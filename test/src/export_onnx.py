"""Export TinyEncoder and TinyPredictor to fixed-shape ONNX models.

ONNX 是 PyTorch 和端侧推理/转换工具之间常用的中间表示。
这一步相当于把 PyTorch 的动态图模型保存成一个可被其他工具读取的计算图。
"""

from __future__ import annotations

from pathlib import Path

import torch

from models import MobileNetActionPolicy, TinyEncoder, TinyPredictor


def export_encoder_onnx(
    encoder: TinyEncoder,
    image: torch.Tensor,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """Export the encoder with image [1, 3, 224, 224] as input."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # eval() 会关闭 dropout/batchnorm 训练行为。虽然 TinyEncoder 没有这些层，
    # 但导出部署模型前统一切到 eval 是一个好习惯。
    encoder.eval()

    torch.onnx.export(
        encoder,
        # image 是 tracing 用的示例输入，ONNX 导出器用它记录计算图和 shape。
        image,
        output_path,
        # export_params=True 表示把模型权重一起写进 ONNX 文件。
        export_params=True,
        opset_version=opset_version,
        # 常量折叠会提前计算图里的常量子表达式，让导出的图更简洁。
        do_constant_folding=True,
        # 明确输入输出名字，后面 ONNX Runtime feed 数据时就用这些 key。
        input_names=["image"],
        output_names=["latent"],
    )
    return output_path


def export_predictor_onnx(
    predictor: TinyPredictor,
    latent: torch.Tensor,
    action: torch.Tensor,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """Export the predictor with latent [1, 256] and action [1, 4]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.eval()

    torch.onnx.export(
        predictor,
        # 多输入模型要用 tuple 传入示例输入。
        (latent, action),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["latent", "action"],
        output_names=["next_latent"],
    )
    return output_path


def export_action_policy_onnx(
    policy: MobileNetActionPolicy,
    image: torch.Tensor,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """Export a direct image -> action policy model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy.eval()

    torch.onnx.export(
        policy,
        image,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["action"],
    )
    return output_path
