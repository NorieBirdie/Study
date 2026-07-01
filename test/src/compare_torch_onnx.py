"""Compare PyTorch outputs with ONNX Runtime outputs.

这一步的目的不是追求速度，而是验证“导出的 ONNX 模型是否仍然等价于
原来的 PyTorch 模型”。如果这里误差很大，后面做量化或 QNN 转换就没有意义。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from models import MobileNetActionPolicy, TinyEncoder, TinyPredictor


def error_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return simple absolute-error metrics for two tensors."""
    # 绝对误差能直观看出两个后端输出差多少。
    # max 关注最坏的单个元素，mean 关注整体平均偏差。
    abs_error = np.abs(reference - candidate)
    return {
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
    }


def run_onnx(model_path: Path, feeds: dict[str, np.ndarray]) -> np.ndarray:
    """Run one ONNX model and return its first output."""
    # CPUExecutionProvider 表示在本机 CPU 上跑 ONNX Runtime。
    # 真实部署到 Qualcomm HTP/NPU 时，会换成 QNN 相关 runtime/provider。
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    # session.run(None, feeds) 表示计算所有输出。
    # feeds 的 key 必须和导出 ONNX 时设置的 input_names 一致。
    outputs = session.run(None, feeds)
    return outputs[0]


def compare_encoder_torch_onnx(
    encoder: TinyEncoder,
    image: torch.Tensor,
    encoder_onnx_path: Path,
) -> dict[str, float]:
    """Compare encoder latent from the same model instance and same input."""
    encoder.eval()
    with torch.no_grad():
        # torch.no_grad() 关闭梯度计算，推理更快，也避免产生训练用的中间状态。
        torch_latent = encoder(image).cpu().numpy()

    # 同一份 image 输入给 ONNX Runtime，保证比较只反映后端/格式差异。
    onnx_latent = run_onnx(
        encoder_onnx_path,
        {"image": image.cpu().numpy()},
    )
    return error_stats(torch_latent, onnx_latent)


def compare_predictor_torch_onnx(
    predictor: TinyPredictor,
    latent: torch.Tensor,
    action: torch.Tensor,
    predictor_onnx_path: Path,
) -> dict[str, float]:
    """Compare predictor next_latent from the same model instance and inputs."""
    predictor.eval()
    with torch.no_grad():
        torch_next_latent = predictor(latent, action).cpu().numpy()

    # predictor 有两个输入，所以 feeds 里要同时提供 latent 和 action。
    onnx_next_latent = run_onnx(
        predictor_onnx_path,
        {
            "latent": latent.cpu().numpy(),
            "action": action.cpu().numpy(),
        },
    )
    return error_stats(torch_next_latent, onnx_next_latent)


def compare_action_policy_torch_onnx(
    policy: MobileNetActionPolicy,
    image: torch.Tensor,
    policy_onnx_path: Path,
) -> dict[str, float]:
    """Compare direct image -> action outputs from PyTorch and ONNX."""
    policy.eval()
    with torch.no_grad():
        torch_action = policy(image).cpu().numpy()

    onnx_action = run_onnx(
        policy_onnx_path,
        {"image": image.cpu().numpy()},
    )
    return error_stats(torch_action, onnx_action)
