"""Run a more realistic image -> action policy deployment demo.

这个脚本使用 torchvision 里现成的 MobileNetV3-Small 架构，把图像直接映射
成 4 维动作。它比 TinyEncoder/TinyPredictor 更接近“视觉策略模型”的形态：

    image [1, 3, 224, 224] -> action [1, 4]

它仍然是前期验证 demo：默认不下载预训练权重，也没有真实机器人数据训练。
重点是验证一个 Mac 跑得动的真实轻量 CNN 架构能否走通 ONNX/INT8 链路。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from compare_fp32_int8 import compare_action_policy_fp32_int8
from compare_torch_onnx import run_onnx
from compare_torch_onnx import compare_action_policy_torch_onnx
from export_onnx import export_action_policy_onnx
from models import SEED, MobileNetActionPolicy, create_demo_inputs, set_seed
from quantize_onnx import quantize_dynamic_int8
from run_pipeline import format_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def configure_torch_cache() -> None:
    """Store downloaded torchvision weights inside this demo directory."""
    os.environ.setdefault("TORCH_HOME", str(OUTPUT_DIR / "torch_cache"))


def parse_args() -> argparse.Namespace:
    """Parse optional image input for the policy demo."""
    parser = argparse.ArgumentParser(
        description="Run MobileNetV3 image -> action ONNX/INT8 validation.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path. If omitted, a fixed random demo image is used.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help=(
            "Use ImageNet pretrained MobileNetV3 backbone. "
            "The action head is still untrained for robot control."
        ),
    )
    return parser.parse_args()


def load_policy_image(image_path: Path | None) -> tuple[torch.Tensor, str]:
    """Load a real image or fall back to the fixed random demo image."""
    if image_path is None:
        image, _ = create_demo_inputs()
        return image, "fixed random demo image"

    # MobileNet 类模型通常接收 224x224 RGB 图像。这里做 resize/crop/toTensor
    # 和 ImageNet 标准化，得到固定 [1, 3, 224, 224] 输入。
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    pil_image = Image.open(image_path).convert("RGB")
    image = preprocess(pil_image).unsqueeze(0)
    return image, str(image_path)


def format_action(name: str, action: torch.Tensor | list[float]) -> str:
    """Format one action vector in a compact readable form."""
    if isinstance(action, torch.Tensor):
        values = action.detach().cpu().flatten().tolist()
    else:
        values = action
    joined = ", ".join(f"{value:.6f}" for value in values)
    return f"{name}: [{joined}]"


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_torch_cache()

    set_seed(SEED)
    policy = MobileNetActionPolicy(use_pretrained_backbone=args.pretrained_backbone)
    policy.eval()

    image, image_source = load_policy_image(args.image)

    policy_onnx = OUTPUT_DIR / "mobilenet_action_policy.onnx"
    policy_int8 = OUTPUT_DIR / "mobilenet_action_policy_int8.onnx"
    summary_path = OUTPUT_DIR / "policy_result_summary.txt"

    print("Exporting MobileNetV3 image -> action FP32 ONNX model...")
    export_action_policy_onnx(policy, image, policy_onnx)

    print("Comparing PyTorch and ONNX Runtime policy outputs...")
    torch_onnx_policy = compare_action_policy_torch_onnx(policy, image, policy_onnx)

    with torch.no_grad():
        torch_action = policy(image)
    onnx_action = run_onnx(policy_onnx, {"image": image.cpu().numpy()})

    print("Applying INT8 dynamic quantization to policy model...")
    quantize_dynamic_int8(policy_onnx, policy_int8)

    print("Comparing FP32 ONNX and INT8 ONNX policy outputs...")
    fp32_int8_policy = compare_action_policy_fp32_int8(
        policy_onnx,
        policy_int8,
        image,
    )
    int8_action = run_onnx(policy_int8, {"image": image.cpu().numpy()})

    summary = "\n".join(
        [
            "MobileNetV3 Image-to-Action Policy Demo",
            f"Seed: {SEED}",
            "Model: torchvision.models.mobilenet_v3_small + action head",
            f"Pretrained backbone: {args.pretrained_backbone}",
            f"Image source: {image_source}",
            "Fixed shapes:",
            "  image  : [1, 3, 224, 224]",
            "  action : [1, 4]",
            "",
            "Note:",
            "  This uses a real lightweight CNN architecture, but no trained robot policy weights.",
            "  ImageNet pretrained backbone improves visual features, but does not train the action head.",
            "  The action values are useful for deployment validation, not robot control.",
            "",
            "Action outputs:",
            f"  {format_action('PyTorch action', torch_action)}",
            f"  {format_action('FP32 ONNX action', onnx_action.flatten().tolist())}",
            f"  {format_action('INT8 ONNX action', int8_action.flatten().tolist())}",
            "",
            format_stats("Torch vs ONNX Policy", torch_onnx_policy),
            format_stats("FP32 ONNX vs INT8 ONNX Policy", fp32_int8_policy),
            "Generated files:",
            f"  {policy_onnx.name}",
            f"  {policy_int8.name}",
            f"  {summary_path.name}",
        ]
    )

    summary_path.write_text(summary + "\n", encoding="utf-8")
    print()
    print(summary)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
