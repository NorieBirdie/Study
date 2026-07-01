"""Run a real pretrained MobileNetV3 image classifier demo.

这个脚本回答“下载预训练权重后，能不能输出真实正确结果”的部分。

MobileNetV3-Small 的 torchvision 预训练权重是在 ImageNet 分类任务上训练的，
所以它能输出真实的 ImageNet 类别概率，例如 dog/cat/car 等分类结果。

注意：ImageNet 分类权重不能直接给出真实机器人动作。真实动作需要一个在
对应机器人数据集和任务上训练好的 policy/action head。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from compare_torch_onnx import error_stats, run_onnx
from models import SEED, set_seed
from quantize_onnx import quantize_dynamic_int8
from run_pipeline import format_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def configure_torch_cache() -> None:
    """Store downloaded torchvision weights inside this demo directory."""
    os.environ.setdefault("TORCH_HOME", str(OUTPUT_DIR / "torch_cache"))


def parse_args() -> argparse.Namespace:
    """Parse image path and top-k settings."""
    parser = argparse.ArgumentParser(
        description="Run pretrained MobileNetV3-Small ImageNet classification.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a real RGB image.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top ImageNet classes to print.",
    )
    return parser.parse_args()


def load_imagenet_image(image_path: Path, weights: MobileNet_V3_Small_Weights) -> torch.Tensor:
    """Load one image and apply the official preprocessing for these weights."""
    pil_image = Image.open(image_path).convert("RGB")
    preprocess = weights.transforms()
    return preprocess(pil_image).unsqueeze(0)


def export_classifier_onnx(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_path: Path,
) -> Path:
    """Export pretrained classifier to ONNX with image -> logits interface."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        image,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
    )
    return output_path


def topk_lines(
    logits: np.ndarray,
    categories: list[str],
    topk: int,
    prefix: str,
) -> list[str]:
    """Format top-k ImageNet predictions from logits."""
    probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    top_indices = probabilities.argsort()[-topk:][::-1]
    lines = [prefix]
    for rank, index in enumerate(top_indices, start=1):
        label = categories[int(index)]
        probability = probabilities[int(index)]
        lines.append(f"  {rank}. {label}: {probability:.4f}")
    return lines


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_torch_cache()

    set_seed(SEED)

    # 这里会使用 torchvision 的官方 ImageNet 预训练权重。
    # 如果本机没有缓存，torchvision 会首次下载权重文件。
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    image = load_imagenet_image(args.image, weights)

    fp32_onnx = OUTPUT_DIR / "mobilenet_v3_small_imagenet.onnx"
    int8_onnx = OUTPUT_DIR / "mobilenet_v3_small_imagenet_int8.onnx"
    summary_path = OUTPUT_DIR / "pretrained_classifier_summary.txt"

    print("Running PyTorch pretrained classifier...")
    with torch.no_grad():
        torch_logits = model(image).cpu().numpy()

    print("Exporting pretrained classifier to FP32 ONNX...")
    export_classifier_onnx(model, image, fp32_onnx)

    print("Running FP32 ONNX classifier...")
    onnx_logits = run_onnx(fp32_onnx, {"image": image.cpu().numpy()})
    torch_onnx_stats = error_stats(torch_logits, onnx_logits)

    print("Applying INT8 dynamic quantization...")
    quantize_dynamic_int8(fp32_onnx, int8_onnx)

    print("Running INT8 ONNX classifier...")
    int8_logits = run_onnx(int8_onnx, {"image": image.cpu().numpy()})
    fp32_int8_stats = error_stats(onnx_logits, int8_logits)

    categories = weights.meta["categories"]
    topk = min(args.topk, len(categories))
    summary_lines = [
        "Pretrained MobileNetV3-Small ImageNet Classifier Demo",
        f"Seed: {SEED}",
        f"Image source: {args.image}",
        "Model: torchvision.models.mobilenet_v3_small",
        f"Weights: {weights}",
        "Fixed shapes:",
        "  image  : [1, 3, 224, 224]",
        "  logits : [1, 1000]",
        "",
        "Important note:",
        "  These are real ImageNet classification results.",
        "  They are not robot actions. Correct actions require a policy trained on action data.",
        "",
        *topk_lines(torch_logits, categories, topk, "PyTorch top predictions:"),
        "",
        *topk_lines(onnx_logits, categories, topk, "FP32 ONNX top predictions:"),
        "",
        *topk_lines(int8_logits, categories, topk, "INT8 ONNX top predictions:"),
        "",
        format_stats("Torch logits vs ONNX logits", torch_onnx_stats),
        format_stats("FP32 ONNX logits vs INT8 ONNX logits", fp32_int8_stats),
        "Generated files:",
        f"  {fp32_onnx.name}",
        f"  {int8_onnx.name}",
        f"  {summary_path.name}",
    ]
    summary = "\n".join(summary_lines)
    summary_path.write_text(summary + "\n", encoding="utf-8")

    print()
    print(summary)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
