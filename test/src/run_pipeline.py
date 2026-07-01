"""Run the complete LeWorldModel QNN validation demo pipeline.

从项目根目录执行：

    python src/run_pipeline.py

这个脚本把所有步骤串起来：
1. 创建 TinyEncoder/TinyPredictor 和固定输入
2. 导出 FP32 ONNX
3. 比较 PyTorch 和 ONNX Runtime 输出
4. 做 INT8 动态量化
5. 比较 FP32 ONNX 和 INT8 ONNX 输出
6. 写出 result_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import torch

from compare_fp32_int8 import compare_encoder_fp32_int8, compare_predictor_fp32_int8
from compare_torch_onnx import compare_encoder_torch_onnx, compare_predictor_torch_onnx
from export_onnx import export_encoder_onnx, export_predictor_onnx
from models import SEED, TinyEncoder, TinyPredictor, create_demo_inputs, set_seed
from quantize_onnx import quantize_dynamic_int8


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def format_stats(title: str, stats: dict[str, float]) -> str:
    """Format one comparison block for terminal and result_summary.txt."""
    return (
        f"{title}\n"
        f"  max_abs_error : {stats['max_abs_error']:.8f}\n"
        f"  mean_abs_error: {stats['mean_abs_error']:.8f}\n"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先固定 seed，再创建模型和输入。
    # 顺序很重要：模型权重初始化和 torch.randn 都会消耗随机数。
    set_seed(SEED)
    encoder = TinyEncoder()
    predictor = TinyPredictor()

    # 本 demo 只做推理/导出，不训练模型，所以统一切到 eval 模式。
    encoder.eval()
    predictor.eval()

    # The same image/action tensors are reused throughout the pipeline.
    image, action = create_demo_inputs()

    # Predictor export and comparison should use the exact latent produced by
    # this encoder instance, matching the real encoder -> predictor data flow.
    with torch.no_grad():
        latent = encoder(image)

    # 所有产物集中放到 outputs，便于后续交给 QNN 转换工具或做归档。
    encoder_onnx = OUTPUT_DIR / "tiny_encoder.onnx"
    predictor_onnx = OUTPUT_DIR / "tiny_predictor.onnx"
    encoder_int8 = OUTPUT_DIR / "tiny_encoder_int8.onnx"
    predictor_int8 = OUTPUT_DIR / "tiny_predictor_int8.onnx"
    summary_path = OUTPUT_DIR / "result_summary.txt"

    print("Exporting FP32 ONNX models...")
    # 这一步验证 PyTorch 模型能不能成功变成 ONNX 计算图。
    export_encoder_onnx(encoder, image, encoder_onnx)
    export_predictor_onnx(predictor, latent, action, predictor_onnx)

    print("Comparing PyTorch and ONNX Runtime outputs...")
    # 如果这里误差很小，说明 ONNX 导出基本可信。
    torch_onnx_encoder = compare_encoder_torch_onnx(encoder, image, encoder_onnx)
    torch_onnx_predictor = compare_predictor_torch_onnx(
        predictor,
        latent,
        action,
        predictor_onnx,
    )

    print("Applying INT8 dynamic quantization...")
    # 这一步生成量化模型，模拟真实 QNN 量化前的数值风险评估。
    quantize_dynamic_int8(encoder_onnx, encoder_int8)
    quantize_dynamic_int8(predictor_onnx, predictor_int8)

    print("Comparing FP32 ONNX and INT8 ONNX outputs...")
    # 如果量化误差过大，真实部署前就需要调整模型结构、量化策略或校准方式。
    fp32_int8_encoder = compare_encoder_fp32_int8(encoder_onnx, encoder_int8, image)
    fp32_int8_predictor = compare_predictor_fp32_int8(
        predictor_onnx,
        predictor_int8,
        latent,
        action,
    )

    # summary 是这个 demo 的最终检查报告，方便直接贴给别人看结果。
    summary = "\n".join(
        [
            "LeWorldModel QNN Quantization Validation Demo",
            f"Seed: {SEED}",
            "Fixed shapes:",
            "  image       : [1, 3, 224, 224]",
            "  latent      : [1, 256]",
            "  action      : [1, 4]",
            "  next_latent : [1, 256]",
            "",
            format_stats("Torch vs ONNX Encoder", torch_onnx_encoder),
            format_stats("Torch vs ONNX Predictor", torch_onnx_predictor),
            format_stats("FP32 ONNX vs INT8 ONNX Encoder", fp32_int8_encoder),
            format_stats("FP32 ONNX vs INT8 ONNX Predictor", fp32_int8_predictor),
            "Generated files:",
            f"  {encoder_onnx.name}",
            f"  {predictor_onnx.name}",
            f"  {encoder_int8.name}",
            f"  {predictor_int8.name}",
            f"  {summary_path.name}",
        ]
    )

    summary_path.write_text(summary + "\n", encoding="utf-8")
    print()
    print(summary)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
