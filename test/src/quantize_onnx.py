"""Apply ONNX Runtime INT8 dynamic quantization.

动态量化主要把权重从 FP32 压到 INT8，并在推理时动态处理激活值量化。
它不需要校准数据集，适合做前期验证；真实 QNN 任务可能还会做静态量化和校准。
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_dynamic_int8(fp32_model_path: Path, int8_model_path: Path) -> Path:
    """Create an INT8 dynamically quantized ONNX model."""
    int8_model_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_model_path),
        model_output=str(int8_model_path),
        # QInt8 表示权重量化成 signed int8。
        weight_type=QuantType.QInt8,
        # Dynamic quantization of Conv may emit ConvInteger, which is not
        # implemented by every ONNX Runtime CPU build. Linear layers are enough
        # for this first-pass validation demo and keep the pipeline portable.
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    return int8_model_path
