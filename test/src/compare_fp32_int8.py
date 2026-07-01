"""Compare FP32 ONNX models with INT8 ONNX models.

PyTorch vs ONNX 关注“模型格式转换是否正确”。
FP32 ONNX vs INT8 ONNX 关注“量化以后数值偏差有多大”。
"""

from __future__ import annotations

from pathlib import Path

import torch

from compare_torch_onnx import error_stats, run_onnx


def compare_encoder_fp32_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    image: torch.Tensor,
) -> dict[str, float]:
    """Compare encoder outputs from FP32 ONNX and INT8 ONNX."""
    # FP32 和 INT8 必须吃同一份输入，否则误差没有可解释性。
    feeds = {"image": image.cpu().numpy()}
    fp32_latent = run_onnx(fp32_onnx_path, feeds)
    int8_latent = run_onnx(int8_onnx_path, feeds)
    return error_stats(fp32_latent, int8_latent)


def compare_predictor_fp32_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    latent: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, float]:
    """Compare predictor outputs from FP32 ONNX and INT8 ONNX."""
    # predictor 的量化误差通常比 encoder 更明显，因为这里有多层 Linear。
    feeds = {
        "latent": latent.cpu().numpy(),
        "action": action.cpu().numpy(),
    }
    fp32_next_latent = run_onnx(fp32_onnx_path, feeds)
    int8_next_latent = run_onnx(int8_onnx_path, feeds)
    return error_stats(fp32_next_latent, int8_next_latent)


def compare_action_policy_fp32_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    image: torch.Tensor,
) -> dict[str, float]:
    """Compare image -> action outputs from FP32 ONNX and INT8 ONNX."""
    feeds = {"image": image.cpu().numpy()}
    fp32_action = run_onnx(fp32_onnx_path, feeds)
    int8_action = run_onnx(int8_onnx_path, feeds)
    return error_stats(fp32_action, int8_action)
