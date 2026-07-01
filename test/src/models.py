"""Tiny neural network modules for the LeWorldModel QNN validation demo.

这个文件只负责定义网络结构和固定输入。真实 LeWorldModel 会更复杂，
但部署验证时最重要的是先把接口关系固定下来：

    image -> encoder -> latent
    latent + action -> predictor -> next_latent
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torch import nn


# 固定随机种子后，每次运行都会得到相同的模型初始化和相同的 dummy input。
# 这对于误差对比很重要：否则你不知道误差来自模型转换，还是来自随机数变化。
SEED = 20260701

# 这里刻意使用固定 shape，不做动态 batch/dynamic axes。
# 端侧部署和 QNN 转换前期验证通常更喜欢明确稳定的输入输出规格。
IMAGE_SHAPE = (1, 3, 224, 224)
LATENT_SHAPE = (1, 256)
ACTION_SHAPE = (1, 4)


def set_seed(seed: int = SEED) -> None:
    """Make model initialization and demo inputs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyEncoder(nn.Module):
    """A small CNN that mimics image -> latent in LeWorldModel.

    输入是一张图像，输出是一个 256 维 latent 向量。
    真实 encoder 可能是 ResNet、ViT 或更复杂的视觉模块，这里只用几层
    Conv2d 来模拟它的输入输出行为。
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # stride=2 会降低空间分辨率，相当于逐步提取更抽象的图像特征。
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            # 把任意空间大小压到 1x1，得到每个通道一个全局特征。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # 64 个通道特征映射到 256 维 latent，模拟真实世界模型的状态表示。
        self.projection = nn.Linear(64, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        # [1, 64, 1, 1] -> [1, 64]，Linear 层需要二维输入。
        x = torch.flatten(x, start_dim=1)
        latent = self.projection(x)
        return latent


class TinyPredictor(nn.Module):
    """A small MLP that mimics latent + action -> next_latent.

    predictor 可以理解成一个简化的 dynamics model：
    给定当前状态 latent 和动作 action，预测执行动作后的下一个状态。
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # latent 和 action 拼接后是 260 维：[1, 256] + [1, 4] -> [1, 260]。
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # dim=1 表示在特征维拼接，batch 维仍然保持为 1。
        x = torch.cat([latent, action], dim=1)
        next_latent = self.network(x)
        return next_latent


class MobileNetActionPolicy(nn.Module):
    """A more realistic image -> action policy based on MobileNetV3-Small.

    这个模型直接从图像输出 4 维动作：

        image [1, 3, 224, 224] -> action [1, 4]

    它使用 torchvision 里现成的 MobileNetV3-Small 架构。MobileNet 系列本来就
    面向移动端/边缘设备设计，比大型 ResNet/ViT 更适合在 Mac 和端侧验证。

    注意：这里默认不下载 pretrained 权重，也没有机器人数据集训练过。
    所以它是“真实轻量视觉策略网络结构”，不是“已经能控制真实机器人的策略”。
    """

    def __init__(self, action_dim: int = 4, use_pretrained_backbone: bool = False) -> None:
        super().__init__()

        # weights=None 不需要联网下载，保证 demo 在没有网络时也能跑。
        # 如果本机已经有 torchvision 缓存权重，也可以把 use_pretrained_backbone
        # 改成 True 做视觉特征初始化，但动作头仍然需要任务数据训练。
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if use_pretrained_backbone
            else None
        )
        backbone = mobilenet_v3_small(weights=weights)

        # MobileNetV3 的 classifier 原本输出 ImageNet 1000 类。
        # 这里替换成一个动作头，把视觉特征映射成 4 维连续动作。
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=False),
            nn.Linear(256, action_dim),
        )
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw_action = self.backbone(image)
        # tanh 把输出限制在 [-1, 1]，这很常见于连续控制动作。
        action = torch.tanh(raw_action)
        return action


def create_demo_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed demo inputs with the required shapes."""
    # 这里用随机张量代替真实相机图像和控制动作。
    # 因为 set_seed 已经固定，随机输入在每次运行时也是可复现的。
    image = torch.randn(*IMAGE_SHAPE, dtype=torch.float32)
    action = torch.randn(*ACTION_SHAPE, dtype=torch.float32)
    return image, action
