"""Model architectures provided by the Mini ViT project."""

import torch
from torch import nn

from src.models.cnn import ConvBlock, SimpleCNN
from src.models.minivit import MiniViT

__all__ = [
    "ConvBlock",
    "MiniViT",
    "SimpleCNN",
    "build_model",
]


def build_model(model_config: dict[str, object], device: torch.device) -> nn.Module:
    """Use the same architecture factory for training and inference."""

    # Copy before removing the factory-only name; preserve the saved config.
    # 创建副本再取出 name，保证训练与推理都不会修改配置快照。
    constructor_config = dict(model_config)
    model_name = str(constructor_config.pop("name", "minivit")).strip().lower()
    if model_name == "minivit":
        model = MiniViT(**constructor_config)
    elif model_name == "cnn":
        model = SimpleCNN(**constructor_config)
    else:
        raise ValueError("model.name must be one of: minivit, cnn.")
    return model.to(device)
