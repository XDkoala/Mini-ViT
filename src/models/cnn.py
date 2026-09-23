"""Convolutional baseline models for CIFAR-10 classification."""

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Extract local features and halve their spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                "in_channels and out_channels must be positive."
            )

        # 3x3 卷积只观察局部邻域；padding=1 使卷积前后的 H、W 不变。
        # bias=False 是因为紧随其后的 BatchNorm 已有可学习平移参数。
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 每个输出通道独立标准化。训练时使用当前 batch 统计量，
        # 同时维护 running mean/variance，验证和推理时改用运行统计量。
        self.norm = nn.BatchNorm2d(out_channels)

        # inplace=True 复用输入张量的存储，减少少量显存占用。
        self.activation = nn.ReLU(inplace=True)

        # 在每个 2x2 区域保留最大响应，使 H、W 各缩小一半。
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform ``[B, C_in, H, W]`` into ``[B, C_out, H/2, W/2]``."""

        features = self.conv(inputs)
        features = self.norm(features)
        features = self.activation(features)
        return self.pool(features)


class SimpleCNN(nn.Module):
    """A compact CNN baseline with configurable feature widths."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: Sequence[int] = (96, 192, 384, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        channel_tuple = tuple(channels)

        if not channel_tuple:
            raise ValueError("channels must contain at least one value.")

        if any(channel <= 0 for channel in channel_tuple):
            raise ValueError("all channels values must be positive.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channel_tuple

        # 相邻的通道数两两配对，依次创建卷积块。
        # 将模块放入 ModuleList 后，PyTorch 才能注册并管理全部参数。
        block_channels = zip(
            (in_channels, *channel_tuple[:-1]),
            channel_tuple,
            strict=True,
        )
        self.blocks = nn.ModuleList([
            ConvBlock(
                in_channels=block_in_channels,
                out_channels=block_out_channels,
            )
            for block_in_channels, block_out_channels in block_channels
        ])

        # 不论卷积块输出的空间尺寸是多少，都汇聚为 1x1。
        # 因此分类头只依赖通道数，不需要把 32x32 写死在 Linear 中。
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout 只在训练模式随机屏蔽特征，验证模式会自动关闭。
        self.dropout = nn.Dropout(dropout)

        # 全局池化后每张图片只剩 channels[-1] 个特征，
        # Linear 将它们映射成 num_classes 个未归一化 logits。
        self.classifier = nn.Linear(
            in_features=channel_tuple[-1],
            out_features=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map image batches ``[B, C, H, W]`` to logits ``[B, classes]``."""

        if images.ndim != 4:
            raise ValueError(
                "expected input shape [B, C, H, W], "
                f"but received {tuple(images.shape)}."
            )

        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"but received {images.shape[1]}."
            )

        # n 个池化层要求 H、W 至少为 2**n，否则会缩小到 0。
        minimum_size = 2 ** len(self.blocks)
        height, width = images.shape[-2:]

        if height < minimum_size or width < minimum_size:
            raise ValueError(
                "input height and width must each be at least "
                f"{minimum_size}."
            )

        features = images

        # 每个 block 扩充通道数并把空间尺寸减半。
        for block in self.blocks:
            features = block(features)

        features = self.global_pool(features)

        # [B, C, 1, 1] -> [B, C]；从第 1 维开始展平，保留 batch 维。
        features = torch.flatten(features, start_dim=1)
        features = self.dropout(features)
        return self.classifier(features)
