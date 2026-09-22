"""Mini Vision Transformer components."""

import torch
from torch import Tensor, nn

# Stage 2, Step 2
class PatchEmbedding(nn.Module):
    """Convert images into a sequence of patch tokens."""

    def __init__(
        self,
        in_channels: int = 3,   # 默认 RGB 三通道
        embed_dim: int = 192,   # 默认每个 patch 将投影为 192 维的 token
        patch_size: int = 4,    # 默认 patch 尺寸: 4*4
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be greater than 0.")

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")

        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0.")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 创建并注册用于 patch 投影的卷积子模块
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size, # 卷积核尺寸（每次覆盖一个 patch）
            stride=patch_size,      # 卷积核步长（patch 互不重叠）
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Convert an image batch into a sequence of patch tokens."""
        # 输入：一个 batch [B, C:3, H:32, W:32]
        if x.ndim != 4:
            raise ValueError(
                f"expected input shape [B, C, H, W], got {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {x.shape[1]}."
            )

        height, width = x.shape[2], x.shape[3]

        # 确保至少能切割出一个 patch
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                "image height and width must be at least patch_size."
            )

        # 确保能完整切割
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "image height and width must be divisible by patch_size."
            )

        # 执行卷积投影
        # [B, C, H, W] → [B, D, H/P, W/P]
        # [B, 3, 32, 32] → [B, 192, 8, 8]（每个 patch：有192个特征，patch 数量：8*8）
        patch_grid = self.projection(x)
        
        # 展平：flatten() 将多个维度合并成一个维度。
        # → [B, D, N = patch_count]
        # → [B, 192, 64]
        patch_features = patch_grid.flatten(start_dim=2)
        
        # 使用 transpose() 交换 N 和 D
        # → [B, N, D]（Transformer 格式）
        # → [B, 64, 192]
        tokens = patch_features.transpose(1, 2)

        return tokens
    

# Stage 2, Step 3
class MultiHeadSelfAttention(nn.Module):
    """Apply multi-head self-attention to a token sequence."""

    def __init__(
        self,
        embed_dim: int = 192,   # 每个 token 的完整特征维度
        num_heads: int = 6,     # 注意力头数
        dropout: float = 0.1,   # 输出投影后的 dropout 概率
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.embed_dim = embed_dim              # 192
        self.num_heads = num_heads              # 6
        self.head_dim = embed_dim // num_heads  # 32

        # 每个头都使用 head_dim 个特征计算点积。
        # 计算缩放因子：1 / √32
        self.scale = self.head_dim ** -0.5

        # 一次性生成 Q、K、V：
        # [B, N, D] -> [B, N, 3D]
        self.qkv_projection = nn.Linear(
            in_features=embed_dim,
            out_features=3 * embed_dim,
        )

        # 合并多个头后，学习如何融合各头的信息。
        # [B, N, D] -> [B, N, D]
        self.output_projection = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )

        # 训练时随机丢弃部分输出，减少过拟合。
        self.dropout = nn.Dropout(p=dropout)
        
    def _split_heads(self, tensor: Tensor) -> Tensor:
        """Split token features into multiple attention heads."""

        batch_size, num_tokens, _ = tensor.shape

        # 将完整特征维度拆成 head 数和每个 head 的维度。
        # [B, N, D] -> [B, N, H, Dh]
        tensor = tensor.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        # 把 head 维移到 token 维之前，方便并行计算注意力。
        # [B, N, H, Dh] -> [B, H, N, Dh]
        tensor = tensor.transpose(1, 2)

        return tensor
    
    def forward(
        self,
        x: Tensor,  # [B, N:65, D:192]  N = 64 个 patch token + 1 个 CLS token
        return_attention: bool = False,  # 是否需要返回注意力权重
        # 如果 return_attention = False，只需要返回最终的 output；反之还需返回 weights
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention to a batch of token sequences."""
        # 输入必须是：[batch, token, embedding]
        if x.ndim != 3:
            raise ValueError(
                f"expected input shape [B, N, D], got {tuple(x.shape)}."
            )

        # 输入 token 的特征维度必须与初始化配置一致。
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding dimension {self.embed_dim}, "
                f"got {x.shape[-1]}."
            )

        batch_size, num_tokens, _ = x.shape

        """ 
            x [B,N,D]
            ↓ qkv_projection
            qkv [B,N,3D]
            ↓ chunk
            Q、K、V [B,N,D]
            ↓ split heads
            Q、K、V [B,H,N,Dh]
            ↓ QKᵀ × scale
            scores [B,H,N,N]
            ↓ softmax(dim=-1)
            attention_weights [B,H,N,N]
            ↓ attention_weights @ V
            head_outputs [B,H,N,Dh]
            ↓ transpose + reshape
            combined [B,N,D]
            ↓ output_projection
            output [B,N,D]
            ↓ dropout
            output [B,N,D]
        """

        # 一次性生成 Q、K、V。
        # [B, N, D] -> [B, N, 3D]
        qkv = self.qkv_projection(x)

        # 沿最后一维将融合结果平均切成三份。
        # 每一份的 shape 都是 [B, N, D]。
        query, key, value = qkv.chunk(3, dim=-1)

        # 每个头都从完整投影结果中获得自己的低维表示。
        # [B, N, D] -> [B, H, N, Dh]
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # 每个头分别计算所有 token 两两之间的匹配分数。
        # Q @ Kᵀ = [B, H, N, Dh] @ [B, H, Dh, N] -> [B, H, N, N]
        scores = query @ key.transpose(-2, -1)

        # 按每个头的特征维度缩放分数。
        scaled_scores = scores * self.scale

        # 每个 Query 对所有 Key 的注意力权重之和为 1。
        # [B, H, N, N] -> [B, H, N, N]
        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1,
        )

        # 每个头使用自己的注意力权重聚合 Value。
        # [B, H, N, N] @ [B, H, N, Dh] -> [B, H, N, Dh]
        head_outputs = attention_weights @ value

        # 将 token 维移到 head 维之前。
        # [B, H, N, Dh] -> [B, N, H, Dh]
        output = head_outputs.transpose(1, 2)

        # 拼接同一个 token 在所有头中的输出。
        # [B, N, H, Dh] -> [B, N, D]
        output = output.reshape(
            batch_size,
            num_tokens,
            self.embed_dim,
        )

        # 学习如何融合不同注意力头提供的信息。
        # [B, N, D] -> [B, N, D]
        output = self.output_projection(output)

        # 训练时随机丢弃部分输出，降低过拟合风险。
        output = self.dropout(output)

        if return_attention:
            return output, attention_weights

        return output
    
    
# Stage 2, Step 4
class MLP(nn.Module):
    """Apply a feed-forward network to each token independently."""

    def __init__(
        self,
        embed_dim: int = 192,   # 输入和输出 token 的特征维度
        hidden_dim: int = 768,  # MLP 内部扩展后的特征维度
        dropout: float = 0.1,   # 两处 dropout 使用的丢弃概率
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 将每个 token 从 D 维扩展到 M 维。
        # [B, N, D] -> [B, N, M]
        self.fc1 = nn.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
        )

        # 为两层 Linear 之间加入非线性。
        self.activation = nn.GELU()

        # 随机丢弃部分隐藏特征。
        self.dropout1 = nn.Dropout(p=dropout)

        # 将每个 token 从 M 维投影回 D 维。
        # [B, N, M] -> [B, N, D]
        self.fc2 = nn.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
        )

        # 随机丢弃部分 MLP 输出。
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to each token."""

        # 输入必须是：[batch, token, embedding]
        if x.ndim != 3:
            raise ValueError(
                f"expected input shape [B, N, D], got {tuple(x.shape)}."
            )

        # token 的特征维度必须与第一层 Linear 的输入一致。
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding dimension {self.embed_dim}, "
                f"got {x.shape[-1]}."
            )

        """
            x [B,N,D]
            ↓ fc1
            hidden [B,N,M]
            ↓ GELU
            hidden [B,N,M]
            ↓ dropout1
            hidden [B,N,M]
            ↓ fc2
            output [B,N,D]
            ↓ dropout2
            output [B,N,D]
        """

        # 扩展每个 token 的特征维度。
        # [B, N, D] -> [B, N, M]
        hidden = self.fc1(x)

        # 对隐藏特征逐元素施加非线性变换。
        # [B, N, M] -> [B, N, M]
        hidden = self.activation(hidden)

        # 训练时随机丢弃部分隐藏特征。
        hidden = self.dropout1(hidden)

        # 将隐藏特征投影回原 embedding 维度。
        # [B, N, M] -> [B, N, D]
        output = self.fc2(hidden)

        # 训练时随机丢弃部分 MLP 输出。
        output = self.dropout2(output)

        return output
    
    
# Stage 2, Step 5
class TransformerEncoderBlock(nn.Module):
    """Combine self-attention and an MLP with pre-norm residual paths."""

    def __init__(
        self,
        embed_dim: int = 192,   # 每个 token 的特征维度
        num_heads: int = 6,     # Attention 的注意力头数
        mlp_dim: int = 768,     # MLP 的隐藏特征维度
        dropout: float = 0.1,   # Attention 和 MLP 使用的 dropout 概率
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        if mlp_dim <= 0:
            raise ValueError("mlp_dim must be greater than 0.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.embed_dim = embed_dim

        # 为 Attention 分支准备尺度稳定的输入。
        self.norm1 = nn.LayerNorm(
            normalized_shape=embed_dim,
        )

        # 在所有 token 之间交换信息。
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 为 MLP 分支准备尺度稳定的输入。
        self.norm2 = nn.LayerNorm(
            normalized_shape=embed_dim,
        )

        # 独立加工每个 token 的特征。
        self.mlp = MLP(
            embed_dim=embed_dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
        )
        
    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the pre-norm Transformer encoder block."""

        # 输入必须是：[batch, token, embedding]
        if x.ndim != 3:
            raise ValueError(
                f"expected input shape [B, N, D], got {tuple(x.shape)}."
            )

        # token 的特征维度必须与 Block 配置一致。
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding dimension {self.embed_dim}, "
                f"got {x.shape[-1]}."
            )

        """x1 = x0 + Attention(LN1(x0))"""

        # Pre-Norm：只归一化送入 Attention 分支的数据。
        # [B, N, D] -> [B, N, D]
        normalized = self.norm1(x)

        if return_attention:
            # 可视化或测试时，同时取得每个 head 的注意力权重。
            attention_output, attention_weights = self.attention(
                normalized,
                return_attention=True,
            )
        else:
            # 普通训练和推理只需要 Attention 输出。
            attention_output = self.attention(normalized)

        # 第一条残差连接保留未经 norm1 修改的原始 x。
        # [B, N, D] + [B, N, D] -> [B, N, D]
        x = x + attention_output

        """x2 = x1 + MLP(LN2(x1))"""
        
        # Pre-Norm：norm2 处理 Attention 更新后的 x。
        # [B, N, D] -> [B, N, D]
        normalized = self.norm2(x)

        # 独立加工每个 token 的特征。
        # [B, N, D] -> [B, N, D]
        mlp_output = self.mlp(normalized)

        # 第二条残差连接保留 Attention 更新后的 x。
        # [B, N, D] + [B, N, D] -> [B, N, D]
        x = x + mlp_output

        if return_attention:
            return x, attention_weights

        return x
    

# Stage 3, Step 1
class MiniViT(nn.Module):
    """Classify images with a compact Vision Transformer."""

    def __init__(
        self,
        image_size: int = 32,    # 输入图片的高度和宽度
        in_channels: int = 3,    # 输入图片通道数
        num_classes: int = 10,   # 分类类别数
        patch_size: int = 4,     # 每个正方形 patch 的边长
        embed_dim: int = 192,    # 每个 token 的特征维度
        depth: int = 6,          # Encoder Block 数量
        num_heads: int = 6,      # 每个 Block 的 Attention head 数
        mlp_dim: int = 768,      # 每个 Block 的 MLP 隐藏维度
        dropout: float = 0.1,    # Attention 和 MLP 的 dropout 概率
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than 0."
            )

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than 0."
            )

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be greater than 0."
            )

        if patch_size <= 0:
            raise ValueError(
                "patch_size must be greater than 0."
            )

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size."
            )

        if embed_dim <= 0:
            raise ValueError(
                "embed_dim must be greater than 0."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than 0."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than 0."
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        if mlp_dim <= 0:
            raise ValueError(
                "mlp_dim must be greater than 0."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        patches_per_side = image_size // patch_size
        self.num_patches = patches_per_side ** 2
        self.num_tokens = self.num_patches + 1

        # 将图片转换成 patch token 序列。
        # [B, C, H, W] -> [B, Np, D]，默认为 [B, 64, 192]
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # 每张图token数：64 → 65（加入了CLS）→ 65（引入位置编码，数值层面上的直接相加）

        # CLS: 每张图片额外加的一个 token，用于汇总信息，服务于最终分类
        # 所有图片共享一份可学习的初始 CLS token: [1, 1, D]
        self.cls_token = nn.Parameter(
            torch.empty(
                1,
                1,
                embed_dim,
            )
        )

        # 可学习的位置编码参数 [1, Np + 1, D]
        self.position_embedding = nn.Parameter(
            torch.empty(
                1,
                self.num_tokens,
                embed_dim,
            )
        )

        # 创建并注册结构相同、参数独立的 Encoder Blocks。
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # 对 Pre-Norm Encoder Stack 的最终输出做归一化。
        self.final_norm = nn.LayerNorm(
            normalized_shape=embed_dim,
        )

        # 将最终 CLS 表征映射为类别 logits。
        # [B, D] -> [B, K]
        self.classifier = nn.Linear(
            in_features=embed_dim,
            out_features=num_classes,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the learned token and positional parameters."""

        # 由于这两个参数是由 torch.empty() 创建的，所以必须显式填入数值
        nn.init.trunc_normal_(
            self.cls_token,
            mean=0.0,
            std=0.02,
            a=-0.04,
            b=0.04,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
            a=-0.04,
            b=0.04,
        )
        
    def forward(
        self,
        images: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Convert an image batch into classification logits."""

        # 输入必须是：[batch, channel, height, width]
        if images.ndim != 4:
            raise ValueError(
                f"expected input shape [B, C, H, W], "
                f"got {tuple(images.shape)}."
            )

        # 图片通道数必须与模型配置一致。
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {images.shape[1]}."
            )

        height, width = images.shape[2], images.shape[3]

        # 可学习位置编码长度固定，因此图片尺寸必须与配置一致。
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected image size "
                f"{self.image_size}x{self.image_size}, "
                f"got {height}x{width}."
            )

        # 将图片转换成 patch token 序列。
        # [B, C, H, W] -> [B, Np, D]
        patch_tokens = self.patch_embedding(images)

        # 防止实际 patch 数与位置编码配置不一致。
        if patch_tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"patch embedding produced "
                f"{patch_tokens.shape[1]} tokens, "
                f"but the model expected {self.num_patches}."
            )

        batch_size = patch_tokens.shape[0]

        # 为当前 batch 扩展共享的 CLS 参数。（形式上复制，内存上共享）
        # [1, 1, D] -> [B, 1, D]
        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # 将 CLS 放在所有 patch tokens 之前。
        # [B, 1, D] + [B, Np, D] -> [B, Np + 1, D]
        tokens = torch.cat(
            [cls_tokens, patch_tokens],
            dim=1,
        )

        # 为 CLS 和每个 patch 加入固定位置对应的可学习表示。（数值上直接相加）
        # [B, N, D] + [1, N, D] -> [B, N, D]
        tokens = tokens + self.position_embedding

        # 普通模式只传递 tokens；可视化模式额外保留最后一层 Attention。
        last_attention: Tensor | None = None

        # 进入核心循环
        for block_index, block in enumerate(self.blocks):
            is_last_block = block_index == len(self.blocks) - 1
            # 最后一层单独处理，只返回最后一层的注意力权重
            if return_attention and is_last_block:
                tokens, last_attention = block(
                    tokens,
                    return_attention=True,
                )
            else:
                tokens = block(tokens)

        # 对整个 Pre-Norm Encoder Stack 的最终输出做归一化。
        # [B, N, D] -> [B, N, D]
        tokens = self.final_norm(tokens)

        # 读取第 0 个 token（CLS） 作为整张图片的表示。
        # [B, N, D] -> [B, D]
        cls_representation = tokens[:, 0, :]

        # 将 CLS 表征映射为每个类别的原始分数。
        # [B, D] -> [B, K]
        logits = self.classifier(
            cls_representation
        )

        if return_attention:
            if last_attention is None:
                raise RuntimeError(
                    "last attention weights were not produced."
                )

            return logits, last_attention

        return logits