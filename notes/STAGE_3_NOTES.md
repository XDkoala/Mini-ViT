# 阶段 3：组装完整 MiniViT

本阶段把阶段 2 已通过测试的 Patch Embedding、Multi-Head Self-Attention、MLP 和 Transformer Encoder Block 组装成完整图像分类模型，使一批 CIFAR-10 图片能够输出 10 类 logits。

开始日期：2026-09-17  
完成日期：2026-09-19  
当前状态：已完成  
当前位置：完整 MiniViT 已实现并通过结构、数值与梯度验收

## 阶段概览

### 阶段目标

完成从图片到分类 logits 的端到端前向与反向闭环：

```text
images
  → patch tokens
  → prepend CLS token
  → add positional embedding
  → encoder stack
  → final LayerNorm
  → select CLS representation
  → classification head
  → logits
```

### 阶段边界

本阶段修改：

- `src/models/minivit.py`：实现完整 `MiniViT`。
- `tests/test_model.py`：补充完整模型的契约与梯度测试。
- `notes/STAGE_3_NOTES.md`：记录组装原理、shape、接口与验收结论。

本阶段不做：

- 正式训练循环、验证指标与 checkpoint。
- 用训练结果判断模型是否具有学习能力。
- CNN 基线、消融实验和最终可视化。

本阶段通过只说明完整模型在结构、数值和梯度上可以运行。阶段 4 将用 tiny overfit 验证它是否真的能够学习。

## 进度总览

- [x] CLS token：作用、参数注册、batch 扩展与序列拼接
- [x] 位置编码：必要性、可学习参数、广播与 shape
- [x] Encoder Stack：重复层、独立参数与 `nn.ModuleList`
- [x] 最终 LayerNorm、CLS 读取与分类头
- [x] 完整 `MiniViT.__init__()` 与参数初始化
- [x] 完整 `MiniViT.forward()` 与可选 Attention 返回接口
- [x] 完整模型自动化测试
- [x] 阶段 3 总验收

## 符号与 shape 速查

| 符号 | 含义 | 第一版默认值 |
|---|---|---:|
| `B` | batch size | 测试常用 2 |
| `C` | 图片通道数 | 3 |
| `H, W` | 图片高、宽 | 32, 32 |
| `P` | patch 边长 | 4 |
| `Np` | patch token 数 | `(H/P)×(W/P)=64` |
| `N` | 加入 CLS 后的总 token 数 | `Np+1=65` |
| `D` | token embedding 维度 | 192 |
| `A` | Attention head 数 | 6 |
| `Dh` | 每个 head 的维度 | `D/A=32` |
| `M` | MLP 隐藏维度 | 768 |
| `L` | Encoder Block 数量 | 6 |
| `K` | 分类类别数 | 10 |

完整默认 shape 路线：

```text
images                         [B,3,32,32]
Patch Embedding                [B,64,192]
prepend CLS token              [B,65,192]
add positional embedding       [B,65,192]
6 × Encoder Block              [B,65,192]
final LayerNorm                [B,65,192]
select token index 0            [B,192]
Linear classification head     [B,10]
```

## 1. CLS Token

### 1.1 职责与学习来源

CLS 不是图片中切出的 patch，而是一份由所有图片共享的可训练初始向量：

```python
self.cls_token = nn.Parameter(torch.empty(1, 1, D))
```

它最初不包含某张图片的信息；加入序列后，作为 Query 在多层 Attention 中读取该图片的 patch，逐渐变成与当前图片相关的汇总表示。分类 loss 只通过最终 CLS 产生 logits，因此会迫使它学习提取对分类有用的信息。它的功能来自模型结构与损失路径，而不是变量名称。

`nn.Parameter` 使该 Tensor 进入 `model.parameters()`、`state_dict()` 和优化器；普通 `requires_grad=True` Tensor 虽可计算梯度，但普通属性赋值不会自动把它注册为模型参数。

### 1.2 batch 扩展与拼接

```text
cls_token     [1,1,D]
  → expand
cls_tokens    [B,1,D]

patch_tokens  [B,Np,D]
  → cat(dim=1)
tokens        [B,Np+1,D]
```

`expand()` 通常通过 stride 0 创建共享底层数据的视图，不复制 B 套参数；不要对扩展视图做原地修改。反向传播时，各样本对 CLS 的梯度贡献汇总回同一份 `[1,1,D]` 参数。

`torch.cat(..., dim=1)` 沿 token 维把 CLS 放到 index 0。最终使用 `tokens[:,0,:]` 读取 CLS。Mean pooling 是另一种合理方案，但本项目选择 CLS 以学习贯穿 Encoder Stack 的专用汇总位置。

## 2. 位置编码

### 2.1 为什么需要位置编码

没有位置机制时，token-wise Linear、LayerNorm、MLP 和 Self-Attention 对 patch 排列具有等变性：

```text
Attention(PX) = P Attention(X)
```

只重排 patch 而固定 CLS 时，最终 CLS 对这组 patch 的汇总近似排列不变，模型会把图片视为一袋无序 patch，无法仅凭内容知道上下左右。多层堆叠不会凭空恢复输入中没有提供的位置信息。

本项目使用可学习绝对位置编码：

```python
self.position_embedding = nn.Parameter(
    torch.empty(1, Np + 1, D)
)
tokens = tokens + self.position_embedding
```

`[1,N,D]` 在 batch 维广播到 `[B,N,D]`；第 0 个向量属于 CLS，其余向量与固定行优先顺序的 patches 一一对应。相加保持 embedding 维度 D 不变。

### 2.2 随机初始化如何获得位置含义

位置编码只在模型创建时随机初始化一次，随后作为普通参数训练，并非每次 forward 重新随机。随机初值先让不同位置具有稳定、可区分的“地址”；因为 token index 与图片空间区域始终固定绑定，分类梯度会逐渐赋予各地址对任务有用的含义。

简单的 `p_i=i×k` 虽能编码一维顺序，但所有位置位于同一特征方向、尺度随 i 增长，且一维相邻不等于二维相邻。固定 sin/cos、多维二维编码、相对位置偏置等都是合法替代方案；当前方案胜在简单、灵活且适合固定尺寸 CIFAR-10。

### 2.3 内容与位置相加后的可辨识性

```text
z_i = x_i + p_i
```

后续网络不能仅从 `z_i` 唯一还原内容 `x_i` 和位置 `p_i`，也不需要显式分离二者。QKV 投影会同时包含内容—内容、内容—位置和位置—位置交互。计算图仍能把梯度分别传给 Patch Embedding 和位置参数，因为二者是加法的不同父节点。

可学习位置编码没有被约束为只记录纯几何坐标。它还可能吸收：

- 与位置绑定的数据集空间先验；
- 中心或边缘等位置的重要性偏置；
- CLS 位置的特殊汇总角色；
- 对共享投影和其他模块偏差的校准。

它不能单独记录当前图片特有的内容，因为同一位置参数由所有图片共享。更准确的名称是“与序列位置绑定的可学习参数”。例如给所有内容表示加公共向量 c、同时从全部位置向量减去 c，合成表示不变，说明二者语义分工并非完全唯一。

### 2.4 固定长度约束

`[1,65,D]` 只能直接匹配 64 个 patch 加一个 CLS。当前完整模型严格检查图片尺寸；若未来支持不同分辨率，需要插值位置编码或改用可变长度的位置机制。

## 3. Encoder Stack

### 3.1 独立层与注册

一层 Attention 已能读取全局，但多层使 token 和 CLS 能反复交流、加工和修正表示。每层保持 `[B,N,D] → [B,N,D]`，因此可以直接串联。

```python
self.blocks = nn.ModuleList(
    [
        TransformerEncoderBlock(...)
        for _ in range(depth)
    ]
)
```

`ModuleList` 注册内部所有层，使其参数被优化器、设备移动、模式切换和 `state_dict()` 正确管理。普通 Python list 不会注册内部模块。

列表推导式必须每次调用构造函数以创建独立参数。禁止使用：

```python
block = TransformerEncoderBlock(...)
self.blocks = nn.ModuleList([block] * depth)
```

列表乘法只重复同一对象的引用，会意外变成跨层权重共享。权重共享是合法的特殊架构，但不是本项目的标准 MiniViT。

### 3.2 前向循环与最后一层 Attention

```python
for block_index, block in enumerate(self.blocks):
    if return_attention and block_index == len(self.blocks) - 1:
        tokens, last_attention = block(
            tokens,
            return_attention=True,
        )
    else:
        tokens = block(tokens)
```

第一版只暴露最后一层 Attention `[B,A,N,N]`，避免接口和存储复杂化。前面层仍然计算权重用于 Value 聚合，只是不向上层返回。最后一层 CLS-to-patch 权重可用于热力图，但 Attention 分布不能直接等同于完整因果解释。

## 4. 最终归一化与分类头

### 4.1 最终 LayerNorm 与 CLS 读取

Pre-Norm Block 的 LayerNorm 位于分支输入，残差相加后的整体输出不会立即归一化，因此所有 Blocks 后使用独立的：

```python
self.final_norm = nn.LayerNorm(D)
tokens = self.final_norm(tokens)
cls_representation = tokens[:, 0, :]
```

`tokens[:,0,:]` 使用整数索引移除 token 维，将 `[B,N,D]` 变为 `[B,D]`。不能复用最后一个 Block 的 `norm2`，因为它服务于最后一个 MLP 输入，之后仍有残差相加。

### 4.2 分类 logits

```python
self.classifier = nn.Linear(D, K)
logits = self.classifier(cls_representation)
```

`[B,D] → [B,K]`。logits 是无范围、和不必为 1 的原始类别分数。训练时直接传给 `nn.CrossEntropyLoss`，因为它内部稳定地组合 `log_softmax` 与负对数似然；若模型先做 softmax，CrossEntropyLoss 会错误地把概率再次当 logits。

推理展示概率时才调用 `torch.softmax(logits, dim=-1)`；预测类别可直接使用 `logits.argmax(dim=-1)`。

## 5. 参数初始化

第一版遵循简单、可解释的方案：

- Linear、Conv2d 和 LayerNorm 暂用 PyTorch 默认初始化。
- CLS token 和位置编码使用 `mean=0,std=0.02`、边界 `[-0.04,0.04]` 的截断正态分布。
- 不混用多个来源不明的初始化策略。

`torch.empty()` 只分配内存，因此两个特殊参数必须立即显式初始化。`nn.init.trunc_normal_()` 原地填充且不把初始化操作加入训练计算图。PyTorch 的 `a`、`b` 是绝对数值边界；若意图为均值正负两个标准差，需要明确写成 `±0.04`。

## 6. 完整 MiniViT

### 6.1 接口与输入契约

```python
class MiniViT(nn.Module):
    def forward(
        self,
        images: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ...
```

返回契约：

```text
return_attention=False → logits [B,K]
return_attention=True  → (logits [B,K], last_attention [B,A,N,N])
```

当前第一版假设正方形固定尺寸输入。模型入口检查：

- 图片必须为 `[B,C,H,W]`；
- C 必须等于构造时的 `in_channels`；
- H、W 必须同时等于 `image_size`；
- Patch Embedding 产生的 token 数必须等于位置编码对应的 `num_patches`。

### 6.2 完整数据流

```text
images [B,C,H,W]
  → PatchEmbedding
patch_tokens [B,Np,D]
  → expand + prepend CLS
tokens [B,Np+1,D]
  → add position_embedding
tokens [B,N,D]
  → L × TransformerEncoderBlock
tokens [B,N,D]
  → final_norm
tokens [B,N,D]
  → select index 0
cls_representation [B,D]
  → classifier
logits [B,K]
```

默认配置对应：

```text
[B,3,32,32] → [B,64,192] → [B,65,192]
→ 6 × Encoder → [B,192] → [B,10]
```

## 7. 自动化测试与阶段总验收

- [x] 默认输入 `[2,3,32,32]` 输出 logits `[2,10]`。
- [x] 自定义小配置符合通用 shape 契约，默认尺寸未被写死。
- [x] 可选返回最后一层 Attention 权重 `[B,A,N,N]`。
- [x] Attention 权重在 Key 维求和接近 1。
- [x] 错误图片 shape、尺寸和通道数给出清晰异常。
- [x] 输出和 Attention 权重均不含 NaN 或 Inf。
- [x] `CrossEntropyLoss` 能从 logits 与类别标签计算标量 loss。
- [x] `loss.backward()` 成功。
- [x] Patch Embedding、CLS token、位置编码、Attention、MLP、最终 LayerNorm 和分类头的关键参数均有非空且有限的梯度。
- [x] `tests/test_model.py` 中完整模型测试通过。
- [x] 项目全部测试通过。

2026-09-19 验收通过：

- 默认模型：`[2,3,32,32] → logits [2,10]`，最后一层 Attention 为 `[2,6,65,65]`。
- 自定义灰度小模型：`image_size=8,P=2,D=8,L=2,A=2,M=13,K=5`，输出 `[3,5]`，Attention 为 `[3,2,17,17]`。
- CLS 为 `[1,1,D]`、位置编码为 `[1,N,D]`，均正确注册且初始化值位于指定截断范围。
- Encoder Blocks 是独立模块，未发生意外权重共享。
- Attention 权重在 Key 维求和接近 1，logits、loss 和 Attention 均为有限值。
- CrossEntropyLoss 成功产生标量 loss；从 loss 到全部注册参数的梯度均非空且有限。
- 非法构造参数、错误图片维数、通道数和尺寸均被拒绝。
- `tests/test_model.py`：`73 passed`；项目全部测试：`84 passed`。

阶段 3 已完成。上述结果确认完整模型的结构和梯度链路正确；是否能够拟合数据将在阶段 4 的 tiny overfit 中验证。
