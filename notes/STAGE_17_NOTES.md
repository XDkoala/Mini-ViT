# 阶段 17：Attention Rollout 与单图推理

本阶段为冻结的 MiniViT 增加跨层注意力可视化与可复用的单图推理入口。注意力图用于检查模型内部的信息混合方式，不作为因果解释。

开始日期：2026-09-29

完成日期：2026-10-11

当前状态：已完成

当前位置：全部层注意力、Rollout、单图 CLI、固定案例、CPU/CUDA 和完整回归均已验收

## 进度

- [x] 核对现有模型的注意力返回接口
- [x] 明确最后一层 Attention 与 Rollout 的解释对象和边界
- [x] 理解跨层矩阵乘法方向与残差连接
- [x] 冻结 Rollout 的多头融合规则及解释边界
- [x] 设计兼容现有调用的全部层注意力接口
- [x] 实现全部层注意力返回及契约测试
- [x] 实现 head 融合、残差、归一化和跨层矩阵连乘
- [x] 实现 CLS-to-patch 热图转换
- [x] 完成热图叠图与末层/rollout 对照
- [x] 实现单图推理 CLI 与一致的预处理/类别映射
- [x] 比较正确/错误、clean/corrupted、末层/rollout 图像
- [x] 阶段 17 总验收

## 1. 从现有接口理解 Rollout 的问题

现有 `MiniViT.forward(images, return_attention=True)` 返回分类 logits 和**最后一个 Encoder Block** 的多头注意力张量，形状为 `[B, H, N, N]`。当前模型对 32×32 图像、4×4 patch 有 64 个 patch token，另有 1 个 CLS token，所以 `N=65`。行代表正在读取其他 token 的 query，列代表被读取的 key。例如最后一层 `A[CLS, patch_j]` 是该层 CLS 对第 `j` 个 patch 的注意力权重。

但最后一层的 patch token 本身已经经过前面多层信息混合。即使最后一层 CLS 主要读取 patch 7，也不能断言最终判断只依赖原图 patch 7：这个 token 可能早已含有其他位置的信息。最后一层 Attention 描述**一层中的直接读取权重**；Attention Rollout 试图在简化假设下，把这些读取关系沿网络层数向前追踪，得到最终 CLS 对原始输入 token 的一种**近似信息路径权重**。

第一步暂不把它称作“重要性”或“因果贡献”。Softmax 权重不是像素敏感度，也没有直接计入 Value 内容、MLP、LayerNorm、分类头及非线性效应。后续热图只能作为探索性可视化，不能单独证明模型为何做出某个预测。

### 后续实现约束

1. 普通 `forward(images)` 和现有 `return_attention=True` 的返回形式保持兼容；新增显式选项返回**全部层** `[depth, B, H, N, N]` 或等价的逐层序列。
2. 先明确如何合并每层多个 head，再给融合后的注意力加入单位矩阵近似残差路径，对每行重新归一化。
3. 按 token 的 query→key 方向从浅层到深层组合矩阵，取最终 CLS 行的 patch 列，再映射到 8×8 patch 网格。
4. 真实图像叠图必须标明使用的是末层 Attention 还是 Rollout，以及二者都不是因果解释。

## 2. 为什么跨层注意力要相乘

记第 `l` 层融合多头之后的注意力矩阵为 `A_l`，形状 `[N, N]`。`A_l[i,j]` 表示该层输出位置 `i` 从上一层位置 `j` 读取信息的权重。注意行是“读取者”，列是“被读取者”；每一行和为 1。

用三个 token `CLS、P1、P2` 举例：第二层 `CLS` 完全读取第一层 `P1`，即 `A_2[CLS,P1]=1`；而第一层 `P1` 完全读取原始 `P2`，即 `A_1[P1,P2]=1`。虽然第二层 `CLS` 没有直接读取第一层 `P2`，原始 `P2` 仍可沿 `P2 → P1 → CLS` 到达最终 CLS。

所有中间位置 `j` 的路径权重相加：

~~~text
R_2[CLS, P2] = Σ_j A_2[CLS, j] × A_1[j, P2]
              = (A_2 @ A_1)[CLS, P2]
~~~

因此约定 `R_0 = I`，逐层计算 `R_l = A_l @ R_(l-1)`。**后层矩阵在左，先层矩阵在右**：这样 `R_l[i,j]` 才表示第 `l` 层位置 `i` 对原始位置 `j` 的近似路径权重。这里的“路径权重”不是精确的输出贡献，因为真实 Transformer 还作用了 Value 投影、输出投影、MLP 和归一化等。

### 2.1 残差为什么不能忽略

Encoder Block 的注意力子层并非只输出 `Attention(x)`，而是有 `x + Attention(x)`。原位置的信息可沿 `x` 直接通过，不需要经过注意力矩阵中其他 token 的边。Rollout 通常用单位矩阵 `I` 近似这条自连接：

~~~text
Ã_l = row_normalize(A_l + I)
R_0 = I
R_l = Ã_l @ R_(l-1)
~~~

若 `A_l` 本身每行和为 1，那么 `(A_l + I)` 每行和为 2，故此时 `Ã_l = (A_l + I)/2`；实现仍显式按行归一化，便于校验数值。比如一层 `CLS` 完全读取 `P1`，加入残差后，在这一近似中它会分别保留 0.5 的 CLS 自身路径和 0.5 的 P1 路径。这里的 0.5 只是 Rollout 的建模约定，**不是**真实网络中残差和注意力分支贡献严格各占一半。

最终取 `R_depth[CLS, patch]`，不取整行：CLS 自身路径不对应图像 patch。若为了画图把 patch 值重新缩放到 `[0,1]`，要注明那是可视化处理，不再是原矩阵行概率。

## 3. 多头先如何合并

每层有 6 个 attention heads，返回的权重形状为 `[B, H, N, N]`；前一节的矩阵乘法要求每层每张图只有一个 `[N,N]` 矩阵。第一版固定为**按 head 等权平均**：

~~~text
A_l[b, i, j] = (1/H) Σ_h A_l[b, h, i, j]
[B, H, N, N] --mean(dim=1)--> [B, N, N]
~~~

假如两个头在 CLS 行对 `[CLS,P1,P2]` 的权重分别是 `[0.1,0.7,0.2]` 与 `[0.2,0.1,0.7]`，等权平均得到 `[0.15,0.4,0.45]`。它表示两个头的平均关注关系，不表示模型只使用了一个“平均头”。真实多头输出仍是各头的 Value 加权结果拼接后，再经过可学习的输出投影；我们仅在**事后画 Rollout 图**时平均注意力矩阵。

每个头的注意力按行 softmax、行和为 1，所以有限且有效时等权平均后仍每行和为 1。随后再加单位矩阵表示残差，并按行归一化，再与前面层的 Rollout 矩阵相乘。对上述 CLS 行，加入残差后是 `[0.575,0.2,0.225]`；这个比例只是所选可视化规则下的近似路径权重。

第一版不用 `max` 融合：它容易被某一个头的高权重主导，而且逐位置取最大值后行和通常不再为 1，仍需额外归一化。等权平均并不“最正确”，只是简单、可复现、不按结果挑选的方法。可在未来作为独立可视化敏感性分析比较其他融合规则；不能从均值图推断各头的真实因果贡献。

## 4. 返回全部层注意力的接口契约

现有 `MiniViT` 与 `TransferMiniViT` 接受 `return_attention: bool`。默认 `False` 返回 logits `[B,C]`；传 `True` 返回 `(logits, last_attention)`，后者形状 `[B,H,N,N]`。`scripts/visualize.py` 和既有测试依赖这两个行为，因此不能突然让 `True` 改成返回全部层。

已在 `forward_features`、`forward` 和 `TransferMiniViT.forward` 增加 `attention_layers: Literal["last", "all"] = "last"`：

| 调用 | 返回 | 含义 |
|---|---|---|
| `model(x)` | `logits [B,C]` | 最常用路径，不收集注意力 |
| `model(x, return_attention=True)` | `(logits, [B,H,N,N])` | 兼容既有的末层图 |
| `model(x, return_attention=True, attention_layers="all")` | `(logits, [L,B,H,N,N])` | 按网络从浅到深收集全部 `L=depth` 层 |

若 `return_attention=False` 却要求 `attention_layers="all"`，或传入未知模式，直接抛出 `ValueError`，不默默忽略。`[L,B,H,N,N]` 的第一维是网络层次，不是 batch。每一层在最后一维上的行和都应接近 1，所有值有限；在 `eval()` 下三种调用的 logits 应一致，`all[-1]` 应与既有的 `last` 结果一致。

正常训练/推理仍走不收集注意力的分支，不额外堆叠所有层。只有需要 Rollout 时才请求 `all`；图像数量通常很少，以免 `[L,B,H,N,N]` 占用过多显存。`TransferMiniViT` 的 Linear Probe 冻结行为不能因新参数而改变，上述约定均由契约测试复核。

## 5. 全部层接口的实现验收

在 `MiniViT.forward_features` 内先验证参数组合：`attention_layers` 只接受 `last/all`；若设 `all`，还必须显式 `return_attention=True`。Encoder Block 循环中，普通路径不收集权重；`last` 只请求末层权重；`all` 每层都请求并放入列表，最后 `torch.stack` 得到 `[L,B,H,N,N]`。`forward` 将所选注意力与 logits 一起返回。`TransferMiniViT.forward` 将同一参数传给父类的 `forward_features`，保留 Linear Probe 的 `no_grad` 与 encoder `eval` 规则。

契约测试检查：普通、末层、全部层三种调用在 `eval()` 下 logits 相同；`all[-1]` 与旧接口完全一致；每层行和约为 1 且数值有限；无效参数组合会报错；Linear Probe 收集全部层后，encoder 仍无梯度而 classifier 仍可反传。既有可视化脚本继续使用旧接口。定向测试 91 passed，完整套件 443 passed。

## 6. Rollout 数值函数

`src/visualization.py` 的 `attention_rollout(all_attention)` 接受 `[L,B,H,N,N]`，返回 `[B,N-1]`，即最终 CLS 到各图像 patch 的近似路径权重。步骤：检查形状、有限性、非负性和每行和为 1；各 head 等权平均；加单位矩阵近似残差并按行归一化；按 `joint = layer_attention @ joint` 从浅层迭代到深层；取最终 joint 的 `[:, CLS, patches]`。输入若为 float16/bfloat16，连乘先转为 float32 以减轻累计舍入误差。

**这里没有把 patch 权重再次归一化到和为 1**：总质量的一部分可以留在 CLS 自身路径。测试用 `CLS → P1 → P2` 的两层例子区分矩阵乘法方向，验证残差与多头均值，并用真实 MiniViT 的全部层输出做端到端数值冒烟。下一节将 `[B,N-1]` reshape、插值并叠加图片；这些显示用缩放不得被误认作路径权重本身。

## 7. 从 patch 权重到显示热图

`patch_weights_to_heatmaps(weights, image_size)` 将 `[B,P]` 先恢复成 `[B,1,√P,√P]` patch 网格，再用双线性插值扩到 `[B,image_size,image_size]`，最后**每张图独立**做 min-max 缩放到 `[0,1]`。它接受 Rollout 输出；原有 `attention_to_heatmaps` 也复用同一个转换步骤，所以末层图与 Rollout 图遵循相同显示规则。若所有 patch 权重相同，热图全零，表示在此色彩约定下没有空间差异，而不是“模型完全没有关注图片”。

数值含义要分开：`attention_rollout` 的 `[B,P]` 保留近似路径权重；`patch_weights_to_heatmaps` 的 `[B,H,W]` 仅表示图内相对高低。两张图的原始权重即使相差十倍，只要空间比例相同，独立缩放后的热图也可能一样。因此热图颜色不能直接比较不同样本的绝对关注强度，更不能当成因果贡献。所有叠图均写明方法标签和相对色彩缩放。

## 8. 单图推理：从 checkpoint 到可用结果

入口 `scripts/predict.py` 支持直接运行与 `python -m scripts.predict`。`--attention` 可选 `none / last / rollout / both`，`--device` 可选 `auto / cpu / cuda`。结果保存为 `prediction.json`；请求注意力时另存 `attention.png`。前者包括 checkpoint SHA-256、epoch、数据集、完整类别顺序、预处理说明、图片哈希、top-k 和原始 patch 权重。Softmax 分数未经校准，不能视作经过可靠性验证的正确概率。

### 8.1 代码块各自负责什么

| 文件或函数 | 功能 |
|---|---|
| `src/models.build_model` | 共用模型工厂，训练与推理依据同一模型配置构建结构；旧的 `scripts.train.build_model` 导入仍可使用 |
| `src/inference.load_inference_runtime` | 校验 checkpoint，识别数据集与类别映射，严格加载完整权重并切到 eval 模式 |
| `InferenceRuntime.prepare_image` | 处理 EXIF、RGB，复用确定性的正式评估变换，并保留实际输入像素 |
| `src/inference.predict_batch` | 单次前向得到分类与所选注意力；处理 head 均值、Rollout 和显示热图 |
| `top_k_predictions` | 概率降序排列；同分按标签索引排序，拒绝越界 k |
| `plot_attention_comparison` | 原图、末层图、Rollout 并排显示，并标明相对颜色与解释边界 |
| `scripts/predict.py` | 解析命令、读单图、调用共用推理模块、保存 JSON/PNG，给出清楚的路径与格式错误 |
| `scripts/visualize_rollout.py` | 按固定验证集选择规则生成配对诊断案例；复用同一推理与绘图逻辑 |

### 8.2 预处理与类别为何要绑定 checkpoint

CIFAR-10 与 CORe50 恰好同为十类，但标签含义不同。加载器根据项目现有 checkpoint schema 识别 CIFAR 的 `data_dir` 或 CORe50 的 `transfer + manifest` 配置；若配置含义不清或分类头维度与映射不符，直接报错。类别不从图片文件名猜测。CORe50 的目标 checkpoint 已含训练好的新分类头，单图推理无需下载数据集或再次加载源域 checkpoint。

原生 32×32 CIFAR 图像复用 `build_transforms("none")` 的评估变换；外部尺寸的 CIFAR 输入先明确使用 bilinear+antialias 缩到 32×32。CORe50 复用 `build_core50_transforms("none")`，包含固定的 32×32 resize。二者归一化都沿用源任务的 mean/std，且不使用随机裁剪或翻转。热图叠在**实际缩放后的模型输入**上，避免把 32×32 的信息路径错当成高分辨率定位结果。CNN checkpoint 也可执行普通分类；注意力模式仅适用于 MiniViT。

这里的外部图像缩放是直接压成正方形，长方形图片可能产生形状拉伸。它是本版明确记录的输入约定，不等同于已经验证过的任意现实照片泛化。

### 8.3 可以直接运行的命令

~~~powershell
# 目标域 CORe50：使用本阶段已经保存的原始验证样本
python -m scripts.predict --checkpoint outputs/advanced/stage16/formal/fine_tune/train_42/best.pt --image assets/advanced/stage17/samples/core50_fine_tune_correct_1.png --top-k 5 --attention both --device auto --output-dir outputs/predictions/core50_demo

# CIFAR-10：只保存 Rollout，可将 --image 替换为自己的图片路径
python -m scripts.predict --checkpoint outputs/advanced/stage12/formal/r3_cutmix_alpha05/split_42__train_42/best.pt --image assets/advanced/stage17/samples/advanced_minivit_final_correct_1.png --top-k 5 --attention rollout --device auto --output-dir outputs/predictions/cifar_demo

# 重建固定的四组对照图
python -m scripts.visualize_rollout --config configs/advanced/stage17/diagnostics.yaml
~~~

复用同一输出目录会更新此前生成的报告和图；需要保留多次推理时应指定不同目录。输出路径若会覆盖输入图片或 checkpoint，命令明确拒绝。Gradio 是工作流中的可选展示层，本阶段采用上述 CLI 作为完整推理入口，无新增依赖。

## 9. 固定案例与观察

案例协议位于 `configs/advanced/stage17/diagnostics.yaml`，固定两个 checkpoint 的哈希。只读取 validation：CIFAR 使用官方 train 数据中的 split-42 验证索引；CORe50 使用冻结 manifest 的 s8。按验证集原始顺序，为 clean 预测正确/错误各选前两张**不同真实类别**的图片，最多扫描 512 张；不按热图是否好看筛选。实际 CIFAR 扫描 7 张、CORe50 扫描 159 张就满足条件。

对每张选中的原始图片，配对展示 clean 与 Gaussian noise severity 2（std=0.08、seed=1501）。扰动先作用于 `[0,1]` 像素，再归一化，顺序与阶段 15 相同。正确/错误分组由 clean 预测决定；扰动后重新计算预测，标题如实标明其是否改变。

| 产物 | 内容 |
|---|---|
| `assets/advanced/stage17/advanced_minivit_final_correct_comparison.png` | CIFAR 两个 clean 正确案例及其噪声版本 |
| `assets/advanced/stage17/advanced_minivit_final_wrong_comparison.png` | CIFAR 两个 clean 错误案例及其噪声版本 |
| `assets/advanced/stage17/core50_fine_tune_correct_comparison.png` | CORe50 两个 clean 正确案例及其噪声版本 |
| `assets/advanced/stage17/core50_fine_tune_wrong_comparison.png` | CORe50 两个 clean 错误案例及其噪声版本 |
| `assets/advanced/stage17/diagnostics.json` | 16 行完整案例、标签与预测、原始末层/Rollout 权重、输入与 checkpoint 哈希、配置/源码哈希 |
| `assets/advanced/stage17/samples/` | 8 张原始案例图片，可直接用于单图 CLI |

在所选 CIFAR 案例中，frog 与 bird 加噪后仍预测正确；truck 的 clean 错误预测 automobile 加噪后变为 truck；deer 由错误的 cat 变为错误的 frog。CORe50 两个 clean 正确样本（plug_adapters、mobile_phones）加噪后分别错判为 scissors、glasses；两个 clean 错误样本加噪后仍错。这些仅是固定案例，不能由“某一张图片加噪后变对”推出噪声总体有益。

末层图与 Rollout 在这些图中呈现不同的空间分布；即使图中高权重落在物体附近，也可能同时包含手部或背景信息。错误预测也能产生集中的热区。因此“热图看起来合理”不能验证分类正确、对象定位或因果解释。案例用于理解和诊断，不替代阶段 15/16 的整体评估，本阶段未使用 test 图片重新选择配方。

可直接打开 [CIFAR 正确案例](../assets/advanced/stage17/advanced_minivit_final_correct_comparison.png)、[CIFAR 错误案例](../assets/advanced/stage17/advanced_minivit_final_wrong_comparison.png)、[CORe50 正确案例](../assets/advanced/stage17/core50_fine_tune_correct_comparison.png)、[CORe50 错误案例](../assets/advanced/stage17/core50_fine_tune_wrong_comparison.png)。

## 10. 总验收与复现边界

定向测试覆盖旧接口、各层权重与矩阵方向、原始权重/显示值区分、类别映射与正式预处理一致、top-k 排序、无效图片/路径/配置、输入文件保护、固定案例选择以及 CPU/CUDA 可移植性。最后完整测试套件为 **463 passed**，Git diff 空白检查通过。四张最终对照图已实际打开检查，诊断 JSON 的源码哈希与当前文件一致。

使用真实冻结 checkpoint，各自实际运行了 CPU 与 CUDA 的 `--attention both`：CIFAR 示例 top-1 都为 frog（约 97.42%），CORe50 示例 top-1 都为 plug_adapters（约 99.94%）。结果位于 `outputs/advanced/stage17/smoke/{cifar_cpu,cifar_cuda,core50_cpu,core50_cuda}/`。GPU/CPU 浮点结果不要求逐位相同：小模型测试中观察到默认 cuDNN/TF32 引起约 `1.76e-4` 的 logits 差异、`2.60e-6` 的原始 Rollout 差异，契约测试采用有依据的数值容差并要求首选类别一致。

当前 Codex 沙箱的系统临时目录在验收时发生写入权限错误，因此测试命令将临时文件放进项目 `outputs/`；没有为此修改模型、基准算法或重新安装依赖。复现该验收命令：

~~~powershell
New-Item -ItemType Directory -Path outputs/stage17_system_temp -Force | Out-Null
$env:TEMP = (Resolve-Path outputs/stage17_system_temp).Path
$env:TMP = $env:TEMP
python -m pytest -q -p no:cacheprovider --basetemp outputs/stage17_verified_pytest
~~~

这些环境变量只影响当前终端进程；`--basetemp` 指定的目录应专用于 pytest。两套真实模型的 checkpoint 哈希保持冻结值，本阶段没有新增训练或修改正式实验指标。代码、配置、案例图片和诊断 JSON 由 Git 跟踪；单次推理的 `outputs/` 文件按项目既有约定不进入 Git。下一阶段是阶段 18 的最终冻结、报告与包装。
