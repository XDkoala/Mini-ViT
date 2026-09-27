# 阶段 14：效率与资源基准

本阶段回答：模型获得当前准确率需要多少参数、计算、时间、显存和存储空间。所有性能数据必须来自固定协议下的重复测量，不能用单次终端耗时替代基准测试。

开始日期：2026-09-27

当前状态：已完成

当前位置：静态分析、CUDA benchmark、原始数据保存与结果分析均已完成

## 进度

- [x] 确定阶段目标与候选模型
- [x] 明确 Parameters 与 Model Size
- [x] 明确 MACs/FLOPs、Latency、Throughput、Memory
- [x] 冻结设备、输入、precision、warmup 与重复次数
- [x] 实现并测试 benchmark 脚本
- [x] 保存原始 JSON/CSV
- [x] 完成 CNN、基础 MiniViT、advanced MiniViT 测量
- [x] 联合分析准确率与成本
- [x] 阶段 14 总验收

## 1. 候选矩阵

至少比较：

| 候选 | 作用 |
|---|---|
| CNN baseline | 具有卷积归纳偏置的效率与精度锚点 |
| MiniViT baseline | 基础 Transformer 锚点 |
| advanced MiniViT final | O3 + AMP + CutMix 0.5 的冻结最终模型 |

advanced MiniViT 与基础 MiniViT 的推理网络结构相同。CutMix、optimizer、scheduler、gradient clipping 都只属于训练过程，导出的推理模型中不存在这些运算。因此两者应具有相同参数量、理论 MACs 和 FP32 权重大小；验证准确率提升代表没有增加推理图结构复杂度，但实际 latency 仍需在统一协议下测量并视为重复运行噪声。

## 2. 指标分组

### 2.1 静态结构指标

- **Parameters**：可学习标量数量，与 batch size 无关；
- **MACs**：一次前向传播中的乘加次数；
- **FLOPs**：浮点运算次数，本项目明确采用 `1 MAC = 2 FLOPs`；
- **Model Size**：实际序列化权重文件大小，同时注明保存 dtype。

静态指标由模型结构决定，不应因 GPU 型号、系统负载或随机 seed 改变。

### 2.2 运行时指标

- **Latency**：完成一次推理需要多久；batch 1 报告 p50 与 p95；
- **Throughput**：单位时间处理的图片数；使用固定 batch size；
- **Peak Memory**：推理或训练过程中达到的峰值显存；训练与推理必须分开；
- **Data Time**：数据读取与预处理成本，和纯模型时间分开。

运行时指标依赖硬件、dtype、batch size、软件版本、温度和后台负载，不能脱离测量环境比较。

## 3. 为什么 latency 和 throughput 不是一回事

batch 1 latency 关心单张请求多久返回，适合实时相机或机器人控制；batch throughput 关心 GPU 一秒总共能处理多少张图，适合离线批处理。

更大的 batch 能提高 GPU 并行利用率，从而增加 images/s，却可能让单个 batch 等待更久。因此“吞吐量更高”不自动等于“单张响应更快”。本阶段固定：

~~~text
latency：batch_size = 1
throughput：batch_size = 128
input shape：[B, 3, 32, 32]
~~~

## 4. 公平计时的最低要求

正式测量必须：

1. 使用 `model.eval()` 与 `torch.inference_mode()`；
2. 先 warmup，使 CUDA context、kernel 与缓存进入稳定状态；
3. CUDA 计时边界调用 synchronize，防止只测到异步任务提交时间；
4. 重复多次保存原始样本，而不是只报一次；
5. latency 报告 median/p50 与 p95；
6. FP32 与 CUDA AMP 分开；
7. 每轮显存测量前重置 peak memory stats；
8. 记录 GPU、PyTorch、CUDA、输入 shape、batch 和 dtype。

第一轮 benchmark 只测纯模型前向，不包含 DataLoader 与磁盘读取。训练显存和数据管线成本作为单独项目记录，避免把不同来源的时间混成一个无法解释的数字。

## 5. 参数量与模型大小

参数量按所有可学习参数的元素数量求和：

~~~python
sum(parameter.numel() for parameter in model.parameters())
~~~

若参数均为 FP32，理论参数字节数为 `parameter_count × 4`。但必须区分：

- `parameters`：参与优化的可学习 Tensor；
- `model_state`：参数加持久 buffer，例如 BatchNorm running statistics；
- 训练 checkpoint：还包含 optimizer、scheduler、AMP scaler、history 和 config；
- 部署权重：通常只需 `model_state`，不需要 optimizer state。

对现有 best checkpoint 的只读拆分结果：

| 模型 | Parameters | 参数字节 | model_state | optimizer state | 完整 checkpoint |
|---|---:|---:|---:|---:|---:|
| MiniViT baseline | 2,693,578 | 10.275 MiB | 10.275 MiB | 20.551 MiB | 30.909 MiB |
| CNN baseline | 2,609,002 | 9.953 MiB | 9.962 MiB | 19.905 MiB | 29.891 MiB |
| advanced MiniViT | 2,693,578 | 10.275 MiB | 10.275 MiB | 20.551 MiB | 30.919 MiB |

AdamW 为每个参数维护一阶、二阶矩估计，因此优化器 Tensor 状态约为 FP32 参数字节的两倍。完整 checkpoint 约为纯模型状态的三倍并不表示推理需要三倍空间。

CNN 的 `model_state` 略大于其参数字节，因为 BatchNorm 的 running mean、running variance 和计数器属于 buffer：它们随模型保存和移动设备，但不由优化器更新。MiniViT baseline 与 advanced MiniViT 的推理结构相同；两者完整 checkpoint 相差约 0.01 MiB 只来自 history/config 等元数据，并非 advanced 模型更大。

## 6. MACs/FLOPs 口径与结果

本项目统计 Conv2d、Linear 与 Self-Attention 中的 `QK^T`、`attention @ V`；不统计 normalization、activation、softmax、pooling、残差加法、reshape/transpose 和内存搬运。bias addition 也不计入。采用 `1 MAC = 2 FLOPs`：

| 模型 | MACs / image | FLOPs / image |
|---|---:|---:|
| CNN | 115.906 M | 231.811 M |
| MiniViT | 182.850 M | 365.699 M |
| advanced MiniViT | 182.850 M | 365.699 M |

MiniViT 主干 MACs 约为 CNN 的 1.58 倍。计数器通过 forward hooks 统计 Conv/Linear，并显式补充 Attention 的两次矩阵乘法；专项测试用手算结果锁定 CNN `115,905,536` 与 MiniViT `182,849,664` MACs。

## 7. 正式测量协议

~~~text
device：NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch / CUDA：2.14.0+cu130 / 13.0
input：[B, 3, 32, 32]
precision：FP32 与 CUDA FP16 autocast
latency：batch 1，50 warmup + 200 measured iterations
throughput：batch 128，20 warmup + 100 measured iterations
training memory：batch 128，2 warmup steps 后测一个完整 AdamW step
mode：eval + inference_mode（推理）；纯合成输入，不含 DataLoader
timing：每次 CUDA synchronize 前后包围真实前向
Git：5c1dbfb，worktree clean
~~~

原始逐次样本保存在：

- `outputs/advanced/stage14/benchmark_raw.json`；
- `outputs/advanced/stage14/benchmark_summary.csv`。

由于 `outputs/` 不进入 Git，commit `5c1dbfb` 对应的只读报告副本保存在 `assets/advanced/stage14/benchmark_raw_5c1dbfb.json` 与 `benchmark_summary_5c1dbfb.csv`，用于 GitHub 复核。后续重跑不得静默覆盖这两个带 commit 标识的归档。

训练显存使用普通硬标签合成 batch，反映相同网络的 forward/backward/AdamW 主体成本，不包含 CutMix 数据构造；因此 advanced MiniViT 与基础 MiniViT 应相同。reserved memory 是 CUDA caching allocator 向驱动保留的池，allocated memory 才是当前 Tensor 实际占用，两者不可混写。

## 8. 正式结果

### 8.1 Batch-1 latency

| 模型 | FP32 p50 / p95 | AMP p50 / p95 |
|---|---:|---:|
| CNN | 0.414 / 0.479 ms | 0.539 / 0.660 ms |
| MiniViT baseline | 1.692 / 1.931 ms | 2.440 / 2.606 ms |
| advanced MiniViT | 1.644 / 1.829 ms | 2.450 / 2.654 ms |

batch 1 下 AMP 反而更慢。输入太小时，autocast、dtype 转换和更多 kernel 调度开销超过 Tensor Core 的计算收益。不能因为 AMP 提高训练速度，就推断它一定降低在线单张延迟。

### 8.2 Batch-128 throughput

| 模型 | FP32 | AMP | AMP 加速 |
|---|---:|---:|---:|
| CNN | 31,229 img/s | 54,757 img/s | 1.75× |
| MiniViT baseline | 10,298 img/s | 20,274 img/s | 1.97× |
| advanced MiniViT | 10,417 img/s | 20,456 img/s | 1.96× |

batch 128 能提供足够并行工作，AMP 的吞吐收益显现。基础与 advanced MiniViT 的小幅差异属于重复运行噪声，而非权重使相同计算图产生稳定复杂度差异。

### 8.3 峰值 allocated 显存

| 模型 | 推理 FP32 | 推理 AMP | 训练 FP32 | 训练 AMP |
|---|---:|---:|---:|---:|
| CNN | 58.12 MiB | 81.78 MiB | 374.29 MiB | 254.42 MiB |
| MiniViT baseline | 75.02 MiB | 74.89 MiB | 860.43 MiB | 632.59 MiB |
| advanced MiniViT | 75.02 MiB | 74.89 MiB | 860.43 MiB | 632.59 MiB |

AMP 使 CNN 训练峰值 allocated 减少约 32.0%，使 MiniViT 减少约 26.5%。CNN 的 batch-1 AMP 推理峰值反而更高，说明 autocast 临时 workspace/cache 的固定开销在极小 workload 下不可忽略；这与其 batch-128 吞吐加速并不矛盾。

## 9. 精度与成本联合结论

| 模型 | Best Val Acc | Params | MACs | FP32 p50 | AMP throughput |
|---|---:|---:|---:|---:|---:|
| CNN baseline | **89.58%** | 2.609 M | **115.906 M** | **0.414 ms** | **54,757 img/s** |
| MiniViT baseline | 81.84% | 2.694 M | 182.850 M | 1.692 ms | 20,274 img/s |
| advanced MiniViT | 86.92% | 2.694 M | 182.850 M | 1.644 ms | 20,456 img/s |

在当前 CIFAR-10 小图像任务上，CNN 同时具有最高验证准确率、最低 MACs、最低延迟、最高吞吐量和最低训练显存，体现了卷积局部性与平移归纳偏置的优势。

advanced MiniViT 相对基础 MiniViT 提升 5.08 pp，而参数量、MACs、模型状态大小和显存相同；阶段 11–13 的优化主要改善了学习过程，没有增加部署端计算图。但 advanced MiniViT 仍比 CNN 低 2.66 pp，FP32 batch-1 延迟约为 CNN 的 3.97 倍，AMP batch-128 吞吐量约为 CNN 的 37.4%。

这不表示 Transformer 架构普遍低效，只说明在 32×32、数据量有限、模型规模约 270 万参数的当前实验条件下，CNN 是更好的精度—成本方案。MiniViT 的价值在于完成可解释的 Transformer 学习与后续注意力、迁移和鲁棒性研究。

## 10. 阶段总验收

参数量与已有记录一致；MACs 约定明确；正式 benchmark 包含 warmup、CUDA synchronize 和重复采样；latency 报告 p50/p95；throughput 固定 batch 与 precision；训练/推理显存分开；原始 JSON/CSV 已保存；全量回归测试为 351 passed。阶段 14 验收完成。
