# 阶段 14：效率与资源基准

本阶段回答：模型获得当前准确率需要多少参数、计算、时间、显存和存储空间。所有性能数据必须来自固定协议下的重复测量，不能用单次终端耗时替代基准测试。

开始日期：2026-09-27

当前状态：进行中

当前位置：建立指标定义、候选矩阵与公平测量协议

## 进度

- [x] 确定阶段目标与候选模型
- [x] 明确 Parameters 与 Model Size
- [ ] 明确 MACs/FLOPs、Latency、Throughput、Memory
- [ ] 冻结设备、输入、precision、warmup 与重复次数
- [ ] 实现并测试 benchmark 脚本
- [ ] 保存原始 JSON/CSV
- [ ] 完成 CNN、基础 MiniViT、advanced MiniViT 测量
- [ ] 联合分析准确率与成本
- [ ] 阶段 14 总验收

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
