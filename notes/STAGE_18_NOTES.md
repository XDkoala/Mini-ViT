# 阶段 18：冻结评估、报告与包装

本阶段不再开发性调参，也不新增训练。补齐冻结模型的最终测试，把阶段 0–17 的证据整理成可复核的项目结论和展示材料。

开始日期：2026-10-11

当前状态：已完成（接受基础旧结果的历史追溯例外）

完成日期：2026-10-11。仓库扫描、图表、权重与预测核验、独立环境安装/冒烟/推理/全量回归均通过；项目所有者已确认接受基础旧结果的历史追溯例外。

## 进度

- [x] 确认本阶段范围与已有结果的复用规则
- [x] 核验三个冻结 checkpoint 的 SHA-256
- [x] 核验 CIFAR-10 评估入口与冻结版本一致
- [x] 补齐最终评估协议：运行版本、输出位置与产物身份
- [x] 仅补评估 seed 123、2026，复用 seed 42 的 clean test
- [x] 汇总三个 seed 的 test mean 与 sample std
- [x] 汇总 clean、corruption、迁移与效率证据
- [x] 完成最终报告，保留失败实验及结论边界
- [x] 整理复现命令、资源预算、项目介绍与技术提纲
- [x] 更新 README、结果总表与工作流
- [x] 核验仓库候选文件、原始预测、权重身份与展示路径
- [x] 在独立虚拟环境按 requirements.txt 安装依赖
- [x] 在独立环境完成冒烟、推理与全量回归
- [x] 确认基础旧结果的历史追溯例外（保留缺口，不补造运行 commit）
- [x] 完成最终代码、图表与仓库验收（带上述例外）

## 1. 最终评估不是再调一次参

validation 用于选择配方和最佳 epoch；test 用于评价已经选定的模型。因此不能看完 test 后换 checkpoint、删掉不利 seed，或新增 seed 来改善均值。

完整配方、checkpoint 路径、SHA-256 和训练提交见 [冻结清单](ADVANCED_MINIVIT_FINAL.md)。本阶段仍使用 split seed 42，以及 train seeds 42、123、2026。

2026-10-11 核验：三个本地 `best.pt` 的 SHA-256 均与冻结清单完全一致；`scripts/evaluate.py` 的 Git blob 为 `4ce2edd171f61e2c579774a2efb98430b737389f`，也与冻结版本一致。该核验只读取文件，不访问测试图片。

准备评估时 HEAD 为 `4d9d4687111df1d215eb8045d2579fb1d315c3df`，但阶段 17 的实现与文档尚未提交。为避免把旧 HEAD 冒充当前评估版本，先将实现、测试及协议保存为本地提交，再从 clean worktree 执行评估；实际运行提交号保存在结果 JSON 中。

冻结协议为 `configs/advanced/stage18/final_evaluation.json`，入口为 `python -m scripts.final_evaluation`。入口在读测试图片前核对 clean commit、评估源码、三个权重和阶段 15 产物哈希，并拒绝覆盖已有运行目录。新增预测明细保存在 `outputs/advanced/stage18/clean_test/`，可提交的汇总保存在 `assets/advanced/stage18/clean_test_summary.json`。

口径备注：seed 42 复用阶段 15 的 FP32 batch 256；两个新增 seed 按冻结 checkpoint 的配置使用 FP32 batch 128。三者使用相同测试集与确定性测试变换，不做测试时增强；浮点运算可能存在批大小相关的微小差异，不通过重跑选择更有利结果。

### 1.1 哪些结果复用，哪些需要补齐

| 内容 | 本阶段处理 | 已有证据 |
|---|---|---|
| CIFAR-10 clean，seed 42 | 复用，不重复挑选输出 | `assets/advanced/stage15/robustness_raw.json` |
| CIFAR-10 clean，seed 123、2026 | 已对各自冻结 `best.pt` 补一次完整评估 | `assets/advanced/stage18/clean_test_summary.json` |
| corruption 类型、强度与 seed | 复用阶段 15 冻结协议和结果 | `assets/advanced/stage15/` |
| CORe50 三种迁移方法的 test | 复用阶段 16 的最终结果 | `assets/advanced/stage16/test_summary.json` |
| 参数、计算量、延迟、吞吐和显存 | 复用阶段 14 的固定硬件测量 | `assets/advanced/stage14/benchmark_raw_5c1dbfb.json` |
| 正确/错误、clean/noise、Attention/Rollout | 复用阶段 17 的 validation 案例 | `assets/advanced/stage17/diagnostics.json` |

阶段 17 的案例来自 validation，只用于定性诊断，不当作新的 test 指标。已有冻结结果不覆盖、不改写。

### 1.2 mean ± std 表示什么

最终 clean test 对三个训练 seed 分别计算准确率，再计算均值与**样本标准差**（分母为 `n-1`）。它描述在同一数据划分下，训练随机性带来的结果波动。

三个 seed 共用同一个测试集，不能当作三个独立采样的数据集；`± std` 也不是置信区间，更不能凭均值大小直接宣称统计显著。

## 2. 最终 clean test

正式补评估于 2026-10-11 从 clean commit `2ed2a93fb105f7518dd360adf105f334c8f02238` 执行，没有新增训练或改选模型。两个新增 seed 均完整评价官方 10,000 张测试图片。

| Train seed | Best epoch | Test accuracy | Test loss | 来源 |
|---:|---:|---:|---:|---|
| 42 | 98 | 86.38% | 0.436853 | 复用阶段 15 |
| 123 | 98 | 86.32% | 0.437951 | 阶段 18 补评估 |
| 2026 | 94 | 86.08% | 0.429282 | 阶段 18 补评估 |
| Mean ± sample std | — | **86.260 ± 0.159%** | — | 三个冻结 seed，全保留 |

可复核汇总：[clean_test_summary.json](../assets/advanced/stage18/clean_test_summary.json)。其中记录每组权重哈希、评估 commit、原始指标和预测明细的哈希。两个新增预测 CSV 均有 10,000 行，样本索引连续且无重复；正确数分别为 8,632 和 8,608，与保存准确率一致。

汇总文件 SHA-256：`4B3CA6FC3EAED43408B5286F4D8385FB74227C068C8C2CE3127927CF39D7569C`。评估命令为 `python -m scripts.final_evaluation`，已执行完成；它不是需要反复运行的训练命令，已有产物时会拒绝覆盖。此次只创建本地冻结提交，未上传 GitHub。

如何解读：三个 seed 的 clean test 表现接近，但这不证明跨数据划分、跨数据集或跨硬件同样稳定。seed 2026 的 validation 数值最高，test 却最低，也说明 validation 排名并不保证测试排名；不能据此反向改选 seed。相较基础 MiniViT 的单次 test 80.65%，最终均值数值上高 5.61 pp，但这不是多 seed 配对显著性检验，且当前 CNN 基线仍更高。

本微步验收：新增冻结/统计契约测试 14 passed；当前项目全量测试 **477 passed，117.92 秒**。测试使用项目 `outputs/` 内的临时目录，避免系统临时目录的权限限制。这是已有虚拟环境中的回归验证，不等同于干净环境重装验收。

## 3. 报告与交付清单

已完成以下材料，均为可编辑 Markdown：

| 文件 | 作用 |
|---|---|
| [FINAL_EXPERIMENT_REPORT.md](FINAL_EXPERIMENT_REPORT.md) | 按八个主题汇总基础到进阶结果，区分 validation/test、单 seed/多 seed与数值差异/显著性 |
| [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) | 环境、证据入口、单图演示、隔离新实验、历史资源预算和权重依赖 |
| [PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md) | 约 60 秒介绍、5 分钟提纲、简历表述与常见追问 |

报告保留无收益或偏离预期的实验：小 patch 和加深未改善基础结果，Label Smoothing 未超过 R0，warmup/clipping 未显示明确最终精度收益，短跑排名未延续，clip 持续触发，以及迁移 validation→test 明显下降。既不把它们删去，也不把可能的原因写成已经证明的因果关系。

资源预算直接来自完整训练 history 与 benchmark，不作为新设备上的时间/显存承诺。简历数字与报告一致，也如实保留 Codex 协作开发背景，不声称独立编写所有代码或生产部署。

材料核验：新文档的本地链接存在、代码围栏配对，关键指标与结构化产物一致，`git diff --check` 通过。训练、迁移、数据准备和聚合入口的 `--help` 均成功；没有执行新长训练或重新读取 test。复现指南中的单图命令已在已有环境实跑，CUDA 预测 `plug_adapters` 99.94%，保存 JSON 与末层/Rollout PNG 于 `outputs/predictions/reproduction_demo/`。本微步只修改文档，代码回归沿用上一微步的 477 passed。

最终验收区分已有环境、独立新环境与干净克隆。独立新环境已通过安装和运行检查，但复用了本地数据与原始冻结权重，不能声称完成从零下载的干净克隆验证。本阶段不默认上传 GitHub、创建 release 或替用户选择许可证。

## 4. 最终验收记录

### 4.1 仓库与证据

核验清单见 [evidence_audit.json](../assets/advanced/stage18/evidence_audit.json)。检查 Git 已跟踪与未忽略的待提交文件，没有发现原始数据/运行输出/虚拟环境、大权重、超过 10 MiB 的文件、机器绝对路径或常见密钥格式。密钥格式扫描不等于安全认证，图表中的轻量数据样例也不等同于分发整个数据集。

7 组基础 test、进阶 clean test 和三组迁移 test 的预测明细、样本顺序、正确数、指标和当前权重哈希均重新核对；没有重跑正式 test。阶段 15 与阶段 16 的原始结果文件哈希仍与冻结记录一致。

阶段 17 原诊断 JSON 的两个 checkpoint 展示路径改为项目相对路径，加入 `presentation_revision` 记录；未修改任何预测、热图权重、样本或原生成版本信息。原文件备份保存在 `outputs/stage18_acceptance/original_diagnostics.json`，SHA-256 为 `79668F96888157C9970E1CB2BE8223D181456BEB28B075329ADA0373B2369D50`。生成入口同步采用相对路径，新增两项契约测试；不要把原生成源码哈希误当作修订后的当前源码哈希。

训练曲线有 epoch、loss、accuracy 单位、图例与最佳 epoch；混淆矩阵有真实/预测方向、比例与计数；错误与 Attention 图有标签及预测分数；扰动网格有类型和 severity，Rollout 图保留相对尺度与非因果说明。图像示例不需要伪造坐标轴，统计图才需要相应坐标和单位。

### 4.2 历史追溯缺口

基础阶段的 7 组测试保存了 checkpoint 哈希、指标与逐样本预测，但**没有保存原评估运行的 Git commit**。可核对其结果首次归档于 `1ff4137`，但归档 commit 不等于运行 commit，不能事后补造。进阶 clean/corruption 结果有原运行 commit；迁移评估的冻结代码与协议可核对到 `5b59654`，运行上下文由阶段 16 笔记记录，而非原 JSON 内嵌。

2026-10-11，项目所有者在了解 Git commit 的作用后明确回复“OK听你的建议”，接受这些基础结果作为带限制的历史参照，并以此例外完成阶段 18 验收。报告继续披露缺口；原运行 commit 保持未知，不用归档 commit 替代，也不为补齐记录重训或重跑正式 test。

Git commit 标识一次保存的代码版本；checkpoint 的 SHA-256 标识具体权重文件。两者分别回答“用什么代码评估”和“评估哪份权重”。运行时工作区若有未提交修改，还需记录这些修改，不能只记 HEAD。缺少原运行 commit 不会自动推翻已核验的指标，但限制了旧评估代码版本的精确追溯。因此本次为**带明确历史例外的验收完成**，不声称全部历史结果均有完整运行版本证据。

### 4.3 独立环境

新环境保存在 `outputs/stage18_acceptance/venv/`，不继承原虚拟环境的 site-packages。首次 ensurepip 因系统临时目录权限失败，改为验收专用临时目录后成功。首次依赖安装因包源 DNS 解析失败而退出；获得联网权限后，严格按现有 requirements.txt 安装成功，未修改原 `.venv` 或固定依赖版本。安装日志保存在验收目录内，接着验证依赖导入、短冒烟、单图推理与全量测试。

最终结果见 [environment_acceptance.json](../assets/advanced/stage18/environment_acceptance.json)：

| 项目 | 结果 |
|---|---|
| 依赖隔离 | `include-system-site-packages=false`；在 `-I` 模式下核对主要依赖均来自新环境 |
| 依赖完整性 | `pip check` 通过；CUDA 可用，主要版本与 requirements.txt 一致 |
| README 冒烟 | 2 epochs，每轮 2 train/val batches，状态 completed；明确排除正式统计 |
| 单图推理 | CUDA，冻结 CORe50 Fine-tune，`plug_adapters` 99.94%；JSON/PNG 均生成 |
| 原环境全量回归 | **479 passed，152.99 秒** |
| 新环境全量回归 | **479 passed，154.97 秒** |
| 文档 | README 与 notes 共 29 份 Markdown 的本地链接均存在；diff whitespace 检查通过 |

requirements.txt 固定主要依赖，未将全部传递依赖逐项锁定；此次新环境的完整安装包版本已写入验收 JSON。验收复用本地 CIFAR-10 数据和原始目标域 checkpoint，没有新增正式长训练或重新进行官方 test 评估。原环境依赖检查也再次通过。

独立环境、安装日志、冒烟输出、预测和测试临时目录均保留在 Git 忽略的验收目录中，未自动删除；之后若需要释放空间，应明确选择专用验收目录，不触碰原实验权重。GitHub 上传、release 和代码许可证仍未操作。
