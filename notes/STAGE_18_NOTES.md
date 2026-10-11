# 阶段 18：冻结评估、报告与包装

本阶段不再开发性调参，也不新增训练。补齐冻结模型的最终测试，把阶段 0–17 的证据整理成可复核的项目结论和展示材料。

开始日期：2026-10-11

当前状态：进行中

当前位置：已核对冻结 checkpoint 与评估入口；尚未执行 seed 123、2026 的最终 test。

## 进度

- [x] 确认本阶段范围与已有结果的复用规则
- [x] 核验三个冻结 checkpoint 的 SHA-256
- [x] 核验 CIFAR-10 评估入口与冻结版本一致
- [x] 补齐最终评估协议：运行版本、输出位置与产物身份
- [ ] 仅补评估 seed 123、2026，复用 seed 42 的 clean test
- [ ] 汇总三个 seed 的 test mean 与 sample std
- [ ] 汇总 clean、corruption、迁移与效率证据
- [ ] 完成最终报告，保留失败实验及结论边界
- [ ] 整理复现命令、资源预算、项目介绍与技术提纲
- [ ] 更新 README、结果总表与工作流
- [ ] 完成最终代码、图表与仓库验收

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
| CIFAR-10 clean，seed 123、2026 | 对各自冻结 `best.pt` 补一次完整评估 | 尚未生成 |
| corruption 类型、强度与 seed | 复用阶段 15 冻结协议和结果 | `assets/advanced/stage15/` |
| CORe50 三种迁移方法的 test | 复用阶段 16 的最终结果 | `assets/advanced/stage16/test_summary.json` |
| 参数、计算量、延迟、吞吐和显存 | 复用阶段 14 的固定硬件测量 | `assets/advanced/stage14/benchmark_raw_5c1dbfb.json` |
| 正确/错误、clean/noise、Attention/Rollout | 复用阶段 17 的 validation 案例 | `assets/advanced/stage17/diagnostics.json` |

阶段 17 的案例来自 validation，只用于定性诊断，不当作新的 test 指标。已有冻结结果不覆盖、不改写。

### 1.2 mean ± std 表示什么

最终 clean test 对三个训练 seed 分别计算准确率，再计算均值与**样本标准差**（分母为 `n-1`）。它描述在同一数据划分下，训练随机性带来的结果波动。

三个 seed 共用同一个测试集，不能当作三个独立采样的数据集；`± std` 也不是置信区间，更不能凭均值大小直接宣称统计显著。

## 2. 报告与交付清单

报告按“问题 → 方法 → 证据 → 限制”组织，覆盖基础 ViT/CNN 对照、优化与正则化、多 seed、精度—效率、鲁棒性、迁移和 Rollout。失败或接近的实验也保留，避免只展示获胜方案。

同步整理：复现命令与资源预算、60 秒项目介绍、5 分钟技术提纲，以及只使用已核验数字的简历表述。

最终验收需区分“已有环境通过”和“干净环境验证”。目前不能提前声称重新安装的干净环境已通过，也不在本阶段默认上传 GitHub、创建 release 或替用户选择许可证。
