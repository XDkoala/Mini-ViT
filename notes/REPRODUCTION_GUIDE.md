# MiniViT 复现入口与资源预算

复现分为查看已保存证据、使用原始冻结 checkpoint 推理，以及重新训练新实验三种情况。`data/` 和 `outputs/` 不进入 Git，克隆仓库不等于已经获得数据或权重；重新训练也不能保证文件哈希或指标逐位相同。

当前状态（2026-10-11）：阶段 0–18 已完成并上传 GitHub。本指南区分已验证的本地入口与尚未验证的从零数据/权重获取，不把仓库发布等同于完整实验产物分发。

## 1 环境与前置条件

历史验收环境为 Windows、Python 3.14、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130 和 RTX 5060 Laptop GPU。`requirements.txt` 固定 CUDA 13.0 wheel 与主要依赖版本，传递依赖未逐项锁定；其他系统或没有兼容 CUDA 驱动时，不保证该清单可直接安装。2026-10-11 已在独立新环境按该清单安装成功，完整包版本见[环境验收 JSON](../assets/advanced/stage18/environment_acceptance.json)。

在仓库根目录、已有 Python 3.14 的 PowerShell 中：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

已有 `.venv` 时只需激活，不重复创建。若 PowerShell 禁止激活，可直接用 `.\.venv\Scripts\python.exe` 替代后文的 `python`；不必修改全局执行策略。

## 2 不训练也能复核的证据

| 问题 | 冻结证据 |
|---|---|
| 基础 ViT/CNN 与结构消融 | [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) |
| 优化、正则化与三 seed validation | [ADVANCED_EXPERIMENT_RESULTS.md](ADVANCED_EXPERIMENT_RESULTS.md) |
| 最终配方与权重身份 | [ADVANCED_MINIVIT_FINAL.md](ADVANCED_MINIVIT_FINAL.md) |
| 三 seed clean test | [clean_test_summary.json](../assets/advanced/stage18/clean_test_summary.json) |
| 效率 | [benchmark_summary_5c1dbfb.csv](../assets/advanced/stage14/benchmark_summary_5c1dbfb.csv) |
| 扰动与全部 severity | [robustness_conditions.csv](../assets/advanced/stage15/robustness_conditions.csv) |
| 迁移与逐类/错误信息 | [test_summary.json](../assets/advanced/stage16/test_summary.json) |
| Attention/Rollout | [diagnostics.json](../assets/advanced/stage17/diagnostics.json) |

图表保持基础训练曲线、混淆矩阵、错误案例、Attention，以及阶段 15 的扰动网格和阶段 17 的四张固定对照图。它们分别属于既定 test 或 validation 协议，不能合并当作新评估样本。

## 3 有冻结权重时的单图演示

先按冻结清单确认本地 `best.pt` 存在且 SHA-256 一致。只需 checkpoint 和图片，不需要训练集或源域 checkpoint。

```powershell
python -m scripts.predict --checkpoint outputs/advanced/stage16/formal/fine_tune/train_42/best.pt --image assets/advanced/stage17/samples/core50_fine_tune_correct_1.png --attention both --device auto --expected-sha256 14e52af049050e20550507feff08bc79874edf66a1b544778972629089b95841 --output-dir outputs/predictions/reproduction_demo
```

输出 `prediction.json` 和 `attention.png`。CLI 中的 top-k softmax 分数未经校准；CORe50 图片应使用 CORe50 分类头，不能用 CIFAR-10 checkpoint 来解释目标标签。

此命令于 2026-10-11 在原环境和独立新环境均实跑，使用 CUDA，top-1 为 `plug_adapters` 99.94%，JSON 和 PNG 均已生成。它使用本地原始冻结权重，不替代远程权重获取验收。

## 4 在独立目录启动新实验

以下命令只供用户主动启动，不在整理报告时执行。CIFAR-10 应已准备在 `data/`；缺失时数据接口会尝试下载，受网络与证书影响。不要复用原实验输出目录。示例目录如已存在，应换一个新名称；只有明确续训才复用运行目录。

```powershell
# Isolated smoke, excluded from formal statistics. / 独立冒烟，不进入正式统计。
python -m scripts.train --config configs/advanced/stage12/formal/r3_cutmix_alpha05.yaml --epochs 2 --max-batches 2 --output-dir outputs/reproduction/smoke_cutmix

# Full new run; never overwrite the original checkpoint. / 完整新运行，不覆盖原权重。
python -m scripts.train --config configs/advanced/stage12/formal/r3_cutmix_alpha05.yaml --output-dir outputs/reproduction/cutmix_seed42
python -m scripts.train --config configs/advanced/stage13/r3_cutmix05_seed123.yaml --output-dir outputs/reproduction/cutmix_seed123
python -m scripts.train --config configs/advanced/stage13/r3_cutmix05_seed2026.yaml --output-dir outputs/reproduction/cutmix_seed2026
python -m scripts.aggregate --input-root outputs/reproduction
```

正式训练前保存 clean Git 版本；三条长训练顺序运行。聚合不读取 test，检查其 skipped 表，smoke 应被排除。重评训练所得模型前，需为新运行单独冻结路径与哈希，不能改写本轮 `final_evaluation.json`。

CORe50 原始归档与解压目录按[数据卡](DATASET_CARD.md)准备。准备 manifest 时也使用新产物目录，避免覆盖既有数据卡摘要：

```powershell
python -m scripts.prepare_core50 --manifest data/core50/manifests/reproduction.csv --summary outputs/reproduction/core50_dataset_summary.json
```

注意：上面的新 manifest 路径不会自动被原始迁移 YAML 使用。原配置的 `data.manifest` 指向 `data/core50/manifests/stage16_core50.csv`。若使用 `reproduction.csv`，先复制迁移 YAML 为新的复现配置，修改 `data.manifest`，并将其实际 SHA-256 写入 `data.manifest_sha256`；不要修改历史配置或降低哈希校验要求。可用 PowerShell 的 `Get-FileHash data/core50/manifests/reproduction.csv -Algorithm SHA256` 核对。

原始配置的迁移入口为 `python -m scripts.train_transfer --config configs/advanced/stage16/fine_tune.yaml --output-dir outputs/reproduction/core50_fine_tune`，仅在其指定的原 manifest 与**原始冻结源域权重**均存在且哈希匹配时可直接使用。采用新 manifest 时，必须将命令的 `--config` 换成上一步创建的复现配置。如果使用自己新训练的源模型，还需记录新 source checkpoint 与哈希，并对 Scratch/Probe/Fine-tune 一起冻结；不能假定新权重会匹配历史文件哈希。

阶段 14–18 的 benchmark、corruption 和最终 test 入口均使用冻结协议，主要用于复核既定运行。特别是 `python -m scripts.final_evaluation` 已执行并拒绝覆盖；不把它作为每次实验都要重跑的通用命令。

## 5 资源预算

| 项目 | 已观测资源或时间 | 解释 |
|---|---|---|
| 三个源域 CutMix 运行 | epoch time 合计分别 26.91、27.25、36.61 分钟 | 来自 100 轮 history；不是新设备上的耗时承诺 |
| CORe50 Scratch / Probe / Fine-tune | 合计分别 10.98、8.86、11.17 分钟 | 每组 50 轮；不含下载和数据准备 |
| ViT AMP 训练峰值 allocated | 约 632.59 MiB | batch 128 的基准；不等于全部进程显存需求 |
| ViT FP32 模型 state dict | 约 10.30 MiB | 完整训练 checkpoint 约 30.92 MiB；每组还保存 best/last、日志 |
| CORe50 原始归档 | 5,892,103,007 字节，约 5.49 GiB | 解压图片需额外空间，下载等待不计入训练时间 |

计时只反映历史运行，可能受系统负载与硬件状态影响；为数据解压、运行日志和测试临时目录额外留空间。不要用峰值 allocated 数字来判断任意显卡一定能够完成训练。

## 6 测试与发布边界

```powershell
python -m pytest -q -p no:cacheprovider --basetemp outputs/reproduction_pytest_temp
```

`--basetemp` 是 pytest 专用目录，重复使用时会清理其旧内容，不能指向模型目录或重要文件。若系统临时目录受限，将当前进程的 `TEMP`/`TMP` 指向 `outputs/` 下另一个专用临时目录；不修改全局环境变量。

2026-10-11 原环境和独立新环境的全量回归均为 **479 passed**；新环境还通过依赖检查、2轮受限冒烟和单图推理。验证复用了本地数据及原始 checkpoint，不等同于干净克隆后从零获取数据/权重；主要入口已验证，不表示每组正式长训练都重跑。记录见[环境验收 JSON](../assets/advanced/stage18/environment_acceptance.json)。

代码、轻量结果和文档已提交并上传 GitHub。完整数据与 checkpoint 未分发，远程权重获取未验收，也未创建 release。7 组基础旧评估缺原运行 commit 的历史例外已由项目所有者接受，详见[证据核验记录](../assets/advanced/stage18/evidence_audit.json)。原创代码与文档采用 [MIT 许可证](../LICENSE)，署名 Soren Yao；数据来源与归属见数据卡，不能把代码许可自动套用于数据集图片、含图片的可视化或其他第三方材料。完整授权范围见 [README 许可证说明](../README.md#许可证)。
