# 阶段 9：README 与仓库收尾

本阶段把已经完成的代码、实验和图表整理为他人可以理解与复现的最终项目入口，并对仓库内容做发布前检查。

完成日期：2026-09-23
当前状态：已完成

## 进度

- [x] 将空白 README 重构为最终项目说明
- [x] 写明 MiniViT 架构与完整 shape 数据流
- [x] 写明环境安装、数据准备和测试命令
- [x] 写明冒烟训练、正式训练和断点续训命令
- [x] 写明消融实验、最终评估和可视化命令
- [x] 加入 MiniViT、CNN 与五个消融实验的最终结果
- [x] 嵌入训练曲线、混淆矩阵、错误案例和 Attention 图
- [x] 解释主要实验结论与 Attention 的解释边界
- [x] 记录可复现性约束、已知限制和后续方向
- [x] 检查 Markdown 本地链接与图片路径
- [x] 检查 Git 未跟踪数据、权重、虚拟环境或运行输出
- [x] 清除文档中的个人绝对路径
- [x] 运行最终全量自动化测试

## 1. README 的定位

README 不再是项目规划稿，而是仓库首页。读者无需先阅读阶段笔记，即可从 README 回答：

1. 项目实现了什么；
2. 为什么比较 MiniViT 与 CNN；
3. MiniViT 输入输出 shape 如何变化；
4. 如何安装环境并运行测试；
5. 如何训练、续训、评估和生成图表；
6. 最终结果是什么；
7. 三组消融实验说明了什么；
8. 项目的限制和下一步是什么。

详细学习过程仍保留在 `notes/`，README 只呈现理解和复现项目所需的主线信息。

## 2. 最终 README 结构

```text
项目简介与亮点
  → 最终结果与核心结论
  → MiniViT 架构和 shape
  → 正式可视化结果
  → 项目目录与模块职责
  → 环境安装和快速开始
  → 训练、续训与消融命令
  → 冻结模型评估与可视化命令
  → 配置、测试和可复现性
  → 已知限制与后续方向
  → 学习笔记与参考资料
```

结果数字统一来自 `notes/EXPERIMENT_RESULTS.md`；图片统一来自 `assets/`，没有在 README 中制造新的实验事实。

## 3. 复现入口

README 提供以下可直接运行的入口：

```powershell
python -m pytest -q

python -m scripts.train --config configs/minivit.yaml --epochs 2 --max-batches 2 --output-dir outputs/smoke_minivit
python -m scripts.train --config configs/minivit.yaml
python -m scripts.train --config configs/minivit.yaml --resume outputs/minivit_baseline/last.pt

python -m scripts.evaluate --checkpoint outputs/minivit_baseline/best.pt
python -m scripts.visualize
```

大型 checkpoint 和 CIFAR-10 数据不提交 Git，因此最终评估前必须先完成相应训练。README 已明确说明这一点。

## 4. 仓库卫生检查

Git 跟踪清单中没有：

- `.venv/`；
- `data/`；
- `outputs/`；
- `*.pt`、`*.pth` 或 `*.ckpt`；
- 本地环境变量或 IDE 配置。

阶段 0 笔记中原有的三处本机绝对路径已改为 `<项目目录>` 占位符。仓库文档和源代码不再包含用户目录或桌面目录的机器专属前缀。

## 5. 最终验收

### 5.1 自动化测试

```text
184 passed
```

pytest 仍报告一个 Windows `.pytest_cache` 创建警告：目标缓存目录已存在。该警告不影响测试执行、代码结果或正式产物。

### 5.2 文档和资源

- README 中所有本地 Markdown 链接均可解析；
- 7 张正式 PNG 均存在；
- 工作流、阶段笔记和实验结果总账均位于 `notes/`；
- README 结果与 `EXPERIMENT_RESULTS.md` 一致；
- `git diff --check` 无补丁格式错误。

### 5.3 已知非阻塞项

仓库目前没有 LICENSE 文件，因此 README 明确提醒在复用或分发前补充许可证。这不影响代码运行和阶段 9 的文档闭环，但正式开放协作前应选择合适许可证。

## 6. 阶段结论

项目现在具备完整的学习记录、自动化测试、可复现实验入口、最终指标、图表和仓库首页。阶段 0–9 的基础项目主线已经闭环。

后续若加入 scheduler、AMP、Mixup/CutMix、多随机种子或新数据集，应创建新的实验阶段和独立评估协议，不覆盖当前冻结结果。
