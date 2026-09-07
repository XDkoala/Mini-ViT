# 阶段 0：环境与项目骨架

本笔记记录 Mini ViT 项目阶段 0 的操作、原理和验收结果。

## 进度

- [x] 创建并使用独立 Python 虚拟环境
- [x] 安装并验证项目依赖与 CUDA
- [x] 创建项目目录和空文件
- [x] 填写并验证 `requirements.txt`
- [x] 配置 `.gitignore`
- [ ] 初始化 Git 并完成首次提交

## 1. Python 虚拟环境

### 1.1 首次创建

```powershell
Set-Location -LiteralPath 'D:\Desktop\Mini ViT'
python --version
python -m venv .venv
```

- `Set-Location`：进入项目目录；路径含空格，所以用引号包住。
- `python -m venv .venv`：用当前 Python 在项目内创建 `.venv`。
- 创建命令只需执行一次；第三方包会安装到 `.venv\Lib\site-packages`。

### 1.2 激活

```powershell
.\.venv\Scripts\Activate.ps1
```

激活后，提示符前通常出现 `(.venv)`。其原理是临时调整当前终端的 `PATH`，让 `python` 优先指向 `.venv`，并非启动虚拟机。

若 PowerShell 阻止脚本运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

`-Scope Process` 只影响当前 PowerShell，关闭终端后失效。

### 1.3 确认环境

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

两条命令的输出路径都应包含：

```text
D:\Desktop\Mini ViT\.venv
```

安装包时优先写 `python -m pip`，确保 pip 属于当前 Python，避免误装进全局环境。

### 1.4 退出与重新进入

退出：

```powershell
deactivate
```

退出不会删除 `.venv` 或其中的包。

下次打开 PowerShell：

```powershell
Set-Location -LiteralPath 'D:\Desktop\Mini ViT'
.\.venv\Scripts\Activate.ps1
```

不激活时也可直接调用：

```powershell
.\.venv\Scripts\python.exe --version
```

## 2. 安装项目依赖

### 2.1 安装 PyTorch（CUDA 13.0）

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

- `torch`：张量、自动求导、神经网络和 GPU 计算。
- `torchvision`：CIFAR-10、图像变换和视觉工具。
- `--index-url .../cu130`：从 PyTorch 官方源安装 CUDA 13.0 构建。

PyTorch wheel 自带所需的 CUDA 运行库。普通训练通常只需兼容的 NVIDIA 驱动，不必另外安装完整 CUDA Toolkit 或 cuDNN。

### 2.2 安装其余依赖

```powershell
python -m pip install numpy pyyaml matplotlib scikit-learn pytest
```

| 安装名 | 导入名 | 用途 |
|---|---|---|
| `numpy` | `numpy` | 数值处理 |
| `pyyaml` | `yaml` | 读取 YAML 配置 |
| `matplotlib` | `matplotlib` | 绘图 |
| `scikit-learn` | `sklearn` | 指标与混淆矩阵 |
| `pytest` | `pytest` | 单元测试 |

### 2.3 验收

```powershell
python -c "import sys, torch, torchvision, numpy, yaml, matplotlib, sklearn, pytest; print('Python:', sys.version.split()[0]); print('Interpreter:', sys.executable); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

实际执行一次 GPU 运算：

```powershell
python -c "import torch; x = torch.rand(3, 3, device='cuda'); print(x.device); print(torch.isfinite(x).all().item())"
```

预期输出包含 `cuda:0` 和 `True`。

### 2.4 本机验收记录（2026-09-07）

| 检查项 | 实测结果 |
|---|---|
| Python | `3.14.0` |
| 解释器 | `.venv\Scripts\python.exe` |
| PyTorch | `2.14.0+cu130` |
| torchvision | `0.29.0+cu130` |
| NumPy | `2.5.2` |
| PyYAML | `6.0.3` |
| Matplotlib | `3.11.1` |
| scikit-learn | `1.9.0` |
| pytest | `9.1.1` |
| CUDA 可用 | `True` |
| CUDA runtime | `13.0` |
| GPU | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| GPU 张量运算 | 通过，设备为 `cuda:0`，数值有限 |

结论：虚拟环境、项目依赖和 GPU 加速均已通过验收。

## 3. 项目目录骨架

### 3.1 创建命令

```powershell
New-Item -ItemType Directory -Path 'configs', 'src', 'scripts', 'tests', 'data', 'outputs', 'assets'
New-Item -ItemType Directory -Path 'src\models'
New-Item -ItemType File -Path 'README.md', 'requirements.txt', '.gitignore'
New-Item -ItemType File -Path 'configs\minivit.yaml', 'configs\cnn.yaml'
New-Item -ItemType File -Path 'src\__init__.py', 'src\config.py', 'src\data.py', 'src\engine.py', 'src\metrics.py', 'src\utils.py', 'src\visualization.py'
New-Item -ItemType File -Path 'src\models\__init__.py', 'src\models\minivit.py', 'src\models\cnn.py'
New-Item -ItemType File -Path 'scripts\train.py', 'scripts\evaluate.py', 'scripts\visualize.py'
New-Item -ItemType File -Path 'tests\test_data.py', 'tests\test_model.py', 'tests\test_tiny_overfit.py'
```

`New-Item` 创建文件或目录；`-ItemType` 指定类型；`-Path` 后可用逗号列出多个目标。首次创建文件时不使用 `-Force`，避免覆盖已有内容。

### 3.2 结构与职责

```text
Mini ViT/                              # 项目根目录
├── README.md                          # 项目介绍、安装方法、运行命令与实验结果
├── PROJECT_PLAN.md                    # 项目目标、技术路线和整体规划
├── BASIC_PROJECT_WORKFLOW.md          # 从环境搭建到项目完成的分阶段流程
├── STAGE_0_NOTES.md                   # 阶段 0 的命令、原理、进度和验收记录
├── requirements.txt                   # 项目所需的 Python 直接依赖
├── .gitignore                         # 声明不应提交到 Git 的文件和目录
├── configs/                           # 模型与训练超参数配置
│   ├── minivit.yaml                   # MiniViT 的结构和训练配置
│   └── cnn.yaml                       # CNN 对照模型的结构和训练配置
├── src/                               # 可复用的核心 Python 代码
│   ├── __init__.py                    # 将 src 标记为可导入的 Python 包
│   ├── config.py                      # 读取、解析和校验配置
│   ├── data.py                        # 下载、划分和加载 CIFAR-10
│   ├── engine.py                      # 训练、验证和 checkpoint 流程
│   ├── metrics.py                     # 准确率等评价指标
│   ├── utils.py                       # 随机种子、设备选择等通用工具
│   ├── visualization.py               # 曲线、混淆矩阵和注意力图
│   └── models/                        # 模型结构实现
│       ├── __init__.py                # 将 models 标记为 Python 子包
│       ├── minivit.py                 # Mini Vision Transformer 实现
│       └── cnn.py                     # CNN 对照基线实现
├── scripts/                           # 用户直接运行的程序入口
│   ├── train.py                       # 启动训练和断点续训
│   ├── evaluate.py                    # 加载权重并执行最终评估
│   └── visualize.py                   # 从日志和预测结果生成图表
├── tests/                             # pytest 自动化测试
│   ├── test_data.py                   # 测试数据数量、划分、变换和 shape
│   ├── test_model.py                  # 测试模型 shape、数值和梯度
│   └── test_tiny_overfit.py           # 测试模型能否记住极小数据集
├── data/                              # 数据集目录，不提交到 Git
├── outputs/                           # 权重、日志和实验输出，不提交大文件
└── assets/                            # README 展示用的可提交图片
```

`src` 保存可复用功能，`scripts` 只负责组合这些功能并启动任务。Git 不记录空目录，所以 `data`、`outputs` 和 `assets` 在没有文件时不会出现在提交中，这不影响使用。

### 3.3 验收结果

2026-09-07 已逐项检查：所有计划目录和空文件均存在，`src\__init__.py` 与 `src\models\__init__.py` 命名正确。

## 4. 依赖清单

`requirements.txt` 只记录项目主动选择的直接依赖，并固定已经验收的版本；pip 会自动安装它们的间接依赖。

```text
--extra-index-url https://download.pytorch.org/whl/cu130

torch==2.14.0+cu130
torchvision==0.29.0+cu130
numpy==2.5.2
PyYAML==6.0.3
matplotlib==3.11.1
scikit-learn==1.9.0
pytest==9.1.1
```

- `==` 固定准确版本，提高复现一致性。
- `+cu130` 指定 PyTorch 的 CUDA 13.0 构建。
- `--extra-index-url` 保留默认 PyPI，并增加 PyTorch CUDA 13.0 官方源。
- 不直接采用完整 `pip freeze`，避免把大量自动安装的间接依赖混入项目清单。

安装和检查：

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

`-r` 表示从文件读取安装要求；`pip check` 用于检查缺失依赖和版本冲突。

2026-09-07 验收通过：文件内容正确，所有包均可导入，`pip check` 输出 `No broken requirements found.`，CUDA 13.0 与 RTX 5060 正常可用。

## 5. Git 忽略规则

`.gitignore` 只排除本机环境、自动生成内容、数据、训练输出、权重和秘密配置；源码、配置、测试、文档及 `assets/` 保持可提交。

```gitignore
# Python environment and caches
/.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.ipynb_checkpoints/

# Data, experiment outputs and weights
/data/
/outputs/
*.pt
*.pth
*.ckpt

# Secrets, editors, operating systems and logs
.env
.env.*
!.env.example
/.vscode/
/.idea/
.DS_Store
Thumbs.db
*.log
```

规则要点：

- 开头的 `/` 将匹配限制在项目根目录；结尾的 `/` 表示目录。
- `*` 是通配符，`*.py[cod]` 匹配 `.pyc`、`.pyo` 和 `.pyd`。
- `!` 取消忽略，因此秘密 `.env` 不提交，但无真实秘密的 `.env.example` 可以提交。
- `.gitignore` 只影响未跟踪文件；不能自动清除已经进入 Git 历史的文件或秘密。
- 虚拟环境由 `requirements.txt` 重建，不提交 `.venv` 本身。

查看规则：

```powershell
Get-Content -LiteralPath '.gitignore'
```

2026-09-07 验收通过：文件非空，所有计划规则存在，没有忽略 `assets/`、项目源码、配置、测试或 Markdown 文档。实际匹配行为将在 Git 初始化后复核。

## 6. 下一步

初始化 Git，验证忽略规则，并完成项目首次提交。
