# CORe50 数据卡

本文件记录阶段 16 使用的目标域数据、筛选规则和数据划分。它只描述数据协议，不记录模型选择结果。

冻结日期：2026-09-28

状态：原始数据、固定 manifest、图片解码与 32×32 样例视觉验收均已通过

## 1. 数据集身份

| 字段 | 内容 |
|---|---|
| 名称 | CORe50（Continuous Object Recognition 50） |
| 官方主页 | https://vlomonaco.github.io/core50/ |
| 官方代码 | https://github.com/vlomonaco/core50 |
| 论文 | Lomonaco & Maltoni, *CORe50: a New Dataset and Benchmark for Continuous Object Recognition*, CoRL 2017 |
| 获取日期 | 2026-09-28 |
| 使用版本 | cropped 128×128 RGB images |
| 原始规模 | 164,866 张 RGB-D 图像，11 sessions，50 objects |
| 本项目任务 | 10 类 category-level RGB image classification |

CORe50 从操作者视角拍摄手持日常物品。不同 session 改变背景、光照、室内/室外环境、持物手和遮挡，适合研究从 CIFAR-10 到具身视角物体图像的表征迁移。

## 2. 许可与引用

官方项目页面声明数据及基准材料被免费发布；官方代码仓库的 `LICENSE` 声明该 work 使用 Creative Commons Attribution 4.0 International（CC BY 4.0）。本项目必须保留数据集名称、作者、论文和主页归属信息。

原始 CORe50 图片保存在 Git 忽略的 `data/` 中，不提交到本仓库，也不随模型代码再分发。若将来需要单独再分发原始图片，必须重新核对数据归档本身当时展示的许可条款，而不能只依赖本数据卡。

建议引用：

~~~bibtex
@inproceedings{lomonaco2017core50,
  title={CORe50: a New Dataset and Benchmark for Continuous Object Recognition},
  author={Lomonaco, Vincenzo and Maltoni, Davide},
  booktitle={Conference on Robot Learning},
  year={2017}
}
~~~

## 3. 类别定义

本项目不预测 50 个具体物体实例，而将每连续 5 个 object ID 合并成一个语义类别：

| Target label | Object IDs | 官方类别名 |
|---:|---|---|
| 0 | o1–o5 | plug adapters |
| 1 | o6–o10 | mobile phones |
| 2 | o11–o15 | scissors |
| 3 | o16–o20 | light bulbs |
| 4 | o21–o25 | cans |
| 5 | o26–o30 | glasses |
| 6 | o31–o35 | balls |
| 7 | o36–o40 | markers |
| 8 | o41–o45 | cups |
| 9 | o46–o50 | remote controls |

标签公式为：

~~~text
category_label = (object_id - 1) // 5
~~~

分类头输出维度固定为 10。CIFAR-10 原分类头不会复用。

## 4. 纳入与排除规则

纳入：

- cropped 128×128 color images；
- session s1–s11；
- object o1–o50；
- 能从路径或文件名解析 session、object 和 frame ID 的有效 RGB 图片。

排除：

- depth images；
- 350×350 full-size images；
- detection、segmentation 与额外 cluttered test 数据；
- 解码失败、路径格式非法或标签无法映射的文件。

任何排除数量都必须在准备脚本的清单中报告，不能静默跳过。

## 5. 固定抽帧规则

原始数据来自约 20 fps 的短视频，相邻帧高度相关。第一版固定保留：

~~~text
frame_id % 10 == 0
~~~

即每 10 帧取 1 帧。该规则对所有 session、object 和 split 完全相同，不由随机种子决定，也不能在看到验证结果后调整。

实际保留 16,425 张图像。抽帧减少冗余和训练成本，不代表从总体中独立同分布地随机抽样。

## 6. 固定数据划分

| Split | Sessions | 预计抽帧后规模 | 用途 |
|---|---|---:|---|
| train | s1, s2, s4, s5, s6, s9, s11 | 10,452 | 参数学习 |
| validation | s8 | 1,491 | 模型选择与早停判断 |
| test | s3, s7, s10 | 4,482 | 协议冻结后的最终评估 |

s3、s7、s10 是 CORe50 官方固定测试 sessions；s8 在查看任何迁移结果前被预注册为 validation。划分以完整 session 为单位，同一 session 的任何相邻帧都不能跨 split。

该划分测量的是同一批物体实例跨采集环境/session 的泛化，而不是对全新物体实例的泛化。后者若开展，必须建立独立协议，不能改写本实验的含义。

## 7. 输入处理边界

- 数据读取：RGB，shape 从 128×128 开始；
- 模型输入：统一缩放到 32×32，以直接兼容 CIFAR-10 MiniViT 的 patch 数与位置编码；
- 三种迁移设置必须共享同一 train/validation/test manifest 和输入处理；
- 第一版沿用 CIFAR-10 normalization，使预训练 encoder 接收与源任务一致的数值约定；
- 目标域 normalization 和位置编码插值只能作为明确标记的后续扩展，不能混入主对照。

缩放到 32×32 会损失小物体纹理和细节，结果只说明当前低分辨率协议下的迁移能力。

正式数据管线固定为：

~~~text
train:
  RGB → Resize(32×32, bilinear, antialias)
      → RandomCrop(32, padding=4)
      → RandomHorizontalFlip(p=0.5)
      → ToTensor
      → CIFAR-10 Normalize

validation/test:
  RGB → Resize(32×32, bilinear, antialias)
      → ToTensor
      → CIFAR-10 Normalize
~~~

训练增强只制造轻微位移和水平镜像；validation/test 不含随机变换。`train_seed` 控制训练 RandomSampler，并在多 worker 模式下参与 worker seed；`num_workers=0` 时，随机 transform 还依赖训练入口统一设置的进程级 PyTorch seed。split 成员完全由 manifest 决定，不存在 CORe50 `split_seed`。

## 8. 下载后必须补充的完整性信息

- [x] 官方下载入口及实际文件名
- [x] 下载日期
- [x] 压缩包字节数与 SHA-256
- [x] 解压后 RGB 文件总数
- [x] 非法或损坏文件数
- [x] 抽帧后 train/validation/test 精确数量
- [x] 每个 split 的逐类数量
- [x] manifest SHA-256
- [x] 确定性抽样图像的人工视觉验收

实际完整性记录：

~~~text
archive: core50_128x128.zip
bytes: 5,892,103,007
SHA-256: f3d65017490dfc4392bc5e41b66930600f1d00898cae47fc4bd1a01cf92e1c6a
ZIP entries: 165,428
CRC bad entry: none
RGB PNG files: 164,866
invalid/corrupt selected images: 0 / 16,425
manifest SHA-256: d12f46f688851a08287f0220c85f49c5d2ee05e0c084b81066d9b79d101633d0
~~~

完整统计保存在 `assets/advanced/stage16/dataset_summary.json`；固定 manifest 保存在 Git 忽略的 `data/core50/manifests/stage16_core50.csv`。视觉网格保存在 `assets/advanced/stage16/core50_samples.png`。

视觉样本只从 train split 中选择，每类取排序后中间记录，未查看 validation/test，也没有人工挑选“最好看”的图片。十类在 32×32 下仍保留可辨认的整体轮廓与主要颜色；插头和马克笔等类别损失较多细节，但主体仍可见。图片中手部和地砖背景占比较高，模型可能利用环境捷径，因此 session-level split 是必要约束。该观察不改变已经冻结的抽帧与划分协议。
