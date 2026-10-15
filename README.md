# LingBot-VLA 2.0 · RoboTwin 训练与评测

基于 LingBot-VLA 2.0，在 RoboTwin 的 50 个 Aloha-AgileX 任务上进行联合训练与闭环评测，记录从环境、数据归一化到续训和推理的完整实验过程。仓库包含上游源码快照、个人改动、实验配置、结果与学习笔记，按私有比赛工作仓库维护。

训练仅使用官方 **clean** 数据；**randomized 仅用于测试**。模型权重、数据和视频保存在本地，GitHub 保存复现说明、配置、结果摘要与源码快照。

## 最新评测结果

两种 setting 各评测 **50 个任务 × 10 回合**。step 5000 和 step 10000 的以下结果均已完成，所有任务验证通过。

| Setting | 场景 / 指令 | step 5000 | step 10000 | 提升 |
| --- | --- | ---: | ---: | ---: |
| clean2clean | `demo_clean` / **seen** | 66.0%（330/500） | **70.4%（352/500）** | **+4.4 个百分点** |
| clean2random | `demo_randomized` / **unseen** | 32.6%（163/500） | **41.2%（206/500）** | **+8.6 个百分点** |

[完整结果与 50 个任务对比](experiments/comparisons/step5000_vs_step10000/README.md) · [逐任务 CSV](experiments/comparisons/step5000_vs_step10000/per_task_comparison.csv) · [实验索引与配置](experiments/README.md)

推理使用 FP32、`compile=True`、action chunk 长度 50、入口 `seed=0`，4 张 GPU 每卡两路。step 10000 的两种 setting 共 1000 回合约耗时 **2 小时 11 分钟**，关闭视频录制。

这些是每任务 10 回合的本地评测结果，不等同于官方最终提交协议。每任务相差 10 个百分点对应 1 次成功；两轮录像设置不同，会影响部分渲染调用与随机光照轨迹，不能把全部成绩变化直接归因于权重变化。

## 训练进展

状态快照：**2026-10-15**。本页记录整理时的状态，实时进度保存在本地运行目录。

| 阶段 | 学习率 | 每卡 micro batch × 累积次数 | Global batch | 状态 |
| --- | --- | ---: | ---: | --- |
| 初始训练 → step 5000 | `1e-4 → 5e-5` | 8 × 5 | 160 | 已完成 |
| step 5000 → 10000 | `5e-5 → 2.5e-5` | 20 × 2 | 160 | 已完成 |
| step 10000 → 15000 | `2.5e-5 → 1.25e-5` | 20 × 2 | 160 | 已配置并排队，等待四卡资源 |

当前续训方案使用 **4 × A100 80GB、FP32、dist_muon、余弦学习率衰减、梯度检查点开启**，从完整 checkpoint 恢复优化器及已保存的训练状态；每 1000 steps 保存完整 checkpoint 和 HF 权重。数据、归一化与辅助损失设置沿用上一阶段。

micro batch 20、累积 2 时实测约 **11.4 秒/step**。关闭梯度检查点只提速约 **0.17%**，峰值显存却从约 **50.19 GiB** 增至 **79.19 GiB**，因此继续开启。短测方法与 loss 走势见[本轮报告](experiments/comparisons/step5000_vs_step10000/README.md)。

下一轮训练完成后，计划分别评测 **step 12000 和 step 15000**：每个 checkpoint 都跑 clean + seen、randomized + unseen，各 50×10 回合，关闭录像，输出使用独立目录。

[下一轮配置与预检查](experiments/records/resume_10000_to15000_micro20_acc2_20261015_132111/config.json) · [step 10000 训练记录](experiments/records/resume_5000_to10000_gc_compare_20261014_154104/summary.json)

## 复现与学习入口

| 内容 | 入口 |
| --- | --- |
| 环境搭建与运行入口 | [复现指南](docs/reproduction.md) |
| 本轮训练、速度与逐任务分析 | [step 5000 → 10000 对比报告](experiments/comparisons/step5000_vs_step10000/README.md) |
| 配置、源码快照与结果摘要 | [实验索引](experiments/README.md) |
| 数据管线与归一化学习记录 | [学习笔记](docs/learning.md) |
| 环境与评测排错 | [排错记录](docs/troubleshooting.md) |
| 上游版本与个人改动 | [贡献清单](contributions/README.md) |
| 目录与 VS Code 提交流程 | [仓库维护](docs/workflow.md) |
| 第三方来源与许可证 | [第三方说明](THIRD_PARTY_NOTICES.md) |

## 目录约定

```text
LingBot/
├── README.md
├── docs/                  # 学习、复现、排错与维护说明
├── tools/                 # 可复用的小工具，不依赖具体实验时间戳
├── experiments/           # 可提交的配置和小型结果摘要
│   ├── records/           # 每次实验的快照，汇总命令生成
│   └── comparisons/       # 跨 checkpoint 的逐任务对比与分析
├── contributions/         # 个人改动、上游版本及 PR 状态
├── RoboTwin/              # 仿真源码，保留原路径
├── lingbot-vla-v2/         # 模型源码，保留原路径
└── reproduce_artifacts/   # 本地运行产物；历史已跟踪文件继续保留
```

新实验的脚本实现放在 `tools/` 或所属源码仓库；运行时复制的脚本、日志、视频、权重留在 `reproduce_artifacts/`。数据、模型与原始下载内容分别留在被忽略的 `datasets/`、`models/`、`raw/`。

## 更新实验记录

从主仓库根目录运行，Python 3.8 及以上即可，无需 GPU 或训练环境：

```bash
python tools/archive_experiments.py
```

命令兼容旧版 `run_config.json` 和新版 `plan.json` 评测记录，按 setting 分别归档；同时保存续训配置、状态、loss 分段与选定源码的 SHA-256。未完成评测仅统计已验证任务，并标为阶段结果。原始数据、权重、视频、日志和缓存继续留在本地。

归档是读取时快照，不会启动训练、重新仿真或解码视频。更新结果后再提交与推送；详细流程见[仓库维护](docs/workflow.md)。

## 上游项目

- [LingBot VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)：模型、训练和推理代码。
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)：双臂机器人仿真平台。
- [个人 LingBot VLA fork](https://github.com/Arborsen/lingbot-vla-v2)：用于维护模型侧改动和准备上游 PR。

上游源码遵循各目录内的许可证，详见[第三方说明](THIRD_PARTY_NOTICES.md)。
