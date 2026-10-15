# LingBot 学习与实验记录

围绕 LingBot VLA 和 RoboTwin，记录环境搭建、数据处理、训练复现、仿真评测和可提交给上游的改进。仓库包含上游源码快照和个人实验记录，个人改动见[贡献清单](contributions/README.md)。

当前按私有比赛工作仓库维护，可以记录成功率和逐任务结果。实验摘要是本地评测记录，评测设置与完成状态见[实验索引](experiments/README.md)。

最新完成：step 10000 在 clean + seen 为 **70.4%（352/500）**，randomized + unseen 为 **41.2%（206/500）**。相较 step 5000，分别增加 4.4、8.6 个百分点。训练设置、50 个任务明细和比较限制见[本轮结果报告](experiments/comparisons/step5000_vs_step10000/README.md)。10000→15000 续训已配置，归档时仍在等待显卡资源。

## 阅读入口

| 内容 | 入口 |
| --- | --- |
| 后续目录与 VS Code 提交流程 | [仓库维护](docs/workflow.md) |
| 环境搭建与运行入口 | [复现指南](docs/reproduction.md) |
| 数据管线与归一化学习记录 | [学习笔记](docs/learning.md) |
| 环境与评测排错 | [排错记录](docs/troubleshooting.md) |
| 配置、逐任务结果与成功率 | [实验索引](experiments/README.md) |
| 上游版本与个人改动 | [贡献清单](contributions/README.md) |
| 第三方来源与许可证 | [第三方说明](THIRD_PARTY_NOTICES.md) |

## 目录约定

```text
LingBot/
├── README.md
├── docs/                  # 学习、复现、排错与维护说明
├── tools/                 # 可复用的小工具，不依赖具体实验时间戳
├── experiments/           # 可提交的配置和小型结果摘要
│   └── records/           # 每次实验的快照，汇总命令生成
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

命令读取各实验的配置与结果，更新 `experiments/records/` 和索引，不修改原始评测文件。未完成实验的成功率只使用已验证任务，单独标记为阶段统计；视频、日志和进程状态不归档。没有配置的旧诊断目录不自动汇总。

随后在 VS Code 中选择 **LingBot 主仓库**，查看并暂存希望提交的文件，再 commit/push。Git 不会自动执行汇总命令；每次需要更新结果时先运行一次即可。

## 上游项目

- [LingBot VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)：模型、训练和推理代码。
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)：双臂机器人仿真平台。
- [个人 LingBot VLA fork](https://github.com/Arborsen/lingbot-vla-v2)：用于维护模型侧改动和准备上游 PR。

上游源码遵循各目录内的许可证，详见[第三方说明](THIRD_PARTY_NOTICES.md)。
