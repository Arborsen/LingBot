# 上游版本与个人改动

此目录记录个人修改和上游贡献状态。下列清单依据 2026 年 10 月 14 日本地工作区整理，状态“本地改进”不代表已经提交或被上游接受。

## 源码关系

| 项目 | 来源 | 整理时的本地版本 |
| --- | --- | --- |
| LingBot VLA 2.0 | [上游](https://github.com/Robbyant/lingbot-vla-v2)／[个人 fork](https://github.com/Arborsen/lingbot-vla-v2) | 个人 main `2f77a62`；本地 upstream/main `be969b8` |
| RoboTwin | [上游](https://github.com/RoboTwin-Platform/RoboTwin) | 本地 HEAD `e8246d7`，另有未提交修改 |

这些版本是整理时状态，不是历史实验执行版本，也不保证对应上游最新分支。主仓库保存源码快照，内部仓库用于独立开发；提交方式见[维护指南](../docs/workflow.md)。

## 个人修改

| 改动 | 文件 | 当前状态与下一步 |
| --- | --- | --- |
| 环境安装依赖调整 | `lingbot-vla-v2/tools/create_train_env.sh` | 本地改进；独立确认依赖兼容和安装差异后准备小型 PR |
| clean 数据归一化与复现输入 | `lingbot-vla-v2/reproduce_configs/`、`lingbot-vla-v2/reproduce_artifacts/robotwin_clean_v30_norm.json` | 个人实验配置；归一化路径需要与通用默认配置区分 |
| 机器人配置指定个人归一化路径 | `lingbot-vla-v2/configs/robot_configs/robotwin.yaml` | 本地实验改动；准备上游 PR 时去除个人绝对路径 |
| 仿真期间写入视频帧 | `RoboTwin/envs/_base_task.py` | 内部 RoboTwin 仓库尚未提交的改动；验证后考虑独立 PR |
| 评测调度、服务与视频工具 | `reproduce_artifacts/` 的历史脚本 | 本地实验工具；保持运行版本，后续逐步提取通用参数和实现 |
| 目录规范与结果归档 | 主仓库 `docs/`、`tools/`、`experiments/` | 本仓库维护内容 |

## 提交上游贡献

每个分支只解决一个可说明的问题，记录触发条件、改动前后行为和验证方式。剔除个人路径、比赛专用配置和无关实验文件；在个人 fork 发起 PR。得到 PR URL 后在此增加链接、状态和合并版本。

已经上传至个人 GitHub 或在主仓库保存的改动，应写作个人改进；已提交 PR 和已合并贡献分别记录。
