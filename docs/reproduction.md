# 环境搭建与复现入口

模型训练与 RoboTwin 仿真使用不同环境。先分别验证，再连接推理服务和仿真客户端。下列入口沿用当前源码位置；具体参数以脚本和归档配置为准。

## 模型环境

[环境创建脚本](../lingbot-vla-v2/tools/create_train_env.sh)默认创建 Python 3.12 环境，安装模型、深度分支及固定版本依赖。从模型仓库运行：

```bash
cd lingbot-vla-v2
bash tools/create_train_env.sh --help
```

安装参数与 Flash Attention wheel 的用法见[模型 README](../lingbot-vla-v2/README.md)。当前个人改动涉及 LeRobot 安装来源、MoGe 的依赖安装方式以及 huggingface_hub 版本，见[贡献清单](../contributions/README.md)。

## 仿真与数据处理

RoboTwin 环境和资源准备见 [RoboTwin README](../RoboTwin/README.md)。数据转换入口见[模型侧 RoboTwin 指南](../lingbot-vla-v2/experiment/robotwin/README.md)。数据格式与归一化输入见[学习笔记](learning.md)。

## 训练与评测

- 训练配置说明：[Training Config](../lingbot-vla-v2/configs/vla/Training_Config.md)。
- 通用评测启动器：[start_robotwin_infer_and_eval.sh](../lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh)。
- 历史训练与评测脚本：`reproduce_artifacts/` 中已跟踪的源码。
- 评测配置和结果摘要：[实验索引](../experiments/README.md)。

历史脚本包含个人机器绝对路径，运行前要检查模型、数据、Conda 环境和输出路径。新的通用启动脚本应接受参数，不依赖某一台机器的目录。

## 复现时需要固定的信息

记录上游 commit、个人改动、运行命令、环境版本、数据选择、归一化文件及其 hash、模型检查点、任务集合、指令类型、seed 和推理精度。旧实验没有记录的执行时 commit 不用当前 HEAD 补填；当前工作区版本只代表整理时的源码状态。

实验汇总命令只读取已有结果，不会重新运行实验或验证视频。现有结果中声明的 `validated` 与 `complete` 会被用于归档，归档不是独立复测。
