# 第三方来源与许可证

主仓库包含第三方源码快照。各项目自己的版权与许可证文件继续适用于对应源码，不能用主仓库的一个标签概括全部内容。

| 目录 | 来源 | 许可证入口 |
| --- | --- | --- |
| `RoboTwin/` | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | [MIT](RoboTwin/LICENSE) |
| `lingbot-vla-v2/` | [LingBot VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) | [Apache 2.0](lingbot-vla-v2/LICENSE) |
| `reproduce_artifacts/robotwin_eval10/python_overlay/` | msgpack、websockets 的本地依赖副本 | [msgpack](reproduce_artifacts/robotwin_eval10/python_overlay/msgpack/LICENSE)、[websockets](reproduce_artifacts/robotwin_eval10/python_overlay/websockets/LICENSE) |

源码树包含其他嵌套项目，例如 RoboTwin 的 policy 实现以及模型视觉模块；它们各自目录内的许可证和说明也应保留。模型权重与数据来源、条款和下载位置单独记录。

本仓库个人新增的工具和文档采用根目录的[MIT许可证](LICENSE)。该许可证不替代第三方文件已有的版权或许可证：`lingbot-vla-v2/`、`RoboTwin/`、视觉模块、policy子项目及依赖副本继续保留其原声明。

RTC方法引用Kevin Black、Manuel Y. Galliker、Sergey Levine的[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)（2025）。本仓库的RTC说明为项目应用笔记，不将该方法归为原创；未在本次公开快照中复制RTC上游实现或发布其模型权重。

模型权重、官方演示数据及仿真资产应从对应上游渠道另行获取，并遵守各自条款。根目录MIT许可证覆盖的新增内容不改变这些资源的许可。
