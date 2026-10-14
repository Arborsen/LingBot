# 数据管线与评测学习记录

## RoboTwin 数据到 LeRobot

当前数据准备路径包括原始 RoboTwin 轨迹、HDF5 转换及 LeRobot 格式整理。上游[转换指南](../lingbot-vla-v2/experiment/robotwin/README.md)以 v2.1 为例，当前工作区也有 v3.0 数据的复现配置。数据格式、相机字段和状态动作映射需要一起核对，不能只检查目录是否存在。

先用单样本验证数据加载、图像字段、状态动作形状和前向计算，再扩大训练。个人 fork 中的“单样本数据管线通过”提交是这段工作的记录入口，详见[贡献清单](../contributions/README.md)。

## 归一化与模型输入

[机器人配置](../lingbot-vla-v2/configs/robot_configs/robotwin.yaml)把原始观测映射到模型输入，并指定归一化文件。当前工作区有个人计算的 clean 数据归一化输入和对应复现配置：

- [归一化数据](../lingbot-vla-v2/reproduce_artifacts/robotwin_clean_v30_norm.json)。
- [归一化复现配置](../lingbot-vla-v2/reproduce_configs/robotwin_clean_norm.yaml)。

训练与推理应明确使用哪一份统计值。记录文件 hash 可用于检查归一化输入是否在实验间变化；个人数据统计和个人路径属于实验配置，不应直接替代上游通用默认值。

## 评测条件与统计

`demo_clean` 和 `demo_randomized` 是不同仿真条件，`seen` 和 `unseen` 是不同指令设置，分别记录和比较。clean 训练数据上的模型在不同评测条件下运行，不能据此称为使用 randomized 数据训练。

完整实验成功率使用完整目标任务集合的已验证 episode。阶段统计使用已完成并验证的任务，其结果可能受到任务完成顺序影响。视频文件名和进度计数有助于观察运行状态，最终汇总仍要检查每个任务的完成数量、校验状态和错误。

## 继续记录的方式

增加笔记时写清具体问题、原因、解决办法和验证证据。实验数字放在对应实验档案里，笔记链接过去，避免同一结果在多篇文档里更新不同步。
