# 任务采样与训练恢复模块

此目录公开实际实验中的算法与校验模块：

| 模块 | 作用 |
| --- | --- |
| `stage_data.py` | 任务加权采样、rank分片、draw级RNG、学习率；另保留15k早期30/40/30增强和动态执行实验函数 |
| `task_mapping.py` | 针对2500条官方clean演示导出的任务映射校验 |
| `training_hooks.py` | 新采样阶段切换、完整恢复核验、实际LR检查和非有限数值拦截 |
| `paired_eval.py` | 环境seed、seen/unseen指令和初始场景冻结 |
| `validate_replay.py` | 在本地运行目录中重建场景，检查任务特有的成功判定状态 |
| `test_sampling.py` | 使用合成任务映射的CPU测试，不需要演示数据或权重 |

17k以后的10/90增强由[continuation_17000](../continuation_17000/)提供，不能把本文件夹保留的早期增强默认值误当成最终训练配方。动态执行长度是独立实验函数，公开固定50步成绩未启用它。

安装PyTorch和NumPy后可运行：

```bash
python tools/continuation_15000/test_sampling.py
```

恢复钩子与任务映射包含本次模型、数据导出的约束，例如优化器状态计数和明确复核过的指令。它们应由训练入口接入，而不是通用于任意模型或任意LeRobot导出的插件。具体接入见[复现指南](../../docs/reproduction.md)。本机准备、资源轮询和一次性迁移脚本未列为通用复现入口。
