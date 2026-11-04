# 评测校验与场景重放

公开内容：

- `gates.py`：训练完成、完整checkpoint及冻结任务manifest的启动门槛。
- `scene_repair.py`：独立杂物边界、专家有效场景状态的捕获、恢复和严格校验。
- `render_device.py`：按PCI地址把SAPIEN Vulkan渲染绑定到指定GPU；当前实验允许物理GPU1/2。
- `aggregate_scene_campaign.py`：验证完整50回合任务、修正两项成绩、核对配对条件并生成对比。
- `run_scene_campaign.py`：以本地`campaign.json`为输入的实验队列示例，包含源码补丁保护和失败保留。

队列示例需要完整的本地运行配置、环境、权重和历史manifest，不能把公开结果摘要直接当成启动配置。不会自动下载数据或重跑GPU实验。

标准库测试：

```bash
python -m unittest discover -s tools/eval_after_training -p 'test_gates.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_campaign_aggregation.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_native_scene_patch.py' -v
```

安装SAPIEN、PyTorch和NumPy的仿真环境还可运行`test_scene_repair.py`，使用CPU物理场景检查位姿、速度与RNG恢复。

真实仿真重复恢复和GPU1→GPU2的跨进程校验已执行；它验证初始状态一致性，不承诺后续物理轨迹或GPU内核逐bit确定。已公开结果及未完成补测见[实验报告](../../experiments/benchmarks/step22000_step25000/README.md)。
