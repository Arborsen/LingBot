# 复现指南

公开仓库提供源码、核心算法模块、轻量测试和已校验结果摘要。完整训练/仿真还需要从官方渠道准备模型、演示与资产；本项目不随Git发布这些大型文件。

## 1. 不依赖GPU的检查

Python3.10+，从仓库根目录执行：

```bash
python -m unittest discover -s tools/tests -v
python -m unittest discover -s tools/eval_after_training -p 'test_gates.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_campaign_aggregation.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_native_scene_patch.py' -v
```

这些检查覆盖归档保留、完整任务统计、修复成绩替换、部分结果拒绝、配对前提和源码补丁保护，不会下载权重或启动GPU任务。

已安装PyTorch、NumPy、Pillow和OpenCV的环境还可执行：

```bash
python tools/continuation_15000/test_sampling.py
python tools/continuation_17000/test_augmentation.py
```

采样测试使用合成50任务映射，不需要官方演示。安装SAPIEN的仿真环境可运行`python tools/eval_after_training/test_scene_repair.py`检查CPU物理状态恢复。

## 2. 环境与外部资源

| 用途 | 实验验证过的环境 |
| --- | --- |
| 训练/模型服务 | Python3.12.15、PyTorch2.8.0+cu128、Transformers4.57.3、NumPy1.26.4 |
| RoboTwin仿真 | Python3.10.22、PyTorch2.4.1+cu121、SAPIEN3.0.0b1、NumPy1.26.4、cuRobo |
| 硬件 | NVIDIA A10080GB；4卡短测、后续2卡训练及评测 |

使用两套独立环境，安装步骤以[模型环境脚本](../lingbot-vla-v2/tools/create_train_env.sh)、[模型README](../lingbot-vla-v2/README.md)及[RoboTwin指南](../RoboTwin/README.md)为准。具体驱动、CUDA和Flash Attention wheel需要按机器匹配；上述是实验环境记录。

准备：官方LingBot-VLA2.0预训练权重、Qwen3-VL-4B-Instruct、模型要求的辅助视觉权重、RoboTwin仿真资产、50任务各50条官方clean演示。数据转换入口见[上游RoboTwin指南](../lingbot-vla-v2/experiment/robotwin/README.md)。本项目训练集合与上游clean+randomized联合训练示例不同，必须显式使用clean数据列表。

## 3. 数据映射与归一化

核对14维机器人状态/动作和三相机输入，使用[robotwin.yaml](../lingbot-vla-v2/configs/robot_configs/robotwin.yaml)的字段映射。该文件保留实验路径，使用前需要替换为自己的资源路径。

训练和推理使用同一clean归一化文件，保存并核验其SHA-256。[学习笔记](learning.md)记录已有归一化与格式转换入口。[task_mapping.py](../tools/continuation_15000/task_mapping.py)用于审核本次2500条episode的导出：它包含该导出的明确措辞例外，换数据版本时需重新审计。

## 4. 训练与恢复接入

基础启动方式与参数说明见[Training Config](../lingbot-vla-v2/configs/vla/Training_Config.md)。核心模块按以下位置接入：

| 模块 | 接入时机 |
| --- | --- |
| `TaskSampler` / `DrawDataset` | DataLoader构造时；设置`LINGBOT_TASK_MAPPING`到审核后的映射JSON |
| `training_hooks` | 完整checkpoint加载、调度器恢复、优化器更新前和阶段状态保存时 |
| `continuation_17000.augmentation.install()` | 图像变换初始化后、训练数据读取前；当前/未来图像传递相同增强参数 |

固定global batch160。2卡配置为micro batch20、累积4次；FP32、dist_muon、梯度检查点开启。LR、阶段切换、各rank恢复校验和保存频率见[技术设计](design.md)。完整checkpoint用于继续训练，导出的`hf_ckpt`用于推理，二者用途不同。

这些模块针对本次模型配置验证过。完整实验恢复还依赖本地checkpoint及对应运行布局；公开的结果JSON和源码快照不能直接替代可执行的训练配置。历史脚本里记录的本机路径需要重新配置。

## 5. 闭环评测

基础服务/仿真启动器为[start_robotwin_infer_and_eval.sh](../lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh)，运行其`--help`可查看模型、环境、工作目录、输出、并行数及`--no_video`参数。

正式比较需显式固定：

- `demo_clean + seen`与`demo_randomized + unseen`；不要仅改场景而遗漏指令类型。
- FP32、动作预测50步、固定执行50步、Flow Matching10次迭代。
- 同一归一化、任务集、每任务50回合、冻结有效环境seed/完整指令/推理seed。
- 独立模型服务和唯一输出目录；本次后续运行只使用物理GPU1/2，SAPIEN渲染通过PCI地址绑定。
- 重测两个偏移任务时使用已修复的杂物边界及`scene_repair.py`状态重放入口；仅相同seed不足以证明同一初始场景。

公开[结果导出器](../tools/export_public_results.py)只从已完成本地任务导出计数及哈希。持有原始本地运行目录时可运行`python tools/export_public_results.py`。普通克隆者可以复核公开计数与代码，但仍需另行准备checkpoint和资产才能独立重跑GPU评测。

## 6. RTC推理记录

RTC的预测/执行/迭代长度、旧动作时间对齐与Flow Matching引导见[RTC说明](rtc.md)。本次公开内容提供方案与接口映射，RTC专用运行代码与专项性能对照未包含在此快照中。固定50步结果不作为RTC性能证明。
