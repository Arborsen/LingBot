# 贡献与维护

欢迎围绕任务映射、可恢复训练、时序增强、场景重放和结果校验提出改进。

- 说明具体触发条件、改动后的行为和验证方式；新增实验数字链接到结果摘要。
- 上游模型/仿真代码与本仓库新增工具分别保留归属和许可证。
- 通过参数或配置提供本机路径；不要提交权重、演示、录像、原始日志或凭据。
- 部分结果应保留为partial；不同任务数、场景、指令或执行策略的分数分别展示。
- 大型GPU实验需要显式运行，不放进提交钩子或默认CI。

轻量检查：

```bash
python -m unittest discover -s tools/tests -v
python -m unittest discover -s tools/eval_after_training -p 'test_gates.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_campaign_aggregation.py' -v
python -m unittest discover -s tools/eval_after_training -p 'test_native_scene_patch.py' -v
git diff --check
```

依赖PyTorch/图像库或SAPIEN的可选CPU测试见[复现指南](docs/reproduction.md)。详细目录约定见[维护指南](docs/workflow.md)。
