# 修改与贡献约定

目录、实验记录和 VS Code 提交流程见[仓库维护](docs/workflow.md)。当前按私有学习与比赛仓库维护。

- 文档写清来源、问题、处理与验证；实验数字链接到归档，减少重复维护。
- 新工具放在 `tools/`，个人参数由配置、参数或环境变量提供。
- 新运行产物放在 `reproduce_artifacts/`；使用 `python tools/archive_experiments.py` 导出配置和结果摘要。
- 提交模型或仿真改动时确认所属仓库和 remote；贡献状态记录在 [contributions](contributions/README.md)。

修改汇总逻辑后，从根目录执行：

```bash
python -m unittest discover -s tools/tests -v
python -m py_compile tools/archive_experiments.py
git diff --check
```

轻量检查不依赖 GPU，也不替代模型训练、仿真验证或上游要求的检查。
