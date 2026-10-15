# 实验索引

使用 `python tools/archive_experiments.py` 从本地运行目录更新。
配置与结果是读取时快照，视频、日志、进程号和服务器端口不归档。

快照时间：2026-10-15T13:37:41+08:00（Asia/Shanghai）。

成功率按已验证任务的成功数 / episode 数计算。只有最终报告声明完成、
全部目标任务验证通过且汇总一致时，状态才标记为 complete；其他均为阶段或诊断记录。
不同数据、指令、任务集和精度的实验分别比较，不能跨行加总；
历史调度扩容实验可能复用同一批结果。校验状态来自原始评测记录，归档工具不会复测视频。

| 实验 | 仿真 / 指令 | 状态 | 已验证任务 | 已验证 episodes / 目标 | 成功数 | 成功率 |
| --- | --- | --- | --- | --- | --- | --- |
| [robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2clean](records/robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2clean/summary.json) | demo_clean / seen | complete | 50/50 | 500/500 | 352 | 70.40% |
| [robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2random](records/robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2random/summary.json) | demo_randomized / unseen | complete | 50/50 | 500/500 | 206 | 41.20% |
| [robotwin_eval_5000_50x10_video_2pergpu_20261013_100920](records/robotwin_eval_5000_50x10_video_2pergpu_20261013_100920/summary.json) | demo_clean / unseen | interrupted | 24/50 | 240/500 | 155 | 64.58%（阶段） |
| [robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148](records/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/summary.json) | demo_clean / seen | complete | 50/50 | 500/500 | 330 | 66.00% |
| [robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148](records/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/summary.json) | demo_randomized / unseen | complete | 50/50 | 500/500 | 163 | 32.60% |

## 训练记录

| 实验 | 状态 | 起始 / 当前 / 目标 step | 学习率区间 |
| --- | --- | --- | --- |
| [resume_10000_to15000_micro20_acc2_20261015_132111](records/resume_10000_to15000_micro20_acc2_20261015_132111/summary.json) | waiting_for_free_gpus | 10000 / 10000 / 15000 | 2.5e-05 → 1.25e-05 |
| [resume_5000_to10000_gc_compare_20261014_154104](records/resume_5000_to10000_gc_compare_20261014_154104/summary.json) | complete | 5000 / 10000 / 10000 | 5e-05 → 2.5e-05 |
| [resume_5000_to10000_micro20_acc2_20261014_150224](records/resume_5000_to10000_micro20_acc2_20261014_150224/summary.json) | failed | 5000 / 5000 / 10000 | 5e-05 → 2.5e-05 |

## 归档说明

每个评测目录保存 config.json、summary.json 和 task_results.csv；训练目录另存配置、状态与恢复记录。
可自行增加 notes.md，记录命令、执行时 commit、环境版本和结论；汇总不会覆盖 notes.md。
summary.json 中的 source_sha256 对应读取到的原始 JSON，setting_correction 保留设置纠正信息。
未记录的历史执行版本不自动补填；official_final_protocol 未声明时不推断为官方评测。
路径去掉个人机器的工作区前缀；归档配置用于追溯，不是可直接执行的启动配置。

部分记录附 source_snapshot/ 与 source_manifest.json，保存实际运行实现及 SHA-256，便于审查复现。
这些是冻结的实验源码快照；仍需原始运行目录布局、环境和本地资源才能执行。

兼容旧 run_config.json 格式和含 settings 的新 plan.json 格式；新格式按 setting 分别归档。
不含上述配置的旧诊断目录不在自动索引中。已有归档在原始目录删除后仍保留，
但不列入本次生成的表格。更新方法见[仓库维护](../docs/workflow.md)。
