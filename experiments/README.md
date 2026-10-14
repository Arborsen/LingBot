# 实验索引

使用 `python tools/archive_experiments.py` 从本地运行目录更新。
配置与结果是读取时快照，视频、日志、进程号和服务器端口不归档。

快照时间：2026-10-14T10:29:48+08:00（Asia/Shanghai）。

成功率按已验证任务的成功数 / episode 数计算。只有最终报告声明完成、
全部目标任务验证通过且汇总一致时，状态才标记为 complete；其他均为阶段或诊断记录。
不同数据、指令、任务集和精度的实验分别比较，不能跨行加总；
历史调度扩容实验可能复用同一批结果。校验状态来自原始评测记录，归档工具不会复测视频。

| 实验 | 仿真 / 指令 | 状态 | 已验证任务 | 已验证 episodes / 目标 | 成功数 | 成功率 |
| --- | --- | --- | --- | --- | --- | --- |
| [robotwin_eval_5000_50x10_video_2pergpu_20261013_100920](records/robotwin_eval_5000_50x10_video_2pergpu_20261013_100920/summary.json) | demo_clean / unseen | interrupted | 24/50 | 240/500 | 155 | 64.58%（阶段） |
| [robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148](records/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/summary.json) | demo_clean / seen | complete | 50/50 | 500/500 | 330 | 66.00% |
| [robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148](records/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/summary.json) | demo_randomized / unseen | partial | 32/50 | 320/500 | 95 | 29.69%（阶段） |

## 归档说明

每个目录保存 config.json、summary.json 和 task_results.csv。
可自行增加 notes.md，记录命令、执行时 commit、环境版本和结论；汇总不会覆盖 notes.md。
summary.json 中的 source_sha256 对应读取到的原始 JSON，setting_correction 保留设置纠正信息。
未记录的历史执行版本不自动补填；official_final_protocol 未声明时不推断为官方评测。
路径去掉个人机器的工作区前缀；归档配置用于追溯，不是可直接执行的启动配置。

没有 run_config.json 的旧诊断目录不在自动索引中。已有归档在原始目录删除后仍保留，
但不列入本次生成的表格。更新方法见[仓库维护](../docs/workflow.md)。
