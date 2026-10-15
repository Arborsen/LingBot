# LingBot-VLA 2.0：global_step_5000 的 RoboTwin 评测结果

整理时间：2026-10-14 15:12:25（Asia/Shanghai）。

同一份 `global_step_5000` 权重在 **clean + seen** 下取得 **66.0%（330/500）**，在 **randomized + unseen** 下取得 **32.6%（163/500）**。两组均已完成，每组 50 个任务、每任务 10 回合；共 **1,000 个视频已保存，评测收尾的全量解码校验通过**。

## 1. 总体结果

| Setting | 场景配置 | 指令 | 完成任务 | 成功回合 / 总回合 | 最终成功率 | 视频及结果校验 |
|---|---|---|---:|---:|---:|---|
| clean2clean | `demo_clean` | seen | 50/50 | 330/500 | **66.0%** | 500 个视频，全部通过 |
| clean2random | `demo_randomized` | unseen | 50/50 | 163/500 | **32.6%** | 500 个视频，全部通过 |

两组成功率相差 **33.4 个百分点**。由于场景和指令类型同时变化，这个差值反映两种 setting 的整体差异，不能单独归因于场景随机化。

clean 有 **10 个任务达到 10/10**、**1 个任务为 0/10**；randomized 有 **1 个任务达到 10/10**、**10 个任务为 0/10**。

## 2. 评测设置

- 权重：[global_step_5000 / hf_ckpt](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/official_dist_muon_batch160_seed42_5000_20261011_163354/checkpoints/global_step_5000/hf_ckpt)；两组使用同一份权重。
- 归一化：[训练时归档的 clean norm_stats.json](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/official_dist_muon_batch160_seed42_5000_20261011_163354/inference_assets/norm_stats.json)；两组使用同一份归一化参数。
- 训练数据沿用 clean 数据；randomized 仅用于测试。
- 任务数：50；每任务有效回合数：10；每组总回合数：500。
- 评测入口 `seed=0`；有效环境种子由仿真专家检查后选取，具体种子记录在各任务结果中。
- 推理：FP32，`compile=True`，action chunk 长度 50。
- 并行：GPU 0–3，每张卡 2 路，共 8 路。
- 视频：10 fps，按回合保存；文件名标注 `success` 或 `failure`。
- 本文仅统计修正后的 clean + seen 和 randomized + unseen；此前 clean + unseen 的诊断结果未纳入。
- 本次为每任务 10 回合的本地评测。设置名称及指令类型按比赛要求对齐，官方最终提交的回合数、种子和执行协议仍以官方规定为准。

## 3. 50 个任务的结果对比

各任务均评测 10 回合。差值 = randomized 成功率 − clean 成功率，单位为百分点。任务顺序沿用评测任务清单。

| 任务 | clean + seen | randomized + unseen | 差值（百分点） |
|---|---:|---:|---:|
| `lift_pot` | 5/10（50%） | 0/10（0%） | -50 |
| `hanging_mug` | 4/10（40%） | 0/10（0%） | -40 |
| `stack_bowls_three` | 7/10（70%） | 3/10（30%） | -40 |
| `scan_object` | 7/10（70%） | 2/10（20%） | -50 |
| `handover_block` | 1/10（10%） | 0/10（0%） | -10 |
| `click_bell` | 0/10（0%） | 5/10（50%） | +50 |
| `put_object_cabinet` | 4/10（40%） | 4/10（40%） | +0 |
| `open_microwave` | 2/10（20%） | 1/10（10%） | -10 |
| `stack_blocks_three` | 5/10（50%） | 3/10（30%） | -20 |
| `place_shoe` | 9/10（90%） | 7/10（70%） | -20 |
| `adjust_bottle` | 10/10（100%） | 5/10（50%） | -50 |
| `beat_block_hammer` | 10/10（100%） | 4/10（40%） | -60 |
| `blocks_ranking_rgb` | 8/10（80%） | 7/10（70%） | -10 |
| `blocks_ranking_size` | 2/10（20%） | 0/10（0%） | -20 |
| `click_alarmclock` | 3/10（30%） | 3/10（30%） | +0 |
| `dump_bin_bigbin` | 9/10（90%） | 7/10（70%） | -20 |
| `grab_roller` | 10/10（100%） | 7/10（70%） | -30 |
| `handover_mic` | 8/10（80%） | 2/10（20%） | -60 |
| `move_can_pot` | 5/10（50%） | 0/10（0%） | -50 |
| `move_pillbottle_pad` | 9/10（90%） | 2/10（20%） | -70 |
| `move_playingcard_away` | 10/10（100%） | 4/10（40%） | -60 |
| `place_cans_plasticbox` | 7/10（70%） | 4/10（40%） | -30 |
| `place_container_plate` | 10/10（100%） | 5/10（50%） | -50 |
| `place_dual_shoes` | 5/10（50%） | 5/10（50%） | +0 |
| `place_empty_cup` | 9/10（90%） | 5/10（50%） | -40 |
| `place_fan` | 1/10（10%） | 0/10（0%） | -10 |
| `place_mouse_pad` | 6/10（60%） | 0/10（0%） | -60 |
| `place_object_basket` | 8/10（80%） | 3/10（30%） | -50 |
| `place_object_scale` | 9/10（90%） | 1/10（10%） | -80 |
| `place_object_stand` | 9/10（90%） | 7/10（70%） | -20 |
| `place_phone_stand` | 6/10（60%） | 1/10（10%） | -50 |
| `move_stapler_pad` | 1/10（10%） | 0/10（0%） | -10 |
| `open_laptop` | 8/10（80%） | 2/10（20%） | -60 |
| `pick_diverse_bottles` | 3/10（30%） | 3/10（30%） | +0 |
| `pick_dual_bottles` | 6/10（60%） | 2/10（20%） | -40 |
| `place_a2b_left` | 6/10（60%） | 1/10（10%） | -50 |
| `place_a2b_right` | 2/10（20%） | 3/10（30%） | +10 |
| `place_bread_basket` | 8/10（80%） | 6/10（60%） | -20 |
| `place_bread_skillet` | 9/10（90%） | 2/10（20%） | -70 |
| `place_burger_fries` | 9/10（90%） | 8/10（80%） | -10 |
| `place_can_basket` | 8/10（80%） | 0/10（0%） | -80 |
| `press_stapler` | 10/10（100%） | 7/10（70%） | -30 |
| `rotate_qrcode` | 7/10（70%） | 1/10（10%） | -60 |
| `shake_bottle_horizontally` | 10/10（100%） | 8/10（80%） | -20 |
| `shake_bottle` | 10/10（100%） | 10/10（100%） | +0 |
| `stack_blocks_two` | 10/10（100%） | 6/10（60%） | -40 |
| `stack_bowls_two` | 10/10（100%） | 5/10（50%） | -50 |
| `stamp_seal` | 7/10（70%） | 0/10（0%） | -70 |
| `turn_switch` | 5/10（50%） | 1/10（10%） | -40 |
| `put_bottles_dustbin` | 3/10（30%） | 1/10（10%） | -20 |
| **合计** | **330/500（66.0%）** | **163/500（32.6%）** | **-33.4** |

## 4. 视频和结果完整性

- 两组 `results.json` 均记录 `complete: true`、`error: null`，退出码均为 0。
- 每组 50 个任务均为 `validated: true`，每任务包含 10 个视频，回合编号完整覆盖 0–9。
- 评测器已校验视频帧数、时长、有效回合种子及成功率记录，并在收尾时对全部视频执行 FFmpeg 解码校验。
- 本次整理再次核对：两组各 500 个唯一视频文件均存在且非空，文件名的成功标记与各任务成绩一致。
- 两组均记录 `preserved_videos_unchanged: true`，既有视频保持不变。

## 5. 成绩与视频位置

| 文件 | clean + seen | randomized + unseen |
|---|---|---|
| 最终成绩 | [stats.txt](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/stats.txt) | [stats.txt](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/stats.txt) |
| 详细结果及有效种子 | [results.json](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/results.json) | [results.json](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/results.json) |
| 任务成绩表 | [task_results.csv](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/task_results.csv) | [task_results.csv](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/task_results.csv) |
| 实际评测配置 | [run_config.json](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/run_config.json) | [run_config.json](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/run_config.json) |
| 视频目录 | [eval_results](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/eval_results) | [eval_results](/data2/hs/User/Program/Compete/LingBot/reproduce_artifacts/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/eval_results) |

每个视频目录下按任务分子目录，视频命名为 `episode0_success.mp4`、`episode1_failure.mp4` 等。
