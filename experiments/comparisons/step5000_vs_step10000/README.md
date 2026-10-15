# step 5000 → step 10000：训练与逐任务评测对比

记录时间：2026-10-15（Asia/Shanghai）。

step 10000 在 clean + seen 达到 **70.4%（352/500）**，randomized + unseen 达到 **41.2%（206/500）**。
两种 setting 各 50 个任务、每任务 10 回合，全部任务结果验证通过，评测退出码为 0。

| Setting | step 5000 | step 10000 | 变化（百分点） | 提升 / 退步 / 持平任务数 |
| --- | ---: | ---: | ---: | ---: |
| clean + seen | 66.0%（330/500） | 70.4%（352/500） | +4.4 | 20 / 14 / 16 |
| randomized + unseen | 32.6%（163/500） | 41.2%（206/500） | +8.6 | 30 / 9 / 11 |

## 训练与推理设置

- 从 step 5000 完整恢复模型、优化器、数据游标及 checkpoint 中保存的随机状态；训练至 step 10000。
- 4 张 A100 80GB；每卡 micro batch 20，累积 2 次，global batch 160；FP32、dist_muon、梯度检查点开启。
- 学习率在 step 5000→10000 的区间内从 5e-5 余弦降至 2.5e-5，无 warmup；每 1000 steps 保存完整 checkpoint 和 HF 权重。
- 训练数据和归一化仍来自官方 clean；randomized 仅用于测试。
- 训练于 10 月 15 日 08:06 正常结束，step 10000 的四个 rank、最终学习率与 HF 分片均校验通过。
- 评测于 08:30 启动、10:41 结束，总耗时约 2 小时 11 分钟；FP32、compile=True、chunk=50、seed=0，GPU 0–3 每卡 2 路。
- clean2clean 使用 demo_clean + seen；clean2random 使用 demo_randomized + unseen。
- step 5000 保存视频；step 10000 关闭视频和逐动作录像取图补丁，仍保存逐回合结果与计时记录。
- 两轮录像设置不同；额外渲染调用可能改变 randomized 动态光照的随机数轨迹。因此本对比不构成只改变权重的严格消融，也不能据此计算纯录像开关的提速倍数。
- 每任务仅 10 回合，相差 10 个百分点对应 1 次成功；小幅差异需要更多回合确认。本地 10 回合评测不等同于官方最终提交协议。

## 收敛观察

以下数值为 rank 0 的分段平均值，不是验证集 loss。

| Step 区间 | 总 loss | VLA loss | 梯度范数 |
| --- | ---: | ---: | ---: |
| 5001–5500 | 0.02495 | 0.02085 | 0.4348 |
| 8501–9000 | 0.01927 | 0.01544 | 0.3387 |
| 9001–9500 | 0.01883 | 0.01503 | 0.3295 |
| 9501–10000 | 0.01840 | 0.01462 | 0.3219 |

最后 500 steps 的平均 VLA loss 比前 500 steps 约低 2.7%；未出现非有限 loss/梯度值。
训练仍在改善，但仅凭两个 checkpoint 和训练 loss，不能断言成功率已到平台。

## 显存与速度实测

同为 FP32、micro batch 20、累积 2、global batch 160，从同一完整 step 5000 checkpoint 恢复，短测 50 steps，排除前 10 步后比较：

| 梯度检查点 | 平均每步耗时 | nvidia-smi 峰值显存 |
| --- | ---: | ---: |
| 开启 | 11.421 秒 | 约 50.19 GiB |
| 关闭 | 11.402 秒 | 约 79.19 GiB |

关闭只提速约 0.17%，显存接近 80 GiB，正式续训保持开启。
此前一段训练曾因切换到关闭梯度检查点的短测而主动 SIGTERM；随后从原 step 5000 完整重启，最终训练和本次评测均正常完成。

## 各任务结果

箭头为 step 5000 → step 10000；Δ 单位为百分点。

| 任务 | clean + seen | Δ | randomized + unseen | Δ |
| --- | ---: | ---: | ---: | ---: |
| `lift_pot` | 50% → 60% | +10 | 0% → 0% | +0 |
| `hanging_mug` | 40% → 40% | +0 | 0% → 10% | +10 |
| `stack_bowls_three` | 70% → 40% | -30 | 30% → 40% | +10 |
| `scan_object` | 70% → 100% | +30 | 20% → 10% | -10 |
| `handover_block` | 10% → 90% | +80 | 0% → 10% | +10 |
| `click_bell` | 0% → 0% | +0 | 50% → 50% | +0 |
| `put_object_cabinet` | 40% → 30% | -10 | 40% → 0% | -40 |
| `open_microwave` | 20% → 70% | +50 | 10% → 30% | +20 |
| `stack_blocks_three` | 50% → 90% | +40 | 30% → 70% | +40 |
| `place_shoe` | 90% → 80% | -10 | 70% → 50% | -20 |
| `adjust_bottle` | 100% → 100% | +0 | 50% → 70% | +20 |
| `beat_block_hammer` | 100% → 100% | +0 | 40% → 40% | +0 |
| `blocks_ranking_rgb` | 80% → 80% | +0 | 70% → 100% | +30 |
| `blocks_ranking_size` | 20% → 0% | -20 | 0% → 20% | +20 |
| `click_alarmclock` | 30% → 10% | -20 | 30% → 40% | +10 |
| `dump_bin_bigbin` | 90% → 100% | +10 | 70% → 70% | +0 |
| `grab_roller` | 100% → 90% | -10 | 70% → 90% | +20 |
| `handover_mic` | 80% → 90% | +10 | 20% → 10% | -10 |
| `move_can_pot` | 50% → 70% | +20 | 0% → 0% | +0 |
| `move_pillbottle_pad` | 90% → 80% | -10 | 20% → 30% | +10 |
| `move_playingcard_away` | 100% → 90% | -10 | 40% → 30% | -10 |
| `place_cans_plasticbox` | 70% → 90% | +20 | 40% → 60% | +20 |
| `place_container_plate` | 100% → 100% | +0 | 50% → 70% | +20 |
| `place_dual_shoes` | 50% → 90% | +40 | 50% → 30% | -20 |
| `place_empty_cup` | 90% → 90% | +0 | 50% → 60% | +10 |
| `place_fan` | 10% → 40% | +30 | 0% → 20% | +20 |
| `place_mouse_pad` | 60% → 30% | -30 | 0% → 10% | +10 |
| `place_object_basket` | 80% → 90% | +10 | 30% → 50% | +20 |
| `place_object_scale` | 90% → 70% | -20 | 10% → 30% | +20 |
| `place_object_stand` | 90% → 100% | +10 | 70% → 80% | +10 |
| `place_phone_stand` | 60% → 80% | +20 | 10% → 30% | +20 |
| `move_stapler_pad` | 10% → 20% | +10 | 0% → 0% | +0 |
| `open_laptop` | 80% → 100% | +20 | 20% → 10% | -10 |
| `pick_diverse_bottles` | 30% → 40% | +10 | 30% → 20% | -10 |
| `pick_dual_bottles` | 60% → 60% | +0 | 20% → 60% | +40 |
| `place_a2b_left` | 60% → 70% | +10 | 10% → 30% | +20 |
| `place_a2b_right` | 20% → 20% | +0 | 30% → 30% | +0 |
| `place_bread_basket` | 80% → 80% | +0 | 60% → 60% | +0 |
| `place_bread_skillet` | 90% → 90% | +0 | 20% → 50% | +30 |
| `place_burger_fries` | 90% → 80% | -10 | 80% → 50% | -30 |
| `place_can_basket` | 80% → 80% | +0 | 0% → 40% | +40 |
| `press_stapler` | 100% → 100% | +0 | 70% → 70% | +0 |
| `rotate_qrcode` | 70% → 80% | +10 | 10% → 20% | +10 |
| `shake_bottle_horizontally` | 100% → 100% | +0 | 80% → 100% | +20 |
| `shake_bottle` | 100% → 100% | +0 | 100% → 100% | +0 |
| `stack_blocks_two` | 100% → 90% | -10 | 60% → 80% | +20 |
| `stack_bowls_two` | 100% → 90% | -10 | 50% → 60% | +10 |
| `stamp_seal` | 70% → 40% | -30 | 0% → 30% | +30 |
| `turn_switch` | 50% → 60% | +10 | 10% → 10% | +0 |
| `put_bottles_dustbin` | 30% → 30% | +0 | 10% → 30% | +20 |

clean 提升最大的是 handover_block（+80）和 open_microwave（+50）；randomized 提升最大的是 pick_dual_bottles、place_can_basket、stack_blocks_three（均 +40）。
两种 setting 都退步的任务为 put_object_cabinet、place_shoe、move_playingcard_away、place_burger_fries；优先复查 put_object_cabinet 的 randomized 40%→0%。

## 下一轮计划及状态

step 10000→15000：完整恢复；学习率 2.5e-5→1.25e-5 余弦衰减；其余训练设置保持不变。
每 1000 steps 保存；训练结束后分别评测 step 12000、15000，每个 checkpoint 都跑两种 setting，各 50×10 回合，关闭录像。
**归档时状态为 waiting_for_free_gpus，尚未开始实际训练**；后台控制器等待 0–3 号卡全部释放后自动启动。
训练本身预计约 16 小时，不含资源等待和评测；实际进度以本地运行状态为准，本 GitHub 记录为时间快照。

## 可追溯记录

- [逐任务 CSV](per_task_comparison.csv)、[对比汇总 JSON](summary.json)。
- [step 5000 clean](../../records/robotwin_eval_5000_clean2clean_seen_50x10_video_2pergpu_20261013_132148/summary.json)、[step 5000 randomized](../../records/robotwin_eval_5000_clean2random_unseen_50x10_video_2pergpu_20261013_132148/summary.json)。
- [step 10000 clean](../../records/robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2clean/summary.json)、[step 10000 randomized](../../records/robotwin_eval_10000_clean_random_novideo_scheduled_20261014_204708__clean2random/summary.json)。
- [5000→10000 训练配置](../../records/resume_5000_to10000_gc_compare_20261014_154104/config.json)、[loss 分段记录](../../records/resume_5000_to10000_gc_compare_20261014_154104/loss_windows.json)。
- [10000→15000 训练配置](../../records/resume_10000_to15000_micro20_acc2_20261015_132111/config.json)、[CPU 预检查](../../records/resume_10000_to15000_micro20_acc2_20261015_132111/preflight_report.json)。
- 各记录内的 source_snapshot/ 与 source_manifest.json 保存冻结实现及 SHA-256；未提交模型、视频、原始日志、缓存或进程信息。

历史运行没有完整记录执行时 Git commit，故不使用本次整理提交冒充执行版本；源码快照 hash 用于补充追溯。
