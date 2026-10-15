#!/usr/bin/env python3
"""Export small evaluation snapshots without changing any runtime artifacts."""

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


CONFIG_FIELDS = (
    "checkpoint", "normalization", "tasks", "episodes_per_task",
    "expected_episodes", "task_config", "instruction_type", "setting",
    "seed", "precision", "compile", "chunk_length", "video_fps",
    "gpus", "slots_per_gpu", "training_data", "official_final_protocol", "eval_video_log",
)
CORRECTION_FIELDS = (
    "status", "original_setting", "required_setting", "replacement",
    "randomized_replacement", "note",
)


def load_json(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return data, hashlib.sha256(raw).hexdigest()


def portable_path(value):
    """Retain workspace-relative asset references, omitting machine prefixes."""
    if not isinstance(value, str):
        return value
    marker = "/LingBot/"
    if marker in value:
        return value.split(marker, 1)[1]
    if value.startswith("/"):
        return Path(value).name
    return value


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def summarize(config, progress=None, report=None):
    """Only validated task results enter rates; final status needs a full report."""
    progress = progress or {}
    report = report or {}
    final_claim = report.get("complete") is True
    source = report if final_claim else (progress or report)
    results = source.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("results must be an object indexed by task name")
    tasks = config.get("tasks", [])
    if not isinstance(tasks, list) or any(not isinstance(t, str) for t in tasks):
        raise ValueError("config.tasks must be a list of task names")
    if len(tasks) != len(set(tasks)):
        raise ValueError("config.tasks contains duplicate task names")
    per_task = config.get("episodes_per_task")
    if not integer(per_task) or per_task == 0:
        raise ValueError("episodes_per_task must be a positive integer")
    expected = len(tasks) * per_task
    if "expected_episodes" in config and config["expected_episodes"] != expected:
        raise ValueError("expected_episodes disagrees with the task list")
    rows = []
    for task in tasks:
        result = results.get(task, {})
        if not isinstance(result, dict):
            raise ValueError("task result must be an object: {}".format(task))
        successes, episodes = result.get("successes"), result.get("episodes")
        numeric = (integer(successes) and integer(episodes)
                   and episodes > 0 and successes <= episodes)
        valid = (numeric and episodes == per_task
                 and result.get("validated") is True
                 and result.get("exit_code", 0) == 0
                 and not result.get("errors")
                 and result.get("task", task) == task)
        rows.append({
            "task": task, "successes": successes, "episodes": episodes,
            "validated": bool(valid),
            "success_rate": successes / episodes if valid else None,
            "source": portable_path(result.get("source")),
        })
    valid_rows = [row for row in rows if row["validated"]]
    successes = sum(row["successes"] for row in valid_rows)
    episodes = sum(row["episodes"] for row in valid_rows)
    complete = (final_claim and bool(tasks) and len(valid_rows) == len(tasks)
                and not report.get("error")
                and report.get("preserved_videos_unchanged", True) is True
                and not progress.get("duplicate_episode_ids")
                and report.get("successes", successes) == successes
                and report.get("expected_episodes", expected) == expected)
    status = "complete" if complete else "partial"
    if not source:
        status = "no_results"
    elif final_claim and not complete:
        status = "needs_review"
    elif report.get("error"):
        status = "interrupted"
    return {
        "status": status,
        "task_count": len(tasks), "validated_tasks": len(valid_rows),
        "expected_episodes": expected, "validated_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "rate_scope": "full_run" if complete else "validated_tasks_only",
        "reported_completed_episodes": progress.get("completed_episodes"),
        "reported_phase": progress.get("phase"),
        "duplicate_episode_ids": progress.get("duplicate_episode_ids", []),
        "official_final_protocol": config.get("official_final_protocol"),
    }, rows


def collect_run(run, captured_at):
    loaded = {name: load_json(run / name) for name in (
        "run_config.json", "progress.json", "results.json", "setting_correction.json"
    )}
    if loaded["run_config.json"] is None:
        return None
    config = loaded["run_config.json"][0]
    progress = loaded["progress.json"][0] if loaded["progress.json"] else None
    report = loaded["results.json"][0] if loaded["results.json"] else None
    summary, rows = summarize(config, progress, report)
    slim_config = {key: config[key] for key in CONFIG_FIELDS if key in config}
    for key in ("checkpoint", "normalization"):
        if key in slim_config:
            slim_config[key] = portable_path(slim_config[key])
    correction = loaded["setting_correction.json"]
    slim_correction = {}
    if correction:
        slim_correction = {k: portable_path(v) for k, v in correction[0].items()
                           if k in CORRECTION_FIELDS}
    summary.update({
        "run": run.name, "captured_at": captured_at,
        "source_directory": "reproduce_artifacts/" + run.name,
        "source_sha256": {k: v[1] for k, v in loaded.items() if v},
        "setting_correction": slim_correction,
    })
    return slim_config, summary, rows


def indexed_results(rows):
    """Reject duplicates before converting the newer list format to a task index."""
    if not isinstance(rows, list):
        raise ValueError("planned evaluation tasks/results must be a list")
    results = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task"), str):
            raise ValueError("planned evaluation result needs a task name")
        if row["task"] in results:
            raise ValueError("duplicate planned evaluation task: {}".format(row["task"]))
        results[row["task"]] = row
    return results


def collect_planned_run(run, artifacts, captured_at):
    """Export each setting independently from the no-video runner's plan/status format."""
    loaded_plan = load_json(run / "plan.json")
    if not loaded_plan or not isinstance(loaded_plan[0].get("settings"), list):
        return []
    plan = loaded_plan[0]
    loaded_status = load_json(run / "status.json")
    status = loaded_status[0] if loaded_status else {}
    entries = []
    seen_settings = set()
    for setting in plan["settings"]:
        key = setting["setting"]
        if key in seen_settings or key not in ("clean2clean", "clean2random"):
            raise ValueError("duplicate or unknown evaluation setting: {}".format(key))
        seen_settings.add(key)
        expected_pair = {"clean2clean": ("demo_clean", "seen"),
                         "clean2random": ("demo_randomized", "unseen")}[key]
        if (setting["task_config"], setting["instruction_type"]) != expected_pair:
            raise ValueError("scene/instruction setting mismatch: {}".format(key))
        config = {k: plan[k] for k in CONFIG_FIELDS if k in plan}
        config.update(setting)
        config["expected_episodes"] = len(plan["tasks"]) * plan["episodes_per_task"]
        config["normalization"] = str(Path(plan["train"]) / "inference_assets/norm_stats.json")
        slots = plan.get("slots", [])
        if slots:
            config["gpus"] = sorted({slot["gpu"] for slot in slots})
            counts = [sum(s["gpu"] == gpu for s in slots) for gpu in config["gpus"]]
            if len(set(counts)) == 1:
                config["slots_per_gpu"] = counts[0]
        progress = dict(status.get("settings", {}).get(key, {}))
        progress["results"] = indexed_results(progress.get("results", []))
        progress["phase"] = status.get("phase")
        loaded_report = load_json(run / key / "summary.json")
        report = None
        if loaded_report:
            raw = loaded_report[0]
            report = {"complete": raw.get("validated") is True,
                      "results": indexed_results(raw.get("tasks", [])),
                      "successes": raw.get("successes"),
                      "expected_episodes": raw.get("episodes")}
            for task, result in report["results"].items():
                if any(result.get(field) != setting[field] for field in
                       ("setting", "task_config", "instruction_type")):
                    raise ValueError("task result setting mismatch: {}".format(task))
            if raw.get("setting") != key or raw.get("instruction_type") != setting["instruction_type"]:
                raise ValueError("final report setting mismatch: {}".format(key))
            if (not integer(raw.get("successes")) or not integer(raw.get("episodes"))
                    or raw["episodes"] == 0
                    or raw.get("success_rate") != raw["successes"] / raw["episodes"]):
                report["error"] = "Reported aggregate rate is inconsistent"
        summary, rows = summarize(config, progress, report)
        for field in ("checkpoint", "normalization"):
            config[field] = portable_path(config[field])
        relative = run.relative_to(artifacts)
        hashes = {"plan.json": loaded_plan[1]}
        if loaded_status:
            hashes["status.json"] = loaded_status[1]
        if loaded_report:
            hashes[key + "/summary.json"] = loaded_report[1]
        summary.update(run="__".join(relative.parts) + "__" + key,
                       captured_at=captured_at,
                       source_directory="reproduce_artifacts/" + relative.as_posix(),
                       source_sha256=hashes, setting_correction={},
                       video_recording=plan.get("eval_video_log"))
        entries.append((config, summary, rows))
    return entries


def portable_tree(value):
    if isinstance(value, dict):
        return {key: portable_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_tree(item) for item in value]
    return portable_path(value)


def source_snapshot(run):
    """Only explicitly selected implementation files; never scan runtime/cache trees."""
    files = {}
    names = ("run_pipeline.py", "preflight.py", "training/config.yaml", "source/train_continuation.py",
             "source/resume_support.py", "source/strict_checkpointer.py",
             "run_scheduled.py", "policy_server.py", "eval_no_video.py", "client_instrumented.py",
             "verify_checkpoint.py", "check_scheduled.py")
    for name in names:
        path = run / name
        if path.is_file():
            data = path.read_bytes()
            if len(data) > 512 * 1024:
                raise ValueError("Source snapshot exceeds size limit: {}".format(path))
            data.decode("utf-8")
            files[name] = data
    return files


def collect_training_run(run, captured_at):
    loaded = load_json(run / "plan.json")
    if not loaded or not all(key in loaded[0] for key in
                             ("formal_restart_step", "formal_stop_step", "full_checkpoint")):
        return None
    plan = loaded[0]
    loaded_state = load_json(run / "training_status.json") or load_json(run / "status.json")
    state = loaded_state[0] if loaded_state else {}
    decision = load_json(run / "decision.json")
    complete = (state.get("phase") == "complete" and state.get("final_step") == plan["formal_stop_step"]
                and (run / "training/exit_code.txt").exists()
                and (run / "training/exit_code.txt").read_text().strip() == "0")
    fields = ("original_run", "full_checkpoint", "gpus", "micro_batch_size", "gradient_accumulation_steps",
              "global_batch_size", "precision", "gradient_checkpointing", "formal_restart_step",
              "formal_stop_step", "additional_optimizer_updates", "learning_rate", "save_steps",
              "training_data", "evaluate_steps", "evaluation_settings", "episodes_per_task",
              "evaluation_tasks", "evaluation_video", "queue_policy")
    config = {key: portable_tree(plan[key]) for key in fields if key in plan}
    if decision:
        config["gradient_checkpointing"] = decision[0]["gradient_checkpointing"]
    summary = dict(run=run.name, captured_at=captured_at,
        status="complete" if complete else state.get("phase", "unknown"),
        start_step=plan["formal_restart_step"], target_step=plan["formal_stop_step"],
        last_step=state.get("last_step", state.get("final_step", plan["formal_restart_step"])),
        final_lr=state.get("final_lr") if complete else None, full_restore_verified=state.get("full_restore_verified"),
        source_directory="reproduce_artifacts/" + run.name, source_sha256={"plan.json": loaded[1]})
    if loaded_state:
        summary["source_sha256"]["training_status.json" if (run / "training_status.json").exists() else "status.json"] = loaded_state[1]
    additional = {}
    if decision:
        additional["gradient_checkpointing_decision.json"] = portable_tree(decision[0])
    for name in ("preflight_report.json", "restore_verified.json"):
        value = load_json(run / name)
        if value:
            additional[name] = portable_tree(value[0])
    for rank in range(4):
        name = "resume_audit_rank{}.json".format(rank)
        value = load_json(run / "training" / name)
        if value:
            additional[name] = portable_tree(value[0])
    metrics = run / "training/metrics/rank0.jsonl"
    if metrics.exists():
        data = [json.loads(line) for line in metrics.read_text().splitlines()]
        windows = []
        first = plan["formal_restart_step"] + 1
        for low in range(first, plan["formal_stop_step"] + 1, 500):
            rows = [r for r in data if low <= r["step"] < low + 500]
            if rows:
                windows.append(dict(first_step=low, last_step=rows[-1]["step"], updates=len(rows),
                    **{key: statistics.mean(r[key] for r in rows) for key in
                       ("loss", "vla_loss", "grad_norm", "seconds", "lr_after_step")}))
        additional["loss_windows.json"] = windows
    sources = source_snapshot(run)
    for plan_path in sorted((run / "evaluations").glob("step*/plan.json")):
        evaluation = load_json(plan_path)[0]
        fields = CONFIG_FIELDS + ("settings", "checkpoint_step", "training_final_step", "expected_checkpoint_lr")
        additional["evaluation_plan_{}.json".format(plan_path.parent.name)] = portable_tree(
            {key: evaluation[key] for key in fields if key in evaluation})
        for name, data in source_snapshot(plan_path.parent).items():
            sources["evaluations/{}/{}".format(plan_path.parent.name, name)] = data
    return config, summary, additional, sources


def write_sources(record, files):
    if not files:
        return
    manifest = {}
    for name, data in files.items():
        path = record / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (record / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def rate_text(rate):
    return "—" if rate is None else "{:.2%}".format(rate)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def export(root):
    captured_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    entries = []
    training_entries = []
    evaluation_sources = {}
    # Read and validate every input before writing any archive.
    for run in sorted((root / "reproduce_artifacts").iterdir()):
        if not run.is_dir():
            continue
        entry = collect_run(run, captured_at)
        if entry is not None:
            entries.append(entry)
        else:
            planned = collect_planned_run(run, root / "reproduce_artifacts", captured_at)
            entries.extend(planned)
            for _, summary, _ in planned:
                evaluation_sources[summary["run"]] = source_snapshot(run)
        training = collect_training_run(run, captured_at)
        if training:
            training_entries.append(training)
        for plan in sorted((run / "evaluations").glob("step*/plan.json")):
            if (plan.parent / "status.json").exists():
                entries.extend(collect_planned_run(plan.parent, root / "reproduce_artifacts", captured_at))
    destination = root / "experiments"
    destination.mkdir(exist_ok=True)
    lines = [
        "# 实验索引", "",
        "使用 `python tools/archive_experiments.py` 从本地运行目录更新。",
        "配置与结果是读取时快照，视频、日志、进程号和服务器端口不归档。", "",
        "快照时间：{}（Asia/Shanghai）。".format(captured_at), "",
        "成功率按已验证任务的成功数 / episode 数计算。只有最终报告声明完成、",
        "全部目标任务验证通过且汇总一致时，状态才标记为 complete；其他均为阶段或诊断记录。",
        "不同数据、指令、任务集和精度的实验分别比较，不能跨行加总；",
        "历史调度扩容实验可能复用同一批结果。校验状态来自原始评测记录，归档工具不会复测视频。", "",
        "| 实验 | 仿真 / 指令 | 状态 | 已验证任务 | 已验证 episodes / 目标 | 成功数 | 成功率 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config, summary, rows in entries:
        run_name = summary["run"]
        record = destination / "records" / run_name
        record.mkdir(parents=True, exist_ok=True)
        write_sources(record, evaluation_sources.get(run_name, {}))
        for name, data in (("config.json", config), ("summary.json", summary)):
            (record / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)
                                       + "\n", encoding="utf-8")
        with (record / "task_results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "task", "successes", "episodes", "validated", "success_rate", "source"
            ), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        rate = rate_text(summary["success_rate"])
        if summary["status"] != "complete" and summary["success_rate"] is not None:
            rate += "（阶段）"
        lines.append(
            "| [{}](records/{}/summary.json) | {} / {} | {} | {}/{} | {}/{} | {} | {} |".format(
                run_name, run_name, cell(config.get("task_config", "未知")),
                cell(config.get("instruction_type", "未知")), summary["status"],
                summary["validated_tasks"], summary["task_count"],
                summary["validated_episodes"], summary["expected_episodes"],
                summary["successes"], rate,
            )
        )
    if not entries:
        lines.extend(["", "未找到包含 run_config.json 的实验目录。"])
    if training_entries:
        lines.extend(["", "## 训练记录", "",
                      "| 实验 | 状态 | 起始 / 当前 / 目标 step | 学习率区间 |",
                      "| --- | --- | --- | --- |"])
        for config, summary, additional, sources in training_entries:
            name = summary["run"]
            record = destination / "records" / name
            record.mkdir(parents=True, exist_ok=True)
            for filename, value in [("config.json", config), ("summary.json", summary)] + list(additional.items()):
                (record / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_sources(record, sources)
            rate = config.get("learning_rate", {})
            lines.append("| [{}](records/{}/summary.json) | {} | {} / {} / {} | {} → {} |".format(
                name, name, summary["status"], summary["start_step"], summary["last_step"], summary["target_step"],
                rate.get("start_lr", "未知"), rate.get("end_lr", "未知")))
    lines.extend([
        "", "## 归档说明", "",
        "每个评测目录保存 config.json、summary.json 和 task_results.csv；训练目录另存配置、状态与恢复记录。",
        "可自行增加 notes.md，记录命令、执行时 commit、环境版本和结论；汇总不会覆盖 notes.md。",
        "summary.json 中的 source_sha256 对应读取到的原始 JSON，setting_correction 保留设置纠正信息。",
        "未记录的历史执行版本不自动补填；official_final_protocol 未声明时不推断为官方评测。",
        "路径去掉个人机器的工作区前缀；归档配置用于追溯，不是可直接执行的启动配置。", "",
        "部分记录附 source_snapshot/ 与 source_manifest.json，保存实际运行实现及 SHA-256，便于审查复现。",
        "这些是冻结的实验源码快照；仍需原始运行目录布局、环境和本地资源才能执行。", "",
        "兼容旧 run_config.json 格式和含 settings 的新 plan.json 格式；新格式按 setting 分别归档。",
        "不含上述配置的旧诊断目录不在自动索引中。已有归档在原始目录删除后仍保留，",
        "但不列入本次生成的表格。更新方法见[仓库维护](../docs/workflow.md)。", "",
    ])
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        count = export(args.root.resolve())
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, "Archive failed: {}\n".format(error))
    print("Archived {} evaluation snapshots to experiments/".format(count))


if __name__ == "__main__":
    main()
