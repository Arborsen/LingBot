#!/usr/bin/env python3
"""Export small evaluation snapshots without changing any runtime artifacts."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


CONFIG_FIELDS = (
    "checkpoint", "normalization", "tasks", "episodes_per_task",
    "expected_episodes", "task_config", "instruction_type", "setting",
    "seed", "precision", "compile", "chunk_length", "video_fps",
    "gpus", "slots_per_gpu", "training_data", "official_final_protocol",
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


def rate_text(rate):
    return "—" if rate is None else "{:.2%}".format(rate)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def export(root):
    captured_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    entries = []
    # Read and validate every input before writing any archive.
    for run in sorted((root / "reproduce_artifacts").iterdir()):
        if not run.is_dir() or not (run / "run_config.json").exists():
            continue
        entry = collect_run(run, captured_at)
        if entry is not None:
            entries.append(entry)
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
    lines.extend([
        "", "## 归档说明", "",
        "每个目录保存 config.json、summary.json 和 task_results.csv。",
        "可自行增加 notes.md，记录命令、执行时 commit、环境版本和结论；汇总不会覆盖 notes.md。",
        "summary.json 中的 source_sha256 对应读取到的原始 JSON，setting_correction 保留设置纠正信息。",
        "未记录的历史执行版本不自动补填；official_final_protocol 未声明时不推断为官方评测。",
        "路径去掉个人机器的工作区前缀；归档配置用于追溯，不是可直接执行的启动配置。", "",
        "没有 run_config.json 的旧诊断目录不在自动索引中。已有归档在原始目录删除后仍保留，",
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
