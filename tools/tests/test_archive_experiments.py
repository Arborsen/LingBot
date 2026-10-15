import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "archive_experiments", Path(__file__).resolve().parents[1] / "archive_experiments.py"
)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.config = {"tasks": ["a", "b"], "episodes_per_task": 10}
        self.results = {
            "a": {"successes": 6, "episodes": 10, "validated": True},
            "b": {"successes": 2, "episodes": 10, "validated": True},
        }

    def test_final_rate_requires_complete_report(self):
        progress = {"results": self.results}
        summary, _ = archive.summarize(self.config, progress)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["rate_scope"], "validated_tasks_only")
        report = {"complete": True, "results": self.results, "successes": 8}
        summary, _ = archive.summarize(self.config, progress, report)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["success_rate"], 0.4)

    def test_partial_rate_excludes_unvalidated_and_running_episodes(self):
        self.results["b"]["validated"] = False
        summary, _ = archive.summarize(self.config, {
            "results": self.results, "completed_episodes": 19, "successes_so_far": 9
        })
        self.assertEqual(summary["validated_episodes"], 10)
        self.assertEqual(summary["successes"], 6)
        self.assertEqual(summary["success_rate"], 0.6)

    def test_incomplete_or_invalid_tasks_cannot_be_final(self):
        for key, value in (("episodes", 9), ("successes", 11),
                           ("exit_code", 1), ("errors", ["bad video"]),
                           ("task", "wrong-task")):
            with self.subTest(key=key):
                result = dict(self.results["b"], **{key: value})
                report = {"complete": True, "results": {"a": self.results["a"], "b": result}}
                summary, _ = archive.summarize(self.config, report=report)
                self.assertEqual(summary["status"], "needs_review")

    def test_duplicate_episodes_and_inconsistent_final_total_need_review(self):
        report = {"complete": True, "results": self.results, "successes": 8}
        summary, _ = archive.summarize(self.config, {"duplicate_episode_ids": ["a:0"]}, report)
        self.assertEqual(summary["status"], "needs_review")
        report["successes"] = 9
        summary, _ = archive.summarize(self.config, report=report)
        self.assertEqual(summary["status"], "needs_review")

    def test_duplicate_tasks_rejected(self):
        with self.assertRaises(ValueError):
            archive.summarize({"tasks": ["a", "a"], "episodes_per_task": 10})

    def test_final_report_takes_precedence_over_older_progress(self):
        progress = {"results": {"a": self.results["a"]}}
        report = {"complete": True, "results": self.results, "successes": 8}
        summary, _ = archive.summarize(self.config, progress, report)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["validated_tasks"], 2)

    def test_malformed_source_does_not_overwrite_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["a_valid", "b_incomplete_write"]:
                run = root / "reproduce_artifacts" / name
                run.mkdir(parents=True)
                (run / "run_config.json").write_text(json.dumps(self.config))
            (root / "reproduce_artifacts/b_incomplete_write/progress.json").write_text('{"results":')
            destination = root / "experiments"
            destination.mkdir()
            (destination / "README.md").write_text("Previous archive")
            with self.assertRaises(ValueError):
                archive.export(root)
            self.assertEqual((destination / "README.md").read_text(), "Previous archive")
            self.assertFalse((destination / "records").exists())

    def test_export_preserves_sources_and_notes_omits_process_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "reproduce_artifacts" / "test_run"
            run.mkdir(parents=True)
            config = dict(self.config, slots=[{"server_pid": 123}], api_token="private")
            (run / "run_config.json").write_text(json.dumps(config))
            (run / "progress.json").write_text(json.dumps({"results": self.results}))
            record = root / "experiments" / "records" / "test_run"
            record.mkdir(parents=True)
            (record / "notes.md").write_text("Keep this note")
            original = {p.name: p.read_bytes() for p in run.iterdir()}
            self.assertEqual(archive.export(root), 1)
            self.assertEqual(original, {p.name: p.read_bytes() for p in run.iterdir()})
            self.assertEqual((record / "notes.md").read_text(), "Keep this note")
            saved = json.loads((record / "config.json").read_text())
            self.assertNotIn("slots", saved)
            self.assertNotIn("api_token", saved)
            self.assertTrue((record / "task_results.csv").exists())

    def test_planned_run_settings_are_separate_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "reproduce_artifacts" / "planned"
            run.mkdir(parents=True)
            settings = [dict(setting="clean2clean", task_config="demo_clean", instruction_type="seen"),
                        dict(setting="clean2random", task_config="demo_randomized", instruction_type="unseen")]
            plan = dict(self.config, train="/machine/LingBot/reproduce_artifacts/train/training",
                        checkpoint="/machine/LingBot/checkpoints/global_step_10000/hf_ckpt",
                        settings=settings, eval_video_log=False,
                        slots=[dict(gpu=0, slot=0, port=123, server_pid=999)])
            (run / "plan.json").write_text(json.dumps(plan))
            for setting in settings:
                directory = run / setting["setting"]
                directory.mkdir()
                tasks = [dict(result, task=task, **setting) for task, result in self.results.items()]
                report = dict(**setting, validated=True, successes=8, episodes=20,
                              success_rate=0.4, tasks=tasks)
                (directory / "summary.json").write_text(json.dumps(report))
            self.assertEqual(archive.export(root), 2)
            saved = root / "experiments/records/planned__clean2random"
            summary = json.loads((saved / "summary.json").read_text())
            config = json.loads((saved / "config.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["success_rate"], 0.4)
            self.assertEqual(config["instruction_type"], "unseen")
            self.assertIs(config["eval_video_log"], False)
            self.assertNotIn("slots", config)
            self.assertEqual(config["slots_per_gpu"], 1)
            report["tasks"].append(report["tasks"][0])
            (run / "clean2random/summary.json").write_text(json.dumps(report))
            previous = (root / "experiments/README.md").read_text()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                archive.export(root)
            self.assertEqual((root / "experiments/README.md").read_text(), previous)

    def test_planned_run_rejects_wrong_instructions_and_inconsistent_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            run = artifacts / "planned"
            directory = run / "clean2clean"
            directory.mkdir(parents=True)
            setting = dict(setting="clean2clean", task_config="demo_clean", instruction_type="seen")
            plan = dict(self.config, train="train", checkpoint="checkpoint", settings=[setting])
            (run / "plan.json").write_text(json.dumps(plan))
            tasks = [dict(result, task=task, **setting) for task, result in self.results.items()]
            report = dict(**setting, validated=True, successes=9, episodes=20, success_rate=0.45, tasks=tasks)
            (directory / "summary.json").write_text(json.dumps(report))
            entries = archive.collect_planned_run(run, artifacts, "test")
            self.assertEqual(entries[0][1]["status"], "needs_review")
            plan["settings"][0]["instruction_type"] = "unseen"
            (run / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "setting mismatch"):
                archive.collect_planned_run(run, artifacts, "test")

    def test_training_snapshot_preserves_queue_status_and_excludes_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "training_run"
            (run / "training").mkdir(parents=True)
            plan = dict(formal_restart_step=10000, formal_stop_step=15000,
                        full_checkpoint="/machine/LingBot/checkpoints/global_step_10000",
                        gradient_checkpointing=True)
            (run / "plan.json").write_text(json.dumps(plan))
            (run / "training_status.json").write_text(json.dumps({"phase":"waiting_for_free_gpus","last_step":10000}))
            (run / "run_pipeline.py").write_text("print('example')\n")
            (run / "training/config.yaml").write_text("train:\n  max_steps: 15000\n")
            (run / "weights.pt").write_bytes(b"not a real checkpoint")
            (run / "server.log").write_text("runtime")
            config, summary, extra, sources = archive.collect_training_run(run, "test")
            self.assertEqual(summary["status"], "waiting_for_free_gpus")
            self.assertIsNone(summary["final_lr"])
            self.assertEqual(config["full_checkpoint"], "checkpoints/global_step_10000")
            self.assertEqual(set(sources), {"run_pipeline.py","training/config.yaml"})
            (run / "decision.json").write_text(json.dumps({"gradient_checkpointing":False}))
            config, _, _, _ = archive.collect_training_run(run, "test")
            self.assertIs(config["gradient_checkpointing"], False)


if __name__ == "__main__":
    unittest.main()
