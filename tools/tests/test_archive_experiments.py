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


if __name__ == "__main__":
    unittest.main()
