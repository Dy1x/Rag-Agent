import json
import os
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.investigation.investigation_store import create_run, load_run, run_writer
from eval.investigation.run_agent import DEFAULT_QUESTION, load_case


ROOT = Path(__file__).resolve().parents[1]
RESUME_WITH_REPLAY_CHECK = """
import sys
from dataclasses import replace
from unittest.mock import patch
from eval.investigation import run_agent

called = []
factory = run_agent.dependencies_for
def forbidden(name):
    def fail(*args):
        called.append(name)
        raise AssertionError("Unexpected replay of " + name)
    return fail

def dependencies(db_path, mode):
    overrides = {name: forbidden(name) for name in sys.argv[1].split(",") if name}
    return replace(factory(db_path, mode), **overrides)

with patch.object(run_agent, "dependencies_for", side_effect=dependencies), \
     patch.object(run_agent, "load_case", side_effect=forbidden("reload fixture")):
    run_agent.main(sys.argv[2:])
if called:
    raise SystemExit("Unexpected replay: " + ", ".join(called))
"""


def cohort_counts(comparison):
    return [
        (comparison[name]["offline_device_count"], comparison[name]["device_count"])
        for name in ("target_group", "control_group")
    ]


class InvestigationPersistenceTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runs_dir = Path(directory.name)

    def cli(self, *args, forbidden=None, succeeds=True):
        command = ["-m", "eval.investigation.run_agent"]
        if forbidden is not None:
            command = ["-c", RESUME_WITH_REPLAY_CHECK, ",".join(forbidden)]
        env = {
            key: value for key, value in os.environ.items()
            if key not in {"DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY"}
        }
        env.update(PYTHONUTF8="1", LANGSMITH_TRACING="false")
        result = subprocess.run(
            [sys.executable, *command, *args, "--runs-dir", str(self.runs_dir)],
            cwd=ROOT, env=env, capture_output=True, encoding="utf-8", timeout=30,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            if args[0] == "start":
                self.assertIn(f"run_id={output['run_id']}", result.stderr)
            return output
        self.assertEqual(result.returncode, 2, result.stderr)
        return result.stderr

    def test_resume_in_another_process_with_durable_cohorts_budget_and_run_isolation(self):
        paused = self.cli(
            "start", "--case", "firmware", "--planner", "offline",
            "--pause-after", "compare_firmware_cohorts",
        )
        run_id = paused["run_id"]
        self.assertEqual(paused["status"], "in_progress")
        self.assertEqual(paused["next_nodes"], ["plan"])
        self.assertEqual(paused["planning_rounds"], 1)
        self.assertEqual(paused["followups_remaining"], 1)
        self.assertEqual(cohort_counts(paused["firmware_comparison"]["data"]), [(6, 8), (1, 8)])

        # Both the saved investigation data and checkpoint tables live in this run's file.
        path = self.runs_dir / f"{run_id}.sqlite3"
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertTrue({"devices", "alarm_events", "firmware_rollouts", "checkpoints", "investigation_run"} <= tables)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM devices").fetchone()[0], 16)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM alarm_events").fetchone()[0], 8)
            self.assertEqual(db.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall(), [(run_id,)])

        other = self.cli("start", "--case", "site", "--planner", "offline")
        self.assertEqual(other["status"], "completed")
        self.assertNotEqual(other["run_id"], run_id)
        self.assertEqual(other["report"]["impact"]["offline_device_count"], 6)

        inspected = self.cli("status", "--run-id", run_id, forbidden=(
            "alarm_timeline", "firmware_rollouts", "firmware_compare", "site_compare", "planner", "lookup_known_issue",
        ))
        self.assertEqual(inspected, paused)

        completed = self.cli("resume", "--run-id", run_id, forbidden=(
            "alarm_timeline", "firmware_rollouts", "firmware_compare", "site_compare",
        ))
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["execution_trace"][:3], paused["execution_trace"])
        self.assertEqual(completed["execution_path"][3:], ["lookup_known_issue", "finish"])
        self.assertEqual(completed["followups_remaining"], 0)
        self.assertEqual(completed["planning_rounds"], 2)
        self.assertEqual(cohort_counts(completed["report"]["firmware_comparison"]), [(6, 8), (1, 8)])

        again = self.cli("resume", "--run-id", run_id, forbidden=(
            "alarm_timeline", "firmware_rollouts", "firmware_compare", "site_compare", "planner", "lookup_known_issue",
        ))
        self.assertEqual(again["checkpoint_id"], completed["checkpoint_id"])
        self.assertEqual(again["report"], completed["report"])
        self.assertEqual(self.cli("status", "--run-id", other["run_id"]), other)

    def test_resume_pending_run_without_reloading_fixture(self):
        devices, case = load_case("site")
        stored = create_run(self.runs_dir, devices, case, "offline", DEFAULT_QUESTION)
        pending = self.cli("status", "--run-id", stored.run_id, forbidden=())
        self.assertEqual(pending["status"], "pending")
        self.assertIsNone(pending["checkpoint_id"])
        completed = self.cli("resume", "--run-id", stored.run_id, forbidden=())
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["report"]["impact"]["offline_device_count"], 6)

    def test_resume_rejects_concurrent_writer_without_advancing_checkpoint(self):
        paused = self.cli("start", "--case", "firmware", "--pause-after", "compare_firmware_cohorts")
        run = load_run(self.runs_dir, paused["run_id"])
        with run_writer(run):
            error = self.cli("resume", "--run-id", run.run_id, succeeds=False)
            self.assertIn("正在另一个进程中执行", error)
            self.assertEqual(self.cli("status", "--run-id", run.run_id), paused)

    def test_status_rejects_unknown_id_without_creating_database(self):
        error = self.cli("status", "--run-id", "0" * 32, succeeds=False)
        self.assertIn("找不到调查", error)
        self.assertEqual(list(self.runs_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
