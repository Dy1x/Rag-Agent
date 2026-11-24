"""Keep each investigation's input tables and checkpoints in one SQLite file."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from eval.investigation.run_baseline import create_fixture_db
from service.investigation_data_service import InvestigationScope


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "reports" / "investigation-runs"


@dataclass(frozen=True)
class StoredInvestigation:
    run_id: str
    db_path: Path
    planner: str
    question: str
    scope: dict

    def graph_input(self) -> dict:
        return {"run_id": self.run_id, "question": self.question, "scope": self.scope}

    def graph_config(self) -> dict:
        return {"configurable": {"thread_id": self.run_id}}


def create_run(
    runs_dir: Path, devices: list[dict], case: dict, planner: str, question: str,
) -> StoredInvestigation:
    question = question.strip()
    if not question:
        raise ValueError("调查问题不能为空。")
    scope = InvestigationScope(**case["scope"]).to_dict()
    run_id = uuid4().hex
    runs_dir = runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.sqlite3"

    # Publish the database only after both the fixture and the resume metadata are durable.
    with TemporaryDirectory(prefix=".prepare-", dir=runs_dir) as directory:
        staging = Path(directory) / path.name
        create_fixture_db(staging, devices, case)
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute(
                """CREATE TABLE investigation_run (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    planner TEXT NOT NULL CHECK (planner IN ('offline', 'deepseek')),
                    question TEXT NOT NULL,
                    scope_json TEXT NOT NULL
                )"""
            )
            db.execute(
                "INSERT INTO investigation_run VALUES (?, ?, ?, ?)",
                (run_id, planner, question, json.dumps(scope, ensure_ascii=False)),
            )
        staging.replace(path)
    return StoredInvestigation(run_id, path, planner, question, scope)


def load_run(runs_dir: Path, run_id: str) -> StoredInvestigation:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("run_id 必须是 32 位小写十六进制字符。")
    path = runs_dir.resolve() / f"{run_id}.sqlite3"
    if not path.is_file():
        raise ValueError(f"找不到调查 {run_id}。")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            "SELECT planner, question, scope_json FROM investigation_run WHERE run_id = ?", (run_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"调查数据库中没有与 {run_id} 匹配的记录。")
    planner, question, scope_json = row
    return StoredInvestigation(run_id, path, planner, question, json.loads(scope_json))


@contextmanager
def run_writer(run: StoredInvestigation):
    """Exclude simultaneous resumes. The OS releases the lock if the process exits."""
    with run.db_path.with_suffix(".lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"调查 {run.run_id} 正在另一个进程中执行。") from error
        yield
