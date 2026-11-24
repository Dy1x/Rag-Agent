"""Run the multi-agent investigation on the shared offline SQLite fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.investigation.run_agent import DEFAULT_QUESTION
from eval.investigation.run_baseline import create_fixture_db, load_case
from service.investigation_data_service import InvestigationScope
from service.investigation_service import multi_dependencies_for, run_investigation


def run_case(name: str, mode: str, question: str = DEFAULT_QUESTION) -> dict:
    devices, case = load_case(name)
    scope = InvestigationScope(**case["scope"])
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "investigation.sqlite3"
        create_fixture_db(db_path, devices, case)
        return run_investigation(
            question, scope, multi_dependencies_for(db_path, mode), mode=mode, simulated_data=True,
        )


def prepare_fixture(name: str, db_path: Path) -> dict:
    devices, case = load_case(name)
    scope = InvestigationScope(**case["scope"])
    try:
        db_path.touch(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("请为模拟数据指定尚不存在的 SQLite 路径") from error
    try:
        create_fixture_db(db_path, devices, case)
    except Exception:
        db_path.unlink()
        raise
    return {"db_path": str(db_path.resolve()), "simulated_data": True, "scope": scope.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a coordinated firmware and site investigation.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--planner", choices=("offline", "deepseek"), default="offline")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--prepare-db", type=Path, help="写入模拟 SQLite 数据后退出")
    args = parser.parse_args()
    try:
        result = (
            prepare_fixture(args.case, args.prepare_db)
            if args.prepare_db
            else run_case(args.case, args.planner, args.question)
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
