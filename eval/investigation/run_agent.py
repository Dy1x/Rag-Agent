"""Run the LangGraph investigation with fixture data and an injectable planner."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import tracing_context

from agent.investigation_graph import InvestigationDependencies, create_investigation_graph
from agent.investigation_planner import plan_offline, plan_with_deepseek
from eval.investigation.investigation_store import (
    DEFAULT_RUNS_DIR, StoredInvestigation, create_run, load_run, run_writer,
)
from eval.investigation.run_baseline import create_fixture_db, load_case
from service.investigation_data_service import InvestigationScope
from service.investigation_service import lookup_known_issue_offline, lookup_known_issue_with_rag


DEFAULT_QUESTION = "昨晚多个站点掉线，优先查哪里？"


def dependencies_for(db_path: Path, mode: str, *, planner=None) -> InvestigationDependencies:
    default_planner, lookup = (
        (plan_offline, lookup_known_issue_offline)
        if mode == "offline"
        else (plan_with_deepseek, lookup_known_issue_with_rag)
    )
    return InvestigationDependencies(db_path, planner or default_planner, lookup)


def run_case(name: str, mode: str, question: str = DEFAULT_QUESTION) -> dict:
    devices, case = load_case(name)
    scope = InvestigationScope(**case["scope"])
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "investigation.sqlite3"
        create_fixture_db(db_path, devices, case)
        graph = create_investigation_graph(dependencies_for(db_path, mode))
        with tracing_context(enabled=False) if mode == "offline" else nullcontext():
            return graph.invoke({"run_id": uuid4().hex, "question": question, "scope": scope.to_dict()})


def result_output(state: dict) -> dict:
    trace = state.get("execution_trace", [])
    return {
        "run_id": state["run_id"],
        "question": state["question"],
        "scope": state["scope"],
        "execution_path": [entry["action"] for entry in trace],
        "execution_trace": trace,
        "planning_rounds": state.get("planning_rounds"),
        "followups_remaining": state.get("followups_remaining"),
        "report": state.get("report"),
    }


def persisted_result(run: StoredInvestigation, *, execute: bool, pause_after: str | None = None) -> dict:
    lock = run_writer(run) if execute else nullcontext()
    tracing = tracing_context(enabled=False) if run.planner == "offline" else nullcontext()
    uri = run.db_path.as_uri() + "?mode=rw"
    with lock, tracing, closing(sqlite3.connect(uri, uri=True, check_same_thread=False)) as connection:
        graph = create_investigation_graph(
            dependencies_for(run.db_path, run.planner),
            checkpointer=SqliteSaver(connection),
            interrupt_after=[pause_after] if pause_after else None,
        )
        config = run.graph_config()
        snapshot = graph.get_state(config)
        if execute and (snapshot.next or not snapshot.values):
            graph.invoke(None if snapshot.values else run.graph_input(), config, durability="sync")
            snapshot = graph.get_state(config)

    values = {**run.graph_input(), **snapshot.values}
    completed = bool(values.get("report")) and not snapshot.next
    return {
        **result_output(values),
        "status": "completed" if completed else ("in_progress" if snapshot.values else "pending"),
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "next_nodes": list(snapshot.next),
        "firmware_comparison": values.get("firmware_comparison"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run, inspect, or resume an investigation.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run with a temporary database.")
    start = commands.add_parser("start", help="Create and run a persistent investigation.")
    for command in (run, start):
        command.add_argument("--case", required=True)
        command.add_argument("--planner", choices=("offline", "deepseek"), default="offline")
        command.add_argument("--question", default=DEFAULT_QUESTION)
    start.add_argument("--pause-after", choices=("compare_firmware_cohorts", "compare_site_devices", "lookup_known_issue"))
    start.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    for name, help_text in (("status", "Show saved progress."), ("resume", "Continue a saved investigation.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run-id", required=True)
        command.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith(("--case", "--question", "--planner")):
        argv.insert(0, "run")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = result_output(run_case(args.case, args.planner, args.question))
        elif args.command == "start":
            devices, case = load_case(args.case)
            stored = create_run(args.runs_dir, devices, case, args.planner, args.question)
            print(f"run_id={stored.run_id}", file=sys.stderr, flush=True)
            result = persisted_result(stored, execute=True, pause_after=args.pause_after)
        else:
            stored = load_run(args.runs_dir, args.run_id)
            result = persisted_result(stored, execute=args.command == "resume")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
