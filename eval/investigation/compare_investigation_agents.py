"""Compare one planner and two specialists over the same simulated evidence."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from langsmith import tracing_context

from agent.investigation_graph import COMPARISON_FOLLOWUP_BUDGET, _failure_reason, create_investigation_graph
from agent.investigation_planner import plan_comparison_offline, plan_with_deepseek
from agent.investigation_specialists import INITIAL_TOOL_BUDGET, REVIEW_TOOL_BUDGET
from eval.investigation.run_agent import DEFAULT_QUESTION, dependencies_for
from eval.investigation.run_baseline import create_fixture_db, load_case
from eval.investigation.run_evaluation import METRICS, score_report
from service.investigation_data_service import InvestigationScope
from service.investigation_service import multi_dependencies_for, run_investigation


CASES = ("rollout_site_overlap", "firmware", "missing")
INITIAL_CALLS = {"get_alarm_timeline", "get_firmware_rollouts"}


def _single(db_path: Path, scope: InvestigationScope, mode: str) -> dict:
    planner = (
        plan_comparison_offline if mode == "offline"
        else partial(plan_with_deepseek, prompt_name="investigation_comparison_plan.txt")
    )
    graph = create_investigation_graph(dependencies_for(db_path, mode, planner=planner), extended_tools=True)
    state = graph.invoke({"run_id": uuid4().hex, "question": DEFAULT_QUESTION, "scope": scope.to_dict()})
    calls = [entry for entry in state["execution_trace"] if entry["action"] != "finish"]
    rechecks = [entry for entry in calls if entry["action"] == "compare_firmware_cohorts_by_site"]
    return {
        "run_id": state["run_id"], "report": state["report"], "tool_calls": calls,
        "planning_decisions": state["planning_rounds"],
        "review": {"request": None, "tool_calls": rechecks},
    }


def _multi(db_path: Path, scope: InvestigationScope, mode: str) -> dict:
    outcome = run_investigation(
        DEFAULT_QUESTION, scope, multi_dependencies_for(db_path, mode), mode=mode, simulated_data=True,
    )
    calls = [
        {"action": action, **outcome["shared_queries"][field]}
        for field, action in (
            ("alarm_timeline", "get_alarm_timeline"), ("firmware_rollouts", "get_firmware_rollouts"),
        )
    ]
    review = outcome["review"] or {}
    specialists = [*outcome["initial"].values(), review]
    return {
        "run_id": outcome["run_id"], "report": outcome["report"],
        "tool_calls": [*calls, *(call for role in specialists for call in role.get("tool_calls", []))],
        "planning_decisions": sum(len(role.get("decisions", [])) for role in specialists),
        "review": {
            "request": outcome["review_request"], "tool_calls": review.get("tool_calls", []),
            "judgment": review.get("finding", {}).get("judgment"),
        },
    }


def _measure(run, db_path, scope, mode, expected, device_ids) -> dict:
    start = perf_counter()
    try:
        outcome = run(db_path, scope, mode)
    except Exception as error:
        reason = _failure_reason(error)
        return {
            "elapsed_seconds": round(perf_counter() - start, 4),
            "actual": None, "passed": dict.fromkeys(METRICS, False),
            "failures": {metric: [reason] for metric in METRICS},
        }
    elapsed = round(perf_counter() - start, 4)
    report = outcome.pop("report")
    calls = outcome["tool_calls"]
    return {
        **outcome, "elapsed_seconds": elapsed,
        "tool_call_count": len(calls),
        "followup_call_count": sum(call["action"] not in INITIAL_CALLS for call in calls),
        "knowledge_call_count": sum(call["action"] == "lookup_known_issue" for call in calls),
        **score_report(report, expected, device_ids, mode),
    }


def compare(mode: str) -> dict:
    if mode not in ("offline", "deepseek"):
        raise ValueError("对照评测规划器必须是 offline 或 deepseek")
    if mode == "deepseek":
        from model.factory import get_chat_model

        get_chat_model()  # Fail visibly on missing provider configuration, before evaluating cases.

    expected = json.loads(Path(__file__).with_name("expected_results.json").read_text(encoding="utf-8"))
    comparison_expected = json.loads(Path(__file__).with_name("comparison_expected.json").read_text(encoding="utf-8"))
    cases = []
    tracing = tracing_context(enabled=False) if mode == "offline" else nullcontext()
    with tracing, TemporaryDirectory() as directory:
        for name in CASES:
            devices, case = load_case(name)
            specification = comparison_expected[name] if name in comparison_expected else expected["cases"][name]
            device_ids = specification.get("device_ids", expected["device_ids"])
            scope = InvestigationScope(**case["scope"])
            db_path = Path(directory) / f"{name}.sqlite3"
            create_fixture_db(db_path, devices, case)
            cases.append({
                "case": name, "scope": scope.to_dict(),
                "single": _measure(_single, db_path, scope, mode, specification, device_ids),
                "multi": _measure(_multi, db_path, scope, mode, specification, device_ids),
            })
    return {
        "planner": mode, "question": DEFAULT_QUESTION, "simulated_data": True,
        "knowledge_mode": "simulated" if mode == "offline" else "configured_rag",
        "budgets": {
            "initial_queries_per_run": len(INITIAL_CALLS), "knowledge_calls_per_run": 1,
            "single_followups": COMPARISON_FOLLOWUP_BUDGET,
            "multi_followups": 2 * INITIAL_TOOL_BUDGET + REVIEW_TOOL_BUDGET,
            "multi_initial_per_role": INITIAL_TOOL_BUDGET, "multi_review": REVIEW_TOOL_BUDGET,
        },
        "cases": cases,
        "metrics": {
            design: {
                metric: {
                    "numerator": sum(case[design]["passed"][metric] for case in cases),
                    "denominator": len(cases),
                }
                for metric in METRICS
            }
            for design in ("single", "multi")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single- and multi-agent investigations.")
    parser.add_argument("--planner", choices=("offline", "deepseek"), required=True)
    args = parser.parse_args()
    try:
        result = compare(args.planner)
    except (RuntimeError, ValueError) as error:
        parser.error(_failure_reason(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
