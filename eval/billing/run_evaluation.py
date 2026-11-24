import argparse
import json
from time import perf_counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from agent.cases.billing.handler import BillingDisputeHandler
from service.billing.db import (
    apply_schema,
    create_database,
    drop_database,
    get_billing_engine,
)
from service.billing.models import ApprovalDecision, Identity
from service.billing.scenarios import SCENARIOS, build_scenario
from service.billing.simulator_service import SimulatorService
from service.billing.facts_service import BillingFactsService
from utils.config_handler import billing_conf, rag_conf
from .checks import (
    CASES,
    check_report,
    expected_actions,
    proposal_errors,
    reference_evidence,
)
from .mutations import injection_blocker, mutate

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def model_settings():
    return {
        "chat_model_name": rag_conf["chat_model_name"],
        **{
            key: billing_conf[key]
            for key in (
                "checker_max_tool_calls",
                "maker_max_tool_calls",
                "max_revisions",
            )
        },
    }


@contextmanager
def evaluation_database(keep=False):
    name = "cb_eval_" + uuid4().hex[:8]
    create_database(name)
    try:
        engine = get_billing_engine(name)
        apply_schema(engine)
        yield engine
    finally:
        if not keep:
            drop_database(name)


def evaluate_case(engine, name, planner, *, design="C", mutation=None):
    sim = SimulatorService(engine)
    scenario = build_scenario(sim, name)
    evidence = reference_evidence(engine, scenario)
    injected = False
    injection_reason = "运行未生成首版方案" if mutation else None

    def transform(proposal, facts):
        nonlocal injected, injection_reason
        altered = mutate(mutation, proposal, facts)
        injected = altered != proposal
        injection_reason = (
            None
            if injected
            else injection_blocker(mutation, proposal, facts) or "变异未改变首版方案"
        )
        return altered

    handler = BillingDisputeHandler(
        engine,
        InMemorySaver(),
        design=design,
        transform=transform if mutation else None,
    )
    started = perf_counter()
    identity = Identity(role="customer", actor_id=scenario.customer_id)
    r = handler.start(
        identity,
        scenario.suggested_question,
        scenario.suggested_locator,
        planner=planner,
    )
    if r.status == "action_pending_approval":
        r = handler.resume(
            r.run_id, ApprovalDecision(decision="approve", approver="evaluation")
        )
    errors = check_report(scenario, r.report, evidence)
    actual = sorted(
        (a["action_type"], a["target_txn_id"], a["amount_fen"])
        for a in r.report["action_results"]
        if a["status"] == "succeeded"
    )
    correct_outcome = r.status == CASES[name]["status"] and actual == expected_actions(
        scenario
    )
    if mutation and not injected:
        errors.append(f"首版方案未注入指定变异（{injection_reason}）")
    trace = handler.store.get_trace(r.run_id)
    after = None
    rerun_report = None
    if "after_backfill" in CASES[name] and r.status == "pending_evidence":
        session = BillingFactsService(engine).get_session(
            identity, scenario.refs["session"]
        )
        sim.backfill_end_reading(session.session_id, 4000, session.charge_end_at)
        rerun = handler.rerun(r.case_id, identity)
        after = rerun.status
        rerun_report = rerun.report
        trace.extend(handler.store.get_trace(rerun.run_id))
        if after != CASES[name]["after_backfill"]:
            errors.append("补传后重跑状态错误：" + after)
            details = rerun.report or {}
            if details.get("failure"):
                errors.append(details["failure"])
            elif details.get("reviews"):
                errors.extend(i["detail"] for i in details["reviews"][-1]["issues"])
            correct_outcome = False
    first_review = next(iter(r.report["reviews"]), None)
    first_correct = bool(
        first_review
        and not proposal_errors(r.report["proposals"][0], scenario, evidence)
    )
    return dict(
        name=name,
        design=design,
        mutation=mutation,
        injected=injected,
        not_injected_reason=injection_reason,
        passed=not errors,
        errors=errors,
        correct_outcome=correct_outcome,
        first_correct=first_correct,
        first_rejected=bool(first_review and not first_review["approved"]),
        revisions=max(0, len(r.report["proposals"]) - 1),
        after_backfill=after,
        **({"after_backfill_report": rerun_report} if rerun_report else {}),
        unauthorized_calls=sum(e["event_type"] == "unauthorized_tool" for e in trace),
        llm_calls=sum(e["event_type"] == "model_call" for e in trace),
        tokens=sum(
            e["payload"].get("total_tokens", 0)
            for e in trace
            if e["event_type"] == "model_usage"
        ),
        seconds=round(perf_counter() - started, 3),
        report=r.report,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["offline", "deepseek"], default="offline")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()
    with evaluation_database(args.keep_db) as engine:

        def run(name):
            try:
                result = evaluate_case(engine, name, args.planner)
            except Exception as e:
                result = dict(
                    name=name, passed=False, errors=[f"{type(e).__name__}: {e}"]
                )
            print(name, "PASS" if result["passed"] else result["errors"], flush=True)
            return result

        with ThreadPoolExecutor(
            max_workers=3 if args.planner == "deepseek" else 1
        ) as pool:
            cases = list(pool.map(run, SCENARIOS))
        report = dict(
            planner=args.planner,
            metadata=model_settings(),
            database=engine.url.database,
            passed=sum(r["passed"] for r in cases),
            total=len(cases),
            cases=cases,
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"billing-eval-{args.planner}-{uuid4().hex[:8]}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"{report['passed']}/{report['total']} {path}")
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
