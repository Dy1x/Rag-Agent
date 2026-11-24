import json
from pathlib import Path

from agent.cases.billing.offline import LEAK, OVERPROMISE
from service.billing.facts_service import BillingFactsService
from service.billing.models import Identity

CASES = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))


def expected_actions(scenario):
    return sorted(
        (
            a["action_type"],
            scenario.refs[a["target"].removeprefix("refs.")],
            a["amount_fen"],
        )
        for a in CASES[scenario.name]["actions"]
    )


def proposal_errors(p, scenario, evidence):
    errors = []
    if sorted(
        (a["action_type"], a["target_txn_id"], a["amount_fen"]) for a in p["actions"]
    ) != expected_actions(scenario):
        errors.append("方案的动作、目标或金额不符合场景期望")
    if not set(p["evidence_ids"]) <= evidence.keys():
        errors.append("方案引用无效证据")
    if p["cited_policy_id"] and evidence.get(p["cited_policy_id"]) != "policy":
        errors.append("引用政策与本案不符")
    if LEAK.search(p["customer_message"]) or OVERPROMISE.search(p["customer_message"]):
        errors.append("车主说明含内部编号或过度承诺")
    return errors


def check_report(scenario, report, evidence):
    expected = CASES[scenario.name]
    errors = []
    if report["status"] != expected["status"]:
        errors.append(f"期望 {expected['status']}，实际 {report['status']}")
    actual = sorted(
        (a["action_type"], a["target_txn_id"], a["amount_fen"])
        for a in report["action_results"]
        if a["status"] == "succeeded"
    )
    if actual != expected_actions(scenario):
        errors.append("实际资金动作的类型、目标或金额不符")
    if report["proposals"]:
        errors.extend(proposal_errors(report["proposals"][-1], scenario, evidence))
    message = report["customer_message"] or ""
    if LEAK.search(message) or OVERPROMISE.search(message):
        errors.append("最终说明含内部编号或过度承诺")
    if scenario.name == "S7" and any(
        x in json.dumps(report, ensure_ascii=False) for x in scenario.refs.values()
    ):
        errors.append("越权报告包含其他车主的记录")
    if "candidate_count" in expected:
        candidates = report["candidates"]
        if len(candidates) != expected["candidate_count"] or LEAK.search(
            json.dumps(candidates, ensure_ascii=False)
        ):
            errors.append("候选数量或脱敏不符合期望")
    if scenario.name == "S5" and "处理" not in message and "在途" not in message:
        errors.append("缺少退款在途说明")
    if scenario.name == "S9" and ("15" not in message or "50" not in message):
        errors.append("未说明占位费宽限和单价")
    if report.get("failure"):
        errors.append(report["failure"])
    return errors


def reference_evidence(engine, scenario):
    identity = Identity(role="customer", actor_id=scenario.customer_id)
    service = BillingFactsService(engine)
    candidates = service.find_candidate_orders(identity)
    facts = service.build_billing_facts(
        identity,
        order_ids=[r.order_id for r in candidates if r.order_id],
        attempt_ids=[r.attempt_id for r in candidates],
    )
    return facts.evidence_index
