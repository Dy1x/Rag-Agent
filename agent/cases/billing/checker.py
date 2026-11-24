from collections import Counter

from agent.cases.agent_spec import AgentSpec
from agent.cases.fund_actions import ACTIONS
from service.billing.models import ReviewIssue, SoftReviewOpinion
from utils.config_handler import billing_conf

SPEC = AgentSpec(
    "checker",
    "billing_checker.txt",
    [
        "get_order",
        "get_transactions",
        "get_session",
        "get_policy",
        "recompute_order",
        "compute_refundable",
    ],
    SoftReviewOpinion,
    billing_conf["checker_max_tool_calls"],
)


def hard_checks(proposal, case_file, service, identity):
    facts = service.build_billing_facts(
        identity, order_ids=case_file.order_ids, attempt_ids=case_file.attempt_ids
    )
    issues = []

    def issue(code, detail):
        issues.append(ReviewIssue(code=code, detail=detail))

    if (proposal.resolution == "action") != bool(proposal.actions):
        issue("H_RESOLUTION_SHAPE", "方案种类与动作列表不一致")
    if any(k not in facts.evidence_index for k in proposal.evidence_ids):
        issue("H_EVIDENCE_UNKNOWN", "引用了本案以外的证据")
    if proposal.cited_policy_id:
        if not facts.orders or any(
            o.policy_id != proposal.cited_policy_id for o in facts.orders
        ):
            issue("H_POLICY_MISMATCH", "引用政策与订单不符")
        else:
            policy = service.get_policy(identity, proposal.cited_policy_id)
            if any(
                o.session.started_at < policy.effective_from
                or (policy.effective_to and o.session.started_at >= policy.effective_to)
                for o in facts.orders
            ):
                issue("H_POLICY_MISMATCH", "政策不在有效期")
    ids = [a.target_txn_id for a in proposal.actions]
    if len(set(ids)) != len(ids):
        issue("H_DUPLICATE_ACTION", "动作目标重复")
    totals = Counter()
    for a in proposal.actions:
        spec = ACTIONS.get(a.action_type)
        code = (
            spec.hard_check(a, facts, service, identity) if spec else "H_TARGET_INVALID"
        )
        if code:
            issue(code, "资金目标、金额或状态与证据不符")
        if a.action_type == "refund":
            for order in facts.orders:
                if a.target_txn_id in {t.txn_id for t in order.deducts}:
                    totals[order.order.order_id] += a.amount_fen
    for o in facts.orders:
        if totals[o.order.order_id] > (o.order_refundable_fen or 0):
            issue("H_REFUND_EXCEEDS", "订单合计退款超出可退额")
    return facts, issues
