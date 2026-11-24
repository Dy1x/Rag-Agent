import re

from service.billing.ids import new_id
from service.billing.models import (
    BillingFacts,
    Proposal,
    ProposedAction,
    ReviewIssue,
    SoftReviewOpinion,
)

RELEVANT = {
    "outage_billing": {
        "DUPLICATE_DEDUCT",
        "UNRELEASED_FREEZE",
        "MISSING_END_READING",
        "TARIFF_MISMATCH",
        "IDLE_FEE_MISMATCH",
        "PENDING_REFUND",
    },
    "double_charge": {"DUPLICATE_DEDUCT"},
    "failed_start_charged": {"UNRELEASED_FREEZE"},
    "frozen_balance": {"UNRELEASED_FREEZE", "MISSING_END_READING"},
    "wrong_tariff": {"TARIFF_MISMATCH"},
    "idle_fee": {"IDLE_FEE_MISMATCH"},
    "refund_status": {"PENDING_REFUND"},
    "recharge_missing": {"PENDING_RECHARGE"},
}
OVERPROMISE = re.compile("立即到账|保证(?!金)|一定|全额赔偿")
LEAK = re.compile(r"(?:CUS|TXN|ORD|SES|ATT|MTR|POLICY|PRP|RUN|ACT)-", re.I)


def remedies(facts, kind):
    actions = []
    for anomaly in facts.anomalies:
        if anomaly.code not in RELEVANT.get(kind, set()):
            continue
        if anomaly.code == "UNRELEASED_FREEZE":
            actions.append(
                ProposedAction(
                    action_type="unfreeze",
                    target_txn_id=anomaly.evidence_ids[-1],
                    amount_fen=anomaly.amount_fen,
                    reason=anomaly.detail,
                )
            )
    for order in facts.orders:
        remaining = order.order_refundable_fen or 0
        target_orders = [
            a
            for a in facts.anomalies
            if order.order.order_id in a.evidence_ids
            and a.code in RELEVANT.get(kind, set())
        ]
        if not target_orders or not remaining:
            continue
        targets = order.deducts[1:] if len(order.deducts) > 1 else order.deducts
        for t in targets:
            reserved = sum(
                r.amount_fen
                for r in order.refunds
                if r.related_txn_id == t.txn_id and r.status in ("pending", "succeeded")
            )
            amount = min(remaining, t.amount_fen - reserved)
            if amount > 0:
                actions.append(
                    ProposedAction(
                        action_type="refund",
                        target_txn_id=t.txn_id,
                        amount_fen=amount,
                        reason=target_orders[0].detail,
                    )
                )
                remaining -= amount
    return actions


def offline_maker(payload, tools):
    facts = BillingFacts.model_validate(payload["facts"])
    kind = payload["case_file"]["dispute_type"]
    actions = remedies(facts, kind)
    order = facts.orders[0] if facts.orders else None
    policy = tools["get_policy"](order.policy_id) if order else None
    relevant = [a for a in facts.anomalies if a.code in RELEVANT.get(kind, set())]
    ids = list(dict.fromkeys(k for a in relevant for k in a.evidence_ids))
    if order:
        ids.extend([order.order.order_id, order.policy_id])
    ids.extend(a.target_txn_id for a in actions)
    message = "已核对扣款与计费规则，当前没有需要退回的差额。"
    if actions:
        message = (
            "核查发现"
            + ("冻结尚未解除" if actions[0].action_type == "unfreeze" else "多收费用")
            + f"，拟处理 {sum(a.amount_fen for a in actions)} 分，需经人工审批。"
        )
    elif kind in ("refund_status", "outage_billing") and any(
        a.code == "PENDING_REFUND" for a in relevant
    ):
        message = "这笔退款正在处理中，完成后会退回钱包。"
    elif kind == "idle_fee" and policy:
        message = f"停止充电后宽限 {policy['idle_grace_minutes']} 分钟，超过后每分钟 {policy['idle_fee_fen_per_minute']} 分，本单占位费 {order.recomputed.idle_fee_fen} 分。"
    if not ids:
        ids = list(facts.evidence_index)[:1]
    return Proposal(
        proposal_id=new_id("PRP"),
        version=payload["version"],
        resolution="action" if actions else "explained",
        actions=actions,
        cited_policy_id=order.policy_id if order else None,
        rationale="、".join(a.code for a in relevant) or "订单与政策核对一致",
        evidence_ids=list(dict.fromkeys(ids)),
        customer_message=message,
    )


def customer_message_issues(message: str) -> list[ReviewIssue]:
    issues = []
    if OVERPROMISE.search(message):
        issues.append(ReviewIssue(code="S_OVERPROMISE", detail="车主说明包含过度承诺"))
    if LEAK.search(message):
        issues.append(
            ReviewIssue(
                code="S_LEAK",
                detail="请移除车主说明中的内部标识，将其保留在 evidence_ids 中",
            )
        )
    return issues


def offline_soft(payload, tools, facts):
    p = Proposal.model_validate(payload["proposal"])
    issues = []

    def flag(code, detail):
        issues.append(ReviewIssue(code=code, detail=detail))

    actual = {a.code for a in facts.anomalies}
    claimed = set(re.findall(r"\b[A-Z]+(?:_[A-Z]+)+\b", p.rationale))
    if claimed - actual:
        flag("S_RATIONALE_INCONSISTENT", "理由声称的异常缺少证据")
    expected = remedies(facts, payload["case_file"]["dispute_type"])
    if {(a.action_type, a.target_txn_id, a.amount_fen) for a in expected} != {
        (a.action_type, a.target_txn_id, a.amount_fen) for a in p.actions
    }:
        flag("S_MISSED_REMEDY", "资金处置与本次诉求的已核实差额不符")
    message = p.customer_message
    issues.extend(customer_message_issues(message))
    if "占位" in message and facts.orders:
        policy = tools["get_policy"](facts.orders[0].policy_id)
        for pattern, field in [
            (r"宽限\s*(\d+)\s*分钟", "idle_grace_minutes"),
            (r"每分钟\s*(\d+)\s*分", "idle_fee_fen_per_minute"),
        ]:
            match = re.search(pattern, message)
            if match and int(match[1]) != policy[field]:
                flag("S_POLICY_MISREAD", "占位费说明与政策不一致")
    return SoftReviewOpinion(approved=not issues, issues=issues)
