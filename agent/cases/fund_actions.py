from dataclasses import dataclass
from typing import Callable

from service.billing import ledger


@dataclass(frozen=True)
class FundActionSpec:
    hard_check: Callable
    revalidate: Callable
    apply: Callable


ACTIONS: dict[str, FundActionSpec] = {}


def register_fund_action(action_type: str, *, hard_check, revalidate, apply) -> None:
    ACTIONS[action_type] = FundActionSpec(hard_check, revalidate, apply)


def _refund_check(action, facts, service, identity):
    for order in facts.orders:
        targets = [t.txn_id for t in order.deducts]
        if action.target_txn_id not in targets:
            continue
        if len(targets) > 1 and action.target_txn_id == targets[0]:
            return "H_TARGET_INVALID"
        available = service.compute_refundable(
            identity, action.target_txn_id
        ).refundable_fen
        if order.order_refundable_fen is None or action.amount_fen > min(
            available, order.order_refundable_fen
        ):
            return "H_REFUND_EXCEEDS"
        return None
    return "H_TARGET_INVALID"


def _unfreeze_check(action, facts, service, identity):
    for t in facts.freezes:
        if t.txn_id == action.target_txn_id:
            if t.amount_fen == action.amount_fen and any(
                a.code == "UNRELEASED_FREEZE" and t.txn_id in a.evidence_ids
                for a in facts.anomalies
            ):
                return None
            return "H_UNFREEZE_INVALID"
    return "H_TARGET_INVALID"


def _refund_latest(c, action, target):
    if target["txn_type"] != "deduct" or target["status"] != "succeeded":
        return "目标不是成功扣款"
    if action.amount_fen > ledger.refundable(c, target):
        return "超过当前扣款可退额"
    from service.billing.facts_service import current_order_refundable

    cap = current_order_refundable(c, action.customer_id, target["order_id"])
    if cap is None or action.amount_fen > cap:
        return "超过当前订单可退额"
    return None


def _unfreeze_latest(c, action, target):
    if target["txn_type"] != "freeze" or target["status"] != "succeeded":
        return "目标不是成功冻结"
    if action.amount_fen != target["amount_fen"] or ledger.release_txn_id(
        c, target["txn_id"], action.customer_id
    ):
        return "冻结已解除或金额不符"
    return None


def _apply(c, action, target, key):
    return ledger.transaction(
        c,
        action.customer_id,
        action.action_type,
        action.amount_fen,
        source="fund_executor",
        related_txn_id=target["txn_id"],
        attempt_id=target["attempt_id"],
        order_id=target["order_id"],
        idempotency_key=key,
    )


register_fund_action(
    "refund", hard_check=_refund_check, revalidate=_refund_latest, apply=_apply
)
register_fund_action(
    "unfreeze", hard_check=_unfreeze_check, revalidate=_unfreeze_latest, apply=_apply
)
