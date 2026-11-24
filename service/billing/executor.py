from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from agent.cases.fund_actions import ACTIONS
from . import ledger
from .errors import BillingError
from .ids import new_id
from .models import ApprovedAction, FundActionResult
from .timeutil import utc_now


class FundActionExecutor:
    def __init__(self, engine: Engine):
        self.engine = engine

    def _previous(self, c, key, action):
        previous = ledger.rows(
            c, "SELECT * FROM cb_fund_actions WHERE idempotency_key=:k", k=key
        )
        if previous:
            row = previous[0]
            if any(
                row[k] != getattr(action, k)
                for k in ("customer_id", "action_type", "target_txn_id", "amount_fen")
            ):
                raise BillingError("IDEMPOTENCY_CONFLICT", "同一审批动作的参数发生变化")
            return FundActionResult(**row)

    def execute(self, action: ApprovedAction) -> FundActionResult:
        key = f"{action.proposal_id}:{action.action_index}"
        try:
            with self.engine.begin() as c:
                ledger.lock_wallet(c, action.customer_id)
                previous = self._previous(c, key, action)
                if previous:
                    return previous
                targets = ledger.rows(
                    c,
                    """SELECT * FROM cb_wallet_transactions
                    WHERE txn_id=:t AND customer_id=:owner FOR UPDATE""",
                    t=action.target_txn_id,
                    owner=action.customer_id,
                )
                spec = ACTIONS.get(action.action_type)
                reason = "不可访问"
                if targets:
                    reason = (
                        spec.revalidate(c, action, targets[0])
                        if spec
                        else "不支持的资金动作"
                    )
                txn = None
                if reason is None:
                    txn = spec.apply(c, action, targets[0], key)
                result = FundActionResult(
                    action_id=new_id("ACT"),
                    action_type=action.action_type,
                    target_txn_id=action.target_txn_id,
                    amount_fen=action.amount_fen,
                    result_txn_id=txn.txn_id if txn else None,
                    status="rejected" if reason else "succeeded",
                    reject_reason=reason,
                )
                ledger.insert(
                    c,
                    "fund_actions",
                    **result.model_dump(),
                    proposal_id=action.proposal_id,
                    customer_id=action.customer_id,
                    action_index=action.action_index,
                    idempotency_key=key,
                    created_at=utc_now(),
                )
                return result
        except IntegrityError:
            with self.engine.begin() as c:
                previous = self._previous(c, key, action)
                if previous:
                    return previous
            raise
