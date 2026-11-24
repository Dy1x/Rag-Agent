from datetime import datetime
from sqlalchemy import Engine

from utils.config_handler import billing_conf
from . import ledger
from .errors import BillingError
from .ids import is_record_suffix
from .models import (
    Anomaly,
    AttemptView,
    BillingFacts,
    FreezeView,
    Identity,
    OrderCandidate,
    OrderFacts,
    OrderView,
    PolicyView,
    RefundableView,
    SessionView,
    TxnView,
)
from .tariff import AmountBreakdown, compute_order_amount


def _scope(identity: Identity, alias: str, params: dict) -> str:
    if identity.role == "operator":
        return "1=1"
    params["owner"] = identity.actor_id
    return f"{alias}.customer_id=:owner"


def _accessible(c, identity, sql, alias, **params):
    data = ledger.rows(c, sql + " AND " + _scope(identity, alias, params), **params)
    if not data:
        raise BillingError("FORBIDDEN", "不可访问")
    return data[0]


def _order(c, identity, key):
    return OrderView(
        **_accessible(
            c, identity, "SELECT o.* FROM cb_orders o WHERE o.order_id=:k", "o", k=key
        )
    )


def _attempt(c, identity, key):
    return AttemptView(
        **_accessible(
            c,
            identity,
            "SELECT a.* FROM cb_charging_attempts a WHERE a.attempt_id=:k",
            "a",
            k=key,
        )
    )


def _session(c, identity, key):
    row = _accessible(
        c,
        identity,
        """SELECT s.* FROM cb_charging_sessions s JOIN cb_charging_attempts a
        USING(attempt_id) WHERE s.session_id=:k""",
        "a",
        k=key,
    )
    readings = ledger.rows(
        c,
        "SELECT * FROM cb_meter_readings WHERE session_id=:k ORDER BY occurred_at,reading_id",
        k=key,
    )
    return SessionView(**row, readings=readings)


def _recompute(policy, session):
    meter = {r.reading_type: r.energy_wh for r in session.readings}
    return compute_order_amount(
        policy,
        session.started_at,
        session.charge_end_at,
        session.unplugged_at,
        meter.get("start"),
        meter.get("end"),
        billing_conf["local_timezone"],
    )


def _transactions(c, identity, *, order_id=None, attempt_id=None, txn_ids=None):
    params = {}
    clauses = [_scope(identity, "t", params)]
    for field, value in (("order_id", order_id), ("attempt_id", attempt_id)):
        if value is not None:
            params[field] = value
            clauses.append(f"t.{field}=:{field}")
    if txn_ids is not None:
        if not txn_ids:
            return []
        binds = {f"t{i}": k for i, k in enumerate(txn_ids)}
        clauses.append("t.txn_id IN (" + ",".join(":" + k for k in binds) + ")")
        params.update(binds)
    query = (
        "SELECT t.* FROM cb_wallet_transactions t WHERE "
        + " AND ".join(clauses)
        + " ORDER BY t.created_at,t.txn_id"
    )
    return [TxnView(**r) for r in ledger.rows(c, query, **params)]


class BillingFactsService:
    def session_locator(self, identity: Identity, session_id: str) -> dict:
        with self.engine.begin() as c:
            return _accessible(
                c,
                identity,
                """SELECT a.attempt_id,o.order_id FROM cb_charging_sessions s
                JOIN cb_charging_attempts a USING(attempt_id) LEFT JOIN cb_orders o ON o.session_id=s.session_id
                WHERE s.session_id=:key""",
                "a",
                key=session_id,
            )

    def get_readings(self, identity: Identity, reading_id: str):
        from .models import ReadingView

        params = {"key": reading_id}
        query = """SELECT m.* FROM cb_meter_readings m JOIN cb_charging_sessions s USING(session_id)
            JOIN cb_charging_attempts a USING(attempt_id) WHERE m.reading_id=:key AND """
        with self.engine.begin() as conn:
            return [
                ReadingView(**r)
                for r in ledger.rows(
                    conn, query + _scope(identity, "a", params), **params
                )
            ]

    def __init__(self, engine: Engine):
        self.engine = engine

    def find_candidate_orders(
        self,
        identity: Identity,
        *,
        order_id: str | None = None,
        charger_id: str | None = None,
        connector_id: str | None = None,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        limit: int = 5,
    ) -> list[OrderCandidate]:
        params = {"lim": max(1, min(100, limit))}
        clauses = [_scope(identity, "a", params)]
        if order_id is not None:
            if is_record_suffix(order_id):
                clauses.append(
                    "UPPER(COALESCE(o.order_id, a.attempt_id)) LIKE :order_suffix"
                )
                params["order_suffix"] = f"%{order_id.upper()}"
            else:
                clauses.append("o.order_id=:o")
                params["o"] = order_id
        for col, op, key, val in [
            ("a.charger_id", "=", "ch", charger_id),
            ("a.connector_id", "=", "n", connector_id),
            ("a.requested_at", ">=", "start", start_utc),
            ("a.requested_at", "<", "end", end_utc),
        ]:
            if val is not None:
                clauses.append(f"{col}{op}:{key}")
                params[key] = val
        with self.engine.begin() as c:
            found = ledger.rows(
                c,
                """SELECT a.*,o.order_id FROM cb_charging_attempts a LEFT JOIN cb_orders o
                USING(attempt_id) WHERE """
                + " AND ".join(clauses)
                + " ORDER BY a.requested_at DESC,a.attempt_id LIMIT :lim",
                **params,
            )
            return [OrderCandidate(**r) for r in found]

    def get_attempts(
        self, identity: Identity, attempt_ids: list[str]
    ) -> list[AttemptView]:
        with self.engine.begin() as c:
            return [_attempt(c, identity, k) for k in attempt_ids]

    def get_session(self, identity: Identity, session_id: str) -> SessionView:
        with self.engine.begin() as c:
            return _session(c, identity, session_id)

    def get_order(self, identity: Identity, order_id: str) -> OrderView:
        with self.engine.begin() as c:
            return _order(c, identity, order_id)

    def get_transactions(
        self, identity: Identity, *, order_id=None, attempt_id=None, txn_ids=None
    ) -> list[TxnView]:
        with self.engine.begin() as c:
            return _transactions(
                c, identity, order_id=order_id, attempt_id=attempt_id, txn_ids=txn_ids
            )

    def get_policy(self, identity: Identity, policy_id: str) -> PolicyView:
        with self.engine.begin() as c:
            return ledger.policy(c, policy_id)

    def recompute_order(self, identity: Identity, order_id: str) -> AmountBreakdown:
        with self.engine.begin() as c:
            order = _order(c, identity, order_id)
            return _recompute(
                ledger.policy(c, order.policy_id),
                _session(c, identity, order.session_id),
            )

    def compute_refundable(
        self, identity: Identity, deduct_txn_id: str
    ) -> RefundableView:
        with self.engine.begin() as c:
            t = _accessible(
                c,
                identity,
                "SELECT * FROM cb_wallet_transactions t WHERE txn_id=:t",
                "t",
                t=deduct_txn_id,
            )
            if t["txn_type"] != "deduct" or t["status"] != "succeeded":
                raise BillingError("VALIDATION", "目标须为成功扣款")
            return RefundableView(
                txn_id=deduct_txn_id, refundable_fen=ledger.refundable(c, t)
            )

    def build_billing_facts(
        self, identity: Identity, *, order_ids: list[str], attempt_ids: list[str]
    ) -> BillingFacts:
        with (
            self.engine.connect().execution_options(
                isolation_level="REPEATABLE READ"
            ) as c,
            c.begin(),
        ):
            orders = [_order(c, identity, k) for k in dict.fromkeys(order_ids)]
            attempts = [
                _attempt(c, identity, k)
                for k in dict.fromkeys([*attempt_ids, *(o.attempt_id for o in orders)])
            ]
            owners = {a.customer_id for a in attempts}
            if len(owners) > 1:
                raise BillingError("VALIDATION", "同一案件只能核查一位车主")
            customer = next(iter(owners), identity.actor_id)
            facts = BillingFacts(customer_id=customer, attempts=attempts)
            for a in attempts:
                facts.evidence_index[a.attempt_id] = "attempt"
                txns = _transactions(c, identity, attempt_id=a.attempt_id)
                for t in txns:
                    facts.evidence_index[t.txn_id] = "txn"
                    if t.txn_type == "freeze" and t.status == "succeeded":
                        release = ledger.release_txn_id(c, t.txn_id, customer)
                        facts.freezes.append(
                            FreezeView(**t.model_dump(), released_by_txn_id=release)
                        )
                        if release:
                            facts.evidence_index[release] = "txn"
                        if (
                            a.result in ("timeout", "rejected")
                            or any(
                                o.attempt_id == a.attempt_id and o.status == "settled"
                                for o in orders
                            )
                        ) and release is None:
                            _flag(
                                facts,
                                "UNRELEASED_FREEZE",
                                "冻结流水尚未解冻",
                                t.amount_fen,
                                [a.attempt_id, t.txn_id],
                            )
            for order in orders:
                self._order_facts(c, identity, facts, order)
            pending = ledger.rows(
                c,
                """SELECT * FROM cb_wallet_transactions WHERE customer_id=:c
                AND status='pending' AND txn_type='recharge' ORDER BY created_at""",
                c=customer,
            )
            for r in pending:
                t = TxnView(**r)
                facts.pending_recharges.append(t)
                facts.evidence_index[t.txn_id] = "txn"
                _flag(
                    facts,
                    "PENDING_RECHARGE",
                    "充值回调待确认",
                    t.amount_fen,
                    [t.txn_id],
                )
                facts.missing_evidence.append("recharge_callback")
            if not ledger.wallet(c, customer).consistent:
                _flag(facts, "WALLET_PROJECTION_MISMATCH", "钱包与流水不一致", None, [])
            return facts

    def _order_facts(self, c, identity, facts, order):
        snapshot = order_snapshot(c, identity, order)
        facts.orders.append(snapshot)
        session = snapshot.session
        amount = snapshot.recomputed
        deducts = snapshot.deducts
        pending = [t for t in snapshot.refunds if t.status == "pending"]
        facts.evidence_index.update(
            {
                order.order_id: "order",
                session.session_id: "session",
                order.policy_id: "policy",
            }
        )
        facts.evidence_index.update({r.reading_id: "reading" for r in session.readings})
        if len(deducts) > 1:
            _flag(
                facts,
                "DUPLICATE_DEDUCT",
                "同一订单多笔扣款",
                sum(t.amount_fen for t in deducts[1:]),
                [order.order_id, *[t.txn_id for t in deducts[1:]]],
            )
        if "end_wh" in amount.missing:
            facts.missing_evidence.append("end_wh")
            _flag(
                facts,
                "MISSING_END_READING",
                "缺少结束电表读数",
                None,
                [order.order_id, session.session_id],
            )
        else:
            for code, fields in [
                ("TARIFF_MISMATCH", ["electricity_fee_fen", "service_fee_fen"]),
                ("IDLE_FEE_MISMATCH", ["idle_fee_fen"]),
            ]:
                delta = sum(getattr(order, f) - getattr(amount, f) for f in fields)
                if any(getattr(order, f) != getattr(amount, f) for f in fields):
                    _flag(
                        facts,
                        code,
                        "订单与政策重算不符",
                        delta,
                        [order.order_id, order.policy_id],
                    )
        for t in pending:
            _flag(
                facts,
                "PENDING_REFUND",
                "退款处理中",
                t.amount_fen,
                [t.txn_id, order.order_id],
            )


def _flag(facts, code, detail, amount, ids):
    facts.anomalies.append(
        Anomaly(code=code, detail=detail, amount_fen=amount, evidence_ids=ids)
    )


def order_snapshot(c, identity, order) -> OrderFacts:
    session = _session(c, identity, order.session_id)
    amount = _recompute(ledger.policy(c, order.policy_id), session)
    txns = _transactions(c, identity, order_id=order.order_id)
    deducts = [t for t in txns if t.txn_type == "deduct" and t.status == "succeeded"]
    refunds = [t for t in txns if t.txn_type == "refund"]
    net = sum(t.amount_fen for t in deducts) - sum(
        t.amount_fen for t in refunds if t.status == "succeeded"
    )
    pending = [t for t in refunds if t.status == "pending"]
    cap = (
        None
        if amount.total_fen is None
        else max(0, net - amount.total_fen - sum(t.amount_fen for t in pending))
    )
    return OrderFacts(
        order=order,
        session=session,
        policy_id=order.policy_id,
        recomputed=amount,
        charged_net_fen=net,
        deducts=deducts,
        refunds=refunds,
        order_refundable_fen=cap,
    )


def current_order_refundable(conn, customer_id: str, order_id: str) -> int | None:
    identity = Identity(role="customer", actor_id=customer_id)
    return order_snapshot(
        conn, identity, _order(conn, identity, order_id)
    ).order_refundable_fen
