"""Connection-scoped operations shared by simulation, reconciliation and settlement."""

from datetime import datetime, timedelta
from sqlalchemy import text

from .errors import BillingError
from .ids import new_id
from .models import PolicyView, TxnView, WalletView
from .timeutil import utc_now


def rows(conn, sql, **params) -> list[dict]:
    return [dict(r) for r in conn.execute(text(sql), params).mappings()]


def one(conn, sql, **params) -> dict:
    result = rows(conn, sql, **params)
    if not result:
        raise BillingError("NOT_FOUND", "记录不存在")
    return result[0]


def insert(conn, table: str, **fields) -> None:
    # Table and column names are internal constants. Values always use bindings.
    columns = ", ".join(fields)
    binds = ", ".join(":" + col for col in fields)
    conn.execute(text(f"INSERT INTO cb_{table} ({columns}) VALUES ({binds})"), fields)


def lock_wallet(conn, customer: str) -> dict:
    return one(
        conn, "SELECT * FROM cb_wallets WHERE customer_id=:c FOR UPDATE", c=customer
    )


def wallet(conn, customer: str) -> WalletView:
    w = one(
        conn,
        """SELECT w.*, COALESCE(SUM(CASE
          WHEN t.txn_type IN ('recharge','refund') THEN t.amount_fen
          WHEN t.txn_type='deduct' THEN -t.amount_fen ELSE 0 END),0) ledger_balance,
        COALESCE(SUM(CASE WHEN t.txn_type='freeze' THEN t.amount_fen
          WHEN t.txn_type='unfreeze' THEN -t.amount_fen ELSE 0 END),0) ledger_frozen
        FROM cb_wallets w LEFT JOIN cb_wallet_transactions t
          ON t.customer_id=w.customer_id AND t.status='succeeded'
        WHERE w.customer_id=:c GROUP BY w.customer_id""",
        c=customer,
    )
    return WalletView(
        **w,
        available_fen=w["balance_fen"] - w["frozen_fen"],
        consistent=(w["balance_fen"], w["frozen_fen"])
        == (w["ledger_balance"], w["ledger_frozen"]),
    )


def policy(conn, policy_id: str) -> PolicyView:
    p = one(conn, "SELECT * FROM cb_tariff_policies WHERE policy_id=:p", p=policy_id)
    periods = rows(
        conn,
        "SELECT * FROM cb_tariff_periods WHERE policy_id=:p ORDER BY start_local",
        p=policy_id,
    )
    for period in periods:
        for field in ("start_local", "end_local"):
            if isinstance(period[field], timedelta):
                period[field] = (datetime.min + period[field]).time()
    return PolicyView(**p, periods=periods)


def project(conn, customer: str, kind: str, amount: int) -> None:
    balance, frozen = {
        "recharge": (amount, 0),
        "deduct": (-amount, 0),
        "refund": (amount, 0),
        "freeze": (0, amount),
        "unfreeze": (0, -amount),
    }[kind]
    conn.execute(
        text("""UPDATE cb_wallets SET balance_fen=balance_fen+:b, frozen_fen=frozen_fen+:f,
        version=version+1, updated_at=UTC_TIMESTAMP(3) WHERE customer_id=:c"""),
        dict(c=customer, b=balance, f=frozen),
    )


def transaction(
    conn,
    customer: str,
    kind: str,
    amount: int,
    *,
    status="succeeded",
    source="simulator",
    at=None,
    **links,
) -> TxnView:
    if type(amount) is not int or amount <= 0:
        raise BillingError("VALIDATION", "金额须为正整数分")
    when = at or utc_now()
    fields = dict(
        txn_id=new_id("TXN"),
        customer_id=customer,
        txn_type=kind,
        amount_fen=amount,
        status=status,
        source=source,
        created_at=when,
        finished_at=when if status == "succeeded" else None,
        **links,
    )
    insert(conn, "wallet_transactions", **fields)
    if status == "succeeded":
        project(conn, customer, kind, amount)
    return TxnView(**fields)


def refundable(conn, target: dict) -> int:
    used = one(
        conn,
        """SELECT COALESCE(SUM(amount_fen),0) used FROM cb_wallet_transactions
        WHERE related_txn_id=:t AND txn_type='refund' AND status IN ('pending','succeeded')""",
        t=target["txn_id"],
    )["used"]
    return max(0, target["amount_fen"] - int(used))


def release_txn_id(conn, target_id: str, customer_id: str) -> str | None:
    return conn.scalar(
        text("""SELECT txn_id FROM cb_wallet_transactions WHERE related_txn_id=:t
            AND customer_id=:c AND txn_type='unfreeze' AND status='succeeded'
            ORDER BY created_at,txn_id LIMIT 1"""),
        {"t": target_id, "c": customer_id},
    )
