from datetime import datetime, timedelta
from sqlalchemy import Engine, text

from utils.config_handler import billing_conf
from . import ledger
from .errors import BillingError
from .ids import new_id
from .models import (
    AttemptView,
    ChargerView,
    CustomerView,
    OrderView,
    TxnView,
    WalletView,
)
from .tariff import compute_order_amount
from .timeutil import as_utc, utc_now


def _fault(value, *valid):
    if value is not None and value not in valid:
        raise BillingError("VALIDATION", "未知故障选项")


class SimulatorService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def list_customers(self) -> list[CustomerView]:
        with self.engine.begin() as c:
            return [
                CustomerView(**r)
                for r in ledger.rows(
                    c, "SELECT * FROM cb_customers ORDER BY created_at,customer_id"
                )
            ]

    def create_customer(self, nickname: str) -> CustomerView:
        if not 1 <= len(nickname.strip()) <= 64:
            raise BillingError("VALIDATION", "昵称须为 1 至 64 个字符")
        with self.engine.begin() as c:
            customer = CustomerView(
                customer_id=new_id("CUS"), nickname=nickname.strip()
            )
            ledger.insert(c, "customers", **customer.model_dump(), created_at=utc_now())
            ledger.insert(
                c, "wallets", customer_id=customer.customer_id, updated_at=utc_now()
            )
            return customer

    def get_wallet(self, customer_id: str) -> WalletView:
        with self.engine.begin() as c:
            return ledger.wallet(c, customer_id)

    def list_chargers(self) -> list[ChargerView]:
        with self.engine.begin() as c:
            return [
                ChargerView(**r)
                for r in ledger.rows(
                    c, "SELECT * FROM cb_chargers ORDER BY charger_id,connector_id"
                )
            ]

    def recharge(
        self, customer_id: str, amount_fen: int, *, fault=None, at=None
    ) -> TxnView:
        _fault(fault, "callback_pending")
        with self.engine.begin() as c:
            ledger.lock_wallet(c, customer_id)
            return ledger.transaction(
                c,
                customer_id,
                "recharge",
                amount_fen,
                at=at,
                status="pending" if fault else "succeeded",
                external_txn_no=new_id("EXT"),
            )

    def confirm_recharge(self, txn_id: str) -> TxnView:
        with self.engine.begin() as c:
            query = "SELECT * FROM cb_wallet_transactions WHERE txn_id=:t"
            target = ledger.one(c, query, t=txn_id)
            ledger.lock_wallet(c, target["customer_id"])
            target = ledger.one(c, query + " FOR UPDATE", t=txn_id)
            if target["txn_type"] != "recharge" or target["status"] != "pending":
                raise BillingError("INVALID_STATE", "只能确认待入账的充值")
            c.execute(
                text(
                    "UPDATE cb_wallet_transactions SET status='succeeded', finished_at=UTC_TIMESTAMP(3) WHERE txn_id=:t"
                ),
                dict(t=txn_id),
            )
            ledger.project(c, target["customer_id"], "recharge", target["amount_fen"])
            return TxnView(**ledger.one(c, query, t=txn_id))

    def start_charging(
        self,
        customer_id: str,
        charger_id: str,
        connector_id: str,
        *,
        at: datetime | None = None,
        start_wh: int = 0,
        fault=None,
    ) -> AttemptView:
        _fault(fault, "start_timeout", "timeout_keep_frozen")
        if type(start_wh) is not int or start_wh < 0:
            raise BillingError("VALIDATION", "起始读数须为非负整数")
        with self.engine.begin() as c:
            w = ledger.lock_wallet(c, customer_id)
            charger = ledger.one(
                c,
                "SELECT * FROM cb_chargers WHERE charger_id=:ch AND connector_id=:n",
                ch=charger_id,
                n=connector_id,
            )
            amount = ledger.policy(c, charger["policy_id"]).freeze_amount_fen
            if w["balance_fen"] - w["frozen_fen"] < amount:
                raise BillingError("INSUFFICIENT_BALANCE", "可用余额不足")
            when = as_utc(at or utc_now())
            attempt = AttemptView(
                attempt_id=new_id("ATT"),
                customer_id=customer_id,
                charger_id=charger_id,
                connector_id=connector_id,
                requested_at=when,
                result="timeout" if fault else "started",
                session_id=None if fault else new_id("SES"),
            )
            freeze = ledger.transaction(
                c, customer_id, "freeze", amount, at=when, attempt_id=attempt.attempt_id
            )
            attempt.freeze_txn_id = freeze.txn_id
            ledger.insert(c, "charging_attempts", **attempt.model_dump())
            if attempt.session_id:
                ledger.insert(
                    c,
                    "charging_sessions",
                    session_id=attempt.session_id,
                    attempt_id=attempt.attempt_id,
                    charger_id=charger_id,
                    connector_id=connector_id,
                    started_at=when,
                    status="charging",
                )
                self._reading(c, attempt.session_id, "start", start_wh, when, when)
            elif fault == "start_timeout":
                ledger.transaction(
                    c,
                    customer_id,
                    "unfreeze",
                    amount,
                    at=when,
                    attempt_id=attempt.attempt_id,
                    related_txn_id=freeze.txn_id,
                )
            return attempt

    @staticmethod
    def _reading(c, session, kind, energy, occurred, received):
        ledger.insert(
            c,
            "meter_readings",
            reading_id=new_id("MTR"),
            session_id=session,
            reading_type=kind,
            energy_wh=energy,
            occurred_at=occurred,
            received_at=received,
        )

    def _locked_session(self, c, session_id):
        query = """SELECT s.*, a.customer_id, a.freeze_txn_id, ch.policy_id FROM cb_charging_sessions s
            JOIN cb_charging_attempts a USING(attempt_id) JOIN cb_chargers ch
            ON ch.charger_id=s.charger_id AND ch.connector_id=s.connector_id WHERE s.session_id=:s"""
        session = ledger.one(c, query, s=session_id)
        ledger.lock_wallet(c, session["customer_id"])
        return ledger.one(c, query + " FOR UPDATE", s=session_id)

    def stop_charging(
        self,
        session_id: str,
        *,
        end_wh: int,
        charge_end_at: datetime,
        unplugged_at: datetime | None,
        fault=None,
    ) -> OrderView:
        _fault(
            fault,
            "missing_end_reading",
            "duplicate_deduct",
            "wrong_tariff_peak",
            "idle_fee_ignore_grace",
        )
        with self.engine.begin() as c:
            s = self._locked_session(c, session_id)
            end = as_utc(charge_end_at)
            unplug = as_utc(unplugged_at) if unplugged_at else None
            if s["status"] != "charging":
                raise BillingError("INVALID_STATE", "会话已结束")
            if end < s["started_at"] or (unplug and unplug < end):
                raise BillingError("VALIDATION", "结束时间或拔枪时间早于上一步")
            c.execute(
                text("""UPDATE cb_charging_sessions SET status='finished',charge_end_at=:end,
                unplugged_at=:unplug WHERE session_id=:s"""),
                dict(end=end, unplug=unplug, s=session_id),
            )
            order_id = new_id("ORD")
            ledger.insert(
                c,
                "orders",
                order_id=order_id,
                customer_id=s["customer_id"],
                session_id=session_id,
                attempt_id=s["attempt_id"],
                policy_id=s["policy_id"],
                status="pending_settlement",
                created_at=end,
            )
            if fault != "missing_end_reading":
                self._settle(c, s, order_id, end_wh, end, unplug, end, end, fault)
            return OrderView(
                **ledger.one(c, "SELECT * FROM cb_orders WHERE order_id=:o", o=order_id)
            )

    def _settle(self, c, s, order_id, end_wh, end, unplug, occurred, received, fault):
        policy = ledger.policy(c, s["policy_id"])
        if fault == "idle_fee_ignore_grace":
            policy.idle_grace_minutes = 0
        if fault == "wrong_tariff_peak":
            peak = next(p for p in policy.periods if p.period_type == "peak")
            for period in policy.periods:
                period.electricity_fen_per_kwh = peak.electricity_fen_per_kwh
                period.service_fen_per_kwh = peak.service_fen_per_kwh
        start = ledger.one(
            c,
            "SELECT energy_wh FROM cb_meter_readings WHERE session_id=:s AND reading_type='start'",
            s=s["session_id"],
        )
        if type(end_wh) is not int:
            raise BillingError("VALIDATION", "结束读数须为整数 Wh")
        bill = compute_order_amount(
            policy,
            s["started_at"],
            end,
            unplug,
            start["energy_wh"],
            end_wh,
            billing_conf["local_timezone"],
        )
        links = dict(attempt_id=s["attempt_id"], order_id=order_id)
        freeze = ledger.one(
            c,
            "SELECT * FROM cb_wallet_transactions WHERE txn_id=:t",
            t=s["freeze_txn_id"],
        )
        w = ledger.lock_wallet(c, s["customer_id"])
        count = 2 if fault == "duplicate_deduct" else 1
        if (
            bill.total_fen * count
            > w["balance_fen"] - w["frozen_fen"] + freeze["amount_fen"]
        ):
            raise BillingError("INSUFFICIENT_BALANCE", "可结算余额不足")
        self._reading(c, s["session_id"], "end", end_wh, occurred, received)
        c.execute(
            text("""UPDATE cb_orders SET energy_wh=:energy_wh,electricity_fee_fen=:electricity_fee_fen,
            service_fee_fen=:service_fee_fen,idle_fee_fen=:idle_fee_fen,total_fen=:total_fen,
            status='settled',settled_at=:settled WHERE order_id=:o"""),
            dict(**bill.model_dump(), settled=received, o=order_id),
        )
        if bill.total_fen:
            for i in range(count):
                ledger.transaction(
                    c,
                    s["customer_id"],
                    "deduct",
                    bill.total_fen,
                    at=received + timedelta(milliseconds=i),
                    **links,
                )
        ledger.transaction(
            c,
            s["customer_id"],
            "unfreeze",
            freeze["amount_fen"],
            at=received,
            related_txn_id=freeze["txn_id"],
            **links,
        )

    def backfill_end_reading(
        self, session_id: str, end_wh: int, occurred_at: datetime
    ) -> OrderView:
        with self.engine.begin() as c:
            s = self._locked_session(c, session_id)
            query = "SELECT * FROM cb_orders WHERE session_id=:s"
            order = ledger.one(c, query, s=session_id)
            if order["status"] != "pending_settlement":
                raise BillingError("INVALID_STATE", "订单已结算")
            s["policy_id"] = order["policy_id"]
            self._settle(
                c,
                s,
                order["order_id"],
                end_wh,
                s["charge_end_at"],
                s["unplugged_at"],
                as_utc(occurred_at),
                utc_now(),
                None,
            )
            return OrderView(**ledger.one(c, query, s=session_id))

    def create_ops_refund(
        self, deduct_txn_id: str, amount_fen: int, status: str
    ) -> TxnView:
        with self.engine.begin() as c:
            t = ledger.one(
                c,
                "SELECT * FROM cb_wallet_transactions WHERE txn_id=:t",
                t=deduct_txn_id,
            )
            ledger.lock_wallet(c, t["customer_id"])
            if (
                t["txn_type"] != "deduct"
                or t["status"] != "succeeded"
                or status not in ("pending", "succeeded")
            ):
                raise BillingError("VALIDATION", "退款目标或状态无效")
            if amount_fen > ledger.refundable(c, t):
                raise BillingError("VALIDATION", "超过可退金额")
            return ledger.transaction(
                c,
                t["customer_id"],
                "refund",
                amount_fen,
                status=status,
                source="ops_manual",
                related_txn_id=deduct_txn_id,
                order_id=t["order_id"],
                attempt_id=t["attempt_id"],
            )

    def list_transactions(self, customer_id: str) -> list[TxnView]:
        with self.engine.begin() as c:
            return [
                TxnView(**r)
                for r in ledger.rows(
                    c,
                    "SELECT * FROM cb_wallet_transactions WHERE customer_id=:c ORDER BY created_at,txn_id",
                    c=customer_id,
                )
            ]

    def list_orders(self, customer_id: str) -> list[OrderView]:
        with self.engine.begin() as c:
            return [
                OrderView(**r)
                for r in ledger.rows(
                    c,
                    "SELECT * FROM cb_orders WHERE customer_id=:c ORDER BY created_at,order_id",
                    c=customer_id,
                )
            ]

    def list_attempts(self, customer_id: str) -> list[AttemptView]:
        with self.engine.begin() as c:
            return [
                AttemptView(**r)
                for r in ledger.rows(
                    c,
                    "SELECT * FROM cb_charging_attempts WHERE customer_id=:c ORDER BY requested_at,attempt_id",
                    c=customer_id,
                )
            ]
