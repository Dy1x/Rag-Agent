from datetime import datetime, timedelta
from functools import partial

from utils.config_handler import billing_conf
from .models import ScenarioHandle
from .simulator_service import SimulatorService
from .timeutil import local_to_utc


QUESTIONS = {
    "S1": "扫了两次，第一次没启动，余额怎么少了一大块？",
    "S2": "一次充电扣了我两次钱",
    "S3": "我充了 100 元怎么没到账？",
    "S4": "早上八点半充的，怎么按高峰价算？",
    "S5": "重复扣的钱退了吗？",
    "S6": "充完电半天了怎么还冻结着钱，也没账单？",
    "S7": "帮我查一下订单",
    "S8": "充得太慢了，全部退给我",
    "S9": "为什么收了我 10 块占位费？",
    "S10": "我十分钟就拔枪了，为什么有占位费？",
    "S11": "今天在 CP-A01 充电的费用不对",
}


def _build(name: str, sim: SimulatorService) -> ScenarioHandle:
    customer = sim.create_customer("场景 " + name).customer_id
    start = local_to_utc(datetime(2026, 11, 20, 8, 30), billing_conf["local_timezone"])
    sim.recharge(customer, 20000, at=start - timedelta(hours=1))
    refs = {}
    locator = {}
    if name == "S3":
        pending = sim.recharge(customer, 10000, fault="callback_pending", at=start)
        refs["recharge"] = pending.txn_id
    else:
        if name == "S1":
            failed = sim.start_charging(
                customer,
                "CP-A01",
                "1",
                at=start - timedelta(minutes=10),
                fault="timeout_keep_frozen",
            )
            refs.update(
                first_attempt=failed.attempt_id, first_freeze=failed.freeze_txn_id
            )
        attempt = sim.start_charging(customer, "CP-A01", "1", at=start)
        end = start + timedelta(hours=1)
        fault = {
            "S2": "duplicate_deduct",
            "S4": "wrong_tariff_peak",
            "S5": "duplicate_deduct",
            "S6": "missing_end_reading",
            "S10": "idle_fee_ignore_grace",
        }.get(name)
        order = sim.stop_charging(
            attempt.session_id,
            end_wh=4000,
            charge_end_at=end,
            unplugged_at=end + timedelta(minutes=35 if name == "S9" else 10),
            fault=fault,
        )
        refs.update(
            order=order.order_id,
            attempt=attempt.attempt_id,
            session=attempt.session_id,
            freeze=attempt.freeze_txn_id,
        )
        locator["order_id"] = order.order_id
        deducts = [t for t in sim.list_transactions(customer) if t.txn_type == "deduct"]
        if deducts:
            refs["first_deduct"] = deducts[0].txn_id
        if len(deducts) > 1:
            refs["second_deduct"] = deducts[1].txn_id
        if name == "S5":
            refs["pending_refund"] = sim.create_ops_refund(
                refs["second_deduct"], 560, "pending"
            ).txn_id
        if name == "S11":
            later = start + timedelta(hours=5, minutes=30)
            a = sim.start_charging(customer, "CP-A01", "1", at=later)
            refs["second_order"] = sim.stop_charging(
                a.session_id,
                end_wh=4000,
                charge_end_at=later + timedelta(hours=1),
                unplugged_at=later + timedelta(hours=1),
            ).order_id
        if name in ("S1", "S11"):
            locator = dict(
                charger_id="CP-A01",
                start_local="2026-11-20T00:00:00",
                end_local="2026-11-21T00:00:00",
            )
    question = QUESTIONS[name]
    if name == "S7":
        customer = "CUS-DEMO-BOB"
        question += " " + refs["order"]
    return ScenarioHandle(
        name=name,
        customer_id=customer,
        refs=refs,
        suggested_question=question,
        suggested_locator=locator,
    )


SCENARIOS = {name: partial(_build, name) for name in QUESTIONS}


def build_scenario(sim: SimulatorService, name: str) -> ScenarioHandle:
    return SCENARIOS[name](sim)
