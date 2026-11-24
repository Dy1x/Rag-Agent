import re
from datetime import datetime

from agent.cases.agent_spec import AgentSpec
from service.billing.ids import is_record_suffix
from service.billing.models import CaseFile
from service.billing.timeutil import to_local
from utils.config_handler import billing_conf
from .tools import locator_args


SPEC = AgentSpec("intake", "billing_intake.txt", ["find_candidate_orders"], CaseFile, 2)


def dispute_type(question):
    for kind, words in [
        (
            "refund_status",
            (
                "退了吗",
                "退款进度",
                "退到",
                "退款在途",
                "退款状态",
                "退款还没",
                "退款没到账",
                "退款未到账",
            ),
        ),
        (
            "recharge_missing",
            ("充值", "充了 100", "没到账", "充钱", "没入账", "未入账"),
        ),
        (
            "failed_start_charged",
            (
                "没启动",
                "未启动",
                "启动失败",
                "启动超时",
                "扫了两次",
                "扫码两次",
                "刷了两次",
                "没充上",
                "未充上",
                "充电未开始",
            ),
        ),
        (
            "double_charge",
            ("两次钱", "重复扣", "扣了两次", "重复收费", "扣款两次", "重复计费"),
        ),
        ("idle_fee", ("占位", "拔枪", "超时停车费")),
        (
            "wrong_tariff",
            ("高峰", "峰价", "平段", "峰时", "谷时", "费率不对", "计价时段"),
        ),
        ("frozen_balance", ("冻结", "没账单", "未出账单", "余额被锁", "余额锁定")),
    ]:
        if any(word in question for word in words):
            return kind
    return "other"


def offline_intake(payload, tools):
    kind = dispute_type(payload["question"])
    locator = locator_args(payload["locator"])
    if match := re.search(r"ORD-[A-Za-z0-9]+", payload["question"]):
        locator.setdefault("order_id", match.group())
    if kind == "recharge_missing" and not locator:
        return CaseFile(status="located", dispute_type=kind)
    return resolve_candidates(
        kind,
        tools["find_candidate_orders"](**locator),
        order_id=locator.get("order_id"),
    )


def resolve_candidates(kind, candidates, *, order_id=None) -> CaseFile:
    if not candidates and order_id and not is_record_suffix(order_id):
        return CaseFile(status="forbidden", dispute_type=kind)
    if kind == "failed_start_charged" and not order_id:
        failed = [r for r in candidates if r["result"] != "started"]
        if len(failed) == 1:
            candidates = failed
    if len(candidates) == 1:
        row = candidates[0]
        return CaseFile(
            status="located",
            dispute_type=kind,
            order_ids=[row["order_id"]] if row["order_id"] else [],
            attempt_ids=[row["attempt_id"]],
        )
    return clarification(kind, candidates)


def clarification(kind, candidates) -> CaseFile:
    masked = [
        {
            "订单尾号": (r["order_id"] or r["attempt_id"])[-4:],
            "桩号": r["charger_id"],
            "本地时间": to_local(
                datetime.fromisoformat(r["requested_at"]),
                billing_conf["local_timezone"],
            ).isoformat(),
        }
        for r in candidates
    ]
    return CaseFile(
        status="clarify",
        dispute_type=kind,
        candidates=masked,
        clarification_question="请提供完整订单号，或确认充电桩和启动时间。",
    )
