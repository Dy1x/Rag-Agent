from datetime import datetime

from service.billing.timeutil import local_to_utc
from utils.config_handler import billing_conf


def locator_args(locator: dict) -> dict:
    args = {
        k: v
        for k, v in locator.items()
        if k in ("order_id", "charger_id", "connector_id") and v
    }
    for edge in ("start", "end"):
        for suffix in ("local", "utc"):
            key = f"{edge}_{suffix}"
            if value := locator.get(key):
                at = datetime.fromisoformat(value) if isinstance(value, str) else value
                args[f"{edge}_utc"] = local_to_utc(
                    at, billing_conf["local_timezone"] if suffix == "local" else "UTC"
                )
    return args


def read_tools(service, identity, trace, planner):
    def query(name, **args):
        trace("tool_call", {"name": name, "arguments": args})
        return getattr(service, name)(identity, **args)

    def find_candidate_orders(
        order_id: str | None = None,
        charger_id: str | None = None,
        connector_id: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
    ):
        """查找当前车主的订单和启动候选。"""
        return [
            r.model_dump(mode="json")
            for r in query("find_candidate_orders", **locator_args(locals()))
        ]

    def get_policy(policy_id: str):
        """读取指定政策及全部计费时段。"""
        return query("get_policy", policy_id=policy_id).model_dump(mode="json")

    def get_order(order_id: str):
        """回查当前车主的订单快照。"""
        return query("get_order", order_id=order_id).model_dump(mode="json")

    def get_session(session_id: str):
        """读取会话及电表读数。"""
        return query("get_session", session_id=session_id).model_dump(mode="json")

    def recompute_order(order_id: str):
        """用政策和读数重新计算订单金额。"""
        return query("recompute_order", order_id=order_id).model_dump(mode="json")

    def compute_refundable(deduct_txn_id: str):
        """读取一笔扣款当前剩余可退分数。"""
        return query("compute_refundable", deduct_txn_id=deduct_txn_id).model_dump()

    def get_transactions(
        order_id: str | None = None,
        attempt_id: str | None = None,
        txn_ids: list[str] | None = None,
    ):
        """读取指定订单、启动尝试或流水号的记录。"""
        return [
            r.model_dump(mode="json")
            for r in query(
                "get_transactions",
                order_id=order_id,
                attempt_id=attempt_id,
                txn_ids=txn_ids,
            )
        ]

    def search_billing_faq(question: str):
        """查询一般充电规则及钱包常见问题。"""
        trace("tool_call", {"name": "search_billing_faq"})
        if planner == "offline":
            return {
                "simulated": True,
                "answer": "充电费用按订单适用政策计算，退款返回钱包。",
            }
        try:
            from rag.rag_service import RagSummerizeService

            result = RagSummerizeService().answer_question(question)
        except Exception as error:
            trace("faq_unavailable", {"reason": type(error).__name__})
            return {
                "status": "insufficient",
                "text": "计费知识库暂不可用，请依据本案账本与适用政策核查。",
                "evidence": [],
            }
        return result.model_dump(mode="json")

    return {
        f.__name__: f
        for f in [
            find_candidate_orders,
            get_policy,
            get_order,
            get_session,
            recompute_order,
            compute_refundable,
            get_transactions,
            search_billing_faq,
        ]
    }
