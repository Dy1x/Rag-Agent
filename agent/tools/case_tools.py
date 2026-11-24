from langchain.tools import ToolRuntime, tool

from agent.cases.registry import additional_handlers, get_handler
from service.billing.errors import BillingError
from service.billing.models import Identity
from utils.config_handler import billing_conf


def _context(runtime):
    return runtime.context or {} if runtime else {}


def _start(handler, question, locator, runtime):
    context = _context(runtime)
    if not context.get("customer_id"):
        return {"status": "need_login"}
    try:
        result = handler.start(
            Identity(role="customer", actor_id=context["customer_id"]),
            question,
            locator,
            planner=billing_conf["planner"],
            chat_session_id=context.get("session_id"),
        )
    except BillingError as error:
        return {"status": error.code.lower(), "customer_message": error.message}
    reply = result.model_dump(
        include={"case_id", "status", "customer_message", "clarification_question"}
    )
    if result.status == "clarify":
        reply["candidates"] = (result.report or {}).get("candidates", [])
    return reply


@tool
def handle_billing_dispute(
    question: str,
    order_id: str | None = None,
    charger_id: str | None = None,
    connector_id: str | None = None,
    start_local: str | None = None,
    end_local: str | None = None,
    runtime: ToolRuntime[dict | None] = None,
) -> dict:
    """核查当前车主的充电扣费、冻结、退款、充值到账争议。

    question 保留车主原话中的关键表述。order_id 仅填完整订单号（ORD-…）。
    车主选定脱敏候选后，传候选的 charger_id 以及候选时间前后各一分钟的
    start_local 和 end_local（本地 ISO 格式）。候选尾号不可填入 order_id。
    """
    if not _context(runtime).get("customer_id"):
        return {"status": "need_login"}
    locator = {
        k: v for k, v in locals().items() if k not in ("question", "runtime") and v
    }
    from agent.cases.billing.tools import locator_args

    try:
        locator_args(locator)
    except ValueError:
        return {
            "status": "clarify",
            "clarification_question": "请提供 ISO 格式的本地日期时间。",
        }
    return _start(get_handler("billing_dispute"), question, locator, runtime)


@tool
def get_dispute_case_status(
    case_id: str, runtime: ToolRuntime[dict | None] = None
) -> dict:
    """查询当前车主自己的费用争议案件状态与处理结果。"""
    context = _context(runtime)
    if not context.get("customer_id"):
        return {"status": "need_login"}
    try:
        handler = get_handler("billing_dispute")
        case = handler.store.get_case_detail(
            Identity(role="customer", actor_id=context["customer_id"]), case_id
        )
    except BillingError:
        return {"status": "forbidden", "customer_message": "不可访问"}
    latest = next(r for r in case["runs"] if r["run_id"] == case["latest_run_id"])
    return {
        "case_id": case_id,
        "status": case["status"],
        "customer_message": (latest["report"] or {}).get("customer_message")
        or (
            "已提交人工审批。"
            if case["status"] == "action_pending_approval"
            else "案件正在处理中。"
        ),
    }


def case_tools():
    def make(handler):
        @tool(handler.name, description=handler.description)
        def start_case(
            question: str, locator: dict, runtime: ToolRuntime[dict | None] = None
        ) -> dict:
            return _start(handler, question, locator, runtime)

        return start_case

    return [
        handle_billing_dispute,
        get_dispute_case_status,
        *[make(h) for h in additional_handlers()],
    ]
