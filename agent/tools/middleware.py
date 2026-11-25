from hashlib import sha256
from typing import Any, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest, before_agent, before_model, dynamic_prompt, wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy.exc import SQLAlchemyError

from rag.evidence import RetrievalBudget
from service.billing.db import get_billing_engine
from service.billing.models import Identity
from service.memory.compactor import compact_old_tools, context_chars
from service.memory.customer_memory_service import CustomerMemoryService
from service.memory.summarizer import update_summary
from service.memory.summary_store import get_summary_store
from utils.config_handler import memory_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


# Agent 最多保留最近 10 个用户对话轮次
MAX_CONVERSATION_ROUNDS = 10


@before_agent
def initialize_rag_budget(state: AgentState, runtime: Runtime) -> None:
    # 工具并行共享运行上下文。预算对象无需写进消息状态或 Redis。
    if runtime.context is not None:
        runtime.context["rag_budget"] = RetrievalBudget()


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    监控 Agent 的工具调用。
    """

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    logger.info(f"[tool monitor]执行工具：{tool_name}")
    if tool_name != "save_customer_memory":
        logger.info(f"[tool monitor]传入参数：{tool_args}")

    try:
        result = handler(request)

        logger.info(f"[tool monitor]工具{tool_name}调用成功")

        if tool_name == "fill_context_for_report":
            request.tool_call["report"] = True

        return result

    except Exception as e:
        logger.error(
            f"[tool monitor]工具{tool_name}调用失败，原因：{str(e)}"
        )
        raise

@before_model
def trim_conversation_history(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """
    在每次调用大模型前裁剪聊天历史。

    规则：
    1. 以 HumanMessage 数量作为对话轮数。
    2. 最多保留最近 10 个用户轮次。
    3. 保留这 10 轮之间产生的 AIMessage 和 ToolMessage。
    4. 保留可能存在的 SystemMessage。
    5. 删除更早的消息，防止上下文无限增长。
    """

    messages = list(state.get("messages", []))

    if not messages:
        return None

    # 找出所有用户消息在消息列表中的位置
    human_message_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]

    # 用户消息没有超过 10 轮，不需要裁剪
    if len(human_message_indexes) <= MAX_CONVERSATION_ROUNDS:
        return None

    # 找到倒数第 10 个用户消息的位置
    start_index = human_message_indexes[-MAX_CONVERSATION_ROUNDS]

    # 如果状态中存在 SystemMessage，需要继续保留
    system_messages = [
        message
        for message in messages[:start_index]
        if isinstance(message, SystemMessage)
    ]

    # 保留最近 10 轮，以及这些轮次中的 AI/工具消息
    recent_messages = messages[start_index:]

    kept_messages = [
        *system_messages,
        *recent_messages,
    ]

    removed_count = len(messages) - len(kept_messages)

    logger.info(
        "[conversation memory]裁剪聊天记录："
        f"原消息数={len(messages)}，"
        f"删除消息数={removed_count}，"
        f"保留消息数={len(kept_messages)}，"
        f"保留用户轮数={MAX_CONVERSATION_ROUNDS}"
    )

    # 先删除原消息，再写入裁剪后的消息
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *kept_messages,
        ]
    }


@before_model
def compact_and_summarize_history(state: AgentState, runtime: Runtime) -> dict | None:
    original = state["messages"]
    messages = compact_old_tools(
        original, keep_rounds=memory_conf["compact_keep_tool_rounds"],
        threshold=memory_conf["tool_compact_chars"],
    )
    starts = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    keep = len(starts)
    if keep > memory_conf["summary_trigger_rounds"]:
        keep = min(keep - 1, max(1, memory_conf["recent_rounds"]))
    limit = memory_conf["max_context_chars"]
    if len(starts) > 2 and context_chars(messages) > limit:
        keep = min(keep, max(2, memory_conf["recent_rounds"]))
        while keep > 2 and context_chars(messages[starts[-keep]:]) > limit:
            keep -= 1

    context = runtime.context or {}
    thread_id = context.get("session_id")
    if thread_id and 0 < keep < len(starts):
        split = starts[-keep]
        evicted = [msg for msg in messages[:split] if not isinstance(msg, SystemMessage)]
        fingerprint = sha256("|".join(str(msg.id) for msg in evicted).encode()).hexdigest()
        summary = get_summary_store().advance(
            thread_id, fingerprint,
            lambda previous: update_summary(previous, evicted, mode=memory_conf["summarizer_mode"]),
        )
        context["conversation_summary"] = summary
        system = [msg for msg in messages[:split] if isinstance(msg, SystemMessage)]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *system, *messages[split:]]}

    updates = [after for before, after in zip(original, messages) if before is not after]
    return {"messages": updates} if updates else None


@dynamic_prompt
def inject_memory_prompt(request: ModelRequest) -> str:
    context = request.runtime.context or {}
    parts = [load_system_prompts()]
    if thread_id := context.get("session_id"):
        if "conversation_summary" not in context:
            context["conversation_summary"] = get_summary_store().get(thread_id)
        if summary := context["conversation_summary"]:
            parts.extend([
                f"[L1 早期对话摘要（已压缩前 {summary.evicted_rounds} 轮，仅作上下文参考，"
                "涉及金额与订单状态请以实时工具结果为准）]",
                summary.render(memory_conf["max_summary_chars"]),
            ])
    if memory_conf["enable_long_term_memory"] and (customer := context.get("customer_id")):
        try:
            memories = CustomerMemoryService(get_billing_engine()).list_active(
                Identity(role="customer", actor_id=customer),
                limit=memory_conf["max_injected_memories"],
            )
        except SQLAlchemyError:
            memories = []
        if memories:
            parts.append(
                "[L2 当前车主长期记忆（仅用于个性化接待与辅助定位，"
                "绝不可替代实时账本或作为退款依据）]"
            )
            parts.extend(
                f"- [{m.category}] {m.memory_key} {m.memory_value}（更新于 {m.updated_at.date()}）"
                for m in memories
            )
    return "\n\n".join(parts)


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
) -> None:
    """
    在调用模型之前输出消息数量和最后一条消息。
    """

    messages = state.get("messages", [])

    logger.info(
        f"[log_before_model]即将调用模型，带有{len(messages)}条消息，{context_chars(messages)}字符。"
    )

    if messages:
        latest_message = messages[-1]

        logger.debug(f"[log_before_model] {type(latest_message).__name__}")

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    """
    根据当前上下文切换普通客服提示词和报告生成提示词。
    """

    is_report = (request.runtime.context or {}).get("report", False)

    if is_report:
        return load_report_prompts()

    return load_system_prompts()
