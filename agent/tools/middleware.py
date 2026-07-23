from typing import Callable,Any

from langchain.agents.middleware import wrap_tool_call,dynamic_prompt,ModelRequest,before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage,HumanMessage,RemoveMessage,SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts


# Agent 最多保留最近 10 个用户对话轮次
MAX_CONVERSATION_ROUNDS = 10

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
def log_before_model(
    state: AgentState,
    runtime: Runtime,
) -> None:
    """
    在调用模型之前输出消息数量和最后一条消息。
    """

    messages = state.get("messages", [])

    logger.info(
        f"[log_before_model]即将调用模型，带有{len(messages)}条消息。"
    )

    if messages:
        latest_message = messages[-1]

        logger.debug(
            f"[log_before_model] "
            f"{type(latest_message).__name__} | "
            f"{latest_message.content}"
        )

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    """
    根据当前上下文切换普通客服提示词和报告生成提示词。
    """

    is_report = request.runtime.context.get("report", False)

    if is_report:
        return load_report_prompts()

    return load_system_prompts()