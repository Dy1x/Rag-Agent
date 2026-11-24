from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import BaseModel


@dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt_file: str
    tools: list[str]
    output_schema: type[BaseModel]
    max_tool_calls: int


def build_agent_node(
    spec: AgentSpec,
    planner: str,
    tool_registry: dict,
    offline_impl: Callable,
    *,
    record=None,
    dialogue=None,
) -> Callable:
    allowed = {name: tool_registry[name] for name in spec.tools}
    if planner == "deepseek":
        return _deepseek(spec, allowed, record, dialogue)
    if planner != "offline":
        raise ValueError("未知 planner")

    def invoke(payload):
        return spec.output_schema.model_validate(offline_impl(payload, allowed))

    return invoke


def _deepseek(spec, allowed, record, dialogue=None):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain.agents.structured_output import (
        ToolStrategy,
        StructuredOutputValidationError,
    )
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tools import StructuredTool
    from model.factory import get_chat_model
    from utils.path_tool import get_abs_path
    from utils.config_handler import rag_conf

    if rag_conf.get("chat_model_provider") != "deepseek":
        raise ValueError("DeepSeek 模式需要配置 deepseek 模型提供方")
    from service.billing.models import Proposal, SoftReviewOpinion
    from service.billing.errors import BillingError

    formats = (
        [spec.output_schema] if dialogue is None else [Proposal, SoftReviewOpinion]
    )
    permitted = set(allowed) | {fmt.__name__ for fmt in formats}

    class Usage(BaseCallbackHandler):
        def on_chat_model_start(self, *args, **kwargs):
            if record:
                record("model_call", {})

        def on_llm_end(self, response, **kwargs):
            if record:
                for generations in response.generations:
                    for generation in generations:
                        for call in getattr(
                            getattr(generation, "message", None), "tool_calls", []
                        ):
                            if call["name"] not in permitted:
                                record("unauthorized_tool", {"name": call["name"]})
                        usage = getattr(
                            getattr(generation, "message", None), "usage_metadata", None
                        )
                        if usage:
                            record("model_usage", dict(usage))

    from functools import wraps
    from threading import Lock

    model = get_chat_model()
    prompt = Path(get_abs_path("prompts/" + spec.prompt_file)).read_text(
        encoding="utf-8"
    )

    def invoke(payload):
        remaining = spec.max_tool_calls
        # Include structured output and its correction in the model's tool-call limit.
        run_limit = spec.max_tool_calls + 2
        lock = Lock()

        def bounded(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                nonlocal remaining
                with lock:
                    if remaining == 0:
                        return {"error": "只读工具预算已用完，请提交结构化结果"}
                    remaining -= 1
                try:
                    return fn(*args, **kwargs)
                except BillingError as error:
                    if record and error.code == "FORBIDDEN":
                        record("unauthorized_tool", {"name": fn.__name__})
                    return {"error": error.code, "message": error.message}

            return call

        tools = [StructuredTool.from_function(bounded(f)) for f in allowed.values()]
        messages = [
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }
        ]
        if dialogue is not None:
            messages = dialogue.get("messages", []) + messages
        # Keep one conversation for design A, but apply the active role's tool budget.
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=prompt
            if dialogue is None
            else prompt
            + " 同一会话内也负责自查。收到自查指令时对已提交方案给出 SoftReviewOpinion。",
            middleware=[
                ToolCallLimitMiddleware(run_limit=run_limit, exit_behavior="continue")
            ],
            response_format=ToolStrategy(
                spec.output_schema
                if dialogue is None
                else Proposal | SoftReviewOpinion,
                handle_errors=False,
            ),
        )
        for attempt in range(2):
            try:
                result = agent.invoke(
                    {"messages": messages},
                    config={
                        "callbacks": [Usage()],
                        # Each sequential call traverses model, middleware and tool nodes.
                        "recursion_limit": 3 * run_limit + 3,
                    },
                )
                output = spec.output_schema.model_validate(
                    result.get("structured_response")
                )
                if dialogue is not None:
                    dialogue["messages"] = result["messages"]
                return output
            except (StructuredOutputValidationError, ValueError) as e:
                if attempt:
                    raise ValueError(
                        "两次结构化输出未通过 " + spec.output_schema.__name__ + " 校验"
                    ) from e
                messages.append(
                    {
                        "role": "user",
                        "content": "上次输出不符合结构，请使用指定结构化工具重新提交。",
                    }
                )

    return invoke
