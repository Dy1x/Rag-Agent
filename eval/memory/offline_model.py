"""Deterministic intent routing for testing memory availability without a model API."""

import re
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from service.memory.compactor import message_text, tool_payload


class ReplayModel(BaseChatModel):
    operation: dict = {}

    @property
    def _llm_type(self):
        return "memory-offline-replay"

    def bind_tools(self, tools, **kwargs):
        self._tool_names = {tool.name for tool in tools}
        return self

    def _tool(self, name, **args):
        if name not in self._tool_names:
            return AIMessage(content="不知道，当前没有可用的长期记忆。")
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": uuid4().hex},
        ])

    def _respond(self, messages):
        last_human = max(i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage))
        history = "\n".join(message_text(msg) for msg in messages[:last_human])
        results = [msg for msg in messages[last_human + 1:] if isinstance(msg, ToolMessage)]
        kind = self.operation.get("kind")
        args = self.operation.get("args", {})
        if kind == "write":
            slots = list(args.items())
            if len(results) < len(slots):
                key, value = slots[len(results)]
                return self._tool("save_customer_memory", memory_key=key, memory_value=value)
            status = tool_payload(results[-1].content).get("status")
            return AIMessage(content="已保存。" if status == "ok" else "不能保存这项记忆。")
        if kind == "profile":
            if not results:
                return self._tool("get_customer_memory")
            records = tool_payload(results[-1].content).get("memories", [])
            by_key = {m["memory_key"]: m["memory_value"] for m in records}
            values = [by_key[key] for key in self.operation["keys"] if key in by_key]
            return AIMessage(content="、".join(values) or "不知道，当前没有这项长期记忆。")
        if kind == "case_status":
            if results:
                data = tool_payload(results[-1].content)
                return AIMessage(content=f"{data.get('case_id', '')} {data.get('status', '')}")
            match = re.search(r"case:(CASE-[0-9A-F]{12})", history)
            if match:
                return self._tool("get_dispute_case_status", case_id=match[1])
            return AIMessage(content="不知道案件编号，请提供。")
        if kind in ("early_case", "early_charger", "early_direction"):
            pattern = {
                "early_case": r"CASE-[0-9A-F]{12}",
                "early_charger": r"CP-[A-Z0-9]+(?:-[A-Z0-9]+)*",
                "early_direction": r"[A-Z] 站网关",
            }[kind]
            match = re.search(pattern, history)
            return AIMessage(content=match[0] if match else "不知道。")
        if kind in ("billing", "outage", "rag", "forget"):
            name = {
                "billing": "handle_billing_dispute", "outage": "investigate_outage",
                "rag": "rag_summerize", "forget": "delete_customer_memory",
            }[kind]
            if not results:
                return self._tool(name, **args)
            if kind == "billing":
                return AIMessage(content=tool_payload(results[-1].content).get("customer_message") or "已核查。")
            return AIMessage(content="已处理。")
        return AIMessage(content="收到。")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])
