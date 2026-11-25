import unittest
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.tools.middleware import compact_and_summarize_history, inject_memory_prompt
from service.memory.summary_store import SummaryStore
from test_investigation_chat import ToolCallingModel


class RecordingModel(ToolCallingModel):
    prompts: list = []

    def _generate(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


class LayeredMiddlewareTests(unittest.TestCase):
    def build(self, turns):
        model = RecordingModel(responses=[AIMessage(content="收到") for _ in range(turns)])
        saver = InMemorySaver()
        graph = create_agent(
            model, middleware=[compact_and_summarize_history, inject_memory_prompt],
            checkpointer=saver, context_schema=dict,
        )
        return graph, model

    def test_twelve_turns_store_summary_and_inject_into_system_prompt(self):
        graph, model = self.build(12)
        store = SummaryStore()
        config = {"configurable": {"thread_id": "thread"}}
        with patch("agent.tools.middleware.get_summary_store", return_value=store):
            for index in range(12):
                graph.invoke({"messages": [HumanMessage(content="CP-A01" if index == 0 else f"第 {index} 轮")]},
                             config, context={"session_id": "thread"})
        summary = store.get("thread")
        self.assertEqual(summary.evicted_rounds, 6)
        self.assertEqual(summary.mentioned_entities["chargers"], ["CP-A01"])
        final = model.prompts[-1]
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in final), 6)
        prompt = next(m.content for m in final if isinstance(m, SystemMessage))
        self.assertIn("[L1 早期对话摘要", prompt)
        self.assertIn("CP-A01", prompt)
        self.assertIsNone(store.get("different-thread"))

    def test_character_threshold_retains_two_whole_rounds(self):
        graph, model = self.build(1)
        store = SummaryStore()
        messages = []
        for index in range(4):
            messages.extend([HumanMessage(content=f"{index}" * 150), AIMessage(content="旧回复")])
        with (
            patch("agent.tools.middleware.get_summary_store", return_value=store),
            patch.dict("agent.tools.middleware.memory_conf", {"max_context_chars": 350}),
        ):
            graph.invoke({"messages": messages},
                         {"configurable": {"thread_id": "chars"}}, context={"session_id": "chars"})
        self.assertEqual(store.get("chars").evicted_rounds, 2)
        self.assertEqual(sum(isinstance(m, HumanMessage) for m in model.prompts[-1]), 2)
