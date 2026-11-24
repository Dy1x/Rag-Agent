import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.react_agent import ReactAgent
from agent.tools.investigation_tools import investigate_outage
from eval.investigation.run_multi_agent import prepare_fixture
from service.investigation_service import run_investigation


SCOPE = {
    "site_ids": ["A", "B"],
    "start_utc": "2026-11-21T14:00Z",
    "end_utc": "2026-11-22T00:00Z",
    "question": "A、B 站掉线影响多少设备，优先查哪里？",
}


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def request_tool(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "chat-tool-1"}])


class InvestigationChatTest(unittest.TestCase):
    def run_chat(self, query, *responses):
        saver = InMemorySaver()
        with (
            patch("agent.react_agent.get_chat_model", return_value=ToolCallingModel(responses=list(responses))),
            patch("agent.react_agent.get_chat_checkpointer", return_value=saver),
        ):
            agent = ReactAgent("investigation-chat-test")
            messages = list(agent.execute_stream(query))
        checkpoint = saver.get(agent.get_checkpoint_config())
        tools = [item for item in checkpoint["channel_values"]["messages"] if isinstance(item, ToolMessage)]
        return messages, tools

    def test_faq_uses_rag_tool(self):
        with (
            patch("agent.tools.agent_tools.rag") as rag,
            patch("agent.tools.investigation_tools.run_investigation") as investigate,
        ):
            rag.rag_summerize.return_value = "手册片段【E1】"
            question = "充电桩有哪些通用的离线排查步骤？"
            messages, tools = self.run_chat(
                question, request_tool("rag_summerize", {"query": question}),
                AIMessage(content="请按手册步骤排查【E1】。"),
            )
        self.assertEqual(messages, ["请按手册步骤排查【E1】。"])
        self.assertEqual([(item.name, item.content) for item in tools], [("rag_summerize", "手册片段【E1】")])
        self.assertEqual(rag.rag_summerize.call_args.args, (question,))
        investigate.assert_not_called()

    def test_missing_time_and_site_can_ask_without_a_tool_call(self):
        reply = "请提供站点 ID、调查开始和结束时间以及时区。"
        with patch("agent.tools.investigation_tools.run_investigation") as investigate:
            messages, tools = self.run_chat("昨晚有设备掉线，查一下。", AIMessage(content=reply))
        self.assertEqual((messages, tools), ([reply], []))
        investigate.assert_not_called()

    def test_complete_scope_runs_the_real_graph_through_the_registered_tool(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "chat.sqlite3"
            prepare_fixture("rollout_site_overlap", db)
            config = {
                "INVESTIGATION_DB_PATH": str(db),
                "INVESTIGATION_FIRMWARE_SOURCE": "available",
                "INVESTIGATION_PLANNER": "offline",
                "INVESTIGATION_SIMULATED_DATA": "1",
            }
            service_results = []

            def capture(*args, **kwargs):
                result = run_investigation(*args, **kwargs)
                service_results.append(result)
                return result

            with (
                patch.dict(os.environ, config),
                patch("agent.tools.investigation_tools.run_investigation", side_effect=capture) as investigate,
            ):
                _, tools = self.run_chat(
                    "调查 A、B 站 UTC 2026-11-21T14:00Z 至 2026-11-22T00:00Z 的掉线。",
                    request_tool("investigate_outage", SCOPE), AIMessage(content="调查已返回。"),
                )
        investigate.assert_called_once()
        self.assertEqual([item.name for item in tools], ["investigate_outage"])
        result = json.loads(tools[0].content)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["simulated_data"])
        self.assertEqual(result["run_id"], service_results[0]["run_id"])
        self.assertEqual(result["review_request"], service_results[0]["review_request"])
        self.assertIn("gw-A-01", result["review_request"]["trigger_evidence_ids"])
        report = result["report"]
        self.assertEqual((report["impact"]["offline_device_count"], report["impact"]["device_count"]), (15, 30))
        self.assertEqual((report["priority"]["direction"], report["priority"]["site_id"]), ("site", "A"))
        self.assertEqual([row["offline_rate_difference"] for row in report["firmware_by_site"]["sites"]], [0, 0])
        self.assertEqual(set(investigate_outage.tool_call_schema.model_json_schema()["properties"]), set(SCOPE))

    def test_invalid_scope_and_unconfigured_data_have_explicit_statuses(self):
        with patch.dict(os.environ, {"INVESTIGATION_DB_PATH": "", "INVESTIGATION_FIRMWARE_SOURCE": ""}):
            self.assertEqual(investigate_outage.invoke({**SCOPE, "site_ids": []})["status"], "invalid_scope")
            self.assertEqual(investigate_outage.invoke({**SCOPE, "end_utc": "昨晚"})["status"], "invalid_scope")
            self.assertEqual(investigate_outage.invoke(SCOPE)["status"], "data_source_unconfigured")
