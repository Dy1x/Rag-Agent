import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from service.memory.compactor import MARKER, compact_old_tools, tool_payload


class CompactorTests(unittest.TestCase):
    def test_message_pair_preserved_and_recent_results_untouched(self):
        request = AIMessage(content="", id="a1", tool_calls=[
            {"name": "handle_billing_dispute", "id": "call1", "args": {}},
        ])
        old = ToolMessage(
            id="t1", tool_call_id="call1", name="handle_billing_dispute",
            content=json.dumps({
                "case_id": "CASE-1234567890AB", "status": "clarify",
                "customer_message": "请补充信息" * 200,
                "candidates": [{"订单尾号": "AB01", "private": "ORD-SECRET"}],
            }, ensure_ascii=False),
        )
        latest = ToolMessage(id="t2", tool_call_id="call2", content="X" * 1000)
        messages = [HumanMessage(content="旧"), request, old, HumanMessage(content="新"), latest]
        result = compact_old_tools(messages, keep_rounds=1, threshold=600)
        self.assertIs(result[1], request)
        self.assertIs(result[-1], latest)
        self.assertEqual(result[2].model_dump(exclude={"content"}), old.model_dump(exclude={"content"}))
        self.assertLessEqual(len(result[2].content), 220)
        payload = tool_payload(result[2].content)
        self.assertEqual(payload["case_id"], "CASE-1234567890AB")
        self.assertEqual(payload["candidates"], ["AB01"])
        self.assertNotIn("ORD-SECRET", result[2].content)
        self.assertIs(compact_old_tools(result, keep_rounds=1, threshold=1)[2], result[2])

    def test_rag_citations_and_investigation_provenance_survive(self):
        messages = [
            HumanMessage(content="查故障"),
            ToolMessage(tool_call_id="r", name="rag_summerize",
                        content="先核查站点。\n" + "正文" * 500 + "\n**知识库出处**\n- 【E1】手册\n- 【E2】FAQ"),
            ToolMessage(tool_call_id="o", name="investigate_outage", content=json.dumps({
                "run_id": "RUN-1234567890AB", "status": "completed",
                "report": {"priority": {"direction": "site", "site_id": "A"},
                           "impact": {"offline_device_count": 15, "device_count": 30},
                           "source_ids": {"site_events": ["gw-A-01"], "alarm_events": ["alarm-A01"]}},
                "padding": "a" * 1000,
            })),
            HumanMessage(content="后续"),
        ]
        result = compact_old_tools(messages, keep_rounds=1, threshold=600)
        self.assertIn(MARKER + "先核查站点。", result[1].content)
        self.assertIn("【E1】手册", result[1].content)
        self.assertIn("【E2】FAQ", result[1].content)
        self.assertEqual(tool_payload(result[2].content)["evidence_ids"], ["gw-A-01", "alarm-A01"])
