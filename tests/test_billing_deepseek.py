import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.cases.agent_spec import build_agent_node
from agent.cases.billing.intake import SPEC
from agent.cases.billing.checker import SPEC as CHECKER
from agent.cases.billing.maker import SPEC as MAKER
from agent.cases.billing.offline import customer_message_issues
from agent.cases.billing.tools import read_tools
from rag.evidence import EvidenceChunk, RagResult


class FakeToolsModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class DeepSeekTests(unittest.TestCase):
    def test_message_guard_distinguishes_deposit_from_promises(self):
        self.assertEqual(customer_message_issues("50 元保证金未释放"), [])
        for message in ("保证立即到账", "保证为您全额退回", "保证处理"):
            with self.subTest(message=message):
                self.assertEqual(
                    [issue.code for issue in customer_message_issues(message)],
                    ["S_OVERPROMISE"],
                )

    def test_single_agent_keeps_history_and_refreshes_checker_budget(self):
        calls = []

        def get_order(order_id: str):
            """读取当前订单。"""
            calls.append(order_id)
            return {"order_id": order_id}

        def response(name, args, key):
            return AIMessage(
                content="", tool_calls=[dict(name=name, args=args, id=key)]
            )

        model = FakeToolsModel(
            responses=[
                response(
                    "Proposal",
                    dict(
                        proposal_id="PRP-test",
                        version=1,
                        resolution="explained",
                        rationale="测试",
                        evidence_ids=["ORD-test"],
                        customer_message="已核对。",
                    ),
                    "proposal",
                ),
                *(
                    response("get_order", {"order_id": str(i)}, f"read-{i}")
                    for i in range(8)
                ),
                response("SoftReviewOpinion", {"approved": True}, "verdict"),
            ]
        )
        dialogue = {}
        with patch("model.factory.get_chat_model", return_value=model):
            for spec, count, payload in [
                (MAKER, 3, {"question": "核对本次费用"}),
                (CHECKER, 8, {"instruction": "自查"}),
            ]:
                result = build_agent_node(
                    replace(
                        spec,
                        prompt_file=MAKER.prompt_file,
                        tools=["get_order"],
                        max_tool_calls=count,
                    ),
                    "deepseek",
                    {"get_order": get_order},
                    None,
                    dialogue=dialogue,
                )(payload)
        self.assertTrue(result.approved)
        self.assertEqual(calls, [str(i) for i in range(8)])
        self.assertIn("核对本次费用", dialogue["messages"][0].content)

    def test_checker_can_submit_after_using_all_eight_reads(self):
        calls = []

        def get_order(order_id: str):
            """读取当前订单。"""
            calls.append(order_id)
            return {"order_id": order_id}

        reads = [
            dict(name="get_order", args={"order_id": str(i)}, id=f"read-{i}")
            for i in range(9)
        ]
        model = FakeToolsModel(
            responses=[
                *(AIMessage(content="", tool_calls=[read]) for read in reads),
                AIMessage(
                    content="",
                    tool_calls=[
                        dict(
                            name="SoftReviewOpinion",
                            args={"approved": True},
                            id="verdict",
                        )
                    ],
                ),
            ]
        )
        with patch("model.factory.get_chat_model", return_value=model):
            invoke = build_agent_node(
                replace(CHECKER, tools=["get_order"], max_tool_calls=8),
                "deepseek",
                {"get_order": get_order},
                None,
            )
            self.assertTrue(invoke({}).approved)
        self.assertEqual(calls, [str(i) for i in range(8)])

    def test_faq_returns_json_evidence_or_explicit_unavailability(self):
        trace = Mock()
        faq = read_tools(None, None, trace, "deepseek")["search_billing_faq"]
        result = RagResult(
            status="answered",
            text="按订单政策核查【E1】。",
            evidence=[
                EvidenceChunk(id="E1", source="计费手册", text="按订单政策核查。")
            ],
        )
        with patch("rag.rag_service.RagSummerizeService") as rag:
            rag.return_value.answer_question.return_value = result
            self.assertEqual(faq("如何计费？"), result.model_dump(mode="json"))
            rag.return_value.answer_question.side_effect = ConnectionError("redis")
            fallback = faq("如何计费？")
            self.assertEqual(RagResult.model_validate(fallback).status, "insufficient")
            self.assertEqual(fallback["evidence"], [])
            trace.assert_called_with("faq_unavailable", {"reason": "ConnectionError"})

    def test_tool_strategy_scoped_read_and_retry(self):
        called = []

        def find_candidate_orders(order_id: str):
            """读取当前车主的候选。"""
            called.append(order_id)
            return []

        def msg(name, args, i):
            return AIMessage(
                content="", tool_calls=[dict(name=name, args=args, id=f"call-{i}")]
            )

        model = FakeToolsModel(
            responses=[
                msg("find_candidate_orders", dict(order_id="ORD-demo"), 1),
                msg("CaseFile", dict(status="invalid"), 2),
                msg("CaseFile", dict(status="forbidden", dispute_type="other"), 3),
            ]
        )
        with patch("model.factory.get_chat_model", return_value=model):
            invoke = build_agent_node(
                SPEC, "deepseek", {"find_candidate_orders": find_candidate_orders}, None
            )
            result = invoke(dict(question="查订单", locator={"order_id": "ORD-demo"}))
        self.assertEqual(result.status, "forbidden")
        self.assertEqual(called, ["ORD-demo"])

    def test_read_budget_includes_validation_retry(self):
        calls = []

        def find_candidate_orders(order_id: str):
            """查找当前车主的订单。"""
            calls.append(order_id)
            return []

        def call(name, args, i):
            return AIMessage(
                content="", tool_calls=[dict(name=name, args=args, id=str(i))]
            )

        model = FakeToolsModel(
            responses=[
                call("find_candidate_orders", {"order_id": "1"}, 1),
                call("find_candidate_orders", {"order_id": "2"}, 2),
                call("CaseFile", {"status": "invalid"}, 3),
                call("find_candidate_orders", {"order_id": "3"}, 4),
                call("CaseFile", {"status": "forbidden", "dispute_type": "other"}, 5),
            ]
        )
        with patch("model.factory.get_chat_model", return_value=model):
            result = build_agent_node(
                SPEC, "deepseek", {"find_candidate_orders": find_candidate_orders}, None
            )({})
        self.assertEqual(result.status, "forbidden")
        self.assertEqual(calls, ["1", "2"])
