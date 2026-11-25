import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from types import SimpleNamespace

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from billing_mysql_fixture import MysqlCase
from test_investigation_chat import ToolCallingModel, request_tool
from agent.react_agent import ReactAgent
from agent.cases.billing.handler import BillingDisputeHandler
from agent.tools.case_tools import handle_billing_dispute, get_dispute_case_status
from agent.tools.memory_tools import CUSTOMER_MEMORY_TOOLS
from service.billing.models import Identity
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService


class ChatSchemaTests(unittest.TestCase):
    def test_default_agent_exposes_only_charging_tools_and_registered_handlers(self):
        expected = {
            "rag_summerize",
            "handle_billing_dispute",
            "get_dispute_case_status",
            "investigate_outage",
            "get_memory_context",
            "save_memory_context",
            "clear_memory_context",
            *(tool.name for tool in CUSTOMER_MEMORY_TOOLS),
        }
        extra = SimpleNamespace(
            name="charger_service_case", description="充电桩维修案件"
        )
        for handlers in ([], [extra]):
            with (
                self.subTest(handlers=handlers),
                patch(
                    "agent.tools.case_tools.additional_handlers", return_value=handlers
                ),
                patch("agent.react_agent.get_chat_model"),
                patch("agent.react_agent.get_chat_checkpointer"),
                patch("agent.react_agent.create_agent") as create,
            ):
                ReactAgent()
                names = [tool.name for tool in create.call_args.kwargs["tools"]]
                self.assertCountEqual(names, expected | {h.name for h in handlers})

    def test_identity_is_runtime_only(self):
        for t in [handle_billing_dispute, get_dispute_case_status, *CUSTOMER_MEMORY_TOOLS]:
            self.assertNotIn(
                "runtime", t.tool_call_schema.model_json_schema()["properties"]
            )
            self.assertNotIn(
                "customer_id", t.tool_call_schema.model_json_schema()["properties"]
            )
        self.assertEqual(
            handle_billing_dispute.func("查扣款"), {"status": "need_login"}
        )


class BillingChatTests(MysqlCase):
    def test_second_chat_turn_resolves_a_masked_candidate_or_narrow_time_window(self):
        h = build_scenario(SimulatorService(self.engine), "S11")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        runtime = SimpleNamespace(
            context={"customer_id": h.customer_id, "session_id": "chat-s11-followup"}
        )
        with (
            patch("agent.tools.case_tools.get_handler", return_value=handler),
            patch.dict("agent.tools.case_tools.billing_conf", {"planner": "offline"}),
        ):
            first = handle_billing_dispute.func(
                h.suggested_question, **h.suggested_locator, runtime=runtime
            )
            self.assertEqual(first["status"], "clarify")
            self.assertEqual(len(first["candidates"]), 2)
            latest = max(first["candidates"], key=lambda r: r["本地时间"])
            self.assertEqual(latest["订单尾号"], h.refs["second_order"][-4:])
            when = datetime.fromisoformat(latest["本地时间"])
            minute = timedelta(minutes=1)
            for locator in [
                {"order_id": latest["订单尾号"], "start_local": when.isoformat()},
                {
                    "start_local": (when - minute).isoformat(),
                    "end_local": (when + minute).isoformat(),
                },
            ]:
                with self.subTest(locator=locator):
                    reply = handle_billing_dispute.func(
                        f"{h.suggested_question}，查最近的一次。",
                        charger_id=latest["桩号"],
                        **locator,
                        runtime=runtime,
                    )
                    self.assertEqual(reply["status"], "explained")
                    detail = handler.store.get_case_detail(
                        Identity(role="customer", actor_id=h.customer_id),
                        reply["case_id"],
                    )
                    (matched,) = detail["runs"][0]["report"]["chain"]
                    self.assertEqual(matched["order_id"], h.refs["second_order"])

    def test_real_chat_tool_for_approval_clarification_and_status(self):
        sim = SimulatorService(self.engine)
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        for scenario, status in [("S2", "action_pending_approval"), ("S11", "clarify")]:
            h = build_scenario(sim, scenario)
            model = ToolCallingModel(
                responses=[
                    request_tool(
                        "handle_billing_dispute",
                        dict(question=h.suggested_question, **h.suggested_locator),
                    ),
                    AIMessage(content="请查看处理结果"),
                ]
            )
            saver = InMemorySaver()
            with (
                patch("agent.react_agent.get_chat_model", return_value=model),
                patch("agent.react_agent.get_chat_checkpointer", return_value=saver),
                patch("agent.tools.case_tools.get_handler", return_value=handler),
            ):
                agent = ReactAgent("chat-" + scenario)
                list(
                    agent.execute_stream(
                        h.suggested_question, customer_id=h.customer_id
                    )
                )
                messages = saver.get(agent.get_checkpoint_config())["channel_values"][
                    "messages"
                ]
                result = json.loads(
                    next(m.content for m in messages if isinstance(m, ToolMessage))
                )
                self.assertEqual(result["status"], status)
                self.assertEqual(
                    get_dispute_case_status.func(
                        result["case_id"],
                        SimpleNamespace(context={"customer_id": h.customer_id}),
                    )["status"],
                    status,
                )
                self.assertEqual(
                    get_dispute_case_status.func(
                        result["case_id"],
                        SimpleNamespace(context={"customer_id": "CUS-DEMO-BOB"}),
                    )["status"],
                    "forbidden",
                )

    def test_switching_drivers_changes_chat_thread(self):
        history_by_thread = {}

        def make_agent(*, session_id):
            mock = SimpleNamespace(
                session_id=session_id,
                get_chat_history=lambda sid=session_id: list(
                    history_by_thread.get(sid, [])
                ),
            )
            return mock

        with (
            patch("service.billing.db.get_billing_engine", return_value=self.engine),
            patch("agent.react_agent.ReactAgent", side_effect=make_agent) as agent,
        ):
            app = AppTest.from_file("app.py", default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "新能源充电桩智能客服")
            first = agent.call_args.kwargs["session_id"]
            history_by_thread[first] = [{"role": "user", "content": "Alice 历史提问"}]

            # 切到 Bob 时使用独立线程
            app.sidebar.selectbox[0].select("CUS-DEMO-BOB").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["customer_id"], "CUS-DEMO-BOB")
            second = agent.call_args.kwargs["session_id"]
            self.assertNotEqual(first, second)
            self.assertEqual(app.session_state["message"], [])

            # 切回 Alice 时恢复原 session_id 与历史记录
            app.sidebar.selectbox[0].select("CUS-DEMO-ALICE").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["customer_id"], "CUS-DEMO-ALICE")
            self.assertEqual(agent.call_args.kwargs["session_id"], first)
            self.assertEqual(
                app.session_state["message"],
                [{"role": "user", "content": "Alice 历史提问"}],
            )

            # 模拟多页面跳转导致 URL 参数清空后切回首页：仍保留原车主与 session_id
            app.query_params.clear()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["customer_id"], "CUS-DEMO-ALICE")
            self.assertEqual(app.session_state["agent_thread_id"], first)
            self.assertEqual(app.query_params["customer_id"], ["CUS-DEMO-ALICE"])
            self.assertEqual(app.query_params["session_id"], [first.split(":")[0]])
