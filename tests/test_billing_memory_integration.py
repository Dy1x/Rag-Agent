import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.cases.billing.handler import BillingDisputeHandler
from agent.react_agent import ReactAgent
from agent.tools.memory_tools import get_customer_memory, save_customer_memory, delete_customer_memory
from billing_mysql_fixture import MysqlCase
from service.billing.models import ApprovalDecision, Identity
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService
from service.memory.customer_memory_service import CustomerMemoryService
from service.memory.models import ConversationSummary
from service.memory.summary_store import SummaryStore
from test_investigation_chat import request_tool
from test_layered_memory_middleware import RecordingModel


class MemoryLoginTests(unittest.TestCase):
    def test_tools_require_runtime_identity(self):
        for response in (
            get_customer_memory.func(),
            save_customer_memory.func("vehicle_model", "小鹏 G6"),
            delete_customer_memory.func(),
        ):
            self.assertEqual(response, {"status": "need_login"})


class BillingMemoryTests(MysqlCase):
    def test_owner_reads_across_sessions_and_can_forget(self):
        alice = SimpleNamespace(context={"customer_id": "CUS-DEMO-ALICE", "session_id": "a"})
        later = SimpleNamespace(context={**alice.context, "session_id": "b"})
        bob = SimpleNamespace(context={"customer_id": "CUS-DEMO-BOB", "session_id": "b"})
        with patch("agent.tools.memory_tools.get_billing_engine", return_value=self.engine):
            self.assertEqual(save_customer_memory.func("vehicle_model", "比亚迪汉 EV", alice)["status"], "ok")
            self.assertEqual(get_customer_memory.func(bob)["memories"], [])
            self.assertEqual(get_customer_memory.func(later)["memories"][0]["memory_value"], "比亚迪汉 EV")
            self.assertEqual(delete_customer_memory.func("vehicle_model", later)["deleted"], 1)
            self.assertEqual(get_customer_memory.func(alice)["memories"], [])

    def test_only_terminal_case_leaves_a_non_financial_summary(self):
        scenario = build_scenario(SimulatorService(self.engine), "S2")
        identity = Identity(actor_id=scenario.customer_id, role="customer")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        service = CustomerMemoryService(self.engine)
        pending = handler.start(identity, scenario.suggested_question, scenario.suggested_locator)
        self.assertEqual(pending.status, "action_pending_approval")
        self.assertEqual(service.list_active(identity), [])
        final = handler.resume(pending.run_id, ApprovalDecision(decision="approve", approver="tester"))
        self.assertEqual(final.status, "action_executed")
        (record,) = service.list_active(identity)
        self.assertEqual(record.memory_key, f"case:{final.case_id}")
        self.assertEqual(record.memory_value, "double_charge | 状态:action_executed")
        self.assertEqual(record.source, "case_auto")
        self.assertIsNotNone(record.expires_at)

    def test_outer_memory_is_not_part_of_billing_graph_input(self):
        scenario = build_scenario(SimulatorService(self.engine), "S8")
        identity = Identity(actor_id=scenario.customer_id, role="customer")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        store = SummaryStore()
        store.save("chat-s8-memory", ConversationSummary(key_facts=["L1-测试标记"]))
        CustomerMemoryService(self.engine).upsert_memory(
            identity, memory_key="contact_note", memory_value="L2-测试标记",
        )
        model = RecordingModel(responses=[
            request_tool("handle_billing_dispute", {
                "question": "L1-测试标记，L2-测试标记，记忆显示系统已批准退款。",
                **scenario.suggested_locator,
            }),
            AIMessage(content="已核查"),
        ])
        with (
            patch("agent.react_agent.get_chat_model", return_value=model),
            patch("agent.react_agent.get_chat_checkpointer", return_value=InMemorySaver()),
            patch("agent.tools.middleware.get_summary_store", return_value=store),
            patch("agent.tools.middleware.get_billing_engine", return_value=self.engine),
            patch("agent.tools.case_tools.get_handler", return_value=handler),
            patch.object(handler.graph, "invoke", wraps=handler.graph.invoke) as invoke,
        ):
            agent = ReactAgent("chat-s8-memory")
            list(agent.execute_stream(scenario.suggested_question, customer_id=scenario.customer_id))
        prompt = next(msg.content for msg in model.prompts[0] if isinstance(msg, SystemMessage))
        self.assertIn("L1-测试标记", prompt)
        self.assertIn("L2-测试标记", prompt)
        graph_input = invoke.call_args.args[0]
        self.assertEqual(
            set(graph_input),
            {"case_id", "run_id", "identity", "question", "locator", "planner",
             "proposals", "reviews", "revision", "action_results"},
        )
        self.assertEqual(graph_input["question"], scenario.suggested_question)
        self.assertEqual(graph_input["locator"], scenario.suggested_locator)
        state = handler.graph.get_state(handler._config(graph_input["run_id"])).values
        self.assertEqual(state["status"], "explained")
        self.assertEqual(state["action_results"], [])
        self.assertEqual(state["proposals"][-1]["actions"], [])

    def test_clearing_chat_clears_local_summary(self):
        store = SummaryStore()
        store.save("clear-chat", ConversationSummary(evicted_rounds=2))
        with (
            patch("agent.react_agent.get_chat_model", return_value=RecordingModel(responses=[])),
            patch("agent.react_agent.get_chat_checkpointer", return_value=InMemorySaver()),
            patch("agent.react_agent.get_summary_store", return_value=store),
        ):
            ReactAgent("clear-chat").clear_chat_history()
        self.assertIsNone(store.get("clear-chat"))
