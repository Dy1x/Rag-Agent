from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from streamlit.testing.v1 import AppTest

from billing_mysql_fixture import MysqlCase
from service.billing.models import Identity
from service.memory.customer_memory_service import CustomerMemoryService
from service.memory.models import ConversationSummary
from service.memory.summary_store import SummaryStore


class MemoryViewTests(MysqlCase):
    def test_owner_switch_new_session_and_forget(self):
        service = CustomerMemoryService(self.engine)
        alice = Identity(role="customer", actor_id="CUS-DEMO-ALICE")
        for car in ("特斯拉 Model 3", "比亚迪汉 EV"):
            service.upsert_memory(alice, memory_key="vehicle_model", memory_value=car)
        record = service.list_active(alice)[0]
        store = SummaryStore()
        thread = f"{uuid4()}:{alice.actor_id}"
        store.save(thread, ConversationSummary(evicted_rounds=3, key_facts=["核查 CP-A01"]))
        app = AppTest.from_file("app.py", default_timeout=15)
        app.query_params["session_id"] = thread.split(":")[0]
        app.query_params["customer_id"] = alice.actor_id
        with (
            patch("service.billing.db.get_billing_engine", return_value=self.engine),
            patch("service.memory.summary_store.get_summary_store", return_value=store),
            patch("agent.react_agent.ReactAgent",
                  side_effect=lambda session_id: SimpleNamespace(session_id=session_id, get_chat_history=list)),
        ):
            app.run()
            self.assertFalse(app.exception)
            texts = [item.value for item in app.sidebar.text]
            self.assertTrue(any("核查 CP-A01" in text for text in texts), texts)
            self.assertTrue(any("特斯拉 Model 3 → 比亚迪汉 EV" in text for text in texts))
            app.sidebar.selectbox[0].select("CUS-DEMO-BOB").run()
            self.assertFalse(app.exception)
            self.assertFalse(any("比亚迪汉 EV" in text.value for text in app.sidebar.text))
            app.sidebar.selectbox[0].select(alice.actor_id).run()
            next(button for button in app.sidebar.button if button.label == "新建会话").click().run()
            self.assertFalse(app.exception)
            self.assertNotEqual(app.session_state["agent_thread_id"], thread)
            self.assertFalse(any("核查 CP-A01" in text.value for text in app.sidebar.text))
            app.sidebar.button(key=f"forget-{record.memory_id}").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(service.list_active(alice), [])
            self.assertTrue(any("没有活跃记忆" in item.value for item in app.sidebar.caption))
