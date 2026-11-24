from pathlib import Path
import os
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest
from billing_mysql_fixture import MysqlCase
from service.billing.case_store import CaseStore
from service.billing.simulator_service import SimulatorService
from utils.config_handler import billing_conf


class SimulatorPageTests(MysqlCase):
    def open_scenario(self, name):
        db = patch("service.billing.db.get_billing_engine", return_value=self.engine)
        self.enterContext(db)
        self.enterContext(
            patch(
                "agent.cases.billing.handler.get_case_checkpointer",
                return_value=InMemorySaver(),
            )
        )
        app = AppTest.from_file("pages/1_车主模拟器.py", default_timeout=15).run()
        next(s for s in app.selectbox if s.label == "选择场景").select(name).run()
        next(b for b in app.button if b.label == "生成场景").click().run()
        self.assertFalse(app.exception)
        return app

    def test_direct_offline_check_shows_approval_and_uses_scenario_identity(self):
        app = self.open_scenario("S2")
        self.assertEqual(
            app.selectbox(key="direct_planner").value, billing_conf["planner"]
        )
        app.selectbox(key="direct_planner").select("offline").run()
        next(b for b in app.button if b.label == "直接发起费用核查").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        case, run = [block.value for block in app.code]
        self.assertTrue(case.startswith("CASE-"))
        self.assertEqual(CaseStore(self.engine).get_run(run)["case_id"], case)
        self.assertIn(
            "action_pending_approval", " ".join(m.value for m in app.markdown)
        )
        self.assertIn("资金审批台", app.info[0].value)
        next(s for s in app.selectbox if s.label == "选择场景").select("S7").run()
        next(b for b in app.button if b.label == "生成场景").click().run()
        self.assertFalse(app.code)
        app.selectbox(key="sim_driver").select("CUS-DEMO-ALICE").run()
        next(b for b in app.button if b.label == "直接发起费用核查").click().run()
        self.assertFalse(app.exception)
        self.assertIn("不可访问", [m.value for m in app.markdown])
        result = app.session_state["direct_result"]
        self.assertEqual(result["status"], "forbidden")
        with self.engine.connect() as conn:
            owner = conn.exec_driver_sql(
                "SELECT customer_id FROM cb_dispute_cases WHERE case_id=%s",
                (result["case_id"],),
            ).scalar_one()
        self.assertEqual(owner, app.session_state["scenario"]["customer_id"])

    def test_direct_deepseek_errors_are_rendered(self):
        app = self.open_scenario("S2")
        app.selectbox(key="direct_planner").select("deepseek").run()
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}),
            patch("agent.cases.billing.handler.BillingDisputeHandler.start") as start,
        ):
            next(b for b in app.button if b.label == "直接发起费用核查").click().run()
            start.assert_not_called()
            self.assertFalse(app.exception)
            self.assertIn("DEEPSEEK_API_KEY", app.error[0].value)
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}),
            patch(
                "model.factory.get_chat_model",
                side_effect=RuntimeError("provider unavailable"),
            ),
        ):
            next(b for b in app.button if b.label == "直接发起费用核查").click().run()
            self.assertFalse(app.exception)
            self.assertIn("DeepSeek", app.error[0].value)
            self.assertEqual(app.session_state["direct_result"]["status"], "escalated")

    def test_scenario_and_manual_operations(self):
        page = Path(__file__).resolve().parents[1] / "pages" / "1_车主模拟器.py"
        with patch("service.billing.db.get_billing_engine", return_value=self.engine):
            app = AppTest.from_file(str(page), default_timeout=15).run()

            def click(label):
                next(b for b in app.button if b.label == label).click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)

            def select(label, value):
                next(b for b in app.selectbox if b.label == label).select(value).run()

            self.assertFalse(app.exception)
            app.text_input[0].set_value("页面测试").run()
            click("开户")
            select("充值故障", "callback_pending")
            click("充值入账")
            click("确认充值")
            click("扫码启动充电")
            select("结算故障", "missing_end_reading")
            click("结束并结算")
            click("提交补传")
            click("记录人工退款")
            for i in range(1, 12):
                select("选择场景", f"S{i}")
                click("生成场景")
                self.assertEqual(app.session_state["scenario"]["name"], f"S{i}")
            self.assertTrue(
                next(b for b in app.button if b.label == "重置并恢复种子数据").disabled
            )
            app.checkbox[0].check().run()
            click("重置并恢复种子数据")
            self.assertEqual(len(SimulatorService(self.engine).list_customers()), 2)
