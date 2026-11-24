import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from langgraph.checkpoint.memory import InMemorySaver
from billing_mysql_fixture import MysqlCase
from agent.cases.billing.handler import BillingDisputeHandler
from agent.cases.billing.checkpoints import get_case_checkpointer
from service.billing.models import Identity, ApprovalDecision
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService


class ApprovalTests(MysqlCase):
    def test_restart_approve_modified_and_repeat(self):
        sim = SimulatorService(self.engine)
        h = build_scenario(sim, "S2")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "checkpoints.sqlite")
            saver = get_case_checkpointer(path)
            self.addCleanup(get_case_checkpointer.cache_clear)
            try:
                handler = BillingDisputeHandler(self.engine, saver)
                r = handler.start(
                    Identity(role="customer", actor_id=h.customer_id),
                    h.suggested_question,
                    h.suggested_locator,
                )
                code = """import sys
from agent.cases.billing.handler import BillingDisputeHandler
from agent.cases.billing.checkpoints import get_case_checkpointer
from service.billing.db import get_billing_engine
from service.billing.models import ApprovalDecision,Proposal
h=BillingDisputeHandler(get_billing_engine(sys.argv[1]),get_case_checkpointer(sys.argv[2]))
r=sys.argv[3]
p=Proposal(**h.graph.get_state(h._config(r)).values['proposals'][-1])
p.actions[0].amount_fen=200
d=ApprovalDecision(decision='approve_modified',approver='ops',modified_actions=p.actions)
assert h.resume(r,d).status=='action_executed'
assert h.resume(r,d).report['action_results'][0]['amount_fen']==200
"""
                subprocess.run(
                    [sys.executable, "-c", code, self.database, path, r.run_id],
                    check=True,
                )
                self.assertEqual(
                    [
                        t.amount_fen
                        for t in sim.list_transactions(h.customer_id)
                        if t.txn_type == "refund"
                    ],
                    [200],
                )
            finally:
                saver.conn.close()

    def test_modified_unfreeze_must_repass_hard_checks(self):
        h = build_scenario(SimulatorService(self.engine), "S1")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        r = handler.start(
            Identity(role="customer", actor_id=h.customer_id),
            h.suggested_question,
            h.suggested_locator,
        )
        a = handler.graph.get_state(handler._config(r.run_id)).values["proposals"][-1][
            "actions"
        ][0]
        d = ApprovalDecision(
            decision="approve_modified",
            approver="ops",
            modified_actions=[dict(a, amount_fen=1)],
        )
        result = handler.resume(r.run_id, d)
        self.assertEqual(result.status, "action_rejected")
        self.assertIn("H_UNFREEZE_INVALID", result.report["failure"])

    def test_approval_and_trace_pages(self):
        h = build_scenario(SimulatorService(self.engine), "S4")
        saver = InMemorySaver()
        handler = BillingDisputeHandler(self.engine, saver)
        r = handler.start(
            Identity(role="customer", actor_id=h.customer_id),
            h.suggested_question,
            h.suggested_locator,
        )
        with (
            patch("service.billing.db.get_billing_engine", return_value=self.engine),
            patch(
                "agent.cases.billing.handler.get_case_checkpointer", return_value=saver
            ),
        ):
            page = AppTest.from_file("pages/2_审批台.py", default_timeout=15).run()
            page.selectbox[0].select(r.case_id).run()
            page.button[0].click().run()
            self.assertFalse(page.exception)
            self.assertEqual(
                page.session_state["last_approval"]["status"], "action_executed"
            )
            trace = AppTest.from_file("pages/3_运行轨迹.py").run()
            trace.text_input[0].set_value(r.case_id).run()
            self.assertFalse(trace.exception)
            self.assertTrue(any("resume" in e.label for e in trace.expander))
            self.assertEqual(
                [button.label for button in trace.get("download_button")],
                ["下载 JSON"],
            )
