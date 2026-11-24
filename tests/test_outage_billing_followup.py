import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from billing_mysql_fixture import MysqlCase
from agent.cases.billing.handler import BillingDisputeHandler
from eval.investigation.run_multi_agent import prepare_fixture
from service.billing.case_store import CaseStore
from service.billing.db import reset_business_data
from service.billing.errors import BillingError
from service.billing.models import Identity, ApprovalDecision
from service.billing.simulator_service import SimulatorService
from service.investigation_data_service import InvestigationScope
from service.investigation_mysql import (
    MySQLInvestigationSource,
    apply_investigation_schema,
    import_sqlite,
)
from service.investigation_service import multi_dependencies_for, run_investigation


class OutageBillingTests(MysqlCase):
    def test_only_confirmed_sessions_get_one_child_and_require_approval(self):
        apply_investigation_schema(self.engine)
        sim = SimulatorService(self.engine)
        start = datetime(2026, 11, 21, 14)
        owners = []
        sessions = []
        for index, fault in enumerate(
            ("duplicate_deduct", "missing_end_reading", None, None, None)
        ):
            owner = sim.create_customer("联动测试" + str(index)).customer_id
            sim.recharge(owner, 20000, at=start)
            a = sim.start_charging(owner, "CP-A01", "1", at=start)
            if index != 4:
                sim.stop_charging(
                    a.session_id,
                    end_wh=4000,
                    charge_end_at=start + timedelta(minutes=10),
                    unplugged_at=start + timedelta(minutes=11),
                    fault=fault,
                )
            owners.append(owner)
            sessions.append(a.session_id)
        source = MySQLInvestigationSource(self.engine, "followup")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.sqlite"
            scope = InvestigationScope(
                **prepare_fixture("rollout_site_overlap", path)["scope"]
            )
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("""CREATE TABLE session_interruptions(interruption_id TEXT PRIMARY KEY,device_id TEXT,
                    charger_id TEXT,session_id TEXT,site_event_id TEXT,alarm_event_id TEXT,cause TEXT,occurred_at TEXT)""")
                for i in range(5):
                    device = f"A{i + 1:02d}"
                    db.execute(
                        "INSERT INTO session_interruptions VALUES (?,?,?,?,?,?,?,?)",
                        (
                            f"confirmed-{i}",
                            device,
                            "CP-A02" if i == 3 else "CP-A01",
                            sessions[i],
                            "gw-A-01",
                            "overlap-alarm-" + device,
                            "power_loss" if i == 2 else "gateway_restart",
                            "2026-11-21T14:07:00Z",
                        ),
                    )
            import_sqlite(path, source)

        saver = InMemorySaver()

        def investigate(_):
            return run_investigation(
                "核查中断",
                scope,
                multi_dependencies_for(source, "offline"),
                mode="offline",
                simulated_data=True,
            )

        with (
            patch(
                "agent.cases.billing.handler.get_case_checkpointer", return_value=saver
            ),
            ThreadPoolExecutor(2) as pool,
        ):
            first, second = list(pool.map(investigate, [1, 2]))
        self.assertEqual(
            {c["case_id"] for c in first["billing_followups"]},
            {c["case_id"] for c in second["billing_followups"]},
        )
        children = {c["interruption_id"]: c for c in first["billing_followups"]}
        self.assertEqual(set(children), {"confirmed-0", "confirmed-1", "confirmed-4"})
        self.assertEqual(children["confirmed-0"]["status"], "action_pending_approval")
        for key in ("confirmed-1", "confirmed-4"):
            self.assertEqual(children[key]["status"], "pending_evidence")
        store = CaseStore(self.engine)
        child = children["confirmed-0"]
        with self.assertRaises(BillingError):
            store.get_case_detail(
                Identity(role="customer", actor_id=owners[1]), child["case_id"]
            )
        self.assertFalse(
            any(t.txn_type == "refund" for t in sim.list_transactions(owners[0]))
        )
        handler = BillingDisputeHandler(self.engine, saver)
        decision = ApprovalDecision(decision="approve", approver="test")
        for _ in range(2):
            self.assertEqual(
                handler.resume(child["run_id"], decision).status, "action_executed"
            )
        self.assertEqual(
            [
                t.amount_fen
                for t in sim.list_transactions(owners[0])
                if t.txn_type == "refund"
            ],
            [560],
        )
        sim.backfill_end_reading(sessions[1], 4000, start + timedelta(minutes=10))
        self.assertEqual(
            handler.rerun(
                children["confirmed-1"]["case_id"],
                Identity(role="customer", actor_id=owners[1]),
            ).status,
            "explained",
        )
        for result in [first, second]:
            self.assertEqual(
                len(
                    [
                        e
                        for e in store.get_trace(result["run_id"])
                        if e["event_type"] == "child_case"
                    ]
                ),
                3,
            )
        reset_business_data(self.engine)
        with self.engine.connect() as c:
            self.assertEqual(c.scalar(text("SELECT COUNT(*) FROM ci_case_links")), 0)
            self.assertEqual(
                c.scalar(text("SELECT COUNT(*) FROM ci_billing_children")), 0
            )
