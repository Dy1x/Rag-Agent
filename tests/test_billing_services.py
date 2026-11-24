from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from billing_mysql_fixture import MysqlCase

from agent.cases.billing.handler import BillingDisputeHandler
from service.billing.errors import BillingError
from service.billing.facts_service import BillingFactsService
from service.billing.ids import new_id
from service.billing.models import Identity
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService


class ServiceTests(MysqlCase):
    def test_suffix_lookup_is_case_insensitive_and_scoped_for_both_record_types(self):
        ids = {
            "ORD": iter(["ORD-000000004AC5"]),
            "ATT": iter(["ATT-000000001A18", "ATT-000000009999"]),
        }
        with patch(
            "service.billing.simulator_service.new_id",
            side_effect=lambda prefix: next(ids[prefix])
            if prefix in ids
            else new_id(prefix),
        ):
            h = build_scenario(SimulatorService(self.engine), "S1")
        owner = Identity(role="customer", actor_id=h.customer_id)
        bob = Identity(role="customer", actor_id="CUS-DEMO-BOB")
        service = BillingFactsService(self.engine)
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        for suffix, order, attempt, status in [
            ("4ac5", h.refs["order"], h.refs["attempt"], "explained"),
            ("1a18", None, h.refs["first_attempt"], "action_pending_approval"),
        ]:
            with self.subTest(suffix=suffix):
                (row,) = service.find_candidate_orders(owner, order_id=suffix)
                self.assertEqual((row.order_id, row.attempt_id), (order, attempt))
                self.assertFalse(service.find_candidate_orders(bob, order_id=suffix))
                own = handler.start(owner, h.suggested_question, {"order_id": suffix})
                self.assertEqual(own.status, status)
                outsider = handler.start(
                    bob, h.suggested_question, {"order_id": suffix}
                )
                self.assertEqual(outsider.status, "clarify")
                self.assertEqual(outsider.report["candidates"], [])
                self.assertEqual(outsider.report["chain"], [])
        outsider = handler.start(
            bob, h.suggested_question, {"order_id": h.refs["order"]}
        )
        self.assertEqual(outsider.status, "forbidden")
        self.assertEqual(outsider.report["chain"], [])

    def test_pending_refund_and_settled_freeze_have_independent_evidence(self):
        from agent.cases.billing.tools import read_tools

        sim = SimulatorService(self.engine)
        service = BillingFactsService(self.engine)
        h = build_scenario(sim, "S5")
        who = Identity(role="customer", actor_id=h.customer_id)
        facts = service.build_billing_facts(
            who, order_ids=[h.refs["order"]], attempt_ids=[]
        )
        order = facts.orders[0]
        tools = read_tools(service, who, lambda *args: None, "offline")
        amount = tools["recompute_order"](order.order.order_id)
        self.assertEqual(amount["started_at_local"], "2026-11-20T08:30:00+08:00")
        self.assertEqual(amount["period_type"], "flat")
        self.assertEqual(
            tools["get_session"](order.session.session_id)["started_at"],
            "2026-11-20T00:30:00Z",
        )
        self.assertEqual(order.order_refundable_fen, 0)
        self.assertEqual(order.refunds[0].related_txn_id, h.refs["second_deduct"])
        freeze = facts.freezes[0]
        (release,) = tools["get_transactions"](txn_ids=[freeze.released_by_txn_id])
        self.assertEqual(
            (release["related_txn_id"], release["txn_type"], release["status"]),
            (freeze.txn_id, "unfreeze", "succeeded"),
        )
        self.assertIn(release["txn_id"], facts.evidence_index)

    def test_backfill_replaces_open_hold_with_linked_release(self):
        sim = SimulatorService(self.engine)
        service = BillingFactsService(self.engine)
        h = build_scenario(sim, "S6")
        who = Identity(role="customer", actor_id=h.customer_id)

        def read():
            return service.build_billing_facts(
                who, order_ids=[h.refs["order"]], attempt_ids=[]
            )

        before = read()
        self.assertIsNone(before.freezes[0].released_by_txn_id)
        sim.backfill_end_reading(
            h.refs["session"], 4000, before.orders[0].session.charge_end_at
        )
        after = read()
        self.assertEqual(after.freezes[0].txn_id, before.freezes[0].txn_id)
        self.assertIn(after.freezes[0].released_by_txn_id, after.evidence_index)
        self.assertEqual(
            (
                after.orders[0].recomputed.total_fen,
                after.orders[0].order_refundable_fen,
            ),
            (560, 0),
        )
        self.assertFalse(after.anomalies)

    def test_all_scenarios_and_reconciliation(self):
        sim = SimulatorService(self.engine)
        facts = BillingFactsService(self.engine)
        expected = [
            ["UNRELEASED_FREEZE"],
            ["DUPLICATE_DEDUCT"],
            ["PENDING_RECHARGE"],
            ["TARIFF_MISMATCH"],
            ["DUPLICATE_DEDUCT", "PENDING_REFUND"],
            ["MISSING_END_READING"],
            [],
            [],
            [],
            ["IDLE_FEE_MISMATCH"],
            [],
        ]
        for i, codes in enumerate(expected, 1):
            with self.subTest(scenario=i):
                h = build_scenario(sim, f"S{i}")
                who = Identity(role="customer", actor_id=h.customer_id)
                self.assertTrue(sim.get_wallet(h.customer_id).consistent)
                if i == 7:
                    with self.assertRaisesRegex(BillingError, "不可访问"):
                        facts.get_order(who, h.refs["order"])
                    self.assertEqual(
                        facts.find_candidate_orders(who, order_id=h.refs["order"]), []
                    )
                    continue
                candidates = facts.find_candidate_orders(who)
                data = facts.build_billing_facts(
                    who,
                    order_ids=[r.order_id for r in candidates if r.order_id],
                    attempt_ids=[r.attempt_id for r in candidates],
                )
                self.assertCountEqual([a.code for a in data.anomalies], codes)
                if i == 6:
                    s = data.orders[0].session
                    sim.backfill_end_reading(s.session_id, 4000, s.charge_end_at)
                    self.assertEqual(sim.get_wallet(h.customer_id).frozen_fen, 0)
                    self.assertEqual(
                        facts.recompute_order(who, h.refs["order"]).total_fen, 560
                    )
                if i == 3:
                    sim.confirm_recharge(h.refs["recharge"])
                    self.assertEqual(sim.get_wallet(h.customer_id).balance_fen, 30000)
                self.assertTrue(sim.get_wallet(h.customer_id).consistent)

    def test_insufficient_balance_and_immediate_timeout_release(self):
        sim = SimulatorService(self.engine)
        customer = sim.create_customer("启动测试").customer_id
        with self.assertRaisesRegex(BillingError, "余额不足"):
            sim.start_charging(customer, "CP-A01", "1")
        sim.recharge(customer, 5000)
        sim.start_charging(customer, "CP-A01", "1", fault="start_timeout")
        w = sim.get_wallet(customer)
        self.assertEqual((w.available_fen, w.consistent), (5000, True))
