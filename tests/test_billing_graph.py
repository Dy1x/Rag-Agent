import json
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from billing_mysql_fixture import MysqlCase

from agent.cases.billing.handler import BillingDisputeHandler
from service.billing.ids import new_id
from service.billing.models import ApprovalDecision, CaseFile, Identity
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService


class GraphTests(MysqlCase):
    def test_rewritten_failed_start_question_selects_only_the_failed_attempt(self):
        h = build_scenario(SimulatorService(self.engine), "S1")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        who = Identity(role="customer", actor_id=h.customer_id)
        result = handler.start(
            who,
            "用户扫码两次，第一次充电未启动，但余额减少了一大块，怀疑产生了扣费或冻结。",
            {},
            planner="offline",
        )
        self.assertEqual(result.status, "action_pending_approval")
        detail = handler.store.get_case_detail(who, result.case_id)
        self.assertEqual(detail["dispute_type"], "failed_start_charged")
        (action,) = detail["proposals"][0]["actions"]
        self.assertEqual(
            (action["action_type"], action["target_txn_id"]),
            ("unfreeze", h.refs["first_freeze"]),
        )

    def test_deepseek_suffix_validation_uses_real_ids_even_if_model_asks_to_clarify(
        self,
    ):
        ids = {
            "ORD": iter(["ORD-000000001A18"]),
            "ATT": iter(["ATT-000000001A18", "ATT-000000008888"]),
        }
        with patch(
            "service.billing.simulator_service.new_id",
            side_effect=lambda prefix: next(ids[prefix])
            if prefix in ids
            else new_id(prefix),
        ):
            h = build_scenario(SimulatorService(self.engine), "S1")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        who = Identity(role="customer", actor_id=h.customer_id)

        def use_offline_after_intake(spec, planner, tools, offline, **kwargs):
            if spec.name == "intake":
                return lambda payload: CaseFile(
                    status="clarify"
                    if "start_local" in payload["locator"]
                    else "located",
                    dispute_type="failed_start_charged",
                )
            return lambda payload: offline(payload, tools)

        with patch(
            "agent.cases.billing.graph.build_agent_node",
            side_effect=use_offline_after_intake,
        ):
            for ref, status, order_ids, attempt_id, start, end in [
                (
                    "order",
                    "explained",
                    [h.refs["order"]],
                    h.refs["attempt"],
                    "08:30",
                    "08:31",
                ),
                (
                    "first_attempt",
                    "action_pending_approval",
                    [],
                    h.refs["first_attempt"],
                    "08:20",
                    "08:21",
                ),
            ]:
                with self.subTest(ref=ref):
                    result = handler.start(
                        who,
                        h.suggested_question,
                        {
                            "order_id": h.refs[ref][-4:],
                            "charger_id": "CP-A01",
                            "start_local": f"2026-11-20T{start}:00",
                            "end_local": f"2026-11-20T{end}:00",
                        },
                        planner="deepseek",
                    )
                    self.assertEqual(result.status, status)
                    case = handler.graph.get_state(
                        handler._config(result.run_id)
                    ).values["case_file"]
                    self.assertEqual(case["order_ids"], order_ids)
                    self.assertEqual(case["attempt_ids"], [attempt_id])
            ambiguous = handler.start(
                who, h.suggested_question, {"order_id": "1A18"}, planner="deepseek"
            )
            self.assertEqual(ambiguous.status, "clarify")
            self.assertEqual(
                [r["订单尾号"] for r in ambiguous.report["candidates"]],
                ["1A18", "1A18"],
            )
            self.assertEqual(ambiguous.report["chain"], [])

    def test_offline_cases_and_trace(self):
        sim = SimulatorService(self.engine)
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        expected = [
            "action_executed",
            "action_executed",
            "pending_evidence",
            "action_executed",
            "explained",
            "pending_evidence",
            "forbidden",
            "explained",
            "explained",
            "action_executed",
            "clarify",
        ]
        actions = {
            1: ("unfreeze", "first_freeze", 5000),
            2: ("refund", "second_deduct", 560),
            4: ("refund", "first_deduct", 240),
            10: ("refund", "first_deduct", 500),
        }
        for i, status in enumerate(expected, 1):
            with self.subTest(i=i):
                h = build_scenario(sim, f"S{i}")
                who = Identity(role="customer", actor_id=h.customer_id)
                result = handler.start(who, h.suggested_question, h.suggested_locator)
                if i in actions:
                    self.assertEqual(result.status, "action_pending_approval")
                    result = handler.resume(
                        result.run_id,
                        ApprovalDecision(decision="approve", approver="tester"),
                    )
                    kind, target, amount = actions[i]
                    a = result.report["action_results"][0]
                    self.assertEqual(
                        (a["action_type"], a["target_txn_id"], a["amount_fen"]),
                        (kind, h.refs[target], amount),
                    )
                self.assertEqual(result.status, status)
                trace = handler.store.get_trace(result.run_id)
                self.assertEqual(
                    [r["seq"] for r in trace], list(range(1, len(trace) + 1))
                )
                self.assertEqual(trace[-1]["node"], "finalize")
                if i == 6:
                    from service.billing.facts_service import BillingFactsService

                    s = BillingFactsService(self.engine).get_session(
                        who, h.refs["session"]
                    )
                    sim.backfill_end_reading(s.session_id, 4000, s.charge_end_at)
                    self.assertEqual(
                        handler.rerun(result.case_id, who).status, "explained"
                    )
                if i == 7:
                    self.assertNotIn(h.refs["order"], json.dumps(result.report))
                if i == 11:
                    self.assertEqual(len(result.report["candidates"]), 2)
                if i == 9:
                    self.assertIn("15 分钟", result.customer_message)
                    self.assertIn("50 分", result.customer_message)

    def test_checker_rejects_then_maker_revises(self):
        def inflate(proposal, facts):
            proposal.actions[0].amount_fen += 100
            return proposal

        sim = SimulatorService(self.engine)
        h = build_scenario(sim, "S2")
        handler = BillingDisputeHandler(self.engine, InMemorySaver(), transform=inflate)
        result = handler.start(
            Identity(role="customer", actor_id=h.customer_id),
            h.suggested_question,
            h.suggested_locator,
        )
        result = handler.resume(
            result.run_id, ApprovalDecision(decision="approve", approver="tester")
        )
        self.assertEqual(
            [r["approved"] for r in result.report["reviews"]], [False, True]
        )
        self.assertEqual([p["version"] for p in result.report["proposals"]], [1, 2])

    def test_wallet_question_uses_pending_recharge_before_clarification(self):
        h = build_scenario(SimulatorService(self.engine), "S3")
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        with patch(
            "agent.cases.billing.intake.offline_intake",
            return_value=CaseFile(status="clarify", dispute_type="recharge_missing"),
        ):
            result = handler.start(
                Identity(role="customer", actor_id=h.customer_id),
                h.suggested_question,
                {},
            )
        self.assertEqual(result.status, "pending_evidence")

    def test_deepseek_clarification_only_displays_scoped_masked_candidates(self):
        sim = SimulatorService(self.engine)
        handler = BillingDisputeHandler(self.engine, InMemorySaver())
        for name, status in [("S11", "clarify"), ("S7", "forbidden")]:
            h = build_scenario(sim, name)
            answer = CaseFile(
                status="clarify",
                dispute_type="other",
                candidates=[{"订单": h.refs["order"]}],
            )
            with patch(
                "agent.cases.billing.graph.build_agent_node",
                return_value=lambda payload: answer,
            ):
                result = handler.start(
                    Identity(role="customer", actor_id=h.customer_id),
                    h.suggested_question,
                    h.suggested_locator,
                    planner="deepseek",
                )
            self.assertEqual(result.status, status)
            self.assertNotIn(h.refs["order"], json.dumps(result.report))
            self.assertEqual(
                len(result.report["candidates"]), 2 if name == "S11" else 0
            )
