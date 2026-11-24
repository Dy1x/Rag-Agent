from contextlib import redirect_stdout
from io import StringIO
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from billing_mysql_fixture import MysqlCase
from eval.billing.compare_designs import run_design, summarize
from eval.billing.mutations import ACTION_REQUIRED, mutate
from eval.billing.run_evaluation import evaluate_case
from service.billing.models import Proposal, SoftReviewOpinion


class EvaluationRunnerTests(TestCase):
    def test_action_mutations_leave_an_empty_proposal_unchanged(self):
        proposal = Proposal(
            proposal_id="PRP-test",
            version=1,
            resolution="explained",
            evidence_ids=["example"],
            rationale="test",
            customer_message="已核对。",
        )
        for mutation in ACTION_REQUIRED:
            with self.subTest(mutation=mutation):
                self.assertIs(mutate(mutation, proposal, None), proposal)

    def test_parallel_samples_keep_input_order_and_report_failures(self):
        second_started = Event()

        def evaluate(engine, name, planner, *, design, mutation):
            if name == "S2":
                second_started.set()
                raise RuntimeError("sample failed")
            self.assertTrue(second_started.wait(3))
            return dict(
                name=name, mutation=mutation, passed=True, seconds=0.1, tokens=42
            )

        with (
            patch("eval.billing.compare_designs.evaluate_case", side_effect=evaluate),
            redirect_stdout(StringIO()) as progress,
        ):
            rows = run_design(
                None, [("S1", None), ("S2", "M_WRONG_TARGET")], "offline", "C", 2
            )
        self.assertEqual([row["name"] for row in rows], ["S1", "S2"])
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[1]["errors"], ["RuntimeError: sample failed"])
        self.assertIn("注入状态未记录", rows[1]["not_injected_reason"])
        self.assertEqual(len(progress.getvalue().splitlines()), 2)
        self.assertEqual(summarize(rows)["avg_tokens"], 21)


class EvaluationTests(MysqlCase):
    def test_missing_unfreeze_is_rejected_even_when_proposal_explains_it(self):
        row = evaluate_case(
            self.engine, "S1", "offline", design="C", mutation="M_MISSED_REMEDY"
        )
        first, corrected = row["report"]["proposals"]
        self.assertEqual(first["resolution"], "explained")
        self.assertEqual(first["actions"], [])
        self.assertIn("冻结尚未解除", first["customer_message"])
        self.assertTrue(row["first_rejected"])
        self.assertIn(
            "S_MISSED_REMEDY",
            [issue["code"] for issue in row["report"]["reviews"][0]["issues"]],
        )
        self.assertEqual(corrected["actions"][0]["action_type"], "unfreeze")
        self.assertTrue(row["passed"])

    def test_only_c_has_a_deepseek_customer_message_guard(self):
        def simulated_model(spec, planner, tools, offline, **kwargs):
            if spec.output_schema is SoftReviewOpinion:
                return lambda payload: SoftReviewOpinion(approved=True)
            return lambda payload: offline(payload, tools)

        with patch(
            "agent.cases.billing.graph.build_agent_node", side_effect=simulated_model
        ):
            rows = [
                evaluate_case(
                    self.engine,
                    "S2",
                    "deepseek",
                    design=design,
                    mutation="M_OVERPROMISE",
                )
                for design in "ABC"
            ]
        self.assertEqual([row["first_rejected"] for row in rows], [False, False, True])
        self.assertEqual(
            rows[2]["report"]["reviews"][0]["issues"][0]["code"], "S_OVERPROMISE"
        )

    def test_first_review_and_final_execution_are_independent_metrics(self):
        rows = [
            evaluate_case(
                self.engine, "S2", "offline", design=d, mutation="M_OVERPROMISE"
            )
            for d in "BC"
        ]
        self.assertFalse(rows[0]["first_rejected"])
        self.assertTrue(rows[0]["correct_outcome"])
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(rows[1]["first_rejected"])
        self.assertTrue(rows[1]["passed"])
        clean = evaluate_case(self.engine, "S9", "offline")
        metrics = summarize([clean, *rows])
        self.assertEqual(metrics["mutation_interception"], dict(n=1, d=2, rate=0.5))
        self.assertEqual(metrics["false_interception"], dict(n=0, d=1, rate=0))

    def test_single_agent_and_independent_review_use_different_inputs(self):
        for design in ("A", "B", "C"):
            row = evaluate_case(
                self.engine, "S2", "offline", design=design, mutation="M_FAKE_EVIDENCE"
            )
            self.assertEqual(row["first_rejected"], design != "A")
            self.assertTrue(row["correct_outcome"])
