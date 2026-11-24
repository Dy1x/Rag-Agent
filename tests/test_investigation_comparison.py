import unittest

from eval.investigation.compare_investigation_agents import compare
from eval.investigation.run_evaluation import METRICS


class InvestigationComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = compare("offline")
        cls.cases = {entry["case"]: entry for entry in cls.result["cases"]}

    def test_overlap_both_designs_can_get_and_use_site_evidence(self):
        overlap = self.cases["rollout_site_overlap"]
        for design in ("single", "multi"):
            with self.subTest(design=design):
                side = overlap[design]
                self.assertTrue(all(side["passed"].values()), side["failures"])
                self.assertEqual(side["actual"]["priority_site_id"], "A")
                counts = side["actual"]["key_numbers"]
                self.assertEqual(counts["impact"], [15, 30])
                self.assertEqual(counts["gateway"]["lead_seconds"], 120)
                self.assertEqual(counts["firmware_by_site"]["A"]["target"], [8, 10])
                self.assertEqual(counts["firmware_by_site"]["A"]["control"], [4, 5])
                self.assertIn("gw-A-01", side["actual"]["source_ids"]["site_events"])
                self.assertEqual(
                    [call["action"] for call in side["review"]["tool_calls"]], ["compare_firmware_cohorts_by_site"],
                )
        self.assertIsNone(overlap["single"]["review"]["request"])
        self.assertIn("gw-A-01", overlap["multi"]["review"]["request"]["trigger_evidence_ids"])

    def test_budget_and_metric_denominators_report_actual_results(self):
        self.assertEqual(self.result["planner"], "offline")
        self.assertEqual(self.result["knowledge_mode"], "simulated")
        self.assertTrue(self.result["simulated_data"])
        budget = self.result["budgets"]
        self.assertEqual(budget["single_followups"], budget["multi_followups"])
        for design in ("single", "multi"):
            sides = [case[design] for case in self.cases.values()]
            for side in sides:
                calls = [entry["action"] for entry in side["tool_calls"]]
                self.assertEqual(len(calls), side["tool_call_count"])
                self.assertEqual(len(calls), len(set(calls)))
                self.assertLessEqual(side["followup_call_count"], budget[f"{design}_followups"])
                self.assertLessEqual(side["knowledge_call_count"], budget["knowledge_calls_per_run"])
                self.assertGreaterEqual(side["elapsed_seconds"], 0)
            for metric in METRICS:
                self.assertEqual(self.result["metrics"][design][metric], {
                    "numerator": sum(side["passed"][metric] for side in sides), "denominator": 3,
                })
        for design in ("single", "multi"):
            stats = self.cases["firmware"][design]["actual"]["key_numbers"]["firmware_cohorts"]
            self.assertEqual((stats["target"], stats["control"]), ([6, 8], [1, 8]))
        missing = self.cases["missing"]["multi"]
        self.assertEqual(missing["actual"]["firmware_source_status"], "unavailable")
        self.assertEqual(missing["actual"]["key_numbers"]["sites"], {"A": [3, 8], "B": [2, 8]})
        self.assertFalse(missing["passed"]["investigation_direction"])
        self.assertIn("undetermined", missing["failures"]["investigation_direction"][0])
