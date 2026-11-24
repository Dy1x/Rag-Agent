import json
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.investigation.run_agent import run_case
from eval.investigation.run_evaluation import METRICS, run_evaluation, score_report


class InvestigationEvaluationTest(unittest.TestCase):
    def test_offline_batch_passes_six_independently_scored_cases(self):
        result = run_evaluation("offline")
        self.assertEqual(result["planner"], "offline")
        self.assertEqual(result["knowledge_mode"], "simulated")
        self.assertEqual(result["metrics"], {
            metric: {"numerator": 6, "denominator": 6} for metric in METRICS
        })
        by_case = {case["case"]: case["actual"] for case in result["cases"]}
        quiet = by_case["upgraded_no_offline"]
        self.assertEqual((quiet["key_numbers"]["impact"], quiet["key_numbers"]["rollouts"]), ([0, 16], 8))
        empty = by_case["available_no_rollouts"]
        self.assertEqual((empty["firmware_source_status"], empty["key_numbers"]["rollouts"]), ("available", 0))
        self.assertIsNone(by_case["missing"]["key_numbers"]["rollouts"])
        equal = by_case["equal_rates"]
        self.assertEqual(equal["direction"], "undetermined")
        self.assertEqual(equal["key_numbers"]["firmware_cohorts"]["offline_rate_difference"], 0)
        self.assertEqual(equal["key_numbers"]["sites"], {"A": [2, 8], "B": [2, 8]})

    def test_scorer_reports_mismatches_in_all_four_dimensions(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "investigation" / "expected_results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        report = run_case("missing", "offline")["report"]
        report["priority"]["direction"] = "firmware"
        report["impact"]["offline_device_count"] = 100
        report["source_ids"]["alarm_events"] = ["invented-id"]
        report["missing_evidence"] = []

        scored = score_report(report, data["cases"]["missing"], data["device_ids"], "offline")
        self.assertEqual(scored["passed"], dict.fromkeys(METRICS, False))
        self.assertTrue(all(scored["failures"].values()))
        self.assertEqual(scored["actual"]["key_numbers"]["impact"], [100, 16])
        self.assertIn("invented-id", scored["failures"]["traceable_evidence"][0])

    def test_a_failed_case_is_still_in_each_denominator(self):
        def run(name, mode):
            if name == "firmware":
                raise RuntimeError("test failure")
            return run_case(name, mode)

        with patch("eval.investigation.run_evaluation.run_case", side_effect=run):
            result = run_evaluation("offline")

        self.assertEqual(len(result["cases"]), 6)
        failed = result["cases"][0]
        self.assertIsNone(failed["actual"])
        self.assertTrue(all("RuntimeError" in reasons[0] for reasons in failed["failures"].values()))
        self.assertEqual(result["metrics"], {
            metric: {"numerator": 5, "denominator": 6} for metric in METRICS
        })


if __name__ == "__main__":
    unittest.main()
