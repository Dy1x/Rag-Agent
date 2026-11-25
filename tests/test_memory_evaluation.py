from billing_mysql_fixture import MysqlCase
from eval.memory.run_memory_eval import compare, deepseek_passes, markdown, offline_passes


class MemoryEvaluationTests(MysqlCase):
    def test_offline_ablation_runs_all_scenarios(self):
        results = compare("offline", self.engine)
        self.assertTrue(offline_passes(results), results)
        self.assertEqual(sum(results["M0"]["MEM-1"].values()), 0)
        self.assertEqual(sum(results["M1"]["MEM-1"].values()), 3)
        self.assertFalse(results["M1"]["MEM-3"]["live_status"])
        self.assertIsNone(results["M2"]["usage"]["prompt_tokens"])
        for group in results.values():
            self.assertEqual(group["summary_outcomes"], {"success": 0, "fallback": 0})
            self.assertEqual(group["evidence_inputs"], {"outage": True, "rag": True})
        self.assertIn("| M2 | 3/3 | 4/4 | 1/1 | 2/2 | 2/2 | 1/1 |", markdown(results, "offline"))
        self.assertFalse(deepseek_passes(results))
        for design in ("M1", "M2"):
            results[design]["summary_outcomes"]["success"] = 12
        self.assertTrue(deepseek_passes(results))
        results["M2"]["summary_outcomes"]["fallback"] = 1
        self.assertFalse(deepseek_passes(results))
        results["M2"]["summary_outcomes"]["fallback"] = 0
        results["M2"]["evidence_inputs"]["outage"] = False
        self.assertFalse(deepseek_passes(results))
