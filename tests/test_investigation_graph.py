import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent.investigation_graph import InvestigationDependencies, KnowledgeResult, create_investigation_graph
from agent.investigation_planner import plan_offline, plan_with_deepseek
from eval.investigation.run_agent import (
    DEFAULT_QUESTION, lookup_known_issue_offline, lookup_known_issue_with_rag,
)
from eval.investigation.run_baseline import create_fixture_db
from service.investigation_data_service import (
    InvestigationScope, compare_site_devices, get_alarm_timeline, get_firmware_rollouts,
)


def decision(action, target_version=None):
    return {"action": action, "reason": "测试规划理由。", "target_version": target_version}


def path(state):
    return [entry["action"] for entry in state["execution_trace"]]


INITIAL = ["get_alarm_timeline", "get_firmware_rollouts"]


class InvestigationGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).resolve().parents[1] / "eval" / "investigation" / "cases.json"
        cls.fixture = json.loads(fixture.read_text(encoding="utf-8"))

    def run_graph(self, name="firmware", *, case=None, **overrides):
        case = self.fixture["cases"][name] if case is None else case
        scope = InvestigationScope(**case["scope"])
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.sqlite3"
            create_fixture_db(db_path, self.fixture["devices"], case)
            options = {
                "planner": plan_offline, "lookup_known_issue": lookup_known_issue_offline,
                **overrides,
            }
            graph = create_investigation_graph(InvestigationDependencies(db_path, **options))
            return graph.invoke({"run_id": "test-run", "question": DEFAULT_QUESTION, "scope": scope.to_dict()})

    def test_same_question_follows_three_evidence_driven_paths(self):
        firmware, site, missing = [self.run_graph(name) for name in ("firmware", "site", "missing")]
        self.assertEqual(path(firmware), INITIAL + ["compare_firmware_cohorts", "lookup_known_issue", "finish"])
        self.assertEqual(path(site), INITIAL + ["compare_site_devices", "finish"])
        self.assertEqual(path(missing), INITIAL + ["compare_site_devices", "lookup_known_issue", "finish"])
        for state in (firmware, site, missing):
            self.assertEqual(state["question"], DEFAULT_QUESTION)
            self.assertEqual(state["planning_rounds"], 2)
            self.assertEqual(json.loads(json.dumps(state)), state)
            self.assertTrue(all(entry["reason"] for entry in state["execution_trace"]))

        report = firmware["report"]
        self.assertEqual(report["priority"]["direction"], "firmware")
        self.assertEqual(report["impact"]["offline_device_count"], 7)
        self.assertEqual(report["impact"]["offline_event_count"], 8)
        comparison = report["firmware_comparison"]
        self.assertEqual(comparison["target_version"], "2.4.0")
        self.assertEqual(
            [(group["offline_device_count"], group["device_count"])
             for group in (comparison["target_group"], comparison["control_group"])],
            [(6, 8), (1, 8)],
        )
        self.assertIn("fw-alarm-04", report["source_ids"]["alarm_events"])
        self.assertIn("fw-rollout-B01", report["source_ids"]["firmware_rollouts"])
        self.assertTrue(report["knowledge_base"]["simulated"])
        self.assertEqual(report["source_ids"]["knowledge_base"][0]["id"], "SIM-KB-01")

        report = site["report"]
        self.assertEqual(report["priority"]["direction"], "site")
        self.assertEqual(
            [(item["site_id"], item["offline_device_count"], item["device_count"])
             for item in report["site_comparison"]["sites"]],
            [("A", 6, 8), ("B", 0, 8)],
        )
        self.assertEqual(site["followups_remaining"], 1)
        self.assertIsNone(report["knowledge_base"])

        report = missing["report"]
        self.assertEqual(report["firmware_source"]["status"], "unavailable")
        self.assertEqual(report["impact"]["offline_device_count"], 5)
        self.assertIn("firmware_rollouts", [item["source"] for item in report["missing_evidence"]])
        self.assertIsNone(report["firmware_comparison"])

    def test_initial_queries_are_parallel_and_plan_waits_for_both(self):
        barrier = Barrier(2)
        alarms_done, firmware_done = Event(), Event()
        plan_inputs = []

        def read_in_parallel(query, completed):
            def run(db_path, scope):
                barrier.wait(timeout=5)
                result = query(db_path, scope)
                completed.set()
                return result
            return run

        def observe(request):
            plan_inputs.append((alarms_done.is_set(), firmware_done.is_set(), request))
            return decision("finish")

        state = self.run_graph(
            planner=observe,
            alarm_timeline=read_in_parallel(get_alarm_timeline, alarms_done),
            firmware_rollouts=read_in_parallel(get_firmware_rollouts, firmware_done),
        )
        self.assertEqual(path(state), INITIAL + ["finish"])
        self.assertEqual(len(plan_inputs), 1)
        alarms_finished, firmware_finished, request = plan_inputs[0]
        self.assertTrue(alarms_finished and firmware_finished)
        self.assertEqual(set(request), {"question", "scope", "evidence"})
        self.assertEqual(set(request["evidence"]), {"alarm_timeline", "firmware_rollouts"})
        self.assertEqual(request["evidence"]["alarm_timeline"]["data"]["offline_device_count"], 7)
        self.assertEqual(request["evidence"]["firmware_rollouts"]["data"]["rollout_count"], 8)
        payload = json.dumps(request)
        self.assertNotIn("fw-alarm-", payload)
        self.assertNotIn("fw-rollout-", payload)
        self.assertIn("fw-alarm-01", state["report"]["source_ids"]["alarm_events"])

    def test_primary_source_failure_keeps_the_other_evidence(self):
        reader = Mock(side_effect=RuntimeError("告警端暂不可读"))
        state = self.run_graph(alarm_timeline=reader)
        self.assertEqual(state["alarm_timeline"]["status"], "failed")
        self.assertEqual(state["firmware_rollouts"]["status"], "available")
        self.assertEqual(state["report"]["firmware_rollout_count"], 8)
        self.assertEqual(state["report"]["impact"]["offline_device_count"], 7)
        self.assertIn("告警端暂不可读", state["missing_evidence"][0]["reason"])

        state = self.run_graph(firmware_rollouts=Mock(side_effect=RuntimeError("升级端连接断开")))
        self.assertEqual(state["firmware_rollouts"]["status"], "failed")
        self.assertEqual(state["alarm_timeline"]["status"], "available")
        self.assertEqual(state["report"]["impact"]["offline_device_count"], 7)
        self.assertIn("升级端连接断开", state["report"]["firmware_source"]["reason"])

    def test_bad_plan_or_planner_error_falls_back_to_site(self):
        for response, expected in (
            ("broken JSON", "结构校验失败"),
            (decision("delete_device"), "结构校验失败"),
            (decision("compare_firmware_cohorts", "unknown"), "候选版本"),
            (decision("lookup_known_issue"), "依赖不可用"),
            (RuntimeError("模拟模型异常"), "模拟模型异常"),
        ):
            with self.subTest(response=response):
                planner = Mock(side_effect=[response, decision("finish")])
                state = self.run_graph(planner=planner, lookup_known_issue=None)
                self.assertEqual(path(state), INITIAL + ["compare_site_devices", "finish"])
                self.assertIn(expected, state["execution_trace"][2]["planner_error"])
                self.assertEqual(state["report"]["impact"]["offline_device_count"], 7)
                self.assertEqual(state["report"]["firmware_rollout_count"], 8)

    def test_firmware_comparison_requires_available_source_and_candidate(self):
        for status, expected in (("unavailable", "固件来源不可用"), ("available", "没有可选的目标版本")):
            with self.subTest(status=status):
                case = deepcopy(self.fixture["cases"]["missing"])
                case["scope"]["firmware_source"] = status
                if status == "available":
                    case["scope"].pop("firmware_source_reason")
                planner = Mock(side_effect=[decision("compare_firmware_cohorts", "2.4.0"), decision("finish")])
                compare = Mock()
                state = self.run_graph(case=case, planner=planner, firmware_compare=compare)
                compare.assert_not_called()
                self.assertEqual(path(state), INITIAL + ["compare_site_devices", "finish"])
                self.assertIn(expected, state["execution_trace"][2]["planner_error"])
                self.assertEqual(state["report"]["firmware_source"]["status"], status)

    def test_repeated_actions_are_blocked_even_when_budget_remains(self):
        compare = Mock(wraps=compare_site_devices)
        planner = Mock(side_effect=[decision("compare_site_devices"), decision("compare_site_devices")])
        site = self.run_graph(planner=planner, site_compare=compare)
        compare.assert_called_once()
        self.assertEqual(path(site), INITIAL + ["compare_site_devices", "finish"])
        self.assertIn("已尝试过", site["execution_trace"][-1]["planner_error"])
        self.assertEqual(site["followups_remaining"], 1)

        lookup = Mock(wraps=lookup_known_issue_offline)
        planner = Mock(side_effect=[decision("lookup_known_issue"), decision("lookup_known_issue")])
        knowledge = self.run_graph(planner=planner, lookup_known_issue=lookup)
        lookup.assert_called_once()
        self.assertEqual(path(knowledge), INITIAL + ["lookup_known_issue", "compare_site_devices", "finish"])
        self.assertIn("已尝试过", knowledge["execution_trace"][3]["planner_error"])

    def test_two_action_budget_and_one_replan_are_hard_limits(self):
        planner = Mock(side_effect=[
            decision("compare_site_devices"), decision("lookup_known_issue"),
            decision("compare_firmware_cohorts", "2.4.0"),
        ])
        state = self.run_graph(planner=planner)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(state["planning_rounds"], 2)
        self.assertEqual(state["followups_remaining"], 0)
        self.assertEqual(path(state), INITIAL + ["compare_site_devices", "lookup_known_issue", "finish"])
        self.assertIn("动作上限", state["execution_trace"][-1]["reason"])

    def test_missing_or_failed_rag_does_not_discard_data_results(self):
        for status, lookup in (
            ("insufficient", Mock(return_value=KnowledgeResult(status="insufficient", text="缺少版本文档。"))),
            ("clarify", Mock(return_value=KnowledgeResult(status="clarify", text="请补充设备型号。"))),
            ("failed", Mock(side_effect=RuntimeError("知识库超时"))),
        ):
            with self.subTest(status=status):
                state = self.run_graph(lookup_known_issue=lookup)
                lookup.assert_called_once()
                self.assertIn("2.4.0", lookup.call_args.args[0])
                self.assertEqual(path(state)[-2:], ["lookup_known_issue", "finish"])
                self.assertEqual(state["report"]["knowledge_base"]["status"], status)
                self.assertEqual(state["report"]["firmware_comparison"]["target_group"]["offline_device_count"], 6)
                gap = next(item for item in state["missing_evidence"] if item["source"] == "known_issue")
                self.assertEqual(gap["status"], status)
                self.assertEqual(gap["reason"], state["report"]["knowledge_base"]["text"])

    def test_deepseek_adapter_requests_json_and_sends_only_the_planning_input(self):
        request = {"question": DEFAULT_QUESTION, "scope": {"site_ids": ["A"]}, "evidence": {}}
        respond = Mock(return_value=AIMessage(content=json.dumps(decision("compare_site_devices"))))
        model = Mock()
        model.bind.return_value = RunnableLambda(respond)
        factory = SimpleNamespace(get_chat_model=Mock(return_value=model))
        with patch.dict(sys.modules, {"model.factory": factory}):
            plan = plan_with_deepseek(request)

        self.assertEqual(plan.action, "compare_site_devices")
        self.assertEqual(model.bind.call_args.kwargs["response_format"], {"type": "json_object"})
        messages = respond.call_args.args[0]
        self.assertEqual(json.loads(messages[1].content), request)

    def test_rag_adapter_preserves_rag_status_and_citations(self):
        source = [{"id": "E1", "source": "故障记录.txt"}]
        reply = SimpleNamespace(status="answered", text="按日志核验【E1】。", sources=source)
        service = Mock()
        service.answer_question.return_value = reply
        module = SimpleNamespace(RagSummerizeService=Mock(return_value=service))
        with patch.dict(sys.modules, {"rag.rag_service": module}):
            knowledge = lookup_known_issue_with_rag("查询掉线线索")

        service.answer_question.assert_called_once_with("查询掉线线索")
        self.assertEqual((knowledge.status, knowledge.text, knowledge.sources), ("answered", reply.text, source))
        self.assertFalse(knowledge.simulated)


if __name__ == "__main__":
    unittest.main()
