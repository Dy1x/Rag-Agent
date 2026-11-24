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
from langsmith import tracing_context

from agent.investigation_specialists import (
    SpecialistDependencies, plan_specialist_offline, plan_specialist_with_deepseek,
)
from agent.multi_agent_investigation_graph import create_multi_agent_investigation_graph
from eval.investigation.run_agent import DEFAULT_QUESTION, lookup_known_issue_offline
from eval.investigation.run_baseline import create_fixture_db, load_case
from service.investigation_data_service import (
    InvestigationScope,
    compare_firmware_cohorts,
    compare_firmware_cohorts_by_site,
    compare_site_devices,
    get_site_event_timeline,
)


def actions(result):
    return [call["action"] for call in result["tool_calls"]]


def counts(group):
    return group["offline_device_count"], group["device_count"], group["offline_rate"]


class MultiAgentInvestigationTest(unittest.TestCase):
    def investigate(self, name="rollout_site_overlap", *, site_event_rows=None, scope=None, **overrides):
        devices, case = load_case(name)
        if site_event_rows is not None:
            case["site_events"] = site_event_rows
        if scope is None:
            scope = InvestigationScope(**case["scope"]).to_dict()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "investigation.sqlite3"
            create_fixture_db(path, devices, case)
            options = {
                "planner": plan_specialist_offline,
                "lookup_known_issue": lookup_known_issue_offline,
                **overrides,
            }
            graph = create_multi_agent_investigation_graph(SpecialistDependencies(path, **options))
            with tracing_context(enabled=False):
                return graph.invoke({"run_id": "test-run", "question": DEFAULT_QUESTION, "scope": scope})

    def test_parallel_specialists_join_and_review_with_actual_peer_evidence(self):
        barrier = Barrier(2)
        firmware_done, site_done = Event(), Event()
        reviews = []
        by_site = Mock(wraps=compare_firmware_cohorts_by_site)

        def parallel(query):
            def run(*args):
                barrier.wait(timeout=5)
                return query(*args)
            return run

        def read_knowledge(query):
            result = lookup_known_issue_offline(query)
            firmware_done.set()
            return result

        def read_site_events(*args):
            result = get_site_event_timeline(*args)
            site_done.set()
            return result

        def plan(role, request):
            if request["review_request"]:
                reviews.append((role, firmware_done.is_set(), site_done.is_set(), request))
            return plan_specialist_offline(role, request)

        state = self.investigate(
            planner=plan, firmware_compare=parallel(compare_firmware_cohorts),
            site_compare=parallel(compare_site_devices), lookup_known_issue=read_knowledge,
            site_events=read_site_events, firmware_compare_by_site=by_site,
        )
        self.assertEqual(actions(state["firmware_initial"]), ["compare_firmware_cohorts", "lookup_known_issue"])
        self.assertEqual(actions(state["site_initial"]), ["compare_site_devices", "get_site_event_timeline"])
        self.assertEqual([row["action"] for row in state["coordination_trace"]], [
            "initial_findings_joined", "request_review", "review_finished",
        ])
        self.assertEqual(len(reviews), 1)
        role, finished_firmware, finished_site, ctx = reviews[0]
        self.assertEqual(role, "firmware")
        self.assertTrue(finished_firmware and finished_site)
        request = state["review_request"]
        self.assertEqual(request, ctx["review_request"])
        self.assertEqual((request["triggered_by"], request["responsible_role"]), ("site", "firmware"))
        self.assertEqual(request["original_judgment"], "overall_association")
        self.assertEqual(request["trigger_evidence_ids"], ["gw-A-01", "overlap-alarm-A01"])
        self.assertEqual(ctx["peer_finding"], state["site_initial"]["finding"])
        self.assertEqual(ctx["prior_finding"], state["firmware_initial"]["finding"])
        self.assertIn("gw-A-01", ctx["peer_finding"]["source_ids"]["site_events"])
        self.assertEqual(ctx["remaining_tools"], 1)

        by_site.assert_called_once()
        self.assertEqual(by_site.call_args.args[1].site_ids, ("A", "B"))
        self.assertEqual(by_site.call_args.args[2], "2.4.0")
        self.assertEqual(actions(state["review_result"]), ["compare_firmware_cohorts_by_site"])
        self.assertEqual(state["review_result"]["tool_calls"][0]["review_request_id"], request["request_id"])
        self.assertIn("overlap-rollout-A01", state["review_result"]["tool_calls"][0]["source_ids"]["firmware_rollouts"])

        report = state["report"]
        self.assertEqual(report["initial_judgments"], {"firmware": "overall_association", "site": "gateway_clue"})
        self.assertEqual(report["review_judgment"], "no_within_site_difference")
        self.assertEqual(counts(report["firmware_overall"]["target_group"]), (9, 15, 0.6))
        self.assertEqual(counts(report["firmware_overall"]["control_group"]), (6, 15, 0.4))
        a, b = report["firmware_by_site"]["sites"]
        self.assertEqual((counts(a["target_group"]), counts(a["control_group"])), ((8, 10, 0.8), (4, 5, 0.8)))
        self.assertEqual((counts(b["target_group"]), counts(b["control_group"])), ((1, 5, 0.2), (2, 10, 0.2)))
        self.assertEqual(report["gateway_alignment"]["lead_seconds"], 120.0)
        self.assertEqual((report["priority"]["direction"], report["priority"]["site_id"]), ("site", "A"))
        self.assertIn("归因仍需", report["interpretation"])
        self.assertEqual(report["source_ids"]["site_events"], ["gw-A-01"])
        self.assertEqual(report["source_ids"]["knowledge_base"], ["SIM-KB-01"])
        json.dumps(state)

    def test_missing_knowledge_keeps_database_evidence_and_allows_review(self):
        for lookup, status in ((None, "unavailable"), (Mock(side_effect=RuntimeError("知识库超时")), "failed")):
            with self.subTest(status=status):
                state = self.investigate(lookup_known_issue=lookup)
                self.assertEqual(state["firmware_initial"]["finding"]["key_data"]["knowledge"]["status"], status)
                self.assertEqual(state["report"]["review_judgment"], "no_within_site_difference")
                gap = next(gap for gap in state["report"]["missing_evidence"] if gap["source"] == "lookup_known_issue")
                self.assertEqual(gap["status"], status)
                self.assertTrue(gap["reason"])

    def test_missing_gateway_evidence_closes_without_review(self):
        comparison = Mock(wraps=compare_firmware_cohorts_by_site)
        state = self.investigate(site_event_rows=[], firmware_compare_by_site=comparison)
        self.assertIsNone(state["review_request"])
        self.assertNotIn("review_result", state)
        self.assertEqual(state["reviews_remaining"], 1)
        comparison.assert_not_called()
        self.assertEqual(state["report"]["priority"]["direction"], "firmware")
        self.assertEqual(counts(state["report"]["firmware_overall"]["target_group"]), (9, 15, 0.6))
        self.assertIn("gateway_alignment", [item["source"] for item in state["report"]["missing_evidence"]])

    def test_site_agent_stops_when_queried_sites_have_no_offline_devices(self):
        events = Mock()
        state = self.investigate("upgraded_no_offline", site_events=events)
        events.assert_not_called()
        self.assertEqual(actions(state["site_initial"]), ["compare_site_devices"])
        self.assertIn("未发现掉线", state["site_initial"]["decisions"][-1]["reason"])
        self.assertIsNone(state["review_request"])
        self.assertEqual(state["report"]["impact"]["offline_device_count"], 0)

    def test_late_gateway_or_gateway_without_provenance_cannot_trigger_review(self):
        _, case = load_case("rollout_site_overlap")
        late = deepcopy(case["site_events"])
        late[0]["occurred_at"] = "2026-11-21T14:05:00Z"
        state = self.investigate(site_event_rows=late)
        self.assertIsNone(state["review_request"])
        self.assertIsNone(state["report"]["gateway_alignment"])

        def unproven(*args):
            timeline = get_site_event_timeline(*args)
            timeline["source_ids"]["site_events"] = []
            return timeline

        state = self.investigate(site_events=unproven)
        self.assertIsNone(state["review_request"])
        self.assertIsNone(state["report"]["gateway_alignment"])

    def test_one_specialist_query_failure_retains_the_other_finding(self):
        state = self.investigate(firmware_compare=Mock(side_effect=RuntimeError("固件查询失败")))
        self.assertEqual(state["firmware_initial"]["tool_calls"][0]["status"], "failed")
        self.assertIsNone(state["review_request"])
        self.assertEqual(state["report"]["initial_judgments"]["site"], "gateway_clue")
        self.assertEqual(state["report"]["priority"]["site_id"], "A")
        self.assertIn("gw-A-01", state["report"]["source_ids"]["site_events"])
        self.assertIn("compare_firmware_cohorts", [gap["source"] for gap in state["report"]["missing_evidence"]])

        state = self.investigate(site_events=Mock(side_effect=RuntimeError("站点事件查询失败")))
        self.assertIsNone(state["review_request"])
        self.assertEqual(state["report"]["priority"]["direction"], "firmware")
        self.assertEqual(counts(state["report"]["firmware_overall"]["control_group"]), (6, 15, 0.4))
        self.assertIn("get_site_event_timeline", [gap["source"] for gap in state["report"]["missing_evidence"]])

    def test_budgets_limit_initial_tools_and_one_review(self):
        planner = Mock(side_effect=plan_specialist_offline)
        by_site = Mock(wraps=compare_firmware_cohorts_by_site)
        state = self.investigate(planner=planner, firmware_compare_by_site=by_site)
        self.assertEqual(len(state["firmware_initial"]["tool_calls"]), 2)
        self.assertEqual(len(state["site_initial"]["tool_calls"]), 2)
        self.assertEqual(state["firmware_initial"]["remaining_tools"], 0)
        self.assertEqual(state["site_initial"]["remaining_tools"], 0)
        self.assertEqual(len(state["review_result"]["tool_calls"]), 1)
        self.assertEqual(state["review_result"]["remaining_tools"], 0)
        self.assertEqual(state["reviews_remaining"], 0)
        self.assertEqual(sum(entry["action"] == "request_review" for entry in state["coordination_trace"]), 1)
        self.assertEqual(sum(call.args[1]["review_request"] is not None for call in planner.call_args_list), 1)
        by_site.assert_called_once()

    def test_failed_review_preserves_both_initial_findings(self):
        review = Mock(side_effect=RuntimeError("分站查询失败"))
        state = self.investigate(firmware_compare_by_site=review)
        review.assert_called_once()
        self.assertEqual(state["review_result"]["tool_calls"][0]["status"], "failed")
        self.assertEqual(state["report"]["initial_judgments"], {"firmware": "overall_association", "site": "gateway_clue"})
        self.assertEqual(state["report"]["review_judgment"], "insufficient")
        self.assertEqual(state["report"]["source_ids"]["site_events"], ["gw-A-01"])
        self.assertTrue(any("分站查询失败" in item["reason"] for item in state["report"]["missing_evidence"]))

    def test_role_actions_candidate_versions_and_repeat_are_validated(self):
        def plan(role, request):
            if role == "site":
                return {"action": "compare_firmware_cohorts", "target_version": "2.4.0", "reason": "越权测试"}
            return {"action": "compare_firmware_cohorts", "target_version": "invented-version", "reason": "版本测试"}

        state = self.investigate(planner=plan)
        self.assertEqual(actions(state["firmware_initial"]), [])
        self.assertEqual(actions(state["site_initial"]), [])
        self.assertIn("目标版本", state["firmware_initial"]["decisions"][0]["planner_error"])
        self.assertIn("角色权限", state["site_initial"]["decisions"][0]["planner_error"])
        self.assertEqual(state["report"]["impact"]["offline_device_count"], 15)

        def repeat(role, request):
            if role == "site":
                return {"action": "compare_site_devices", "reason": "重复测试"}
            return {"action": "finish", "reason": "结束测试"}

        state = self.investigate(planner=repeat)
        self.assertEqual(actions(state["site_initial"]), ["compare_site_devices"])
        self.assertIn("已执行", state["site_initial"]["decisions"][-1]["planner_error"])

    def test_invalid_scope_records_gap_without_dispatching_tools(self):
        query = Mock()
        state = self.investigate(scope={}, alarm_timeline=query, firmware_rollouts=query)
        query.assert_not_called()
        self.assertNotIn("firmware_initial", state)
        self.assertNotIn("site_initial", state)
        self.assertEqual(state["report"]["missing_evidence"][0]["source"], "scope")
        self.assertEqual(state["report"]["priority"]["direction"], "undetermined")

    def test_role_prompts_and_json_validation_use_mocked_deepseek(self):
        from langchain_core.messages import SystemMessage

        reply = Mock(side_effect=[
            AIMessage(content='{"action":"compare_firmware_cohorts","reason":"test","target_version":"2.4.0"}'),
            AIMessage(content='{"action":"get_site_event_timeline","reason":"test","target_version":null}'),
        ])
        model = Mock()
        model.bind.return_value = RunnableLambda(reply)
        factory = SimpleNamespace(get_chat_model=Mock(return_value=model))
        with patch.dict(sys.modules, {"model.factory": factory}):
            firmware = plan_specialist_with_deepseek("firmware", {"candidate_versions": ["2.4.0"]})
            site = plan_specialist_with_deepseek("site", {"candidate_versions": []})

        self.assertEqual(firmware.target_version, "2.4.0")
        self.assertEqual(site.action, "get_site_event_timeline")
        self.assertEqual(model.bind.call_count, 2)
        self.assertEqual(model.bind.call_args.kwargs["response_format"], {"type": "json_object"})
        self.assertIsInstance(reply.call_args_list[0].args[0][0], SystemMessage)
        self.assertIn("固件调查员", reply.call_args_list[0].args[0][0].content)
        self.assertIn("站点调查员", reply.call_args_list[1].args[0][0].content)

    def test_deepseek_early_finish_cannot_skip_necessary_site_event_query(self):
        import json

        response = Mock(return_value=AIMessage(content='{"action":"finish","reason":"提前结束"}'))
        model = Mock()
        model.bind.return_value = RunnableLambda(response)

        def planner(role, request):
            if role == "site":
                return plan_specialist_with_deepseek(role, request)
            return plan_specialist_offline(role, request)

        with patch.dict(sys.modules, {"model.factory": SimpleNamespace(get_chat_model=lambda: model)}):
            state = self.investigate(planner=planner)

        self.assertEqual(actions(state["site_initial"]), ["compare_site_devices", "get_site_event_timeline"])
        self.assertIn("gw-A-01", state["review_request"]["trigger_evidence_ids"])
        payload = json.loads(response.call_args.args[0][-1].content)
        self.assertEqual(payload["required_action"], "get_site_event_timeline")
        self.assertNotIn("compare_site_devices", payload["allowed_actions"])


if __name__ == "__main__":
    unittest.main()
