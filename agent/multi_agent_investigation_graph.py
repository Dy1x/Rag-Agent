"""Parallel specialist investigation and a single cross-role review."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.investigation_graph import InvestigationInput, _failure_reason, _read_source
from agent.investigation_specialists import (
    ReviewRequest,
    Role,
    SpecialistDependencies,
    align_gateway_to_offline,
    collect_source_ids,
    create_firmware_agent,
    create_site_agent,
)
from service.investigation_data_service import InvestigationScope


class MultiAgentState(InvestigationInput, total=False):
    scope_error: str
    shared_evidence: dict
    firmware_initial: dict
    site_initial: dict
    review_request: dict | None
    review_result: dict
    reviews_remaining: int
    coordination_trace: list[dict]
    report: dict


def _summary_rate(cohort: dict) -> str:
    rate = cohort["offline_rate"]
    percentage = f"（{rate:.0%}）" if rate is not None else "（无可比较分母）"
    return f"{cohort['offline_device_count']}/{cohort['device_count']}{percentage}"


def _report(state: MultiAgentState) -> dict:
    if state.get("scope_error"):
        return {
            "priority": {"direction": "undetermined", "reason": "请先补齐有效的站点和 UTC 调查窗口。"},
            "missing_evidence": [{"source": "scope", "status": "invalid", "reason": state["scope_error"]}],
            "source_ids": {},
            "interpretation": "故障归因待验证。",
        }

    firmware = state["firmware_initial"]["finding"]
    site = state["site_initial"]["finding"]
    revised = state.get("review_result", {}).get("finding")
    review = state.get("review_request")
    overall = firmware["key_data"]["overall"]
    # Some planners can request a by-site comparison before coordination.
    by_site = (revised["key_data"]["by_site"] if revised else None) or firmware["key_data"]["by_site"]
    alignment = site["key_data"]["gateway_alignment"]
    alarm = state["shared_evidence"]["alarm_timeline"]
    alarm_data = alarm["data"] if alarm["status"] == "available" else {}
    impact = {key: alarm_data[key] for key in (
        "device_count", "offline_device_count", "offline_device_ids", "offline_event_count",
    )} if alarm_data else None
    rollout = state["shared_evidence"]["firmware_rollouts"]
    rollout_count = rollout["data"]["rollout_count"] if rollout["status"] == "available" else None
    if revised and revised["judgment"] == "no_within_site_difference" and alignment:
        direction = "site"
        reason = (
            f"总体固件差异在分站后消失，优先核验 {alignment['site_id']} 站网关事件 "
            f"{alignment['site_event_id']}。"
        )
    elif revised and revised["judgment"] == "within_site_comparison" and alignment:
        direction = "undetermined"
        reason = "分站对比与网关线索仍需结合更多证据，分别核查版本记录与站点日志。"
    elif alignment:
        direction = "site"
        reason = f"优先核验 {alignment['site_id']} 站网关和设备日志，固件关联仍需复核。"
    elif firmware["judgment"] == "overall_association":
        direction = "firmware"
        reason = "总体固件分组有差异，先核对升级记录和设备诊断，并补齐站点证据。"
    else:
        direction = "undetermined"
        reason = "保留已取得的统计，补齐能够区分固件和站点因素的证据。"

    findings = [firmware, site, *([revised] if revised else [])]
    gaps = {
        (item["source"], item["status"], item["reason"]): item
        for finding in findings for item in finding["missing_evidence"]
    }
    if impact and impact["offline_device_count"]:
        gaps["site_network_power_logs", "not_collected"] = {
            "source": "site_network_power_logs", "status": "not_collected",
            "reason": "还需同一时间窗口的网络、供电和完整网关日志。",
        }
        if direction in ("firmware", "undetermined"):
            gaps["device_diagnostics", "not_collected"] = {
                "source": "device_diagnostics", "status": "not_collected",
                "reason": "还需设备侧升级与掉线日志、版本复测或回滚结果。",
            }
    interpretation = "以上结果属于观测关联，故障归因仍需结合设备、网关及网络日志验证。"
    lines = [reason]
    if impact:
        lines.insert(0, f"调查范围内 {impact['offline_device_count']}/{impact['device_count']} 台设备掉线。")
    if overall:
        lines.append(
            f"总体目标版本组 {_summary_rate(overall['target_group'])}，对照组 {_summary_rate(overall['control_group'])}。"
        )
    if alignment:
        lines.append(
            f"{alignment['site_event_id']} 于 {alignment['site_event_at']} 发生，"
            f"早于同站首条掉线 {alignment['first_offline_event_id']} {alignment['lead_seconds']:g} 秒。"
        )
    if by_site:
        lines.extend(
            f"{row['site_id']} 站复核目标版本组 {_summary_rate(row['target_group'])}，"
            f"对照组 {_summary_rate(row['control_group'])}。"
            for row in by_site["sites"]
        )
    lines.append(interpretation)
    return {
        "priority": {"direction": direction, "site_id": alignment["site_id"] if direction == "site" else None, "reason": reason},
        "impact": impact,
        "firmware_source": {key: rollout[key] for key in ("status", "reason")},
        "firmware_rollout_count": rollout_count,
        "firmware_overall": overall,
        "firmware_by_site": by_site,
        "site_comparison": site["key_data"]["site_comparison"],
        "gateway_alignment": alignment,
        "knowledge_base": firmware["key_data"]["knowledge"],
        "initial_judgments": {finding["role"]: finding["judgment"] for finding in (firmware, site)},
        "review_request_id": review["request_id"] if review else None,
        "review_judgment": revised["judgment"] if revised else None,
        "source_ids": collect_source_ids(state["shared_evidence"], *findings),
        "missing_evidence": list(gaps.values()),
        "interpretation": interpretation,
        "summary": "\n".join(lines),
    }


def create_multi_agent_investigation_graph(dependencies: SpecialistDependencies):
    agents = {"firmware": create_firmware_agent(dependencies), "site": create_site_agent(dependencies)}
    tasks = {
        "firmware": "核查升级版本与设备掉线的关联，提供总体固件分组和版本核验线索。",
        "site": "调查站点掉线分布，将站点事件与同站掉线告警对齐。",
    }

    def prepare(state: MultiAgentState) -> dict:
        try:
            scope = InvestigationScope(**(state.get("scope") or {}))
        except (TypeError, ValueError) as error:
            return {"scope_error": _failure_reason(error), "coordination_trace": [], "reviews_remaining": 1}
        return {
            "scope": scope.to_dict(),
            "shared_evidence": {
                "alarm_timeline": _read_source(dependencies.alarm_timeline, dependencies.db_path, scope),
                "firmware_rollouts": _read_source(dependencies.firmware_rollouts, dependencies.db_path, scope),
            },
            "reviews_remaining": 1,
            "review_request": None,
        }

    def initial_node(role: Role):
        def invoke(state: MultiAgentState) -> dict:
            return {f"{role}_initial": agents[role].invoke({
                "question": state["question"], "scope": state["scope"],
                "task": tasks[role], "shared_evidence": state["shared_evidence"],
            })}
        return invoke

    def coordinate(state: MultiAgentState) -> dict:
        firmware = state["firmware_initial"]["finding"]
        site_result = state["site_initial"]
        site = site_result["finding"]
        joined = {
            "action": "initial_findings_joined",
            "judgments": {"firmware": firmware["judgment"], "site": site["judgment"]},
        }
        alignment = align_gateway_to_offline(
            state["shared_evidence"], site_result["evidence"], InvestigationScope(**state["scope"]),
        )
        overall = firmware["key_data"]["overall"]
        if state["reviews_remaining"] > 0 and alignment and overall:
            target, control = overall["target_group"]["offline_rate"], overall["control_group"]["offline_rate"]
            if target is not None and control is not None and target > control:
                request = ReviewRequest(
                    request_id=f"{state['run_id']}:review-1",
                    triggered_by="site",
                    responsible_role="firmware",
                    question=f"{alignment['site_id']} 站有先于掉线的网关重启。按站分组后，总体固件差异是否仍存在？",
                    trigger_evidence_ids=[alignment["site_event_id"], alignment["first_offline_event_id"]],
                    original_judgment=firmware["judgment"],
                    site_id=alignment["site_id"],
                    target_version=overall["target_version"],
                ).model_dump()
                return {
                    "review_request": request,
                    "reviews_remaining": state["reviews_remaining"] - 1,
                    "coordination_trace": [joined, {"action": "request_review", **request}],
                }
        return {
            "review_request": None,
            "coordination_trace": [joined, {
                "action": "finish_without_review",
                "reason": "现有证据未同时表明总体固件组差异与可对齐的站点网关事件。",
            }],
        }

    def run_review(state: MultiAgentState) -> dict:
        request = state["review_request"]
        role = request["responsible_role"]
        reviewed = agents[role].invoke({
            "question": state["question"], "scope": state["scope"], "task": request["question"],
            "shared_evidence": state["shared_evidence"], "review_request": request,
            "prior_finding": state[f"{role}_initial"]["finding"],
            "peer_finding": state[f"{request['triggered_by']}_initial"]["finding"],
        })
        return {
            "review_result": reviewed,
            "coordination_trace": [*state["coordination_trace"], {
                "action": "review_finished", "review_request_id": request["request_id"],
                "role": role, "judgment": reviewed["finding"]["judgment"],
                "tool_actions": [call["action"] for call in reviewed["tool_calls"]],
            }],
        }

    graph = StateGraph(MultiAgentState, input_schema=InvestigationInput)
    graph.add_node("prepare", prepare)
    graph.add_node("firmware_initial", initial_node("firmware"))
    graph.add_node("site_initial", initial_node("site"))
    graph.add_node("coordinate", coordinate)
    graph.add_node("review", run_review)
    graph.add_node("report", lambda state: {"report": _report(state)})
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", lambda state: "report" if state.get("scope_error") else ["firmware_initial", "site_initial"],
        {name: name for name in ("firmware_initial", "site_initial", "report")},
    )
    graph.add_edge(["firmware_initial", "site_initial"], "coordinate")
    graph.add_conditional_edges(
        "coordinate", lambda state: "review" if state["review_request"] else "report", {"review": "review", "report": "report"},
    )
    graph.add_edge("review", "report")
    graph.add_edge("report", END)
    return graph.compile()
