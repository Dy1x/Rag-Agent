"""Single-agent investigation graph with independent evidence sources."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agent.investigation_planner import (
    FIRMWARE_COMPARISON_ACTIONS, PlanDecision, PlanningInput, firmware_candidates, parse_decision,
)
from service.investigation_data_service import (
    InvestigationScope,
    align_site_gateway_restart,
    compare_firmware_cohorts,
    compare_firmware_cohorts_by_site,
    compare_site_devices,
    get_alarm_timeline,
    get_firmware_rollouts,
    get_site_event_timeline,
)


class InvestigationInput(TypedDict):
    run_id: str
    question: str
    scope: dict


class InvestigationState(InvestigationInput, total=False):
    alarm_timeline: dict
    firmware_rollouts: dict
    firmware_comparison: dict
    firmware_comparison_by_site: dict
    site_comparison: dict
    site_event_timeline: dict
    known_issue: dict
    execution_trace: list[dict]
    followups_remaining: int
    planning_rounds: int
    missing_evidence: list[dict]
    decision: dict
    report: dict


class KnowledgeResult(BaseModel):
    status: Literal["answered", "clarify", "insufficient", "unavailable", "failed"]
    text: str = Field(min_length=1)
    sources: list[dict[str, str]] = Field(default_factory=list)
    simulated: bool = False


ScopedQuery = Callable[[str | Path, InvestigationScope], dict]


@dataclass(frozen=True)
class InvestigationDependencies:
    db_path: str | Path
    planner: Callable[[PlanningInput], PlanDecision | dict | str]
    lookup_known_issue: Callable[[str], KnowledgeResult | dict] | None = None
    alarm_timeline: ScopedQuery | None = get_alarm_timeline
    firmware_rollouts: ScopedQuery | None = get_firmware_rollouts
    firmware_compare: Callable[[str | Path, InvestigationScope, str], dict] | None = compare_firmware_cohorts
    site_compare: ScopedQuery | None = compare_site_devices
    firmware_compare_by_site: Callable[[str | Path, InvestigationScope, str], dict] | None = compare_firmware_cohorts_by_site
    site_events: ScopedQuery | None = get_site_event_timeline


COMPARISON_FOLLOWUP_BUDGET = 5


def _failure_reason(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_context=False, include_url=False)
        return "结构校验失败（" + "、".join(
            ".".join(str(key) for key in item["loc"]) or item["type"] for item in details
        ) + "）"
    # Provider error bodies may contain credentials. Keep the HTTP status for diagnosis.
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return f"{type(error).__name__}（HTTP {status_code}）"
    message = str(error).splitlines()
    if not message:
        return type(error).__name__
    reason = message[0]
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            reason = reason.replace(secret, "[已隐藏]")
    return f"{type(error).__name__}：{reason}"


def _read_source(operation: Callable[..., dict] | None, *args) -> dict:
    if not callable(operation):
        return {"status": "unavailable", "data": None, "reason": "查询依赖不可用。"}
    try:
        data = operation(*args)
        status = data.get("status", "available")
        if status not in ("available", "unavailable", "failed"):
            raise ValueError("查询返回了未知的来源状态")
        reason = data.get("reason")
        if status != "available" and not reason:
            reason = "查询未返回可用证据。"
        return {"status": status, "data": data, "reason": reason}
    except Exception as error:
        return {"status": "failed", "data": None, "reason": _failure_reason(error)}


def _missing(source: str, status: str, reason: str) -> dict:
    return {"source": source, "status": status, "reason": reason}


def _record_followup(state: InvestigationState, field: str, result: dict, source: str) -> dict:
    decision = state["decision"]
    entry = {**decision, "status": result["status"]}
    if result.get("reason"):
        entry["result_reason"] = result["reason"]
    missing = state["missing_evidence"]
    if result["status"] not in ("available", "answered"):
        missing = [*missing, _missing(source, result["status"], result.get("reason") or result["text"])]
    return {
        field: result,
        "execution_trace": [*state["execution_trace"], entry],
        "followups_remaining": state["followups_remaining"] - 1,
        "missing_evidence": missing,
    }


def _source_data(state: InvestigationState, key: str) -> dict:
    source = state.get(key)
    return source["data"] if source and source["status"] == "available" else {}


def _planner_view(value):
    # Fixture record IDs contain case labels. Keep provenance in the report.
    if isinstance(value, dict):
        return {
            key: _planner_view(item) for key, item in value.items()
            if key not in ("event_id", "rollout_id", "source_ids")
        }
    if isinstance(value, list):
        return [_planner_view(item) for item in value]
    return value


def _gateway_alignment(state: InvestigationState) -> dict | None:
    return align_site_gateway_restart(
        _source_data(state, "alarm_timeline"), _source_data(state, "site_comparison"),
        _source_data(state, "site_event_timeline"), InvestigationScope(**state["scope"]),
    )


def _compile_report(state: InvestigationState, *, extended_tools: bool = False) -> dict:
    alarm = _source_data(state, "alarm_timeline")
    rollouts = _source_data(state, "firmware_rollouts")
    firmware = _source_data(state, "firmware_comparison")
    sites = _source_data(state, "site_comparison")
    knowledge = state.get("known_issue")
    site_rows = sites.get("sites", [])
    groups = [firmware[key] for key in ("target_group", "control_group") if firmware.get(key)]
    by_site = _source_data(state, "firmware_comparison_by_site")
    timeline = _source_data(state, "site_event_timeline")
    gateway = _gateway_alignment(state) if extended_tools else None
    sources: dict[str, set[str]] = {"devices": set(), "alarm_events": set(), "firmware_rollouts": set()}
    for summary in (alarm, rollouts, firmware, *groups, *site_rows, by_site, timeline):
        for table, ids in summary.get("source_ids", {}).items():
            sources.setdefault(table, set()).update(ids)

    affected_sites = (
        sorted({row["site_id"] for row in alarm["events"] if row["event_type"] == "offline"})
        if alarm else sites.get("sites_with_offline_devices", [])
    )
    impact = {
        "site_ids": state["scope"]["site_ids"],
        "affected_sites": affected_sites if alarm or sites else None,
        "device_count": alarm["device_count"] if alarm else (
            sum(row["device_count"] for row in site_rows) if sites else None
        ),
        "offline_device_count": alarm["offline_device_count"] if alarm else (
            sum(row["offline_device_count"] for row in site_rows) if sites else None
        ),
        "offline_device_ids": alarm["offline_device_ids"] if alarm else (
            sorted(device for row in site_rows for device in row["offline_device_ids"]) if sites else None
        ),
        "offline_event_count": alarm.get("offline_event_count"),
    }

    if impact["offline_device_count"] == 0:
        direction, reason = "none", "调查窗口内没有掉线设备记录。"
    elif gateway and by_site and any(row["offline_rate_difference"] != 0 for row in by_site["sites"]):
        direction, reason = "undetermined", "分站固件对比与网关线索仍需一起核验。"
    elif gateway:
        direction = "site"
        reason = f"优先核验 {gateway['site_id']} 站网关事件 {gateway['site_event_id']} 和同站掉线告警。"
    elif sites and len(affected_sites) == 1 and len(site_rows) > 1:
        direction, reason = "site", f"掉线集中在 {affected_sites[0]} 站，优先核对该站网络、供电和网关记录。"
    elif len(affected_sites) > 1 and firmware.get("offline_rate_difference") == 0:
        direction, reason = "undetermined", "掉线涉及多个站点且两组掉线率相同，需核对跨站共用链路和设备日志。"
    elif (firmware.get("offline_rate_difference") or 0) > 0:
        direction, reason = "firmware", f"目标版本 {firmware['target_version']} 组掉线率较高，优先核对升级和设备诊断记录。"
    elif sites and affected_sites:
        direction, reason = "site", "先按站点统计核对网络和供电记录，同时补齐能区分各解释的证据。"
    else:
        direction, reason = "undetermined", "当前证据无法确定优先方向，先补齐缺失来源。"

    firmware_source = state["firmware_rollouts"]
    missing = list(state["missing_evidence"])
    if direction in ("firmware", "undetermined") and impact["offline_device_count"]:
        missing.append(_missing("device_diagnostics", "not_collected", "还需设备侧升级与掉线日志、版本复测或回滚结果。"))
    if direction in ("firmware", "site", "undetermined") and impact["offline_device_count"]:
        missing.append(_missing("site_network_power_logs", "not_collected", "还需同一时间窗口的站点网络、供电和网关记录。"))
    interpretation = "以上结果描述观测关联，故障归因待验证。"
    lines = [
        f"调查覆盖 {'、'.join(impact['site_ids'])} 站，窗口为 UTC {state['scope']['start']} 至 {state['scope']['end']}。",
        "掉线影响暂无法确认。" if impact["offline_device_count"] is None
        else f"调查范围内 {impact['offline_device_count']}/{impact['device_count']} 台设备掉线。",
        reason,
    ]
    if len(groups) == 2:
        target, control = groups
        lines.append(
            f"目标版本组有 {target['offline_device_count']}/{target['device_count']} 台掉线，"
            f"对照组有 {control['offline_device_count']}/{control['device_count']} 台掉线。"
        )
    if site_rows:
        counts = "、".join(
            f"{row['site_id']} 站 {row['offline_device_count']}/{row['device_count']}" for row in site_rows
        )
        lines.append(f"各站掉线比例为 {counts}。")
    if gateway:
        lines.append(
            f"{gateway['site_event_id']} 早于同站首条掉线 {gateway['first_offline_event_id']} "
            f"{gateway['lead_seconds']:g} 秒。"
        )
    if by_site:
        lines.extend(
            f"{row['site_id']} 站目标版本组 {row['target_group']['offline_device_count']}/"
            f"{row['target_group']['device_count']}，对照组 {row['control_group']['offline_device_count']}/"
            f"{row['control_group']['device_count']}。"
            for row in by_site["sites"]
        )
    lines.extend(item["reason"] for item in missing)
    lines.append(interpretation)
    report = {
        "impact": impact,
        "priority": {"direction": direction, "reason": reason},
        "alarm_source": {key: state["alarm_timeline"][key] for key in ("status", "reason")},
        "firmware_source": {key: firmware_source[key] for key in ("status", "reason")},
        "firmware_rollout_count": rollouts.get("rollout_count"),
        "firmware_comparison": firmware or None,
        "site_comparison": sites or None,
        "knowledge_base": knowledge,
        "source_ids": {**{table: sorted(ids) for table, ids in sources.items()},
                       "knowledge_base": knowledge["sources"] if knowledge else []},
        "missing_evidence": missing,
        "interpretation": interpretation,
        "summary": "\n".join(lines),
    }
    if extended_tools:
        report.update(firmware_by_site=by_site or None, gateway_alignment=gateway)
        if gateway and direction == "site":
            report["priority"]["site_id"] = gateway["site_id"]
    return report


def create_investigation_graph(
    dependencies: InvestigationDependencies,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_after: list[str] | None = None,
    extended_tools: bool = False,
):
    followup_budget = COMPARISON_FOLLOWUP_BUDGET if extended_tools else 2
    tool_dependencies = {
        "compare_firmware_cohorts": dependencies.firmware_compare,
        "compare_site_devices": dependencies.site_compare,
        "lookup_known_issue": dependencies.lookup_known_issue,
    }
    if extended_tools:
        tool_dependencies.update({
            "compare_firmware_cohorts_by_site": dependencies.firmware_compare_by_site,
            "get_site_event_timeline": dependencies.site_events,
        })

    def initialize(state: InvestigationInput) -> dict:
        if not state["run_id"].strip() or not state["question"].strip():
            raise ValueError("运行 ID 和问题均不能为空")
        return {
            "scope": InvestigationScope(**state["scope"]).to_dict(),
            "question": state["question"].strip(),
            "followups_remaining": followup_budget,
            "planning_rounds": 0,
        }

    def read_alarms(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        return {"alarm_timeline": _read_source(dependencies.alarm_timeline, dependencies.db_path, scope)}

    def read_firmware(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        return {"firmware_rollouts": _read_source(dependencies.firmware_rollouts, dependencies.db_path, scope)}

    def collect_initial_evidence(state: InvestigationState) -> dict:
        trace, missing = [], []
        initial = [
            ("alarm_timeline", "get_alarm_timeline", "alarm_events", "读取调查窗口内的告警及受影响设备。"),
            ("firmware_rollouts", "get_firmware_rollouts", "firmware_rollouts", "读取调查开始前两小时的成功升级记录。"),
        ]
        for field, action, source, reason in initial:
            result = state[field]
            trace.append({"action": action, "status": result["status"], "reason": result["reason"] or reason})
            if result["status"] != "available":
                missing.append(_missing(source, result["status"], result["reason"]))
        if state["firmware_rollouts"]["status"] == "available" and not firmware_candidates(state["firmware_rollouts"]):
            missing.append(_missing("firmware_rollouts", "no_records", "固件来源可用，调查前两小时没有可供分组的升级记录。"))
        return {"execution_trace": trace, "missing_evidence": missing}

    def validation_error(state: InvestigationState, decision: PlanDecision) -> str | None:
        action = decision.action
        if action == "finish":
            return None
        if state["followups_remaining"] <= 0:
            return "后续调查额度已用完。"
        if any(entry["action"] == action for entry in state["execution_trace"]):
            return f"动作 {action} 已尝试过。"
        if not callable(tool_dependencies.get(action)):
            return f"动作 {action} 的依赖不可用。"
        if action in FIRMWARE_COMPARISON_ACTIONS:
            if state["alarm_timeline"]["status"] != "available":
                return "告警证据不可用，无法进行固件分组对比。"
            if state["firmware_rollouts"]["status"] != "available":
                return "固件来源不可用，无法进行分组对比。"
            candidates = firmware_candidates(state["firmware_rollouts"])
            if not candidates:
                return "当前调查窗口内没有可选的目标版本。"
            if decision.target_version not in candidates:
                return "所选目标版本不在已取得证据的候选版本内。"
        return None

    def plan(state: InvestigationState) -> dict:
        evidence_fields = ("alarm_timeline", "firmware_rollouts", "firmware_comparison", "site_comparison", "known_issue")
        if extended_tools:
            evidence_fields += ("firmware_comparison_by_site", "site_event_timeline")
        evidence = {field: state[field] for field in evidence_fields if field in state}
        if extended_tools and (gateway := _gateway_alignment(state)):
            evidence["gateway_alignment"] = gateway
        request = _planner_view({
            "question": state["question"],
            "scope": state["scope"],
            "evidence": evidence,
        })
        try:
            decision = parse_decision(dependencies.planner(request))
            error = validation_error(state, decision)
        except Exception as exception:
            error = _failure_reason(exception)
        if error:
            decision = PlanDecision(action="compare_site_devices", reason="规划不可用，降级为站点对比。")
            if validation_error(state, decision):
                decision = PlanDecision(action="finish", reason="规划不可用，保留已有证据和缺口并收口。")
        chosen = decision.model_dump(exclude_none=True)
        if error:
            chosen["planner_error"] = error
        return {"decision": chosen, "planning_rounds": state["planning_rounds"] + 1}

    def run_firmware_comparison(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        result = _read_source(
            dependencies.firmware_compare, dependencies.db_path, scope, state["decision"]["target_version"],
        )
        return _record_followup(state, "firmware_comparison", result, "firmware_comparison")

    def run_site_comparison(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        result = _read_source(dependencies.site_compare, dependencies.db_path, scope)
        return _record_followup(state, "site_comparison", result, "site_comparison")

    def run_firmware_comparison_by_site(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        result = _read_source(
            dependencies.firmware_compare_by_site, dependencies.db_path, scope, state["decision"]["target_version"],
        )
        return _record_followup(state, "firmware_comparison_by_site", result, "firmware_cohorts_by_site")

    def read_site_events(state: InvestigationState) -> dict:
        scope = InvestigationScope(**state["scope"])
        result = _read_source(dependencies.site_events, dependencies.db_path, scope)
        return _record_followup(state, "site_event_timeline", result, "site_events")

    def run_knowledge_lookup(state: InvestigationState) -> dict:
        comparison = _source_data(state, "firmware_comparison")
        version = comparison.get("target_version")
        query = f"充电站设备掉线的已知问题及核验线索。用户问题：{state['question']}"
        if version and extended_tools:
            query = f"充电站设备升级至 {version} 后掉线，有哪些已知问题和核验线索？{state['question']}"
        elif version:
            query += f" 本次已对比的固件目标版本为 {version}。"
        try:
            result = KnowledgeResult.model_validate(dependencies.lookup_known_issue(query))
        except Exception as error:
            result = KnowledgeResult(status="failed", text=_failure_reason(error))
        return _record_followup(state, "known_issue", result.model_dump(), "known_issue")

    def after_action(state: InvestigationState) -> Literal["plan", "finish"]:
        return "plan" if state["followups_remaining"] > 0 and state["planning_rounds"] < followup_budget else "finish"

    def finish(state: InvestigationState) -> dict:
        decision = state["decision"]
        if decision["action"] != "finish":
            decision = {"action": "finish", "reason": "已达到后续调查动作上限，汇总取得的证据。"}
        return {
            "execution_trace": [*state["execution_trace"], {**decision, "status": "completed"}],
            "report": _compile_report(state, extended_tools=extended_tools),
        }

    graph = StateGraph(InvestigationState, input_schema=InvestigationInput)
    graph.add_node("initialize", initialize)
    graph.add_node("get_alarm_timeline", read_alarms)
    graph.add_node("get_firmware_rollouts", read_firmware)
    graph.add_node("collect_initial_evidence", collect_initial_evidence)
    graph.add_node("plan", plan)
    graph.add_node("finish", finish)
    followups = {
        "compare_firmware_cohorts": run_firmware_comparison,
        "compare_site_devices": run_site_comparison,
        "lookup_known_issue": run_knowledge_lookup,
    }
    if extended_tools:
        followups.update({
            "compare_firmware_cohorts_by_site": run_firmware_comparison_by_site,
            "get_site_event_timeline": read_site_events,
        })
    for name, node in followups.items():
        graph.add_node(name, node)
        graph.add_conditional_edges(name, after_action)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "get_alarm_timeline")
    graph.add_edge("initialize", "get_firmware_rollouts")
    graph.add_edge(["get_alarm_timeline", "get_firmware_rollouts"], "collect_initial_evidence")
    graph.add_edge("collect_initial_evidence", "plan")
    graph.add_conditional_edges(
        "plan", lambda state: state["decision"]["action"],
        {name: name for name in (*followups, "finish")},
    )
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
