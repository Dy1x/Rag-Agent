"""Bounded firmware and site investigation subgraphs."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.investigation_graph import KnowledgeResult, _failure_reason, _read_source
from agent.investigation_planner import firmware_candidates
from service.investigation_data_service import (
    InvestigationSource,
    InvestigationScope,
    _parse_utc,
    align_site_gateway_restart,
    compare_firmware_cohorts,
    compare_firmware_cohorts_by_site,
    compare_site_devices,
    get_alarm_timeline,
    get_firmware_rollouts,
    get_site_event_timeline,
)


Role = Literal["firmware", "site"]
Action = Literal[
    "compare_firmware_cohorts", "compare_firmware_cohorts_by_site", "lookup_known_issue",
    "compare_site_devices", "get_site_event_timeline", "finish",
]
FIRMWARE_ACTIONS = ("compare_firmware_cohorts", "compare_firmware_cohorts_by_site", "lookup_known_issue")
SITE_ACTIONS = ("compare_site_devices", "get_site_event_timeline")
INITIAL_TOOL_BUDGET = 2
REVIEW_TOOL_BUDGET = 1


class Finding(BaseModel):
    role: Role
    judgment: str
    summary: str
    key_data: dict
    source_ids: dict[str, list[str]]
    missing_evidence: list[dict[str, str]]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(min_length=1)
    triggered_by: Role
    responsible_role: Role
    question: str = Field(min_length=1)
    trigger_evidence_ids: list[str] = Field(min_length=1)
    original_judgment: str = Field(min_length=1)
    site_id: str = Field(min_length=1)
    target_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_cross_role_and_sources(self) -> ReviewRequest:
        if self.triggered_by == self.responsible_role:
            raise ValueError("复核必须由另一角色触发")
        if len(set(self.trigger_evidence_ids)) != len(self.trigger_evidence_ids):
            raise ValueError("复核依据 ID 必须唯一")
        return self


class SpecialistDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Action
    reason: str = Field(min_length=1)
    target_version: str | None = None

    @model_validator(mode="after")
    def check_version(self) -> SpecialistDecision:
        if self.action in FIRMWARE_ACTIONS:
            if not self.target_version:
                raise ValueError("固件操作需要目标版本")
        elif self.target_version is not None:
            raise ValueError("此操作不接受目标版本")
        return self


class SpecialistInput(TypedDict, total=False):
    question: str
    task: str
    scope: dict
    shared_evidence: dict
    prior_finding: dict
    peer_finding: dict
    review_request: dict


class SpecialistState(SpecialistInput, total=False):
    role: Role
    remaining_tools: int
    evidence: dict
    decisions: list[dict]
    tool_calls: list[dict]
    finding: dict


ScopedQuery = Callable[[InvestigationSource, InvestigationScope], dict]
VersionQuery = Callable[[InvestigationSource, InvestigationScope, str], dict]


@dataclass(frozen=True)
class SpecialistDependencies:
    db_path: InvestigationSource
    planner: Callable[[Role, dict], SpecialistDecision | dict | str]
    lookup_known_issue: Callable[[str], KnowledgeResult | dict] | None = None
    alarm_timeline: ScopedQuery = get_alarm_timeline
    firmware_rollouts: ScopedQuery = get_firmware_rollouts
    firmware_compare: VersionQuery = compare_firmware_cohorts
    firmware_compare_by_site: VersionQuery = compare_firmware_cohorts_by_site
    site_compare: ScopedQuery = compare_site_devices
    site_events: ScopedQuery = get_site_event_timeline


def _data(source: dict | None) -> dict:
    return source["data"] if source and source["status"] == "available" else {}


def collect_source_ids(*records: dict) -> dict[str, list[str]]:
    ids: dict[str, set[str]] = {}

    def visit(record):
        if isinstance(record, dict):
            for table, values in record.get("source_ids", {}).items():
                ids.setdefault(table, set()).update(values)
            for key, value in record.items():
                if key != "source_ids":
                    visit(value)
        elif isinstance(record, list):
            for value in record:
                visit(value)

    for record in records:
        visit(record)
    return {table: sorted(values) for table, values in ids.items()}


def align_gateway_to_offline(shared: dict, evidence: dict, scope: InvestigationScope) -> dict | None:
    return align_site_gateway_restart(
        _data(shared.get("alarm_timeline")), _data(evidence.get("compare_site_devices")),
        _data(evidence.get("get_site_event_timeline")), scope,
    )


def _review_error(request: dict, peer: dict, prior: dict, scope: dict, role: Role) -> str | None:
    event = peer.get("key_data", {}).get("gateway_alignment") or {}
    peer_ids = peer.get("source_ids", {})
    relevant_ids = {event.get("site_event_id"), event.get("first_offline_event_id")}
    if (
        request["responsible_role"] != role
        or request["triggered_by"] != peer.get("role")
        or prior.get("role") != role
        or request["original_judgment"] != prior.get("judgment")
        or request["site_id"] not in scope["site_ids"]
        or request["site_id"] != event.get("site_id")
        or request["target_version"] != prior.get("key_data", {}).get("target_version")
        or None in relevant_ids
        or set(request["trigger_evidence_ids"]) != relevant_ids
        or event["site_event_id"] not in peer_ids.get("site_events", [])
        or event["first_offline_event_id"] not in peer_ids.get("alarm_events", [])
    ):
        return "复核请求未能与初判及对方的网关和掉线证据对应。"
    start, end = _parse_utc(scope["start"]), _parse_utc(scope["end"])
    if not start <= _parse_utc(event["site_event_at"]) < _parse_utc(event["first_offline_at"]) < end:
        return "复核证据的时间先后或调查窗口无效。"
    return None


def plan_specialist_offline(role: Role, request: dict) -> SpecialistDecision:
    evidence = request["evidence"]
    review = request.get("review_request")
    if review:
        return SpecialistDecision(
            action="compare_firmware_cohorts_by_site",
            target_version=review["target_version"],
            reason=f"站点 Agent 提供了 {request['peer_finding']['key_data']['gateway_alignment']['site_event_id']}，"
                   "按站分组复核总体固件关联。",
        )
    if role == "site":
        if "compare_site_devices" not in evidence:
            return SpecialistDecision(action="compare_site_devices", reason="先比较各站的受影响设备比例。")
        sites = _data(evidence["compare_site_devices"])
        if sites and not sites["sites_with_offline_devices"]:
            return SpecialistDecision(action="finish", reason="站点对比未发现掉线，无需继续对齐异常事件。")
        if "get_site_event_timeline" not in evidence:
            return SpecialistDecision(action="get_site_event_timeline", reason="查询站点事件并与同站掉线告警对齐。")
        return SpecialistDecision(action="finish", reason="已取得站点统计和时间线。")

    candidates = request["candidate_versions"]
    if not candidates:
        return SpecialistDecision(action="finish", reason="升级证据没有提供可比较的版本。")
    if "compare_firmware_cohorts" not in evidence:
        return SpecialistDecision(
            action="compare_firmware_cohorts", target_version=candidates[0],
            reason="先用升级记录中的候选版本比较总体掉线率。",
        )
    overall = _data(evidence["compare_firmware_cohorts"])
    if (overall.get("offline_rate_difference") or 0) > 0 and "lookup_known_issue" not in evidence:
        return SpecialistDecision(
            action="lookup_known_issue", target_version=overall["target_version"],
            reason="目标版本组总体掉线率较高，查询可供核验的版本资料。",
        )
    return SpecialistDecision(action="finish", reason="保留已取得的固件证据并提交初判。")


def plan_specialist_with_deepseek(role: Role, request: dict) -> SpecialistDecision:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    from model.factory import get_chat_model

    required = None
    if "evidence" in request:
        next_step = plan_specialist_offline(role, request)
        if next_step.action in (
            "compare_site_devices", "get_site_event_timeline",
            "compare_firmware_cohorts", "compare_firmware_cohorts_by_site",
        ):
            required = next_step
        allowed = [
            action for action in request["allowed_actions"] if action not in request["evidence"]
            and (action != "compare_firmware_cohorts_by_site" or request.get("review_request"))
        ]
        request = {
            **request,
            "allowed_actions": allowed,
            "required_action": required.action if required else None,
        }

    prompt = Path(__file__).resolve().parents[1] / "prompts" / f"investigation_{role}.txt"
    chain = get_chat_model().bind(response_format={"type": "json_object"}, max_tokens=600) | StrOutputParser()
    response = chain.invoke([
        SystemMessage(content=prompt.read_text(encoding="utf-8")),
        HumanMessage(content=json.dumps(request, ensure_ascii=False)),
    ])
    decision = SpecialistDecision.model_validate_json(response)
    if required and decision.action != required.action:
        return required.model_copy(update={
            "reason": f"{required.reason} 模型选择了 {decision.action}，按必要取证步骤继续。",
        })
    if "evidence" in request and decision.action not in request["allowed_actions"]:
        raise ValueError("模型选择了当前阶段不可用的动作")
    return decision


def _decision(value: SpecialistDecision | dict | str) -> SpecialistDecision:
    if isinstance(value, str):
        return SpecialistDecision.model_validate_json(value)
    return SpecialistDecision.model_validate(value.model_dump() if isinstance(value, SpecialistDecision) else value)


def _make_finding(role: Role, state: SpecialistState) -> Finding:
    evidence, shared = state["evidence"], state["shared_evidence"]
    relevant_shared = shared if role == "firmware" else {"alarm_timeline": shared["alarm_timeline"]}
    gaps = [
        {"source": name, "status": result["status"], "reason": result["reason"]}
        for name, result in {**relevant_shared, **evidence}.items()
        if result["status"] not in ("available", "answered")
    ]
    for decision in state["decisions"]:
        if decision.get("planner_error"):
            gaps.append({"source": "specialist_planner", "status": "failed", "reason": decision["planner_error"]})
    sources = collect_source_ids(*evidence.values())
    if role == "firmware":
        overall = _data(evidence.get("compare_firmware_cohorts"))
        by_site = _data(evidence.get("compare_firmware_cohorts_by_site"))
        knowledge = evidence.get("lookup_known_issue", {}).get("data")
        if knowledge:
            sources["knowledge_base"] = [citation["id"] for citation in knowledge["sources"]]
        rows = by_site.get("sites", [])
        if len(rows) > 1 and all(row["offline_rate_difference"] == 0 for row in rows):
            judgment, summary = "no_within_site_difference", "各站内目标版本组和对照组的掉线率相同。"
        elif by_site:
            judgment, summary = "within_site_comparison", "已取得分站组间比较，各站结果需结合站点证据复核。"
        elif (overall.get("offline_rate_difference") or 0) > 0:
            judgment, summary = "overall_association", "目标版本组总体掉线率较高，需核验站点分布的影响。"
        elif overall:
            judgment, summary = "no_positive_overall_difference", "总体比较未观察到目标版本组掉线率更高。"
        else:
            judgment, summary = "insufficient", "当前未取得可比较的固件分组结果。"
        if shared["firmware_rollouts"]["status"] == "available" and not firmware_candidates(shared["firmware_rollouts"]):
            gaps.append({"source": "firmware_rollouts", "status": "no_records", "reason": "窗口内没有可分组的升级记录。"})
        if rows and any(row["offline_rate_difference"] is None for row in rows):
            gaps.append({"source": "firmware_cohorts_by_site", "status": "incomplete", "reason": "部分站点缺少可比较的设备组。"})
        key_data = {
            "target_version": (by_site or overall).get("target_version"),
            "overall": overall or None, "by_site": by_site or None, "knowledge": knowledge,
        }
    else:
        scope = InvestigationScope(**state["scope"])
        sites = _data(evidence.get("compare_site_devices"))
        timeline = _data(evidence.get("get_site_event_timeline"))
        alignment = align_gateway_to_offline(shared, evidence, scope)
        key_data = {"site_comparison": sites or None, "site_event_timeline": timeline or None, "gateway_alignment": alignment}
        if alignment:
            judgment, summary = "gateway_clue", f"{alignment['site_id']} 站网关重启早于同站首条掉线，值得复核。"
        else:
            judgment = "site_distribution" if sites else "insufficient"
            summary = "保留已有站点分布与事件，当前没有匹配的网关重启线索。"
            if timeline:
                gaps.append({
                    "source": "gateway_alignment", "status": "no_match",
                    "reason": "窗口内没有可与掉线集中的站点及首条掉线对齐的网关重启。",
                })
    return Finding(
        role=role, judgment=judgment, summary=summary, key_data=key_data, source_ids=sources, missing_evidence=gaps,
    )


def _create_specialist(role: Role, deps: SpecialistDependencies):
    queries = {
        "compare_firmware_cohorts": deps.firmware_compare,
        "compare_firmware_cohorts_by_site": deps.firmware_compare_by_site,
        "compare_site_devices": deps.site_compare,
        "get_site_event_timeline": deps.site_events,
    }

    def initialize(state: SpecialistState) -> dict:
        return {
            "role": role, "evidence": {}, "tool_calls": [], "decisions": [],
            "remaining_tools": REVIEW_TOOL_BUDGET if state.get("review_request") else INITIAL_TOOL_BUDGET,
        }

    def plan(state: SpecialistState) -> dict:
        entry = {}
        try:
            review = state.get("review_request")
            if review:
                review = ReviewRequest.model_validate(review).model_dump()
                problem = _review_error(review, state.get("peer_finding", {}), state.get("prior_finding", {}), state["scope"], role)
                if problem:
                    raise ValueError(problem)
            shared = state["shared_evidence"]
            versions = firmware_candidates(shared["firmware_rollouts"]) if role == "firmware" else {}
            affected = set(_data(shared["alarm_timeline"]).get("offline_device_ids", []))
            candidates = sorted(versions, key=lambda version: (-len(versions[version] & affected), version))
            allowed = FIRMWARE_ACTIONS if role == "firmware" else SITE_ACTIONS
            if review and role == "firmware":
                allowed = ("compare_firmware_cohorts_by_site",)
            request = {
                "question": state["question"], "task": state["task"], "scope": state["scope"],
                "evidence": state["evidence"], "shared_evidence": shared, "candidate_versions": candidates,
                "allowed_actions": [*allowed, "finish"], "remaining_tools": state["remaining_tools"],
                "review_request": review, "peer_finding": state.get("peer_finding"), "prior_finding": state.get("prior_finding"),
            }
            decision = _decision(deps.planner(role, deepcopy(request)))
            if decision.action != "finish":
                if decision.action not in allowed or decision.action in state["evidence"]:
                    raise ValueError("动作超出角色权限、复核范围或已执行。")
                if state["remaining_tools"] <= 0:
                    raise ValueError("角色工具额度已用完。")
                if decision.action in FIRMWARE_ACTIONS and (
                    decision.target_version not in versions or (review and decision.target_version != review["target_version"])
                ):
                    raise ValueError("目标版本缺少有效升级证据，或超出复核范围。")
        except Exception as error:
            decision = SpecialistDecision(action="finish", reason="保留已有证据并报告规划缺口。")
            entry["planner_error"] = _failure_reason(error)
        return {"decisions": [*state["decisions"], {**decision.model_dump(), **entry}]}

    def execute(state: SpecialistState) -> dict:
        decision = state["decisions"][-1]
        action, version = decision["action"], decision["target_version"]
        if action == "lookup_known_issue":
            try:
                if not callable(deps.lookup_known_issue):
                    result = KnowledgeResult(status="unavailable", text="知识库依赖不可用。")
                else:
                    result = KnowledgeResult.model_validate(deps.lookup_known_issue(
                        f"充电站设备升级至 {version} 后掉线，有哪些已知问题和核验线索？{state['question']}",
                    ))
            except Exception as error:
                result = KnowledgeResult(status="failed", text=_failure_reason(error))
            evidence = {
                "status": result.status, "data": result.model_dump(),
                "reason": None if result.status == "answered" else result.text,
            }
        else:
            scope = InvestigationScope(**state["scope"])
            args = (deps.db_path, scope, version) if action in FIRMWARE_ACTIONS else (deps.db_path, scope)
            evidence = _read_source(queries[action], *args)
        review = state.get("review_request") or {}
        call = {
            "role": role, "action": action, "reason": decision["reason"], "target_version": version,
            "status": evidence["status"], "result_reason": evidence["reason"],
            "review_request_id": review.get("request_id"), "source_ids": collect_source_ids(evidence),
        }
        if action == "lookup_known_issue":
            call["source_ids"]["knowledge_base"] = [item["id"] for item in evidence["data"]["sources"]]
        return {
            "evidence": {**state["evidence"], action: evidence},
            "tool_calls": [*state["tool_calls"], call],
            "remaining_tools": state["remaining_tools"] - 1,
        }

    graph = StateGraph(SpecialistState, input_schema=SpecialistInput)
    graph.add_node("initialize", initialize)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("judge", lambda state: {"finding": _make_finding(role, state).model_dump()})
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "plan")
    graph.add_conditional_edges(
        "plan", lambda state: "judge" if state["decisions"][-1]["action"] == "finish" else "execute",
        {"judge": "judge", "execute": "execute"},
    )
    graph.add_conditional_edges(
        "execute", lambda state: "plan" if state["remaining_tools"] else "judge", {"plan": "plan", "judge": "judge"},
    )
    graph.add_edge("judge", END)
    return graph.compile()


def create_firmware_agent(dependencies: SpecialistDependencies):
    return _create_specialist("firmware", dependencies)


def create_site_agent(dependencies: SpecialistDependencies):
    return _create_specialist("site", dependencies)
