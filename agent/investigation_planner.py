"""Decisions for the offline and DeepSeek investigation modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


Action = Literal[
    "compare_firmware_cohorts", "compare_site_devices", "lookup_known_issue",
    "compare_firmware_cohorts_by_site", "get_site_event_timeline", "finish",
]
FIRMWARE_COMPARISON_ACTIONS = ("compare_firmware_cohorts", "compare_firmware_cohorts_by_site")


class PlanningInput(TypedDict):
    question: str
    scope: dict
    evidence: dict[str, dict]


class PlanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Action
    reason: str = Field(min_length=1)
    target_version: str | None = None

    @model_validator(mode="after")
    def check_target_version(self) -> PlanDecision:
        if self.action in FIRMWARE_COMPARISON_ACTIONS:
            if not self.target_version:
                raise ValueError("固件对比需要明确的目标版本")
        elif self.target_version is not None:
            raise ValueError("只有固件对比接受目标版本")
        return self


def parse_decision(value: PlanDecision | dict | str) -> PlanDecision:
    if isinstance(value, str):
        return PlanDecision.model_validate_json(value)
    if isinstance(value, PlanDecision):
        value = value.model_dump()
    return PlanDecision.model_validate(value)


def firmware_candidates(source: dict) -> dict[str, set[str]]:
    """Group devices by the last successful upgrade in the queried UTC window."""
    if source["status"] != "available":
        return {}

    # The data service orders rollouts by completion instant and ID.
    latest = {row["device_id"]: row["to_version"] for row in source["data"]["rollouts"]}
    candidates: dict[str, set[str]] = {}
    for device_id, version in latest.items():
        if version:
            candidates.setdefault(version, set()).add(device_id)
    return candidates


def plan_offline(request: PlanningInput) -> PlanDecision:
    evidence = request["evidence"]
    alarm = evidence["alarm_timeline"]
    if alarm["status"] != "available":
        if "site_comparison" not in evidence:
            return PlanDecision(action="compare_site_devices", reason="告警首查失败，尝试从站点对比入口取得影响统计。")
        return PlanDecision(action="finish", reason="保留可用的调查结果，并报告告警首查的缺口。")

    alarms = alarm["data"]
    offline_devices = set(alarms["offline_device_ids"])
    if not offline_devices:
        return PlanDecision(action="finish", reason="调查窗口内没有掉线设备记录。")

    if "firmware_comparison" in evidence:
        comparison = evidence["firmware_comparison"]
        if comparison["status"] == "available" and (comparison["data"]["offline_rate_difference"] or 0) > 0:
            return PlanDecision(action="lookup_known_issue", reason="目标版本组掉线率更高，补查该版本的已知问题与核验线索。")
        return PlanDecision(action="compare_site_devices", reason="固件分组未提供清晰的方向，继续核对站点分布。")

    affected_sites = {event["site_id"] for event in alarms["events"] if event["event_type"] == "offline"}
    candidates = firmware_candidates(evidence["firmware_rollouts"])
    if "site_comparison" in evidence:
        comparison = evidence["site_comparison"]
        if comparison["status"] == "available" and len(comparison["data"]["sites_with_offline_devices"]) == 1:
            return PlanDecision(action="finish", reason="掉线集中在一个站点，已有统计可用于安排站点侧排查。")
        if not candidates and len(affected_sites) > 1:
            return PlanDecision(action="lookup_known_issue", reason="掉线涉及多个站点且缺少可比较的升级证据，补查通用核验线索。")
        return PlanDecision(action="finish", reason="已完成站点对比，剩余归因需要额外证据。")

    if len(affected_sites) > 1 and candidates:
        version = min(candidates, key=lambda item: (-len(candidates[item] & offline_devices), item))
        if candidates[version] & offline_devices:
            return PlanDecision(
                action="compare_firmware_cohorts",
                target_version=version,
                reason=f"掉线跨站分布，受影响设备中有近期升级到 {version} 的设备，先对比版本组和对照组。",
            )
    return PlanDecision(action="compare_site_devices", reason="先按设备台账比较各站点的掉线分布。")


def plan_comparison_offline(request: PlanningInput) -> PlanDecision:
    """Use the same tools as the multi-agent evaluation, with a single planner."""
    evidence = request["evidence"]
    alarm = evidence["alarm_timeline"]
    if alarm["status"] != "available" or not alarm["data"]["offline_device_count"]:
        return plan_offline(request)
    if "site_comparison" not in evidence:
        return PlanDecision(action="compare_site_devices", reason="按站点和设备台账确认影响分布。")
    if "site_event_timeline" not in evidence:
        return PlanDecision(action="get_site_event_timeline", reason="查询可能早于掉线的站点事件。")
    candidates = firmware_candidates(evidence["firmware_rollouts"])
    if candidates and "firmware_comparison" not in evidence:
        offline_devices = set(alarm["data"]["offline_device_ids"])
        version = min(candidates, key=lambda candidate: (-len(candidates[candidate] & offline_devices), candidate))
        return PlanDecision(
            action="compare_firmware_cohorts", target_version=version,
            reason="使用已取得的升级记录核对总体版本组差异。",
        )
    overall = evidence.get("firmware_comparison", {})
    if overall.get("status") == "available" and (overall["data"]["offline_rate_difference"] or 0) > 0:
        if evidence.get("gateway_alignment") and "firmware_comparison_by_site" not in evidence:
            return PlanDecision(
                action="compare_firmware_cohorts_by_site", target_version=overall["data"]["target_version"],
                reason="总体固件差异与站点事件线索同时出现，按站核对组间差异。",
            )
        if not evidence.get("gateway_alignment") and "known_issue" not in evidence:
            return PlanDecision(action="lookup_known_issue", reason="补查相关版本的核验资料。")
    return PlanDecision(action="finish", reason="已有统计可用于安排下一步排查，披露证据缺口。")


def plan_with_deepseek(
    request: PlanningInput, *, prompt_name: str = "investigation_plan.txt",
) -> PlanDecision:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    from model.factory import get_chat_model

    prompt = Path(__file__).resolve().parents[1] / "prompts" / prompt_name
    chain = get_chat_model().bind(
        response_format={"type": "json_object"}, max_tokens=600,
    ) | StrOutputParser()
    response = chain.invoke([
        SystemMessage(content=prompt.read_text(encoding="utf-8")),
        HumanMessage(content=json.dumps(request, ensure_ascii=False)),
    ])
    return parse_decision(response)
