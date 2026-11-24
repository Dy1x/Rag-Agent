"""Score six investigation reports against separately authored expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.investigation.run_agent import run_case


METRICS = ("investigation_direction", "key_numbers", "traceable_evidence", "gap_disclosure")


def _device_counts(summary: dict | None) -> list[int | None] | None:
    if summary is None:
        return None
    return [summary["offline_device_count"], summary["device_count"]]


def _report_numbers(report: dict) -> dict:
    firmware = report.get("firmware_comparison", report.get("firmware_overall"))
    sites = report["site_comparison"]
    numbers = {
        "impact": _device_counts(report["impact"]),
        "offline_events": (report["impact"] or {}).get("offline_event_count"),
        "rollouts": report["firmware_rollout_count"],
        "firmware_cohorts": None if firmware is None else {
            "target_version": firmware["target_version"],
            "target": _device_counts(firmware["target_group"]),
            "control": _device_counts(firmware["control_group"]),
            "offline_rate_difference": (
                round(firmware["offline_rate_difference"], 12)
                if firmware["offline_rate_difference"] is not None else None
            ),
        },
        "sites": None if sites is None else {row["site_id"]: _device_counts(row) for row in sites["sites"]},
    }
    if "firmware_by_site" in report:
        by_site = report["firmware_by_site"]
        numbers["firmware_by_site"] = None if by_site is None else {
            row["site_id"]: {
                "target": _device_counts(row["target_group"]),
                "control": _device_counts(row["control_group"]),
                "offline_rate_difference": (
                    round(row["offline_rate_difference"], 12)
                    if row["offline_rate_difference"] is not None else None
                ),
            }
            for row in by_site["sites"]
        }
    if "gateway_alignment" in report:
        gateway = report["gateway_alignment"]
        numbers["gateway"] = None if gateway is None else {
            key: gateway[key] for key in ("site_id", "site_event_id", "first_offline_event_id", "lead_seconds")
        }
    return numbers


def score_report(report: dict, expected: dict, device_ids: list[str], mode: str) -> dict:
    failures: dict[str, list[str]] = {metric: [] for metric in METRICS}
    direction = report["priority"]["direction"]
    if direction != expected["direction"]:
        failures["investigation_direction"].append(f"期望 {expected['direction']}，实际 {direction}。")
    if "priority_site_id" in expected and report["priority"].get("site_id") != expected["priority_site_id"]:
        failures["investigation_direction"].append(f"预期优先核验 {expected['priority_site_id']} 站。")

    numbers = _report_numbers(report)
    for key, value in expected["key_numbers"].items():
        if numbers.get(key) != value:
            failures["key_numbers"].append(f"{key} 期望 {value}，实际 {numbers.get(key)}。")

    expected_ids = {"devices": device_ids, **expected["source_ids"]}
    actual_ids = report["source_ids"]
    for table, ids in expected_ids.items():
        actual = actual_ids.get(table)
        if actual is None or sorted(actual) != sorted(ids):
            absent = sorted(set(ids) - set(actual or []))
            extra = sorted(set(actual or []) - set(ids))
            failures["traceable_evidence"].append(
                f"{table} 的来源 ID 不匹配，缺少 {absent}，多出 {extra}。"
            )

    knowledge = report["knowledge_base"]
    if knowledge and knowledge["status"] == "answered":
        cited = knowledge["sources"]
        if not cited or any(not item.get("id") or not item.get("source") for item in cited):
            failures["traceable_evidence"].append("知识库回答缺少可追溯的引用 ID 或出处。")
        reported = actual_ids.get("knowledge_base")
        if cited != reported and [item["id"] for item in cited] != reported:
            failures["traceable_evidence"].append("报告的知识库来源 ID 与实际引用不一致。")
        if knowledge["simulated"] != (mode == "offline"):
            failures["traceable_evidence"].append("知识库结果的模拟标记与运行模式不一致。")

    firmware_status = report["firmware_source"]["status"]
    if firmware_status != expected["firmware_source_status"]:
        failures["gap_disclosure"].append(
            f"固件来源期望 {expected['firmware_source_status']}，实际 {firmware_status}。"
        )
    gaps = report["missing_evidence"]
    actual_gaps = {gap["source"]: gap["status"] for gap in gaps if gap["reason"].strip()}
    required_gaps = dict(expected["required_gaps"])
    if knowledge and knowledge["status"] != "answered":
        knowledge_gap = "known_issue" if "firmware_comparison" in report else "lookup_known_issue"
        required_gaps[knowledge_gap] = knowledge["status"]
    for source, status in required_gaps.items():
        if actual_gaps.get(source) != status:
            failures["gap_disclosure"].append(f"{source} 应披露为 {status} 并附原因。")
    if "firmware_rollouts" not in required_gaps and any(gap["source"] == "firmware_rollouts" for gap in gaps):
        failures["gap_disclosure"].append("存在可用升级记录，报告却将固件记录列为缺失。")

    return {
        "actual": {
            "direction": direction,
            "priority_site_id": report["priority"].get("site_id"),
            "key_numbers": numbers,
            "source_ids": actual_ids,
            "firmware_source_status": firmware_status,
            "missing_evidence": gaps,
            "knowledge_base": knowledge,
        },
        "passed": {metric: not failures[metric] for metric in METRICS},
        "failures": failures,
    }


def run_evaluation(mode: str) -> dict:
    expected = json.loads(Path(__file__).with_name("expected_results.json").read_text(encoding="utf-8"))
    results = []
    for name, specification in expected["cases"].items():
        try:
            state = run_case(name, mode)
            scored = score_report(state["report"], specification, expected["device_ids"], mode)
            results.append({
                "case": name, "run_id": state["run_id"],
                "execution_trace": state["execution_trace"], **scored,
            })
        except Exception as error:
            reason = f"调查或评分执行失败（{type(error).__name__}）。"
            results.append({
                "case": name,
                "actual": None,
                "passed": dict.fromkeys(METRICS, False),
                "failures": {metric: [reason] for metric in METRICS},
            })
    return {
        "planner": mode,
        "knowledge_mode": "simulated" if mode == "offline" else "configured_rag",
        "cases": results,
        "metrics": {
            metric: {
                "numerator": sum(item["passed"][metric] for item in results),
                "denominator": len(results),
            }
            for metric in METRICS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score six investigation cases.")
    parser.add_argument("--planner", choices=("offline", "deepseek"), required=True)
    args = parser.parse_args()
    result = run_evaluation(args.planner)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(counts["numerator"] != counts["denominator"] for counts in result["metrics"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
