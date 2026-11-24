"""Run a fixed, offline sequence of investigation queries against a fixture."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from service.investigation_data_service import (
    InvestigationScope,
    compare_firmware_cohorts,
    compare_firmware_cohorts_by_site,
    compare_site_devices,
    get_alarm_timeline,
    get_firmware_rollouts,
    get_site_event_timeline,
)


def load_case(name: str) -> tuple[list[dict], dict]:
    data = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    if name not in data["cases"]:
        raise ValueError(f"未知样例 {name}，可选样例为 {'、'.join(data['cases'])}。")
    case = data["cases"][name]
    devices = case["devices"] if "devices" in case else data["devices"]
    return devices, case


def create_fixture_db(db_path: Path, devices: list[dict], case: dict) -> None:
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with closing(sqlite3.connect(db_path)) as db:
        db.executescript(schema)
        with db:
            db.executemany(
                "INSERT INTO devices (device_id, site_id, model) VALUES (:device_id, :site_id, :model)",
                devices,
            )
            db.executemany(
                """
                INSERT INTO alarm_events (event_id, device_id, event_type, occurred_at)
                VALUES (:event_id, :device_id, :event_type, :occurred_at)
                """,
                case["alarm_events"],
            )
            db.executemany(
                """
                INSERT INTO firmware_rollouts
                    (rollout_id, device_id, from_version, to_version, completed_at)
                VALUES (:rollout_id, :device_id, :from_version, :to_version, :completed_at)
                """,
                case["firmware_rollouts"],
            )
            db.executemany(
                """
                INSERT INTO site_events (event_id, site_id, event_type, occurred_at)
                VALUES (:event_id, :site_id, :event_type, :occurred_at)
                """,
                case.get("site_events", []),
            )


def run_baseline(
    db_path: Path, scope: InvestigationScope, target_version: str, *, include_site_evidence: bool = False,
) -> dict:
    alarm_summary = get_alarm_timeline(db_path, scope)
    rollouts = get_firmware_rollouts(db_path, scope)
    firmware_comparison = compare_firmware_cohorts(db_path, scope, target_version)
    site_comparison = compare_site_devices(db_path, scope)
    firmware_comparison.update(rollout_count=rollouts["rollout_count"], rollouts=rollouts["rollouts"])

    missing_evidence = []
    if rollouts["status"] == "unavailable":
        missing_evidence.append({
            "source": "firmware_rollouts",
            "status": rollouts["status"],
            "reason": rollouts["reason"],
        })

    result = {
        "scope": scope.to_dict(),
        "alarm_summary": alarm_summary,
        "site_comparison": site_comparison,
        "firmware_comparison": firmware_comparison,
        "missing_evidence": missing_evidence,
        "interpretation": "站点分布和组间掉线率差异是观测关联。归因仍需核对站点网络、供电或设备诊断记录。",
    }
    if include_site_evidence:
        timeline = get_site_event_timeline(db_path, scope)
        result["site_event_timeline"] = timeline
        result["firmware_comparison_by_site"] = compare_firmware_cohorts_by_site(db_path, scope, target_version)
        result["interpretation"] += " 优先核验早于同站掉线的网关事件，并对照网络和设备日志。"
        if timeline["status"] == "unavailable":
            missing_evidence.append({
                "source": "site_events",
                "status": timeline["status"],
                "reason": timeline["reason"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline charging-station investigation baseline.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--include-site-evidence", action="store_true")
    args = parser.parse_args()
    try:
        devices, case = load_case(args.case)
    except ValueError as error:
        parser.error(str(error))
    scope = InvestigationScope(**case["scope"])
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "investigation.sqlite3"
        create_fixture_db(db_path, devices, case)
        result = run_baseline(db_path, scope, case["target_version"], include_site_evidence=args.include_site_evidence)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
