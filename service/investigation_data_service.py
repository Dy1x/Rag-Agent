"""Offline investigation queries over devices, alarms, rollouts and site events."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ContextManager, Literal, Protocol


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("时间必须是带 UTC 偏移的 ISO 8601 字符串")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("时间必须使用 UTC（Z 或 +00:00）")
    return timestamp


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _utc_microseconds(value: str) -> int:
    delta = _parse_utc(value) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True)
class InvestigationScope:
    site_ids: tuple[str, ...]
    start: str
    end: str
    firmware_source: Literal["available", "unavailable"]
    firmware_source_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.site_ids, (list, tuple)) or not self.site_ids:
            raise ValueError("调查范围必须包含至少一个站点 ID")
        if any(not isinstance(site, str) or not site.strip() for site in self.site_ids):
            raise ValueError("站点 ID 必须是非空字符串")
        sites = tuple(site.strip() for site in self.site_ids)
        if len(set(sites)) != len(sites):
            raise ValueError("调查范围中的站点 ID 必须唯一")

        start, end = _parse_utc(self.start), _parse_utc(self.end)
        if start >= end:
            raise ValueError("调查开始时间必须早于结束时间")
        if self.firmware_source not in ("available", "unavailable"):
            raise ValueError("固件来源状态必须是 available 或 unavailable")
        reason = self.firmware_source_reason
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("固件来源说明必须是非空字符串")

        object.__setattr__(self, "site_ids", sites)
        object.__setattr__(self, "start", _format_utc(start))
        object.__setattr__(self, "end", _format_utc(end))

    def to_dict(self) -> dict:
        return {**asdict(self), "site_ids": list(self.site_ids)}

    @property
    def firmware_window(self) -> dict[str, str]:
        start = _parse_utc(self.start) - timedelta(hours=2)
        return {"start": _format_utc(start), "end": self.start}


class OutageReader(Protocol):
    def devices(self, scope: InvestigationScope) -> list[dict]: ...
    def alarm_events(self, scope: InvestigationScope) -> list[dict]: ...
    def firmware_rollouts(self, scope: InvestigationScope) -> list[dict]: ...
    def site_events(self, scope: InvestigationScope) -> list[dict] | None: ...


class EvidenceSource(Protocol):
    def open(self) -> ContextManager[OutageReader]: ...


InvestigationSource = str | Path | EvidenceSource


@contextmanager
def _read_db(db_path: InvestigationSource):
    if not isinstance(db_path, (str, Path)):
        with db_path.open() as reader:
            yield reader
        return
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        # Compare exact instants even when UTC strings have different fractional precision.
        db.create_function("utc_microseconds", 1, _utc_microseconds, deterministic=True)
        db.execute("BEGIN")
        yield SQLiteReader(db)


def _site_placeholders(scope: InvestigationScope) -> str:
    return ", ".join("?" for _ in scope.site_ids)


class SQLiteReader:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def devices(self, scope: InvestigationScope) -> list[dict]:
        rows = self.db.execute(
            f"""
            SELECT device_id, site_id FROM devices
            WHERE site_id IN ({_site_placeholders(scope)})
            ORDER BY device_id
            """,
            scope.site_ids,
        )
        return [dict(row) for row in rows]


    def alarm_events(self, scope: InvestigationScope) -> list[dict]:
        rows = self.db.execute(
            f"""
            SELECT a.event_id, a.device_id, d.site_id, a.event_type, a.occurred_at
            FROM alarm_events AS a JOIN devices AS d ON d.device_id = a.device_id
            WHERE d.site_id IN ({_site_placeholders(scope)})
              AND utc_microseconds(a.occurred_at) >= ?
              AND utc_microseconds(a.occurred_at) < ?
            ORDER BY utc_microseconds(a.occurred_at), a.event_id
            """,
            (*scope.site_ids, _utc_microseconds(scope.start), _utc_microseconds(scope.end)),
        )
        return [dict(row) for row in rows]


    def firmware_rollouts(self, scope: InvestigationScope) -> list[dict]:
        window = scope.firmware_window
        rows = self.db.execute(
            f"""
            SELECT f.rollout_id, f.device_id, d.site_id, f.from_version, f.to_version, f.completed_at
            FROM firmware_rollouts AS f JOIN devices AS d ON d.device_id = f.device_id
            WHERE d.site_id IN ({_site_placeholders(scope)})
              AND utc_microseconds(f.completed_at) >= ?
              AND utc_microseconds(f.completed_at) < ?
            ORDER BY utc_microseconds(f.completed_at), f.rollout_id
            """,
            (*scope.site_ids, _utc_microseconds(window["start"]), _utc_microseconds(window["end"])),
        )
        return [dict(row) for row in rows]


    def site_events(self, scope: InvestigationScope) -> list[dict] | None:
        table = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", ("site_events",),
        ).fetchone()
        if table is None:
            return None
        rows = self.db.execute(
            f"""
            SELECT event_id, site_id, event_type, occurred_at FROM site_events
            WHERE site_id IN ({_site_placeholders(scope)})
              AND utc_microseconds(occurred_at) >= ?
              AND utc_microseconds(occurred_at) < ?
            ORDER BY utc_microseconds(occurred_at), event_id
            """,
            (*scope.site_ids, _utc_microseconds(scope.start), _utc_microseconds(scope.end)),
        )
        return [dict(row) for row in rows]


def _offline_events_by_device(events: list[dict]) -> dict[str, list[str]]:
    offline_events = {}
    for event in events:
        if event["event_type"] == "offline":
            offline_events.setdefault(event["device_id"], []).append(event["event_id"])
    return offline_events


def _device_summary(device_ids: list[str], offline_events: dict[str, list[str]]) -> dict:
    affected = [device_id for device_id in device_ids if device_id in offline_events]
    return {
        "device_count": len(device_ids),
        "offline_device_count": len(affected),
        "offline_rate": len(affected) / len(device_ids) if device_ids else None,
        "offline_device_ids": affected,
        "source_ids": {
            "devices": device_ids,
            "alarm_events": [event_id for device_id in affected for event_id in offline_events[device_id]],
        },
    }


def _firmware_status(scope: InvestigationScope) -> dict:
    reason = None
    if scope.firmware_source == "unavailable":
        reason = scope.firmware_source_reason or "调查范围标记固件发布来源不可用。"
    return {"status": scope.firmware_source, "reason": reason, "window": scope.firmware_window}


def get_alarm_timeline(db_path: InvestigationSource, scope: InvestigationScope) -> dict:
    """Return all scoped alarms and count affected devices from offline events."""
    with _read_db(db_path) as db:
        devices = db.devices(scope)
        events = db.alarm_events(scope)
    offline_events = _offline_events_by_device(events)
    summary = _device_summary([row["device_id"] for row in devices], offline_events)
    summary["source_ids"]["alarm_events"] = [event["event_id"] for event in events]
    return {
        **summary,
        "window": {"start": scope.start, "end": scope.end},
        "event_count": len(events),
        "offline_event_count": sum(len(event_ids) for event_ids in offline_events.values()),
        "events": events,
    }


def get_site_event_timeline(db_path: InvestigationSource, scope: InvestigationScope) -> dict:
    """Read scoped site events and distinguish an older database from an empty window."""
    window = {"start": scope.start, "end": scope.end}
    with _read_db(db_path) as db:
        events = db.site_events(scope)
        if events is None:
            return {
                "status": "unavailable",
                "reason": "调查数据库缺少 site_events 表。",
                "window": window,
                "event_count": None,
                "events": None,
                "source_ids": {"site_events": []},
            }
    return {
        "status": "available",
        "reason": None,
        "window": window,
        "event_count": len(events),
        "events": events,
        "source_ids": {"site_events": [event["event_id"] for event in events]},
    }


def get_firmware_rollouts(db_path: InvestigationSource, scope: InvestigationScope) -> dict:
    """Return successful rollouts in the two hours before the investigation."""
    result = _firmware_status(scope)
    if result["status"] == "unavailable":
        return {**result, "rollout_count": None, "rollouts": None, "source_ids": {"firmware_rollouts": []}}

    with _read_db(db_path) as db:
        rollouts = db.firmware_rollouts(scope)
    return {
        **result,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
        "source_ids": {"firmware_rollouts": [rollout["rollout_id"] for rollout in rollouts]},
    }


def _firmware_comparison_context(scope: InvestigationScope, target_version: str) -> dict:
    if not isinstance(target_version, str) or not target_version.strip():
        raise ValueError("目标版本必须是非空字符串")
    return {
        **_firmware_status(scope),
        "target_version": target_version.strip(),
        "definition": {
            "target_group": "升级窗口内最后一次成功升级的目标版本等于指定版本。",
            "control_group": "范围内其余台账设备，包含窗口内无升级或最后升级至其他版本的设备。",
        },
        "source_ids": {"firmware_rollouts": []},
    }


def _read_firmware_comparison_data(
    db_path: InvestigationSource, scope: InvestigationScope,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    with _read_db(db_path) as db:
        devices = db.devices(scope)
        rollouts = db.firmware_rollouts(scope)
        offline_events = _offline_events_by_device(db.alarm_events(scope))
    return devices, rollouts, offline_events


def _cohort_comparison(
    device_ids: list[str], latest: dict[str, dict], offline_events: dict[str, list[str]], target_version: str,
) -> dict:
    target_ids, control_ids = [], []
    for device_id in device_ids:
        rollout = latest.get(device_id)
        group = target_ids if rollout and rollout["to_version"] == target_version else control_ids
        group.append(device_id)

    result = {"offline_rate_difference": None}
    for name, cohort_ids in (("target_group", target_ids), ("control_group", control_ids)):
        summary = _device_summary(cohort_ids, offline_events)
        summary["source_ids"]["firmware_rollouts"] = [
            latest[device_id]["rollout_id"] for device_id in cohort_ids if device_id in latest
        ]
        result[name] = summary

    if target_ids and control_ids:
        result["offline_rate_difference"] = (
            result["target_group"]["offline_rate"] - result["control_group"]["offline_rate"]
        )
    return result


def _device_ids_by_site(scope: InvestigationScope, devices: list[dict]) -> dict[str, list[str]]:
    sites = {site_id: [] for site_id in scope.site_ids}
    for device in devices:
        sites[device["site_id"]].append(device["device_id"])
    return sites


def compare_firmware_cohorts(
    db_path: InvestigationSource, scope: InvestigationScope, target_version: str,
) -> dict:
    """Compare offline rates by the last successful rollout in the firmware window."""
    result = _firmware_comparison_context(scope, target_version)
    if result["status"] == "unavailable":
        return {**result, "target_group": None, "control_group": None, "offline_rate_difference": None}

    devices, rollouts, offline = _read_firmware_comparison_data(db_path, scope)
    latest = {rollout["device_id"]: rollout for rollout in rollouts}
    result["source_ids"]["firmware_rollouts"] = [rollout["rollout_id"] for rollout in rollouts]
    return {
        **result,
        **_cohort_comparison([row["device_id"] for row in devices], latest, offline, result["target_version"]),
    }


def compare_firmware_cohorts_by_site(
    db_path: InvestigationSource, scope: InvestigationScope, target_version: str,
) -> dict:
    """Compare the same firmware cohorts within each scoped site's device roster."""
    result = _firmware_comparison_context(scope, target_version)
    if result["status"] == "unavailable":
        return {**result, "sites": None}

    devices, rollouts, offline = _read_firmware_comparison_data(db_path, scope)
    latest = {rollout["device_id"]: rollout for rollout in rollouts}
    rollout_ids_by_site = {site_id: [] for site_id in scope.site_ids}
    for rollout in rollouts:
        rollout_ids_by_site[rollout["site_id"]].append(rollout["rollout_id"])
    result["source_ids"]["firmware_rollouts"] = [rollout["rollout_id"] for rollout in rollouts]
    return {
        **result,
        "sites": [
            {
                "site_id": site_id,
                **_cohort_comparison(device_ids, latest, offline, result["target_version"]),
                "source_ids": {"firmware_rollouts": rollout_ids_by_site[site_id]},
            }
            for site_id, device_ids in _device_ids_by_site(scope, devices).items()
        ],
    }


def compare_site_devices(db_path: InvestigationSource, scope: InvestigationScope) -> dict:
    """Compare offline rates using the complete device roster for each requested site."""
    with _read_db(db_path) as db:
        devices = db.devices(scope)
        offline_events = _offline_events_by_device(db.alarm_events(scope))

    sites = [
        {"site_id": site_id, **_device_summary(device_ids, offline_events)}
        for site_id, device_ids in _device_ids_by_site(scope, devices).items()
    ]
    return {
        "sites": sites,
        "sites_with_offline_devices": [site["site_id"] for site in sites if site["offline_device_count"]],
    }


def align_site_gateway_restart(alarms: dict, sites: dict, timeline: dict, scope: InvestigationScope) -> dict | None:
    """Find a prior, sourced restart at the site with the highest scoped offline rate."""
    ranked = sorted(
        (row for row in sites.get("sites", []) if row["offline_rate"] is not None),
        key=lambda row: row["offline_rate"], reverse=True,
    )
    if not alarms or not timeline or len(ranked) < 2:
        return None
    site, next_site = ranked[:2]
    if site["offline_rate"] <= next_site["offline_rate"] or site["offline_device_count"] < 2:
        return None
    site_id = site["site_id"]
    if site_id not in scope.site_ids:
        return None

    source_alarms = set(alarms["source_ids"]["alarm_events"]) & set(site["source_ids"]["alarm_events"])
    first = min(
        (row for row in alarms["events"] if row["site_id"] == site_id and row["event_type"] == "offline"),
        key=lambda row: _parse_utc(row["occurred_at"]), default=None,
    )
    if first is None or first["event_id"] not in source_alarms:
        return None
    start, end = _parse_utc(scope.start), _parse_utc(scope.end)
    first_at = _parse_utc(first["occurred_at"])
    if not start <= first_at < end:
        return None

    site_event_ids = set(timeline["source_ids"]["site_events"])
    restart = max(
        (row for row in timeline["events"]
         if row["site_id"] == site_id and row["event_type"] == "gateway_restart"
         and row["event_id"] in site_event_ids and start <= _parse_utc(row["occurred_at"]) < first_at),
        key=lambda row: _parse_utc(row["occurred_at"]), default=None,
    )
    if restart is None:
        return None
    return {
        "site_id": site_id,
        "site_event_id": restart["event_id"],
        "site_event_at": restart["occurred_at"],
        "first_offline_event_id": first["event_id"],
        "first_offline_at": first["occurred_at"],
        "lead_seconds": (first_at - _parse_utc(restart["occurred_at"])).total_seconds(),
    }
