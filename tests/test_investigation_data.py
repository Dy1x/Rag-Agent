import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.investigation.run_baseline import create_fixture_db, run_baseline
from eval.investigation.run_baseline import load_case as load_fixture
from service.investigation_data_service import (
    InvestigationScope,
    compare_firmware_cohorts,
    compare_firmware_cohorts_by_site,
    compare_site_devices,
    get_alarm_timeline,
    get_firmware_rollouts,
    get_site_event_timeline,
)


class InvestigationDataTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db_path = Path(directory.name) / "investigation.sqlite3"

    def load_case(self, name):
        devices, case = load_fixture(name)
        create_fixture_db(self.db_path, devices, case)
        return InvestigationScope(**case["scope"]), case["target_version"]

    def create_boundary_db(self):
        devices = [
            {"device_id": device_id, "site_id": site_id, "model": "DC-120"}
            for device_id, site_id in (("A01", "A"), ("A02", "A"), ("A03", "A"), ("A04", "A"), ("B01", "B"))
        ]
        create_fixture_db(self.db_path, devices, {"alarm_events": [], "firmware_rollouts": []})
        return InvestigationScope(
            ("A", "EMPTY"), "2026-11-21T14:00:00+00:00", "2026-11-21T15:00:00Z", "available",
        )

    def test_firmware_fixture_compares_target_version_across_both_sites(self):
        scope, version = self.load_case("firmware")
        result = run_baseline(self.db_path, scope, version)
        self.assertEqual(result["scope"]["site_ids"], ["A", "B"])
        self.assertEqual(result["alarm_summary"]["device_count"], 16)
        self.assertEqual(result["alarm_summary"]["offline_device_count"], 7)
        self.assertEqual(
            [(site["site_id"], site["offline_device_count"], site["device_count"])
             for site in result["site_comparison"]["sites"]],
            [("A", 4, 8), ("B", 3, 8)],
        )

        firmware = result["firmware_comparison"]
        self.assertEqual(firmware["status"], "available")
        self.assertEqual(firmware["rollout_count"], 8)
        self.assertEqual(
            [(group["offline_device_count"], group["device_count"], group["offline_rate"])
             for group in (firmware["target_group"], firmware["control_group"])],
            [(6, 8, 0.75), (1, 8, 0.125)],
        )
        self.assertEqual(firmware["offline_rate_difference"], 0.625)
        self.assertEqual(
            [sum(rollout["site_id"] == site for rollout in firmware["rollouts"]) for site in scope.site_ids],
            [4, 4],
        )
        self.assertIn("fw-rollout-A01", firmware["target_group"]["source_ids"]["firmware_rollouts"])
        self.assertIn("fw-rollout-B01", firmware["target_group"]["source_ids"]["firmware_rollouts"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertNotIn("site_event_timeline", result)
        self.assertNotIn("firmware_comparison_by_site", result)

    def test_overlap_fixture_totals_and_per_site_cohorts_disagree(self):
        scope, version = self.load_case("rollout_site_overlap")
        self.assertEqual(version, "2.4.0")
        result = run_baseline(self.db_path, scope, version, include_site_evidence=True)

        alarms = result["alarm_summary"]
        self.assertEqual(
            (alarms["device_count"], alarms["offline_device_count"], alarms["offline_event_count"]),
            (30, 15, 16),
        )
        self.assertEqual(
            [(row["site_id"], row["offline_device_count"], row["device_count"])
             for row in result["site_comparison"]["sites"]],
            [("A", 12, 15), ("B", 3, 15)],
        )

        overall = result["firmware_comparison"]
        per_site = result["firmware_comparison_by_site"]
        self.assertEqual(overall["rollout_count"], 15)
        self.assertEqual(overall["source_ids"], per_site["source_ids"])
        self.assertAlmostEqual(overall["offline_rate_difference"], 0.2)

        def counts(group):
            return group["offline_device_count"], group["device_count"], group["offline_rate"]

        self.assertEqual(counts(overall["target_group"]), (9, 15, 0.6))
        self.assertEqual(counts(overall["control_group"]), (6, 15, 0.4))
        a, b = per_site["sites"]
        self.assertEqual((a["site_id"], b["site_id"]), ("A", "B"))
        self.assertEqual(counts(a["target_group"]), (8, 10, 0.8))
        self.assertEqual(counts(a["control_group"]), (4, 5, 0.8))
        self.assertEqual(counts(b["target_group"]), (1, 5, 0.2))
        self.assertEqual(counts(b["control_group"]), (2, 10, 0.2))
        self.assertEqual((a["offline_rate_difference"], b["offline_rate_difference"]), (0.0, 0.0))

        a_sources = a["target_group"]["source_ids"]
        self.assertIn("A01", a_sources["devices"])
        self.assertIn("overlap-alarm-A01", a_sources["alarm_events"])
        self.assertIn("overlap-alarm-A01-repeat", a_sources["alarm_events"])
        self.assertEqual(a["target_group"]["offline_device_ids"].count("A01"), 1)
        self.assertIn("overlap-rollout-A01", a_sources["firmware_rollouts"])
        self.assertNotIn("overlap-rollout-B01", a["source_ids"]["firmware_rollouts"])
        self.assertEqual(b["control_group"]["source_ids"]["alarm_events"], ["overlap-alarm-B06", "overlap-alarm-B07"])
        self.assertEqual(result["site_event_timeline"]["source_ids"], {"site_events": ["gw-A-01"]})

    def test_overlap_gateway_restart_precedes_same_site_dropout_by_two_minutes(self):
        scope, _ = self.load_case("rollout_site_overlap")
        timeline = get_site_event_timeline(self.db_path, scope)
        self.assertEqual((timeline["status"], timeline["event_count"]), ("available", 1))
        restart, = timeline["events"]
        self.assertEqual(
            (restart["event_id"], restart["site_id"], restart["event_type"]),
            ("gw-A-01", "A", "gateway_restart"),
        )
        first_offline = next(
            event for event in get_alarm_timeline(self.db_path, scope)["events"]
            if event["site_id"] == "A" and event["event_type"] == "offline"
        )
        self.assertEqual(first_offline["event_id"], "overlap-alarm-A01")
        restart_time = datetime.fromisoformat(restart["occurred_at"].replace("Z", "+00:00"))
        dropout_time = datetime.fromisoformat(first_offline["occurred_at"].replace("Z", "+00:00"))
        self.assertEqual(dropout_time - restart_time, timedelta(minutes=2))

    def test_site_fixture_keeps_site_b_with_zero_offline_devices(self):
        scope, version = self.load_case("site")
        result = run_baseline(self.db_path, scope, version)
        sites = result["site_comparison"]
        self.assertEqual(sites["sites_with_offline_devices"], ["A"])
        self.assertEqual(
            [(site["site_id"], site["offline_device_count"], site["device_count"], site["offline_rate"])
             for site in sites["sites"]],
            [("A", 6, 8, 0.75), ("B", 0, 8, 0.0)],
        )
        firmware = result["firmware_comparison"]
        self.assertEqual(firmware["target_group"]["offline_device_ids"], ["A01", "A02", "A03"])
        self.assertEqual(firmware["control_group"]["offline_device_ids"], ["A05", "A06", "A07"])
        self.assertEqual(firmware["offline_rate_difference"], 0.0)
        self.assertEqual(
            [sum(rollout["site_id"] == site for rollout in firmware["rollouts"]) for site in scope.site_ids],
            [4, 4],
        )

    def test_repeated_alarm_is_kept_in_timeline_and_device_is_counted_once(self):
        scope, _ = self.load_case("firmware")
        alarm = get_alarm_timeline(self.db_path, scope)
        self.assertEqual(alarm["offline_event_count"], 8)
        self.assertEqual(alarm["offline_device_count"], 7)
        self.assertEqual(alarm["offline_device_ids"].count("A01"), 1)
        self.assertEqual(
            [event["event_id"] for event in alarm["events"] if event["device_id"] == "A01"],
            ["fw-alarm-01", "fw-alarm-04"],
        )
        self.assertEqual(len(alarm["source_ids"]["alarm_events"]), 8)

    def test_unavailable_firmware_and_empty_available_firmware_are_distinct(self):
        scope, version = self.load_case("missing")
        result = run_baseline(self.db_path, scope, version)
        self.assertEqual(result["alarm_summary"]["offline_device_count"], 5)
        self.assertEqual(
            [site["offline_device_count"] for site in result["site_comparison"]["sites"]], [3, 2],
        )
        firmware = result["firmware_comparison"]
        self.assertEqual(firmware["status"], "unavailable")
        self.assertIsNone(firmware["rollouts"])
        self.assertIsNone(firmware["rollout_count"])
        self.assertIsNone(firmware["target_group"])
        self.assertIsNone(firmware["control_group"])
        self.assertEqual(result["missing_evidence"], [{
            "source": "firmware_rollouts",
            "status": "unavailable",
            "reason": scope.firmware_source_reason,
        }])

        available = replace(scope, firmware_source="available", firmware_source_reason=None)
        result = run_baseline(self.db_path, available, version)
        firmware = result["firmware_comparison"]
        self.assertEqual(firmware["status"], "available")
        self.assertEqual(firmware["rollouts"], [])
        self.assertEqual(firmware["rollout_count"], 0)
        self.assertEqual(firmware["target_group"]["device_count"], 0)
        self.assertIsNone(firmware["target_group"]["offline_rate"])
        self.assertEqual(firmware["control_group"]["device_count"], 16)
        self.assertEqual(firmware["control_group"]["offline_device_count"], 5)
        self.assertIsNone(firmware["offline_rate_difference"])
        self.assertEqual(result["missing_evidence"], [])

    def test_alarm_window_is_utc_half_open_and_excludes_other_sites(self):
        scope = self.create_boundary_db()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.executemany("INSERT INTO alarm_events VALUES (?, ?, ?, ?)", [
                ("before", "A02", "offline", "2026-11-21T13:59:59.999999Z"),
                ("start", "A01", "offline", "2026-11-21T14:00:00+00:00"),
                ("warning", "A02", "warning", "2026-11-21T14:12:00Z"),
                ("outside-site", "B01", "offline", "2026-11-21T14:30:00Z"),
                ("last", "A03", "offline", "2026-11-21T14:59:59.999999Z"),
                ("end", "A02", "offline", "2026-11-21T15:00:00Z"),
                ("after", "A02", "offline", "2026-11-21T15:00:00.000001Z"),
            ])
        alarm = get_alarm_timeline(self.db_path, scope)
        self.assertEqual(alarm["window"]["start"], "2026-11-21T14:00:00Z")
        self.assertEqual(alarm["source_ids"]["alarm_events"], ["start", "warning", "last"])
        self.assertEqual(alarm["event_count"], 3)
        self.assertEqual(alarm["offline_event_count"], 2)
        self.assertEqual(alarm["offline_device_ids"], ["A01", "A03"])

        a, empty = compare_site_devices(self.db_path, scope)["sites"]
        self.assertEqual(a["offline_rate"], 0.5)
        self.assertEqual(a["source_ids"]["alarm_events"], ["start", "last"])
        self.assertEqual((empty["site_id"], empty["device_count"]), ("EMPTY", 0))
        self.assertIsNone(empty["offline_rate"])

    def test_site_event_window_is_half_open_ordered_and_scoped(self):
        scope = self.create_boundary_db()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.executemany("INSERT INTO site_events VALUES (?, ?, ?, ?)", [
                ("before", "A", "gateway_restart", "2026-11-21T13:59:59.999999Z"),
                ("start-b", "A", "gateway_restart", "2026-11-21T14:00:00+00:00"),
                ("start-a", "A", "gateway_alert", "2026-11-21T14:00:00Z"),
                ("outside-site", "B", "gateway_restart", "2026-11-21T14:30:00Z"),
                ("last", "A", "gateway_restart", "2026-11-21T14:59:59.999999Z"),
                ("end", "A", "gateway_restart", "2026-11-21T15:00:00Z"),
                ("after", "A", "gateway_restart", "2026-11-21T15:00:00.000001Z"),
            ])
        timeline = get_site_event_timeline(self.db_path, scope)
        self.assertEqual(timeline["window"], {"start": scope.start, "end": scope.end})
        self.assertEqual(timeline["event_count"], 3)
        self.assertEqual(timeline["source_ids"]["site_events"], ["start-a", "start-b", "last"])
        self.assertEqual([event["event_id"] for event in timeline["events"]], ["start-a", "start-b", "last"])

    def test_site_event_empty_window_differs_from_older_database(self):
        scope, version = self.load_case("firmware")
        empty = get_site_event_timeline(self.db_path, scope)
        self.assertEqual((empty["status"], empty["event_count"], empty["events"]), ("available", 0, []))
        self.assertEqual(empty["source_ids"], {"site_events": []})

        # Reproduce the schema of a run persisted before site events were added.
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("DROP TABLE site_events")
        result = run_baseline(self.db_path, scope, version, include_site_evidence=True)
        unavailable = result["site_event_timeline"]
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["events"])
        self.assertIsNone(unavailable["event_count"])
        self.assertIn("site_events", unavailable["reason"])
        self.assertEqual(result["missing_evidence"], [{
            "source": "site_events", "status": "unavailable", "reason": unavailable["reason"],
        }])
        self.assertEqual(result["alarm_summary"]["offline_device_count"], 7)

    def test_firmware_window_and_latest_successful_rollout_per_device(self):
        scope = self.create_boundary_db()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.executemany("INSERT INTO firmware_rollouts VALUES (?, ?, ?, ?, ?)", [
                ("before", "A02", "1.0", "2.0", "2026-11-21T11:59:59.999999Z"),
                ("start", "A01", "1.0", "2.0", "2026-11-21T12:00:00+00:00"),
                ("earlier-target", "A03", "1.0", "2.0", "2026-11-21T12:35:00Z"),
                ("outside-site", "B01", "1.0", "2.0", "2026-11-21T13:15:00Z"),
                ("tie-b", "A04", "1.5", "2.0", "2026-11-21T13:45:00+00:00"),
                ("tie-a", "A04", "1.0", "1.5", "2026-11-21T13:45:00Z"),
                ("latest-other", "A03", "2.0", "2.1", "2026-11-21T13:59:59.999999Z"),
                ("end", "A02", "1.0", "2.0", "2026-11-21T14:00:00Z"),
                ("after", "A02", "2.0", "2.1", "2026-11-21T14:00:00.000001Z"),
            ])
        raw = get_firmware_rollouts(self.db_path, scope)
        self.assertEqual(raw["window"], {"start": "2026-11-21T12:00:00Z", "end": scope.start})
        self.assertEqual(
            raw["source_ids"]["firmware_rollouts"],
            ["start", "earlier-target", "tie-a", "tie-b", "latest-other"],
        )
        cohorts = compare_firmware_cohorts(self.db_path, scope, "2.0")
        self.assertEqual(cohorts["target_group"]["source_ids"]["devices"], ["A01", "A04"])
        self.assertEqual(cohorts["target_group"]["source_ids"]["firmware_rollouts"], ["start", "tie-b"])
        self.assertEqual(cohorts["control_group"]["source_ids"]["devices"], ["A02", "A03"])
        self.assertEqual(cohorts["control_group"]["source_ids"]["firmware_rollouts"], ["latest-other"])
        site_a = compare_firmware_cohorts_by_site(self.db_path, scope, "2.0")["sites"][0]
        self.assertEqual(site_a["target_group"], cohorts["target_group"])
        self.assertEqual(site_a["control_group"], cohorts["control_group"])

    def test_firmware_by_site_handles_empty_groups_and_unavailable_source(self):
        scope = self.create_boundary_db()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.executemany("INSERT INTO firmware_rollouts VALUES (?, ?, ?, ?, ?)", [
                (device_id, device_id, "1.0", "2.0", "2026-11-21T13:00:00Z")
                for device_id in ("A01", "A02", "A03", "A04")
            ])

        a, empty = compare_firmware_cohorts_by_site(self.db_path, scope, "2.0")["sites"]
        self.assertEqual(a["target_group"]["device_count"], 4)
        self.assertEqual(a["control_group"]["device_count"], 0)
        self.assertIsNone(a["control_group"]["offline_rate"])
        self.assertIsNone(a["offline_rate_difference"])
        self.assertEqual(empty["site_id"], "EMPTY")
        self.assertIsNone(empty["target_group"]["offline_rate"])
        self.assertIsNone(empty["control_group"]["offline_rate"])
        self.assertIsNone(empty["offline_rate_difference"])

        a = compare_firmware_cohorts_by_site(self.db_path, scope, "3.0")["sites"][0]
        self.assertEqual(a["target_group"]["device_count"], 0)
        self.assertIsNone(a["target_group"]["offline_rate"])
        self.assertIsNone(a["offline_rate_difference"])
        unavailable = compare_firmware_cohorts_by_site(
            self.db_path,
            replace(scope, firmware_source="unavailable", firmware_source_reason="升级来源不可查询。"),
            "2.0",
        )
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["reason"], "升级来源不可查询。")
        self.assertIsNone(unavailable["sites"])
        self.assertEqual(unavailable["source_ids"], {"firmware_rollouts": []})

    def test_scope_validation_and_parameter_binding(self):
        scope, _ = self.load_case("firmware")
        for invalid in (
            {"site_ids": []},
            {"site_ids": ("A", " A ")},
            {"start": scope.end},
            {"start": "2026-11-21T14:00:00"},
            {"end": "2026-11-22T08:00:00+08:00"},
            {"firmware_source": "unknown"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(scope, **invalid)

        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("INSERT INTO devices VALUES (?, ?, ?)", ("C01", "O'Brien", "DC-120"))
            db.execute(
                "INSERT INTO alarm_events VALUES (?, ?, ?, ?)",
                ("quoted-site", "C01", "offline", scope.start),
            )
        quoted_scope = replace(scope, site_ids=("O'Brien",))
        alarm = get_alarm_timeline(self.db_path, quoted_scope)
        self.assertEqual(alarm["source_ids"], {"devices": ["C01"], "alarm_events": ["quoted-site"]})


if __name__ == "__main__":
    unittest.main()
