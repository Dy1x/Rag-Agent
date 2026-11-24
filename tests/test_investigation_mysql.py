import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import text
from billing_mysql_fixture import MysqlCase
from agent.tools.investigation_tools import investigate_outage
from eval.investigation.run_baseline import create_fixture_db, load_case, run_baseline
from eval.investigation.run_multi_agent import prepare_fixture
from service.billing.case_store import CaseStore
from service.investigation_data_service import (
    InvestigationScope,
    get_alarm_timeline,
    get_site_event_timeline,
)
from service.investigation_mysql import (
    MySQLInvestigationSource,
    apply_investigation_schema,
    import_sqlite,
)
from service.investigation_service import multi_dependencies_for, run_investigation


class InvestigationMySQLTests(MysqlCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        apply_investigation_schema(cls.engine)

    def test_all_seven_fixtures_match_sqlite_reports_and_provenance(self):
        names = json.loads(
            (
                Path(__file__).resolve().parents[1] / "eval/investigation/cases.json"
            ).read_text(encoding="utf-8")
        )["cases"]
        for name in names:
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                path = Path(tmp) / "source.sqlite"
                devices, case = load_case(name)
                create_fixture_db(path, devices, case)
                scope = InvestigationScope(**case["scope"])
                source = MySQLInvestigationSource(self.engine, name)
                self.assertTrue(import_sqlite(path, source)["imported"])
                self.assertFalse(import_sqlite(path, source)["imported"])
                baseline = run_baseline(
                    path, scope, case["target_version"], include_site_evidence=True
                )
                self.assertEqual(
                    run_baseline(
                        source,
                        scope,
                        case["target_version"],
                        include_site_evidence=True,
                    ),
                    baseline,
                )

                def investigate(src):
                    return run_investigation(
                        "核查掉线影响",
                        scope,
                        multi_dependencies_for(src, "offline"),
                        mode="offline",
                        simulated_data=True,
                    )

                old = investigate(path)
                new = investigate(source)
                old_report = json.dumps(old["report"], sort_keys=True).replace(
                    old["run_id"], "<run>"
                )
                new_report = json.dumps(new["report"], sort_keys=True).replace(
                    new["run_id"], "<run>"
                )
                self.assertEqual(old_report, new_report)
                self.assertEqual(old["shared_queries"], new["shared_queries"])
                store = CaseStore(self.engine)
                self.assertEqual(store.get_run(new["run_id"])["report"], new)
                events = store.get_trace(new["run_id"])
                self.assertTrue(any(e["event_type"] == "tool_call" for e in events))
                self.assertEqual(events[-1]["payload"]["status"], "completed")

    def test_legacy_site_events_and_atomic_import(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.sqlite"
            cfg = prepare_fixture("site", path)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("DROP TABLE site_events")
            source = MySQLInvestigationSource(self.engine, "legacy")
            import_sqlite(path, source)
            scope = InvestigationScope(**cfg["scope"])
            self.assertEqual(
                get_site_event_timeline(path, scope),
                get_site_event_timeline(source, scope),
            )
            previous = get_alarm_timeline(source, scope)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE alarm_events SET occurred_at='bad'")
            with self.assertRaisesRegex(ValueError, "同名"):
                import_sqlite(path, source)
            self.assertEqual(get_alarm_timeline(source, scope), previous)
            with self.assertRaises(ValueError):
                import_sqlite(
                    path, MySQLInvestigationSource(self.engine, "invalid_snapshot")
                )
            with self.engine.connect() as conn:
                self.assertEqual(
                    conn.scalar(
                        text(
                            "SELECT COUNT(*) FROM ci_datasets WHERE dataset='invalid_snapshot'"
                        )
                    ),
                    0,
                )

    def test_fractional_utc_boundaries_and_case_sensitive_sites(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.sqlite"
            events = [
                dict(
                    event_id=f"E{i}",
                    device_id=device,
                    event_type="offline",
                    occurred_at=stamp,
                )
                for i, (device, stamp) in enumerate(
                    [
                        ("a", "2026-09-10T00:00:00.000009Z"),
                        ("a", "2026-09-10T00:00:00.000010+00:00"),
                        ("a", "2026-09-10T00:00:00.000011Z"),
                        ("A", "2026-09-10T00:00:00.000010Z"),
                    ]
                )
            ]
            create_fixture_db(
                path,
                [dict(device_id=d, site_id=d + "'s", model="X") for d in ("a", "A")],
                {"alarm_events": events, "firmware_rollouts": []},
            )
            source = MySQLInvestigationSource(self.engine, "boundary")
            import_sqlite(path, source)
            scope = InvestigationScope(
                ["a's"],
                "2026-09-10T00:00:00.000010Z",
                "2026-09-10T00:00:00.000011Z",
                "available",
            )
            self.assertEqual(
                get_alarm_timeline(source, scope), get_alarm_timeline(path, scope)
            )
            self.assertEqual(
                get_alarm_timeline(source, scope)["source_ids"]["alarm_events"], ["E1"]
            )

    def test_registered_chat_tool_uses_configured_mysql(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.sqlite"
            cfg = prepare_fixture("rollout_site_overlap", path)
            import_sqlite(path, MySQLInvestigationSource(self.engine, "chat"))
        scope = cfg["scope"]
        env = {
            "INVESTIGATION_BACKEND": "mysql",
            "INVESTIGATION_DATASET": "chat",
            "INVESTIGATION_MYSQL_DATABASE": self.database,
            "INVESTIGATION_DB_PATH": "",
            "INVESTIGATION_FIRMWARE_SOURCE": "available",
            "INVESTIGATION_PLANNER": "offline",
            "INVESTIGATION_SIMULATED_DATA": "1",
        }
        with patch.dict(os.environ, env):
            args = dict(
                site_ids=scope["site_ids"],
                start_utc=scope["start"],
                end_utc=scope["end"],
                question="核查网关重启",
            )
            result = investigate_outage.invoke(args)
            self.assertEqual(result["status"], "completed", result)
            self.assertTrue(result["case_id"].startswith("CASE-"))
            self.assertEqual(result["report"]["impact"]["offline_device_count"], 15)
            with patch.dict(os.environ, {"INVESTIGATION_DATASET": "unknown"}):
                self.assertEqual(
                    investigate_outage.invoke(args)["status"],
                    "data_source_unconfigured",
                )
