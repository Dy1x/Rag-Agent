"""MySQL evidence reader and transactional import of an immutable SQLite snapshot."""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Connection, Engine, bindparam, text

from service.investigation_data_service import InvestigationScope, _utc_microseconds


@dataclass(frozen=True)
class MySQLInvestigationSource:
    engine: Engine
    dataset: str

    def __post_init__(self):
        if not self.dataset.strip() or len(self.dataset) > 64:
            raise ValueError("调查数据集名称须为 1–64 个字符")

    def confirmed_interruptions(self, scope):
        """Match explicit device-side cause records to a gateway event, alarm and billing session."""
        sql = text("""SELECT i.interruption_id,i.session_id,i.site_event_id,i.alarm_event_id,
                d.site_id,s.attempt_id FROM ci_session_interruptions i
            JOIN ci_site_events g ON g.dataset=i.dataset AND g.event_id=i.site_event_id
            JOIN ci_alarm_events a ON a.dataset=i.dataset AND a.event_id=i.alarm_event_id AND a.device_id=i.device_id
            JOIN ci_devices d ON d.dataset=i.dataset AND d.device_id=i.device_id AND d.site_id=g.site_id
            JOIN cb_charging_sessions s ON BINARY s.session_id=BINARY i.session_id AND BINARY s.charger_id=BINARY i.charger_id
            WHERE i.dataset=:dataset AND d.site_id IN :sites AND i.cause='gateway_restart'
              AND g.event_type='gateway_restart' AND a.event_type='offline'
              AND g.instant_us>=:lo AND a.instant_us>=g.instant_us AND a.instant_us<:hi
              AND i.instant_us>=g.instant_us AND i.instant_us<:hi
              AND i.instant_us>=TIMESTAMPDIFF(MICROSECOND,'1970-01-01',s.started_at)
              AND (s.charge_end_at IS NULL OR i.instant_us<=TIMESTAMPDIFF(MICROSECOND,'1970-01-01',s.charge_end_at))
            ORDER BY i.session_id,i.instant_us,i.interruption_id""").bindparams(
            bindparam("sites", expanding=True)
        )
        with self.engine.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    sql,
                    dict(
                        dataset=self.dataset,
                        sites=scope.site_ids,
                        lo=_utc_microseconds(scope.start),
                        hi=_utc_microseconds(scope.end),
                    ),
                ).mappings()
            ]

    @contextmanager
    def open(self):
        with (
            self.engine.connect().execution_options(
                isolation_level="REPEATABLE READ"
            ) as conn,
            conn.begin(),
        ):
            metadata = conn.execute(
                text(
                    "SELECT has_site_events FROM ci_datasets WHERE dataset=:d AND content_sha256 IS NOT NULL"
                ),
                {"d": self.dataset},
            ).first()
            if metadata is None:
                raise ValueError("MySQL 中没有已完成导入的指定调查数据集")
            yield MySQLReader(conn, self.dataset, bool(metadata.has_site_events))


@dataclass
class MySQLReader:
    conn: Connection
    dataset: str
    has_site_events: bool

    def _rows(self, query: str, scope: InvestigationScope, window=None):
        stmt = text(query).bindparams(bindparam("sites", expanding=True))
        params = {"dataset": self.dataset, "sites": scope.site_ids}
        if window:
            params.update(
                lo=_utc_microseconds(window["start"]),
                hi=_utc_microseconds(window["end"]),
            )
        return [dict(row) for row in self.conn.execute(stmt, params).mappings()]

    def devices(self, scope):
        return self._rows(
            """SELECT device_id,site_id FROM ci_devices
            WHERE dataset=:dataset AND site_id IN :sites ORDER BY device_id""",
            scope,
        )

    def alarm_events(self, scope):
        return self._rows(
            """SELECT a.event_id,a.device_id,d.site_id,a.event_type,a.occurred_at
            FROM ci_alarm_events a JOIN ci_devices d USING(dataset,device_id)
            WHERE a.dataset=:dataset AND d.site_id IN :sites AND a.instant_us>=:lo AND a.instant_us<:hi
            ORDER BY a.instant_us,a.event_id""",
            scope,
            {"start": scope.start, "end": scope.end},
        )

    def firmware_rollouts(self, scope):
        return self._rows(
            """SELECT f.rollout_id,f.device_id,d.site_id,f.from_version,f.to_version,f.completed_at
            FROM ci_firmware_rollouts f JOIN ci_devices d USING(dataset,device_id)
            WHERE f.dataset=:dataset AND d.site_id IN :sites AND f.instant_us>=:lo AND f.instant_us<:hi
            ORDER BY f.instant_us,f.rollout_id""",
            scope,
            scope.firmware_window,
        )

    def site_events(self, scope):
        if not self.has_site_events:
            return None
        return self._rows(
            """SELECT event_id,site_id,event_type,occurred_at FROM ci_site_events
            WHERE dataset=:dataset AND site_id IN :sites AND instant_us>=:lo AND instant_us<:hi
            ORDER BY instant_us,event_id""",
            scope,
            {"start": scope.start, "end": scope.end},
        )


def apply_investigation_schema(engine: Engine):
    ddl = Path(__file__).with_suffix(".sql").read_text(encoding="utf-8")
    with engine.begin() as conn:
        for statement in ddl.split(";"):
            if statement.strip():
                conn.exec_driver_sql(statement)


# The leading column is the primary key in each source table.
_TABLES = {
    "devices": ("device_id", "site_id", "model"),
    "alarm_events": ("event_id", "device_id", "event_type", "occurred_at"),
    "firmware_rollouts": (
        "rollout_id",
        "device_id",
        "from_version",
        "to_version",
        "completed_at",
    ),
    "site_events": ("event_id", "site_id", "event_type", "occurred_at"),
    "session_interruptions": (
        "interruption_id",
        "device_id",
        "charger_id",
        "session_id",
        "site_event_id",
        "alarm_event_id",
        "cause",
        "occurred_at",
    ),
}


def import_sqlite(source: Path, destination: MySQLInvestigationSource) -> dict:
    """Publish all four evidence streams atomically; retries verify the source contents."""
    uri = source.resolve().as_uri() + "?mode=ro"
    with (
        closing(sqlite3.connect(uri, uri=True)) as db,
        destination.engine.begin() as conn,
    ):
        db.execute("BEGIN")
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = set(_TABLES) - {"site_events", "session_interruptions"} - tables
        if missing:
            raise ValueError("SQLite 缺少调查表：" + ", ".join(sorted(missing)))
        dataset = destination.dataset
        has_events = "site_events" in tables
        conn.execute(
            text("""INSERT INTO ci_datasets(dataset,has_site_events) VALUES (:d,:has_events)
            ON DUPLICATE KEY UPDATE dataset=dataset"""),
            dict(d=dataset, has_events=has_events),
        )
        saved = conn.scalar(
            text("SELECT content_sha256 FROM ci_datasets WHERE dataset=:d FOR UPDATE"),
            {"d": dataset},
        )
        digest = hashlib.sha256()
        counts = {}
        for table, columns in _TABLES.items():
            if table not in tables:
                continue
            digest.update((table + "\n").encode())
            fields = ", ".join(columns)
            instant = next(
                (c for c in columns if c in ("occurred_at", "completed_at")), None
            )
            names = ["dataset", *columns] + (["instant_us"] if instant else [])
            insert = text(
                f"INSERT INTO ci_{table} ({', '.join(names)}) VALUES ({', '.join(':' + k for k in names)})"
            )
            counts[table] = 0
            cursor = db.execute(
                f"SELECT {fields} FROM {table} ORDER BY {columns[0]} COLLATE BINARY"
            )
            while batch := cursor.fetchmany(500):
                values = []
                for row in batch:
                    digest.update(
                        (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode()
                    )
                    if not saved:
                        data = dict(zip(columns, row), dataset=dataset)
                        if instant:
                            data["instant_us"] = _utc_microseconds(data[instant])
                        values.append(data)
                counts[table] += len(batch)
                if values:
                    conn.execute(insert, values)
        fingerprint = digest.hexdigest()
        if saved and saved != fingerprint:
            raise ValueError("同名调查数据集已有不同内容，请为新快照指定新名称")
        if not saved:
            conn.execute(
                text("UPDATE ci_datasets SET content_sha256=:digest WHERE dataset=:d"),
                {"d": dataset, "digest": fingerprint},
            )
    return {
        "dataset": dataset,
        "rows": counts,
        "sha256": fingerprint,
        "imported": saved is None,
    }
