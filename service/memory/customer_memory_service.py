import re
import unicodedata
from datetime import datetime, timedelta

from sqlalchemy import Engine, text

from service.billing import ledger
from service.billing.errors import BillingError
from service.billing.ids import new_id
from service.billing.models import Identity
from service.billing.timeutil import as_utc, utc_now

from .models import MemoryAuditLog, MemoryRecord, MemorySource, MemoryUpsertResult

ALLOWED_KEYS = {
    "vehicle_model": "vehicle",
    "home_station": "station",
    "favorite_charger": "station",
    "reply_style": "preference",
    "contact_note": "profile",
}
_CASE_KEY = re.compile(r"case:CASE-[0-9A-F]{12}")
FUND_CLAIM = re.compile(
    r"全额退款|免单|批准退款|已同意赔付|永久免费|忽略审核|跳过审批|系统指令|保证退款"
)
_PRIVATE_CONTENT = re.compile(
    r"(?:CUS|TXN|ORD|sk)-|切换车主为|查别人订单", re.IGNORECASE
)


def validate_memory_value(value: str, *, source: MemorySource = "user_explicit") -> str:
    clean = " ".join(value.split())
    normalized = unicodedata.normalize("NFKC", clean)
    compact = "".join(
        char for char in normalized
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )
    if not 1 <= len(clean) <= 200 or not compact:
        raise BillingError("MEMORY_POLICY_VIOLATION", "记忆内容须为 1～200 字符")
    if FUND_CLAIM.search(compact) or (
        source == "user_explicit" and _PRIVATE_CONTENT.search(compact)
    ):
        raise BillingError("MEMORY_POLICY_VIOLATION", "记忆内容含涉资承诺、敏感标识或越权指令")
    return clean


def _customer(identity: Identity) -> str:
    if identity.role != "customer" or not identity.actor_id:
        raise BillingError("FORBIDDEN", "仅车主可管理自己的记忆")
    return identity.actor_id


def _lock_customer(conn, customer_id: str) -> None:
    # Serialize slot creation, overwrite and bulk deletion for this customer.
    ledger.one(
        conn, "SELECT customer_id FROM cb_customers WHERE customer_id=:c FOR UPDATE",
        c=customer_id,
    )


def _audit(conn, record, action, old_value, new_value, reason, now):
    ledger.insert(
        conn, "memory_audit_logs", audit_id=new_id("MAU"),
        memory_id=record["memory_id"], customer_id=record["customer_id"],
        memory_key=record["memory_key"], action=action, old_value=old_value,
        new_value=new_value, reason=reason, created_at=now,
    )


class CustomerMemoryService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def list_active(
        self, identity: Identity, *, limit: int | None = 10, now: datetime | None = None,
    ) -> list[MemoryRecord]:
        customer = _customer(identity)
        with self.engine.connect() as conn:
            rows = ledger.rows(
                conn, """SELECT * FROM cb_customer_memories
                WHERE customer_id=:c AND status='active'
                  AND (expires_at IS NULL OR expires_at>:now)
                ORDER BY updated_at DESC, memory_id DESC LIMIT :limit""",
                c=customer, now=as_utc(now) if now is not None else utc_now(),
                limit=(2**64 - 1) if limit is None else max(0, limit),
            )
        return [MemoryRecord.model_validate(row) for row in rows]

    def upsert_memory(
        self, identity: Identity, *, memory_key: str, memory_value: str,
        source: MemorySource = "user_explicit", session_id: str | None = None,
        ttl_days: int | None = None, reason: str | None = None,
    ) -> MemoryUpsertResult:
        customer = _customer(identity)
        category = None
        if source == "user_explicit":
            category = ALLOWED_KEYS.get(memory_key)
        elif source == "case_auto" and _CASE_KEY.fullmatch(memory_key):
            category = "case_summary"
        if category is None:
            raise BillingError("MEMORY_POLICY_VIOLATION", "不允许写入该记忆槽位")
        value = validate_memory_value(memory_value, source=source)
        now = utc_now()
        record = dict(
            customer_id=customer, memory_key=memory_key, category=category,
            memory_value=value, source=source, source_session_id=session_id,
            status="active", updated_at=now,
            expires_at=now + timedelta(days=ttl_days) if ttl_days is not None else None,
        )
        with self.engine.begin() as conn:
            _lock_customer(conn, customer)
            previous = ledger.rows(
                conn, """SELECT * FROM cb_customer_memories
                WHERE customer_id=:customer AND memory_key=:key""",
                customer=customer, key=memory_key,
            )
            old_value = None
            if not previous:
                action = "create"
                record.update(memory_id=new_id("MEM"), version=1, created_at=now)
                ledger.insert(conn, "customer_memories", **record)
            else:
                old = previous[0]
                old_value = old["memory_value"]
                if old["status"] == "deleted":
                    action = "restore"
                elif value != old_value:
                    action = "update"
                else:
                    action = "unchanged"
                record.update(
                    memory_id=old["memory_id"], created_at=old["created_at"],
                    version=old["version"] + (action != "unchanged"),
                )
                conn.execute(
                    text("""UPDATE cb_customer_memories
                    SET memory_value=:memory_value, source_session_id=:source_session_id,
                        version=:version, status=:status, updated_at=:updated_at,
                        expires_at=:expires_at
                    WHERE customer_id=:customer_id AND memory_key=:memory_key"""),
                    record,
                )
            if action != "unchanged":
                _audit(conn, record, action, old_value, value, reason, now)
        return MemoryUpsertResult(memory=MemoryRecord.model_validate(record), action=action)

    def delete_memory(
        self, identity: Identity, *, memory_key: str | None = None,
        reason: str | None = None,
    ) -> int:
        customer = _customer(identity)
        with self.engine.begin() as conn:
            _lock_customer(conn, customer)
            rows = ledger.rows(
                conn, """SELECT * FROM cb_customer_memories
                WHERE customer_id=:c AND status='active'
                  AND (:key IS NULL OR memory_key=:key)""",
                c=customer, key=memory_key,
            )
            now = utc_now()
            for record in rows:
                conn.execute(
                    text("""UPDATE cb_customer_memories SET status='deleted', updated_at=:now
                    WHERE customer_id=:c AND memory_id=:id"""),
                    {"c": customer, "id": record["memory_id"], "now": now},
                )
                _audit(conn, record, "delete", record["memory_value"], None, reason, now)
            return len(rows)

    def list_audit_logs(self, identity: Identity, *, limit: int = 20) -> list[MemoryAuditLog]:
        customer = _customer(identity)
        with self.engine.connect() as conn:
            rows = ledger.rows(
                conn, """SELECT * FROM cb_memory_audit_logs WHERE customer_id=:c
                ORDER BY created_at DESC, audit_id DESC LIMIT :limit""",
                c=customer, limit=max(0, limit),
            )
        return [MemoryAuditLog.model_validate(row) for row in rows]
