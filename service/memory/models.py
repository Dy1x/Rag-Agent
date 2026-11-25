from typing import Literal

from pydantic import BaseModel, Field

from service.billing.models import UtcTimestamp

MemoryCategory = Literal["profile", "vehicle", "station", "preference", "case_summary"]
MemorySource = Literal["user_explicit", "case_auto"]
EntityKey = Literal["chargers", "stations", "orders", "cases", "vehicles", "time_ranges"]


class CaseRef(BaseModel):
    case_id: str | None = None
    order_id: str | None = None
    status: str | None = None
    summary: str = Field(max_length=120)


class ConversationSummary(BaseModel):
    evicted_rounds: int = Field(default=0, ge=0)
    resolved_cases: list[CaseRef] = Field(default_factory=list, max_length=6)
    active_issues: list[str] = Field(default_factory=list, max_length=5)
    mentioned_entities: dict[EntityKey, list[str]] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list, max_length=4)
    key_facts: list[str] = Field(default_factory=list, max_length=6)

    def render(self, max_chars: int = 800) -> str:
        lines = []
        for case in self.resolved_cases:
            details = " ".join(filter(None, (case.case_id, case.order_id, case.status, case.summary)))
            lines.append(f"- 案件线索 {details}")
        lines.extend(
            f"- {key} {'、'.join(values)}" for key, values in self.mentioned_entities.items()
        )
        for label, items in (
            ("待确认", self.open_questions), ("当前问题", self.active_issues), ("对话记录", self.key_facts),
        ):
            lines.extend(f"- {label} {item}" for item in items)
        result = "\n".join(lines)
        if len(result) <= max_chars:
            return result
        return result[:max(0, max_chars - 1)] + ("…" if max_chars > 0 else "")


class MemoryRecord(BaseModel):
    memory_id: str
    customer_id: str
    category: MemoryCategory
    memory_key: str
    memory_value: str
    source: MemorySource
    source_session_id: str | None
    version: int
    status: Literal["active", "deleted"]
    expires_at: UtcTimestamp | None
    created_at: UtcTimestamp
    updated_at: UtcTimestamp


class MemoryAuditLog(BaseModel):
    audit_id: str
    memory_id: str
    customer_id: str
    memory_key: str
    action: Literal["create", "update", "delete", "restore"]
    old_value: str | None
    new_value: str | None
    reason: str | None
    created_at: UtcTimestamp


class MemoryUpsertResult(BaseModel):
    memory: MemoryRecord
    action: Literal["create", "update", "restore", "unchanged"]
