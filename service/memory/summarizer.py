"""Incremental summary with offline extraction and optional structured model output."""

import json
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from service.memory.compactor import message_text, tool_payload
from service.memory.customer_memory_service import FUND_CLAIM
from service.memory.models import CaseRef, ConversationSummary
from utils.logger_handler import logger

_ENTITIES = {
    "chargers": r"CP-[A-Z0-9]+(?:-[A-Z0-9]+)*",
    "stations": r"科技园\s*[A-Z]\s*站",
    "orders": r"ORD-[0-9A-F]{12}",
    "cases": r"CASE-[0-9A-F]{12}",
    "vehicles": r"(?:比亚迪汉\s*EV|特斯拉\s*Model\s*[3YSX]|小鹏\s*G6)",
    "time_ranges": r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?",
}
_RUN = re.compile(r"RUN-[0-9A-F]{12}")
_BILLING_TOOLS = {"handle_billing_dispute", "get_dispute_case_status"}
_CASE_STATUS = {
    "action_pending_approval", "action_executed", "action_rejected", "explained",
    "escalated", "pending_evidence", "clarify",
}
_summary_outcomes: ContextVar[Counter[str] | None] = ContextVar("summary_outcomes", default=None)


class _SummaryDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_issues: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)

    @field_validator("active_issues", "open_questions", "key_facts", mode="before")
    @classmethod
    def normalize_notes(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [note.strip() for note in value if isinstance(note, str) and note.strip()]
        return value


@contextmanager
def count_summary_outcomes():
    """Collect model outcomes for a single evaluation group."""
    counts = Counter(success=0, fallback=0)
    token = _summary_outcomes.set(counts)
    try:
        yield counts
    finally:
        _summary_outcomes.reset(token)


def _newest(old, new, limit):
    return list(dict.fromkeys([*new, *old]))[:limit]


def _safe_note(text) -> bool:
    return bool(text) and not FUND_CLAIM.search(re.sub(r"\s+", "", text))


def _offline(previous, messages):
    entities = {key: list(values) for key, values in previous.mentioned_entities.items()}
    facts = []
    questions = []
    issues = []
    cases = {record.case_id: record for record in reversed(previous.resolved_cases)}
    runs = []
    for message in messages:
        content = message_text(message)
        for key, pattern in _ENTITIES.items():
            matches = re.findall(pattern, content)
            if matches:
                entities[key] = _newest(entities.get(key, []), reversed(matches), 6)
        for run in _RUN.findall(content):
            runs.insert(0, f"调查编号 {run}")
        if isinstance(message, HumanMessage) and _safe_note(content):
            facts.insert(0, f"用户提到：{content.strip()[:95]}")
        if not isinstance(message, ToolMessage):
            continue
        payload = tool_payload(content)
        if message.name == "investigate_outage":
            report = payload.get("report") or {}
            priority = report.get("priority") or {}
            direction = payload.get("direction") or priority.get("direction")
            site = payload.get("site_id") or priority.get("site_id")
            if direction == "site" and isinstance(site, str) and re.fullmatch(r"[A-Z0-9-]{1,24}", site):
                issues.insert(0, f"掉线调查优先核验 {site} 站网关，故障归因待验证")
        if message.name not in _BILLING_TOOLS:
            continue
        case_id, status = payload.get("case_id"), payload.get("status")
        if status == "clarify" and (question := payload.get("clarification_question")):
            if isinstance(question, str) and _safe_note(question):
                questions.insert(0, question[:100])
        if isinstance(case_id, str) and re.fullmatch(_ENTITIES["cases"], case_id) and status in _CASE_STATUS:
            orders = [
                ref for ref in re.findall(_ENTITIES["orders"], str(payload.get("order_id") or ""))
            ]
            old = cases.pop(case_id, None)
            explanation = payload.get("customer_message") or ""
            cases[case_id] = CaseRef(
                case_id=case_id, order_id=orders[0] if orders else old.order_id if old else None,
                status=status, summary=explanation[:120] if isinstance(explanation, str) and _safe_note(explanation) else "",
            )
    return ConversationSummary(
        evicted_rounds=previous.evicted_rounds + sum(isinstance(m, HumanMessage) for m in messages),
        resolved_cases=list(reversed(cases.values()))[:6],
        active_issues=_newest(previous.active_issues, issues, 5),
        mentioned_entities=entities,
        open_questions=_newest(previous.open_questions, questions, 4),
        key_facts=_newest(previous.key_facts, [*runs, *facts], 6),
    )


def _model_summary(previous, messages) -> _SummaryDraft:
    from model.factory import get_chat_model

    prompt = (Path(__file__).resolve().parents[2] / "prompts/memory_summarize.txt").read_text(encoding="utf-8")
    data = {
        "previous": previous.model_dump(include=set(_SummaryDraft.model_fields)),
        "messages": [
            {"role": msg.type, "tool": getattr(msg, "name", None), "text": message_text(msg)[:1000]}
            for msg in messages
        ],
    }
    response = get_chat_model().bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(data, ensure_ascii=False)),
    ])
    return _SummaryDraft.model_validate_json(response.content)


def update_summary(
    previous: ConversationSummary | None, evicted_messages: list[BaseMessage], *,
    mode: Literal["offline", "deepseek"] = "offline",
) -> ConversationSummary:
    previous = previous or ConversationSummary()
    summary = _offline(previous, evicted_messages)
    if mode != "deepseek":
        return summary
    outcome = "success"
    try:
        model = _model_summary(previous, evicted_messages)
        # Keep authoritative pointers and statuses from deterministic extraction.
        summary = summary.model_copy(update={
            "active_issues": _newest(summary.active_issues, (x for x in model.active_issues if _safe_note(x)), 5),
            "open_questions": _newest(summary.open_questions, (x for x in model.open_questions if _safe_note(x)), 4),
            "key_facts": _newest(
                [f"对话提及：{x[:95]}" for x in model.key_facts if _safe_note(x)],
                summary.key_facts, 6,
            ),
        })
    except Exception as error:
        outcome = "fallback"
        logger.warning(f"[memory] 摘要模型异常，已使用离线提取：{type(error).__name__}")
    if (counts := _summary_outcomes.get()) is not None:
        counts[outcome] += 1
    return summary
