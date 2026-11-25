"""Compact old tool outputs without changing message identity or pairing."""

import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

MARKER = "[COMPACTED]"
MAX_COMPACTED = 220


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tool_payload(content) -> dict:
    if isinstance(content, str):
        try:
            value = json.loads(content.removeprefix(MARKER))
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    return {}


def message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else _json(message.content)


def context_chars(messages) -> int:
    return sum(
        len(message_text(message)) + (len(_json(calls)) if (calls := getattr(message, "tool_calls", [])) else 0)
        for message in messages
    )


def _rag_excerpt(text: str, limit: int) -> str:
    answer, _, references = text.partition("**知识库出处**")
    first = re.split(r"(?<=[。！？])|\n", answer.strip(), maxsplit=1)[0]
    sources = re.findall(r"^\s*-\s*(【E\d+】)([^\n]*)", references, re.MULTILINE)
    pointers = "\n".join(id_ for id_, _ in sources)
    if not pointers:
        return first[:limit]
    first = first[:min(120, max(0, limit - len(pointers) - 1))]
    per_source = max(0, (limit - len(first) - len(pointers) - 1) // len(sources))
    return "\n".join([first, *(id_ + source[:per_source] for id_, source in sources)])[:limit]


def _candidates(payload):
    return [
        item["订单尾号"]
        for item in payload.get("candidates") or []
        if isinstance(item, dict) and re.fullmatch(r"[A-Za-z0-9]{4}", str(item.get("订单尾号")))
    ][:8]


def _investigation(payload):
    report = payload.get("report") or {}
    priority = report.get("priority") or {}
    impact = report.get("impact") or {}
    evidence = report.get("source_ids") or {}
    refs = [
        ref for group in ("site_events", "alarm_events", "firmware_rollouts")
        for ref in (evidence.get(group) or [])[:3]
    ][:3]
    return {
        "run_id": payload.get("run_id"), "status": payload.get("status"),
        "case_id": payload.get("case_id"), "direction": priority.get("direction"),
        "site_id": priority.get("site_id"),
        "offline": impact.get("offline_device_count"), "total": impact.get("device_count"),
        "evidence_ids": refs or None,
    }


def _compact(message: ToolMessage) -> str:
    raw = message_text(message)
    payload = tool_payload(raw)
    limit = MAX_COMPACTED - len(MARKER)
    if message.name == "rag_summerize":
        return MARKER + _rag_excerpt(raw, limit)
    if message.name in ("handle_billing_dispute", "get_dispute_case_status") and payload:
        fields = {
            "case_id": payload.get("case_id"), "status": payload.get("status"),
            "candidates": _candidates(payload) or None,
            "customer_message": str(payload.get("customer_message") or "")[:80] or None,
        }
    elif message.name == "investigate_outage" and payload:
        fields = _investigation(payload)
    else:
        fields = {key: value for key, value in payload.items() if isinstance(value, (str, int, float, bool))}
    if fields:
        result = {}
        for key, value in fields.items():
            if value is not None and len(_json({**result, key: value})) <= limit:
                result[key] = value
        if not result and len(fields) == 1:
            key, value = next(iter(fields.items()))
            if isinstance(value, str) and len(_json({key: ""})) < limit:
                while len(_json({key: value})) > limit:
                    value = value[:len(value) // 2]
                result[key] = value
        if result:
            return MARKER + _json(result)
    return MARKER + raw[:94] + "…" + raw[-94:]


def compact_old_tools(
    messages: list[BaseMessage], *, keep_rounds: int, threshold: int,
) -> list[BaseMessage]:
    starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(starts) <= keep_rounds:
        return messages
    boundary = starts[-keep_rounds] if keep_rounds else len(messages)
    return [
        message.model_copy(update={"content": _compact(message)})
        if index < boundary and isinstance(message, ToolMessage)
        and len(message_text(message)) > threshold and len(message_text(message)) > MAX_COMPACTED
        and not message_text(message).startswith(MARKER)
        else message
        for index, message in enumerate(messages)
    ]
