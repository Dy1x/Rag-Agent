import hashlib
import json
from datetime import UTC, datetime

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status


def _trace_id(case_id):
    return hashlib.sha256(case_id.encode()).digest()[:16]


def _span_id(run_id):
    return hashlib.sha256(run_id.encode()).digest()[:8]


def _unix_nanos(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    elapsed = value - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        (elapsed.days * 86400 + elapsed.seconds) * 1000000 + elapsed.microseconds
    ) * 1000


def encode_otlp(run, events):
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    resource.resource.attributes.add(
        key="service.name"
    ).value.string_value = "charging-support"
    scope = resource.scope_spans.add()
    scope.scope.name = "charging.cases"
    span = scope.spans.add(
        trace_id=_trace_id(run["case_id"]),
        span_id=_span_id(run["run_id"]),
        name="case.run",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=_unix_nanos(run["started_at"]),
    )
    for key in ("case_id", "run_id", "planner", "status"):
        span.attributes.add(key=f"case.{key}").value.string_value = run[key]
    finished = run["finished_at"]
    span.attributes.add(key="case.finished").value.bool_value = finished is not None
    span.end_time_unix_nano = max(
        span.start_time_unix_nano, _unix_nanos(finished) if finished else 0
    )
    if run["status"] == "failed":
        span.status.code = Status.STATUS_CODE_ERROR
    elif finished:
        span.status.code = Status.STATUS_CODE_OK

    for row in events:
        node, event_type, payload = row["node"], row["event_type"], row["payload"]
        event = span.events.add(
            name=f"{node}.{event_type}", time_unix_nano=_unix_nanos(row["created_at"])
        )
        event.attributes.add(key="case.event.seq").value.int_value = row["seq"]
        event.attributes.add(key="case.node").value.string_value = node
        event.attributes.add(key="case.event.type").value.string_value = event_type
        event.attributes.add(key="case.event.payload").value.string_value = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        span.end_time_unix_nano = max(span.end_time_unix_nano, event.time_unix_nano)
        if (
            event_type == "child_case"
            and isinstance(payload, dict)
            and payload.get("case_id")
            and payload.get("run_id")
        ):
            span.links.add(
                trace_id=_trace_id(payload["case_id"]),
                span_id=_span_id(payload["run_id"]),
            )
    return request.SerializeToString()
