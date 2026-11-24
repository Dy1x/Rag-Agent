import json
import unittest
from datetime import UTC, datetime, timedelta, timezone

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Status

from service.billing.trace_export import encode_otlp


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 1, 0, 0, 0, 321987)
        self.run = dict(
            case_id="CASE-1",
            run_id="RUN-1",
            planner="offline",
            started_at=self.start,
            finished_at=None,
            status="running",
        )

    def decode(self, run, events=()):
        request = ExportTraceServiceRequest.FromString(encode_otlp(run, events))
        resource = request.resource_spans[0]
        self.assertEqual(
            resource.resource.attributes[0].value.string_value, "charging-support"
        )
        return resource.scope_spans[0].spans[0]

    def test_event_payload_times_and_child_link_round_trip(self):
        child = dict(self.run, case_id="CASE-2", run_id="RUN-2")
        payload = {
            "case_id": child["case_id"],
            "run_id": child["run_id"],
            "证据": [1, None],
        }
        after = self.start.replace(tzinfo=UTC).astimezone(
            timezone(timedelta(hours=8))
        ) + timedelta(microseconds=5)
        events = [
            dict(
                seq=1,
                node="graph",
                event_type="node_end",
                payload={},
                created_at=self.start,
            ),
            dict(
                seq=2,
                node="billing_followup",
                event_type="child_case",
                payload=payload,
                created_at=after,
            ),
        ]
        span = self.decode(
            dict(self.run, status="completed", finished_at=self.start), events
        )
        self.assertEqual(span.start_time_unix_nano, 1767225600321987000)
        self.assertEqual(span.end_time_unix_nano, span.start_time_unix_nano + 5000)
        self.assertEqual(span.status.code, Status.STATUS_CODE_OK)
        self.assertEqual(
            [e.name for e in span.events],
            ["graph.node_end", "billing_followup.child_case"],
        )
        attrs = {a.key: a.value for a in span.events[1].attributes}
        self.assertEqual(attrs["case.event.seq"].int_value, 2)
        self.assertEqual(json.loads(attrs["case.event.payload"].string_value), payload)
        child_span = self.decode(child)
        self.assertEqual(
            (span.links[0].trace_id, span.links[0].span_id),
            (child_span.trace_id, child_span.span_id),
        )

    def test_incomplete_failed_and_retried_runs(self):
        pending = self.decode(self.run)
        self.assertEqual(pending.status.code, Status.STATUS_CODE_UNSET)
        self.assertEqual(pending.end_time_unix_nano, pending.start_time_unix_nano)
        self.assertEqual(len(pending.trace_id), 16)
        self.assertEqual(len(pending.span_id), 8)
        failed = self.decode(
            dict(self.run, run_id="RUN-2", status="failed", finished_at=self.start)
        )
        self.assertEqual(failed.status.code, Status.STATUS_CODE_ERROR)
        self.assertEqual(failed.trace_id, pending.trace_id)
        self.assertNotEqual(failed.span_id, pending.span_id)
        self.assertEqual(self.decode(self.run).span_id, pending.span_id)
