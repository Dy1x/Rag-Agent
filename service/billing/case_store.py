import json
from contextlib import contextmanager

from sqlalchemy import Engine, text

from . import ledger
from .errors import BillingError
from .ids import new_id
from .timeutil import utc_now


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _decode(row, *fields):
    for field in fields:
        if isinstance(row.get(field), str):
            row[field] = json.loads(row[field])
    return row


class CaseStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def _insert_case(self, c, *, handler, customer_id, question, chat_session_id):
        key = new_id("CASE")
        ledger.insert(
            c,
            "dispute_cases",
            case_id=key,
            handler=handler,
            customer_id=customer_id,
            chat_session_id=chat_session_id,
            question=question,
            status="running",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        return key

    def _insert_run(self, c, case_id, planner):
        key = new_id("RUN")
        ledger.insert(
            c,
            "case_runs",
            run_id=key,
            case_id=case_id,
            planner=planner,
            status="running",
            started_at=utc_now(),
        )
        c.execute(
            text(
                "UPDATE cb_dispute_cases SET latest_run_id=:r,status='running',updated_at=UTC_TIMESTAMP(3) WHERE case_id=:k"
            ),
            dict(k=case_id, r=key),
        )
        return key

    def create_case(self, *, handler, customer_id, question, chat_session_id):
        with self.engine.begin() as c:
            return self._insert_case(
                c,
                handler=handler,
                customer_id=customer_id,
                question=question,
                chat_session_id=chat_session_id,
            )

    def create_run(self, case_id, planner):
        with self.engine.begin() as c:
            return self._insert_run(c, case_id, planner)

    def start_case_run(
        self, *, handler, customer_id, question, chat_session_id, planner
    ):
        with self.engine.begin() as c:
            case_id = self._insert_case(
                c,
                handler=handler,
                customer_id=customer_id,
                question=question,
                chat_session_id=chat_session_id,
            )
            return case_id, self._insert_run(c, case_id, planner)

    def link_outage_session(self, parent_run_id, interruption, planner):
        session_id = interruption["session_id"]
        with self.engine.begin() as c:
            owner = ledger.one(
                c,
                """SELECT a.customer_id FROM cb_charging_sessions s JOIN cb_charging_attempts a
                USING(attempt_id) WHERE s.session_id=:s FOR UPDATE""",
                s=session_id,
            )["customer_id"]
            rows = ledger.rows(
                c,
                """SELECT d.case_id,d.latest_run_id,d.question FROM ci_billing_children b
                JOIN cb_dispute_cases d ON d.case_id=b.child_case_id WHERE b.session_id=:s""",
                s=session_id,
            )
            if rows:
                case_id, run_id, question = (
                    rows[0][f] for f in ("case_id", "latest_run_id", "question")
                )
            else:
                question = "该次充电已确认因网关重启中断，请核查订单计费、扣款与冻结。"
                case_id = self._insert_case(
                    c,
                    handler="billing_dispute",
                    customer_id=owner,
                    question=question,
                    chat_session_id=None,
                )
                run_id = self._insert_run(c, case_id, planner)
                c.execute(
                    text(
                        "INSERT INTO ci_billing_children(session_id,child_case_id) VALUES (:s,:k)"
                    ),
                    dict(s=session_id, k=case_id),
                )
            c.execute(
                text("""INSERT INTO ci_case_links(parent_run_id,child_case_id,interruption_id)
                VALUES (:r,:k,:i) ON DUPLICATE KEY UPDATE child_case_id=child_case_id"""),
                dict(r=parent_run_id, k=case_id, i=interruption["interruption_id"]),
            )
        return dict(
            case_id=case_id, run_id=run_id, customer_id=owner, question=question
        )

    def update_case(self, case_id, *, status, dispute_type=None, latest_run_id=None):
        with self.engine.begin() as c:
            c.execute(
                text("""UPDATE cb_dispute_cases SET status=:s,dispute_type=COALESCE(:d,dispute_type),
                updated_at=UTC_TIMESTAMP(3) WHERE case_id=:k AND (:r IS NULL OR latest_run_id=:r)"""),
                dict(k=case_id, s=status, d=dispute_type, r=latest_run_id),
            )
            if latest_run_id:
                c.execute(
                    text(
                        "UPDATE cb_case_runs SET status=:s WHERE run_id=:r AND case_id=:k"
                    ),
                    dict(s=status, r=latest_run_id, k=case_id),
                )

    def finish_run(self, run_id, status, report):
        with self.engine.begin() as c:
            c.execute(
                text("""UPDATE cb_case_runs SET status=:s,report=:p,finished_at=UTC_TIMESTAMP(3)
                WHERE run_id=:r"""),
                dict(s=status, p=_json(report), r=run_id),
            )
            c.execute(
                text("""UPDATE cb_dispute_cases SET status=:s,dispute_type=:d,updated_at=UTC_TIMESTAMP(3)
                WHERE latest_run_id=:r"""),
                dict(s=status, d=report.get("dispute_type"), r=run_id),
            )

    def save_proposal(self, run_id, proposal):
        with self.engine.begin() as c:
            ledger.insert(
                c,
                "proposals",
                proposal_id=proposal.proposal_id,
                run_id=run_id,
                version=proposal.version,
                payload=proposal.model_dump_json(),
                created_at=utc_now(),
            )

    def save_review(self, proposal_id, verdict):
        key = new_id("REV")
        with self.engine.begin() as c:
            ledger.insert(
                c,
                "reviews",
                review_id=key,
                proposal_id=proposal_id,
                hard_check_passed=verdict.hard_check_passed,
                approved=verdict.approved,
                issues=_json([i.model_dump() for i in verdict.issues]),
                created_at=utc_now(),
            )
        return key

    def save_approval(self, proposal_id, decision):
        key = new_id("APV")
        with self.engine.begin() as c:
            prior = ledger.rows(
                c, "SELECT * FROM cb_approvals WHERE proposal_id=:p", p=proposal_id
            )
            if prior:
                row = _decode(prior[0], "modified_actions")
                if any(
                    row[k] != v for k, v in decision.model_dump(mode="json").items()
                ):
                    raise BillingError("INVALID_STATE", "该方案已经审批")
                return row["approval_id"]
            ledger.insert(
                c,
                "approvals",
                approval_id=key,
                proposal_id=proposal_id,
                approver=decision.approver,
                decision=decision.decision,
                modified_actions=_json(decision.model_dump()["modified_actions"]),
                comment=decision.comment,
                decided_at=utc_now(),
            )
        return key

    def append_trace(self, run_id, node, event_type, payload):
        with self.engine.begin() as c:
            ledger.one(
                c,
                "SELECT run_id FROM cb_case_runs WHERE run_id=:r FOR UPDATE",
                r=run_id,
            )
            seq = ledger.one(
                c,
                "SELECT COALESCE(MAX(seq),0)+1 n FROM cb_trace_events WHERE run_id=:r",
                r=run_id,
            )["n"]
            ledger.insert(
                c,
                "trace_events",
                run_id=run_id,
                seq=seq,
                node=node,
                event_type=event_type,
                payload=_json(payload),
                created_at=utc_now(),
            )

    def list_pending_approvals(self):
        with self.engine.begin() as c:
            return ledger.rows(
                c,
                "SELECT * FROM cb_dispute_cases WHERE status='action_pending_approval' ORDER BY created_at",
            )

    def get_run(self, run_id):
        with self.engine.begin() as c:
            return self._read_run(c, run_id)

    def _read_run(self, c, run_id):
        return _decode(
            ledger.one(c, "SELECT * FROM cb_case_runs WHERE run_id=:r", r=run_id),
            "report",
        )

    def get_case_detail(self, identity, case_id):
        with self.engine.begin() as c:
            result = ledger.rows(
                c,
                """SELECT * FROM cb_dispute_cases WHERE case_id=:k
                AND (:operator OR customer_id=:owner)""",
                k=case_id,
                operator=identity.role == "operator",
                owner=identity.actor_id,
            )
            if not result:
                raise BillingError("FORBIDDEN", "不可访问")
            case = result[0]
            run = case["latest_run_id"]
            case["runs"] = [
                _decode(r, "report")
                for r in ledger.rows(
                    c,
                    "SELECT * FROM cb_case_runs WHERE case_id=:k ORDER BY started_at",
                    k=case_id,
                )
            ]
            case["proposals"] = [
                _decode(r, "payload")["payload"]
                for r in ledger.rows(
                    c,
                    "SELECT payload FROM cb_proposals WHERE run_id=:r ORDER BY version",
                    r=run,
                )
            ]
            case["reviews"] = [
                _decode(r, "issues")
                for r in ledger.rows(
                    c,
                    """SELECT v.* FROM cb_reviews v JOIN cb_proposals p
                USING(proposal_id) WHERE p.run_id=:r ORDER BY p.version""",
                    r=run,
                )
            ]
            return case

    def get_trace(self, run_id):
        with self.engine.begin() as c:
            return self._read_events(c, run_id)

    def _read_events(self, c, run_id):
        return [
            _decode(r, "payload")
            for r in ledger.rows(
                c, "SELECT * FROM cb_trace_events WHERE run_id=:r ORDER BY seq", r=run_id
            )
        ]

    def get_trace_snapshot(self, run_id):
        with self.engine.connect() as c:
            c.execution_options(isolation_level="REPEATABLE READ")
            return self._read_run(c, run_id), self._read_events(c, run_id)

    @contextmanager
    def run_lock(self, run_id):
        key = "billing:" + run_id
        with self.engine.connect() as c:
            if c.scalar(text("SELECT GET_LOCK(:k,10)"), dict(k=key)) != 1:
                raise BillingError("INVALID_STATE", "案件正在处理中")
            try:
                yield
            finally:
                c.execute(text("SELECT RELEASE_LOCK(:k)"), dict(k=key))
