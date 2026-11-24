from .checkpoints import get_case_checkpointer
from langgraph.types import Command
from agent.cases.base import CaseResult

from service.billing.case_store import CaseStore
from service.billing.errors import BillingError
from service.billing.executor import FundActionExecutor
from service.billing.facts_service import BillingFactsService
from service.billing.models import ApprovalDecision, Identity
from .graph import build_graph


class BillingDisputeHandler:
    name = "billing_dispute"
    description = "核查当前车主的充电费用、冻结、退款与充值到账争议"

    def __init__(self, engine, checkpointer=None, *, transform=None, design="C"):
        self.store = CaseStore(engine)
        self.graph = build_graph(
            BillingFactsService(engine),
            self.store,
            FundActionExecutor(engine),
            checkpointer or get_case_checkpointer(),
            transform=transform,
            design=design,
        )

    def start(
        self,
        identity: Identity,
        question: str,
        locator: dict,
        *,
        chat_session_id=None,
        planner="offline",
    ) -> CaseResult:
        if identity.role != "customer":
            raise BillingError("VALIDATION", "发起核查需要车主身份")
        case_id = self.store.create_case(
            handler=self.name,
            customer_id=identity.actor_id,
            question=question,
            chat_session_id=chat_session_id,
        )
        return self._run(case_id, identity, question, locator, planner)

    def _run(self, case_id, identity, question, locator, planner, *, run=None):
        if run is None:
            run = self.store.create_run(case_id, planner)
        initial = dict(
            case_id=case_id,
            run_id=run,
            identity=identity.model_dump(),
            question=question,
            locator=locator,
            planner=planner,
            proposals=[],
            reviews=[],
            revision=0,
            action_results=[],
        )
        state = self.graph.invoke(initial, self._config(run))
        return self._result(state)

    def investigate_linked_session(
        self, case_id, run_id, identity, question, session_id
    ):
        with self.store.run_lock(run_id):
            run = self.store.get_run(run_id)
            if run["status"] != "running":
                return run["status"]
            config = self._config(run_id)
            saved = self.graph.get_state(config)
            if saved.values:
                if saved.interrupts:
                    return self._result(saved.values).status
                return self._result(self.graph.invoke(None, config)).status
            return self._run(
                case_id,
                identity,
                question,
                {"confirmed_outage_session": session_id},
                run["planner"],
                run=run_id,
            ).status

    @staticmethod
    def _config(run):
        return {"configurable": {"thread_id": run}}

    @staticmethod
    def _result(state):
        report = state.get("report") or {}
        message = report.get("customer_message")
        if state["status"] == "action_pending_approval":
            message = "已提交人工审批。"
        return CaseResult(
            case_id=state["case_id"],
            run_id=state["run_id"],
            status=state["status"],
            customer_message=message,
            clarification_question=message if state["status"] == "clarify" else None,
            report=report or None,
        )

    def resume(self, run_id: str, decision: ApprovalDecision) -> CaseResult:
        with self.store.run_lock(run_id):
            config = self._config(run_id)
            saved = self.graph.get_state(config)
            if not saved.values:
                raise BillingError("NOT_FOUND", "未找到运行检查点")
            if saved.values.get("report"):
                return self._result(saved.values)
            if saved.interrupts:
                update = Command(resume=decision.model_dump())
            elif saved.values.get("approval") and saved.next:
                update = None
            else:
                raise BillingError("INVALID_STATE", "当前没有待审批方案")
            return self._result(self.graph.invoke(update, config))

    def rerun(self, case_id: str, identity: Identity) -> CaseResult:
        case = self.store.get_case_detail(identity, case_id)
        if identity.role != "customer" or case["status"] not in (
            "pending_evidence",
            "escalated",
            "clarify",
        ):
            raise BillingError("INVALID_STATE", "案件不支持重新核查")
        state = self.graph.get_state(self._config(case["latest_run_id"])).values
        return self._run(
            case_id, identity, case["question"], state["locator"], state["planner"]
        )
