from typing import Protocol
from pydantic import BaseModel

from service.billing.models import ApprovalDecision, Identity


class CaseResult(BaseModel):
    case_id: str
    run_id: str
    status: str
    customer_message: str | None
    clarification_question: str | None = None
    report: dict | None = None


class CaseHandler(Protocol):
    name: str
    description: str

    def start(
        self,
        identity: Identity,
        question: str,
        locator: dict,
        *,
        chat_session_id: str | None,
        planner: str,
    ) -> CaseResult: ...
    def resume(self, run_id: str, decision: ApprovalDecision) -> CaseResult: ...
    def rerun(self, case_id: str, identity: Identity) -> CaseResult: ...
