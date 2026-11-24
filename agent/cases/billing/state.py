from typing import TypedDict


class BillingCaseState(TypedDict, total=False):
    failure: str | None
    case_id: str
    run_id: str
    identity: dict  # Identity
    question: str
    locator: dict  # order_id / charger_id / connector_id / start_utc / end_utc
    planner: str
    case_file: dict  # CaseFile
    facts: dict  # BillingFacts
    proposals: list[dict]  # 历次 Proposal
    reviews: list[dict]  # 历次 ReviewVerdict
    revision: int
    approval: dict | None  # ApprovalDecision
    action_results: list[dict]
    status: str
    report: dict
