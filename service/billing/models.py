from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer, PositiveInt
from .tariff import AmountBreakdown, TariffPolicy
from .timeutil import as_utc


# MySQL DATETIME is stored as naive UTC. Preserve the Python/DB representation
# while making the offset unambiguous in tool responses and checkpoints.
UtcTimestamp = Annotated[
    datetime,
    AfterValidator(as_utc),
    PlainSerializer(lambda dt: dt.isoformat() + "Z", return_type=str, when_used="json"),
]


class Identity(BaseModel):
    actor_id: str
    role: Literal["customer", "operator"]


class CustomerView(BaseModel):
    customer_id: str
    nickname: str


class WalletView(BaseModel):
    customer_id: str
    balance_fen: int
    frozen_fen: int
    available_fen: int
    consistent: bool


class ChargerView(BaseModel):
    charger_id: str
    connector_id: str
    station_name: str
    policy_id: str


class AttemptView(BaseModel):
    attempt_id: str
    customer_id: str
    charger_id: str
    connector_id: str
    requested_at: UtcTimestamp
    result: str
    session_id: str | None = None
    freeze_txn_id: str | None = None


class ReadingView(BaseModel):
    reading_id: str
    reading_type: str
    energy_wh: int
    occurred_at: UtcTimestamp
    received_at: UtcTimestamp


class SessionView(BaseModel):
    session_id: str
    attempt_id: str
    started_at: UtcTimestamp
    charge_end_at: UtcTimestamp | None
    unplugged_at: UtcTimestamp | None
    status: str
    readings: list[ReadingView] = []


class OrderView(BaseModel):
    order_id: str
    customer_id: str
    attempt_id: str
    session_id: str
    policy_id: str
    energy_wh: int | None
    electricity_fee_fen: int | None
    service_fee_fen: int | None
    idle_fee_fen: int | None
    total_fen: int | None
    status: str
    created_at: UtcTimestamp
    settled_at: UtcTimestamp | None


class TxnView(BaseModel):
    txn_id: str
    customer_id: str
    txn_type: str
    status: str
    amount_fen: int
    order_id: str | None = None
    attempt_id: str | None = None
    related_txn_id: str | None = None
    source: str
    created_at: UtcTimestamp
    finished_at: UtcTimestamp | None = None


class FreezeView(TxnView):
    released_by_txn_id: str | None = None


class PolicyView(TariffPolicy):
    policy_id: str
    policy_text: str
    effective_from: UtcTimestamp
    effective_to: UtcTimestamp | None
    freeze_amount_fen: int


class OrderCandidate(BaseModel):
    attempt_id: str
    order_id: str | None
    charger_id: str
    connector_id: str
    requested_at: UtcTimestamp
    result: str


class RefundableView(BaseModel):
    txn_id: str
    refundable_fen: int


class OrderFacts(BaseModel):
    order: OrderView
    session: SessionView
    policy_id: str
    recomputed: AmountBreakdown
    charged_net_fen: int
    deducts: list[TxnView]
    refunds: list[TxnView]
    order_refundable_fen: int | None


class Anomaly(BaseModel):
    code: str
    detail: str
    amount_fen: int | None
    evidence_ids: list[str]


class BillingFacts(BaseModel):
    customer_id: str
    attempts: list[AttemptView] = []
    orders: list[OrderFacts] = []
    freezes: list[FreezeView] = []
    pending_recharges: list[TxnView] = []
    anomalies: list[Anomaly] = []
    missing_evidence: list[str] = []
    evidence_index: dict[str, str] = {}


class ApprovedAction(BaseModel):
    proposal_id: str
    action_index: int = Field(ge=0)
    action_type: str
    target_txn_id: str
    amount_fen: PositiveInt
    customer_id: str


class FundActionResult(BaseModel):
    action_id: str
    action_type: str
    target_txn_id: str
    amount_fen: int
    result_txn_id: str | None
    status: Literal["succeeded", "rejected"]
    reject_reason: str | None = None


class ScenarioHandle(BaseModel):
    name: str
    customer_id: str
    refs: dict[str, str]
    suggested_question: str
    suggested_locator: dict


class CaseFile(BaseModel):
    status: Literal["located", "clarify", "forbidden"]
    dispute_type: (
        Literal[
            "double_charge",
            "failed_start_charged",
            "wrong_tariff",
            "idle_fee",
            "refund_status",
            "recharge_missing",
            "frozen_balance",
            "outage_billing",
            "other",
        ]
        | None
    )
    order_ids: list[str] = []
    attempt_ids: list[str] = []
    clarification_question: str | None = None
    candidates: list[
        dict
    ] = []  # clarify 时给车主看的候选（脱敏：订单号尾号、时间、桩号）


class ProposedAction(BaseModel):
    action_type: Literal["refund", "unfreeze"]
    target_txn_id: str
    amount_fen: int = Field(gt=0)
    reason: str


class Proposal(BaseModel):
    proposal_id: str
    version: int
    resolution: Literal["explained", "action"]
    actions: list[ProposedAction] = []
    cited_policy_id: str | None = None
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    customer_message: str


class ReviewIssue(BaseModel):
    code: str
    detail: str
    evidence_ids: list[str] = []


class SoftReviewOpinion(BaseModel):
    approved: bool
    issues: list[ReviewIssue] = []


class ReviewVerdict(BaseModel):
    proposal_id: str
    hard_check_passed: bool
    approved: bool
    issues: list[ReviewIssue]


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject", "approve_modified"]
    approver: str
    modified_actions: list[ProposedAction] | None = None
    comment: str | None = None
