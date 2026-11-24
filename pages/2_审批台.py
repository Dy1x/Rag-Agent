import streamlit as st

from agent.cases.billing.handler import BillingDisputeHandler
from service.billing.db import get_billing_engine
from service.billing.errors import BillingError
from service.billing.facts_service import BillingFactsService
from service.billing.models import ApprovalDecision, Identity, Proposal
from utils.config_handler import billing_conf


def run():
    st.title("资金审批台")
    engine = get_billing_engine()
    handler = BillingDisputeHandler(engine)
    if previous := st.session_state.get("last_approval"):
        st.write("最近审批结果", previous["status"])
        st.json(previous["report"]["action_results"])
        if previous["report"].get("failure"):
            st.error(previous["report"]["failure"])
    pending = handler.store.list_pending_approvals()
    if not pending:
        st.info("当前没有待审批案件。")
        return
    cases = {r["case_id"]: r for r in pending}
    case_id = st.selectbox("待审批案件", list(cases))
    case = cases[case_id]
    operator = st.text_input("审批人", billing_conf["default_operator"])
    identity = Identity(role="operator", actor_id=operator)
    detail = handler.store.get_case_detail(identity, case_id)
    st.write("车主问题", detail["question"])
    for p in detail["proposals"]:
        with st.expander(
            f"方案 v{p['version']}", expanded=p == detail["proposals"][-1]
        ):
            st.write(p["customer_message"])
            st.write(p["rationale"])
            st.dataframe(p["actions"])
            reviews = [
                r for r in detail["reviews"] if r["proposal_id"] == p["proposal_id"]
            ]
            st.json(reviews)
    proposal = Proposal(**detail["proposals"][-1])
    service = BillingFactsService(engine)
    owner = Identity(role="customer", actor_id=case["customer_id"])
    with st.expander("查看证据"):
        for evidence in proposal.evidence_ids:
            st.code(evidence)
            if evidence.startswith("TXN-"):
                records = service.get_transactions(owner, txn_ids=[evidence])
            elif evidence.startswith("ATT-"):
                records = service.get_attempts(owner, [evidence])
            elif evidence.startswith("SES-"):
                records = [service.get_session(owner, evidence)]
            elif evidence.startswith("ORD-"):
                records = [service.get_order(owner, evidence)]
            elif evidence.startswith("POLICY-"):
                records = [service.get_policy(owner, evidence)]
            else:
                records = service.get_readings(owner, evidence)
            st.json([r.model_dump(mode="json") for r in records])
    decision = st.radio(
        "审批决定",
        ["approve", "reject", "approve_modified"],
        format_func={
            "approve": "通过",
            "reject": "拒绝",
            "approve_modified": "修改后通过",
        }.get,
    )
    modified = None
    if decision == "approve_modified":
        modified = []
        for index, a in enumerate(proposal.actions):
            amount = st.number_input(
                f"动作 {index + 1} 金额（分，0 表示移除）",
                min_value=0,
                max_value=a.amount_fen,
                value=a.amount_fen,
                key=f"{proposal.proposal_id}_{index}",
            )
            if amount:
                modified.append(a.model_copy(update={"amount_fen": amount}))
    comment = st.text_area("审批说明", max_chars=255)
    if st.button(
        "提交审批",
        disabled=not operator.strip()
        or (decision == "approve_modified" and not comment.strip()),
    ):
        value = ApprovalDecision(
            decision=decision,
            approver=operator,
            comment=comment or None,
            modified_actions=modified,
        )
        result = handler.resume(case["latest_run_id"], value)
        st.session_state.last_approval = result.model_dump()
        st.rerun()


try:
    run()
except BillingError as error:
    st.error(error.message)
