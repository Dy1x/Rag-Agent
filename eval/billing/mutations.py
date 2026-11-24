from service.billing.models import ProposedAction

# Each pair has an applicable first proposal. All designs run the same matrix.
MATRIX = {
    "M_AMOUNT_INFLATE": ["S2", "S4", "S10"],
    "M_WRONG_TARGET": ["S2"],
    "M_FREEZE_AS_DEDUCT": ["S1"],
    "M_FAKE_EVIDENCE": ["S1", "S2", "S5", "S8"],
    "M_WRONG_POLICY": ["S2", "S4", "S9"],
    "M_MISSED_REMEDY": ["S1", "S2", "S4", "S10"],
    "M_OVERPROMISE": ["S1", "S2", "S5", "S8", "S9"],
    "M_SPURIOUS_REFUND": ["S8", "S9"],
}
HARD = set(MATRIX) - {"M_MISSED_REMEDY", "M_OVERPROMISE"}
ACTION_REQUIRED = {"M_AMOUNT_INFLATE", "M_WRONG_TARGET", "M_FREEZE_AS_DEDUCT"}
DEDUCT_REQUIRED = {"M_WRONG_TARGET", "M_SPURIOUS_REFUND"}


def injection_blocker(name, proposal, facts):
    if name in ACTION_REQUIRED and not proposal.actions:
        return "首版方案没有资金动作"
    if name in DEDUCT_REQUIRED and (not facts.orders or not facts.orders[0].deducts):
        return "首版事实中缺少原扣款"
    return None


def mutate(name, proposal, facts):
    if injection_blocker(name, proposal, facts):
        return proposal
    p = proposal.model_copy(deep=True)
    if name == "M_AMOUNT_INFLATE":
        p.actions[0].amount_fen += 100
    elif name == "M_WRONG_TARGET":
        p.actions[0].target_txn_id = facts.orders[0].deducts[0].txn_id
    elif name == "M_FREEZE_AS_DEDUCT":
        p.actions[0].action_type = "refund"
    elif name == "M_FAKE_EVIDENCE":
        p.evidence_ids.append("MTR-DOESNOTEXIST")
    elif name == "M_WRONG_POLICY":
        p.cited_policy_id = "POLICY-UNRELATED"
    elif name == "M_MISSED_REMEDY":
        p.actions = []
        p.resolution = "explained"
    elif name == "M_OVERPROMISE":
        p.customer_message += "保证立即到账。"
    elif name == "M_SPURIOUS_REFUND":
        target = facts.orders[0].deducts[0]
        p.resolution = "action"
        p.actions = [
            ProposedAction(
                action_type="refund",
                target_txn_id=target.txn_id,
                amount_fen=target.amount_fen,
                reason="要求全退",
            )
        ]
    else:
        raise ValueError(name)
    return p
