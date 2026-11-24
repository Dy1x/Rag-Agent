from service.billing.models import Identity
from agent.cases.billing.handler import BillingDisputeHandler


def investigate_interrupted_sessions(
    source, scope, parent, store, *, checkpointer=None
):
    if parent["shared_queries"]["alarm_timeline"]["status"] != "available":
        return []
    interruptions = source.confirmed_interruptions(scope)
    if not interruptions:
        return []
    handler = BillingDisputeHandler(source.engine, checkpointer)
    seen = set()
    children = []
    for event in interruptions:
        session = event["session_id"]
        if session in seen:
            continue
        seen.add(session)
        linked = store.link_outage_session(parent["run_id"], event, parent["planner"])
        status = handler.investigate_linked_session(
            linked["case_id"],
            linked["run_id"],
            Identity(role="customer", actor_id=linked["customer_id"]),
            linked["question"],
            session,
        )
        item = dict(
            case_id=linked["case_id"],
            run_id=linked["run_id"],
            status=status,
            interruption_id=event["interruption_id"],
            site_event_id=event["site_event_id"],
            alarm_event_id=event["alarm_event_id"],
        )
        store.append_trace(parent["run_id"], "billing_followup", "child_case", item)
        children.append(item)
    return children
