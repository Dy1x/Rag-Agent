import json
import streamlit as st

from service.billing.case_store import CaseStore
from service.billing.db import get_billing_engine
from service.billing.errors import BillingError
from service.billing.models import Identity


def run():
    st.title("案件运行轨迹")
    key = st.text_input("案件号或运行号").strip()
    if not key:
        return
    store = CaseStore(get_billing_engine())
    if key.startswith("CASE-"):
        case = store.get_case_detail(Identity(role="operator", actor_id="trace"), key)
        run_id = st.selectbox("选择运行", [r["run_id"] for r in reversed(case["runs"])])
    else:
        run_id = key
    run, events = store.get_trace_snapshot(run_id)
    st.write(run["status"])
    for event in events:
        with st.expander(f"{event['seq']} · {event['node']} · {event['event_type']}"):
            st.caption(str(event["created_at"]) + " UTC")
            st.json(event["payload"])
    st.download_button(
        "下载 JSON",
        json.dumps(
            {"run": run, "events": events}, ensure_ascii=False, indent=2, default=str
        ),
        f"{run_id}.json",
        "application/json",
    )


try:
    run()
except BillingError as error:
    st.error(error.message)
