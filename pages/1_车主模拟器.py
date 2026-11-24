import os
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import streamlit as st

from agent.cases.billing.handler import BillingDisputeHandler
from service.billing.db import get_billing_engine, reset_business_data
from service.billing.errors import BillingError
from service.billing.models import Identity
from service.billing.scenarios import SCENARIOS, build_scenario
from service.billing.simulator_service import SimulatorService
from service.billing.timeutil import local_to_utc
from utils.config_handler import billing_conf


def direct_check(sim, scenario):
    modes = ["offline", "deepseek"]
    planner = st.selectbox(
        "本次核查规划模式",
        modes,
        index=modes.index(billing_conf["planner"]),
        key="direct_planner",
    )
    if st.button("直接发起费用核查"):
        st.session_state.pop("direct_result", None)
        if planner == "deepseek" and not os.getenv("DEEPSEEK_API_KEY", "").strip():
            st.error("请配置 DEEPSEEK_API_KEY 后再发起核查，或选择 offline 模式。")
            return
        try:
            with st.spinner("正在核查费用……"):
                result = BillingDisputeHandler(sim.engine).start(
                    Identity(role="customer", actor_id=scenario["customer_id"]),
                    scenario["suggested_question"],
                    scenario["suggested_locator"],
                    planner=planner,
                )
        except BillingError as error:
            st.error(error.message)
            return
        except Exception:
            if planner != "deepseek":
                raise
            st.error("DeepSeek 核查调用失败，请检查模型配置和服务连接。")
            return
        summary = result.model_dump(
            include={
                "case_id",
                "run_id",
                "status",
                "customer_message",
                "clarification_question",
            }
        )
        summary["model_failed"] = planner == "deepseek" and bool(
            (result.report or {}).get("failure")
        )
        st.session_state.direct_result = summary

    if result := st.session_state.get("direct_result"):
        if result["model_failed"]:
            st.error("DeepSeek 核查调用异常。可用下方运行号查看失败节点。")
        st.write("核查状态", result["status"])
        if message := result["customer_message"] or result["clarification_question"]:
            st.write(message)
        if result["status"] == "action_pending_approval":
            st.info("请在「资金审批台」选择下方案件号，查看方案并审批。")
        st.caption("案件号（case_id）")
        st.code(result["case_id"], language=None)
        st.caption("运行号（run_id），可复制到「案件运行轨迹」查看")
        st.code(result["run_id"], language=None)


def run():
    st.title("车主模拟器")
    sim = SimulatorService(get_billing_engine())
    tz = billing_conf["local_timezone"]
    now = datetime.now(ZoneInfo(tz)).replace(tzinfo=None, second=0, microsecond=0)
    if message := st.session_state.pop("sim_notice", None):
        st.success(message)
    customers = {v.customer_id: v.nickname for v in sim.list_customers()}
    if selected := st.session_state.pop("next_driver", None):
        st.session_state["sim_driver"] = selected
    customer = st.selectbox(
        "当前车主", list(customers), format_func=customers.get, key="sim_driver"
    )
    with st.expander("新建车主"):
        nickname = st.text_input("昵称", max_chars=64)
        if st.button("开户"):
            st.session_state.next_driver = sim.create_customer(nickname).customer_id
            done("已开户")
    w = sim.get_wallet(customer)
    cols = st.columns(3)
    for col, label, value in zip(
        cols,
        ["账面余额", "冻结", "可用余额"],
        [w.balance_fen, w.frozen_fen, w.available_fen],
    ):
        col.metric(label, f"{value:,} 分")
    st.caption("钱包与流水一致" if w.consistent else "钱包与流水不一致")
    txns = sim.list_transactions(customer)
    orders = sim.list_orders(customer)
    tabs = st.tabs(
        ["充值", "扫码启动", "结束充电", "补传读数", "运营人工退款", "一键生成场景"],
        key="sim_operation_tab",
        on_change="rerun",
    )
    with tabs[0]:
        amount = st.number_input("充值金额（分）", min_value=1, value=10000)
        fault = st.selectbox(
            "充值故障", [None, "callback_pending"], format_func=lambda x: x or "无"
        )
        if st.button("充值入账"):
            sim.recharge(customer, amount, fault=fault)
            done("充值已记录")
        pending = [
            t.txn_id for t in txns if t.txn_type == "recharge" and t.status == "pending"
        ]
        target = st.selectbox("待确认充值", pending)
        if st.button("确认充值", disabled=not target):
            sim.confirm_recharge(target)
            done("充值已确认")
    with tabs[1]:
        chargers = {
            f"{c.charger_id} / {c.connector_id}": c for c in sim.list_chargers()
        }
        charger = chargers[st.selectbox("充电桩 / 枪", list(chargers))]
        start = st.datetime_input("启动时间（本地）", value=now)
        wh = st.number_input("起始读数（Wh）", min_value=0)
        fault = st.selectbox(
            "启动故障",
            [None, "start_timeout", "timeout_keep_frozen"],
            format_func=lambda x: x or "无",
        )
        if st.button("扫码启动充电"):
            sim.start_charging(
                customer,
                charger.charger_id,
                charger.connector_id,
                at=local_to_utc(start, tz),
                start_wh=wh,
                fault=fault,
            )
            done("已记录启动尝试")
    with tabs[2]:
        finished = {o.session_id for o in orders}
        active = [
            a.session_id
            for a in sim.list_attempts(customer)
            if a.session_id and a.session_id not in finished
        ]
        session = st.selectbox("进行中的会话", active)
        end_wh = st.number_input("结束读数（Wh）", min_value=0, value=4000)
        end = st.datetime_input("停止时间（本地）", value=now + timedelta(hours=1))
        unplug = st.datetime_input(
            "拔枪时间（本地）", value=now + timedelta(hours=1, minutes=10)
        )
        fault = st.selectbox(
            "结算故障",
            [
                None,
                "missing_end_reading",
                "duplicate_deduct",
                "wrong_tariff_peak",
                "idle_fee_ignore_grace",
            ],
            format_func=lambda x: x or "无",
        )
        if st.button("结束并结算", disabled=not session):
            sim.stop_charging(
                session,
                end_wh=end_wh,
                charge_end_at=local_to_utc(end, tz),
                unplugged_at=local_to_utc(unplug, tz),
                fault=fault,
            )
            done("已记录结束充电")
    with tabs[3]:
        pending = [o for o in orders if o.status == "pending_settlement"]
        target = st.selectbox("缺结束读数的会话", [o.session_id for o in pending])
        wh = st.number_input("补传读数（Wh）", min_value=0, value=4000)
        when = st.datetime_input("读数发生时间（本地）", value=now)
        if st.button("提交补传", disabled=not target):
            sim.backfill_end_reading(target, wh, local_to_utc(when, tz))
            done("已补传并结算")
    with tabs[4]:
        target = st.selectbox(
            "原扣款流水", [t.txn_id for t in txns if t.txn_type == "deduct"]
        )
        amount = st.number_input("退款金额（分）", min_value=1)
        status = st.selectbox("退款状态", ["pending", "succeeded"])
        if st.button("记录人工退款", disabled=not target):
            sim.create_ops_refund(target, amount, status)
            done("人工退款已记录")
    with tabs[5]:
        name = st.selectbox("选择场景", list(SCENARIOS))
        if st.button("生成场景"):
            h = build_scenario(sim, name)
            st.session_state.pop("direct_result", None)
            st.session_state.scenario = h.model_dump()
            st.session_state.next_driver = h.customer_id
            done("已生成 " + name)
        if h := st.session_state.get("scenario"):
            st.caption(f"已生成 {h['name']} · 核查车主 {h['customer_id']}")
            st.write(h["suggested_question"])
            st.json(h["refs"])
            st.link_button(
                "切换到该车主去聊天",
                "/?" + urlencode({"customer_id": h["customer_id"]}),
            )
            direct_check(sim, h)
    st.subheader("订单")
    st.dataframe([o.model_dump() for o in orders], width="stretch")
    st.subheader("钱包流水")
    st.dataframe([t.model_dump() for t in txns], width="stretch")
    with st.expander("重置模拟数据"):
        confirmed = st.checkbox("确认清空当前模拟库的全部车主、订单、钱包和案件")
        if st.button("重置并恢复种子数据", disabled=not confirmed):
            reset_business_data(sim.engine)
            st.session_state.pop("scenario", None)
            st.session_state.pop("direct_result", None)
            st.session_state.next_driver = "CUS-DEMO-ALICE"
            done("已恢复种子数据")


def done(message: str):
    st.session_state.sim_notice = message
    st.rerun()


try:
    run()
except BillingError as error:
    st.error(error.message)
