import time
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from service.billing.db import get_billing_engine
from service.billing.simulator_service import SimulatorService

from agent.react_agent import ReactAgent
from utils.memory_handler import AgentMemory


def read_query_param(name: str) -> str | None:
    """
    安全读取 URL 查询参数，兼容列表返回值。
    """

    value = st.query_params.get(name)
    if isinstance(value, list):
        value = value[-1] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def create_new_session(driver_key: str | None = None) -> None:
    """
    为当前车主创建全新的页面会话。
    """

    new_session_id = str(uuid.uuid4())
    key = driver_key or st.session_state.get("customer_id") or "__default__"
    driver_sessions = st.session_state.setdefault("driver_sessions", {})
    driver_sessions[key] = new_session_id

    st.session_state["session_id"] = (
        new_session_id
    )

    st.session_state.pop("agent_thread_id", None)
    st.session_state.pop("agent", None)
    st.session_state.pop("message", None)

    st.query_params["session_id"] = (
        new_session_id
    )

    st.rerun()


st.title("新能源充电桩智能客服")
st.divider()


try:
    drivers = {c.customer_id: c.nickname for c in SimulatorService(get_billing_engine()).list_customers()}
except SQLAlchemyError:
    drivers = {}
    st.sidebar.info("费用核查暂不可用，请检查 MySQL 并运行计费库初始化。")
previous_driver = st.session_state.get("customer_id")
if drivers:
    driver_ids = list(drivers)
    if st.session_state.get("chat_driver") not in drivers:
        st.session_state.pop("chat_driver", None)
    requested_driver = read_query_param("customer_id") or previous_driver
    selected_driver = st.sidebar.selectbox(
        "当前车主", driver_ids,
        index=driver_ids.index(requested_driver) if requested_driver in drivers else 0,
        format_func=drivers.get, key="chat_driver",
    )
    st.query_params["customer_id"] = selected_driver
else:
    selected_driver = None
st.session_state["customer_id"] = selected_driver

driver_key = selected_driver or "__default__"
driver_sessions = st.session_state.setdefault("driver_sessions", {})
driver_changed = previous_driver is not None and previous_driver != selected_driver

if driver_changed:
    active_session_id = driver_sessions.get(driver_key) or str(uuid.uuid4())
else:
    active_session_id = (
        read_query_param("session_id")
        or driver_sessions.get(driver_key)
        or str(uuid.uuid4())
    )

driver_sessions[driver_key] = active_session_id
st.query_params["session_id"] = active_session_id

target_thread_id = (
    f"{active_session_id}:{selected_driver}"
    if selected_driver
    else active_session_id
)

# 会话 ID 或当前车主发生变化时，切换对应车主的 Agent 与历史记录。
if (
    st.session_state.get("session_id") != active_session_id
    or st.session_state.get("agent_thread_id") != target_thread_id
):
    st.session_state["session_id"] = active_session_id
    st.session_state["agent_thread_id"] = target_thread_id

    st.session_state.pop("agent", None)
    st.session_state.pop("message", None)


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent(
        session_id=target_thread_id
    )


# 页面第一次打开或服务重启后，
# 从Redis恢复最近聊天记录。
if "message" not in st.session_state:
    st.session_state["message"] = (
        st.session_state[
            "agent"
        ].get_chat_history()
    )


memory = AgentMemory()


with st.sidebar:
    st.caption(
        "Session ID:"
    )

    st.code(
        st.session_state["session_id"],
        language=None,
    )

    if st.button(
        "新建会话",
        use_container_width=True,
    ):
        create_new_session()

    if st.button(
        "清空当前会话",
        use_container_width=True,
    ):
        try:
            # 清除Redis中的LangGraph聊天记录。
            st.session_state[
                "agent"
            ].clear_chat_history()

            # 清除user_id、city、month等结构化记忆。
            memory.clear_memory(
                    st.session_state["agent"].session_id
            )

            # 清除页面展示消息。
            st.session_state["message"] = []

            st.success("当前会话已清空")

        except Exception as error:
            st.error(
                f"清空当前会话失败：{error}"
            )


for message in st.session_state["message"]:
    st.chat_message(
        message["role"]
    ).write(
        message["content"]
    )


prompt = st.chat_input(
    "请输入你的问题"
)


if prompt:
    st.chat_message("user").write(prompt)

    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response_parts: list[str] = []

    try:
        with st.spinner(
            "智能客服思考中……"
        ):

            def capture(generator):
                for chunk in generator:
                    response_parts.append(
                        chunk
                    )

                    for character in chunk:
                        time.sleep(0.005)
                        yield character

            response_generator = (
                st.session_state[
                    "agent"
                ].execute_stream(prompt, customer_id=selected_driver)
            )

            st.chat_message(
                "assistant"
            ).write_stream(
                capture(
                    response_generator
                )
            )

        response_text = "".join(
            response_parts
        ).strip()

        if response_text:
            st.session_state[
                "message"
            ].append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

        else:
            error_message = (
                "Agent未生成有效回答"
            )

            st.error(error_message)

            st.session_state[
                "message"
            ].append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

    except Exception as error:
        error_message = (
            f"处理请求失败：{error}"
        )

        st.error(error_message)

        st.session_state[
            "message"
        ].append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )
