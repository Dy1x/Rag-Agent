import time
import uuid

import streamlit as st

from agent.react_agent import ReactAgent
from utils.memory_handler import AgentMemory


def get_url_session_id() -> str:
    """
    从URL读取session_id。

    如果URL中没有session_id，则生成新的并写入URL。
    """

    query_session_id = st.query_params.get(
        "session_id"
    )

    # 兼容可能返回列表的情况。
    if isinstance(query_session_id, list):
        query_session_id = (
            query_session_id[-1]
            if query_session_id
            else None
        )

    if query_session_id:
        query_session_id = str(
            query_session_id
        ).strip()

    if query_session_id:
        return query_session_id

    new_session_id = str(uuid.uuid4())

    st.query_params["session_id"] = (
        new_session_id
    )

    return new_session_id


def create_new_session() -> None:
    """
    创建全新的页面会话。
    """

    new_session_id = str(uuid.uuid4())

    st.session_state["session_id"] = (
        new_session_id
    )

    st.session_state.pop("agent", None)
    st.session_state.pop("message", None)

    st.query_params["session_id"] = (
        new_session_id
    )

    st.rerun()


st.title("机器人智能客服")
st.divider()


url_session_id = get_url_session_id()


# URL中的session_id发生变化时，切换当前会话。
if (
    st.session_state.get("session_id")
    != url_session_id
):
    st.session_state["session_id"] = (
        url_session_id
    )

    st.session_state.pop("agent", None)
    st.session_state.pop("message", None)


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent(
        session_id=st.session_state[
            "session_id"
        ]
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
                st.session_state[
                    "session_id"
                ]
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
                ].execute_stream(prompt)
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