from typing import Iterator

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
)

from agent.tools.agent_tools import (
    clear_memory_context,
    fetch_external_data,
    fill_context_for_report,
    get_cleaning_records_from_db,
    get_current_month,
    get_memory_context,
    get_monthly_report_data_from_db,
    get_user_devices_from_db,
    get_user_id,
    get_user_location,
    get_user_profile_from_db,
    get_weather,
    rag_summerize,
    save_memory_context,
)
from agent.tools.mcp_db_tools import (
    get_cleaning_records_from_mcp,
    get_monthly_report_data_from_mcp,
    get_user_devices_from_mcp,
    get_user_profile_from_mcp,
)
from agent.tools.middleware import (
    log_before_model,
    monitor_tool,
    report_prompt_switch,
    trim_conversation_history,
)
from model.factory import chat_model
from utils.chat_checkpointer import (
    chat_checkpointer,
)
from utils.prompt_loader import (
    load_system_prompts,
)
from utils.session_context import (
    reset_current_session_id,
    set_current_session_id,
)


class ReactAgent:
    """
    机器人智能客服Agent。

    session_id同时作为：

    1. LangGraph thread_id；
    2. 结构化记忆session_id；
    3. Streamlit页面会话标识。
    """

    def __init__(
        self,
        session_id: str = "default-session",
    ):
        if not session_id:
            raise ValueError(
                "session_id不能为空"
            )

        session_id = session_id.strip()

        if not session_id:
            raise ValueError(
                "session_id不能为空"
            )

        self.session_id = session_id

        self.agent = create_agent(
            model=chat_model,
            system_prompt=(
                load_system_prompts()
            ),
            tools=[
                rag_summerize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
                get_user_profile_from_db,
                get_user_devices_from_db,
                get_cleaning_records_from_db,
                get_monthly_report_data_from_db,
                get_memory_context,
                save_memory_context,
                clear_memory_context,
                get_user_profile_from_mcp,
                get_user_devices_from_mcp,
                get_cleaning_records_from_mcp,
                get_monthly_report_data_from_mcp,
            ],
            middleware=[
                # 先裁剪，再记录裁剪后的消息数量
                trim_conversation_history,
                monitor_tool,
                log_before_model,
                report_prompt_switch,
            ],
            checkpointer=chat_checkpointer,
        )

    def get_checkpoint_config(
        self,
    ) -> RunnableConfig:
        """
        获取当前会话的LangGraph配置。
        """

        return {
            "configurable": {
                "thread_id": self.session_id,
            }
        }

    def execute_stream(
        self,
        query: str,
    ) -> Iterator[str]:
        """
        流式执行Agent。

        请求结束或发生异常时，
        自动恢复执行前的ContextVar。
        """

        if not query:
            return

        query = query.strip()

        if not query:
            return

        context_token = (
            set_current_session_id(
                self.session_id
            )
        )

        input_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }

        runtime_context = {
            "report": False,
            "session_id": self.session_id,
        }

        emitted_message_ids: set[str] = (
            set()
        )

        try:
            for chunk in self.agent.stream(
                input_dict,
                config=(
                    self.get_checkpoint_config()
                ),
                context=runtime_context,
                stream_mode="values",
            ):
                messages = chunk.get(
                    "messages",
                    [],
                )

                if not messages:
                    continue

                latest_message = (
                    messages[-1]
                )

                # 不显示用户消息和工具返回消息
                if not isinstance(
                    latest_message,
                    AIMessage,
                ):
                    continue

                # 带tool_calls的AIMessage
                # 只是工具调用请求，不是最终回答
                if latest_message.tool_calls:
                    continue

                message_id = (
                    latest_message.id
                )

                if (
                    message_id
                    and message_id
                    in emitted_message_ids
                ):
                    continue

                content = (
                    latest_message.content
                )

                if not isinstance(
                    content,
                    str,
                ):
                    continue

                content = content.strip()

                if not content:
                    continue

                if message_id:
                    emitted_message_ids.add(
                        message_id
                    )

                yield content

        finally:
            # 无论正常完成还是异常，
            # 都恢复请求前的session_id上下文
            reset_current_session_id(
                context_token
            )

    def get_chat_history(
        self,
    ) -> list[dict[str, str]]:
        """
        从Redis Checkpointer恢复页面聊天记录。

        只返回：

        1. HumanMessage；
        2. 不包含tool_calls的最终AIMessage。

        不显示内部ToolMessage。
        """

        checkpoint = (
            chat_checkpointer.get(
                self.get_checkpoint_config()
            )
        )

        if checkpoint is None:
            return []

        messages = checkpoint.get(
            "channel_values",
            {},
        ).get(
            "messages",
            [],
        )

        history: list[
            dict[str, str]
        ] = []

        for message in messages:
            content = getattr(
                message,
                "content",
                None,
            )

            if not isinstance(
                content,
                str,
            ):
                continue

            content = content.strip()

            if not content:
                continue

            if isinstance(
                message,
                HumanMessage,
            ):
                history.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )

            elif isinstance(
                message,
                AIMessage,
            ):
                if message.tool_calls:
                    continue

                history.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

        return history

    def clear_chat_history(
        self,
    ) -> None:
        """
        删除当前thread_id的Redis Checkpoint。
        """

        chat_checkpointer.delete_thread(
            self.session_id
        )

    def clear_conversation_memory(
        self,
    ) -> None:
        """
        保留旧方法名，兼容已有演示代码。
        """

        self.clear_chat_history()


if __name__ == "__main__":
    agent = ReactAgent(
        session_id=(
            "local-redis-test-session"
        )
    )

    questions = [
        "请记住，我的暗号是青竹-729。",
        "刚才的暗号是什么？",
    ]

    for question in questions:
        print(f"\n用户：{question}")
        print(
            "机器人：",
            end="",
        )

        for response_chunk in (
            agent.execute_stream(
                question
            )
        ):
            print(
                response_chunk,
                end="",
                flush=True,
            )

        print()