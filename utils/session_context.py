"""
当前 Agent 请求的 session_id 上下文。

使用 ContextVar，避免多个并发请求之间互相覆盖 session_id。
"""

from contextvars import ContextVar, Token


_current_session_id: ContextVar[str] = ContextVar(
    "current_agent_session_id",
    default="default-session",
)


def set_current_session_id(
    session_id: str,
) -> Token:
    """
    设置当前执行上下文中的 session_id。
    """

    if not session_id or not session_id.strip():
        raise ValueError("session_id不能为空")

    return _current_session_id.set(
        session_id.strip()
    )


def get_current_session_id() -> str:
    """
    获取当前执行上下文中的 session_id。
    """

    return _current_session_id.get()


def reset_current_session_id(
    token: Token,
) -> None:
    """
    在需要时恢复设置前的 session_id。
    """

    _current_session_id.reset(token)