from typing import Literal

from langchain.tools import ToolRuntime, tool
from sqlalchemy.exc import SQLAlchemyError

from service.billing.db import get_billing_engine
from service.billing.errors import BillingError
from service.billing.models import Identity
from service.memory.customer_memory_service import CustomerMemoryService
from utils.config_handler import memory_conf

MemoryKey = Literal["vehicle_model", "home_station", "favorite_charger", "reply_style", "contact_note"]


def _for_customer(runtime, operation):
    context = (runtime.context or {}) if runtime else {}
    if not (customer := context.get("customer_id")):
        return {"status": "need_login"}
    if not memory_conf["enable_long_term_memory"]:
        return {"status": "disabled"}
    try:
        service = CustomerMemoryService(get_billing_engine())
        return operation(service, Identity(role="customer", actor_id=customer), context)
    except BillingError as error:
        return {"status": error.code.lower(), "message": error.message}
    except SQLAlchemyError:
        return {"status": "unavailable", "message": "长期记忆暂时不可用"}


@tool
def get_customer_memory(runtime: ToolRuntime[dict | None] = None) -> dict:
    """读取当前登录车主的车型、常用站点和充电桩、回复偏好及历史案件索引。"""
    def read(service, identity, _):
        records = service.list_active(identity, limit=memory_conf["max_injected_memories"])
        return {"status": "ok", "memories": [
            record.model_dump(mode="json", exclude={"customer_id", "source_session_id"})
            for record in records
        ]}

    return _for_customer(runtime, read)


@tool
def save_customer_memory(
    memory_key: MemoryKey, memory_value: str, runtime: ToolRuntime[dict | None] = None,
) -> dict:
    """按车主明确表达保存或更新长期画像和偏好。禁止保存退款承诺、账务结论或他人信息。"""
    def save(service, identity, context):
        result = service.upsert_memory(
            identity, memory_key=memory_key, memory_value=memory_value,
            session_id=context.get("session_id"), reason="explicit_chat_request",
        )
        return {"status": "ok", "action": result.action, "version": result.memory.version}

    return _for_customer(runtime, save)


@tool
def delete_customer_memory(
    memory_key: str | None = None, runtime: ToolRuntime[dict | None] = None,
) -> dict:
    """根据车主明确要求遗忘指定长期记忆。仅在车主明确要求全部遗忘时省略记忆键。"""
    def forget(service, identity, _):
        count = service.delete_memory(identity, memory_key=memory_key, reason="explicit_chat_request")
        return {"status": "ok", "deleted": count}

    return _for_customer(runtime, forget)


CUSTOMER_MEMORY_TOOLS = [get_customer_memory, save_customer_memory, delete_customer_memory]
