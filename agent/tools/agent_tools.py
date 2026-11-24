import os

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.rag_service import RagSummerizeService
from service.user_data_service import (
    get_user_profile as query_user_profile,
    get_user_devices as query_user_devices,
    get_cleaning_records as query_cleaning_records,
    get_monthly_report_data,
)
from service.json_handler import to_json_str
from utils.memory_handler import AgentMemory
from utils.session_context import get_current_session_id

import random

rag = RagSummerizeService()
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
external_data = {}
agent_memory = AgentMemory()
DEFAULT_SESSION_ID = "default-session"

@tool(description=(
    "回答新能源充电桩知识库问题，返回带出处的答案、需要用户补充的问题或证据不足说明。"
    "query 须保留问题及已知型号、条件和限制。"
))
def rag_summerize(query: str, runtime: ToolRuntime[dict | None] = None) -> str:
    if runtime is None:
        return rag.rag_summerize(query)

    budget = (runtime.context or {}).get("rag_budget")
    if budget is None:
        return "知识库请求上下文未初始化，无法检索资料。"
    return rag.rag_summerize(query, budget=budget)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26度。"


@tool(description="【模拟工具】随机获取用户所在城市，仅在测试场景使用；真实用户信息优先使用MySQL数据库工具")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="【模拟工具】随机获取一个用户ID，仅当用户没有提供用户ID，且 get_memory_context 中也没有 user_id 时才可以调用。若当前会话记忆中已有 user_id，禁止调用该工具。")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串的形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id":{
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            ...
        },
        "user_id":{
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            ...
        },
        "user_id":{
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            "month":{"特征": xxx,"效率": xxx, ...}
            ...
        },
        ...
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="【旧CSV模拟工具】从CSV读取用户使用记录，仅兼容旧报告流程；真实报告优先使用MySQL数据库工具")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[generate_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")

@tool(description="无入参，无返回值,调用后触发中间件自动为报告生成场景动态注入上线纹信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report以调用"

#查询用户画像
@tool(description="根据用户ID从MySQL数据库查询用户画像信息，包括用户名、所在城市、房屋面积、家庭类型")
def get_user_profile_from_db(user_id: str) -> str:
    data = query_user_profile(user_id)

    if not data:
        return to_json_str({
            "found": False,
            "message": f"未查询到用户{user_id}的基本信息",
            "data": None
        })

    return to_json_str({
        "found": True,
        "data": data
    })

#查询用户设备
@tool(description="根据用户ID从MySQL数据库查询用户绑定的扫地机器人设备信息，包括设备名称、型号、序列号、固件版本、购买日期和状态")
def get_user_devices_from_db(user_id: str) -> str:
    data = query_user_devices(user_id)
    if not data:
        return to_json_str({
            "found": False,
            "message": f"未查询到用户{user_id}的设备",
            "data": None
        })
    return to_json_str({
        "found": True,
        "data": data
    })
#查询清扫记录
@tool(description="根据用户ID和月份从MySQL数据库查询扫地机器人清扫记录")
def get_cleaning_records_from_db(user_id: str, month: str) -> str:
    data = query_cleaning_records(user_id, month)

    if not data:
        return to_json_str({
            "found": False,
            "message": f"未查询到用户{user_id}在{month}的清扫记录",
            "data": []
        })

    return to_json_str({
        "found": True,
        "data": data
    })

#查询月报完整数据
@tool(description="根据用户ID和月份从MySQL数据库查询生成月度使用报告所需的完整数据，包括用户画像、设备信息和清扫记录")
def get_monthly_report_data_from_db(user_id: str, month: str) -> str:
    data = get_monthly_report_data(user_id, month)
    if not data:
        return to_json_str({
            "found": False,
            "message": "未查询到用户信息",
            "data": None
        })
    return to_json_str({
        "found": True,
        "data": data
    })

@tool(description="当用户引用先前保存的设备、地点或咨询意图时，读取当前会话记忆。费用案件的车主身份由服务端提供。")
def get_memory_context() -> str:
    session_id = get_current_session_id()
    memory = agent_memory.get_memory(session_id)

    return to_json_str({
        "session_id": session_id,
        "found": bool(memory),
        "data": memory,
    })

@tool(description="按用户要求保存当前会话的上下文，例如充电设备编号、城市或咨询意图。")
def save_memory_context(
    user_id: str | None = None,
    city: str | None = None,
    month: str | None = None,
    device_id: str | None = None,
    last_intent: str | None = None,
) -> str:
    session_id = get_current_session_id()

    memory = agent_memory.update_memory(
        session_id,
        user_id=user_id,
        city=city,
        month=month,
        device_id=device_id,
        last_intent=last_intent,
    )

    return to_json_str({
        "session_id": session_id,
        "saved": True,
        "data": memory,
    })

@tool(description="清理当前会话中的Agent短期记忆。仅当用户明确要求清空记忆、重置会话或测试记忆功能时调用。")
def clear_memory_context() -> str:
    session_id = get_current_session_id()
    cleared = agent_memory.clear_memory(session_id)

    return to_json_str({
        "session_id": session_id,
        "cleared": cleared,
    })

