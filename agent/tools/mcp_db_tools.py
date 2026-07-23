from langchain_core.tools import tool

from mcp_servers.mysql_mcp_client import call_mysql_mcp_tool


@tool
def get_user_profile_from_mcp(user_id: str) -> str:
    """通过 MCP 查询用户基本信息。用于查询用户姓名、城市、房屋面积、家庭类型。"""
    return call_mysql_mcp_tool(
        "get_user_profile",
        {"user_id": user_id},
    )


@tool
def get_user_devices_from_mcp(user_id: str) -> str:
    """通过 MCP 查询用户绑定的扫地机器人设备信息。"""
    return call_mysql_mcp_tool(
        "get_user_devices",
        {"user_id": user_id},
    )


@tool
def get_cleaning_records_from_mcp(user_id: str, month: str) -> str:
    """通过 MCP 查询用户指定月份的清扫记录。month 格式为 YYYY-MM。"""
    return call_mysql_mcp_tool(
        "get_cleaning_records",
        {
            "user_id": user_id,
            "month": month,
        },
    )


@tool
def get_monthly_report_data_from_mcp(user_id: str, month: str) -> str:
    """通过 MCP 查询生成月度报告所需的完整数据，包括用户、设备、清扫记录。month 格式为 YYYY-MM。"""
    return call_mysql_mcp_tool(
        "get_monthly_report_data",
        {
            "user_id": user_id,
            "month": month,
        },
    )