from datetime import date, datetime
from decimal import Decimal

from mcp.server.fastmcp import FastMCP

from service.user_data_service import (
    get_user_profile as query_user_profile,
    get_user_devices as query_user_devices,
    get_cleaning_records as query_cleaning_records,
    get_monthly_report_data as query_monthly_report_data,
)

mcp = FastMCP("mysql-query-server")


def to_jsonable(data):
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]

    if isinstance(data, dict):
        return {
            key: to_jsonable(value)
            for key, value in data.items()
        }

    if isinstance(data, (datetime, date)):
        return data.isoformat()

    if isinstance(data, Decimal):
        return float(data)

    return data


@mcp.tool()
def get_user_profile(user_id: str) -> dict:
    result = to_jsonable(query_user_profile(user_id))

    if not result:
        return {
            "found": False,
            "message": f"未查询到用户{user_id}的基本信息",
            "data": None,
        }

    return {
        "found": True,
        "data": result,
    }


@mcp.tool()
def get_user_devices(user_id: str) -> dict:
    result = to_jsonable(query_user_devices(user_id))

    if not result:
        return {
            "found": False,
            "message": f"未查询到用户{user_id}绑定的设备信息",
            "data": [],
        }

    return {
        "found": True,
        "data": result,
    }


@mcp.tool()
def get_cleaning_records(user_id: str, month: str) -> dict:
    result = to_jsonable(query_cleaning_records(user_id, month))

    if not result:
        return {
            "found": False,
            "message": f"未查询到用户{user_id}在{month}的清扫记录",
            "data": [],
        }

    return {
        "found": True,
        "data": result,
    }


@mcp.tool()
def get_monthly_report_data(user_id: str, month: str) -> dict:
    result = to_jsonable(query_monthly_report_data(user_id, month))

    if not result or not result.get("records"):
        return {
            "found": False,
            "message": f"未查询到用户{user_id}在{month}的月度报告数据",
            "data": result,
        }

    return {
        "found": True,
        "data": result,
    }


if __name__ == "__main__":
    print("MySQL MCP Server started. Waiting for MCP client...")
    mcp.run()