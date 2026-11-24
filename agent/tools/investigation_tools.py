"""Chat tool for investigating a concrete charging-station incident."""

from os import environ
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from langchain_core.tools import tool

from service.investigation_data_service import InvestigationScope
from service.investigation_service import multi_dependencies_for, run_investigation


@tool
def investigate_outage(
    site_ids: list[str], start_utc: str, end_utc: str, question: str
) -> dict:
    """调查指定充电站和明确 UTC 时间窗内的掉线或升级异常。

    Args:
        site_ids: 用户确认的站点 ID 列表。
        start_utc: 明确的 ISO 8601 UTC 开始时间，包含该时刻。
        end_utc: 明确的 ISO 8601 UTC 结束时间，排除该时刻。
        question: 用户希望调查的具体问题。
    """
    return run_outage(site_ids, start_utc, end_utc, question)


def run_outage(
    site_ids, start_utc, end_utc, question, *, planner=None, chat_session_id=None
):
    try:
        InvestigationScope(site_ids, start_utc, end_utc, "available")
        if not question.strip():
            raise ValueError("请输入需要调查的问题")
    except ValueError as error:
        return {"status": "invalid_scope", "message": str(error)}

    path = environ.get("INVESTIGATION_DB_PATH", "").strip()
    source = environ.get("INVESTIGATION_FIRMWARE_SOURCE", "").strip()
    backend = environ.get("INVESTIGATION_BACKEND", "sqlite").strip()
    if source not in ("available", "unavailable"):
        return {
            "status": "data_source_unconfigured",
            "message": "后端尚未配置调查固件来源状态。",
        }
    mode = planner or environ.get("INVESTIGATION_PLANNER", "deepseek").strip()
    try:
        if backend == "sqlite":
            db_path = Path(path)
            if not path or not db_path.is_file():
                return {
                    "status": "data_source_unconfigured",
                    "message": "后端调查 SQLite 数据文件不存在。",
                }
        elif backend == "mysql":
            from service.billing.db import get_billing_engine
            from service.investigation_mysql import MySQLInvestigationSource

            db_path = MySQLInvestigationSource(
                get_billing_engine(environ.get("INVESTIGATION_MYSQL_DATABASE") or None),
                environ.get("INVESTIGATION_DATASET", ""),
            )
        else:
            raise ValueError("调查数据源须为 sqlite 或 mysql")
        dependencies = multi_dependencies_for(db_path, mode)
        scope = InvestigationScope(
            site_ids,
            start_utc,
            end_utc,
            source,
            environ.get("INVESTIGATION_FIRMWARE_SOURCE_REASON", "").strip() or None,
        )
        result = run_investigation(
            question,
            scope,
            dependencies,
            mode=mode,
            simulated_data=environ.get("INVESTIGATION_SIMULATED_DATA", "").strip()
            == "1",
            chat_session_id=chat_session_id,
        )
    except SQLAlchemyError:
        return {
            "status": "data_source_unconfigured",
            "message": "MySQL 调查库无法访问，请检查连接与导入状态。",
        }
    except ValueError as error:
        return {"status": "data_source_unconfigured", "message": str(error)}

    return {
        "status": "completed",
        **({"case_id": result["case_id"]} if "case_id" in result else {}),
        **(
            {"billing_followup_count": len(result["billing_followups"])}
            if "billing_followups" in result
            else {}
        ),
        "run_id": result["run_id"],
        "scope": result["scope"],
        "planner": result["planner"],
        "simulated_data": result["simulated_data"],
        "report": result["report"],
        "review_request": result["review_request"],
    }
