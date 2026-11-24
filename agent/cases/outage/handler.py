from agent.cases.base import CaseResult
from agent.tools.investigation_tools import run_outage


class OutageInvestigationHandler:
    name = "outage_investigation"
    description = "调查指定充电站和时间窗内的掉线或升级异常"

    def start(
        self, identity, question, locator, *, chat_session_id=None, planner="offline"
    ):
        result = run_outage(
            locator.get("site_ids", []),
            locator.get("start_utc", ""),
            locator.get("end_utc", ""),
            question,
            planner=planner,
            chat_session_id=chat_session_id,
        )
        run = result.get("run_id", "")
        return CaseResult(
            case_id=result.get("case_id",run),
            run_id=run,
            status=result["status"],
            customer_message=result.get("message"),
            report=result,
        )

    def resume(self, run_id, decision):
        raise NotImplementedError("掉线调查不涉及资金审批")

    def rerun(self, case_id, identity):
        raise NotImplementedError("请提供原站点和时间窗重新发起调查")
