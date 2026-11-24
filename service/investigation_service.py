"""Shared entry point for synchronous multi-agent outage investigations."""

from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from langsmith import tracing_context

from agent.investigation_graph import KnowledgeResult
from agent.investigation_specialists import (
    SpecialistDependencies,
    collect_source_ids,
    plan_specialist_offline,
    plan_specialist_with_deepseek,
)
from agent.multi_agent_investigation_graph import create_multi_agent_investigation_graph
from service.investigation_data_service import InvestigationScope, InvestigationSource


def lookup_known_issue_offline(_query: str) -> KnowledgeResult:
    return KnowledgeResult(
        status="answered",
        text="离线模拟线索：对齐设备掉线时间、站点网关日志和升级完成时间，供后续核验。",
        sources=[{"id": "SIM-KB-01", "source": "离线模拟运维核验清单"}],
        simulated=True,
    )


def lookup_known_issue_with_rag(query: str) -> KnowledgeResult:
    from rag.rag_service import RagSummerizeService

    result = RagSummerizeService().answer_question(query)
    return KnowledgeResult(
        status=result.status, text=result.text, sources=result.sources
    )


def multi_dependencies_for(
    db_path: InvestigationSource, mode: str
) -> SpecialistDependencies:
    if mode == "offline":
        return SpecialistDependencies(
            db_path, plan_specialist_offline, lookup_known_issue_offline
        )
    if mode == "deepseek":
        return SpecialistDependencies(
            db_path, plan_specialist_with_deepseek, lookup_known_issue_with_rag
        )
    raise ValueError("调查规划器必须是 offline 或 deepseek")


def _specialist_output(state: dict) -> dict:
    return {
        key: state[key]
        for key in (
            "task",
            "role",
            "decisions",
            "tool_calls",
            "remaining_tools",
            "finding",
        )
    }


def run_investigation(
    question: str,
    scope: InvestigationScope,
    dependencies: SpecialistDependencies,
    *,
    mode: str,
    simulated_data: bool,
    chat_session_id: str | None = None,
) -> dict:
    if not question.strip():
        raise ValueError("请输入需要调查的问题")
    source = dependencies.db_path
    store = None
    if isinstance(source, (str, Path)):
        if not Path(source).is_file():
            raise ValueError("后端调查 SQLite 数据文件不存在")
    else:
        from service.billing.case_store import CaseStore

        with source.open():
            pass  # Check availability before creating an operations case.
        store = CaseStore(source.engine)

    graph = create_multi_agent_investigation_graph(dependencies)
    tracing = tracing_context(enabled=False) if mode == "offline" else nullcontext()
    run_id = uuid4().hex
    if store is not None:
        case_id, run_id = store.start_case_run(
            handler="outage_investigation",
            customer_id="ops:outage",
            question=question.strip(),
            chat_session_id=chat_session_id,
            planner=mode,
        )
    inputs = {"run_id": run_id, "question": question.strip(), "scope": scope.to_dict()}
    with tracing:
        if store is None:
            return _investigation_result(graph.invoke(inputs), mode, simulated_data)
        try:
            store.append_trace(
                run_id,
                "outage",
                "node_start",
                {"dataset": source.dataset, "scope": scope.to_dict()},
            )
            for kind, chunk in graph.stream(inputs, stream_mode=["updates", "values"]):
                if kind == "values":
                    state = chunk
                    continue
                for node, update in chunk.items():
                    for field in ("firmware_initial", "site_initial", "review_result"):
                        for call in (update.get(field) or {}).get("tool_calls", []):
                            store.append_trace(run_id, node, "tool_call", call)
                    store.append_trace(
                        run_id, node, "node_end", {"fields": list(update)}
                    )
            result = _investigation_result(state, mode, simulated_data)
            result.update(case_id=case_id, dataset=source.dataset)
            from agent.cases.outage.billing_followup import (
                investigate_interrupted_sessions,
            )

            result["billing_followups"] = investigate_interrupted_sessions(
                source, scope, result, store
            )
            store.append_trace(run_id, "outage", "node_end", {"status": "completed"})
            store.finish_run(run_id, "completed", result)
            return result
        except Exception as error:
            payload = {
                "case_id": case_id,
                "run_id": run_id,
                "status": "failed",
                "error": type(error).__name__,
            }
            store.append_trace(run_id, "outage", "error", payload)
            store.finish_run(run_id, "failed", payload)
            raise


def _investigation_result(state, mode, simulated_data):
    shared_queries = {
        name: {
            "status": result["status"],
            "reason": result["reason"],
            "source_ids": collect_source_ids(result),
        }
        for name, result in state["shared_evidence"].items()
    }
    review = state.get("review_result")
    return {
        "run_id": state["run_id"],
        "planner": mode,
        "simulated_data": simulated_data,
        "question": state["question"],
        "scope": state["scope"],
        "shared_queries": shared_queries,
        "initial": {
            role: _specialist_output(state[f"{role}_initial"])
            for role in ("firmware", "site")
        },
        "coordination_trace": state["coordination_trace"],
        "review_request": state.get("review_request"),
        "reviews_remaining": state["reviews_remaining"],
        "review": _specialist_output(review) if review else None,
        "report": state["report"],
    }
