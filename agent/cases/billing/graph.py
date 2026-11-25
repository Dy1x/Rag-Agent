from functools import wraps
from dataclasses import replace

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.cases.agent_spec import build_agent_node
from service.billing.errors import BillingError
from service.billing.models import (
    ApprovalDecision,
    ApprovedAction,
    BillingFacts,
    CaseFile,
    Identity,
    Proposal,
    ReviewVerdict,
)
from service.memory.customer_memory_service import CustomerMemoryService
from utils.config_handler import billing_conf, memory_conf
from utils.logger_handler import logger
from . import checker, intake, maker
from .offline import RELEVANT, customer_message_issues, offline_maker, offline_soft
from .state import BillingCaseState
from .tools import locator_args, read_tools


def build_graph(service, store, executor, checkpointer, *, transform=None, design="C"):
    graph = StateGraph(BillingCaseState)
    dialogues = {}

    def trace(s, node, event, payload):
        store.append_trace(s["run_id"], node, event, payload)

    def wrap(name, fn):
        @wraps(fn)
        def node(s):
            trace(s, name, "node_start", {})
            try:
                update = fn(s)
            except Exception as exc:
                trace(s, name, "error", {"reason": str(exc)})
                raise
            trace(s, name, "node_end", {"status": update.get("status")})
            return update

        return node

    def role(s, spec, payload, offline):
        dialogue = None
        if design == "A" and spec.name in ("maker", "checker"):
            dialogue = dialogues.setdefault(s["run_id"], {})
            spec = replace(
                spec,
                name="single_agent",
                prompt_file=maker.SPEC.prompt_file,
                tools=list(dict.fromkeys(maker.SPEC.tools + checker.SPEC.tools)),
            )
        record = lambda event, value: trace(s, spec.name, event, value)
        tools = read_tools(service, Identity(**s["identity"]), record, s["planner"])
        invoke = build_agent_node(
            spec, s["planner"], tools, offline, record=record, dialogue=dialogue
        )
        output = invoke(payload)
        record(
            "llm_output",
            {"planner": s["planner"], "output": output.model_dump(mode="json")},
        )
        return output

    def locate(s):
        if session_id := s["locator"].get("confirmed_outage_session"):
            target = service.session_locator(Identity(**s["identity"]), session_id)
            case = CaseFile(
                status="located",
                dispute_type="outage_billing",
                attempt_ids=[target["attempt_id"]],
                order_ids=[target["order_id"]] if target["order_id"] else [],
            )
            return {"case_file": case.model_dump(), "status": "located"}
        case = role(
            s,
            intake.SPEC,
            {"question": s["question"], "locator": s["locator"]},
            intake.offline_intake,
        )
        # A wallet recharge can be located without a charging order. Check the
        # ledger before asking the driver for information already on file.
        if (
            case.status == "clarify"
            and case.dispute_type == "recharge_missing"
            and not s["locator"].get("order_id")
        ):
            trace(
                s,
                "intake",
                "tool_call",
                {"name": "build_billing_facts", "scope": "wallet"},
            )
            wallet = service.build_billing_facts(
                Identity(**s["identity"]), order_ids=[], attempt_ids=[]
            )
            if len(wallet.pending_recharges) == 1:
                case = CaseFile(status="located", dispute_type=case.dispute_type)
        # Resolve explicit locators and masked candidates using scoped rows.
        if s["planner"] == "deepseek" and (
            case.status == "clarify" or s["locator"].get("order_id")
        ):
            args = locator_args(s["locator"])
            trace(
                s,
                "intake",
                "tool_call",
                {"name": "find_candidate_orders", "scope": "validate_locator"},
            )
            rows = service.find_candidate_orders(Identity(**s["identity"]), **args)
            candidates = [r.model_dump(mode="json") for r in rows]
            if args.get("order_id"):
                case = intake.resolve_candidates(
                    case.dispute_type, candidates, order_id=args["order_id"]
                )
            else:
                case = intake.clarification(case.dispute_type, candidates)
        return {"case_file": case.model_dump(), "status": case.status}

    def gather(s):
        cf = CaseFile(**s["case_file"])
        trace(s, "facts", "tool_call", {"name": "build_billing_facts"})
        try:
            data = service.build_billing_facts(
                Identity(**s["identity"]),
                order_ids=cf.order_ids,
                attempt_ids=cf.attempt_ids,
            )
        except BillingError as e:
            if e.code == "FORBIDDEN":
                return {"status": "forbidden"}
            raise
        pending = (
            RELEVANT.get(cf.dispute_type, set())
            & {a.code for a in data.anomalies}
            & {"PENDING_RECHARGE", "MISSING_END_READING"}
        )
        if any(a.code == "WALLET_PROJECTION_MISMATCH" for a in data.anomalies):
            return {"facts": data.model_dump(mode="json"), "status": "escalated"}
        if cf.dispute_type == "outage_billing" and not data.orders:
            data.missing_evidence.append("settlement_order")
            return {"facts": data.model_dump(mode="json"), "status": "pending_evidence"}
        if not data.evidence_index:
            return {"facts": data.model_dump(mode="json"), "status": "pending_evidence"}
        return {
            "facts": data.model_dump(mode="json"),
            "status": "pending_evidence" if pending else "located",
        }

    def draft(s):
        version = len(s["proposals"]) + 1
        inputs = dict(
            question=s["question"],
            case_file=s["case_file"],
            facts=s["facts"],
            version=version,
            previous=s["proposals"][-1:] or None,
            issues=s["reviews"][-1]["issues"] if s["reviews"] else [],
        )
        proposal = role(s, maker.SPEC, inputs, offline_maker)
        from service.billing.ids import new_id

        proposal.proposal_id = new_id("PRP")
        proposal.version = version
        if transform and version == 1:
            proposal = transform(proposal, BillingFacts(**s["facts"]))
        store.save_proposal(s["run_id"], proposal)
        return {
            "proposals": [*s["proposals"], proposal.model_dump()],
            "revision": version - 1,
            "status": "proposed",
        }

    def review(s):
        p = Proposal(**s["proposals"][-1])
        cf = CaseFile(**s["case_file"])
        if design == "A":
            facts, issues = BillingFacts(**s["facts"]), []
        else:
            trace(
                s,
                "checker",
                "tool_call",
                {"name": "build_billing_facts", "independent": True},
            )
            facts, issues = checker.hard_checks(
                p, cf, service, Identity(**s["identity"])
            )
        hard = not issues
        approved = hard
        if hard and design != "B":
            inputs = dict(
                question=s["question"],
                case_file=s["case_file"],
                proposal=p.model_dump(),
            )
            if design == "A":
                inputs["instruction"] = (
                    "自查刚提交的方案，以 SoftReviewOpinion 返回结果。"
                )
                inputs["facts"] = s["facts"]
            opinion = role(
                s,
                checker.SPEC,
                inputs,
                lambda payload, tools: offline_soft(payload, tools, facts),
            )
            issues = opinion.issues
            if design == "C" and s["planner"] == "deepseek":
                reported = {issue.code for issue in issues}
                issues.extend(
                    i
                    for i in customer_message_issues(p.customer_message)
                    if i.code not in reported
                )
            approved = opinion.approved and not issues
        verdict = ReviewVerdict(
            proposal_id=p.proposal_id,
            hard_check_passed=hard,
            approved=approved,
            issues=issues,
        )
        store.save_review(p.proposal_id, verdict)
        if approved:
            status = "action_pending_approval" if p.actions else "explained"
        else:
            status = (
                "revise"
                if s["revision"] < billing_conf["max_revisions"]
                else "escalated"
            )
        return {"reviews": [*s["reviews"], verdict.model_dump()], "status": status}

    def pending(s):
        store.update_case(
            s["case_id"],
            status="action_pending_approval",
            dispute_type=s["case_file"]["dispute_type"],
            latest_run_id=s["run_id"],
        )
        trace(s, "await_approval", "node_start", {})
        trace(
            s,
            "await_approval",
            "interrupt",
            {"proposal_id": s["proposals"][-1]["proposal_id"]},
        )
        return {}

    def await_approval(s):
        value = interrupt(
            dict(
                case_id=s["case_id"],
                run_id=s["run_id"],
                proposal=s["proposals"][-1],
                review=s["reviews"][-1],
            )
        )
        decision = ApprovalDecision.model_validate(value)
        p = Proposal(**s["proposals"][-1])
        store.save_approval(p.proposal_id, decision)
        trace(s, "await_approval", "resume", decision.model_dump())
        status = "action_rejected" if decision.decision == "reject" else "approved"
        reason = None
        if decision.decision == "approve_modified":
            allowed = {
                (a.action_type, a.target_txn_id): a.amount_fen for a in p.actions
            }
            modified = decision.modified_actions or []
            if not modified or any(
                a.amount_fen > allowed.get((a.action_type, a.target_txn_id), 0)
                for a in modified
            ):
                reason = "修改动作超出原方案"
            else:
                _, issues = checker.hard_checks(
                    p.model_copy(update={"actions": modified}),
                    CaseFile(**s["case_file"]),
                    service,
                    Identity(**s["identity"]),
                )
                if issues:
                    reason = "、".join(i.code for i in issues)
            if reason:
                status = "action_rejected"
        trace(s, "await_approval", "node_end", {"status": status, "reason": reason})
        return {"approval": decision.model_dump(), "status": status, "failure": reason}

    def execute(s):
        proposal = Proposal(**s["proposals"][-1])
        decision = ApprovalDecision(**s["approval"])
        actions = (
            decision.modified_actions
            if decision.decision == "approve_modified"
            else proposal.actions
        )
        results = []
        for a in actions:
            index = next(
                i
                for i, orig in enumerate(proposal.actions)
                if orig.target_txn_id == a.target_txn_id
            )
            action = ApprovedAction(
                **a.model_dump(),
                proposal_id=proposal.proposal_id,
                action_index=index,
                customer_id=s["identity"]["actor_id"],
            )
            results.append(executor.execute(action).model_dump())
        return {
            "action_results": results,
            "status": "action_executed"
            if all(r["status"] == "succeeded" for r in results)
            else "action_rejected",
        }

    def finalize(s):
        status = s["status"]
        cf = s.get("case_file", {})
        data = (
            BillingFacts(**s["facts"])
            if s.get("facts") and status != "forbidden"
            else None
        )
        msg = {
            "forbidden": "不可访问",
            "escalated": "已转交人工进一步核查。",
            "clarify": cf.get("clarification_question"),
            "action_rejected": "本次资金方案未全部执行，请查看审批结果。",
            "action_executed": "审批已通过，资金处理已完成，可查看钱包流水。",
            "pending_evidence": "需要等待充值回调或补充结束读数后再核查。",
        }.get(status)
        if not msg and s["proposals"]:
            msg = s["proposals"][-1]["customer_message"]
        chain = []
        if data:
            for a in data.attempts:
                os = [o for o in data.orders if o.order.attempt_id == a.attempt_id]
                ts = service.get_transactions(
                    Identity(**s["identity"]), attempt_id=a.attempt_id
                )
                chain.append(
                    dict(
                        attempt_id=a.attempt_id,
                        session_id=a.session_id,
                        order_id=os[0].order.order_id if os else None,
                        txn_ids=[t.txn_id for t in ts],
                    )
                )
        report = dict(
            case_id=s["case_id"],
            run_id=s["run_id"],
            planner=s["planner"],
            status=status,
            dispute_type=cf.get("dispute_type"),
            chain=chain,
            candidates=cf.get("candidates", []) if status == "clarify" else [],
            facts_summary={
                "anomalies": [a.model_dump() for a in data.anomalies],
                "missing_evidence": data.missing_evidence,
            }
            if data
            else {},
            proposals=s["proposals"],
            reviews=s["reviews"],
            approval=s.get("approval"),
            action_results=s.get("action_results", []),
            failure=s.get("failure"),
            customer_message=msg,
            simulated_data=True,
        )
        store.finish_run(s["run_id"], status, report)
        if (
            memory_conf["enable_long_term_memory"]
            and memory_conf["auto_record_case_summary"]
            and status in {"explained", "action_executed", "action_rejected", "escalated"}
        ):
            try:
                CustomerMemoryService(store.engine).upsert_memory(
                    Identity(**s["identity"]),
                    memory_key=f"case:{s['case_id']}",
                    memory_value=f"{report['dispute_type'] or 'billing'} | 状态:{status}",
                    source="case_auto",
                    ttl_days=memory_conf["case_summary_retention_days"],
                    reason="billing_case_finalized",
                )
            except Exception as error:
                logger.warning(f"[memory] 案件摘要写入失败：{type(error).__name__}")
        logger.info(
            "billing case_id=%s run_id=%s status=%s", s["case_id"], s["run_id"], status
        )
        return {"report": report}

    def guarded(name, fn):
        def invoke(s):
            try:
                return fn(s)
            except Exception as e:
                if s["planner"] == "offline":
                    raise
                trace(s, name, "error", {"reason": str(e)})
                return {"status": "escalated", "failure": str(e)}

        return invoke

    for name, fn in [
        ("intake", locate),
        ("facts", gather),
        ("maker", draft),
        ("checker", review),
        ("pending", pending),
        ("execute", execute),
        ("finalize", finalize),
    ]:
        graph.add_node(
            name,
            wrap(
                name,
                guarded(name, fn) if name in ("intake", "maker", "checker") else fn,
            ),
        )
    graph.add_node("await_approval", await_approval)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake", lambda s: "facts" if s["status"] == "located" else "finalize"
    )
    graph.add_conditional_edges(
        "facts", lambda s: "maker" if s["status"] == "located" else "finalize"
    )
    graph.add_conditional_edges(
        "maker", lambda s: "checker" if s["status"] == "proposed" else "finalize"
    )
    graph.add_conditional_edges(
        "checker",
        lambda s: {"revise": "maker", "action_pending_approval": "pending"}.get(
            s["status"], "finalize"
        ),
    )
    graph.add_edge("pending", "await_approval")
    graph.add_conditional_edges(
        "await_approval",
        lambda s: "execute" if s["status"] == "approved" else "finalize",
    )
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
