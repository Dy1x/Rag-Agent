from agent.cases.agent_spec import AgentSpec
from service.billing.models import Proposal
from utils.config_handler import billing_conf

SPEC = AgentSpec(
    "maker",
    "billing_maker.txt",
    ["get_policy", "search_billing_faq", "recompute_order", "compute_refundable"],
    Proposal,
    billing_conf["maker_max_tool_calls"],
)
