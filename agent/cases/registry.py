from functools import cache
from .base import CaseHandler

_handlers: dict[str, CaseHandler] = {}


def register_handler(handler: CaseHandler) -> None:
    _handlers[handler.name] = handler


@cache
def _builtin(name):
    if name == "billing_dispute":
        from .billing.handler import BillingDisputeHandler
        from service.billing.db import get_billing_engine

        return BillingDisputeHandler(get_billing_engine())
    if name == "outage_investigation":
        from .outage.handler import OutageInvestigationHandler

        return OutageInvestigationHandler()
    raise KeyError(name)


def get_handler(name: str) -> CaseHandler:
    return _handlers[name] if name in _handlers else _builtin(name)


def additional_handlers() -> list[CaseHandler]:
    return [
        h
        for k, h in _handlers.items()
        if k not in ("billing_dispute", "outage_investigation")
    ]
