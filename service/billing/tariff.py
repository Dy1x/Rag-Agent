from datetime import datetime, time, timedelta

from pydantic import BaseModel, Field
from .errors import BillingError
from .timeutil import as_utc, to_local


class TariffPeriod(BaseModel):
    start_local: time
    end_local: time
    period_type: str
    electricity_fen_per_kwh: int
    service_fen_per_kwh: int


class TariffPolicy(BaseModel):
    periods: list[TariffPeriod]
    idle_grace_minutes: int
    idle_fee_fen_per_minute: int
    idle_fee_cap_fen: int | None


class AmountBreakdown(BaseModel):
    period_type: str
    started_at_local: datetime | None = None
    energy_wh: int | None = None
    electricity_fee_fen: int | None = None
    service_fee_fen: int | None = None
    idle_minutes: int
    idle_fee_fen: int
    total_fen: int | None = None
    missing: list[str] = Field(default_factory=list)


def find_period(policy: TariffPolicy, at_utc: datetime, tz: str) -> TariffPeriod:
    local = to_local(at_utc, tz).time()
    for period in policy.periods:
        start, end = period.start_local, period.end_local
        if start <= local < end or (end <= start and (local >= start or local < end)):
            return period
    raise BillingError("VALIDATION", "计费政策没有覆盖该时段")


def round_half_up_div(numerator: int, denominator: int) -> int:
    if denominator <= 0 or numerator < 0:
        raise BillingError("VALIDATION", "计费除法需要非负分子和正分母")
    return (2 * numerator + denominator) // (2 * denominator)


def compute_order_amount(
    policy: TariffPolicy,
    started_at: datetime,
    charge_end_at: datetime,
    unplugged_at: datetime | None,
    start_wh: int | None,
    end_wh: int | None,
    tz: str,
) -> AmountBreakdown:
    period = find_period(policy, started_at, tz)
    missing = []
    minutes = 0
    if unplugged_at is None:
        missing.append("unplugged_at")
    else:
        elapsed = as_utc(unplugged_at) - as_utc(charge_end_at)
        minutes = max(
            0, -(-elapsed // timedelta(minutes=1)) - policy.idle_grace_minutes
        )
    idle = minutes * policy.idle_fee_fen_per_minute
    if policy.idle_fee_cap_fen is not None:
        idle = min(idle, policy.idle_fee_cap_fen)
    result = AmountBreakdown(
        period_type=period.period_type,
        started_at_local=to_local(started_at, tz),
        idle_minutes=minutes,
        idle_fee_fen=idle,
    )
    result.missing = missing + [
        key for key, val in (("start_wh", start_wh), ("end_wh", end_wh)) if val is None
    ]
    if start_wh is None or end_wh is None:
        return result
    if not 0 <= start_wh <= end_wh:
        raise BillingError("VALIDATION", "电表读数须非负且单调递增")
    result.energy_wh = end_wh - start_wh
    result.electricity_fee_fen = round_half_up_div(
        result.energy_wh * period.electricity_fen_per_kwh, 1000
    )
    result.service_fee_fen = round_half_up_div(
        result.energy_wh * period.service_fen_per_kwh, 1000
    )
    result.total_fen = result.electricity_fee_fen + result.service_fee_fen + idle
    return result
