from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def as_utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC).astimezone(UTC).replace(tzinfo=None)
        if value.tzinfo is None
        else value.astimezone(UTC).replace(tzinfo=None)
    )


def local_to_utc(value: datetime, tz: str) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(tz))
    return as_utc(value)


def to_local(value: datetime, tz: str) -> datetime:
    return as_utc(value).replace(tzinfo=UTC).astimezone(ZoneInfo(tz))
