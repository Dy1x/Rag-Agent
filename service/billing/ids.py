import re
from secrets import token_hex


def new_id(prefix: str) -> str:
    return f"{prefix}-{token_hex(6).upper()}"


def is_record_suffix(value: str) -> bool:
    return re.fullmatch(r"[0-9A-Za-z]{4}", value) is not None
