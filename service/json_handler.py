import json
from datetime import date, datetime
from decimal import Decimal


def to_json_str(data) -> str:
    def default(o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return str(o)

    return json.dumps(data, ensure_ascii=False, default=default)
