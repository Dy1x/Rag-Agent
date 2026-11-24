import argparse
from pathlib import Path

from service.billing.case_store import CaseStore
from service.billing.db import get_billing_engine
from service.billing.trace_export import encode_otlp


def main():
    parser = argparse.ArgumentParser(description="导出案件运行的 OTLP Protobuf 轨迹")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database")
    args = parser.parse_args()
    store = CaseStore(get_billing_engine(args.database))
    data = encode_otlp(*store.get_trace_snapshot(args.run_id))
    with args.output.open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    main()
