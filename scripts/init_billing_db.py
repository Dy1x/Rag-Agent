import argparse

from service.billing.db import (
    apply_schema,
    create_database,
    get_billing_engine,
    reset_business_data,
)
from utils.config_handler import billing_conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=billing_conf["mysql_database"])
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    create_database(args.database)
    engine = get_billing_engine(args.database)
    apply_schema(engine)
    if args.reset:
        reset_business_data(engine)


if __name__ == "__main__":
    main()
