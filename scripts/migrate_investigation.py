import argparse
import json
from pathlib import Path

from service.billing.db import create_database, get_billing_engine, apply_schema
from service.investigation_mysql import (
    MySQLInvestigationSource,
    apply_investigation_schema,
    import_sqlite,
)
from utils.config_handler import billing_conf


def main():
    parser = argparse.ArgumentParser(description="将 SQLite 调查证据按数据集导入 MySQL")
    parser.add_argument("--from-sqlite", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--database", default=billing_conf["mysql_database"])
    args = parser.parse_args()
    if not args.from_sqlite.is_file():
        parser.error("源 SQLite 文件不存在")
    create_database(args.database)
    engine = get_billing_engine(args.database)
    apply_schema(engine)
    apply_investigation_schema(engine)
    print(
        json.dumps(
            import_sqlite(
                args.from_sqlite, MySQLInvestigationSource(engine, args.dataset)
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
