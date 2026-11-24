import re
from functools import cache
from pathlib import Path

from sqlalchemy import URL, Engine, create_engine, text

from utils.config_handler import billing_conf, db_conf
from .errors import BillingError

POLICY = "POLICY-STD-01"
POLICY_TEXT = (
    "按开始时段为整单计价。每千瓦时电费/服务费（分）：谷段23:00至07:00为40/40，"
    "平段07:00至10:00、15:00至18:00、21:00至23:00为80/60，"
    "峰段10:00至15:00、18:00至21:00为120/80。停止充电后15分钟免费，"
    "超过宽限按每分钟50分计收，不足一分钟向上取整，封顶3000分。"
    "启动冻结5000分，结束后从钱包结算并解除冻结，退款退回钱包余额。"
)


def build_billing_url(database: str | None = None, *, server_only: bool = False) -> str:
    url = URL.create(
        "mysql+pymysql",
        username=db_conf["mysql_user"],
        password=str(db_conf["mysql_password"]),
        host=db_conf["mysql_host"],
        port=int(db_conf["mysql_port"]),
        database=None if server_only else database or billing_conf["mysql_database"],
        query={"charset": db_conf.get("mysql_charset", "utf8mb4")},
    )
    return url.render_as_string(hide_password=False)


@cache
def get_billing_engine(database: str | None = None) -> Engine:
    return create_engine(
        build_billing_url(database),
        pool_pre_ping=True,
        pool_recycle=3600,
        isolation_level="READ COMMITTED",
    )


def _database_ddl(name: str, statement: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
        raise BillingError("VALIDATION", "非法数据库名称")
    engine = create_engine(build_billing_url(server_only=True))
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(statement.format(name=name))
    finally:
        engine.dispose()


def create_database(name: str) -> None:
    _database_ddl(name, "CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4")


def drop_database(name: str) -> None:
    if not name.startswith(("cb_test_", "cb_eval_")):
        raise BillingError("VALIDATION", "只允许删除测试或评测库")
    get_billing_engine(name).dispose()
    _database_ddl(name, "DROP DATABASE IF EXISTS `{name}`")


def _schema() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _seed(conn) -> None:
    conn.execute(
        text("""INSERT INTO cb_tariff_policies VALUES
        (:p,'标准充电政策','2020-01-01',NULL,15,50,3000,5000,:policy_text)
        ON DUPLICATE KEY UPDATE policy_id=policy_id"""),
        {"p": POLICY, "policy_text": POLICY_TEXT},
    )
    periods = [
        (23, 7, "valley", 40, 40),
        (7, 10, "flat", 80, 60),
        (10, 15, "peak", 120, 80),
        (15, 18, "flat", 80, 60),
        (18, 21, "peak", 120, 80),
        (21, 23, "flat", 80, 60),
    ]
    for start, end, kind, electricity, service in periods:
        conn.execute(
            text("""INSERT INTO cb_tariff_periods VALUES (:p,:start,:end,:kind,:e,:s)
            ON DUPLICATE KEY UPDATE policy_id=policy_id"""),
            dict(
                p=POLICY,
                start=f"{start:02}:00",
                end=f"{end:02}:00",
                kind=kind,
                e=electricity,
                s=service,
            ),
        )
    for charger, connector in [("CP-A01", "1"), ("CP-A01", "2"), ("CP-A02", "1")]:
        conn.execute(
            text("""INSERT INTO cb_chargers VALUES (:charger,:connector,'科技园 A 站',:p)
            ON DUPLICATE KEY UPDATE charger_id=charger_id"""),
            dict(charger=charger, connector=connector, p=POLICY),
        )
    for nickname in ["ALICE", "BOB"]:
        values = dict(customer=f"CUS-DEMO-{nickname}", nickname=nickname)
        conn.execute(
            text("""INSERT INTO cb_customers VALUES (:customer,:nickname,UTC_TIMESTAMP(3))
            ON DUPLICATE KEY UPDATE customer_id=customer_id"""),
            values,
        )
        conn.execute(
            text("""INSERT INTO cb_wallets (customer_id,updated_at) VALUES (:customer,UTC_TIMESTAMP(3))
            ON DUPLICATE KEY UPDATE customer_id=customer_id"""),
            values,
        )


def apply_schema(engine: Engine) -> None:
    statements = re.sub(r"--[^\n]*", "", _schema()).split(";")
    with engine.begin() as conn:
        for statement in statements:
            if statement.strip():
                conn.exec_driver_sql(statement)
        _seed(conn)


def reset_business_data(engine: Engine) -> None:
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS (cb_[a-z_]+)", _schema())
    with engine.begin() as conn:
        for table in reversed(tables):
            conn.exec_driver_sql(f"DELETE FROM `{table}`")
        _seed(conn)
