from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from utils.config_handler import db_conf
from sqlalchemy import text


def build_mysql_url() -> str:
    return (
        f"mysql+pymysql://{db_conf['mysql_user']}:{db_conf['mysql_password']}"
        f"@{db_conf['mysql_host']}:{db_conf['mysql_port']}"
        f"/{db_conf['mysql_database']}?charset={db_conf.get('mysql_charset', 'utf8mb4')}"
    )


engine = create_engine(
    build_mysql_url(),
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine)


def fetch_one(sql: str, params: dict | None = None):
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        row = result.mappings().first()
        return dict(row) if row else None


def fetch_all(sql: str, params: dict | None = None):
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


def execute(sql: str, params: dict | None = None):
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        return result.rowcount