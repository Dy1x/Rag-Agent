import sqlite3
from functools import cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from utils.config_handler import billing_conf
from utils.path_tool import get_abs_path


@cache
def get_case_checkpointer(path: str | None = None) -> SqliteSaver:
    file = Path(path or get_abs_path(billing_conf["checkpoint_sqlite_path"]))
    file.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(file, check_same_thread=False, timeout=30))
    saver.setup()
    return saver
