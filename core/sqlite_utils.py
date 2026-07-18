"""подключение к sqlite."""

import os
import sqlite3
from contextlib import contextmanager

from core.config import CHATS_DB_PATH


@contextmanager
def connect_db(db_path: str = CHATS_DB_PATH, timeout: int = 10):
    """открыть SQLITE и сохранить или откатить изменения."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
