"""курсы в sqlite. курс - контейнер для материалов и чатов."""

import uuid
import logging
from datetime import datetime

from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)

DEFAULT_COURSE_TITLE = "Мои материалы"


def _connect():
    """открыть sqlite."""
    return connect_db()


def _init_db() -> None:
    """создать таблицу курсов."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id   TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT,
                created_at  TEXT NOT NULL
            );
        """)


def create_course(title: str, description: str | None = None) -> dict:
    """создать курс."""
    course_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO courses (course_id, title, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (course_id, title, description, now),
        )

    logger.info("создали курс %s (%s)", course_id, title)
    return {
        "course_id": course_id,
        "title": title,
        "description": description,
        "created_at": now,
        "document_count": 0,
        "chat_count": 0,
    }


def get_course(course_id: str) -> dict | None:
    """получить курс."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM courses WHERE course_id = ?", (course_id,)
        ).fetchone()
    return dict(row) if row else None


def list_courses() -> list[dict]:
    """получить курсы со счетчиками материалов и чатов."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT c.course_id, c.title, c.description, c.created_at,
                   (SELECT COUNT(*) FROM documents d WHERE d.course_id = c.course_id)
                       AS document_count,
                   (SELECT COUNT(*) FROM chats ch WHERE ch.course_id = c.course_id)
                       AS chat_count,
                   (SELECT d2.filename FROM documents d2 WHERE d2.course_id = c.course_id
                        ORDER BY d2.created_at DESC LIMIT 1)
                       AS last_document
            FROM courses c
            ORDER BY c.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def rename_course(course_id: str, title: str) -> bool:
    """переименовать курс."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE courses SET title = ? WHERE course_id = ?",
            (title, course_id),
        )
    return cur.rowcount > 0


def list_course_document_ids(course_id: str) -> list[str]:
    """получить идентификаторы материалов курса."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE course_id = ?", (course_id,)
        ).fetchall()
    return [r["doc_id"] for r in rows]


def list_course_chat_ids(course_id: str) -> list[str]:
    """получить идентификаторы чатов курса."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT chat_id FROM chats WHERE course_id = ?", (course_id,)
        ).fetchall()
    return [r["chat_id"] for r in rows]


def delete_course(course_id: str) -> bool:
    """удалить курс. материалы и чаты удаляются вызывающей стороной."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM courses WHERE course_id = ?", (course_id,))
    logger.info("удалили курс %s, существовал=%s", course_id, cur.rowcount > 0)
    return cur.rowcount > 0


def ensure_default_course() -> str | None:
    """перенести материалы и чаты без курса в курс по умолчанию."""
    with _connect() as conn:
        orphan_docs = conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE course_id IS NULL"
        ).fetchone()["c"]
        orphan_chats = conn.execute(
            "SELECT COUNT(*) AS c FROM chats WHERE course_id IS NULL"
        ).fetchone()["c"]

    if not orphan_docs and not orphan_chats:
        return None

    with _connect() as conn:
        row = conn.execute(
            "SELECT course_id FROM courses WHERE title = ? ORDER BY created_at LIMIT 1",
            (DEFAULT_COURSE_TITLE,),
        ).fetchone()

    course_id = row["course_id"] if row else create_course(DEFAULT_COURSE_TITLE)["course_id"]

    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET course_id = ? WHERE course_id IS NULL", (course_id,)
        )
        conn.execute(
            "UPDATE chats SET course_id = ? WHERE course_id IS NULL", (course_id,)
        )

    from core.lexical_store import backfill_course_id
    backfill_course_id()

    from core.vector_store import backfill_course_id as backfill_vectors
    backfill_vectors()

    logger.info(
        "перенесли в курс %s материалов: %d, чатов: %d",
        course_id, orphan_docs, orphan_chats,
    )
    return course_id


_init_db()
