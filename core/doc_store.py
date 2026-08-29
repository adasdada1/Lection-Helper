"""документы и конспекты в sqlite. векторы лежат в CHROMADB."""

import logging
from datetime import datetime

from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)

def _init_db() -> None:
    """создать таблицу документов."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                source_type     TEXT NOT NULL,
                transcript      TEXT,
                full_summary    TEXT,
                display_summary TEXT,
                chunk_count_transcript INTEGER DEFAULT 0,
                chunk_count_summary    INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL
            );
        """)

        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(documents)")
        }
        if "course_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN course_id TEXT")
            logger.info("добавили колонку course_id в documents")
        if "checklist" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN checklist TEXT")
            logger.info("добавили колонку checklist в documents")
        if "media_path" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN media_path TEXT")
            logger.info("добавили колонку media_path в documents")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_course_id "
            "ON documents(course_id)"
        )

def _connect():
    """открыть sqlite."""
    return connect_db()


def add_document(
    doc_id: str,
    filename: str,
    source_type: str,
    transcript: str | None,
    full_summary: str | None,
    display_summary: str | None,
    chunk_count_transcript: int = 0,
    chunk_count_summary: int = 0,
    course_id: str | None = None,
    checklist: str | None = None,
    media_path: str | None = None,
) -> None:
    """сохранить документ."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, filename, source_type, transcript, full_summary, display_summary,
                chunk_count_transcript, chunk_count_summary, created_at, course_id,
                checklist, media_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                filename,
                source_type,
                transcript,
                full_summary,
                display_summary,
                chunk_count_transcript,
                chunk_count_summary,
                now,
                course_id,
                checklist,
                media_path,
            ),
        )
    logger.info("сохранили данные документа в SQLITE, doc_id=%s", doc_id)


def get_document(doc_id: str) -> dict | None:
    """получить документ."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents(course_id: str | None = None) -> list[dict]:
    """получить документы без больших текстов."""
    sql = (
        "SELECT doc_id, filename, source_type, chunk_count_transcript, "
        "chunk_count_summary, created_at, course_id, media_path FROM documents"
    )
    params: list = []
    if course_id:
        sql += " WHERE course_id = ?"
        params.append(course_id)
    sql += " ORDER BY created_at DESC"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["chunk_count"] = d["chunk_count_transcript"] + d["chunk_count_summary"]
        result.append(d)
    return result


def rename_document(doc_id: str, title: str) -> bool:
    """переименовать материал во всех хранилищах."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE documents SET filename = ? WHERE doc_id = ?", (title, doc_id),
        )
    if cur.rowcount == 0:
        return False

    from core.lexical_store import rename_document as rename_lexical
    from core.vector_store import rename_document as rename_vectors

    try:
        rename_lexical(doc_id, title)
    except Exception as e:
        logger.warning("не переименовали материал в fts5: %s", e)
    try:
        rename_vectors(doc_id, title)
    except Exception as e:
        logger.warning("не переименовали материал в chromadb: %s", e)

    logger.info("переименовали материал %s в '%s'", doc_id, title)
    return True


def get_media_path(doc_id: str) -> str | None:
    """получить путь к исходному файлу материала."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT media_path FROM documents WHERE doc_id = ?", (doc_id,),
        ).fetchone()
    return row["media_path"] if row else None


def delete_document(doc_id: str) -> bool:
    """удалить документ."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    logger.info("удалили документ %s из sqlite, существовал=%s", doc_id, cur.rowcount > 0)
    return cur.rowcount > 0


_init_db()
