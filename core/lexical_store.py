"""поиск точных слов через sqlite fts5."""

import re
import sqlite3
import logging

from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)

_FTS_AVAILABLE: bool | None = None

_FTS_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS lecture_chunks_fts
    USING fts5(
        chunk_id UNINDEXED,
        doc_id UNINDEXED,
        course_id UNINDEXED,
        filename UNINDEXED,
        source_type UNINDEXED,
        content_kind UNINDEXED,
        chunk_index UNINDEXED,
        text,
        tokenize='unicode61'
    )
"""

_FTS_COLUMNS = (
    "chunk_id, doc_id, course_id, filename, source_type, content_kind, chunk_index, text"
)


def _connect():
    """ОТКРЫТЬ SQLITE."""
    return connect_db()


def _migrate_add_course_id(conn) -> None:
    """пересоздать fts5 с колонкой курса. fts5 не умеет alter table."""
    rows = conn.execute(
        "SELECT chunk_id, doc_id, filename, source_type, content_kind, chunk_index, text "
        "FROM lecture_chunks_fts"
    ).fetchall()
    saved = [
        (r["chunk_id"], r["doc_id"], None, r["filename"], r["source_type"],
         r["content_kind"], r["chunk_index"], r["text"])
        for r in rows
    ]

    conn.execute("DROP TABLE lecture_chunks_fts")
    conn.execute(_FTS_SCHEMA)
    if saved:
        conn.executemany(
            f"INSERT INTO lecture_chunks_fts ({_FTS_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            saved,
        )
    logger.info("пересоздали fts5 с course_id, перенесли фрагментов: %d", len(saved))


def _init_db() -> bool:
    global _FTS_AVAILABLE
    if _FTS_AVAILABLE is not None:
        return _FTS_AVAILABLE

    try:
        with _connect() as conn:
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(lecture_chunks_fts)")
            }
            if existing and "course_id" not in existing:
                _migrate_add_course_id(conn)
            conn.execute(_FTS_SCHEMA)
        _FTS_AVAILABLE = True
    except sqlite3.OperationalError as exc:
        logger.warning("sqlite fts5 недоступен: %s", exc)
        _FTS_AVAILABLE = False
    return _FTS_AVAILABLE


def backfill_course_id() -> int:
    """проставить курс фрагментам по их документам."""
    if not _init_db():
        return 0

    with _connect() as conn:
        pairs = conn.execute(
            "SELECT doc_id, course_id FROM documents WHERE course_id IS NOT NULL"
        ).fetchall()
        updated = 0
        for row in pairs:
            cur = conn.execute(
                "UPDATE lecture_chunks_fts SET course_id = ? "
                "WHERE doc_id = ? AND course_id IS NULL",
                (row["course_id"], row["doc_id"]),
            )
            updated += cur.rowcount

    logger.info("проставили course_id у %d фрагментов fts5", updated)
    return updated


def _tokenize_query(query: str) -> str | None:
    tokens = re.findall(r"[\wа-яА-ЯёЁ]+", query.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return None

    # искать по началу слова и не передавать сырой запрос
    return " OR ".join(f'"{token}"*' for token in tokens[:12])


def add_chunks(
    doc_id: str,
    chunks: list[dict],
    filename: str,
    source_type: str,
    course_id: str | None = None,
) -> int:
    """сохранить фрагменты в fts5."""
    if not _init_db():
        return 0

    rows = []

    for chunk in chunks:
        content_kind = chunk.get("content_kind", "transcript")

        rows.append((
            f"{doc_id}_chunk_{chunk['chunk_index']}",
            doc_id,
            course_id,
            filename,
            source_type,
            content_kind,
            chunk["chunk_index"],
            chunk["text"],
        ))

    with _connect() as conn:
        conn.execute("DELETE FROM lecture_chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            f"INSERT INTO lecture_chunks_fts ({_FTS_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    logger.info("сохранили в SQLITE FTS5 %d фрагментов, doc_id=%s", len(rows), doc_id)
    return len(rows)


def search(
    query: str,
    top_k: int = 5,
    content_kind: str | None = None,
    course_id: str | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """найти фрагменты по bm25."""
    if top_k <= 0 or not _init_db():
        return []

    fts_query = _tokenize_query(query)
    if not fts_query:
        return []

    sql = (
        "SELECT chunk_id, doc_id, course_id, filename, source_type, content_kind, "
        "chunk_index, text, bm25(lecture_chunks_fts) AS score "
        "FROM lecture_chunks_fts WHERE lecture_chunks_fts MATCH ?"
    )
    params: list = [fts_query]
    if content_kind:
        sql += " AND content_kind = ?"
        params.append(content_kind)
    if course_id:
        sql += " AND course_id = ?"
        params.append(course_id)
    if doc_id:
        sql += " AND doc_id = ?"
        params.append(doc_id)
    sql += " ORDER BY score LIMIT ?"
    params.append(top_k)

    try:
        with _connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("ошибка запроса fts5: %s", exc)
        return []

    matches = []
    for row in rows:
        matches.append({
            "id": row["chunk_id"],
            "text": row["text"],
            "metadata": {
                "doc_id": row["doc_id"],
                "course_id": row["course_id"],
                "filename": row["filename"],
                "source_type": row["source_type"],
                "content_kind": row["content_kind"],
                "chunk_index": row["chunk_index"],
            },
            "distance": float(row["score"]),
            "lexical_score": float(row["score"]),
        })
    return matches


def list_chunks(
    doc_id: str,
    content_kind: str = "transcript",
) -> list[dict]:
    if not _init_db():
        return []

    with _connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, course_id, filename, source_type, "
            "content_kind, chunk_index, text FROM lecture_chunks_fts "
            "WHERE doc_id = ? AND content_kind = ? "
            "ORDER BY CAST(chunk_index AS INTEGER)",
            (doc_id, content_kind),
        ).fetchall()

    return [
        {
            "id": row["chunk_id"],
            "text": row["text"],
            "metadata": {
                "doc_id": row["doc_id"],
                "course_id": row["course_id"],
                "filename": row["filename"],
                "source_type": row["source_type"],
                "content_kind": row["content_kind"],
                "chunk_index": row["chunk_index"],
            },
            "distance": 0.0,
        }
        for row in rows
    ]


def rename_document(doc_id: str, title: str) -> int:
    """обновить название материала у его фрагментов."""
    if not _init_db():
        return 0
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE lecture_chunks_fts SET filename = ? WHERE doc_id = ?",
            (title, doc_id),
        )
    return cur.rowcount


def delete_document(doc_id: str) -> int:
    if not _init_db():
        return 0
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM lecture_chunks_fts WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()["c"]
        conn.execute("DELETE FROM lecture_chunks_fts WHERE doc_id = ?", (doc_id,))
    return count
