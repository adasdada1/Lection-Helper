"""поиск точных слов через sqlite fts5."""

import re
import sqlite3
import logging

from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)

_FTS_AVAILABLE: bool | None = None


def _connect():
    """ОТКРЫТЬ SQLITE."""
    return connect_db()


def _init_db() -> bool:
    global _FTS_AVAILABLE
    if _FTS_AVAILABLE is not None:
        return _FTS_AVAILABLE

    try:
        with _connect() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS lecture_chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    filename UNINDEXED,
                    source_type UNINDEXED,
                    content_kind UNINDEXED,
                    chunk_index UNINDEXED,
                    text,
                    tokenize='unicode61'
                )
            """)
        _FTS_AVAILABLE = True
    except sqlite3.OperationalError as exc:
        logger.warning("sqlite fts5 недоступен: %s", exc)
        _FTS_AVAILABLE = False
    return _FTS_AVAILABLE


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
            filename,
            source_type,
            content_kind,
            chunk["chunk_index"],
            chunk["text"],
        ))

    with _connect() as conn:
        conn.execute("DELETE FROM lecture_chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            """INSERT INTO lecture_chunks_fts
               (chunk_id, doc_id, filename, source_type, content_kind, chunk_index, text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    logger.info("сохранили в SQLITE FTS5 %d фрагментов, doc_id=%s", len(rows), doc_id)
    return len(rows)


def search(query: str, top_k: int = 5, content_kind: str | None = None) -> list[dict]:
    """найти фрагменты по bm25."""
    if top_k <= 0 or not _init_db():
        return []

    fts_query = _tokenize_query(query)
    if not fts_query:
        return []

    sql = (
        "SELECT chunk_id, doc_id, filename, source_type, content_kind, chunk_index, "
        "text, bm25(lecture_chunks_fts) AS score "
        "FROM lecture_chunks_fts WHERE lecture_chunks_fts MATCH ?"
    )
    params: list = [fts_query]
    if content_kind:
        sql += " AND content_kind = ?"
        params.append(content_kind)
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
                "filename": row["filename"],
                "source_type": row["source_type"],
                "content_kind": row["content_kind"],
                "chunk_index": row["chunk_index"],
            },
            "distance": float(row["score"]),
            "lexical_score": float(row["score"]),
        })
    return matches


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
