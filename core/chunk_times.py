"""отметки времени фрагментов транскрипта в sqlite."""

import logging

from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)


def _connect():
    """открыть sqlite."""
    return connect_db()


def _init_db() -> None:
    """создать таблицу отметок времени."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunk_times (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                start_time REAL,
                end_time   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_times_doc_id
                ON chunk_times(doc_id);
        """)


def save_times(doc_id: str, chunks: list[dict]) -> int:
    """сохранить время фрагментов документа."""
    rows = [
        (
            f"{doc_id}_chunk_{chunk['chunk_index']}",
            doc_id,
            chunk.get("start_time"),
            chunk.get("end_time"),
        )
        for chunk in chunks
        if chunk.get("start_time") is not None
    ]

    with _connect() as conn:
        conn.execute("DELETE FROM chunk_times WHERE doc_id = ?", (doc_id,))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_times "
                "(chunk_id, doc_id, start_time, end_time) VALUES (?, ?, ?, ?)",
                rows,
            )

    logger.info("сохранили отметки времени для %d фрагментов, doc_id=%s", len(rows), doc_id)
    return len(rows)


def get_times(chunk_ids: list[str]) -> dict[str, dict]:
    """получить время фрагментов по их идентификаторам."""
    ids = [c for c in chunk_ids if c]
    if not ids:
        return {}

    placeholders = ", ".join("?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, start_time, end_time FROM chunk_times "
            f"WHERE chunk_id IN ({placeholders})",
            ids,
        ).fetchall()

    return {
        r["chunk_id"]: {"start_time": r["start_time"], "end_time": r["end_time"]}
        for r in rows
    }


def delete_document(doc_id: str) -> int:
    """удалить отметки времени документа."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM chunk_times WHERE doc_id = ?", (doc_id,))
    return cur.rowcount


def format_timecode(seconds: float | None) -> str | None:
    """перевести секунды в MM:SS или H:MM:SS."""
    if seconds is None:
        return None

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_range(start: float | None, end: float | None) -> str | None:
    """собрать подпись вида 12:34-15:02."""
    start_label = format_timecode(start)
    if start_label is None:
        return None

    end_label = format_timecode(end)
    if end_label is None or end_label == start_label:
        return start_label
    return f"{start_label}–{end_label}"


_init_db()
