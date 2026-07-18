"""память чатов в sqlite."""

import os
import uuid
import logging
import json
from datetime import datetime

from core.config import (
    CHAT_RECENT_WINDOW as RECENT_WINDOW,
    CHAT_SUMMARIZE_THRESHOLD as SUMMARIZE_THRESHOLD,
    DEFAULT_CHAT_TITLE,
)
from core.sqlite_utils import connect_db

logger = logging.getLogger(__name__)

# работа с sqlite ---------------------------------------------------------------------------

def _init_db() -> None:
    """создать таблицы чатов и сообщений."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id    TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'Новый чат',
                summary    TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                model      TEXT,
                sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
                ON messages(chat_id);
        """)

        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(messages)")
        }
        if "model" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        if "sources_json" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN sources_json TEXT")


def _connect():
    """открыть sqlite."""
    return connect_db()


# операции с чатами ---------------------------------------------------------------------------

def create_chat(title: str | None = None) -> dict:
    """создать чат."""
    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    t = title or DEFAULT_CHAT_TITLE

    with _connect() as conn:
        conn.execute(
            "INSERT INTO chats (chat_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (chat_id, t, now, now),
        )

    logger.info("создали чат %s (%s)", chat_id, t)
    return {"chat_id": chat_id, "title": t, "created_at": now, "updated_at": now}


def list_chats() -> list[dict]:
    """получить список чатов."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT chat_id, title, created_at, updated_at "
            "FROM chats ORDER BY updated_at DESC"
        ).fetchall()

    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    """получить чат по id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT chat_id, title, summary, created_at, updated_at "
            "FROM chats WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()

    return dict(row) if row else None


def rename_chat(chat_id: str, new_title: str) -> bool:
    """переименовать чат."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?",
            (new_title, now, chat_id),
        )
    return cur.rowcount > 0


def delete_chat(chat_id: str) -> bool:
    """удалить чат с сообщениями."""
    with _connect() as conn:
        # сообщения удалятся вместе с чатом
        cur = conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    logger.info("удалили чат %s, существовал=%s", chat_id, cur.rowcount > 0)
    return cur.rowcount > 0


# сообщения ---------------------------------------------------------------------------

def add_message(
    chat_id: str,
    role: str,
    content: str,
    *,
    model: str | None = None,
    sources: list[dict] | None = None,
) -> None:
    """добавить сообщение."""
    now = datetime.now().isoformat()
    sources_json = (
        json.dumps(sources, ensure_ascii=False) if sources is not None else None
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages "
            "(chat_id, role, content, model, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, model, sources_json, now),
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
            (now, chat_id),
        )


def get_messages(chat_id: str) -> list[dict]:
    """получить сообщения чата."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, model, sources_json, created_at FROM messages "
            "WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()

    messages = []
    for row in rows:
        message = dict(row)
        raw_sources = message.pop("sources_json", None)
        try:
            sources = json.loads(raw_sources) if raw_sources else []
        except json.JSONDecodeError:
            logger.warning("не удалось прочитать sources_json сообщения")
            sources = []
        message["sources"] = sources if isinstance(sources, list) else []
        messages.append(message)
    return messages


def get_message_count(chat_id: str) -> int:
    """посчитать сообщения чата."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
    return row["cnt"] if row else 0


def get_recent_messages(chat_id: str, limit: int = RECENT_WINDOW) -> list[dict]:
    """получить последние сообщения чата."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()

    # вернуть старые сообщения первыми
    return [dict(r) for r in reversed(rows)]


def get_summary(chat_id: str) -> str | None:
    """получить сводку чата."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT summary FROM chats WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
    return row["summary"] if row else None


def set_summary(chat_id: str, summary: str) -> None:
    """сохранить сводку чата."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE chats SET summary = ?, updated_at = ? WHERE chat_id = ?",
            (summary, now, chat_id),
        )


def set_title(chat_id: str, title: str) -> None:
    """сохранить заголовок чата."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?",
            (title, now, chat_id),
        )


# сжатие истории ---------------------------------------------------------------------------

def _load_summarization_prompt() -> str:
    """загрузить промпт для сжатия истории."""
    path = os.path.join("prompts", "context_summarization_prompt.txt")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Файл промпта для суммаризации контекста не найден: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise RuntimeError(f"Файл промпта пуст: {path}")
    return content


def _format_messages_for_summary(
    messages: list[dict], existing_summary: str | None = None,
) -> str:
    """собрать сообщения в один текст."""
    parts = []
    if existing_summary:
        parts.append(f"Предыдущее резюме диалога:\n{existing_summary}\n")
        parts.append("Новые сообщения:\n")

    for msg in messages:
        role_label = "Студент" if msg["role"] == "user" else "Ассистент"
        content = msg["content"]
        if len(content) > 1000:
            content = content[:1000] + "…"
        parts.append(f"{role_label}: {content}")

    return "\n".join(parts)


def maybe_summarize(chat_id: str) -> None:
    """сжать старые сообщения, если история стала длинной."""
    msg_count = get_message_count(chat_id)
    if msg_count <= SUMMARIZE_THRESHOLD:
        return

    all_messages = get_messages(chat_id)
    if len(all_messages) <= RECENT_WINDOW:
        return

    messages_to_summarize = all_messages[:-RECENT_WINDOW]
    existing_summary = get_summary(chat_id)

    try:
        system_prompt = _load_summarization_prompt()
    except RuntimeError as e:
        logger.warning("сжатие истории пропущено: %s", e)
        return

    user_content = _format_messages_for_summary(
        messages_to_summarize, existing_summary,
    )

    from core.llm import call_llm_for_summarization

    try:
        logger.info(
            "сжимаем %d старых сообщений чата %s",
            len(messages_to_summarize), chat_id,
        )
        result = call_llm_for_summarization([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])
        set_summary(chat_id, result["answer"])

        # удалить уже сжатые сообщения
        with _connect() as conn:
            # оставить последние recent_window сообщений
            conn.execute(
                "DELETE FROM messages WHERE chat_id = ? AND id NOT IN "
                "(SELECT id FROM messages WHERE chat_id = ? "
                "ORDER BY id DESC LIMIT ?)",
                (chat_id, chat_id, RECENT_WINDOW),
            )

        logger.info("сжали историю чата %s", chat_id)
    except Exception as e:
        logger.error("не удалось сжать историю чата %s: %s", chat_id, e)


# заголовок чата ---------------------------------------------------------------------------

def generate_chat_title(chat_id: str) -> str | None:
    """создать заголовок по первым сообщениям."""
    messages = get_messages(chat_id)
    if not messages:
        return None

    # взять первый вопрос и ответ
    first_user = None
    first_assistant = None
    for msg in messages:
        if msg["role"] == "user" and first_user is None:
            first_user = msg["content"]
        elif msg["role"] == "assistant" and first_assistant is None:
            first_assistant = msg["content"]
        if first_user and first_assistant:
            break

    if not first_user:
        return None

    # собрать промпт для заголовка
    context = f"Вопрос пользователя: {first_user[:500]}"
    if first_assistant:
        context += f"\n\nОтвет ассистента: {first_assistant[:500]}"

    from core.llm import call_llm_for_title

    try:
        result = call_llm_for_title([
            {
                "role": "system",
                "content": (
                    "Сгенерируй очень короткое название для этого диалога (2–6 слов). "
                    "Название должно отражать суть вопроса. "
                    "Отвечай только названием, без кавычек и пояснений. "
                    "Язык названия должен совпадать с языком диалога."
                ),
            },
            {"role": "user", "content": context},
        ])
        title = result["answer"].strip().strip('"\'')
        # обрезать длинный заголовок
        if len(title) > 60:
            title = title[:57] + "…"
        if title:
            set_title(chat_id, title)
            logger.info("создали заголовок чата %s: %s", chat_id, title)
            return title
    except Exception as e:
        logger.warning("не удалось создать заголовок чата %s: %s", chat_id, e)

    # взять начало вопроса, если модель не ответила
    fallback = " ".join(first_user.split()[:5])
    if len(fallback) > 50:
        fallback = fallback[:47] + "…"
    set_title(chat_id, fallback)
    return fallback


# запуск sqlite ---------------------------------------------------------------------------
_init_db()
