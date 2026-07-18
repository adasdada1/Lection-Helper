"""rag-поиск по лекциям с памятью чатов."""

import logging
from core.embeddings import get_embedder
from core.vector_store import search
from core.lexical_store import search as lexical_search
from core.llm import call_llm
from core import chat_memory
from core.config import (
    DENSE_TOP_K_SUMMARY,
    DENSE_TOP_K_TRANSCRIPT,
    LEXICAL_TOP_K_SUMMARY,
    LEXICAL_TOP_K_TRANSCRIPT,
    RERANK_TOP_N_SUMMARY,
    RERANK_TOP_N_TRANSCRIPT,
    RETRIEVAL_TOP_K,
    RRF_K,
)

logger = logging.getLogger(__name__)

async def ask(question: str, chat_id: str, top_k: int = RETRIEVAL_TOP_K) -> dict:
    """ответить на вопрос в указанном чате."""
    # 1. вектор вопроса
    embedder = get_embedder()
    query_embedding = embedder.embed_query(question)

    # 2. фрагменты лекций
    matches = _retrieve_candidates(question, query_embedding, limit=top_k)
    from core.reranker import rerank_balanced
    matches = rerank_balanced(
        question,
        matches,
        top_n_summary=RERANK_TOP_N_SUMMARY,
        top_n_transcript=RERANK_TOP_N_TRANSCRIPT,
    )
    # 3. промпт с памятью чата
    messages, sources = _build_messages(question, matches, chat_id)

    # 4. сохранить вопрос
    chat_memory.add_message(chat_id, "user", question)

    # 5. получить ответ
    result = call_llm(messages)

    # 6. сохранить ответ
    chat_memory.add_message(
        chat_id,
        "assistant",
        result["answer"],
        model=result.get("model"),
        sources=sources,
    )

    # 7. при необходимости сжать историю
    chat_memory.maybe_summarize(chat_id)

    # 8. создать заголовок
    msg_count = chat_memory.get_message_count(chat_id)
    chat_data = chat_memory.get_chat(chat_id)
    if msg_count == 2 and chat_data and chat_data["title"] == chat_memory.DEFAULT_CHAT_TITLE:
        try:
            chat_memory.generate_chat_title(chat_id)
        except Exception as e:
            logger.warning("не удалось создать заголовок автоматически: %s", e)

    return {
        "answer": result["answer"],
        "model": result["model"],
        "chat_id": chat_id,
        "sources": sources,
    }


def _match_key(match: dict) -> str:
    if match.get("id"):
        return str(match["id"])
    meta = match.get("metadata", {})
    return f"{meta.get('doc_id')}:{meta.get('chunk_index')}:{meta.get('content_kind')}"


def _rrf_merge(result_lists: list[list[dict]], limit: int) -> list[dict]:
    """объединить результаты поиска через rrf."""
    fused: dict[str, dict] = {}

    for matches in result_lists:
        for rank, match in enumerate(matches, start=1):
            key = _match_key(match)
            score = 1.0 / (RRF_K + rank)
            if key not in fused:
                fused[key] = dict(match)
                fused[key]["rrf_score"] = 0.0
                fused[key]["retrieval_hits"] = 0
            fused[key]["rrf_score"] += score
            fused[key]["retrieval_hits"] += 1

    return sorted(
        fused.values(),
        key=lambda item: item.get("rrf_score", 0.0),
        reverse=True,
    )[:limit]


def _retrieve_candidates(question: str, query_embedding: list[float], limit: int) -> list[dict]:
    """найти фрагменты конспекта и транскрипта."""
    dense_summary = search(
        query_embedding,
        top_k=DENSE_TOP_K_SUMMARY,
        content_kind="summary",
    )
    dense_transcript = search(
        query_embedding,
        top_k=DENSE_TOP_K_TRANSCRIPT,
        content_kind="transcript",
    )
    lexical_summary = lexical_search(
        question,
        top_k=LEXICAL_TOP_K_SUMMARY,
        content_kind="summary",
    )
    lexical_transcript = lexical_search(
        question,
        top_k=LEXICAL_TOP_K_TRANSCRIPT,
        content_kind="transcript",
    )
    return _rrf_merge(
        [dense_summary, dense_transcript, lexical_summary, lexical_transcript],
        limit=limit,
    )


def _build_messages(
    question: str,
    matches: list[dict],
    chat_id: str,
) -> tuple[list[dict], list[dict]]:
    """собрать сообщения для llm."""
    sources = []

    # системный промпт
    if matches:
        system_prompt = (
            "Ты - полезный ассистент по лекционным материалам. "
            "Тебе предоставлены фрагменты из лекций и история диалога. "
            "Отвечай на вопросы студента строго на основе лекционных фрагментов. "
            "Используй историю диалога для понимания контекста вопроса, "
            "но приоритет отдавай фактам из лекций. "
            "Если информации недостаточно, скажи об этом честно.\n"
            "ВАЖНОЕ ПРАВИЛО: При написании математических формул используй ТОЛЬКО стандартный синтаксис LaTeX "
            "($...$ для строчных формул и $$...$$ для выключных). Категорически запрещено дублировать "
            "одну и ту же формулу в виде LaTeX и в виде обычного текста/юникода.\n"
            "Отвечай ясно, структурированно и на русском языке."
        )
    else:
        system_prompt = (
            "Ты - полезный ассистент. "
            "В базе знаний пока нет загруженных лекций. "
            "Постарайся ответить на вопрос общими знаниями, "
            "и предложи загрузить лекционные материалы для более точных ответов."
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    # сводка этого чата
    summary = chat_memory.get_summary(chat_id)
    if summary:
        messages.append({
            "role": "system",
            "content": (
                "Краткое резюме предыдущей части диалога:\n"
                f"{summary}"
            ),
        })

    # последние сообщения этого чата
    recent = chat_memory.get_recent_messages(chat_id)
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # вопрос с rag-контекстом
    if matches:
        context_parts = []
        for i, match in enumerate(matches):
            meta = match["metadata"]
            kind = meta.get("content_kind", "transcript")
            kind_label = "Конспект" if kind == "summary" else "Транскрипт"
            label = (
                f"Фрагмент {i + 1} [{kind_label}] "
                f"(из \"{meta.get('filename', 'материал')}\", чанк #{meta.get('chunk_index')})"
            )
            context_parts.append(f"--- {label} ---\n{match['text']}")
            sources.append({
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename", "Без названия"),
                "source_type": meta.get("source_type", "material"),
                "content_kind": kind,
                "content_kind_label": kind_label,
                "chunk_index": meta.get("chunk_index"),
                "text_preview": match["text"][:220] + "…" if len(match["text"]) > 220 else match["text"],
                "distance": round(match["distance"], 4),
                "rerank_score": round(match.get("rerank_score", 0), 4),
            })

        context_block = "\n\n".join(context_parts)
        user_content = (
            f"Контекст из лекций:\n\n{context_block}\n\n"
            f"Вопрос студента: {question}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})

    return messages, sources
