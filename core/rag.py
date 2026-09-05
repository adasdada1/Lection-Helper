"""rag-поиск по лекциям с памятью чатов."""

import logging
import os
import re
import json
from core.embeddings import get_embedder
from core.vector_store import search
from core.lexical_store import list_chunks, search as lexical_search
from core.llm import (
    call_llm_for_answer,
    call_llm_for_semantic_answer,
    call_llm_for_semantic_repair,
    call_llm_for_semantic_validation,
    call_llm_for_validation,
)
from core.model_runtime import LOCAL_MODEL_LOCK
from core.technical_terms import relevant_term_guidance
from core import chat_memory
from core.answer_grounding import (
    GROUNDING_FAILED_ANSWER,
    INSUFFICIENT_ANSWER,
    PROVIDER_FAILED_ANSWER,
    answer_json_schema,
    apply_semantic_validation,
    build_semantic_review_plan,
    claim_validation_json_schema,
    merge_semantic_review,
    validate_grounded_answer,
)
from core.semantic_grounding import (
    apply_section_validation,
    build_response_style,
    build_question_requirements,
    build_section_review_plan,
    combine_semantic_results,
    merge_section_review,
    response_language_matches,
    select_answer_reasoning,
    semantic_answer_json_schema,
    semantic_validation_json_schema,
    validate_semantic_answer,
)
from core.config import (
    DENSE_TOP_K_SUMMARY,
    DENSE_TOP_K_TRANSCRIPT,
    GROUNDING_MODE,
    LEXICAL_TOP_K_SUMMARY,
    LEXICAL_TOP_K_TRANSCRIPT,
    RERANK_TOP_N_SUMMARY,
    RERANK_TOP_N_TRANSCRIPT,
    RETRIEVAL_TOP_K,
    RRF_K,
    SHORT_LECTURE_MAX_CHARS,
    SHORT_LECTURE_MAX_CHUNKS,
)

logger = logging.getLogger(__name__)

async def ask(
    question: str,
    chat_id: str,
    top_k: int = RETRIEVAL_TOP_K,
    scope_doc_id: str | None = None,
    use_local_models: bool = True,
) -> dict:
    """ответить на вопрос в указанном чате."""
    deepseek_calls = []
    chat = chat_memory.get_chat(chat_id)
    course_id = chat.get("course_id") if chat else None

    matches = (
        _load_short_scoped_transcript(scope_doc_id)
        if scope_doc_id else None
    )
    if matches is not None:
        logger.info(
            "передаём все %d фрагментов короткой выбранной лекции",
            len(matches),
        )
    elif use_local_models:
        with LOCAL_MODEL_LOCK:
            embedder = get_embedder()
            query_embedding = embedder.embed_query(question)
            matches = _retrieve_candidates(
                question,
                query_embedding,
                limit=top_k,
                course_id=course_id,
                scope_doc_id=scope_doc_id,
            )
            from core.reranker import rerank_balanced
            matches = rerank_balanced(
                question,
                matches,
                top_n_summary=RERANK_TOP_N_SUMMARY,
                top_n_transcript=RERANK_TOP_N_TRANSCRIPT,
            )
    else:
        logger.info("локальные модели заняты, используем лексический поиск")
        matches = _retrieve_lexical_candidates(
            question,
            limit=top_k,
            course_id=course_id,
            scope_doc_id=scope_doc_id,
        )
    requirements = (
        build_question_requirements(question)
        if GROUNDING_MODE == "semantic" else []
    )
    messages, evidence_sources = _build_messages(
        question,
        matches,
        chat_id,
        semantic_mode=GROUNDING_MODE == "semantic",
        requirements=requirements,
    )

    # 4. сохранить вопрос
    chat_memory.add_message(chat_id, "user", question)

    # 5. получить ответ
    grounding_reason = None
    if evidence_sources:
        try:
            outcome = (
                _generate_semantic_answer(
                    messages,
                    evidence_sources,
                    question,
                    requirements,
                )
                if GROUNDING_MODE == "semantic"
                else _generate_strict_answer(
                    messages,
                    evidence_sources,
                    question,
                )
            )
            answer = outcome["answer"]
            sources = outcome["sources"]
            model = outcome["model"]
            grounding_status = outcome["grounding_status"]
            grounding_reason = outcome["grounding_reason"]
            deepseek_calls.extend(outcome["deepseek_calls"])
        except RuntimeError as exc:
            logger.error("генерация ответа недоступна: %s", exc)
            failure_result = getattr(exc, "result", None)
            if isinstance(failure_result, dict):
                deepseek_calls.append(
                    _usage_record(failure_result, "failed_answer_generation")
                )
            answer = PROVIDER_FAILED_ANSWER
            sources = []
            model = (
                failure_result.get("model")
                if isinstance(failure_result, dict) else None
            )
            grounding_status = "provider_error"
            grounding_reason = str(exc)
    else:
        answer = INSUFFICIENT_ANSWER
        sources = []
        model = None
        grounding_status = "no_evidence"
        grounding_reason = "no_transcript_evidence"

    # 6. сохранить ответ
    chat_memory.add_message(
        chat_id,
        "assistant",
        answer,
        model=model,
        sources=sources,
    )

    # 7. при необходимости сжать историю
    chat_memory.maybe_summarize(chat_id)

    # 8. создать заголовок
    msg_count = chat_memory.get_message_count(chat_id)
    chat_data = chat_memory.get_chat(chat_id)
    if (
        grounding_status in {
            "supported", "supported_partial", "supported_after_retry",
        }
        and msg_count == 2
        and chat_data
        and chat_data["title"] == chat_memory.DEFAULT_CHAT_TITLE
    ):
        try:
            chat_memory.generate_chat_title(chat_id)
        except Exception as e:
            logger.warning("не удалось создать заголовок автоматически: %s", e)

    return {
        "answer": answer,
        "model": model,
        "chat_id": chat_id,
        "sources": sources,
        "grounding_status": grounding_status,
        "grounding_reason": grounding_reason,
        "deepseek_usage": _summarize_usage(deepseek_calls),
    }


def _generate_strict_answer(
    messages: list[dict],
    evidence_sources: dict[str, dict],
    question: str,
) -> dict:
    deepseek_calls = []
    result = call_llm_for_answer(messages, answer_json_schema())
    deepseek_calls.extend(
        _generation_usage_records(result, "answer_generation")
    )
    model = result.get("model")
    grounded = _ground_generated_answer(
        result["answer"], evidence_sources, question,
    )
    deepseek_calls.extend(grounded.get("deepseek_calls", []))
    retried = False
    if not grounded.get("valid") or not grounded.get("claims"):
        logger.warning(
            "первая генерация не прошла проверку: %s",
            grounded.get("failure_reason"),
        )
        retry_result = call_llm_for_answer(
            _build_claim_retry_messages(
                question, evidence_sources, grounded,
            ),
            answer_json_schema(),
        )
        deepseek_calls.extend(
            _generation_usage_records(retry_result, "targeted_retry")
        )
        model = retry_result.get("model")
        grounded = _ground_generated_answer(
            retry_result["answer"], evidence_sources, question,
        )
        deepseek_calls.extend(grounded.get("deepseek_calls", []))
        retried = True

    if grounded.get("valid") and grounded.get("claims"):
        status = "supported"
        if retried:
            status = "supported_after_retry"
        elif grounded.get("rejected_claims"):
            status = "supported_partial"
        return {
            "answer": grounded["answer"],
            "sources": _select_grounded_sources(
                evidence_sources,
                grounded["source_ids"],
                grounded["quotes"],
            ),
            "model": model,
            "grounding_status": status,
            "grounding_reason": None,
            "deepseek_calls": deepseek_calls,
        }

    return {
        "answer": GROUNDING_FAILED_ANSWER,
        "sources": [],
        "model": model,
        "grounding_status": _failure_category(grounded),
        "grounding_reason": grounded.get("failure_reason"),
        "deepseek_calls": deepseek_calls,
    }


def _generate_semantic_answer(
    messages: list[dict],
    evidence_sources: dict[str, dict],
    question: str,
    requirements: list[dict],
) -> dict:
    deepseek_calls = []
    result = call_llm_for_semantic_answer(
        messages,
        semantic_answer_json_schema(requirements),
        select_answer_reasoning(requirements),
    )
    deepseek_calls.extend(
        _generation_usage_records(result, "semantic_answer_generation")
    )
    model = result.get("model")
    grounded = _ground_semantic_answer(
        result["answer"], evidence_sources, question, requirements,
    )
    deepseek_calls.extend(grounded.get("deepseek_calls", []))

    missing_ids = list(dict.fromkeys(
        grounded.get("uncovered_requirement_ids", [])
        + grounded.get("lost_requirement_ids", [])
    ))
    retried = False
    if not grounded.get("valid"):
        missing_ids = [item["requirement_id"] for item in requirements]
    if missing_ids:
        missing_requirements = [
            item for item in requirements
            if item["requirement_id"] in missing_ids
        ]
        try:
            repair_result = call_llm_for_semantic_repair(
                _build_semantic_repair_messages(
                    messages,
                    question,
                    grounded.get("answer", ""),
                    missing_requirements,
                ),
                semantic_answer_json_schema(missing_requirements),
            )
            deepseek_calls.append(
                _usage_record(repair_result, "semantic_targeted_repair")
            )
            model = repair_result.get("model") or model
            repaired = _ground_semantic_answer(
                repair_result["answer"],
                evidence_sources,
                question,
                missing_requirements,
            )
            deepseek_calls.extend(repaired.get("deepseek_calls", []))
            if repaired.get("valid"):
                grounded = combine_semantic_results(
                    grounded,
                    repaired,
                    requirements,
                )
            retried = True
        except RuntimeError as exc:
            logger.error("не удалось дополнить пропущенные части: %s", exc)
            failure_result = getattr(exc, "result", None)
            if isinstance(failure_result, dict):
                deepseek_calls.append(
                    _usage_record(failure_result, "failed_semantic_repair")
                )

    if grounded.get("valid") and grounded.get("sections"):
        missing_ids = grounded.get("uncovered_requirement_ids", [])
        status = "supported"
        if missing_ids:
            status = "supported_partial"
        elif retried:
            status = "supported_after_retry"
        elif grounded.get("rejected_sections"):
            status = "supported_partial"
        reason = (
            "uncovered_requirements:" + ",".join(missing_ids)
            if missing_ids else None
        )
        return {
            "answer": grounded["answer"],
            "sources": _select_semantic_sources(
                evidence_sources,
                grounded["source_ids"],
            ),
            "model": model,
            "grounding_status": status,
            "grounding_reason": reason,
            "deepseek_calls": deepseek_calls,
        }

    return {
        "answer": GROUNDING_FAILED_ANSWER,
        "sources": [],
        "model": model,
        "grounding_status": "grounding_failure",
        "grounding_reason": grounded.get("failure_reason"),
        "deepseek_calls": deepseek_calls,
    }


def _ground_semantic_answer(
    raw: str,
    evidence_sources: dict[str, dict],
    question: str,
    requirements: list[dict],
) -> dict:
    deepseek_calls = []
    grounded = validate_semantic_answer(raw, evidence_sources, requirements)
    if not grounded.get("valid"):
        grounded["deepseek_calls"] = deepseek_calls
        return grounded

    if not response_language_matches(question, grounded.get("sections", [])):
        grounded["valid"] = False
        grounded["failure_reason"] = "response_language_mismatch"
        grounded["uncovered_requirement_ids"] = [
            item["requirement_id"] for item in requirements
        ]
        grounded["deepseek_calls"] = deepseek_calls
        return grounded

    review_plan = build_section_review_plan(
        grounded,
        evidence_sources,
        question,
    )
    logger.info(
        "semantic grounding: %d безопасных, %d требуют llm, %d заблокированы",
        len(review_plan["safe_section_ids"]),
        len(review_plan["review_sections"]),
        len(review_plan["blocked_sections"]),
    )
    if review_plan["risk_reasons"]:
        logger.info("риски semantic-секций: %s", review_plan["risk_reasons"])

    if review_plan["review_sections"]:
        try:
            validation_result = call_llm_for_semantic_validation(
                _build_semantic_validation_messages(
                    review_plan["review_sections"]
                ),
                semantic_validation_json_schema(),
            )
            deepseek_calls.extend(
                _generation_usage_records(
                    validation_result,
                    "semantic_section_validation",
                )
            )
            validation = apply_section_validation(
                review_plan["review_sections"],
                validation_result["answer"],
            )
        except RuntimeError as exc:
            logger.error("semantic validator недоступен: %s", exc)
            failure_result = getattr(exc, "result", None)
            if isinstance(failure_result, dict):
                deepseek_calls.append(
                    _usage_record(failure_result, "failed_semantic_section_validation")
                )
            validation = apply_section_validation(
                review_plan["review_sections"],
                "",
            )
    else:
        validation = {
            "valid": True,
            "accepted_section_ids": [],
            "rejected_sections": [],
            "failure_reason": None,
        }

    merged = merge_section_review(
        grounded,
        review_plan,
        validation,
        requirements,
    )
    merged["deepseek_calls"] = deepseek_calls
    return merged


def _load_short_scoped_transcript(doc_id: str) -> list[dict] | None:
    transcript_chunks = list_chunks(doc_id, content_kind="transcript")
    if not transcript_chunks:
        return None
    total_chars = sum(
        len(chunk.get("text", "")) for chunk in transcript_chunks
    )
    if (
        len(transcript_chunks) <= SHORT_LECTURE_MAX_CHUNKS
        and total_chars <= SHORT_LECTURE_MAX_CHARS
    ):
        summary_chunks = list_chunks(doc_id, content_kind="summary")
        return transcript_chunks + summary_chunks
    return None


def _ground_generated_answer(
    raw: str,
    evidence_sources: dict[str, dict],
    question: str,
) -> dict:
    deepseek_calls = []
    grounded = validate_grounded_answer(raw, evidence_sources)
    if not grounded.get("valid") or not grounded.get("claims"):
        return grounded

    review_plan = build_semantic_review_plan(grounded, question=question)
    review = review_plan["review"]
    logger.info(
        "проверка риска: %d безопасных, %d требуют llm, %d заблокированы",
        len(review_plan["safe_claim_ids"]),
        len(review["claims"]),
        len(review_plan["blocked_claims"]),
    )
    if review_plan["risk_reasons"]:
        logger.info(
            "причины дополнительной проверки: %s",
            review_plan["risk_reasons"],
        )
    if review["claims"]:
        try:
            review, usage = _run_semantic_validation(review)
            deepseek_calls.append(usage)
        except RuntimeError as exc:
            logger.error("смысловая проверка недоступна: %s", exc)
            failure_result = getattr(exc, "result", None)
            if isinstance(failure_result, dict):
                deepseek_calls.append(
                    _usage_record(failure_result, "failed_semantic_validation")
                )
            review = apply_semantic_validation(review, "")
    else:
        logger.info("дополнительная llm-проверка не требуется")
    merged = merge_semantic_review(grounded, review_plan, review)
    merged["deepseek_calls"] = deepseek_calls
    return merged


def _usage_record(result: dict, operation: str) -> dict:
    usage = result.get("usage") or {}
    return {
        "operation": operation,
        "model": result.get("model"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "estimated_cost_upper_usd": float(
            result.get("estimated_cost_upper_usd") or 0.0
        ),
    }


def _generation_usage_records(result: dict, operation: str) -> list[dict]:
    records = [
        _usage_record(prior, f"failed_{operation}")
        for prior in result.get("prior_results", [])
        if isinstance(prior, dict)
    ]
    actual_operation = operation
    if result.get("generation_mode") == "no_thinking_after_length":
        actual_operation = f"{operation}_no_thinking_fallback"
    records.append(_usage_record(result, actual_operation))
    return records


def _summarize_usage(calls: list[dict]) -> dict:
    return {
        "calls": len(calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "estimated_cost_upper_usd": round(
            sum(call["estimated_cost_upper_usd"] for call in calls),
            8,
        ),
        "details": calls,
    }


def _build_claim_retry_messages(
    question: str,
    evidence_sources: dict[str, dict],
    grounded: dict,
) -> list[dict]:
    sources = [
        {"source_id": source_id, "text": value["text"]}
        for source_id, value in evidence_sources.items()
    ]
    payload = {
        "question": question,
        "failure_reason": grounded.get("failure_reason"),
        "rejected_claims": grounded.get("rejected_claims", []),
        "sources": sources,
    }
    return [
        {"role": "system", "content": _load_prompt("claim_retry_prompt.txt")},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _failure_category(grounded: dict) -> str:
    reason = grounded.get("failure_reason")
    if reason == "invalid_model_response":
        return "invalid_output"
    verdicts = [
        item.get("verdict", {}).get("verdict")
        for item in grounded.get("rejected_claims", [])
    ]
    if "validator_failed" in verdicts:
        return "validator_failure"
    return "grounding_failure"


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


def _retrieve_candidates(
    question: str,
    query_embedding: list[float],
    limit: int,
    course_id: str | None = None,
    scope_doc_id: str | None = None,
) -> list[dict]:
    """найти фрагменты конспекта и транскрипта в пределах курса."""
    scope = {"course_id": course_id, "doc_id": scope_doc_id}

    dense_summary = search(
        query_embedding,
        top_k=DENSE_TOP_K_SUMMARY,
        content_kind="summary",
        **scope,
    )
    dense_transcript = search(
        query_embedding,
        top_k=DENSE_TOP_K_TRANSCRIPT,
        content_kind="transcript",
        **scope,
    )
    lexical_summary = lexical_search(
        question,
        top_k=LEXICAL_TOP_K_SUMMARY,
        content_kind="summary",
        **scope,
    )
    lexical_transcript = lexical_search(
        question,
        top_k=LEXICAL_TOP_K_TRANSCRIPT,
        content_kind="transcript",
        **scope,
    )
    return _rrf_merge(
        [dense_summary, dense_transcript, lexical_summary, lexical_transcript],
        limit=limit,
    )


def _retrieve_lexical_candidates(
    question: str,
    limit: int,
    course_id: str | None = None,
    scope_doc_id: str | None = None,
) -> list[dict]:
    return lexical_search(
        question,
        top_k=limit,
        content_kind="transcript",
        course_id=course_id,
        doc_id=scope_doc_id,
    )


_METADATA_BLOCK = re.compile(
    r"<<<RAG_METADATA_START>>>.*?<<<RAG_METADATA_END>>>", re.DOTALL,
)
_METADATA_OPEN = re.compile(r"<<<RAG_METADATA_START>>>.*$", re.DOTALL)
_METADATA_CLOSE = re.compile(r"^.*?<<<RAG_METADATA_END>>>", re.DOTALL)
_METADATA_KEYS = (
    '"rag_metadata"', '"search_phrases"', '"common_mistakes"', '"synonyms"',
)
_MIN_FRAGMENT_CHARS = 40


def _strip_metadata(text: str) -> str:
    """убрать служебный блок индексации из текста фрагмента."""
    cleaned = _METADATA_BLOCK.sub("", text)
    cleaned = _METADATA_OPEN.sub("", cleaned)
    cleaned = _METADATA_CLOSE.sub("", cleaned)

    if any(key in cleaned for key in _METADATA_KEYS):
        return ""

    return cleaned.strip()


def _clean_matches(matches: list[dict]) -> list[dict]:
    """очистить фрагменты от служебных блоков и выбросить пустые."""
    cleaned = []
    for match in matches:
        text = _strip_metadata(match.get("text", ""))
        if len(text) < _MIN_FRAGMENT_CHARS:
            logger.info(
                "фрагмент %s отброшен: после очистки осталось %d символов",
                match.get("id"), len(text),
            )
            continue
        item = dict(match)
        item["text"] = text
        cleaned.append(item)
    return cleaned


def _load_prompt(filename: str) -> str:
    """загрузить системный промпт из файла."""
    path = os.path.join("prompts", filename)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Файл системного промпта не найден: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise RuntimeError(f"Файл промпта пуст: {path}")
    return content


def _question_guidance(question: str) -> str:
    normalized = " ".join(question.casefold().replace("ё", "е").split())
    comparison = (
        re.search(r"\bчем\b.*\bотлич\w*\b.*\bот\b", normalized)
        or re.search(r"\bразниц\w*\b.*\bмежду\b", normalized)
    )
    if comparison:
        return (
            "Это вопрос на сравнение. Первый тезис должен прямо назвать главное "
            "различие между двумя понятиями из вопроса и кратко охарактеризовать "
            "обе стороны. Не заменяй прямое различие сведениями о создании, "
            "атрибутах, методах, синтаксисе или примерами. Такие детали допустимы "
            "только после прямого ответа и только если без них сравнение непонятно."
        )
    return ""


def _build_messages(
    question: str,
    matches: list[dict],
    chat_id: str,
    semantic_mode: bool = False,
    requirements: list[dict] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """собрать сообщения для llm."""
    evidence_sources = {}
    matches = [
        match for match in _clean_matches(matches)
    ]
    transcript_matches = [
        match for match in matches
        if match.get("metadata", {}).get("content_kind") != "summary"
    ]
    summary_matches = [
        match for match in matches
        if match.get("metadata", {}).get("content_kind") == "summary"
    ]
    normalized_context = "\n\n".join(
        match["text"] for match in summary_matches
    )

    # системный промпт
    prompt_filename = (
        "semantic_answer_system_prompt.txt"
        if semantic_mode else "answer_system_prompt.txt"
    )
    system_prompt = _load_prompt(prompt_filename)

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
    if transcript_matches or summary_matches:
        from core.chunk_times import get_times, format_range

        times = get_times([match.get("id") for match in transcript_matches])

        context_parts = []
        for i, match in enumerate(transcript_matches):
            meta = match["metadata"]
            source_id = f"T{i + 1}"

            chunk_time = times.get(match.get("id")) or {}
            time_label = format_range(
                chunk_time.get("start_time"), chunk_time.get("end_time"),
            )
            time_part = f", {time_label}" if time_label else ""

            label = (
                f"Источник {source_id} [Транскрипт — прямая речь лектора] "
                f"(из \"{meta.get('filename', 'материал')}\"{time_part}, "
                f"чанк #{meta.get('chunk_index')})"
            )
            context_parts.append(f"--- {label} ---\n{match['text']}")
            source = {
                "evidence_id": source_id,
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename", "Без названия"),
                "source_type": meta.get("source_type", "material"),
                "content_kind": "transcript",
                "content_kind_label": "Транскрипт",
                "chunk_index": meta.get("chunk_index"),
                "start_time": chunk_time.get("start_time"),
                "end_time": chunk_time.get("end_time"),
                "time_label": time_label,
                "text_preview": match["text"][:220] + "…" if len(match["text"]) > 220 else match["text"],
                "distance": round(match.get("distance", 0), 4),
                "rerank_score": round(match.get("rerank_score", 0), 4),
            }
            evidence_sources[source_id] = {
                "text": match["text"],
                "normalized_context": normalized_context,
                "source": source,
            }

        for match in summary_matches:
            meta = match["metadata"]
            summary_role = (
                "нормализованный учебный материал"
                if semantic_mode
                else "нормализованный контекст, не источник доказательств"
            )
            label = (
                f"Вспомогательный ИИ-конспект [{summary_role}] "
                f"(из \"{meta.get('filename', 'материал')}\", "
                f"чанк #{meta.get('chunk_index')})"
            )
            context_parts.append(f"--- {label} ---\n{match['text']}")

        context_block = "\n\n".join(context_parts)
        guidance = "" if semantic_mode else _question_guidance(question)
        guidance_part = (
            f"\n\nТребование к структуре ответа:\n{guidance}"
            if guidance else ""
        )
        term_guidance = relevant_term_guidance(
            f"{question}\n{context_block}"
        )
        term_guidance_part = (
            "\n\nТехнические алиасы распознавания речи:\n"
            f"{term_guidance}\n"
            + (
                "Используй канонический термин справа."
                if semantic_mode
                else (
                    "В тексте тезиса используй канонический термин справа. "
                    "В поле quote копируй исходный транскрипт без исправлений."
                )
            )
            if term_guidance else ""
        )
        requirements_part = (
            "\n\nОбязательные требования к покрытию вопроса:\n"
            + json.dumps(requirements or [], ensure_ascii=False)
            if semantic_mode else ""
        )
        response_style_part = (
            "\n\nОбязательный профиль длины и плотности ответа:\n"
            + json.dumps(
                build_response_style(question, requirements or []),
                ensure_ascii=False,
            )
            if semantic_mode else ""
        )
        user_content = (
            f"Контекст из лекций:\n\n{context_block}\n\n"
            f"Вопрос студента: {question}{requirements_part}"
            f"{response_style_part}{guidance_part}{term_guidance_part}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})

    return messages, evidence_sources


def _select_grounded_sources(
    evidence_sources: dict[str, dict],
    source_ids: list[str],
    quotes: dict[str, list[str]],
) -> list[dict]:
    sources = []
    for source_id in source_ids:
        evidence = evidence_sources.get(source_id)
        if not evidence:
            continue
        source = dict(evidence["source"])
        source["text_preview"] = " … ".join(quotes.get(source_id, []))
        sources.append(source)
    return sources


def _select_semantic_sources(
    evidence_sources: dict[str, dict],
    source_ids: list[str],
) -> list[dict]:
    sources = []
    for source_id in source_ids:
        evidence = evidence_sources.get(source_id)
        if not evidence:
            continue
        source = dict(evidence["source"])
        text = evidence["text"]
        source["text_preview"] = text[:500] + "…" if len(text) > 500 else text
        sources.append(source)
    return sources


def _build_semantic_validation_messages(sections: list[dict]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": _load_prompt("semantic_validation_prompt.txt"),
        },
        {
            "role": "user",
            "content": json.dumps({"sections": sections}, ensure_ascii=False),
        },
    ]


def _build_semantic_repair_messages(
    original_messages: list[dict],
    question: str,
    existing_answer: str,
    missing_requirements: list[dict],
) -> list[dict]:
    payload = {
        "question": question,
        "existing_answer": existing_answer,
        "missing_requirements": missing_requirements,
        "lecture_context": original_messages[-1].get("content", ""),
    }
    return [
        {
            "role": "system",
            "content": _load_prompt("semantic_repair_prompt.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _build_claim_validation_messages(claims: list[dict]) -> list[dict]:
    system_prompt = _load_prompt("claim_validation_prompt.txt")
    payload = {"claims": claims}
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _run_semantic_validation(
    grounded: dict,
) -> tuple[dict, dict]:
    validation_result = call_llm_for_validation(
        _build_claim_validation_messages(grounded["claims"]),
        claim_validation_json_schema(),
    )
    return (
        apply_semantic_validation(grounded, validation_result["answer"]),
        _usage_record(validation_result, "semantic_validation"),
    )
