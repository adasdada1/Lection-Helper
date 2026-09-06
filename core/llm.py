"""запросы к DeepSeek API."""

import logging

from core.config import (
    DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS,
    DEEPSEEK_ANSWER_MAX_TOKENS,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_REPAIR_MAX_TOKENS,
    DEEPSEEK_VALIDATION_MAX_TOKENS,
    DEEPSEEK_VALIDATION_REASONING_EFFORT,
)
from core.deepseek_client import (
    DeepSeekCompletionError,
    call_deepseek_structured,
    call_deepseek_text,
)

logger = logging.getLogger(__name__)


def call_llm_for_answer(
    messages: list[dict],
    schema: dict,
) -> dict:
    return _call_answer_with_fallback(
        messages,
        schema,
        schema_name="grounded_lecture_answer",
        reasoning_effort=DEEPSEEK_REASONING_EFFORT,
        operation="генерация ответа",
    )


def call_llm_for_semantic_answer(
    messages: list[dict],
    schema: dict,
    reasoning_effort: str,
) -> dict:
    return _call_answer_with_fallback(
        messages,
        schema,
        schema_name="semantic_lecture_answer",
        reasoning_effort=reasoning_effort,
        operation="смысловая генерация ответа",
    )


def _call_answer_with_fallback(
    messages: list[dict],
    schema: dict,
    schema_name: str,
    reasoning_effort: str,
    operation: str,
) -> dict:
    try:
        return call_deepseek_structured(
            messages,
            schema_name=schema_name,
            schema=schema,
            max_tokens=DEEPSEEK_ANSWER_MAX_TOKENS,
            reasoning_effort=reasoning_effort,
            operation=operation,
        )
    except DeepSeekCompletionError as exc:
        if "finish_reason=length" not in str(exc):
            raise
        logger.warning(
            "thinking исчерпал лимит, выполняем один короткий fallback без thinking"
        )
        result = call_deepseek_structured(
            messages,
            schema_name=schema_name,
            schema=schema,
            max_tokens=DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS,
            reasoning_effort=reasoning_effort,
            operation=f"резервная {operation}",
            thinking_enabled=False,
        )
        result["prior_results"] = [exc.result]
        result["generation_mode"] = "no_thinking_after_length"
        return result

# открытые функции ---------------------------------------------------------------------------
def call_llm(messages: list[dict]) -> dict:
    """Создать конспект или чек-лист через DeepSeek без thinking."""
    return call_deepseek_text(
        messages,
        max_tokens=DEEPSEEK_ANSWER_MAX_TOKENS,
        operation="создание конспекта или чек-листа",
        thinking_enabled=False,
    )


def call_llm_for_validation(
    messages: list[dict],
    schema: dict,
) -> dict:
    return call_deepseek_structured(
        messages,
        schema_name="claim_validation",
        schema=schema,
        max_tokens=DEEPSEEK_VALIDATION_MAX_TOKENS,
        reasoning_effort=DEEPSEEK_VALIDATION_REASONING_EFFORT,
        operation="проверка доказательств",
    )


def call_llm_for_semantic_validation(
    messages: list[dict],
    schema: dict,
) -> dict:
    try:
        return call_deepseek_structured(
            messages,
            schema_name="semantic_section_validation",
            schema=schema,
            max_tokens=DEEPSEEK_VALIDATION_MAX_TOKENS,
            reasoning_effort=DEEPSEEK_VALIDATION_REASONING_EFFORT,
            operation="смысловая проверка секций",
        )
    except DeepSeekCompletionError as exc:
        if "finish_reason=length" not in str(exc):
            raise
        logger.warning(
            "semantic validator исчерпал лимит, выполняем короткий fallback без thinking"
        )
        result = call_deepseek_structured(
            messages,
            schema_name="semantic_section_validation",
            schema=schema,
            max_tokens=DEEPSEEK_REPAIR_MAX_TOKENS,
            reasoning_effort="low",
            operation="резервная смысловая проверка секций",
            thinking_enabled=False,
        )
        result["prior_results"] = [exc.result]
        result["generation_mode"] = "no_thinking_after_length"
        return result


def call_llm_for_semantic_repair(
    messages: list[dict],
    schema: dict,
) -> dict:
    return call_deepseek_structured(
        messages,
        schema_name="semantic_answer_repair",
        schema=schema,
        max_tokens=DEEPSEEK_REPAIR_MAX_TOKENS,
        reasoning_effort="low",
        operation="дополнение пропущенных частей",
        thinking_enabled=False,
    )


# сжатие истории ---------------------------------------------------------------------------
def call_llm_for_summarization(messages: list[dict]) -> dict:
    """Сжать историю через DeepSeek без thinking."""
    return call_deepseek_text(
        messages,
        max_tokens=DEEPSEEK_REPAIR_MAX_TOKENS,
        operation="сжатие истории",
        thinking_enabled=False,
    )


# заголовок чата ---------------------------------------------------------------------------
def call_llm_for_title(messages: list[dict]) -> dict:
    """Создать короткий заголовок через DeepSeek без thinking."""
    return call_deepseek_text(
        messages,
        max_tokens=128,
        operation="генерация заголовка чата",
        thinking_enabled=False,
    )
