"""запросы к внешним LLM-провайдерам."""

import time
import logging
import requests

from core.config import (
    BACKOFF_BASE,
    DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS,
    DEEPSEEK_ANSWER_MAX_TOKENS,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_VALIDATION_MAX_TOKENS,
    DEEPSEEK_VALIDATION_REASONING_EFFORT,
    FALLBACK_MODELS,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    PRIMARY_MODEL,
    UTILITY_MODELS,
)
from core.deepseek_client import DeepSeekCompletionError, call_deepseek_structured

logger = logging.getLogger(__name__)


def call_llm_for_answer(
    messages: list[dict],
    schema: dict,
) -> dict:
    try:
        return call_deepseek_structured(
            messages,
            schema_name="grounded_lecture_answer",
            schema=schema,
            max_tokens=DEEPSEEK_ANSWER_MAX_TOKENS,
            reasoning_effort=DEEPSEEK_REASONING_EFFORT,
            operation="генерация ответа",
        )
    except DeepSeekCompletionError as exc:
        if "finish_reason=length" not in str(exc):
            raise
        logger.warning(
            "thinking исчерпал лимит, выполняем один короткий fallback без thinking"
        )
        result = call_deepseek_structured(
            messages,
            schema_name="grounded_lecture_answer",
            schema=schema,
            max_tokens=DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS,
            reasoning_effort=DEEPSEEK_REASONING_EFFORT,
            operation="резервная генерация ответа",
            thinking_enabled=False,
        )
        result["prior_results"] = [exc.result]
        result["generation_mode"] = "no_thinking_after_length"
        return result

# открытые функции ---------------------------------------------------------------------------
def call_llm(messages: list[dict]) -> dict:
    """отправить запрос и при ошибке сменить модель."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не настроен .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    all_models = [PRIMARY_MODEL] + FALLBACK_MODELS
    total_attempts = len(all_models)

    last_error_msg: str | None = None

    for attempt in range(1, total_attempts + 1):
        current_model_idx = (attempt - 1) % len(all_models)
        current_model = all_models[current_model_idx]
        
        payload = {
            "model": current_model,
            "messages": messages,
        }

        try:
            logger.info(
                "отправляем запрос в OPENROUTER, попытка %d/%d, модель %s",
                attempt, total_attempts, current_model,
            )

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            
            # проверить ошибку внутри успешного ответа
            if "error" in data:
                err_msg = data["error"].get("message", "Unknown")
                logger.warning("внутренняя ошибка openrouter, модель %s: %s", current_model, err_msg)
                last_error_msg = f"API error: {err_msg}"
                # перейти к следующей модели
                raise ValueError("API error inside 200 OK JSON")

            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not reply:
                logger.warning("модель %s вернула пустой ответ", current_model)
                last_error_msg = "Один из серверов вернул пустой ответ."
                raise ValueError("Empty reply string")

            logger.info("получили ответ от модели %s", current_model)
            return {
                "answer": reply,
                "model": current_model,
            }

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error_msg = f"Ошибка сети/таймаут: {exc}"
            logger.warning("попытка %d: %s", attempt, last_error_msg)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_error_msg = f"HTTP {status}"
            logger.warning("попытка %d, ошибка http %s", attempt, status)

            if status == 401 or status == 403:
                raise RuntimeError(
                    f"Ошибка авторизации (HTTP {status}). Проверьте API-ключ."
                )

        except ValueError as exc:
            # обработать пустой ответ и ошибку api
            logger.warning("попытка %d не удалась: %s", attempt, exc)

        # подождать перед повтором
        if attempt < total_attempts:
            wait = BACKOFF_BASE ** attempt
            logger.info("через %.1f с попробуем следующую модель", wait)
            time.sleep(wait)

    raise RuntimeError(
        f"Не удалось получить ответ после {total_attempts} попыток. {last_error_msg}"
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


# сжатие истории ---------------------------------------------------------------------------
def call_llm_for_summarization(messages: list[dict]) -> dict:
    """создать сводку через OPENROUTER."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    all_models = UTILITY_MODELS
    total_attempts = len(all_models)

    last_error_msg: str | None = None

    for attempt in range(1, total_attempts + 1):
        current_model = all_models[attempt - 1]

        payload = {
            "model": current_model,
            "messages": messages,
        }

        try:
            logger.info(
                "запрашиваем сводку, попытка %d/%d, модель %s",
                attempt, total_attempts, current_model,
            )

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                err_msg = data["error"].get("message", "Unknown")
                logger.warning(
                    "внутренняя ошибка при создании сводки, модель %s: %s", current_model, err_msg,
                )
                last_error_msg = f"API error: {err_msg}"
                raise ValueError("API error inside 200 OK JSON")

            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not reply:
                logger.warning(
                    "модель сводки %s вернула пустой ответ", current_model,
                )
                last_error_msg = "Пустой ответ модели суммаризации"
                raise ValueError("Empty reply")

            logger.info("получили сводку от модели %s", current_model)
            return {"answer": reply, "model": current_model}

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error_msg = f"Сеть/таймаут: {exc}"
            logger.warning("попытка создать сводку %d: %s", attempt, last_error_msg)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_error_msg = f"HTTP {status}"
            logger.warning("попытка создать сводку %d, http %s", attempt, status)
            if status in (401, 403):
                raise RuntimeError(f"Ошибка авторизации (HTTP {status}). Проверьте API-ключ.")

        except ValueError as exc:
            logger.warning("попытка создать сводку %d не удалась: %s", attempt, exc)

        # подождать перед следующей моделью
        if attempt < total_attempts:
            wait = min(BACKOFF_BASE ** attempt, 5.0)
            time.sleep(wait)

    raise RuntimeError(
        f"Суммаризация не удалась после {total_attempts} попыток. {last_error_msg}"
    )


# заголовок чата ---------------------------------------------------------------------------
def call_llm_for_title(messages: list[dict]) -> dict:
    """создать короткий заголовок чата."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    all_models = UTILITY_MODELS
    total_attempts = len(all_models)
    last_error_msg: str | None = None

    for attempt in range(1, total_attempts + 1):
        current_model = all_models[attempt - 1]

        payload = {
            "model": current_model,
            "messages": messages,
        }

        try:
            logger.info(
                "запрашиваем заголовок, попытка %d/%d, модель %s",
                attempt, total_attempts, current_model,
            )

            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ValueError(data["error"].get("message", "Unknown"))

            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not reply:
                raise ValueError("Empty reply")

            logger.info("получили заголовок от модели %s", current_model)
            return {"answer": reply, "model": current_model}

        except Exception as exc:
            last_error_msg = str(exc)
            logger.warning("попытка создать заголовок %d не удалась: %s", attempt, exc)

        if attempt < total_attempts:
            time.sleep(1)

    raise RuntimeError(f"Генерация заголовка не удалась: {last_error_msg}")
