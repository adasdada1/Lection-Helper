import json
import logging
import threading
import time

import requests

from core.config import (
    BACKOFF_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MODEL,
    RETRYABLE_STATUS_CODES,
)

logger = logging.getLogger(__name__)

_usage_lock = threading.Lock()
_usage_totals = {
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_cost_upper_usd": 0.0,
}


class DeepSeekCompletionError(RuntimeError):
    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def call_deepseek_structured(
    messages: list[dict],
    schema_name: str,
    schema: dict,
    max_tokens: int,
    reasoning_effort: str,
    operation: str,
    thinking_enabled: bool = True,
) -> dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY не настроен в .env")
    return _call_json_output(
        messages,
        schema_name,
        schema,
        max_tokens,
        reasoning_effort,
        operation,
        thinking_enabled,
    )


def get_usage_totals() -> dict:
    with _usage_lock:
        return dict(_usage_totals)


def _call_json_output(
    messages: list[dict],
    schema_name: str,
    schema: dict,
    max_tokens: int,
    reasoning_effort: str,
    operation: str,
    thinking_enabled: bool,
) -> dict:
    payload = _base_payload(
        _with_json_instruction(messages, schema_name, schema),
        max_tokens,
        reasoning_effort,
        thinking_enabled,
    )
    payload["response_format"] = {"type": "json_object"}
    data = _post_with_retries(
        DEEPSEEK_API_URL,
        payload,
        operation,
    )
    choice = _first_choice(data, operation)
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if not content.strip():
        raise DeepSeekCompletionError(
            f"{operation}: DeepSeek вернул пустой JSON Output",
            _result(data, ""),
        )
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise DeepSeekCompletionError(
            f"{operation}: DeepSeek вернул невалидный JSON",
            _result(data, content),
        ) from exc
    return _result(data, content)


def _base_payload(
    messages: list[dict],
    max_tokens: int,
    reasoning_effort: str,
    thinking_enabled: bool,
) -> dict:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload["thinking"] = {
        "type": "enabled" if thinking_enabled else "disabled",
    }
    if thinking_enabled:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def _post_with_retries(
    url: str,
    payload: dict,
    operation: str,
    max_attempts: int | None = None,
) -> dict:
    attempts = max_attempts or DEEPSEEK_MAX_RETRIES
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = "неизвестная ошибка"

    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                "%s через DeepSeek, попытка %d/%d, модель %s",
                operation,
                attempt,
                attempts,
                DEEPSEEK_MODEL,
            )
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                error = data.get("error") or {}
                message = error.get("message", "Unknown")
                raise RuntimeError(f"DeepSeek API error: {message}")
            _record_usage(data, operation)
            return data
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = f"ошибка сети/таймаут: {exc}"
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_error = f"HTTP {status}"
            if status in (401, 403):
                raise RuntimeError(
                    f"DeepSeek: ошибка авторизации HTTP {status}. Проверьте DEEPSEEK_API_KEY."
                ) from exc
            if status == 402:
                raise RuntimeError("DeepSeek: недостаточно средств на балансе") from exc
            if status not in RETRYABLE_STATUS_CODES:
                raise RuntimeError(f"{operation} не выполнена: {last_error}") from exc
        except (ValueError, RuntimeError) as exc:
            last_error = str(exc)

        logger.warning("%s не удалась: %s", operation, last_error)
        if attempt < attempts:
            time.sleep(min(BACKOFF_BASE ** attempt, 3.0))

    raise RuntimeError(f"{operation} не выполнена: {last_error}")


def _first_choice(data: dict, operation: str) -> dict:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{operation}: DeepSeek не вернул choices")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason in {"length", "content_filter", "insufficient_system_resource"}:
        raise DeepSeekCompletionError(
            f"{operation}: DeepSeek завершил ответ с finish_reason={finish_reason}",
            _result(data, ""),
        )
    return choice


def _result(data: dict, answer: str) -> dict:
    usage = data.get("usage") or {}
    return {
        "answer": answer,
        "model": data.get("model") or DEEPSEEK_MODEL,
        "usage": usage,
        "estimated_cost_upper_usd": _estimate_cost_upper(usage),
    }


def _record_usage(data: dict, operation: str) -> None:
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    estimated = _estimate_cost_upper(usage)
    with _usage_lock:
        _usage_totals["requests"] += 1
        _usage_totals["prompt_tokens"] += prompt_tokens
        _usage_totals["completion_tokens"] += completion_tokens
        _usage_totals["estimated_cost_upper_usd"] += estimated
        total = _usage_totals["estimated_cost_upper_usd"]
    logger.info(
        "%s: DeepSeek usage input=%d output=%d, верхняя оценка $%.6f, за запуск $%.6f",
        operation,
        prompt_tokens,
        completion_tokens,
        estimated,
        total,
    )


def _estimate_cost_upper(usage: dict) -> float:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return (
        prompt_tokens * 0.44 / 1_000_000
        + completion_tokens * 1.32 / 1_000_000
    )


def _with_json_instruction(
    messages: list[dict],
    schema_name: str,
    schema: dict,
) -> list[dict]:
    instruction = (
        f"Верни только JSON-объект {schema_name} без Markdown, "
        "соответствующий этой схеме: "
        + json.dumps(schema, ensure_ascii=False)
    )
    result = [dict(message) for message in messages]
    if result and result[0].get("role") == "system":
        result[0]["content"] = result[0].get("content", "") + "\n\n" + instruction
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result
