"""Локальная ограниченная диагностика semantic-проверки, без ключей и reasoning."""

import json
import logging
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from core.config import DATA_DIR

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_handler: RotatingFileHandler | None = None


def record_semantic_diagnostic(trace_id: str, question: str, stage: str, **details) -> None:
    """Писать только явно переданные данные ответа, не HTTP-запросы провайдеров."""
    global _handler
    try:
        payload = {
            "trace_id": trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "question": question,
            **details,
        }
        text = json.dumps(payload, ensure_ascii=False)
        with _lock:
            if _handler is None:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                _handler = RotatingFileHandler(
                    DATA_DIR / "semantic_diagnostics.jsonl",
                    maxBytes=2_000_000,
                    backupCount=2,
                    encoding="utf-8",
                    delay=True,
                )
                _handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord(__name__, logging.INFO, __file__, 0, text, (), None)
            _handler.handle(record)
    except (OSError, TypeError, ValueError):
        _logger.warning("Не удалось сохранить semantic-диагностику", exc_info=True)
