"""настройки проекта."""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
# заглушка для запуска без pydantic
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env_int(name: str, default: int) -> int:
    """прочитать положительное число из .env или взять запасное."""
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def env_reasoning(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().casefold()
    return value if value in {"low", "medium"} else default


# файлы и хранилища
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = str(BASE_DIR / "uploads")
CHATS_DB_PATH = str(DATA_DIR / "chats.db")
CHROMA_DIR = str(DATA_DIR / "chroma")
CHROMA_COLLECTION_NAME = "lecture_chunks"


# форматы загрузки
VIDEO_EXTENSIONS = {".mp4", ".ts", ".avi",
                    ".mkv", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


# поиск rag и повторное ранжирование
RETRIEVAL_TOP_K = 10
DENSE_TOP_K_SUMMARY = 5
DENSE_TOP_K_TRANSCRIPT = 7
LEXICAL_TOP_K_SUMMARY = 5
LEXICAL_TOP_K_TRANSCRIPT = 7
RERANK_TOP_N_SUMMARY = 3
RERANK_TOP_N_TRANSCRIPT = 4
RERANK_SCORE_FLOOR = 0.0
RRF_K = 60
RERANKER_BATCH_SIZE = env_int("RERANKER_BATCH_SIZE", 4)


# обработка длинных лекций по частям
DIRECT_LLM_CHAR_LIMIT = 20_000
MAP_CHUNK_CHARS = 14_000
MAP_CHUNK_OVERLAP = 800
CHECKLIST_MERGE_CHAR_LIMIT = 40_000


# память чатов
CHAT_RECENT_WINDOW = 8
CHAT_SUMMARIZE_THRESHOLD = 12
DEFAULT_CHAT_TITLE = "Новый чат"


# выбор моделей openrouter и LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "minimax/minimax-m3:free"

FALLBACK_MODELS = [
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

UTILITY_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "poolside/laguna-s-2.1:free",
]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
BACKOFF_BASE = 2.0

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/chat/completions",
)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_REASONING_EFFORT = env_reasoning(
    "DEEPSEEK_REASONING_EFFORT", "medium",
)
DEEPSEEK_VALIDATION_REASONING_EFFORT = env_reasoning(
    "DEEPSEEK_VALIDATION_REASONING_EFFORT", "medium",
)
DEEPSEEK_MAX_RETRIES = env_int("DEEPSEEK_MAX_RETRIES", 2)
DEEPSEEK_ANSWER_MAX_TOKENS = env_int("DEEPSEEK_ANSWER_MAX_TOKENS", 16384)
DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS = env_int(
    "DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS", 2048,
)
DEEPSEEK_VALIDATION_MAX_TOKENS = env_int(
    "DEEPSEEK_VALIDATION_MAX_TOKENS", 8192,
)
DEEPSEEK_REPAIR_MAX_TOKENS = env_int("DEEPSEEK_REPAIR_MAX_TOKENS", 2048)
GROUNDING_MODE = os.getenv("GROUNDING_MODE", "semantic").strip().casefold()
if GROUNDING_MODE not in {"semantic", "strict"}:
    GROUNDING_MODE = "semantic"
SHORT_LECTURE_MAX_CHUNKS = env_int("SHORT_LECTURE_MAX_CHUNKS", 8)
SHORT_LECTURE_MAX_CHARS = env_int("SHORT_LECTURE_MAX_CHARS", 32_000)
