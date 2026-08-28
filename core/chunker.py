"""разбивает текст лекции на фрагменты с перекрытием."""

import logging
import nltk

logger = logging.getLogger(__name__)

# скачать токенизатор, если его нет
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def chunk_text(
    text: str,
    max_chunk_chars: int = 2000,   # около 500 токенов
    overlap_chars: int = 200,      # около 50 токенов
) -> list[dict]:
    """разбить текст по предложениям с перекрытием."""
    if not text or not text.strip():
        return []

    sentences = nltk.sent_tokenize(text)

    if not sentences:
        return []

    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)

        # если предложение не помещается, закончить текущий фрагмент
        if current_len + sentence_len > max_chunk_chars and current_sentences:
            _finalize_chunk(chunks, current_sentences)

            # взять последние предложения для перекрытия
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) <= overlap_chars:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    # добавить остаток
    if current_sentences:
        _finalize_chunk(chunks, current_sentences)

    logger.info(
        "текст разделен на %d фрагментов (лимит=%d, перекрытие=%d)",
        len(chunks), max_chunk_chars, overlap_chars,
    )
    return chunks


def chunk_segments(
    segments: list[dict],
    max_chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> list[dict]:
    """разбить фрагменты whisper на чанки, сохраняя время."""
    if not segments:
        return []

    chunks: list[dict] = []
    current: list[dict] = []
    current_len = 0

    for segment in segments:
        segment_len = len(segment["text"])

        if current_len + segment_len > max_chunk_chars and current:
            _finalize_segment_chunk(chunks, current)

            overlap: list[dict] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s["text"]) <= overlap_chars:
                    overlap.insert(0, s)
                    overlap_len += len(s["text"])
                else:
                    break

            current = overlap
            current_len = overlap_len

        current.append(segment)
        current_len += segment_len

    if current:
        _finalize_segment_chunk(chunks, current)

    logger.info(
        "транскрипт разделен на %d фрагментов из %d сегментов",
        len(chunks), len(segments),
    )
    return chunks


def _finalize_segment_chunk(
    chunks: list[dict],
    segments: list[dict],
) -> None:
    """добавить готовый фрагмент с границами по времени."""
    starts = [s["start"] for s in segments if s.get("start") is not None]
    ends = [s["end"] for s in segments if s.get("end") is not None]

    chunks.append({
        "text": " ".join(s["text"] for s in segments),
        "chunk_index": len(chunks),
        "start_time": min(starts) if starts else None,
        "end_time": max(ends) if ends else None,
    })


def _finalize_chunk(
    chunks: list[dict],
    sentences: list[str],
) -> None:
    """добавить готовый фрагмент."""
    chunk_text_str = " ".join(sentences)

    chunks.append({
        "text": chunk_text_str,
        "chunk_index": len(chunks),
    })
