"""обрабатывает лекцию от загрузки до сохранения."""

import os
import gc
import re
import json
import uuid
import subprocess
import logging
from pathlib import Path
from datetime import datetime

from core.config import (
    ALLOWED_EXTENSIONS,
    AUDIO_EXTENSIONS,
    CHECKLIST_MERGE_CHAR_LIMIT,
    DIRECT_LLM_CHAR_LIMIT,
    MAP_CHUNK_CHARS,
    MAP_CHUNK_OVERLAP,
    UPLOAD_DIR,
    VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)

class IngestJobManager:
    """хранит статус фоновой обработки."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def create_job(self, filename: str, course_id: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "course_id": course_id,
            "status": "pending",
            "stage": "Ожидание",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "doc_id": None,
            "chunk_count": 0,
            "transcript": None,
            "transcription": None,
            "checklist": None,
            "summary": None,
            "error": None,
        }
        return job_id

    def get_status(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    # служебные методы ------------------------------------------------------------------
    def _update(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    # обработка файла ------------------------------------------------------------------
    def run(self, job_id: str, file_path: str):
        """обработать загруженный файл."""
        filename = self.jobs[job_id]["filename"]
        course_id = self.jobs[job_id].get("course_id")
        doc_id = str(uuid.uuid4())
        self._update(job_id, doc_id=doc_id, status="processing")

        extracted_audio_path: str | None = None  # временное аудио

        try:
            ext = Path(file_path).suffix.lower()
            is_video = ext in VIDEO_EXTENSIONS
            source_type = "video" if is_video else "audio"
            audio_path = file_path

            # 1. извлечь аудио
            if is_video:
                self._update(job_id, stage="Извлечение аудио из видео", progress=5)
                extracted_audio_path = file_path.rsplit(".", 1)[0] + ".wav"
                _extract_audio(file_path, extracted_audio_path)
                audio_path = extracted_audio_path
                logger.info("извлекли аудио: %s", audio_path)

            # 2. получить транскрипцию
            self._update(
                job_id,
                stage="Транскрибация",
                progress=10,
                transcription={"active": True},
            )
            from core.transcriber import MediaProcessor
            processor = MediaProcessor()  # здесь загрузится whisper

            segments = processor.transcribe_segments(audio_path)
            self._update(job_id, transcription=None)
            logger.info(
                "транскрипция готова, сегментов: %d, с отметками времени: %d",
                len(segments),
                sum(1 for s in segments if s.get("start") is not None),
            )

            # 3. очистить текст
            self._update(job_id, stage="Очистка текста", progress=40)
            bad_words_path = "bad_words.json"
            if os.path.exists(bad_words_path):
                for segment in segments:
                    segment["text"] = processor.clean_text(segment["text"], bad_words_path)
                segments = [s for s in segments if s["text"]]
                logger.info("очистили текст по bad_words.json")

            transcript = " ".join(s["text"] for s in segments).strip()
            self._update(job_id, transcript=transcript)

            del processor
            _free_gpu()
            logger.info("выгрузили whisper из видеопамяти")

            # 4. сделать чек-лист
            self._update(job_id, stage="Генерация чек-листа", progress=50)
            checklist, checklist_parts = _generate_checklist(transcript)
            self._update(job_id, checklist=checklist)
            logger.info("чек-лист готов, символов: %d", len(checklist))

            # 5. сделать конспект
            self._update(job_id, stage="Генерация конспекта", progress=60)
            full_summary_with_metadata = _generate_summary(
                transcript, checklist, checklist_parts,
            )
            
            import re
            pattern = r"<<<RAG_METADATA_START>>>\s*(.*?)\s*<<<RAG_METADATA_END>>>"
            match = re.search(pattern, full_summary_with_metadata, re.DOTALL)
            if match:
                display_summary = re.sub(pattern, "", full_summary_with_metadata, flags=re.DOTALL).strip()
            else:
                display_summary = full_summary_with_metadata
                
            self._update(job_id, summary=display_summary)
            logger.info("конспект готов, символов: %d", len(full_summary_with_metadata))

            # 6. разбить на фрагменты
            self._update(job_id, stage="Разбиение на фрагменты", progress=75)
            from core.chunker import chunk_text, chunk_segments
            
            INDEX_SUMMARY = True
            INDEX_TRANSCRIPT = True
            
            chunks = []
            
            if INDEX_SUMMARY:
                summary_chunks = chunk_text(display_summary)
                for c in summary_chunks:
                    c["content_kind"] = "summary"
                chunks.extend(summary_chunks)
                
            if INDEX_TRANSCRIPT:
                transcript_chunks = chunk_segments(segments)
                start_idx = chunks[-1]["chunk_index"] + 1 if chunks else 0
                for i, c in enumerate(transcript_chunks):
                    c["chunk_index"] = start_idx + i
                    c["content_kind"] = "transcript"
                chunks.extend(transcript_chunks)

            logger.info("разделили текст на %d фрагментов", len(chunks))

            # 7. создать векторы
            self._update(job_id, stage="Создание эмбеддингов (BGE-M3)", progress=80)
            from core.embeddings import get_embedder
            embedder = get_embedder()
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_documents(texts)
            logger.info("посчитали векторы: %d", len(embeddings))

            # 8. сохранить в базы
            self._update(job_id, stage="Сохранение в базу знаний", progress=90)
            from core.vector_store import add_chunks
            from core.doc_store import add_document
            from core.lexical_store import add_chunks as add_lexical_chunks
            from core.chunk_times import save_times

            stored = add_chunks(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                filename=filename,
                source_type=source_type,
                course_id=course_id,
            )
            self._update(job_id, chunk_count=stored)

            try:
                add_lexical_chunks(
                    doc_id=doc_id,
                    chunks=chunks,
                    filename=filename,
                    source_type=source_type,
                    course_id=course_id,
                )
            except Exception as e:
                logger.warning("лексический индекс не обновлен: %s", e)

            try:
                save_times(doc_id, chunks)
            except Exception as e:
                logger.warning("отметки времени не сохранены: %s", e)
            
            # сохранить документ в sqlite
            chunk_count_summary = sum(1 for c in chunks if c.get("content_kind") == "summary")
            chunk_count_transcript = sum(1 for c in chunks if c.get("content_kind") == "transcript")
            add_document(
                doc_id=doc_id,
                filename=filename,
                source_type=source_type,
                transcript=transcript,
                checklist=checklist,
                full_summary=full_summary_with_metadata,
                display_summary=display_summary,
                chunk_count_transcript=chunk_count_transcript,
                chunk_count_summary=chunk_count_summary,
                course_id=course_id,
            )

            # готово
            self._update(
                job_id,
                status="done",
                stage="Готово",
                progress=100,
            )
            logger.info(
                "завершили обработку '%s', сохранили фрагментов: %d", filename, stored,
            )

        except Exception as e:
            logger.exception("загрузка задачи %s не удалась", job_id)
            self._update(
                job_id,
                status="error",
                stage="Ошибка",
                error=str(e),
            )

        finally:
            # удалить временные файлы
            _safe_remove(file_path)
            if extracted_audio_path:
                _safe_remove(extracted_audio_path)


# вспомогательные функции ---------------------------------------------------------------------------

def _extract_audio(video_path: str, output_path: str) -> None:
    """извлечь аудио через ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",                       # без видео
            "-acodec", "pcm_s16le",      # несжатый
            "-ar", "16000",       # 16 кгц для whisper. только такой принимает видимо
            "-ac", "1",                  # моно
            "-y",                        # перезаписать
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr[:500]}")


def _load_prompt_file(path: str) -> str:
    """прочитать промпт или остановить обработку."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Файл промпта не найден: {path}. "
            "Создайте файл и перезапустите обработку."
        )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise RuntimeError(
            f"Файл промпта пуст: {path}. "
            "Заполните файл и перезапустите обработку."
        )

    return content


def _truncate_for_llm(transcript: str, max_chars: int = DIRECT_LLM_CHAR_LIMIT) -> str:
    """обрезать текст до лимита."""
    if len(transcript) > max_chars:
        return transcript[:max_chars] + "\n\n[… текст обрезан …]"
    return transcript


def _split_text_for_llm(
    text: str,
    max_chars: int = MAP_CHUNK_CHARS,
    overlap_chars: int = MAP_CHUNK_OVERLAP,
) -> list[str]:
    """разбить длинный текст по пробелам."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            window_start = max(start + max_chars - 1200, start)
            split_at = text.rfind("\n", window_start, hard_end)
            if split_at == -1:
                split_at = text.rfind(" ", window_start, hard_end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap_chars, 0)

    return chunks


def _generate_checklist(transcript: str) -> tuple[str, list[str]]:
    """сделать чек-лист из транскрипта. вернуть общий чек-лист и его части."""
    checklist_prompt_path = os.path.join("prompts", "check_list_prompt.txt")
    checklist_system_prompt = _load_prompt_file(checklist_prompt_path)

    from core.llm import call_llm

    def _call_for_chunk(chunk: str, index: int | None = None, total: int | None = None) -> str:
        part_note = f"\nЭто часть {index}/{total} длинной лекции." if index and total else ""
        result = call_llm([
            {"role": "system", "content": checklist_system_prompt + part_note},
            {
                "role": "user",
                "content": (
                    f"[ВСТАВЬ СЮДА ASR]\n{chunk}\n\n"
                    "[ВСТАВЬ СЮДА ИЗОБРАЖЕНИЯ / OCR / ОПИСАНИЯ]\n"
                    "(Нет изображений в данном контексте)"
                ),
            },
        ])
        answer = result["answer"]
        if not answer:
            raise RuntimeError("LLM вернул пустой чек-лист")
        return answer

    if len(transcript) <= DIRECT_LLM_CHAR_LIMIT:
        logger.info("запрашиваем чек-лист у llm")
        checklist = _call_for_chunk(transcript)
        logger.info("получили чек-лист")
        return checklist, [checklist]

    chunks = _split_text_for_llm(transcript)
    logger.info("чек-лист создается по частям, частей: %d", len(chunks))
    partials = [
        _call_for_chunk(chunk, index=i, total=len(chunks))
        for i, chunk in enumerate(chunks, start=1)
    ]
    return _merge_checklists(partials), partials


def _merge_checklists(partials: list[str]) -> str:
    """свести частичные чек-листы в один без потери пунктов."""
    current = list(partials)

    while len(current) > 1:
        merged: list[str] = []
        batch: list[str] = []
        size = 0

        for part in current:
            if batch and size + len(part) > CHECKLIST_MERGE_CHAR_LIMIT:
                merged.append(_merge_checklist_batch(batch) if len(batch) > 1 else batch[0])
                batch, size = [], 0
            batch.append(part)
            size += len(part)

        if batch:
            merged.append(_merge_checklist_batch(batch) if len(batch) > 1 else batch[0])

        if len(merged) >= len(current):
            logger.warning("склейка чек-листа не уменьшает объем, оставляем как есть")
            return "\n\n".join(current)

        logger.info("склейка чек-листа: %d -> %d частей", len(current), len(merged))
        current = merged

    return current[0]


def _merge_checklist_batch(batch: list[str]) -> str:
    """объединить несколько частичных чек-листов одним запросом."""
    from core.llm import call_llm

    merge_prompt = _load_prompt_file(
        os.path.join("prompts", "checklist_merge_prompt.txt")
    )
    joined = "\n\n".join(
        f"## Часть {i}\n{part}" for i, part in enumerate(batch, start=1)
    )
    result = call_llm([
        {"role": "system", "content": merge_prompt},
        {"role": "user", "content": joined},
    ])
    merged = result["answer"]
    if not merged:
        raise RuntimeError("LLM вернул пустой итоговый чек-лист")
    return merged


def _generate_summary(
    transcript: str,
    checklist: str,
    checklist_parts: list[str] | None = None,
) -> str:
    """сделать конспект из транскрипта."""
    summary_prompt_path = os.path.join("prompts", "summary_system_prompt.txt")
    system_prompt_template = _load_prompt_file(summary_prompt_path)

    from core.llm import call_llm

    def _call_for_chunk(
        chunk: str,
        part_checklist: str,
        index: int | None = None,
        total: int | None = None,
    ) -> str:
        system_prompt = system_prompt_template.replace("{checklist}", part_checklist)
        part_note = ""
        if index and total:
            part_note = (
                f"\nСейчас обрабатывается часть {index}/{total} длинной лекции. "
                "Чек-лист выше относится только к этой части: закрой все его пункты "
                "и не пиши о том, чего в этой части нет."
            )
        result = call_llm([
            {"role": "system", "content": system_prompt + part_note},
            {"role": "user", "content": chunk},
        ])
        answer = result["answer"]
        if not answer:
            raise RuntimeError("LLM вернул пустой конспект")
        return answer

    if len(transcript) <= DIRECT_LLM_CHAR_LIMIT:
        logger.info("запрашиваем конспект у LLM")
        summary = _call_for_chunk(transcript, checklist)
        logger.info("получили конспект")
        return summary

    chunks = _split_text_for_llm(transcript)
    logger.info("конспект создается по частям, частей: %d", len(chunks))

    parts = checklist_parts or []
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        part_checklist = parts[i - 1] if i - 1 < len(parts) else checklist
        partials.append(_call_for_chunk(chunk, part_checklist, index=i, total=len(chunks)))

    summary = _join_summaries(partials)
    logger.info("склеили %d частей конспекта, символов: %d", len(partials), len(summary))
    return summary


_METADATA_RE = re.compile(
    r"<<<RAG_METADATA_START>>>\s*(.*?)\s*<<<RAG_METADATA_END>>>", re.DOTALL,
)
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _strip_front_matter(text: str) -> str:
    """убрать заголовок документа и оглавление из части конспекта."""
    lines = text.strip().splitlines()
    result: list[str] = []
    skipping = True

    for line in lines:
        if skipping:
            stripped = line.strip()
            if not stripped or stripped.startswith(">") or stripped.startswith("# "):
                continue
            if stripped in ("---", "___"):
                continue
            skipping = False
        result.append(line)

    return "\n".join(result).strip()


def _merge_metadata_blocks(blocks: list[str]) -> str | None:
    """свести служебные блоки частей в один без потери значений."""
    merged: dict = {}
    parsed_any = False

    for block in blocks:
        try:
            data = json.loads(block)
        except (ValueError, TypeError):
            continue
        items = data.get("rag_metadata")
        if not isinstance(items, list):
            continue
        parsed_any = True
        for item in items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if isinstance(value, list):
                    bucket = merged.setdefault(key, [])
                    for v in value:
                        if v not in bucket:
                            bucket.append(v)
                elif key not in merged:
                    merged[key] = value

    if not parsed_any:
        return None

    payload = json.dumps({"rag_metadata": [merged]}, ensure_ascii=False, indent=2)
    return f"<<<RAG_METADATA_START>>>\n{payload}\n<<<RAG_METADATA_END>>>"


def _join_summaries(partials: list[str]) -> str:
    """соединить частичные конспекты без обращения к llm."""
    title = None
    bodies: list[str] = []
    metadata_blocks: list[str] = []

    for part in partials:
        if title is None:
            found = _TITLE_RE.search(part)
            if found:
                title = found.group(1).strip()

        block = _METADATA_RE.search(part)
        if block:
            metadata_blocks.append(block.group(1))
            part = _METADATA_RE.sub("", part)

        body = _strip_front_matter(part)
        if body:
            bodies.append(body)

    body = "\n\n".join(bodies)

    sections = _SECTION_RE.findall(body)
    parts_out: list[str] = []
    if title:
        parts_out.append(f"# {title}")
    if sections:
        toc = ["> [!summary] Оглавление лекции"]
        toc.extend(f"> {i}. {name.strip()}" for i, name in enumerate(sections, start=1))
        parts_out.append("\n".join(toc))
    parts_out.append(body)

    metadata = _merge_metadata_blocks(metadata_blocks)
    if metadata:
        parts_out.append(metadata)

    return "\n\n".join(parts_out).strip()


def _free_gpu() -> None:
    """освободить видеопамять после выгрузки модели."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_remove(path: str) -> None:
    """удалить временный файл."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
