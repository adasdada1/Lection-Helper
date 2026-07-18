"""api и веб-интерфейс приложения."""

import os
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lection Helper")
app.mount("/static", StaticFiles(directory="static"), name="static")

# фоновые задачи загрузки ---------------------------------------------------------------------------
from core.config import ALLOWED_EXTENSIONS, UPLOAD_DIR  # noqa: E402
from core.ingestion import IngestJobManager  # noqa: E402

job_manager = IngestJobManager()

os.makedirs(UPLOAD_DIR, exist_ok=True)


# модели pydantic ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    chat_id: str = Field(..., description="Целевой ID чата.")


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


# фронтенд ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """отдать интерфейс."""
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Фронтенд не найден.")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# чаты ---------------------------------------------------------------------------

@app.post("/api/chats")
async def create_chat():
    """создать чат."""
    from core.chat_memory import create_chat
    chat = create_chat()
    return chat


@app.get("/api/chats")
async def list_chats():
    """получить список чатов."""
    from core.chat_memory import list_chats as _list
    return {"chats": _list()}


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    """получить сообщения чата."""
    from core.chat_memory import get_chat, get_messages

    chat = get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат не найден")

    messages = get_messages(chat_id)
    return {
        "chat_id": chat_id,
        "title": chat["title"],
        "messages": messages,
    }


@app.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, req: RenameChatRequest):
    """переименовать чат."""
    from core.chat_memory import rename_chat as _rename

    if not _rename(chat_id, req.title):
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"chat_id": chat_id, "title": req.title}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """удалить чат"""
    from core.chat_memory import delete_chat as _delete

    if not _delete(chat_id):
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"deleted": True, "chat_id": chat_id}


# сообщения ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """отправить сообщение в чат."""
    from core.rag import ask
    from core.chat_memory import get_chat

    # проверить чат
    if not get_chat(req.chat_id):
        raise HTTPException(status_code=404, detail="Чат не найден")

    try:
        result = await ask(req.message, chat_id=req.chat_id)
        # вернуть обновленный заголовок
        chat_data = get_chat(req.chat_id)
        result["title"] = chat_data["title"] if chat_data else None
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# загрузка ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """загрузить файл и начать обработку."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемый формат файла ({ext}). "
                f"Допустимые: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    safe_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    job_id = job_manager.create_job(file.filename)
    background_tasks.add_task(job_manager.run, job_id, file_path)

    return {"job_id": job_id, "message": "Обработка запущена"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """получить статус обработки."""
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return status


@app.get("/api/documents")
async def list_documents():
    """получить список лекций."""
    from core.doc_store import list_documents as _list
    return {"documents": _list()}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """получить лекцию."""
    from core.doc_store import get_document as _get
    doc = _get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"document": doc}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """удалить лекцию из всех хранилищ."""
    from core.vector_store import delete_document as _delete_vec
    from core.lexical_store import delete_document as _delete_lex
    from core.doc_store import delete_document as _delete_sql
    
    vec_deleted = _delete_vec(doc_id)
    lex_deleted = _delete_lex(doc_id)
    sql_deleted = _delete_sql(doc_id)
    
    if not sql_deleted and vec_deleted == 0 and lex_deleted == 0:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"deleted": True, "doc_id": doc_id, "chunks_deleted": vec_deleted, "lexical_chunks_deleted": lex_deleted}
