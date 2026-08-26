"""api и веб-интерфейс приложения."""

import os
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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
from core import course_store, doc_store, chat_memory, lexical_store  # noqa: E402,F401

job_manager = IngestJobManager()

os.makedirs(UPLOAD_DIR, exist_ok=True)
course_store.ensure_default_course()


# модели pydantic ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    chat_id: str = Field(..., description="Целевой ID чата.")
    scope_doc_id: str | None = Field(None, description="Искать только в этом материале.")


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class CreateChatRequest(BaseModel):
    course_id: str | None = Field(None, description="Курс, которому принадлежит чат.")


class CreateCourseRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(None, max_length=500)


class RenameCourseRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


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


@app.post("/api/courses")
async def create_course(req: CreateCourseRequest):
    """создать курс."""
    return course_store.create_course(req.title, req.description)


@app.get("/api/courses")
async def list_courses():
    """получить список курсов."""
    return {"courses": course_store.list_courses()}


@app.get("/api/courses/{course_id}")
async def get_course(course_id: str):
    """получить курс."""
    course = course_store.get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Курс не найден")
    return {"course": course}


@app.patch("/api/courses/{course_id}")
async def rename_course(course_id: str, req: RenameCourseRequest):
    """переименовать курс."""
    if not course_store.rename_course(course_id, req.title):
        raise HTTPException(status_code=404, detail="Курс не найден")
    return {"course_id": course_id, "title": req.title}


@app.delete("/api/courses/{course_id}")
async def delete_course(course_id: str):
    """удалить курс со всеми материалами и чатами."""
    if not course_store.get_course(course_id):
        raise HTTPException(status_code=404, detail="Курс не найден")

    from core.vector_store import delete_document as _delete_vec
    from core.lexical_store import delete_document as _delete_lex
    from core.doc_store import delete_document as _delete_sql
    from core.chat_memory import delete_chat as _delete_chat

    doc_ids = course_store.list_course_document_ids(course_id)
    for doc_id in doc_ids:
        _delete_vec(doc_id)
        _delete_lex(doc_id)
        _delete_sql(doc_id)

    chat_ids = course_store.list_course_chat_ids(course_id)
    for chat_id in chat_ids:
        _delete_chat(chat_id)

    course_store.delete_course(course_id)
    return {
        "deleted": True,
        "course_id": course_id,
        "documents_deleted": len(doc_ids),
        "chats_deleted": len(chat_ids),
    }


@app.get("/api/courses/{course_id}/documents")
async def list_course_documents(course_id: str):
    """получить материалы курса."""
    if not course_store.get_course(course_id):
        raise HTTPException(status_code=404, detail="Курс не найден")
    return {"documents": doc_store.list_documents(course_id=course_id)}


@app.get("/api/courses/{course_id}/chats")
async def list_course_chats(course_id: str):
    """получить чаты курса."""
    if not course_store.get_course(course_id):
        raise HTTPException(status_code=404, detail="Курс не найден")
    return {"chats": chat_memory.list_chats(course_id=course_id)}


# чаты ---------------------------------------------------------------------------

@app.post("/api/chats")
async def create_chat(req: CreateChatRequest | None = None):
    """создать чат."""
    from core.chat_memory import create_chat

    course_id = req.course_id if req else None
    if course_id and not course_store.get_course(course_id):
        raise HTTPException(status_code=404, detail="Курс не найден")

    chat = create_chat(course_id=course_id)
    return chat


@app.get("/api/chats")
async def list_chats(course_id: str | None = None):
    """получить список чатов."""
    from core.chat_memory import list_chats as _list
    return {"chats": _list(course_id=course_id)}


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
        result = await ask(
            req.message,
            chat_id=req.chat_id,
            scope_doc_id=req.scope_doc_id,
        )
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
    course_id: str | None = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """загрузить файл и начать обработку."""
    if course_id and not course_store.get_course(course_id):
        raise HTTPException(status_code=404, detail="Курс не найден")

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

    job_id = job_manager.create_job(file.filename, course_id=course_id)
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
async def list_documents(course_id: str | None = None):
    """получить список лекций."""
    from core.doc_store import list_documents as _list
    return {"documents": _list(course_id=course_id)}


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
