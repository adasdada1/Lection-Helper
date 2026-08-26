"""фрагменты лекций в chromadb. данные хранятся на диске."""

import os
import logging
import chromadb

from core.config import CHROMA_COLLECTION_NAME as COLLECTION_NAME, CHROMA_DIR

logger = logging.getLogger(__name__)

_client = None # управление всем хранилищем chromadb
_collection = None # конкретная группа записей lecture_chunks. у нее вызываются всякие add query и т.д


def _get_collection():
    """В общем просто ленивое открытие ChromaDB, установка сравнения для алгоритма HNSW по косинусному сходству, установка db в _client и _collection для быстрого обращения."""
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR) #локальный клиент ChromaDB, который хранит данные на диске. вся работа через него. ДАННЫЕ НЕ УДАЛЯЮТСЯ
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "открыли коллекцию CHROMADB '%s', данные хранятся в %s",
            COLLECTION_NAME, CHROMA_DIR,
        )
    return _collection


# запись ---------------------------------------------------------------------------
def add_chunks(
    doc_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    filename: str,
    source_type: str,
    course_id: str | None = None,
) -> int:
    """сохранить фрагменты и их векторы. возвращает длину"""
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])

        content_kind = chunk.get("content_kind", "transcript")
        meta: dict = {
            "doc_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "content_kind": content_kind,
            "chunk_index": chunk["chunk_index"],
        }
        if course_id:
            meta["course_id"] = course_id

        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "сохранил в chromadb %d фрагментов документа '%s', doc_id=%s",
        len(ids), filename, doc_id,
    )
    return len(ids)


# чтение и поиск ---------------------------------------------------------------------------
def search(
    query_embedding: list[float],
    top_k: int = 5,
    content_kind: str | None = None,
    course_id: str | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """найти ближайшие фрагменты по вектору запроса."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    conditions = []
    if content_kind:
        conditions.append({"content_kind": content_kind})
    if course_id:
        conditions.append({"course_id": course_id})
    if doc_id:
        conditions.append({"doc_id": doc_id})

    if len(conditions) == 1:
        query_kwargs["where"] = conditions[0]
    elif conditions:
        query_kwargs["where"] = {"$and": conditions}

    results = collection.query(**query_kwargs)

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return matches


def list_documents() -> list[dict]:
    """получить документы и число их фрагментов."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    all_data = collection.get(include=["metadatas"])

    docs: dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta["filename"],
                "source_type": meta["source_type"],
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


def backfill_course_id() -> int:
    """проставить курс метаданным фрагментов по их документам."""
    collection = _get_collection()

    if collection.count() == 0:
        return 0

    from core.doc_store import list_documents
    course_by_doc = {
        d["doc_id"]: d.get("course_id")
        for d in list_documents()
        if d.get("course_id")
    }
    if not course_by_doc:
        return 0

    all_data = collection.get(include=["metadatas"])

    ids_to_update = []
    metas_to_update = []
    for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if meta.get("course_id"):
            continue
        course_id = course_by_doc.get(meta.get("doc_id"))
        if not course_id:
            continue
        updated = dict(meta)
        updated["course_id"] = course_id
        ids_to_update.append(chunk_id)
        metas_to_update.append(updated)

    if ids_to_update:
        collection.update(ids=ids_to_update, metadatas=metas_to_update)
        logger.info("проставили course_id у %d фрагментов chromadb", len(ids_to_update))

    return len(ids_to_update)


def delete_document(doc_id: str) -> int:
    """удалить фрагменты документа."""
    collection = _get_collection()

    all_data = collection.get(
        where={"doc_id": doc_id},
        include=[],
    )

    ids_to_delete = all_data["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("удалил из CHROMADB %d фрагментов, doc_id=%s", len(ids_to_delete), doc_id)

    return len(ids_to_delete)
