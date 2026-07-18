"""сортирует найденные фрагменты через jina-reranker-v3. локально."""

import logging
from functools import lru_cache
from typing import Optional

from core.config import RERANKER_BATCH_SIZE

logger = logging.getLogger(__name__)


# загрузка модели ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker():
    """загрузить модель при первом вызовее."""
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"загружаем jinaai/jina-reranker-v3 (построено на qwen3-0.6b, так что около 1,2 гб), устройство {device}")
    
    model = AutoModel.from_pretrained(
        "jinaai/jina-reranker-v3",
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    logger.info("загрузили модель JINA-RERANKER-V3")
    return model


# открытые функции ---------------------------------------------------------------------------

def rerank(
    query: str,
    matches: list[dict],
    top_n: Optional[int] = None,
) -> list[dict]:
    """отсортировать фрагменты и вернуть top_n."""
    if not matches:
        return matches

    model = get_reranker()
    documents = [m["text"] for m in matches]

    effective_top_n = top_n if top_n is not None else len(matches)
    
    results = []
    batch_size = RERANKER_BATCH_SIZE
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_res = model.rerank(query, batch, top_n=len(batch))
        for r in batch_res:
            r["index"] += i
            results.append(r)
            
    results.sort(key=lambda x: float(x["relevance_score"]), reverse=True)
    results = results[:effective_top_n]

    reranked = []
    for r in results:
        match = matches[r["index"]]
        match["rerank_score"] = round(float(r["relevance_score"]), 4)
        reranked.append(match)

    logger.debug(
        "обычное ранжирование: получено %d, возвращено %d, top_n=%s",
        len(matches), len(reranked), top_n,
    )
    return reranked


def rerank_balanced(
    query: str,
    matches: list[dict],
    top_n_summary: int = 4,
    top_n_transcript: int = 5,
) -> list[dict]:
    """отдельно отсортировать конспект и транскрипт."""
    if not matches:
        return matches

    model = get_reranker()

    # разделить конспект и транскрипт
    summary_matches    = [m for m in matches if m.get("metadata", {}).get("content_kind") == "summary"]
    transcript_matches = [m for m in matches if m.get("metadata", {}).get("content_kind") != "summary"]

    logger.debug(
        "раздельное ранжирование: %d summary и %d transcript",
        len(summary_matches), len(transcript_matches),
    )

    def _rerank_group(group: list[dict], top_n: int) -> list[dict]:
        """отсортировать одну группу."""
        if not group:
            return []

        documents = [m["text"] for m in group]
        effective_top_n = min(top_n, len(group))
        
        results = []
        batch_size = RERANKER_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_res = model.rerank(query, batch, top_n=len(batch))
            for r in batch_res:
                r["index"] += i
                results.append(r)
                
        results.sort(key=lambda x: float(x["relevance_score"]), reverse=True)
        results = results[:effective_top_n]

        out = []
        for r in results:
            match = group[r["index"]]
            match["rerank_score"] = round(float(r["relevance_score"]), 4)
            out.append(match)
        return out

    reranked_summary    = _rerank_group(summary_matches,    top_n_summary)
    reranked_transcript = _rerank_group(transcript_matches, top_n_transcript)

    combined = reranked_summary + reranked_transcript

    logger.info(
        "после ранжирования: %d summary и %d transcript, всего %d фрагментов для llm",
        len(reranked_summary), len(reranked_transcript), len(combined),
    )
    return combined