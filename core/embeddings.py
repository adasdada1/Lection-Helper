"""создает векторы через bge-m3."""

import logging
import torch

logger = logging.getLogger(__name__)

_embedder = None


class BGEEmbedder:
    """создает векторы через BGE-M3."""

    DIMENSION = 1024  # размер вектора

    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        logger.info(
            "загружаем bge-m3 (fp16=%s, устройство=%s)…",
            use_gpu, device,
        )
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_gpu, device=device)
        logger.info("загрузили модель BGE-M3")

    def embed_documents(
        self, texts: list[str], batch_size: int = 4,
    ) -> list[list[float]]:
        """создать векторы текстов."""
        if not texts:
            return []
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        """создать вектор запроса."""
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"][0].tolist()


def get_embedder() -> BGEEmbedder:
    """загрузить bge-m3 при первом вызове."""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder
