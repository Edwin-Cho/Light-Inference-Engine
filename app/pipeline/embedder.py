from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """
    BAAI/bge-small-en-v1.5 로컬 임베딩 모델.
    FAISS IndexFlatIP와 코사인 유사도 사용을 위해 L2 정규화 적용.
    """

    _instance: "Embedder | None" = None

    def __init__(self) -> None:
        self._model = SentenceTransformer(
            settings.embedding_model_id,
            device="cpu",
        )

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 후 L2 정규화하여 반환.

        Returns:
            np.ndarray of shape (N, embedding_dim), dtype float32
        """
        vectors = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩. shape: (1, embedding_dim)"""
        return self.embed([text])
