from __future__ import annotations

from typing import List, Tuple

from app.config import settings
from app.pipeline.embedder import Embedder
from app.pipeline.store import VectorStore


def retrieve(query: str, top_k: int | None = None) -> List[Tuple[dict, float]]:
    """
    쿼리를 임베딩 후 FAISS Top-K 검색 + 임계값 필터링.

    Returns:
        List of (metadata_dict, cosine_score) sorted by score desc.
        관련 청크가 없으면 빈 리스트 반환.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k

    embedder = Embedder.get()
    query_vector = embedder.embed_one(query)

    store = VectorStore.get()
    results = store.search(query_vector, top_k=k)

    if results:
        top_score = results[0][1]
        cutoff = top_score - settings.retrieval_score_gap
        results = [(meta, score) for meta, score in results if score >= cutoff]

    return results
