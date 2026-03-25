from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings


class VectorStore:
    """
    FAISS IndexFlatIP 기반 벡터 스토어.
    - L2 정규화된 벡터 → IndexFlatIP = 코사인 유사도
    - 메타데이터는 JSON 파일에 별도 영속화
    - 증분 추가(add) 및 문서 단위 삭제 지원
    """

    _instance: "VectorStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(settings.embedding_dim)
        self._metadata: List[dict] = []
        self._load_if_exists()

    @classmethod
    def get(cls) -> "VectorStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        index_path = settings.faiss_index_path
        meta_path = settings.metadata_store_path

        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def save(self) -> None:
        settings.data_index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(settings.faiss_index_path))
        with open(settings.metadata_store_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, metadata_list: List[dict]) -> None:
        """청크 벡터와 메타데이터를 인덱스에 추가."""
        assert vectors.shape[0] == len(metadata_list)
        with self._lock:
            self._index.add(vectors)
            self._metadata.extend(metadata_list)
        self.save()

    def remove_by_document_id(self, document_id: str) -> int:
        """
        document_id에 해당하는 모든 청크를 삭제 후 인덱스 재빌드.
        Returns: 삭제된 청크 수
        """
        with self._lock:
            keep_indices = [
                i for i, m in enumerate(self._metadata)
                if m.get("document_id") != document_id
            ]
            removed = len(self._metadata) - len(keep_indices)
            if removed == 0:
                return 0

            kept_meta = [self._metadata[i] for i in keep_indices]

            if keep_indices:
                all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                kept_vectors = all_vectors[keep_indices].astype(np.float32)
                new_index = faiss.IndexFlatIP(settings.embedding_dim)
                new_index.add(kept_vectors)
                self._index = new_index
            else:
                self._index = faiss.IndexFlatIP(settings.embedding_dim)

            self._metadata = kept_meta

        self.save()
        return removed

    def rebuild(self, vectors: np.ndarray, metadata_list: List[dict]) -> None:
        """전체 인덱스를 새 데이터로 교체."""
        with self._lock:
            new_index = faiss.IndexFlatIP(settings.embedding_dim)
            new_index.add(vectors)
            self._index = new_index
            self._metadata = list(metadata_list)
        self.save()

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[dict, float]]:
        """
        Top-K 검색 후 임계값 필터링.

        Returns:
            List of (metadata_dict, score) sorted by score desc.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            results.append((self._metadata[idx], float(score)))

        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    def get_all_metadata(self) -> List[dict]:
        return list(self._metadata)

    def document_exists(self, document_id: str) -> bool:
        return any(m.get("document_id") == document_id for m in self._metadata)
