"""FAISS-backed vector store for OnDevice Scholar RAG.

Design decisions:
    - ``IndexFlatIP`` (Inner Product) with L2-normalised vectors = cosine similarity.
      Chosen over ``IndexFlatL2`` because cosine similarity is more robust to
      embedding magnitude differences across chunking strategies.
    - Metadata (source_filename, page_number, text, etc.) is stored separately in
      a JSON sidecar file rather than inside FAISS, since FAISS only manages vectors.
    - Thread-safety: a single ``threading.Lock`` guards all write operations so the
      FastAPI server can ingest documents concurrently without index corruption.
    - Singleton pattern: one ``VectorStore`` instance is shared across all requests.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings


class VectorStore:
    """FAISS IndexFlatIP-based vector store with JSON metadata persistence.

    Key properties:
        - L2-normalised vectors + IndexFlatIP = cosine similarity search
        - Metadata (text, source, page, etc.) stored in a JSON sidecar file
        - Supports incremental ``add`` and per-document ``remove_by_document_id``
        - Thread-safe writes via a class-level ``threading.Lock``
        - Singleton: use ``VectorStore.get()`` to get the shared instance
    """

    # 싱글턴 인스턴스 — 서버 전체에서 FAISS 인덱스를 한 번만 로드
    _instance: "VectorStore | None" = None
    # 원자적 쓰기 보호: 동시 진단 중 인덱스 손상 방지
    _lock = threading.Lock()

    def __init__(self) -> None:
        # 384차원 Inner Product 인덱스 초기화 (bge-small-en-v1.5 임베딩 차원)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(settings.embedding_dim)
        self._metadata: List[dict] = []  # FAISS 인덱스와 1:1 매청되는 메타데이터 리스트
        self._load_if_exists()  # 디스크에 저장된 인덱스가 있으면 자동 로드

    @classmethod
    def get(cls) -> "VectorStore":
        """Return the shared VectorStore instance, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ── Persistence ──────────────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        """Load FAISS index and metadata from disk if both files exist."""
        index_path = settings.faiss_index_path
        meta_path = settings.metadata_store_path

        if index_path.exists() and meta_path.exists():
            # FAISS 바이너리 파일과 JSON 메타데이터를 동시 로드
            # 두 파일이 모두 있어야만 로드: 한쪽만 있으면 인덱스-메타 불일치 발생
            loaded_index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded_meta = json.load(f)
            # Flaw fix: ntotal ≠ len(metadata) 시 파일 손상 또는 불완전 저장 → 빈 인덱스로 초기화
            if loaded_index.ntotal != len(loaded_meta):
                return  # 불일치 감지: 부패 방지를 위해 다시 추론하지 않고 빈 상태로 남될
            self._index = loaded_index
            self._metadata = loaded_meta

    def _save_unlocked(self) -> None:
        """Write index and metadata to disk. **Caller must already hold** ``_lock``.

        Separated from the public ``save()`` to allow lock-holding callers
        (``add``, ``remove_by_document_id``, ``rebuild``) to persist atomically
        without releasing the lock between write and save.
        """
        settings.data_index_dir.mkdir(parents=True, exist_ok=True)
        # FAISS 인덱스 바이너리 저장
        faiss.write_index(self._index, str(settings.faiss_index_path))
        # 메타데이터 JSON 저장 (ensure_ascii=False: 한글 제목 등 보존)
        with open(settings.metadata_store_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        """Persist the current FAISS index and metadata to disk (thread-safe).

        Acquires the write lock before writing so the on-disk snapshot is
        always consistent with the in-memory state at the moment of the call.
        """
        # lock 획득 후 저장: 저장 도중 다른 스레드의 상태 변경 차단
        with self._lock:
            self._save_unlocked()

    # ── Write ────────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, metadata_list: List[dict]) -> None:
        """Add new chunk vectors and their metadata to the index.

        Args:
            vectors: Float32 ndarray of shape ``(n, embedding_dim)``, L2-normalised.
            metadata_list: List of ``n`` metadata dicts corresponding to each vector.

        Raises:
            ValueError: If ``vectors`` and ``metadata_list`` have different lengths.
        """
        # Bug fix: assert → raise ValueError
        # python -O (실행 최적화) 모드에서 assert는 비활성화 → 불일치 감지 불가
        if vectors.shape[0] != len(metadata_list):
            raise ValueError(
                f"vectors count ({vectors.shape[0]}) ≠ metadata count ({len(metadata_list)})"
            )
        with self._lock:
            self._index.add(vectors)  # FAISS는 벡터만 관리
            self._metadata.extend(metadata_list)  # 메타데이터는 Python 리스트로 관리
            # Bug fix: lock 안에서 저장 — lock 바깔에서 save() 호출 시 race condition 발생 가능
            self._save_unlocked()

    def remove_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document and rebuild the index.

        Args:
            document_id: Unique document identifier stored in chunk metadata.

        Returns:
            Number of chunks removed (0 if document was not found).
        """
        with self._lock:
            # 삭제 대상이 아닌 청크들의 인덱스 목록
            keep_indices = [
                i for i, m in enumerate(self._metadata)
                if m.get("document_id") != document_id
            ]
            removed = len(self._metadata) - len(keep_indices)
            if removed == 0:
                return 0  # 해당 문서 없음

            kept_meta = [self._metadata[i] for i in keep_indices]

            if keep_indices:
                # FAISS IndexFlatIP는 in-place 삭제 불가 → 전체 벡터 추출 후 재빌드
                all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                kept_vectors = all_vectors[keep_indices].astype(np.float32)
                new_index = faiss.IndexFlatIP(settings.embedding_dim)
                new_index.add(kept_vectors)
                self._index = new_index
            else:
                # 샘플 전체 삭제 시 빈 인덱스로 초기화
                self._index = faiss.IndexFlatIP(settings.embedding_dim)

            self._metadata = kept_meta
            # Bug fix: lock 안에서 저장 (race condition 방지)
            self._save_unlocked()

        return removed

    def rebuild(self, vectors: np.ndarray, metadata_list: List[dict]) -> None:
        """Replace the entire index with a new set of vectors and metadata.

        Used during full re-ingestion (e.g. after re-chunking all documents).

        Args:
            vectors: Float32 ndarray of shape ``(n, embedding_dim)``.
            metadata_list: List of ``n`` metadata dicts.
        """
        with self._lock:
            new_index = faiss.IndexFlatIP(settings.embedding_dim)
            new_index.add(vectors)
            self._index = new_index
            self._metadata = list(metadata_list)  # 복사본 저장으로 외부 참조 차단
            # Bug fix: lock 안에서 저장 (race condition 방지)
            self._save_unlocked()

    # ── Read ────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[dict, float]]:
        """Search the FAISS index and return top-K results above a score threshold.

        Args:
            query_vector: L2-normalised query embedding of shape ``(1, embedding_dim)``.
            top_k: Maximum number of results to retrieve.
            score_threshold: Minimum cosine similarity score to include a result.
                Defaults to ``settings.retrieval_score_threshold`` (0.30).

        Returns:
            List of (metadata_dict, cosine_score) sorted by score descending.
            Returns an empty list if the index is empty.
        """
        if self._index.ntotal == 0:
            return []  # 인덱스가 비어 있으면 검색 불가

        # Flaw fix: FAISS는 (1, dim) 형태를 요구함
        # (dim,) 형태로 입력되면 드리물게 search가 실패하는 사례 있음
        query_vector = query_vector.reshape(1, -1)

        # 인덱스 전체 청크 수보다 크게 요청하면 FAISS 오류 → min 제한
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        # score_threshold 미지정 시 settings.retrieval_score_threshold(0.30) 사용
        threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS는 결과 부족 시 -1로 선언 → 건너뜀
            if score < threshold:
                continue  # 임계값 미만 청크 제외
            results.append((self._metadata[idx], float(score)))

        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    def get_all_metadata(self) -> List[dict]:
        return list(self._metadata)

    def document_exists(self, document_id: str) -> bool:
        return any(m.get("document_id") == document_id for m in self._metadata)
