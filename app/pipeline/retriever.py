"""Retrieval layer for OnDevice Scholar RAG.

Retrieval strategy (2-pass):
    1st pass: ``retrieval_score_threshold`` (0.30) — high-confidence results only
    2nd pass: ``retrieval_fallback_score_threshold`` (0.20) — fallback for vocabulary-mismatch queries

Post-retrieval filtering:
    - Score-gap pruning: drops chunks whose score is more than ``retrieval_score_gap`` (0.25)
      below the top result, reducing noise while keeping contextually related chunks.

P15 — Comparison queries (A vs B) trigger sub-retrieval for each side independently,
    then merge results to guarantee both entities appear in the context.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app.config import settings
from app.pipeline.embedder import Embedder
from app.pipeline.store import VectorStore


def retrieve(query: str, top_k: int | None = None) -> List[Tuple[dict, float]]:
    """Embed the query and search FAISS with 2-pass threshold + score-gap pruning.

    Pass 1: uses ``retrieval_score_threshold`` (0.30) for high-confidence hits.
    Pass 2: relaxes threshold to ``settings.retrieval_fallback_score_threshold`` (0.20) if pass 1
        returns nothing — rescues queries with vocabulary mismatch.
    Final topic-relevance judgment is delegated to the LLM's Rule 3 in SYSTEM_PROMPT.

    Args:
        query: Raw user question string.
        top_k: Number of candidates to retrieve. Defaults to ``settings.retrieval_top_k``.

    Returns:
        List of (metadata_dict, cosine_score) sorted by descending score.
        Returns an empty list if no chunks meet the threshold.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k

    # 쿼리 텍스트 → 384차원 임베딩 벡터 (BAAI/bge-small-en-v1.5)
    embedder = Embedder.get()
    query_vector = embedder.embed_one(query)

    store = VectorStore.get()
    # 1st pass: 기본 임계값(0.30)으로 검색
    results = store.search(query_vector, top_k=k)

    if not results:
        # 2nd pass: 결과 없으면 config의 완화된 임계값으로 재시도 (0.20, 하드코딩 제거)
        results = store.search(query_vector, top_k=k, score_threshold=settings.retrieval_fallback_score_threshold)

    if results:
        # Score-gap 프루닝: top 점수 - retrieval_score_gap(0.25) 이하 청크 제거
        # 이유: 최상위 청크와 너무 차이나는 청크는 맥락 혼재 원인이 됨
        top_score = results[0][1]
        cutoff = top_score - settings.retrieval_score_gap
        results = [(meta, score) for meta, score in results if score >= cutoff]

    return results


# ── P15: Comparison Query Sub-Retrieval ───────────────────────────────────────
# 목적: "A vs B" 형태의 비교 쿼리는 단순 Top-K로는 한쪽 entity만 retrieve될 수 있음
# 해결: 양측을 각각 독립적으로 검색하여 결과를 병합 → 양측 문서 모두 컨텍스트 포함 보장

# 비교 쿼리를 감지하는 키워드 패턴
# 예: "BERT vs GPT", "difference between A and B", "how does X compare to Y"
# Bug fix: \w+ → .+? — \w+는 단일 단어만 매치 → "full fine-tuning" 같은 다단어 entity를 미탐지하여 P15 전체 우회
_COMPARISON_KW_RE = re.compile(
    r'\b(?:vs\.?|versus|compared?\s+(?:to|with)|difference\s+between'
    r'|how\s+does\s+.+?\s+(?:differ|compare))\b',
    re.IGNORECASE,
)

# "vs", "versus", "compared to" 등으로 두 피비교 대상 분리
_COMPARISON_SPLIT_RE = re.compile(
    r'\s+(?:vs\.?|versus|compared?\s+(?:to|with))\s+',
    re.IGNORECASE,
)

# "between A and B" 패턴에서 A, B 추출
# Bug fix: side_b에서 "and" 이후 잔류 entity 제거 (3항 비교 오파싱 방지)
# 예: "BERT and GPT and T5" → side_b = "GPT" (⨉"GPT and T5")
_BETWEEN_AND_RE = re.compile(
    r'\bbetween\s+(.+?)\s+and\s+((?:(?!\band\b).)+)',
    re.IGNORECASE,
)

# 패턴 2(A vs B) 분리 시 side_a에 재위하는 질문 접두어 제거
# 예: "How does LoRA compare to ..." → parts[0] = "How does LoRA" → 접두어가 임베딩 품질 저하
_QUESTION_STEM_RE = re.compile(
    r'^(?:how\s+does|what\s+is\s+the\s+difference\s+between'
    r'|compare|tell\s+me\s+about)\s+',
    re.IGNORECASE,
)


def _strip_question_stem(s: str) -> str:
    """Remove interrogative prefix from a comparison side string.

    Ensures that ``side_a`` from pattern-2 splits does not carry
    question stems like "How does" that degrade embedding quality.

    Args:
        s: Candidate entity string, possibly prefixed with a question stem.

    Returns:
        Cleaned entity string with the prefix removed.
    """
    # "How does LoRA" → "LoRA"를 만듦으로써 retrieve() 호출 시 정확한 entity 임베딩 보장
    return _QUESTION_STEM_RE.sub("", s).strip()


def is_comparison_query(query: str) -> bool:
    """Return True if the query compares two entities.

    Used by the API router to decide whether to call ``retrieve_comparison``
    instead of the standard ``retrieve``.
    """
    return bool(_COMPARISON_KW_RE.search(query))


def _extract_comparison_sides(query: str) -> tuple[str, str] | None:
    """Extract the two entities being compared from a comparison query.

    Tries two patterns in order:
        1. ``between A and B`` form
        2. ``A vs/versus/compared to B`` form

    Args:
        query: User question string.

    Returns:
        ``(side_a, side_b)`` tuple, or ``None`` if extraction fails.
    """
    # 패턴 1: "between A and B"
    m = _BETWEEN_AND_RE.search(query)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip('?.')

    # 패턴 2: "A vs B" / "A compared to B"
    # side_a에 질문 접두어 제거 적용 (예: "How does LoRA" → "LoRA")
    parts = _COMPARISON_SPLIT_RE.split(query, maxsplit=1)
    if len(parts) == 2:
        return _strip_question_stem(parts[0].strip()), parts[1].strip().rstrip('?.')

    return None  # 분해 실패 → retrieve_comparison에서 기본 retrieve로 폴백


def retrieve_comparison(query: str, top_k: int | None = None) -> List[Tuple[dict, float]]:
    """P15: Sub-retrieval for comparison queries (A vs B).

    Splits the query into two sides and retrieves chunks independently for each,
    then merges results to ensure both entities are represented in the context.
    This prevents the standard Top-K from being dominated by one side only.

    Falls back to standard ``retrieve()`` if side extraction fails.

    Args:
        query: User comparison question (e.g. "How does LoRA differ from full fine-tuning?").
        top_k: Total number of results to return after merging.

    Returns:
        Merged and deduplicated list of (metadata_dict, cosine_score),
        sorted by score descending, capped at ``top_k``.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    sides = _extract_comparison_sides(query)

    if not sides:
        # 분해 실패 시 일반 검색으로 폴백
        return retrieve(query, top_k=k)

    side_a, side_b = sides
    # embed_one() 3회 → 2회: main 쿼리 검색 제거, k_each 상향으로 보완
    # main 쿼리 결과는 results_a + results_b와 높은 중복률 → 원가 제한적
    k_each = max(3, (k * 2) // 3)

    # 양측 독립 검색
    results_a = retrieve(side_a, top_k=k_each)
    results_b = retrieve(side_b, top_k=k_each)

    # 중복 청크는 (파일명, 청크 인덱스) 키로 제거, 점수는 최댓값 유지
    seen: dict[tuple, Tuple[dict, float]] = {}
    for meta, score in results_a + results_b:
        key = (meta.get("source_filename", ""), meta.get("chunk_index", 0))
        if key not in seen or score > seen[key][1]:
            seen[key] = (meta, score)

    # score 비대칭 시 한쪽 entity 탈락 방지: 양측 각 k_floor개 우선 보장
    # 예: side_a 점수 [0.85, 0.82] vs side_b [0.55, 0.52] 시도 side_b 콘텍스트 포함 보장
    k_floor = max(2, k // 4)
    priority_keys: set[tuple] = set()
    for meta, score in results_a[:k_floor] + results_b[:k_floor]:
        pk = (meta.get("source_filename", ""), meta.get("chunk_index", 0))
        priority_keys.add(pk)

    guaranteed = sorted(
        [v for ck, v in seen.items() if ck in priority_keys],
        key=lambda x: x[1], reverse=True,
    )
    remainder = sorted(
        [v for ck, v in seen.items() if ck not in priority_keys],
        key=lambda x: x[1], reverse=True,
    )
    return (guaranteed + remainder)[:k]
