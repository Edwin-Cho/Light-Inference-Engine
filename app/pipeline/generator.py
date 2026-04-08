"""Answer generation pipeline for OnDevice Scholar RAG.

Responsibilities (in order of execution inside ``Generator.generate``):
    1. Build a context block from retrieved FAISS chunks  (_build_context_block)
    2. Run Qwen2.5-3B-Instruct with a structured system prompt  (Generator._load_model / generate)
    3. P16 — Post-hoc citation injection for uncited sentences  (_inject_citations_post_hoc)
    4. P13 — Tier-2 numeric scrubbing: replace hallucinated numbers with [?]  (_scrub_hallucinated_numerics)
    5. Citation validation & pruning  (_build_citations, _filter_by_contribution)
    6. Hallucination warnings: numeric existence (P13) + metric-label fidelity (P12)  (_check_numeric_existence, _check_metric_fidelity)
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.models.schemas import Citation
from app.pipeline.ingest import _is_noise_header, _extract_arxiv_id, _extract_paper_title

FALLBACK_ANSWER = "No relevant information found in the provided documents."

# 모델이 "모르겠다"고 답할 때 사용하는 표현 목록
# _is_fallback()에서 이 중 하나라도 포함되면 FALLBACK_ANSWER로 대체
_FALLBACK_SUBSTRINGS = (
    "no relevant information found",
    "i don't know",
    "i do not know",
    "cannot find",
    "not found in the provided",
)

SYSTEM_PROMPT = r"""You are a precise academic research assistant. Answer ONLY using the provided context passages.

Rules:
1. Every factual claim MUST be followed immediately by an inline citation: [Source: filename | Section: X | p.N].
   - BAD:  "BERT uses masked language modeling and next sentence prediction."
   - GOOD: "BERT uses two pre-training objectives: masked language modeling (MLM) and next sentence prediction (NSP) [Source: BERT.pdf | Section: 3.1 | p.4]."
2. PRECISION — Never use vague quantifiers when the context gives exact information.
   - BAD:  "BERT is trained on various pre-training tasks."
   - GOOD: "BERT is trained on exactly two pre-training tasks: MLM and NSP."
   - If the context lists exactly N items, say "exactly N" or list them explicitly.
3. If the context contains NO topic-relevant information whatsoever, respond exactly with: "No relevant information found in the provided documents."
   - BAD: Returning this fallback when the context discusses the topic but lacks specific numbers.
   - GOOD: If the context contains qualitative findings (e.g., "Adam outperforms SGD on average") but no exact figures, report the qualitative finding and state that specific numbers are not provided.
   3a. PARTIAL QUESTIONS — If the context covers only SOME sub-parts of a multi-part question, answer the covered parts precisely and explicitly state for the uncovered parts: "[sub-topic] is not reported in the provided documents." Never silently skip a sub-question.
4. Do NOT fabricate facts, authors, or findings not present in the context.
5. SYNTHESIZE the content in your own words — do NOT copy sentences verbatim from the context.
6. Replace all vague pronouns with explicit names: "Our method" → the paper/method name, "We" → "the authors of [paper]", "they" → the actual subject.
7. Be concise. 3–5 sentences is ideal unless the question demands more.
8. MATH — Wrap all mathematical expressions in LaTeX delimiters so they render correctly.
   - Inline math: $O(n^2)$, $d_{\text{model}}$, $\sqrt{d_k}$
   - Block math: $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   - Never write bare: O(n^2), O(n log n), d_model — always wrap in $...$.
9. TABLE / METRIC FIDELITY — When the context contains a table or list of metrics, report each metric using its EXACT name and value as written in the source. NEVER rename, merge, or reassign metric labels.
   - BAD:  "Person Detection: 31.6%" when the source says "Lane Line IoU: 31.6%".
   - GOOD: "Lane Line IoU: 31.6% [Source: ...]". Preserve every column/row header verbatim.
   - If you are unsure which label belongs to a value, quote the source table directly rather than paraphrasing.
10. TIME-BOUNDED CLAIMS — Never assert that a result is "state-of-the-art", "best", or "superior" without anchoring it to the paper's publication context.
    - BAD:  "HybridNets achieves state-of-the-art performance on BDD100K."
    - GOOD: "At the time of publication, HybridNets reported state-of-the-art performance on BDD100K [Source: ...]."
    - If the context does not mention a publication date or comparison scope, omit the SOTA claim entirely.
11. VERBATIM NUMERICS — Never insert a specific number (percentage, score, count) unless that exact value appears verbatim in the retrieved context.
    - BAD:  "The average improvement is 89.33%" when the context only says "outperforms on average".
    - GOOD: "The context states that Adam pre-training outperforms SGD on average [Source: ...] but provides no specific aggregate figure."
    - If the context gives a trend/direction without an exact figure, describe the trend only.
12. NUMERIC ABSTENTION — When you want to include a precise figure but cannot find it verbatim in the retrieved passages, explicitly state the absence rather than estimating or rounding.
    - BAD:  "LoRA reduces trainable parameters by approximately 60–70% compared to full fine-tuning." ← estimation not grounded in context
    - GOOD: "LoRA drastically reduces the number of trainable parameters by injecting low-rank matrices [Source: LoRA.pdf | Section: 3 | p.4], but the exact reduction percentage is not reported in the retrieved passages."
13. COMPARISON COMPLETENESS — When a question compares two entities (A vs B), if retrieved context covers only one side, explicitly note the missing side rather than answering only about the covered entity.
    - BAD:  Answering only about LoRA when asked "LoRA vs full fine-tuning".
    - GOOD: Answer what the context provides about each side; for the uncovered side write "[entity] details are not present in the retrieved documents." """


def _build_context_block(retrieved: List[Tuple[dict, float]]) -> str:
    """Format retrieved chunks into a single context string for the LLM prompt.

    Each chunk is prefixed with its source metadata (filename, section, page, score)
    so the model can generate grounded [Source: ...] citations.

    Args:
        retrieved: List of (metadata_dict, cosine_score) from the vector store.

    Returns:
        A single string with chunks separated by ``---`` dividers.
    """
    blocks = []
    for meta, score in retrieved:
        filename = meta.get("source_filename", "unknown")
        page = meta.get("page_number", "?")
        raw_header = meta.get("section_header")
        # 노이즈성 헤더(e.g. 'References', 빈 문자열)는 '—'로 대체
        section = (raw_header if raw_header and not _is_noise_header(raw_header) else None) or "—"
        text = meta.get("text", "")
        blocks.append(
            f"[Source: {filename} | Section: {section} | p.{page} | score: {score:.3f}]\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def _is_fallback(answer: str) -> bool:
    """모델이 답변 불가 판정을 했는지 확인."""
    lower = answer.lower()
    return any(kw in lower for kw in _FALLBACK_SUBSTRINGS)


# LLM 답변 내 [Source: filename | ...] 패턴에서 파일명만 추출하는 정규식
_SOURCE_PATTERN = re.compile(r'\[Source:\s*([^|\]]+?)\s*\|', re.IGNORECASE)


def _extract_cited_sources(answer: str) -> set[str]:
    """
    LLM 답변 텍스트에서 [Source: filename | ...] 패턴으로 인용된 파일명 추출.
    """
    return {m.group(1).strip() for m in _SOURCE_PATTERN.finditer(answer)}


# P12/P13 경고 시스템에서 사용: "XX.X%" 형태의 퍼센트 숫자만 추출
# 예: "achieves 92.3% accuracy" → "92.3"
_METRIC_NUM_RE = re.compile(r'\b(\d+\.?\d*)\s*%')

# 퍼센트 앞 문맥 분석 시 의미 없는 일반 단어 제거용 불용어
_METRIC_LABEL_STOPWORDS = frozenset({
    'the', 'for', 'its', 'with', 'and', 'is', 'are', 'was', 'has',
    'achieves', 'score', 'rate', 'value', 'result', 'performance',
    'percentage', 'point', 'reaches', 'obtains', 'shows', 'gets',
})

# "진짜 메트릭 이름"을 식별하는 지표 단어 목록
# 이 중 하나라도 포함되어 있어야 label mismatch 경고 발생 → false positive 방지
_METRIC_INDICATOR_WORDS = frozenset({
    'accuracy', 'acc', 'precision', 'recall', 'f1', 'bleu', 'rouge',
    'top1', 'top5', 'map', 'ndcg', 'mrr', 'perplexity', 'wer', 'cer',
    'flop', 'flops', 'param', 'params', 'latency', 'throughput', 'speedup',
    'reduction', 'improvement', 'compression', 'trainable', 'parameters',
    'benchmark', 'error', 'loss', 'gain', 'drop', 'delta', 'baseline',
})


def _looks_like_metric_label(label: str) -> bool:
    """Return True only if label contains at least one metric-specific indicator word.
    Filters out garbled OCR text or generic English phrases.
    """
    return bool(set(label.lower().split()) & _METRIC_INDICATOR_WORDS)


def _metric_label_context(text: str, window: int = 60) -> dict[str, list[str]]:
    """Extract label context for each percentage figure found in text.

    For each ``XX.X%`` match, captures up to ``window`` characters before it
    and returns the last 5 meaningful words as a label sequence.

    Args:
        text: Raw text to analyze (answer or chunk).
        window: Character window size before the number to inspect.

    Returns:
        Mapping of ``{num_str: [label_word_sequences]}``.
    """
    result: dict[str, list[str]] = {}
    for m in _METRIC_NUM_RE.finditer(text):
        num = m.group(1)
        # 숫자 앞 최대 window 글자를 잘라서 컨텍스트로 사용
        start = max(0, m.start() - window)
        ctx = text[start:m.start()]
        words = [
            w for w in re.findall(r'[A-Za-z]{3,}', ctx.lower())
            if w not in _METRIC_LABEL_STOPWORDS
        ]
        if words:
            result.setdefault(num, []).append(' '.join(words[-5:]))  # 마지막 5개 의미 단어만 유지
    return result


def _check_numeric_existence(
    answer: str,
    retrieved: List[Tuple[dict, float]],
) -> list[str]:
    """P13: Check whether percentage figures in the answer exist verbatim in context.

    Any ``XX.X%`` value found in the answer but absent from all retrieved chunks
    is flagged as a potentially hallucinated figure.

    Args:
        answer: The LLM-generated answer string.
        retrieved: List of (metadata_dict, score) retrieved from FAISS.

    Returns:
        List of warning strings (empty if no hallucination detected).
    """
    # 답변에서 퍼센트 숫자 집합 추출
    answer_nums = {m.group(1) for m in _METRIC_NUM_RE.finditer(answer)}
    if not answer_nums:
        return []

    # 전체 retrieved 청크에서 퍼센트 숫자 집합 추출
    chunk_nums: set[str] = set()
    for meta, _ in retrieved:
        for m in _METRIC_NUM_RE.finditer(meta.get("text", "")):
            chunk_nums.add(m.group(1))

    # 답변에만 있고 청크에는 없는 숫자 = 할루시네이션 의심
    hallucinated = sorted(answer_nums - chunk_nums, key=float)
    if not hallucinated:
        return []
    return [
        f"Numeric hallucination suspected: {', '.join(f'{n}%' for n in hallucinated)} "
        f"not found in retrieved context"
    ]


def _check_metric_fidelity(
    answer: str,
    retrieved: List[Tuple[dict, float]],
) -> list[str]:
    """P12: Detect metric label-value mismatches between answer and source context.

    For each ``XX.X%`` in the answer, compares the surrounding label words
    against the same number's context in the retrieved chunks.
    Zero word-overlap means the model assigned the figure to a wrong metric label.

    Args:
        answer: The LLM-generated answer string.
        retrieved: List of (metadata_dict, score) retrieved from FAISS.

    Returns:
        List of warning strings describing each mismatch (empty if clean).
    """
    # 답변의 각 퍼센트 숫자 → 주변 레이블 단어 매핑
    answer_map = _metric_label_context(answer)
    if not answer_map:
        return []

    # retrieved 청크의 각 퍼센트 숫자 → 주변 레이블 단어 매핑 (ground truth)
    source_map: dict[str, list[str]] = {}
    for meta, _ in retrieved:
        for num, labels in _metric_label_context(meta.get("text", "")).items():
            source_map.setdefault(num, []).extend(labels)

    warnings: list[str] = []
    for num, answer_labels in answer_map.items():
        if num not in source_map:
            continue  # 청크에 없는 숫자는 P13에서 처리, 여기서는 레이블 불일치만 검사

        # 청크 레이블의 모든 단어를 하나의 집합으로 합침
        source_words: set[str] = set()
        for lbl in source_map[num]:
            source_words.update(lbl.split())

        for a_lbl in answer_labels:
            if not _looks_like_metric_label(a_lbl):
                continue  # OCR 잡음이나 일반 문구 → 메트릭 레이블 아님, 건너뜀
            a_words = set(a_lbl.split())
            if a_words and not (a_words & source_words):
                # 답변 레이블과 청크 레이블이 단어 교집합 없음 → 레이블 혼동 의심
                best_src = source_map[num][0] if source_map[num] else "unknown"
                warnings.append(
                    f"Metric label mismatch for {num}%: "
                    f"answer uses '{a_lbl}' but source context shows '{best_src}'"
                )

    return warnings


_CONTRIBUTION_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "are", "from", "have",
    "been", "which", "their", "they", "also", "more", "such", "into",
    "when", "than", "its", "was", "were", "can", "will", "use", "used",
    "using", "each", "both", "very", "well", "may", "paper", "model",
    "method", "approach", "show", "propose", "work", "authors",
})


def _extract_keywords(text: str) -> frozenset[str]:
    """4글자 이상 알파벳 단어에서 contribution stopword 제거 후 반환."""
    words = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
    return frozenset(w for w in words if len(w) >= 4 and w not in _CONTRIBUTION_STOPWORDS)


def _filter_by_contribution(
    answer: str,
    citations: List[Citation],
    retrieved: List[Tuple[dict, float]],
    min_overlap: int = 2,
) -> List[Citation]:
    """P11: Remove citations whose source chunk has low keyword overlap with the answer.

    Prevents low-relevance chunks from appearing as citations even if they
    technically passed the score threshold.

    Args:
        answer: The LLM-generated answer string.
        citations: Candidate citations to filter.
        retrieved: Full list of retrieved (metadata, score) pairs.
        min_overlap: Minimum number of shared keywords required to keep a citation.

    Returns:
        Filtered citation list. Falls back to the original list if all are filtered out.
    """
    answer_kw = _extract_keywords(answer)
    # 답변이 너무 짧으면 키워드 기반 필터 신뢰도 낮음 → 필터 건너뜀
    if len(answer_kw) < min_overlap * 3:
        return citations

    # 파일명 → 모든 청크 텍스트 목록 매핑 (첫 청크만 참조하면 이후 청크의 키워드 누락)
    chunk_map: dict[str, list[str]] = {}
    for meta, _ in retrieved:
        fn = meta.get("source_filename", "")
        if fn:
            chunk_map.setdefault(fn, []).append(meta.get("text", ""))

    # 교집합 키워드 수 >= min_overlap인 citation만 유지 (모든 청크 텍스트 합산)
    filtered = [
        c for c in citations
        if len(answer_kw & _extract_keywords(" ".join(chunk_map.get(c.source_filename, [])))) >= min_overlap
    ]
    # 전부 제거되면 원본 반환 (안전 폴백 — citation 없는 답변 방지)
    return filtered if filtered else citations


_TITLE_STOPWORDS = frozenset({
    'with', 'from', 'that', 'this', 'over', 'using', 'based', 'large',
    'language', 'model', 'models', 'neural', 'deep', 'learning', 'into',
    'beyond', 'towards', 'toward', 'improving', 'improved', 'attention',
})


def _match_citations_by_title(
    answer: str, citations: List[Citation]
) -> List[Citation]:
    """
    [Source:] 패턴 부재 시 fallback: paper_title의 핵심 키워드가
    답변 텍스트에 등장하면 해당 citation 포함.
    """
    answer_lower = answer.lower()
    matched = []
    for c in citations:
        title = c.paper_title or ""
        keywords = [
            w for w in re.findall(r'[A-Za-z]{4,}', title)
            if w.lower() not in _TITLE_STOPWORDS
        ]
        if keywords and any(kw.lower() in answer_lower for kw in keywords):
            matched.append(c)
    return matched


# Flaw 2 fix: 매 generate() 호출마다 PDF 재열기 방지 — 프로세스 수명 동안 제목 캐시 유지
_paper_title_cache: dict[str, str] = {}


def _build_citations(retrieved: List[Tuple[dict, float]]) -> List[Citation]:
    """Build a deduplicated list of Citation objects from retrieved chunks.

    Applies ``citation_min_score`` threshold (default 0.65) to filter noise.
    Deduplicates by filename so each paper appears at most once.
    Falls back to PyMuPDF title extraction if paper_title is missing from metadata.

    Args:
        retrieved: List of (metadata_dict, cosine_score) from the vector store.

    Returns:
        List of Citation objects, one per unique source file above the score threshold.
    """
    seen: set[str] = set()
    citations: List[Citation] = []

    for meta, score in retrieved:
        # citation_min_score(0.65) 미만 청크는 품질이 낮아 citation에서 제외
        if score < settings.citation_min_score:
            continue
        filename = meta.get("source_filename", "")
        if filename and filename not in seen:
            raw_header = meta.get("section_header")
            # 노이즈 헤더(예: 숫자만 있는 헤더, 빈 문자열)는 None 처리
            section_header = None if (raw_header and _is_noise_header(raw_header)) else raw_header
            arxiv_id = meta.get("arxiv_id") or _extract_arxiv_id(filename)
            paper_title = meta.get("paper_title") or ""
            stem_fallback = filename.rsplit(".", 1)[0].replace("_", " ")
            if not paper_title or paper_title == stem_fallback:
                # 메타데이터에 title이 없으면 PDF를 직접 열어 첫 페이지에서 추출 시도
                pdf_path = settings.data_raw_dir / filename
                if pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
                    if filename in _paper_title_cache:
                        paper_title = _paper_title_cache[filename]
                    else:
                        try:
                            # Bug 1 fix: context manager → _extract_paper_title 예외 시도 doc.close() 보장
                            with fitz.open(str(pdf_path)) as doc:
                                paper_title = _extract_paper_title(filename, doc)
                        except Exception:
                            paper_title = stem_fallback
                        # Flaw 2 fix: 이후 호출에서 재열기 방지
                        _paper_title_cache[filename] = paper_title
            citations.append(Citation(
                source_filename=filename,
                paper_title=paper_title or stem_fallback,
                arxiv_id=arxiv_id,
                section_header=section_header,
                page_number=meta.get("page_number"),
                score=round(score, 4),
            ))
            seen.add(filename)  # 파일 단위 중복 방지

    return citations


# 문장 경계 분리: '.', '!', '?' 다음 공백 기준
# lookbehind로 구분자 자체는 유지하면서 분리
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Flaw 1 fix: 학술 약어 마침표 임시 치환 — _inject_citations_post_hoc 오분리 방지
# "Fig. 3" → "Fig\x01 3" 로 치환해 _SENT_SPLIT_RE가 약어 마침표를 분리하지 않도록 보호
_ABBREV_RE = re.compile(
    r'\b(Fig|et al|e\.g|i\.e|Dr|Mr|Mrs|Prof|vs|cf|Eq|No|Vol|Sec)\.',
    re.IGNORECASE,
)
_ABBREV_PLACEHOLDER = '\x01'  # SOH 제어 문자 — 실제 학술 텍스트에 등장 불가

# P13 Tier-2 숫자 탐지 정규식
# 주의: Python re 모듈은 가변 길이 lookbehind 미지원 → citation 보호는 split으로 처리
_NUM_RE = re.compile(
    r"""
    \b
    (
        \d{1,4}                # 정수 (1~4자리, 연도 포함)
        (?:\.\d+)?             # 소수 부분
        \s*(?:%|k|M|B|x)?     # 단위
    )
    \b
    """,
    re.VERBOSE,
)


def _extract_context_numbers(retrieved: List[Tuple[dict, float]]) -> set[str]:
    """Collect all numeric token strings from retrieved chunks.

    Used as the ground-truth set for P13 scrubbing: any number in the answer
    that is NOT in this set is a candidate for replacement with ``[?]``.

    Args:
        retrieved: List of (metadata_dict, score) from FAISS search.

    Returns:
        Set of numeric strings (e.g. ``{'64', '0.1', '90%', '3B'}``).
    """
    nums: set[str] = set()
    for meta, _ in retrieved:
        # text 필드에서만 추출 (content 키는 ingest 스키마에 없음)
        text = meta.get("text", "")
        for m in _NUM_RE.finditer(text):
            nums.add(m.group(1).strip())
    return nums


def _scrub_hallucinated_numerics(
    answer: str,
    retrieved: List[Tuple[dict, float]],
) -> str:
    """P13 Tier-2: Replace numbers in the answer that are absent from retrieved context.

    Works entirely post-hoc (no additional LLM inference).
    Numbers verified in context are left intact; unverified ones become ``[?]``.

    Coverage:
        - Integers, decimals, percentages, and unit suffixes (k, M, B, x)
        - Numbers inside ``[Source: ...]`` citation tags are never modified
        - Years (1000–2099) are excluded to avoid false positives

    Args:
        answer: The LLM-generated answer string (after P16 injection).
        retrieved: List of (metadata_dict, score) from FAISS search.

    Returns:
        Answer string with unverified numbers replaced by ``[?]``.
    """
    # Ground-truth 숫자 집합: retrieved 청크 전체에서 추출
    context_nums = _extract_context_numbers(retrieved)
    if not context_nums:
        # 청크에서 숫자를 전혀 못 읽은 경우 스크러빙 건너뜀 (안전 폴백)
        return answer

    def _replace(m: re.Match) -> str:
        """단일 숫자 매치에 대해 대체 여부를 결정하는 내부 콜백."""
        val = m.group(1).strip()
        if not val:
            return m.group(0)
        # 단위(%,k,M,B,x,공백) 제거 후 순수 숫자 문자열 추출
        bare = re.sub(r'[%kMBx\s]', '', val)
        try:
            num_float = float(bare)
        except ValueError:
            return m.group(0)  # 파싱 불가 → 건드리지 않음
        # 연도 범위(1000~2099)는 false positive 가능성 높아 제외
        if 1000 <= num_float <= 2099:
            return m.group(0)
        # 단위 포함 형태 또는 순수 숫자 중 하나라도 context에 있으면 유지
        if val in context_nums or bare in context_nums:
            return m.group(0)
        # 검증 실패 → [?]로 대체
        return m.group(0).replace(val, '[?]', 1)

    # [Source: ...] 태그를 기준으로 텍스트를 분리하여 태그 내부는 스크러빙 제외
    citation_safe_re = re.compile(r'(\[Source:[^\]]+\])')
    parts = citation_safe_re.split(answer)
    scrubbed_parts = []
    for part in parts:
        if citation_safe_re.fullmatch(part):
            # citation 태그 자체는 그대로 유지
            scrubbed_parts.append(part)
        else:
            # 일반 텍스트 구간에만 스크러빙 적용
            scrubbed_parts.append(_NUM_RE.sub(_replace, part))
    return ''.join(scrubbed_parts)


def _inject_citations_post_hoc(
    answer: str,
    retrieved: List[Tuple[dict, float]],
    min_overlap: float | None = None,
) -> str:
    """P16: Inject [Source: ...] citations into sentences that lack them.

    For each sentence without a citation tag, computes Jaccard-style word overlap
    against each retrieved chunk. If the best overlap meets ``min_overlap``,
    appends the matching chunk's citation at the sentence end.
    No additional LLM inference — O(sentences × chunks) string operations only.

    Args:
        answer: Raw LLM answer before post-processing.
        retrieved: List of (metadata_dict, cosine_score) from FAISS.
        min_overlap: Minimum fraction of sentence words that must appear in
            the best-matching chunk. Defaults to
            ``settings.citation_injection_min_overlap`` (0.15, eval-optimised).
            Pass an explicit value to override in tests.

    Returns:
        Answer string with inline citations injected where applicable.
    """
    # None이면 config에서 최적값 로드 (하드코딩 제거 → 재현성 보장)
    if min_overlap is None:
        min_overlap = settings.citation_injection_min_overlap
    # Flaw 1 fix: 약어 마침표("Fig.", "et al.", "e.g." 등) 보호 후 분리 → 오분리 방지
    guarded = _ABBREV_RE.sub(lambda m: m.group(1) + _ABBREV_PLACEHOLDER, answer)
    sentences_raw = _SENT_SPLIT_RE.split(guarded)
    sentences = [s.replace(_ABBREV_PLACEHOLDER, '.') for s in sentences_raw]
    result: list[str] = []

    for sent in sentences:
        # 빈 문장 또는 이미 citation이 있는 문장은 그대로 패스
        if not sent.strip() or '[Source:' in sent:
            result.append(sent)
            continue

        # 4글자 이상 단어만 추출 → 의미 있는 내용어 기반 overlap 계산
        sent_words = set(re.findall(r'[a-zA-Z]{4,}', sent.lower()))
        # 내용어가 5개 미만이면 짧은 전환문 등 → citation 부적합, 건너뜀
        if len(sent_words) < 5:
            result.append(sent)
            continue

        best_overlap = 0.0
        best_meta: dict | None = None

        for meta, score in retrieved:
            # citation_min_score(0.65) 미만 청크는 citation 출처로 신뢰도 낮음
            if score < settings.citation_min_score:
                continue
            chunk_words = set(re.findall(r'[a-zA-Z]{4,}', meta.get("text", "").lower()))
            if not chunk_words:
                continue
            # overlap = (문장 단어 ∩ 청크 단어) / 문장 단어 수
            overlap = len(sent_words & chunk_words) / len(sent_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_meta = meta

        if best_overlap >= min_overlap and best_meta:
            # 최적 청크의 citation 태그를 문장 끝에 삽입
            filename = best_meta.get("source_filename", "unknown")
            raw_header = best_meta.get("section_header")
            section = (raw_header if raw_header and not _is_noise_header(raw_header) else None) or "—"
            page = best_meta.get("page_number", "?")
            result.append(f"{sent} [Source: {filename} | Section: {section} | p.{page}]")
        else:
            # min_overlap 미달 → citation 없이 원문 유지
            result.append(sent)

    # 원본 문장 구분자(공백 / \n\n 등)를 복원하여 LaTeX 블록 및 단락 구조 보존
    # " ".join() 사용 시 \n\n 단락 구분이 평탄화되어 LaTeX $$ 블록 렌더링이 깨짐
    # guarded 기준으로 구분자 추출 — 약어 오분리 제외 후 실제 문장 경계만 포함
    seps = _SENT_SPLIT_RE.findall(guarded)
    output: list[str] = []
    for i, sent in enumerate(result):
        output.append(sent)
        if i < len(seps):
            output.append(seps[i])
    return "".join(output)


class Generator:
    """Singleton inference engine wrapping Qwen2.5-3B-Instruct.

    Hardware-adaptive loading strategy:
        - CUDA available + load_in_4bit=True → 4-bit NF4 quantization (bitsandbytes)
        - Apple MPS available → float16 on Metal GPU
        - Fallback → float32 on CPU

    Use ``Generator.get()`` to obtain the shared instance.
    Direct instantiation loads the model weights (~3GB), so singleton is critical.
    """

    # 싱글턴 인스턴스 — 서버 전체에서 모델을 한 번만 로드하기 위해 사용
    _instance: "Generator | None" = None

    def __init__(self) -> None:
        self._device, self._model, self._tokenizer = self._load_model()

    @classmethod
    def get(cls) -> "Generator":
        """Return the shared Generator instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Load Qwen2.5-3B-Instruct with hardware-appropriate precision.

        Returns:
            Tuple of (device_str, model, tokenizer).
        """
        model_id = settings.generation_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available() and settings.load_in_4bit:
            # CUDA 환경: bitsandbytes NF4 4-bit 양자화로 VRAM ~2GB 절감
            # double_quant=True → 양자화 상수 자체를 재양자화 (추가 메모리 절감)
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",  # 멀티 GPU 자동 분산
                trust_remote_code=True,
            )
            device = "cuda"
        elif torch.backends.mps.is_available():
            # Apple Silicon (M1/M2/M3): Metal GPU 사용, float16으로 로드
            # Bug fix: `dtype=` → `torch_dtype=` (from_pretrained 공식 파라미터명)
            # `dtype=`는 silently ignore → float32로 로드되어 메모리 2배 낭비
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("mps")
            device = "mps"
        else:
            # CPU fallback: float32 (정밀도 최대, 속도 최저)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            device = "cpu"

        model.eval()  # 추론 모드: dropout 비활성화, BN 통계 고정
        return device, model, tokenizer

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[dict, float]],
    ) -> Tuple[str, List[Citation], str]:
        """
        Retrieved 청크를 컨텍스트로 답변 생성 + Citation Validator 실행.

        Returns:
            Tuple of ``(answer, citations, answer_pre_scrub)``:
            - answer: P16 injection + P13 scrubbing 완료된 최종 답변
            - citations: 검증된 Citation 목록
            - answer_pre_scrub: P13 스크러빙 이전 답변 (P12/P13 경고 계산에 사용)
            관련 정보 없으면 ``(FALLBACK_ANSWER, [], "")`` 반환.
        """
        if not retrieved:
            return FALLBACK_ANSWER, [], ""

        context = _build_context_block(retrieved)
        user_message = (
            "Examples of required answer format:\n\n"
            "Q: What optimizer does GPT-3 use?\n"
            "A: GPT-3 is trained using the Adam optimizer with a peak learning rate of 0.6×10⁻⁴ "
            "[Source: GPT3.pdf | Section: 2.1 | p.8].\n\n"
            "Q: How does ModelFoo perform on BenchmarkX for TaskA, TaskB, and TaskC?\n"
            "A: As reported at the time of publication, ModelFoo achieves TaskA score 92.3% and TaskB score 85.1% on BenchmarkX "
            "[Source: foo_paper.pdf | Section: 4.2 | p.7]. "
            "ModelFoo was noted to obtain state-of-the-art performance on BenchmarkX at the time of publication "
            "[Source: foo_paper.pdf | Section: Abstract | p.1]. "
            "A separate metric for TaskC is not reported in the provided documents.\n\n"
            "Q: What is the key difference between MethodA and MethodB in terms of parameter efficiency?\n"
            "A: MethodA freezes all pretrained weights and introduces small trainable adapter modules, requiring far fewer updated parameters "
            "[Source: methodA.pdf | Section: 3 | p.5]. "
            "The exact percentage reduction in trainable parameters relative to MethodB is not reported in the retrieved passages. "
            "Details on MethodB's parameter efficiency are not present in the retrieved documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (follow the example format — include [Source: ...] after every factual claim, "
            "preserve all metric names verbatim, and time-bound any SOTA claims):"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        gen_kwargs: dict = {
            "max_new_tokens": settings.generation_max_new_tokens,
            "do_sample": settings.generation_do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if settings.generation_do_sample:
            gen_kwargs["temperature"] = settings.generation_temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not answer or _is_fallback(answer):
            return FALLBACK_ANSWER, [], ""

        # ── Post-processing Pipeline ────────────────────────────────────────
        # Step 1 (P16): citation 없는 문장에 word-overlap 기반 citation 삽입
        answer = _inject_citations_post_hoc(answer, retrieved)
        # P12/P13 경고는 스크러빙 이전 답변에 적용해야 함
        # 스크러빙 후에는 숫자가 [?]로 치환되어 _METRIC_NUM_RE가 매치 불가 → 경고 항상 0
        answer_pre_scrub = answer
        # Step 2 (P13 Tier-2): context에 없는 숫자를 [?]로 스크러빙
        answer = _scrub_hallucinated_numerics(answer, retrieved)
        # ───────────────────────────────────────────────────────────────────

        # 전체 retrieved 청크에서 후보 citation 목록 구성 (score >= citation_min_score)
        all_citations = _build_citations(retrieved)

        # 답변 텍스트에서 실제로 인용된 파일명 집합 추출
        cited_sources = _extract_cited_sources(answer)
        if cited_sources:
            # [Source: ...] 태그가 있으면 해당 파일만 유지 (pruning)
            pruned = [c for c in all_citations if c.source_filename in cited_sources]
            citations = pruned if pruned else all_citations
        else:
            # 태그가 없으면 paper_title 키워드 매칭으로 fallback
            by_title = _match_citations_by_title(answer, all_citations)
            citations = by_title if by_title else all_citations

        # P11: 답변 키워드와 청크 키워드 교집합이 적은 citation 제거
        citations = _filter_by_contribution(answer, citations, retrieved)

        return answer, citations, answer_pre_scrub

    def suggest_queries(self, chunks: List[str]) -> List[str]:
        """
        논문 청크 샘플을 기반으로 LLM이 답변 가능한 연구 질문 4개 생성.
        파싱 실패 시 빈 리스트 반환 (프론트엔드 fallback 처리).
        """
        if not chunks:
            return []

        context = "\n\n---\n\n".join(
            f"[Excerpt {i + 1}]\n{chunk[:400]}" for i, chunk in enumerate(chunks)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Based on the paper excerpts provided, "
                    "generate exactly 4 specific and concise research questions that can be "
                    "answered using these documents. "
                    "Return ONLY a valid JSON array of 4 question strings. "
                    'Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"] '
                    "Do NOT include any explanation, markdown code fences, or extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Paper excerpts:\n{context}\n\nJSON array of 4 questions:",
            },
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            # greedy 매치: non-greedy(.*?)는 [MASK], [CLS] 등 NLP 토큰이 포함된 경우
            # 첫 번째 ] 에서 끊겨 파싱 실패 → greedy(.*)로 마지막 ] 까지 매치
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                if isinstance(questions, list):
                    valid = [str(q).strip() for q in questions if str(q).strip()]
                    if len(valid) >= 2:
                        return valid[:4]
        except Exception:
            pass

        return []
