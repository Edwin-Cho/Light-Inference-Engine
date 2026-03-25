from __future__ import annotations

import re
from typing import List, Tuple

import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.models.schemas import Citation
from app.pipeline.ingest import _is_noise_header, _extract_arxiv_id, _extract_paper_title

FALLBACK_ANSWER = "No relevant information found in the provided documents."

_FALLBACK_SUBSTRINGS = (
    "no relevant information found",
    "i don't know",
    "i do not know",
    "cannot find",
    "not found in the provided",
)

SYSTEM_PROMPT = """You are a precise academic research assistant. Answer ONLY using the provided context passages.

Rules:
1. Every factual claim MUST include an inline citation in the format [Source: filename | Section: X | p.N].
2. If the context does not contain enough information to answer, respond exactly with: "No relevant information found in the provided documents."
3. Do NOT fabricate facts, authors, or findings not present in the context.
4. SYNTHESIZE the content in your own words — do NOT copy sentences verbatim from the context.
5. Replace all vague pronouns with explicit names: "Our method" → the paper/method name (e.g. "AWQ"), "We" → "the authors of [paper]", "they" → the actual subject.
6. Be concise and accurate."""


def _build_context_block(retrieved: List[Tuple[dict, float]]) -> str:
    blocks = []
    for meta, score in retrieved:
        filename = meta.get("source_filename", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section_header") or "—"
        text = meta.get("text", "")
        blocks.append(
            f"[Source: {filename} | Section: {section} | p.{page} | score: {score:.3f}]\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def _is_fallback(answer: str) -> bool:
    """모델이 답변 불가 판정을 했는지 확인."""
    lower = answer.lower()
    return any(kw in lower for kw in _FALLBACK_SUBSTRINGS)


_SOURCE_PATTERN = re.compile(r'\[Source:\s*([^|\]]+?)\s*\|', re.IGNORECASE)


def _extract_cited_sources(answer: str) -> set[str]:
    """
    LLM 답변 텍스트에서 [Source: filename | ...] 패턴으로 인용된 파일명 추출.
    """
    return {m.group(1).strip() for m in _SOURCE_PATTERN.finditer(answer)}


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
    """
    P11: 답변-citation 불일치 필터.
    답변 키워드와 청크 텍스트 키워드 교집합이 min_overlap 미만이면 제거.
    전체 필터링 시 원본 반환 (안전 폴백).
    """
    answer_kw = _extract_keywords(answer)
    if len(answer_kw) < min_overlap * 3:
        return citations

    chunk_map: dict[str, str] = {}
    for meta, _ in retrieved:
        fn = meta.get("source_filename", "")
        if fn and fn not in chunk_map:
            chunk_map[fn] = meta.get("text", "")

    filtered = [
        c for c in citations
        if len(answer_kw & _extract_keywords(chunk_map.get(c.source_filename, ""))) >= min_overlap
    ]
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


def _build_citations(retrieved: List[Tuple[dict, float]]) -> List[Citation]:
    """
    Citation Validator: retrieved 청크의 메타데이터를 citation으로 변환.
    파일 단위로 중복 제거하여 반환.
    """
    seen: set[str] = set()
    citations: List[Citation] = []

    for meta, score in retrieved:
        if score < settings.citation_min_score:
            continue
        filename = meta.get("source_filename", "")
        if filename and filename not in seen:
            raw_header = meta.get("section_header")
            section_header = None if (raw_header and _is_noise_header(raw_header)) else raw_header
            arxiv_id = meta.get("arxiv_id") or _extract_arxiv_id(filename)
            paper_title = meta.get("paper_title") or ""
            stem_fallback = filename.rsplit(".", 1)[0].replace("_", " ")
            if not paper_title or paper_title == stem_fallback:
                pdf_path = settings.data_raw_dir / filename
                if pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
                    try:
                        doc = fitz.open(str(pdf_path))
                        paper_title = _extract_paper_title(filename, doc)
                        doc.close()
                    except Exception:
                        paper_title = stem_fallback
            citations.append(Citation(
                source_filename=filename,
                paper_title=paper_title or stem_fallback,
                arxiv_id=arxiv_id,
                section_header=section_header,
                page_number=meta.get("page_number"),
                score=round(score, 4),
            ))
            seen.add(filename)

    return citations


class Generator:
    """
    Qwen2.5-3B-Instruct 추론 엔진.
    - CUDA: 4-bit NF4 quantization (bitsandbytes)
    - MPS / CPU: float16 / float32 자동 fallback
    """

    _instance: "Generator | None" = None

    def __init__(self) -> None:
        self._device, self._model, self._tokenizer = self._load_model()

    @classmethod
    def get(cls) -> "Generator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        model_id = settings.generation_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available() and settings.load_in_4bit:
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
                device_map="auto",
                trust_remote_code=True,
            )
            device = "cuda"
        elif torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                trust_remote_code=True,
            ).to("mps")
            device = "mps"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                trust_remote_code=True,
            )
            device = "cpu"

        model.eval()
        return device, model, tokenizer

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[dict, float]],
    ) -> Tuple[str, List[Citation]]:
        """
        Retrieved 청크를 컨텍스트로 답변 생성 + Citation Validator 실행.

        Returns:
            (answer: str, citations: List[Citation])
            관련 정보 없으면 FALLBACK_ANSWER + 빈 citations 반환.
        """
        if not retrieved:
            return FALLBACK_ANSWER, []

        context = _build_context_block(retrieved)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

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
            return FALLBACK_ANSWER, []

        all_citations = _build_citations(retrieved)

        cited_sources = _extract_cited_sources(answer)
        if cited_sources:
            pruned = [c for c in all_citations if c.source_filename in cited_sources]
            citations = pruned if pruned else all_citations
        else:
            by_title = _match_citations_by_title(answer, all_citations)
            citations = by_title if by_title else all_citations

        citations = _filter_by_contribution(answer, citations, retrieved)

        return answer, citations
