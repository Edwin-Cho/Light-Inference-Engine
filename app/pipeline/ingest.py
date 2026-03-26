from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

from app.config import settings
from app.pipeline.chunker import chunk_text
from app.pipeline.embedder import Embedder
from app.pipeline.store import VectorStore


def _make_document_id(filename: str) -> str:
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


_ARXIV_PATTERN = re.compile(r'\b(\d{4}\.\d{4,5})(v\d+)?\b')
_ARXIV_URL_PATTERN = re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', re.IGNORECASE)


def _extract_arxiv_id(filename: str, doc: Optional[object] = None) -> Optional[str]:
    """
    arXiv ID 추출 (우선순위):
    1. PDF 메타데이터 (title / subject / keywords / creator)
    2. 첫 2페이지 본문 (arXiv:XXXX.XXXXX 또는 arxiv.org/abs/XXXX.XXXXX)
    3. 파일명 (예: 1706.03762.pdf)
    """
    if doc is not None:
        meta_fields = [
            doc.metadata.get("title", ""),
            doc.metadata.get("subject", ""),
            doc.metadata.get("keywords", ""),
            doc.metadata.get("creator", ""),
        ]
        for field in meta_fields:
            if not field:
                continue
            m = _ARXIV_URL_PATTERN.search(field) or _ARXIV_PATTERN.search(field)
            if m:
                return m.group(1)

        for page_num in range(min(2, doc.page_count)):
            page_text = doc[page_num].get_text("text")
            m = _ARXIV_URL_PATTERN.search(page_text)
            if m:
                return m.group(1)
            # "arXiv:1706.03762" 또는 "arXiv: 1706.03762v3" 형태
            m = re.search(r'arXiv\s*:\s*(\d{4}\.\d{4,5})', page_text, re.IGNORECASE)
            if m:
                return m.group(1)

    stem = Path(filename).stem
    if re.match(r'^\d{4}\.\d{4,5}$', stem):
        return stem
    return None


_TITLE_GARBAGE_PATTERNS = [
    re.compile(r'\\[a-zA-Z]+\{'),                         # LaTeX: \title{, \maketitle{
    re.compile(r'^https?://', re.IGNORECASE),              # URL
    re.compile(r'^\d+$'),                                  # 순수 숫자
    re.compile(r'Microsoft Word', re.IGNORECASE),          # 편집 도구 흔적
    re.compile(r'^arXiv:', re.IGNORECASE),                 # arXiv 제출 배너: "arXiv:1810.04805v2 ..."
    re.compile(r'\[(?:cs|eess|q-bio|stat|math|physics)\.[A-Z]'),  # arXiv 카테고리 태그
    re.compile(r'\d{1,2}\s+\w+\s+\d{4}$'),                # 날짜 끝: "24 May 2019"
]


def _is_valid_title(text: str) -> bool:
    """제목으로 사용 가능한 텍스트인지 검증."""
    if not text or len(text) < 6 or len(text) > 250:
        return False
    if any(p.search(text) for p in _TITLE_GARBAGE_PATTERNS):
        return False
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text) >= 0.3


def _extract_paper_title(filename: str, doc: Optional[object] = None) -> str:
    """
    논문 제목 추출 (우선순위):
    1. PDF 메타데이터 title 필드
    2. 첫 페이지 최대 폰트 크기 스팬 집합
    3. 파일명 기반 폴백
    """
    if doc is not None:
        meta_title = doc.metadata.get("title", "").strip()
        if _is_valid_title(meta_title):
            return meta_title

        if doc.page_count > 0:
            try:
                page = doc[0]
                blocks = page.get_text("dict")["blocks"]
                span_list: list[tuple[float, str]] = []
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            t = span.get("text", "").strip()
                            sz = span.get("size", 0.0)
                            if t and sz > 0:
                                span_list.append((sz, t))
                seen_sizes: list[float] = sorted(
                    {sz for sz, _ in span_list}, reverse=True
                )
                for target_sz in seen_sizes:
                    parts = [
                        t for sz, t in span_list
                        if abs(sz - target_sz) < 0.5
                    ]
                    candidate = " ".join(parts).strip()
                    if _is_valid_title(candidate):
                        return candidate
            except Exception:
                pass

    return Path(filename).stem.replace("_", " ")


_NOISE_PATTERNS = [
    re.compile(r'URL\s+https?://', re.IGNORECASE),
    re.compile(r'doi:\s*10\.', re.IGNORECASE),
    re.compile(r'arXiv:', re.IGNORECASE),
    re.compile(r',\s*\d{4}[\.)\]]'),     # bibliography year: ", 2023."
    re.compile(r'^\d+\.\d{2,}'),         # 2+ decimal places: score "9.56 RTN", "12.46 GPTQ"
    re.compile(r'^0\.\d'),               # 0.X ratio/probability: "0.7 Mean..."
    re.compile(r'^\d{3,}\s'),            # 3+ digit leading integer: table row "199 ResNet101x1"
    re.compile(r'\(vector-wise|\(row-wise|\(column-wise', re.IGNORECASE),  # quant table
    re.compile(                          # Figure/Table/Algorithm caption: "Figure 1:", "Table 2."
        r'^(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?|Listing)\s+\d+',
        re.IGNORECASE,
    ),
]


def _is_noise_header(text: str) -> bool:
    """
    셀렉션 헤더 오탐 필터.
    참고문헌 줄 / URL / DOI / arXiv / 표 행 / 차트 레이블 등을 차단.
    """
    if any(p.search(text) for p in _NOISE_PATTERNS):
        return True
    if text.count('.') >= 3 and len(text) > 50:
        return True
    # Float-prefix 휴리스틱: "9.56 RTN", "9 .56 RTN", "12.46 GPTQ", "0.7 Mean..."
    # PyMuPDF 스팬 분리로 "9 .56" 형태가 될 수 있어 앞 2토큰을 합쳐 파싱.
    tokens = text.strip().split()
    for candidate in ([tokens[0]] if tokens else []) + (["".join(tokens[:2])] if len(tokens) >= 2 else []):
        candidate_clean = re.sub(r'\s', '', candidate)
        try:
            val = float(candidate_clean)
            decimal_str = candidate_clean.split('.')[1] if '.' in candidate_clean else ''
            if len(decimal_str) >= 2 or (0.0 < val < 1.0):
                return True
            break
        except ValueError:
            continue
    return False


def _detect_section_header(page_fitz) -> Optional[str]:
    """
    PyMuPDF dict 모드로 폰트 크기 + bold 기반 섹션 헤더 탐지.
    body text보다 큰 폰트 또는 bold 표시된 옵로 첫 번째 추보 입후 반환.
    """
    from collections import Counter

    blocks = page_fitz.get_text("dict")["blocks"]
    all_sizes: list[float] = []
    line_entries: list[tuple[float, str, bool]] = []

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(s.get("text", "").strip() for s in spans).strip()
            sz = max((round(s.get("size", 0), 1) for s in spans), default=0.0)
            is_bold = any(
                bool(s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()
                for s in spans
            )
            if text and sz > 0:
                all_sizes.append(sz)
                line_entries.append((sz, text, is_bold))

    if not all_sizes or not line_entries:
        return None

    body_size = Counter(all_sizes).most_common(1)[0][0]
    running_header = line_entries[0][1] if line_entries else ""

    for sz, text, is_bold in line_entries[1:]:
        if text == running_header or re.match(r'^[\d\.\s,\-\+]+$', text):
            continue
        if _is_noise_header(text):
            continue
        is_larger = sz > body_size * 1.05
        is_section_bold = is_bold and re.match(
            r'^(\d[\d\.]*\.?\s*[A-Z]|[A-Z][A-Z\s\-]{3,}$)', text
        )
        if (is_larger or is_section_bold) and 3 < len(text) < 120:
            return text
    return None


def _parse_pdf(path: Path) -> tuple[list[tuple[int, str, Optional[str]]], Optional[str], str]:
    """
    PDF를 페이지별로 파싱. 섹션 헤더를 폰트 크기 기반으로 탐지하고 forward propagate.
    Returns:
        pages: [(page_number_1indexed, page_text, section_header), ...], 빈 페이지 제외.
        arxiv_id: PDF 메타데이터 / 본문에서 추출한 arXiv ID (없으면 None).
        paper_title: PDF 메타데이터 / 첫 페이지 최대 폰트에서 추출한 제목.
    """
    doc = fitz.open(str(path))
    arxiv_id = _extract_arxiv_id(path.name, doc)
    paper_title = _extract_paper_title(path.name, doc)
    pages: list[tuple[int, str, Optional[str]]] = []
    current_section: Optional[str] = None

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue
        detected = _detect_section_header(page)
        if detected:
            current_section = detected
        pages.append((page_num, text, current_section))

    doc.close()
    return pages, arxiv_id, paper_title


def _extract_numbered_section(text: str) -> Optional[str]:
    """
    청크 텍스트 전체에서 번호 기반 섹션 헤더 탐지 (font-size 탐지 보완).
    단일줄: '3.2 Attention', '3.2.1 Scaled Dot-Product Attention'
    2줄 결합: '3.2\nAttention' 형태 (다단 레이아웃 PDF 대응)
    """
    lines = [l.strip() for l in text.strip().splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        if (
            re.match(r'^\d[\d\.]*\.?\s+[A-Z][a-zA-Z]', line)
            and 10 < len(line) < 100
            and not line.endswith('.')
            and not line.endswith(',')
            and not _is_noise_header(line)
        ):
            return line
        if (
            re.match(r'^[A-Z]\s+[A-Z][a-zA-Z]', line)
            and 10 < len(line) < 80
            and not _is_noise_header(line)
        ):
            return line
        if re.match(r'^\d[\d\.]*\.?$', line) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and re.match(r'^[A-Z][a-zA-Z]{2,}', nxt) and len(nxt) < 80:
                combined = f"{line} {nxt}"
                if not combined.endswith('.') and not combined.endswith(','):
                    return combined
    return None


def _parse_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_file(file_path: Path) -> int:
    """
    PDF / Markdown / TXT 파일을 파싱 → 청킹 → 임베딩 → FAISS 저장.

    Returns:
        chunks_indexed (int)

    Raises:
        ValueError: 지원하지 않는 파일 형식
        RuntimeError: 파싱 실패
    """
    suffix = file_path.suffix.lower()
    filename = file_path.name
    document_id = _make_document_id(filename)
    arxiv_id = _extract_arxiv_id(filename)

    base_metadata = {
        "document_id": document_id,
        "source_filename": filename,
        "paper_title": Path(filename).stem.replace("_", " "),
        "arxiv_id": arxiv_id,
    }

    if suffix == ".pdf":
        try:
            page_data, detected_arxiv_id, detected_title = _parse_pdf(file_path)
            if detected_arxiv_id:
                base_metadata["arxiv_id"] = detected_arxiv_id
            if detected_title:
                base_metadata["paper_title"] = detected_title
        except Exception as exc:
            raise RuntimeError(f"PDF 파싱 실패: {exc}") from exc
        if not page_data:
            raise RuntimeError("문서에서 텍스트를 추출할 수 없습니다.")

        chunks: list[dict] = []
        current_section: Optional[str] = None
        for page_num, page_text, page_section in page_data:
            if page_section:
                current_section = page_section
            page_chunks = chunk_text(page_text, {**base_metadata, "page_number": page_num})
            for chunk in page_chunks:
                override = _extract_numbered_section(chunk["text"])
                if override:
                    current_section = override
                chunk["metadata"]["section_header"] = current_section
            chunks.extend(page_chunks)

    elif suffix in (".md", ".txt"):
        full_text = _parse_text(file_path)
        if not full_text.strip():
            raise RuntimeError("문서에서 텍스트를 추출할 수 없습니다.")
        chunks = chunk_text(full_text, base_metadata)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

    if not chunks:
        raise RuntimeError("청킹 결과가 비어있습니다.")

    texts = [c["text"] for c in chunks]
    meta_list = [{"text": c["text"], **c["metadata"]} for c in chunks]

    embedder = Embedder.get()
    vectors = embedder.embed(texts)

    store = VectorStore.get()
    store.add(vectors, meta_list)

    return len(chunks)
