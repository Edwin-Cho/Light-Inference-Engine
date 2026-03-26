from __future__ import annotations

import json
import random
import re
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.auth.rbac import Role, authenticate_user, require_role
from app.auth.token import create_access_token
from app.config import settings
from app.models.schemas import (
    DeleteResponse,
    DocumentItem,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RebuildResponse,
    SuggestQueriesResponse,
    TokenRequest,
    TokenResponse,
)
from app.pipeline.embedder import Embedder
from app.pipeline.generator import Generator
from app.pipeline.ingest import ingest_file, _make_document_id, _extract_numbered_section
from app.pipeline.retriever import retrieve
from app.pipeline.store import VectorStore

app = FastAPI(
    title="OnDevice Scholar RAG",
    description="Privacy-first, fully offline RAG pipeline for academic research.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """모델 및 인덱스 사전 로드."""
    Embedder.get()
    VectorStore.get()


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Public"])
async def health() -> HealthResponse:
    return HealthResponse()


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(body: TokenRequest) -> TokenResponse:
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token)


# ── Query helpers ───────────────────────────────────────────────────────

_CMP_PATTERN = re.compile(
    r'(?:difference|compare|comparison|between)\s+([A-Za-z0-9][A-Za-z0-9\-\.]+)'
    r'\s+and\s+([A-Za-z0-9][A-Za-z0-9\-\.]+)',
    re.IGNORECASE,
)
_VS_PATTERN = re.compile(
    r'([A-Za-z0-9][A-Za-z0-9\-\.]+)\s+(?:vs\.?|versus)\s+([A-Za-z0-9][A-Za-z0-9\-\.]+)',
    re.IGNORECASE,
)


def _single_source_warnings(query: str, citations) -> list[str]:
    """비교 쿼리에서 한쪽 소스만 retrieval됐을 때 경고 반환."""
    m = _CMP_PATTERN.search(query) or _VS_PATTERN.search(query)
    if not m:
        return []
    e1, e2 = m.group(1).lower(), m.group(2).lower()
    sources = " ".join(c.source_filename.lower() for c in citations)
    warns = []
    if e1 not in sources:
        warns.append(f"Comparison query detected: no source related to '{m.group(1)}' retrieved")
    if e2 not in sources:
        warns.append(f"Comparison query detected: no source related to '{m.group(2)}' retrieved")
    return warns


# ── Suggest Queries ──────────────────────────────────────────────────────────

_SUGGEST_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "suggested_queries.json"


def _sample_diverse_chunks(metadata: list, n_papers: int = 8) -> list[str]:
    """서로 다른 논문에서 청크를 1개씩 샘플링. 텍스트 길이 기준으로 대표 청크 선택."""
    by_paper: dict[str, list[dict]] = {}
    for m in metadata:
        fn = m.get("source_filename", "")
        if fn:
            by_paper.setdefault(fn, []).append(m)

    papers = list(by_paper.keys())
    random.shuffle(papers)
    selected = papers[:n_papers]

    chunks = []
    for paper in selected:
        candidates = by_paper[paper]
        best = max(candidates, key=lambda m: len(m.get("text", "")))
        text = best.get("text", "").strip()
        if text:
            chunks.append(text)
    return chunks


@app.get("/suggest-queries", response_model=SuggestQueriesResponse, tags=["RAG"])
async def suggest_queries(
    _user: dict = Depends(require_role(Role.RESEARCHER)),
) -> SuggestQueriesResponse:
    """
    인덱싱된 논문 청크를 기반으로 LLM이 답변 가능한 연구 질문 4개를 생성.
    최초 1회 생성 후 캐시 파일에 저장; 이후 요청은 캐시에서 즉시 반환.
    캐시는 rebuild-index 호출 시 초기화됨.
    """
    if _SUGGEST_CACHE_PATH.exists():
        try:
            with open(_SUGGEST_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= 2:
                return SuggestQueriesResponse(questions=data, cached=True)
        except Exception:
            pass

    store = VectorStore.get()
    metadata = store.get_all_metadata()
    if not metadata:
        return SuggestQueriesResponse(questions=[])

    chunks = _sample_diverse_chunks(metadata)
    generator = Generator.get()
    questions = generator.suggest_queries(chunks)

    if questions:
        _SUGGEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SUGGEST_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)

    return SuggestQueriesResponse(questions=questions, cached=False)


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    body: QueryRequest,
    _user: dict = Depends(require_role(Role.RESEARCHER)),
) -> QueryResponse:
    retrieved = retrieve(body.query, top_k=body.top_k)
    generator = Generator.get()
    answer, citations = generator.generate(body.query, retrieved)

    warnings: list[str] = []
    for c in citations:
        missing = [f for f in ("section_header", "page_number") if getattr(c, f) is None]
        if missing:
            warnings.append(f"{c.source_filename}: {', '.join(missing)} missing")
    warnings += _single_source_warnings(body.query, citations)

    return QueryResponse(
        answer=answer,
        citations=citations,
        status="partial" if warnings else "ok",
        warnings=warnings,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    file: UploadFile = File(...),
    _user: dict = Depends(require_role(Role.LAB_PI)),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_SUFFIXES}",
        )

    tmp_path = settings.data_raw_dir / f"_tmp_{uuid.uuid4().hex}{suffix}"
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        final_path = settings.data_raw_dir / file.filename
        tmp_path.rename(final_path)

        chunks_indexed = ingest_file(final_path)
    except (ValueError, RuntimeError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingest failed: {exc}",
        )

    return IngestResponse(filename=file.filename, chunks_indexed=chunks_indexed)


# ── Documents List ──────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResponse, tags=["RAG"])
async def list_documents(
    _user: dict = Depends(require_role(Role.RESEARCHER)),
) -> DocumentListResponse:
    """FAISS 메타데이터를 기반으로 인덱싱된 문서 목록 반환 (청크 수 포함)."""
    store = VectorStore.get()
    metadata = store.get_all_metadata()

    seen: dict[str, DocumentItem] = {}
    for m in metadata:
        doc_id = m.get("document_id", "")
        filename = m.get("source_filename", "")
        if not doc_id or not filename:
            continue
        if doc_id not in seen:
            seen[doc_id] = DocumentItem(
                document_id=doc_id,
                source_filename=filename,
                paper_title=m.get("paper_title") or None,
                arxiv_id=m.get("arxiv_id") or None,
                chunk_count=0,
            )
        seen[doc_id].chunk_count += 1

    docs = sorted(seen.values(), key=lambda d: d.source_filename.lower())
    return DocumentListResponse(documents=docs, total_chunks=len(metadata))


# ── Delete ─────────────────────────────────────────────────────────────────────

@app.delete("/document/{document_id}", response_model=DeleteResponse, tags=["RAG"])
async def delete_document(
    document_id: str,
    _user: dict = Depends(require_role(Role.LAB_PI)),
) -> DeleteResponse:
    store = VectorStore.get()
    if not store.document_exists(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in index.",
        )
    removed = store.remove_by_document_id(document_id)
    return DeleteResponse(document_id=document_id, chunks_removed=removed)


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/admin/rebuild-index", response_model=RebuildResponse, tags=["Admin"])
async def rebuild_index(
    _user: dict = Depends(require_role(Role.ADMIN)),
) -> RebuildResponse:
    """
    data/raw/ 의 모든 문서를 재파싱·재임베딩하여 인덱스 전체 재빌드.
    """
    store = VectorStore.get()
    embedder = Embedder.get()

    from app.pipeline.chunker import chunk_text
    from app.pipeline.ingest import _parse_pdf, _parse_text, _make_document_id, _extract_arxiv_id
    import numpy as np

    all_vectors = []
    all_meta = []
    doc_count = 0

    for file_path in sorted(settings.data_raw_dir.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES or file_path.name.startswith("_tmp_"):
            continue

        try:
            base_meta = {
                "document_id": _make_document_id(file_path.name),
                "source_filename": file_path.name,
                "paper_title": file_path.stem.replace("_", " "),
                "arxiv_id": _extract_arxiv_id(file_path.name),
            }

            if suffix == ".pdf":
                page_data, detected_arxiv_id, detected_title = _parse_pdf(file_path)
                if detected_arxiv_id:
                    base_meta["arxiv_id"] = detected_arxiv_id
                if detected_title:
                    base_meta["paper_title"] = detected_title
                if not page_data:
                    continue
                current_sec = None
                for page_num, page_text, page_section in page_data:
                    if page_section:
                        current_sec = page_section
                    page_chunks = chunk_text(page_text, {**base_meta, "page_number": page_num})
                    for chunk in page_chunks:
                        override = _extract_numbered_section(chunk["text"])
                        if override:
                            current_sec = override
                        chunk["metadata"]["section_header"] = current_sec
                    texts = [c["text"] for c in page_chunks]
                    vectors = embedder.embed(texts)
                    all_vectors.append(vectors)
                    all_meta.extend([{"text": c["text"], **c["metadata"]} for c in page_chunks])
            else:
                full_text = _parse_text(file_path)
                if not full_text.strip():
                    continue
                chunks = chunk_text(full_text, base_meta)
                texts = [c["text"] for c in chunks]
                vectors = embedder.embed(texts)
                all_vectors.append(vectors)
                all_meta.extend([{"text": c["text"], **c["metadata"]} for c in chunks])

            doc_count += 1
        except Exception:
            continue

    if all_vectors:
        combined = np.vstack(all_vectors).astype(np.float32)
        store.rebuild(combined, all_meta)

    _SUGGEST_CACHE_PATH.unlink(missing_ok=True)

    return RebuildResponse(
        documents_reindexed=doc_count,
        chunks_total=len(all_meta),
    )


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            reason=str(exc),
        ).model_dump(),
    )
