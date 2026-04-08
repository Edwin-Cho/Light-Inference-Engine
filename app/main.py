"""FastAPI application entry point for OnDevice Scholar RAG.

Endpoint overview:
    GET  /health                — backend liveness check
    POST /auth/token            — JWT login (username + password)
    GET  /suggest-queries       — LLM-generated research questions (cached)
    POST /query                 — main RAG pipeline: retrieve → generate → validate
    POST /ingest                — upload & index a new document (LAB_PI+)
    GET  /documents             — list indexed documents with chunk counts
    DELETE /document/{id}       — remove a document from the index (LAB_PI+)
    POST /admin/rebuild-index   — full re-ingestion from data/raw/ (ADMIN only)

Auth: Bearer JWT via /auth/token. RBAC roles: researcher < lab_pi < admin.
"""
from __future__ import annotations

import json
import logging
import random
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
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
from app.pipeline.chunker import chunk_text
from app.pipeline.embedder import Embedder
from app.pipeline.generator import Generator, _check_metric_fidelity, _check_numeric_existence
from app.pipeline.ingest import (
    ingest_file,
    _make_document_id,
    _extract_numbered_section,
    _parse_pdf,
    _parse_text,
    _extract_arxiv_id,
)
from app.pipeline.retriever import (
    retrieve,
    retrieve_comparison,
    is_comparison_query,
    _extract_comparison_sides,
)
from app.pipeline.store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server startup/shutdown lifecycle handler.

    Replaces deprecated ``@app.on_event("startup")``.
    Pre-loads Embedder and VectorStore singletons to avoid cold-start
    latency on the first real request.
    Generator is intentionally deferred — it is large and loaded on demand.
    """
    Embedder.get()      # BAAI/bge-small-en-v1.5 로드
    VectorStore.get()   # FAISS 인덱스 + 메타 JSON 로드
    yield


app = FastAPI(
    title="OnDevice Scholar RAG",
    description="Privacy-first, fully offline RAG pipeline for academic research.",
    version="1.1.0",
    lifespan=lifespan,
)

# Vite 개발 서버(5173)에서의 요청을 허용
# 배포 시에는 실제 도메인으로 교체해야 함
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 지원하는 업로드 파일 형식 (PDF + 텍스트)
ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}


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


# ── Query helpers ───────────────────────────────────────────────

def _single_source_warnings(query: str, citations) -> list[str]:
    """Warn when a comparison query retrieved sources for only one side.

    Uses ``_extract_comparison_sides`` from retriever (single source of truth)
    to avoid duplicating regex logic and to support multi-word entity names.
    Searches both ``source_filename`` and ``paper_title`` to reduce false positives
    when entity names do not appear in filenames (e.g. ``devlin_2019.pdf`` for BERT).

    Args:
        query: Raw user query string.
        citations: List of Citation objects returned by the generator.

    Returns:
        List of warning strings (empty if both sides are covered or not a comparison query).
    """
    # Flaw 2 fix: _CMP_PATTERN/_VS_PATTERN 제거 — retriever._extract_comparison_sides() 재사용
    sides = _extract_comparison_sides(query)
    if not sides:
        return []
    raw_e1, raw_e2 = sides
    e1, e2 = raw_e1.lower(), raw_e2.lower()
    # Flaw 1 fix: paper_title까지 포함하여 매칭 확장 (파일명에 entity명이 없는 경우 대응)
    sources = " ".join(
        (c.source_filename + " " + (c.paper_title or "")).lower()
        for c in citations
    )
    warns = []
    if e1 not in sources:
        warns.append(f"Comparison query detected: no source related to '{raw_e1}' retrieved")
    if e2 not in sources:
        warns.append(f"Comparison query detected: no source related to '{raw_e2}' retrieved")
    return warns


# ── Suggest Queries ──────────────────────────────────────────────────────────

# Improvement 2 fix: settings.data_index_dir 사용으로 하드코딩 제거 (경로 변경 시 단일 수정 보장)
_SUGGEST_CACHE_PATH = settings.data_index_dir / "suggested_queries.json"


def _sample_diverse_chunks(metadata: list, n_papers: int = 8) -> list[str]:
    """Sample one representative chunk per paper for suggest-queries generation.

    Picks the longest text chunk from each paper to maximise information density
    for the LLM. Shuffles paper order to vary suggestions across calls.

    Args:
        metadata: Full list of chunk metadata dicts from VectorStore.
        n_papers: Number of papers to sample from (default 8).

    Returns:
        List of text strings, one per sampled paper.
    """
    # 사용 논문 파일명 기준으로 그룹화
    by_paper: dict[str, list[dict]] = {}
    for m in metadata:
        fn = m.get("source_filename", "")
        if fn:
            by_paper.setdefault(fn, []).append(m)

    papers = list(by_paper.keys())
    random.shuffle(papers)  # 새로고침(refresh) 시 다양한 질문 생성을 위해 샤플
    selected = papers[:n_papers]

    chunks = []
    for paper in selected:
        candidates = by_paper[paper]
        # 가장 진 청크 선택 → 내용이 풍부한 청크일수록 LLM이 더 지식적인 질문 생성
        best = max(candidates, key=lambda m: len(m.get("text", "")))
        text = best.get("text", "").strip()
        if text:
            chunks.append(text)
    return chunks


@app.get("/suggest-queries", response_model=SuggestQueriesResponse, tags=["RAG"])
async def suggest_queries(
    _user: dict = Depends(require_role(Role.RESEARCHER)),
    refresh: bool = False,
) -> SuggestQueriesResponse:
    """
    인덱싱된 논문 청크를 기반으로 LLM이 답변 가능한 연구 질문 4개를 생성.
    최초 1회 생성 후 캐시 파일에 저장; 이후 요청은 캐시에서 즉시 반환.
    refresh=true 전달 시 캐시를 삭제하고 재생성.
    캐시는 rebuild-index 호출 시 초기화됨.
    """
    if refresh:
        _SUGGEST_CACHE_PATH.unlink(missing_ok=True)

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
    """Main RAG query endpoint.

    Full pipeline:
        1. Retrieve: 2-pass FAISS search (or comparison sub-retrieval for A vs B queries)
        2. Generate: Qwen2.5-3B-Instruct with structured SYSTEM_PROMPT
        3. Post-process: P16 citation injection → P13 numeric scrubbing
        4. Validate: citation metadata completeness + P12 metric fidelity + P13 existence check

    Args:
        body: QueryRequest with ``query`` string and optional ``top_k`` / ``include_chunks``.

    Returns:
        QueryResponse with answer, citations, warnings, and optional retrieved chunks.
    """
    # Step 1: 비교 쿼리(“A vs B”) 여부에 따라 retrieval 전략 분기
    if is_comparison_query(body.query):
        retrieved = retrieve_comparison(body.query, top_k=body.top_k)
    else:
        retrieved = retrieve(body.query, top_k=body.top_k)

    # Step 2: LLM 답변 생성 + citation list 반환
    # answer_pre_scrub: P13 스크러빙 이전 답변 — P12/P13 경고 계산에 사용
    generator = Generator.get()
    answer, citations, answer_pre_scrub = generator.generate(body.query, retrieved)

    # Step 3: 할루시네이션 경고 수집
    warnings: list[str] = []
    # (a) citation 메타 누락 여부
    for c in citations:
        missing = [f for f in ("section_header", "page_number") if getattr(c, f) is None]
        if missing:
            warnings.append(f"{c.source_filename}: {', '.join(missing)} missing")
    # (b) 비교 쿼리단에서 한쪽 entity 누락 여부
    warnings += _single_source_warnings(body.query, citations)
    # (c) P12: 메트릭 레이블-값 불일치 검사 (스크러빙 이전 답변 기준)
    warnings += _check_metric_fidelity(answer_pre_scrub, retrieved)
    # (d) P13: 답변의 퍼센트 숫자가 컨텍스트에 없는지 검사 (스크러빙 이전 답변 기준)
    warnings += _check_numeric_existence(answer_pre_scrub, retrieved)

    # include_chunks=true 시 retrieved 청크 원문을 크라이언트에 전달 (디버깅용)
    retrieved_chunks = (
        [meta["text"] for meta, _score in retrieved if "text" in meta]
        if body.include_chunks else []
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        status="partial" if warnings else "ok",  # 경고 있으면 "partial"
        warnings=warnings,
        retrieved_chunks=retrieved_chunks,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(
    file: UploadFile = File(...),
    _user: dict = Depends(require_role(Role.LAB_PI)),
) -> IngestResponse:
    """Upload and index a new document (PDF, Markdown, or plain text).

    Uses a temporary file path during upload to prevent partial-write
    corruption of the final file. On failure, the temp file is cleaned up.

    Args:
        file: Uploaded file (multipart/form-data).

    Returns:
        IngestResponse with filename and number of chunks indexed.

    Raises:
        422 for unsupported file types or parsing errors.
        500 for unexpected ingestion failures.
    """
    # Bug 2 fix: Path().name으로 디렉토리 컴포넌트 제거 — path traversal 공격 방지
    # 예: file.filename = "../../config.py" → safe_filename = "config.py" (그래도 확장자 검사에서 차단됨)
    safe_filename = Path(file.filename).name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_SUFFIXES}",
        )

    # 업로드 도중 부분 저장된 파일이 노출되는 사태 방지용 임시 경로
    tmp_path = settings.data_raw_dir / f"_tmp_{uuid.uuid4().hex}{suffix}"
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 임시 경로에 저장 후 최종 경로로 이동
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        final_path = settings.data_raw_dir / safe_filename
        # Flaw 4 fix: shutil.move()는 Windows에서도 덮어쓰기 보장
        # Path.rename()은 Windows에서 목적지 파일 존재 시 FileExistsError 발생
        shutil.move(str(tmp_path), str(final_path))

        # 청크 분할 → 임베딩 → FAISS 인덱스 삽입
        chunks_indexed = ingest_file(final_path)
    except (ValueError, RuntimeError) as exc:
        tmp_path.unlink(missing_ok=True)  # 실패 시 임시 파일 정리
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

    return IngestResponse(filename=safe_filename, chunks_indexed=chunks_indexed)


# ── Documents List ──────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResponse, tags=["RAG"])
async def list_documents(
    _user: dict = Depends(require_role(Role.RESEARCHER)),
) -> DocumentListResponse:
    """List all indexed documents with per-document chunk counts.

    Aggregates chunk-level metadata from VectorStore into document-level
    summaries. Documents are sorted alphabetically by filename.

    Returns:
        DocumentListResponse with documents list and total chunk count.
    """
    store = VectorStore.get()
    # 청크 레벨 메타데이터를 문서 레벨로 집계
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
        seen[doc_id].chunk_count += 1  # 같은 문서의 청크마다 카운터 증가

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

def _do_rebuild() -> RebuildResponse:
    """Synchronous full-index rebuild. Called via ``run_in_threadpool``.

    Iterates all files in ``settings.data_raw_dir``, re-parses and re-embeds
    them, then replaces the FAISS index atomically. Files that fail are logged
    and skipped so a single corrupt PDF cannot abort the entire rebuild.

    Returns:
        RebuildResponse with documents_reindexed count and chunks_total.
    """
    store = VectorStore.get()
    embedder = Embedder.get()

    all_vectors: list = []
    all_meta: list[dict] = []
    doc_count = 0
    # Flaw 5 fix: 실패 파일 추적 (silent swallow 제거)
    failed_files: list[dict] = []

    for file_path in sorted(settings.data_raw_dir.iterdir()):
        suffix = file_path.suffix.lower()
        # 지원 형식이 아니거나 업로드 중 임시 파일은 건너땁
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
                # PDF: 페이지 단위 파싱 → 섹션 헤더 추적 → 청크 임베딩
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
                        # 청크 텍스트에서 번호매긴 섹션을 다시 탐지 → 섹션 헤더 업데이트
                        override = _extract_numbered_section(chunk["text"])
                        if override:
                            current_sec = override
                        chunk["metadata"]["section_header"] = current_sec
                    texts = [c["text"] for c in page_chunks]
                    vectors = embedder.embed(texts)
                    all_vectors.append(vectors)
                    all_meta.extend([{"text": c["text"], **c["metadata"]} for c in page_chunks])
            else:
                # 텍스트 파일(md, txt): 전체 파싱 후 청크 분할
                full_text = _parse_text(file_path)
                if not full_text.strip():
                    continue
                chunks = chunk_text(full_text, base_meta)
                texts = [c["text"] for c in chunks]
                vectors = embedder.embed(texts)
                all_vectors.append(vectors)
                all_meta.extend([{"text": c["text"], **c["metadata"]} for c in chunks])

            doc_count += 1
        except Exception as exc:
            # Flaw 5 fix: 실패 파일 기록 + 로그 출력 (silent swallow 제거)
            failed_files.append({"file": file_path.name, "reason": str(exc)})
            continue

    if all_vectors:
        # 모든 벡터를 하나의 float32 행렬로 통합 후 인덱스 전체 교체
        combined = np.vstack(all_vectors).astype(np.float32)
        store.rebuild(combined, all_meta)

    # 내용이 바뀌었으니 기존 제안 쿼리 캐시 무효화
    _SUGGEST_CACHE_PATH.unlink(missing_ok=True)

    if failed_files:
        logger.warning("Rebuild: %d file(s) failed: %s", len(failed_files), failed_files)

    return RebuildResponse(
        documents_reindexed=doc_count,
        chunks_total=len(all_meta),
    )


@app.post("/admin/rebuild-index", response_model=RebuildResponse, tags=["Admin"])
async def rebuild_index(
    _user: dict = Depends(require_role(Role.ADMIN)),
) -> RebuildResponse:
    """Re-parse and re-embed all documents in data/raw/ to rebuild the FAISS index.

    Runs in a thread pool (``run_in_threadpool``) to prevent blocking the
    FastAPI event loop during long-running PDF parsing and embedding operations
    (≈325 docs × embed latency = several minutes).

    Side effects:
        - Clears the suggest-queries cache (suggested_queries.json)
        - Replaces the existing FAISS index and metadata JSON entirely

    Returns:
        RebuildResponse with documents_reindexed count and chunks_total.
    """
    # Flaw 3 fix: 동기 루프를 threadpool으로 위임 → 이벤트 루프 블로킹 방지
    return await run_in_threadpool(_do_rebuild)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    # Improvement 3 fix: 내부 오류 문자열 클라이언트 노출 차단
    # 파일시스템 경로, 모듈명, 변수값 등이 공격자에게 노출될 수 있음
    # 서버 로그에만 상세 기록, 클라이언트에는 제네릭 메시지만 전달
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            reason="An unexpected error occurred. Please try again later.",
        ).model_dump(),
    )
