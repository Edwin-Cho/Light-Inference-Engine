from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


# ── Shared ─────────────────────────────────────────────────────────────────

def new_request_id() -> str:
    return str(uuid.uuid4())


# ── Auth ────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    filename: str
    chunks_indexed: int
    status: str = "ok"


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    source_filename: str
    paper_title: Optional[str] = None
    arxiv_id: Optional[str] = None
    section_header: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    answer: str
    citations: List[Citation] = []
    status: str = "ok"
    warnings: List[str] = []


# ── Document Delete ─────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    document_id: str
    chunks_removed: int
    status: str = "ok"


# ── Admin ───────────────────────────────────────────────────────────────────

class RebuildResponse(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    documents_reindexed: int
    chunks_total: int
    status: str = "ok"


# ── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


# ── Error ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    status: str = "error"
    error: str
    reason: Optional[str] = None
