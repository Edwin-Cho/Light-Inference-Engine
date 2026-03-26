# OnDevice Scholar RAG

> A privacy-first, fully offline Retrieval-Augmented Generation pipeline for academic research labs.  
> [한국어 버전은 하단을 참조하세요.](#한국어-버전)

---

## Overview

| 항목 | 내용 |
| :--- | :--- |
| **Goal** | Query confidential academic documents locally, without external cloud APIs |
| **Generation Model** | Qwen2.5-3B-Instruct (float16, MPS-optimized) |
| **Embedding Model** | BAAI/bge-small-en-v1.5 (local, on-device) |
| **Vector Store** | FAISS IndexFlatIP (cosine similarity) |
| **Backend** | FastAPI + Uvicorn |
| **Target Platform** | Apple Silicon M-series (MPS) |
| **Document Formats** | PDF, Markdown, TXT |

---

## Architecture

![OnDevice Scholar RAG — System Architecture](docs/images/architecture.png)

---

## Project Structure

```text
OnDevice_Scholar_RAG/
├── app/
│   ├── main.py                 # FastAPI 엔트리포인트
│   ├── config.py               # 중앙 설정 (Top-K, 임계값, 모델 경로 등)
│   ├── auth/
│   │   ├── rbac.py             # Role 기반 접근 제어
│   │   └── token.py            # JWT Bearer token 발급/검증
│   ├── pipeline/
│   │   ├── ingest.py           # Document parsing + metadata extraction
│   │   ├── chunker.py          # Recursive Character Text Splitter
│   │   ├── embedder.py         # Local embedding + L2 normalization
│   │   ├── store.py            # FAISS index management
│   │   ├── retriever.py        # Top-K search + threshold filtering
│   │   └── generator.py        # Qwen2.5-3B inference + Citation Validator
│   └── models/
│       └── schemas.py          # Pydantic request/response schemas
├── data/
│   ├── raw/                    # Uploaded source documents (PDF/MD/TXT)
│   ├── processed/              # Chunk + metadata cache
│   └── index/                  # FAISS index files
├── docs/
│   ├── how_to_run.md           # Run guide
│   ├── evaluation_rubric.md    # Evaluation criteria
│   └── images/
│       └── architecture.png    # System architecture diagram
├── frontend/                   # React Web UI (Phase 2 v2.0.0)
│   ├── src/
│   │   ├── pages/              # Login, Query, Documents, Admin
│   │   ├── components/         # Layout (sidebar + health badge)
│   │   ├── contexts/           # AuthContext (JWT)
│   │   └── lib/                # api.ts, types.ts
│   ├── vite.config.ts
│   └── package.json
├── OnDevice_Scholar_RAG_SRS.md # Software Requirements Specification v1.1.0
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Web UI (optional)

```bash
cd frontend && npm install && npm run dev
# Open http://localhost:5173  (login: admin / admin1234)
```

### 4. Get a token (CLI alternative)

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 5. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@paper.pdf"
```

### 6. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the proposed methodology?", "top_k": 3}'
```

---

## API Endpoints

| Method | Endpoint | Role | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/query` | Researcher+ | Natural language query → answer with citations |
| `POST` | `/ingest` | Lab PI+ | Document upload and indexing |
| `DELETE` | `/document/{id}` | Lab PI+ | Delete document and its chunks |
| `POST` | `/admin/rebuild-index` | Admin | Full vector index rebuild |
| `GET` | `/health` | Public | Service health check |

### Query Response Example

```json
{
  "request_id": "uuid-string",
  "answer": "The Transformer achieved a BLEU score of 28.4 on WMT 2014. [Source: Attention_Is_All_You_Need.pdf | Section: 6 Results | p.8]",
  "citations": [
    {
      "source_filename": "Attention_Is_All_You_Need.pdf",
      "paper_title": "Attention Is All You Need",
      "arxiv_id": null,
      "section_header": "6 Results",
      "page_number": 8,
      "score": 0.7407
    }
  ],
  "status": "ok",
  "warnings": []
}
```

When only one side is retrieved in a comparison query:

```json
{
  "status": "partial",
  "warnings": ["Comparison query detected: no source related to 'GPTQ' retrieved"]
}
```

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **Fully Offline** | No external network requests during indexing, inference, or search |
| **Forced Citation** | Every response must include source (filename · section · page) citations |
| **Citation Pruning** | Inline `[Source:]` parse + paper title keyword fallback — only contributing sources included |
| **Citation Contribution Filter** | Keyword overlap check removes citations with no actual contribution to the answer |
| **Cross-domain Leakage Mitigation** | Absolute score threshold + relative score gap filter (top − 0.25) blocks off-topic chunks |
| **Section Header Detection** | Font-size + bold based section tagging; Figure / Table / Algorithm caption mis-tags filtered |
| **arXiv ID Auto-extraction** | Scans PDF metadata, first 2 pages, then filename — no manual input needed |
| **Bias Detection Warning** | Comparison queries with one-sided retrieval return `status=partial` + warning |
| **Safe Fallback** | Returns "No relevant information found" when context is insufficient |
| **RBAC** | Role-based access control: Researcher / Lab PI / Admin |
| **MPS Optimized** | Prioritized optimization for Apple Silicon M-series |
| **Incremental Indexing** | Add new documents without full index rebuild |
| **Metric Label Fidelity (P12)** | Post-generation check: label–value mismatch against retrieved chunks → `partial` warning |
| **Numeric Existence Check (P13)** | Post-generation check: percentage values absent from retrieved context → hallucination warning |
| **Noise Section Header Filter** | `_is_noise_header` applied at query time to both context block and citations — no re-indexing required |
| **Single-source Badge** | UI amber badge when answer cites only 1 document: "1 source — verify independently" |
| **Hallucination Defense Prompt** | SYSTEM_PROMPT Rules 9–11: verbatim metric names, time-bounded SOTA claims, no fabricated numerics |

---

## Notes

- Model weights (Qwen2.5-3B-Instruct, BAAI/bge-small-en-v1.5) must be pre-downloaded before first offline use.
- `bitsandbytes` 4-bit support may be unstable on Apple Silicon MPS. `float16 + MPS` fallback is applied automatically.

### Document Usage Policy

This system is intended for **non-commercial, internal lab use only**.

| Document Type | Allowed | Notes |
| :--- | :---: | :--- |
| arXiv preprint (CC BY / CC BY-NC-SA) | ✅ | Citation enforced |
| Lab's own unpublished papers | ✅ | Core use case of this system |
| Publisher final PDFs (Elsevier, Springer, etc.) | ⚠️ | Check publisher license separately |
| External commercial service distribution | ❌ | Outside the scope of this system |

- RAG uses vector indexing + retrieval + citation, not model fine-tuning. Document content is not internalized into model weights.
- For publisher PDFs, using the **preprint version** posted by the authors on arXiv is recommended.

---

## License

Apache License 2.0

---

## 한국어 버전

학술 연구 환경을 위한 프라이버시 우선, 완전 오프라인 RAG 파이프라인

### 개요

| 항목 | 내용 |
| :--- | :--- |
| **목표** | 기밀 학술 문서를 외부 클라우드 없이 로컬에서 쿼리하는 RAG 시스템 |
| **생성 모델** | Qwen2.5-3B-Instruct (float16, MPS 최적화) |
| **임베딩 모델** | BAAI/bge-small-en-v1.5 (로컬 온디바이스) |
| **벡터 스토어** | FAISS IndexFlatIP (코사인 유사도) |
| **백엔드** | FastAPI + Uvicorn |
| **타겟 플랫폼** | Apple Silicon M-series (MPS 최적화) |
| **지원 문서 형식** | PDF, Markdown, TXT |

### 핵심 기능

| 기능 | 설명 |
| :--- | :--- |
| **완전 오프라인** | 인덱싱/추론/검색 전 과정에서 외부 네트워크 요청 없음 |
| **인용 강제** | 모든 응답에 출처(파일명·섹션·페이지) 인용 필수 |
| **Citation Pruning** | LLM 인라인 인용 파싱 + paper_title 키워드 폴백으로 실제 기여 문서만 포함 |
| **Citation Contribution 검증** | 답변-청크 키워드 오버랩 기반으로 기여 없는 citation 자동 제거 |
| **Cross-domain Leakage 완화** | 절대 threshold + 상대 score gap 필터(top−0.25)로 무관 도메인 청크 차단 |
| **섹션 헤더 탐지** | 폰트 크기 + bold 기반 자동 섹션 태깅, Figure/Table/Algorithm 캡션 오탐 필터 |
| **arXiv ID 자동 추출** | PDF 메타데이터 · 본문 첫 2페이지 스캔 · 파일명 순으로 arXiv ID 자동 탐지 |
| **편향 감지 경고** | 비교 쿼리에서 한쪽 소스만 검색 시 `status=partial` + warning 반환 |
| **안전 폴백** | 관련 정보 없을 시 "No relevant information found" 반환 |
| **RBAC** | Researcher / Lab PI / Admin 역할 기반 접근 제어 |
| **MPS 최적화** | Apple Silicon M-series 우선 최적화 |
| **증분 인덱싱** | 전체 재빌드 없이 신규 문서 추가 가능 |

### 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Web UI 실행 (선택)
cd frontend && npm install && npm run dev
# http://localhost:5173 접속  (로그인: admin / admin1234)

# 토큰 발급 (CLI 방식)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 문서 인덱싱
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@paper.pdf"

# 쿼리
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "제안된 방법론은 무엇인가요?", "top_k": 3}'
```

### 주의사항

- 모델 최초 실행 전 Qwen2.5-3B-Instruct 및 임베딩 모델 가중치를 사전 다운로드해야 합니다.
- Apple Silicon에서 `bitsandbytes` 4-bit 지원이 불안정할 수 있습니다. `float16 + MPS` fallback이 자동 적용됩니다.
- 이 시스템은 **연구실 내부 비상업적 목적**으로만 사용해야 합니다.
