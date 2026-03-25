# How to Run — OnDevice Scholar RAG

**Version:** 1.2.0  
**Target Platform:** macOS (Apple Silicon M-series, MPS)  
[한국어 버전은 하단을 참조하세요.](#한국어-버전)

---

## Prerequisites

| Item | Minimum |
| :--- | :--- |
| Python | 3.11+ |
| macOS | Ventura 13.0+ |
| Unified Memory | 8 GB recommended |
| Free Disk Space | 10 GB+ (including model weights) |

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `bitsandbytes` 4-bit quantization is unstable on Apple Silicon MPS.  
> `generator.py` automatically applies `float16` fallback on MPS.

---

## Step 2 — Pre-download Model Weights

Download model weights before first offline use.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Generation model
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Embedding model
SentenceTransformer("BAAI/bge-small-en-v1.5")
```

---

## Step 3 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
SECRET_KEY=your-secret-key-here          # JWT signing key (32+ characters recommended)
ACCESS_TOKEN_EXPIRE_MINUTES=60
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-admin-password
```

> ⚠️ Never commit `.env` to Git. It must be listed in `.gitignore`.

---

## Step 4 — Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Interactive API docs (dev mode): `http://localhost:8000/docs`

---

## Step 5 — Get a Token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-admin-password"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

Raw response:

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

---

## Step 6 — Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/raw/paper.pdf"
```

Response:

```json
{
  "request_id": "uuid-string",
  "filename": "paper.pdf",
  "chunks_indexed": 42,
  "status": "ok"
}
```

---

## Step 7 — Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the proposed methodology?", "top_k": 3}'
```

Response (normal):

```json
{
  "request_id": "uuid-string",
  "answer": "The Transformer achieved 28.4 BLEU on WMT 2014. [Source: Attention_Is_All_You_Need.pdf | Section: 6 Results | p.8]",
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

When one side is missing in a comparison query:

```json
{
  "status": "partial",
  "warnings": ["Comparison query detected: no source related to 'GPTQ' retrieved"]
}
```

When no relevant documents found:

```json
{
  "request_id": "uuid-string",
  "answer": "No relevant information found in the provided documents.",
  "citations": [],
  "status": "ok",
  "warnings": []
}
```

---

## Admin Commands

### Delete a document

```bash
curl -X DELETE http://localhost:8000/document/<document_id> \
  -H "Authorization: Bearer $TOKEN"
```

### Rebuild the full index

```bash
curl -X POST http://localhost:8000/admin/rebuild-index \
  -H "Authorization: Bearer $TOKEN"
```

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## RBAC Summary

| Role | Allowed Endpoints |
| :--- | :--- |
| **Researcher** | `GET /health`, `POST /query` |
| **Lab PI** | + `POST /ingest`, `DELETE /document/{id}` |
| **Admin** | + `POST /admin/rebuild-index`, full settings access |

---

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Model load failure | Out of memory | Close other apps and retry. Minimum 8 GB free required |
| `bitsandbytes` error | MPS incompatibility | float16 fallback applied automatically. Check logs |
| FAISS index load failure | Corrupted index file | Call `POST /admin/rebuild-index` |
| `section_header: null` returned | Font/bold detection failure | Standard arXiv PDFs are auto-detected after `rebuild-index` |
| `status: partial` returned | Missing metadata or one-sided retrieval | Check content of `warnings` field |
| PDF parse failure | Encrypted or corrupted PDF | Check `detail` field in error response |

---

## 한국어 버전

### 사전 요구사항

| 항목 | 최소 버전 |
| :--- | :--- |
| Python | 3.11+ |
| macOS | Ventura 13.0+ |
| 통합 메모리 | 8GB 이상 권장 |
| 디스크 여유 공간 | 10GB 이상 (모델 가중치 포함) |

### 실행 순서

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 모델 가중치 사전 다운로드 (최초 1회)
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
SentenceTransformer('BAAI/bge-small-en-v1.5')
"

# 3. .env 생성 (SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD 설정)

# 4. 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. 토큰 발급
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 6. 문서 인덱싱
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@paper.pdf"

# 7. 쿼리
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "제안된 방법론은 무엇인가요?", "top_k": 3}'
```

### 역할별 권한

| 역할 | 허용 엔드포인트 |
| :--- | :--- |
| **Researcher** | `GET /health`, `POST /query` |
| **Lab PI** | + `POST /ingest`, `DELETE /document/{id}` |
| **Admin** | + `POST /admin/rebuild-index`, 모든 설정 접근 |

### 문제 해결

| 증상 | 원인 | 해결 |
| :--- | :--- | :--- |
| 모델 로드 실패 | 메모리 부족 | 다른 앱 종료 후 재시도. 최소 8GB 여유 필요 |
| `bitsandbytes` 오류 | MPS 비호환 | float16 fallback 자동 적용됨. 로그 확인 |
| FAISS 인덱스 로드 실패 | 인덱스 파일 손상 | `POST /admin/rebuild-index` 호출 |
| `section_header: null` 반환 | 폰트/bold 탐지 실패 | `rebuild-index` 후 표준 arXiv PDF는 자동 탐지됨 |
| `status: partial` 반환 | 메타데이터 누락 또는 단측 retrieval | `warnings` 필드 내용 확인 |
| PDF 파싱 실패 | 암호화/손상된 PDF | 에러 응답의 `detail` 필드 확인 |
