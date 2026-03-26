# OnDevice Scholar RAG — 업무 일지

---

## 2026-03-24 (Mon)

**작업자:** Edwin Cho
**전체 작업 시간:** 11:00 — 22:30 KST
**주요 목표:** On-Device RAG 파이프라인 설계·구현·정제·문서화 전 과정

---

## 오전 세션 (11:00 — 오후)

### 1. 프로젝트 초기 설계 및 구조 확립

#### 아키텍처 결정

| 구성 요소 | 선택 | 이유 |
| --- | --- | --- |
| LLM | Qwen2.5-3B-Instruct | On-device 크기 + 한/영 지원 |
| Embedding | BAAI/bge-small-en-v1.5 (384d) | 경량 + 높은 검색 정확도 |
| Vector DB | FAISS IndexFlatIP | 설치 불필요, 인메모리, cosine sim |
| Backend | FastAPI + uvicorn | 비동기 I/O, Pydantic 검증 |
| PDF 파싱 | PyMuPDF (fitz) | font/bold 메타데이터 접근 가능 |
| 인증 | JWT (HS256) + RBAC | 역할별 엔드포인트 접근 제어 |

#### 프로젝트 디렉토리 구조 확정

```text
OnDevice_Scholar_RAG/
├── app/
│   ├── main.py              # FastAPI 앱, 라우터
│   ├── config.py            # Pydantic Settings
│   ├── models/schemas.py    # API 스키마 (QueryRequest, Citation 등)
│   └── pipeline/
│       ├── ingest.py        # PDF 파싱, 섹션 헤더 탐지, 청킹, 저장
│       ├── chunker.py       # RecursiveCharacterTextSplitter
│       ├── embedder.py      # SentenceTransformer 래퍼
│       ├── store.py         # FAISS VectorStore
│       ├── retriever.py     # 쿼리 임베딩 + 벡터 검색
│       └── generator.py     # Qwen2.5 추론, citation 생성
├── data/raw/                # 원본 PDF/TXT 저장 경로
├── data/index/              # FAISS 인덱스 + 메타데이터 저장
├── docs/how_to_run.md
├── requirements.txt
└── README.md
```

#### RBAC 역할 정의

| 역할 | 허용 엔드포인트 |
| --- | --- |
| `admin` | 전체 (rebuild-index, delete, ingest, query) |
| `lab_pi` | ingest, query |
| `viewer` | query 전용 |

---

### 2. 핵심 파이프라인 구현

#### `ingest.py` — PDF 파싱 및 청킹

- `_make_document_id()`: SHA-256 기반 16자리 문서 ID
- `_extract_arxiv_id()`: 파일명 패턴 기반 arXiv ID 추출 (초기 버전)
- `_detect_section_header()`: PyMuPDF `dict` 모드로 폰트 크기 + bold 플래그 탐지
  - body text보다 1.05배 이상 크거나 bold + 섹션 패턴이면 헤더로 인정
  - CMBold, NimbusSanL-Bold 등 arXiv PDF 폰트명 대응
- `_extract_numbered_section()`: 청크 텍스트에서 번호 패턴 보완 탐지
  - `"3.2 Attention"`, `"3.2.1 Scaled Dot-Product Attention"` 등
  - 2줄 결합 패턴 `"3.2\nAttention"` 지원 (다단 레이아웃 PDF)
- `ingest_file()`: PDF/TXT → 청킹 → 임베딩 → FAISS 저장

#### `store.py` — VectorStore

- FAISS `IndexFlatIP` (L2 정규화 벡터로 cosine similarity)
- `search()`: score threshold 0.30 이하 결과 필터링
- `rebuild()`: 전체 인덱스 재빌드
- JSON 직렬화로 메타데이터 영속화 (`data/index/metadata.json`)

#### `generator.py` — 추론 엔진

- Qwen2.5-3B-Instruct: MPS (Apple Silicon) float16, CUDA 4-bit NF4
- Prompt engineering:
  - 컨텍스트 청크를 `[Source: filename | Section: X | p.N]` 형태로 주입
  - Verbatim 복사 금지, 대명사 → 고유명사 치환 지시
- `citation_min_score: 0.65` 이상인 청크만 citation 포함
- `_build_citations()`: 파일 단위 중복 제거 후 Citation 객체 생성

#### `main.py` — API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
| --- | --- | --- |
| `/auth/token` | POST | JWT 토큰 발급 |
| `/query` | POST | RAG 쿼리 (검색 + 생성 + citations) |
| `/ingest` | POST | 파일 업로드 + 인덱싱 |
| `/admin/rebuild-index` | POST | 전체 인덱스 재빌드 |
| `/admin/delete-document` | DELETE | 특정 문서 인덱스 삭제 |
| `/health` | GET | 서버 상태 확인 |

---

### 3. P6 — Post-generation Citation Pruning

**배경:** LLM이 답변에 언급하지 않은 출처도 `citations`에 포함되는 노이즈 문제

**구현 (`generator.py`):**

- `_extract_cited_sources()`: 답변 텍스트에서 `[Source: filename | ...]` 패턴 파싱
- `generate()` 내 pruning 로직:
  1. LLM 답변 생성 후 인라인 인용 파싱
  2. 실제 언급된 `source_filename`만 citations에 포함
  3. 파싱 결과 없으면 전체 반환 (안전 폴백)

**결과:** 관련 없는 citation이 응답에서 제거됨

---

### 4. P7 — Single-source Bias Detection

**배경:** 비교 쿼리 ("A vs B", "A and B compare")에서 한쪽 논문 청크만 검색될 때 편향된 답변 생성

**구현 (`main.py`):**

- 쿼리에 비교 키워드 (`vs`, `compare`, `difference between` 등) 포함 여부 탐지
- 검색 결과가 단일 `source_filename`에서만 나올 경우:
  - `status: "partial"` 반환
  - `warnings: ["Single-source bias detected: only X was retrieved..."]` 추가
- `QueryResponse` 스키마에 `warnings: List[str]` 필드 추가

---

### 5. P8 — Section Header 탐지 (1차 구현)

**배경:** 청크에 `section_header` 메타데이터가 없으면 citation 품질 저하

**구현 (`ingest.py`):**

- `_detect_section_header()`: 페이지별 폰트 크기 분포 분석
  - 최빈 body size 계산 (Counter 활용)
  - body size × 1.05 초과 or bold + 섹션 네이밍 패턴 → 헤더로 인정
  - 러닝 헤더(페이지 상단 반복 텍스트) 자동 제거
- `_extract_numbered_section()`: 청크 텍스트 기반 보완 탐지
- Section header forward propagation: 새 헤더 탐지 시 다음 페이지까지 유지

---

### 6. 코드베이스 리팩토링 및 정리

**제거된 디렉토리:**

| 경로 | 이유 |
| --- | --- |
| `LiE/` | 실험용 임시 코드, 미사용 |
| `adapter/` | LoRA adapter 관련, `use_adapter=False`로 비활성 |
| `colab/` | Colab 노트북 스크래치, 미사용 |
| `ui/` | Gradio UI 실험 코드, 미사용 |

**제거된 dead code:**

- `_extract_section_header()` stub 함수 (실제 구현 없이 `pass`만 존재)

**`requirements.txt` 정리:**

| 제거 패키지 | 이유 |
| --- | --- |
| `PyPDF2` | PyMuPDF(fitz)로 대체됨 |
| `pyyaml` | 프로젝트 전체에서 미사용 |
| `peft` | `use_adapter=False`, 실제 로드 코드 없음 |

---

## 저녁 세션 (21:00 — 22:30)

### D — 문서화 업데이트 (영문/한국어 이중언어)

#### `README.md`

- 영문 기본 구조로 전면 재구성, 하단에 `## 한국어 버전` 섹션 추가
- 쿼리 응답 예시 현행화: `paper_title`, `arxiv_id`, `score`, `warnings` 필드 반영
- Key Features 테이블에 P6/P7/P8 추가
- 삭제된 `adapter/` 디렉토리 참조 제거

#### `docs/how_to_run.md`

- Step 1~7 영문 구조로 전환, 하단에 한국어 요약 섹션 추가
- curl 예시 전체를 `$TOKEN` 변수 방식으로 통일
- Response 레이블 / Troubleshooting 표 영문화

---

### A — 스트레스 테스트 + 버그 수정

#### 테스트 결과 (TC-1 ~ TC-9)

| TC | 쿼리 유형 | 결과 | 판정 |
| --- | --- | --- | --- |
| TC-1 | 단순 사실 ("What is attention?") | 정확한 단일 소스 답변 | ✅ |
| TC-2 | 특정 논문 질문 ("Reformer의 핵심 기법") | LSH attention 정확히 답변 | ✅ |
| TC-3 | 비교 쿼리 ("Reformer vs Longformer") | 양쪽 citation 포함, P7 정상 | ✅ |
| TC-4 | 다중 홉 ("attention 개선 논문들의 공통점") | 복수 소스 통합 답변 | ✅ |
| TC-5 | 존재하지 않는 정보 ("GPT-5 architecture") | "No relevant information" 정상 | ✅ |
| TC-6 | 단일 소스 비교 쿼리 | `status: partial` + warnings 반환 | ✅ |
| TC-7 | 효율적 Transformer 포괄 쿼리 (top_k=10) | LLM 인라인 인용 없음 → 무관 citation 7개 | ⚠️ 수정 |
| TC-8 | 수식 포함 쿼리 ("O(n²) 문제") | BigBird만 반환, 정상 | ✅ |
| TC-9 | 빈 쿼리 (`""`) | 422 Unprocessable Entity | ✅ |

#### 수정 1 — TC-7: Citation Pruning Hybrid Fallback

**문제:** LLM이 포괄적 종합 쿼리에서 `[Source: filename | ...]` 인라인 포맷 미사용
→ `_extract_cited_sources()` 파싱 실패 → 전체 citation 7개 반환 (노이즈 포함)

**해결 (`generator.py`):**

- `_TITLE_STOPWORDS`: 빈도 높은 일반 단어 필터셋
- `_match_citations_by_title()`: paper_title 핵심 키워드 → 답변 텍스트 매칭
- 3단계 폴백 구조:
  1. `[Source:...]` 인라인 파싱 (기존)
  2. paper_title 키워드 매칭 (신규)
  3. 전체 반환 (안전 폴백)

**결과:** 7개 → 1개 (Reformer.pdf만 정확히 추출) ✅

#### 수정 2 — P8: Section Header 노이즈 필터

**문제:** 표 행, 참고문헌 줄, 차트 레이블이 `section_header`로 오탐

- `"9.56 RTN"` — 양자화 성능 표 행
- `"12.46 Absmax LLM.int8() (vector-wise + decomp)"` — 표 행
- `"G. Gemini Team. URL https:..."` — 참고문헌 줄

**해결 (`ingest.py`):**

`_NOISE_PATTERNS` 정규식 + `_is_noise_header()` 함수:

- URL / DOI / arXiv 문자열 패턴
- 소수 2자리 이상 숫자 시작 (점수 메트릭)
- `(vector-wise|row-wise|column-wise)` 양자화 표 설명
- 마침표 3개 이상 + 50자 초과 (참고문헌 줄)
- Float 파싱 휴리스틱: `"9 .56 RTN"` (PyMuPDF 스팬 분리 케이스) 대응

**핵심 아키텍처 개선:**
필터 위치를 **인덱싱 시점 → 쿼리 시점** (`_build_citations()`)으로 이동
→ 이후 필터 튜닝 시 rebuild 불필요, 서버 reload만으로 즉시 반영

**결과:** `"9.56 RTN"` → `null`, `"12.46 Absmax..."` → `null` ✅

---

### B — arxiv_id 자동 추출

**기존:** 파일명이 `1706.03762.pdf` 형식일 때만 추출

**신규 3단계 추출 (`ingest.py`):**

1. PDF 메타데이터 스캔 (`title` / `subject` / `keywords` / `creator`)
2. 첫 2페이지 본문에서 `arXiv:XXXX.XXXXX` 또는 `arxiv.org/abs/XXXX.XXXXX` 탐지
3. 파일명 기반 폴백 (기존)

**아키텍처 변경:**

- `_parse_pdf()` 반환값: `list[...]` → `tuple[list[...], Optional[str]]`
- `ingest_file()` + `rebuild_index()`: `detected_arxiv_id`로 `base_metadata` 업데이트
- `_build_citations()` (`generator.py`): `arxiv_id=None`이면 파일명 기반 쿼리 폴백 → rebuild 없이 즉시 반영

**검증:**

```text
BERT.pdf | 1810.04805   ← PDF 내부 텍스트에서 자동 추출 ✅
```

---

### 부가 작업

| 항목 | 변경 전 | 변경 후 | 파일 |
| --- | --- | --- | --- |
| JWT 토큰 만료 시간 | 60분 | **480분** (8시간) | `app/config.py` |
| `adapter_dir` 미사용 필드 | 존재 | 제거 | `app/config.py` |
| uvicorn 강제 종료 | `Ctrl+C` 루프 실패 | `pkill -f "uvicorn app.main:app"` | — |

---

---

## 2026-03-25 (Tue)

**작업자:** Edwin Cho
**작업 시간:** 12:00 — 12:35 KST
**주요 목표:** P9 / P10 / P11 완료

---

### P9 — Figure/표 Caption 오태깅 수정

**문제:** PyMuPDF가 bold + 큰 폰트로 렌더링된 caption (`Figure 1:`, `Table 2.`, `Algorithm 3`)을 section_header로 오탐

**해결 (`ingest.py`):**

`_NOISE_PATTERNS`에 caption 패턴 추가:

```python
re.compile(r'^(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?|Listing)\s+\d+', re.IGNORECASE)
```

- P8과 동일한 쿼리 시점 필터 경로 (`_build_citations`) 사용 → rebuild 불필요
- 정상 헤더 (`2 Related Work`, `3 Experiments`) 영향 없음 ✅

---

### P10 — Cross-domain Semantic Leakage 완화

**원인 분석:**

| 원인 | 설명 |
| --- | --- |
| 절대 threshold만 존재 | top-1 score가 0.82여도 score 0.31짜리 cross-domain 청크가 LLM 컨텍스트에 포함 |
| 공유 어휘 | "attention", "layer", "training" 등 NLP 전반 공통 용어로 임베딩 거리 좁아짐 |
| LLM 컨텍스트 오염 | citation에 없어도 검색된 청크가 답변 생성에 영향 |

**해결 — 상대 score gap 필터:**

- `config.py`에 `retrieval_score_gap: float = 0.25` 추가
- `retriever.py`에서 top-1 score 기준 gap 초과 청크 제거

```python
if results:
    top_score = results[0][1]
    cutoff = top_score - settings.retrieval_score_gap
    results = [(meta, score) for meta, score in results if score >= cutoff]
```

**전체 retrieval 필터 체인 (완성):**

```text
FAISS search (top_k)
  → 절대 threshold ≥ 0.30         [store.py]
  → 상대 score gap (top - 0.25)   [retriever.py]  ← P10 신규
  → LLM 컨텍스트 + 생성
  → citation_min_score ≥ 0.65     [generator.py]
  → P6 inline citation pruning
  → TC-7 paper_title keyword fallback
  → P11 keyword contribution filter
```

**검증:** `top_k=10` 쿼리 → `BERT.pdf (0.815)` 1개만 반환, 나머지 9개 제거 ✅

---

### P11 — 답변-Citation 불일치 추적

**문제:** P6/TC-7 통과 후에도 LLM 답변에 실제 기여하지 않은 청크가 citation으로 남는 경우

**해결 (`generator.py`):**

- `_CONTRIBUTION_STOPWORDS`: 고빈도 일반 단어 제거셋
- `_extract_keywords()`: 4글자 이상 알파벳 단어 추출
- `_filter_by_contribution()`:
  - 답변 키워드 ∩ 청크 텍스트 키워드 < 2개 → citation 제거
  - 답변 키워드 수 < 6개 (너무 짧은 답변) → 검사 스킵
  - 전체 제거 시 원본 반환 (안전 폴백)
- `generate()` 마지막 단계에서 적용

**검증:** `top_k=5` → `BERT.pdf` 1개만 유지, 기여 없는 4개 제거 ✅

---

---

## 2026-03-25 저녁 세션 (20:00 — 21:30 KST)

**작업자:** Edwin Cho
**주요 목표:** Phase 2 — React Web UI 구현 + 프롬프트 품질 개선 + 마무리 정리

---

### Phase 2 — Web UI (SRS v2.0.0 전체 구현)

#### 기술 스택

| 구성 요소 | 선택 | 이유 |
| --- | --- | --- |
| 프레임워크 | React 18 + Vite | 빠른 HMR, TypeScript 지원 |
| 스타일링 | Tailwind CSS v4 | 유틸리티 클래스 기반, 다크 테마 용이 |
| 아이콘 | Lucide React | 경량 SVG 아이콘 셋 |
| HTTP 클라이언트 | Axios | Interceptor 기반 Bearer token 자동 주입 |
| 라우팅 | React Router v6 | 중첩 라우트 + 인증 가드 |
| 수식 렌더링 | KaTeX | LaTeX 수식 클라이언트 사이드 렌더링 |

#### 구현된 페이지

| 페이지 | 경로 | 주요 기능 |
| --- | --- | --- |
| `LoginPage` | `/login` | JWT 발급, 로그인 에러 처리, 글로우 애니메이션 배경 |
| `QueryPage` | `/` | 채팅형 UI, KaTeX 수식 렌더링, 글래스모피즘 citation 카드, score bar |
| `DocumentsPage` | `/documents` | PDF/TXT 업로드 + 문서 삭제 |
| `AdminPage` | `/admin` | 전체 인덱스 rebuild 트리거 |

#### 컴포넌트 구조

```text
frontend/src/
├── App.tsx                  # React Router + AuthProvider 래핑
├── index.css                # Tailwind v4 import + CSS 변수
├── main.tsx                 # React 앱 진입점
├── components/
│   ├── Layout.tsx           # 사이드바 + health badge + role guard 네비
│   └── MathRenderer.tsx     # KaTeX 수식 파서 + 렌더러
├── contexts/
│   └── AuthContext.tsx      # JWT 상태 관리 + role 디코딩
├── lib/
│   ├── api.ts               # Axios 인스턴스 + API 함수
│   └── types.ts             # QueryResponse, Citation 등 타입 정의
└── pages/
    ├── LoginPage.tsx
    ├── QueryPage.tsx
    ├── DocumentsPage.tsx
    └── AdminPage.tsx
```

#### UI 디자인 특징

- 전체 다크 테마 (`#0a0a0f` 배경)
- 사이드바 글래스모피즘 (`rgba(255,255,255,0.025)`)
- 로그인 페이지: 퍼플 그라디언트 글로우 orb 애니메이션
- Citation 카드: 반투명 배경 + 스코어 퍼센트 bar
- 사이드바 Health badge: 30초 폴링, Online(녹색) / Offline(빨강) 실시간 표시

---

### 프롬프트 엔지니어링 개선

#### 문제 1 — 수치 부정확 ("various pre-training tasks")

**해결 (`generator.py` SYSTEM_PROMPT):**
- 정확한 수치 사용 강제 규칙 추가
- 금지어 리스트 (`various`, `several`, `multiple`, `some`) 명시
- "특정 수치가 없으면 생략" 지시

#### 문제 2 — 인라인 citation 미삽입

**해결 (`generator.py` user_message):**
- 실제 쿼리 메시지 앞에 few-shot Q/A 예시 삽입
- 예시 형식: `[Source: BERT.pdf | Section: 3.1 Pre-training Tasks | p.4]`
- 모델이 패턴을 모방해 인라인 citation 생성하도록 유도

**결과:**
- 1차: `"various pre-training tasks"` ❌ → citation 없음 ❌
- 2차: `"two main tasks"` ✅ → citation 없음 ❌
- 3차: `"two pre-training tasks"` ✅ → `[Source: BERT.pdf | Section: 2.3 | p.3]` ✅

---

### JWT Role Guard

**구현:**
- `AuthContext.tsx`: `decodeRole()` — JWT payload `atob()` 디코딩 → `role` 필드 추출
- 역할 계층: `researcher(1) < lab_pi(2) < admin(3)`
- `Layout.tsx`: `hasAccess(role, minRole)` 필터로 nav 아이템 조건부 렌더링

| 역할 | 표시되는 탭 |
| --- | --- |
| `researcher` | Query |
| `lab_pi` | Query, Documents |
| `admin` | Query, Documents, Admin |

---

### KaTeX 수식 렌더링

**구현 (`MathRenderer.tsx`):**

- 지원 패턴: `$...$` (인라인), `$$...$$` (블록), `\(...\)`, `\[...\]`
- 정규식 파싱 → Segment 배열 분리 → KaTeX `renderToString()` 적용
- 렌더 실패 시 `[LaTeX Error]` fallback (빨간 텍스트)
- `QueryPage.tsx` 답변 버블에 적용

---

### 디렉토리 정리 + README 최신화

**삭제된 파일 (Vite 템플릿 잔재):**

| 경로 | 이유 |
| --- | --- |
| `frontend/src/App.css` | 미사용 스타일 |
| `frontend/src/assets/` | 템플릿 SVG/PNG (react.svg, vite.svg, hero.png) |
| `frontend/public/icons.svg` | 미사용 |

**README.md 업데이트:**
- Project Structure에 `frontend/` 디렉토리 추가
- Quick Start에 Web UI 실행 명령어 추가 (영문 + 한국어 섹션 모두)

---

### Git 커밋 이력

| 커밋 해시 | 메시지 |
| --- | --- |
| `16dd31f` | `feat: Phase 2 Web UI + prompt precision upgrade` |
| `f186e9f` | `docs: add frontend start command to Korean quick start` |
| `41a2b83` | `feat: KaTeX math rendering + JWT role guard` |

---

---

## 2026-03-26 (Wed)

**작업자:** Edwin Cho
**작업 시간:** 19:45 — 21:20 KST
**주요 목표:** 시스템 실동작 테스트 + KaTeX 수식 포맷 버그 수정 + UI 디자인 리뷰

---

### 실동작 테스트 쿼리

서버 재기동 후 프론트엔드(`http://localhost:5173`)에서 직접 쿼리 테스트 진행.

| 쿼리 | 결과 | 판정 |
| --- | --- | --- |
| `Performance benchmarks for academic RAG systems` | "No relevant information found" | ✅ 정상 폴백 (해당 논문 없음) |
| `What are the main components of a Transformer architecture?` | 정확한 답변 + `[Source: Attention_Is_All_You_Need.pdf \| Section: 3.1 \| p.3]` | ✅ |
| `What is the computational complexity of self-attention compared to recurrent layers?` | `O(n^2)` (LaTeX 미적용) | ⚠️ → 수정 |

---

### 수식 포맷 버그 수정

**문제:** LLM이 수학 표현을 `O(n^2)` plain text로 출력 → KaTeX 인식 불가

**해결 (`generator.py` SYSTEM_PROMPT Rule 8 추가):**

```python
8. MATH — Wrap all mathematical expressions in LaTeX delimiters:
   - Inline: $O(n^2)$, $d_{\text{model}}$, $\sqrt{d_k}$
   - Block: $$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$$
   - Never write bare: O(n^2), d_model
```

**결과:** `$O(n^2)$` 출력 → KaTeX 정상 렌더링 ✅

**커밋:** `8ebd004` — `feat: enforce LaTeX math delimiters in SYSTEM_PROMPT for KaTeX rendering`

---

### UI 디자인 리뷰 문서 작성

전체 프론트엔드 코드 분석 후 `LiE_Design.md` 작성:

- 페이지별 현행 구조 + CSS 특징 정리
- 아쉬운 점 항목화 (11개 개선 사항)
- 우선순위별 분류 (🔴 High / 🟡 Medium / 🟢 Low)
- 구현 난이도 × 효과 매트릭스

**주요 개선 후보:**

| 우선순위 | 항목 |
| --- | --- |
| 🔴 | Top-K 위치 이동 (헤더 → 입력창 옆) |
| 🔴 | Documents 페이지 — 인덱싱된 문서 목록 조회 |
| 🔴 | QueryPage Empty state — 예시 쿼리 칩 |
| 🟡 | Citation 카드 접기/펼치기 토글 |
| 🟡 | 사이드바 — 로그인 사용자 정보 표시 |
| 🟡 | 답변 복사 버튼 + Clear conversation 버튼 |
| 🟢 | `w-58` → `w-60` 수정, auto-resize textarea |

---

## 미완료 / 다음 세션 후보

| ID | 내용 | 우선순위 |
| --- | --- | --- |
| UI-① | QueryPage Top-K 입력창 옆으로 이동 | 🔴 높음 |
| UI-② | DocumentsPage 인덱싱 문서 목록 조회 | 🔴 높음 |
| UI-③ | QueryPage Empty state 예시 쿼리 칩 | 🔴 높음 |
| UI-④ | Citation 카드 접기/펼치기 | 🟡 중간 |
| UI-⑤ | 사이드바 사용자 정보 (username + role badge) | 🟡 중간 |
| UI-⑥ | 답변 복사 버튼 + Clear 버튼 | 🟡 중간 |
| UI-⑦ | `w-58` → `w-60`, auto-resize textarea | 🟢 낮음 |
| — | 모바일 반응형 레이아웃 (SRS v2.1.0) | � 낮음 |
| — | 다크/라이트 모드 토글 (SRS v2.1.0) | 🟢 낮음 |

---

*작성일: 2026-03-24 ~ 03-26 / 작성자: Edwin Cho*
