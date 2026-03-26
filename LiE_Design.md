# OnDevice Scholar RAG — UI Design Review

## 1. 현행 컴포넌트 구성

```
frontend/src/
├── components/
│   ├── Layout.tsx        # 사이드바 + 헬스 배지 + 네비게이션
│   └── MathRenderer.tsx  # KaTeX 인라인/블록 수식 렌더러
└── pages/
    ├── LoginPage.tsx      # 인증 화면
    ├── QueryPage.tsx      # 채팅형 RAG 쿼리 화면
    ├── DocumentsPage.tsx  # 문서 업로드/삭제
    └── AdminPage.tsx      # 인덱스 재빌드
```

---

## 2. 페이지별 현행 디자인 분석

### 2-1. LoginPage

**구조:**

- 전체 화면 중앙 정렬 (`min-h-screen flex items-center justify-center`)
- 배경: `radial-gradient` 퍼플 + 블루 orb + `#0a0a0f` 베이스
- 카드: `backdrop-blur-md` + `bg-white/[0.04]` 글래스모피즘
- 입력 필드: 좌측 아이콘 (`User`, `Lock`) + `focus:ring-purple-500/70`
- 버튼: `bg-gradient-to-r from-purple-600 to-purple-500` + `shadow-purple-500/20`

**CSS 특징:**

```css
배경: radial-gradient(ellipse at 60% 20%, #2d1b6940, transparent 60%)
카드: bg-white/[0.04] border border-white/10 rounded-2xl backdrop-blur-md
버튼: bg-gradient-to-r from-purple-600 to-purple-500 shadow-purple-500/20
```

**아쉬운 점:**

- 로고가 단순 `BookOpen` 아이콘뿐 — 앱 정체성 부족
- 에러 메시지가 텍스트만 (`Invalid credentials`) — UX 개선 여지
- 배경 orb 애니메이션 없음 (정적)
- `"All queries run locally"` 문구가 `text-slate-700` → 거의 안 보임

---

### 2-2. Layout (Sidebar)

**구조:**

- `w-58` 고정 너비 사이드바 (⚠️ Tailwind 비표준 — `w-56`=224px 또는 `w-60`=240px 권장)
- 상단: 로고 + `Scholar RAG` 텍스트 + `HealthBadge`
- 네비: `NavLink` + active 시 `bg-purple-500/15 text-purple-300` + dot indicator
- 하단: `Sign Out` 버튼 (`hover:text-red-400`)

**CSS 특징:**

```css
사이드바 배경: rgba(255,255,255,0.025)
구분선: rgba(255,255,255,0.07)
Active nav: bg-purple-500/15 text-purple-300
Inactive nav: text-slate-500 hover:text-slate-200 hover:bg-white/5
```

**아쉬운 점:**

- `w-58` 비표준 값 → 실제 렌더 너비 불명확
- 현재 로그인된 사용자 이름/role 표시 없음
- 사이드바 하단에 여백만 있고 빈 공간 활용 없음
- 네비 아이템이 3개뿐 → 공백이 너무 많아 밸런스 불균형

---

### 2-3. QueryPage

**구조:**

- 헤더: `Research Query` 타이틀 + 우측 `Top-K` select
- 메시지 영역: `flex-1 overflow-y-auto` 스크롤
- 입력창: `textarea` (2줄) + `Send` 버튼

**CSS 특징:**

```css
헤더: border-b border-slate-800
유저 버블: bg-purple-600 rounded-xl
어시스턴트 버블: bg-slate-800/60 border border-slate-700/40 rounded-xl
Citation 카드: bg rgba(255,255,255,0.03) border rgba(255,255,255,0.07) rounded-xl
Score bar: h-1 rounded-full (보라/인디고/회색 3단계)
```

**아쉬운 점:**

| 항목 | 문제 |
| --- | --- |
| Top-K 위치 | 헤더 우측에 고립 → 맥락 없이 떠 있음 |
| Empty state | 정적 아이콘+텍스트만 → 예시 쿼리 없음 |
| Citation 카드 | 항상 전체 펼침 → 길 경우 스크롤 과부하 |
| 메시지 | 타임스탬프 없음 |
| 대화 초기화 | Clear 버튼 없음 |
| 입력창 | 자동 높이 조절 없음 (fixed 2줄) |
| 텍스트 복사 | 답변 복사 버튼 없음 |

---

### 2-4. DocumentsPage

**구조:**

- 드래그앤드롭 업로드 존 (`border-2 border-dashed rounded-2xl`)
- 삭제: 텍스트 입력 → Delete 버튼

**아쉬운 점:**

| 항목 | 문제 |
| --- | --- |
| 삭제 방식 | 파일명을 직접 타이핑해야 함 → UX 취약 |
| 업로드된 문서 목록 | 현재 인덱싱된 문서 목록 조회 기능 없음 |
| 진행 상태 | 파일 하나씩 순차 업로드, 전체 진행률 없음 |
| 파일 유효성 | 잘못된 파일 타입 피드백 약함 |

---

### 2-5. AdminPage

**구조:**

- 단일 `Rebuild Index` 카드 + 결과 stats (Documents / Total Chunks)

**아쉬운 점:**

- 기능이 하나뿐 → 페이지가 너무 단순
- 인덱스 상태 (현재 문서 수, 청크 수) 실시간 표시 없음
- `data/raw/` 경로 노출이 운영 환경에서 부적절할 수 있음

---

## 3. 전체 디자인 시스템 분석

### 색상 체계

| 역할 | 현재 값 | 평가 |
| --- | --- | --- |
| 배경 | `#0a0a0f` | ✅ 매우 딥 다크 |
| 포인트 | `purple-500/600` | ✅ 일관성 있음 |
| 텍스트 주요 | `text-white` / `text-slate-200` | ✅ |
| 텍스트 보조 | `text-slate-400/500` | ✅ |
| 경계선 | `rgba(255,255,255,0.07)` | ⚠️ 너무 미묘 |
| 성공 | `green-500` | ✅ |
| 경고 | `yellow-400` | ✅ |
| 위험 | `red-500/600` | ✅ |

### 타이포그래피

| 요소 | 현재 | 평가 |
| --- | --- | --- |
| 페이지 타이틀 | `font-semibold text-white` | ✅ |
| 섹션 헤더 | `text-sm font-medium text-slate-300` | ⚠️ 작음 |
| 본문 | `text-sm text-slate-200` | ✅ |
| 보조 텍스트 | `text-xs text-slate-400/500` | ✅ |
| 코드 | `bg-slate-700 text-slate-300 rounded` | ✅ |

---

## 4. 개선 방향 (우선순위 순)

### 🔴 High Priority

**① QueryPage — Top-K 위치 이동**

- 현재: 헤더 우측에 고립
- 개선: 입력창 좌측에 배치 (전송 버튼과 같은 행)

**② DocumentsPage — 문서 목록 조회**

- 현재: 업로드/삭제만 있고 현재 인덱스 문서 목록 없음
- 개선: `GET /documents` API 추가 또는 Admin 통계 활용

**③ QueryPage — Empty state 예시 쿼리 칩**

```
[What is attention mechanism?]  [Compare BERT and GPT]  [Explain Transformer complexity]
```

### 🟡 Medium Priority

**④ Citation 카드 접기/펼치기**

- 답변 아래 `Sources (N) ▼` 토글로 기본 접힘 처리

**⑤ 사이드바 — 로그인 사용자 정보 표시**

- 하단에 `username + role badge` 추가

**⑥ QueryPage — 답변 복사 버튼**

- 어시스턴트 버블 우측 상단 `Copy` 아이콘 버튼

**⑦ QueryPage — Clear conversation 버튼**

- 헤더에 휴지통 아이콘

### 🟢 Low Priority

**⑧ `w-58` → `w-60` 수정**

**⑨ 입력창 자동 높이 조절 (`auto-resize textarea`)**

**⑩ 메시지 타임스탬프 (hover 시 표시)**

**⑪ 로그인 배경 orb 부드러운 float 애니메이션 추가**

---

## 5. 구현 난이도 × 효과 매트릭스

```
높은 효과
    │  ③ Empty state 칩      ① Top-K 이동
    │  ⑦ Clear 버튼          ② 문서 목록
    │  ⑥ 복사 버튼           ④ Citation 토글
    │  ⑤ 사용자 정보
    │  ⑨ Auto-resize         ⑩ 타임스탬프
    │  ⑪ 로그인 애니         ⑧ w-58 수정
낮은 효과
    └─────────────────────────────────────
       낮은 난이도          높은 난이도
```

---

*작성일: 2026-03-26 / 작성자: Edwin Cho*
