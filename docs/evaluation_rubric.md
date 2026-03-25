# Evaluation Rubric — OnDevice Scholar RAG

**Version:** 1.1.0  
**Reference:** SRS §10 Acceptance Criteria

---

## 평가 목적 | Purpose

이 평가 기준은 OnDevice Scholar RAG 시스템이 SRS v1.1.0에 정의된 기능적·비기능적 요구사항을 충족하는지 검증하기 위한 기준입니다.  
각 항목은 Pass / Fail / Partial 로 평가됩니다.

---

## 1. 프라이버시 및 오프라인 실행 | Privacy & Offline Execution

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| P-1 | 완전 오프라인 실행 | 네트워크 모니터링 도구로 런타임 아웃바운드 요청 확인 | 인덱싱/추론/검색 중 외부 요청 0건 |
| P-2 | 임베딩 API 미호출 | 패킷 캡처 또는 mock 테스트 | 로컬 임베딩 모델만 사용 |
| P-3 | 인증 없는 접근 차단 | 토큰 없이 보호 엔드포인트 호출 | HTTP 401 반환, 내부 상태 미노출 |
| P-4 | RBAC 권한 분리 | Researcher 계정으로 `/ingest` 호출 | HTTP 403 반환 |

---

## 2. 인용 및 근거 생성 | Citation & Grounded Generation

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| C-1 | 인용 포함률 | 30개 쿼리 샘플 검토 | 사실 기반 응답 100%에 인용 포함 |
| C-2 | 인용 형식 정확성 | 응답 파싱 자동화 | `source_filename`, `section_header`, `page_number` 필드 모두 존재 |
| C-3 | Citation Validator 작동 | 인용 없는 응답을 강제 생성 후 반환값 확인 | 안전 폴백 응답으로 대체됨 |
| C-4 | 컨텍스트 외 추론 금지 | 인덱스에 없는 주제로 쿼리 | 사실 날조 없이 폴백 반환 |

---

## 3. 안전 폴백 동작 | Safe Fallback Behavior

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| F-1 | 낮은 유사도 쿼리 | 임계값 이하 쿼리 10건 테스트 | 폴백 응답 반환, 추측 답변 없음 |
| F-2 | 빈 인덱스 쿼리 | 인덱스 비워두고 쿼리 | `"No relevant information found..."` 반환 |
| F-3 | 폴백 메시지 일관성 | 폴백 조건 다수 테스트 | SRS §4.5 FR-25 기준 문구 사용 |

---

## 4. 성능 | Performance

| ID | 항목 | 기준 환경 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| PR-1 | TTFT (Time-To-First-Token) | M3 Pro, 500 청크, 쿼리 ≤ 50 토큰 | **2초 미만** |
| PR-2 | Top-K 검색 지연 | 500 청크 FAISS 인덱스 | 대화형 워크플로우 유지 가능한 수준 |
| PR-3 | 단일 청크 임베딩 지연 | 2048자 청크 기준 | 실용적 로컬 인덱싱 가능한 수준 |

> 참고: NFR-7 기준. 측정은 `time.perf_counter()` 또는 FastAPI middleware 타이머 사용.

---

## 5. 메모리 효율성 | Memory Efficiency

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| M-1 | 런타임 메모리 사용량 | `mps` 또는 `psutil`로 측정 | 전체 파이프라인 **4GB ~ 8GB** 이내 |
| M-2 | 모델 로드 안정성 | 메모리 부족 시 동작 | 서비스 불가 에러 반환, 비정상 종료 없음 |

---

## 6. 신뢰성 및 복구 | Reliability & Recovery

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| R-1 | 인덱스 손상 복구 | FAISS 파일 강제 손상 후 재시작 | persisted 메타데이터로 자동 rebuild |
| R-2 | 서버 재시작 후 인덱스 유지 | 서버 재시작 후 동일 쿼리 | 재인덱싱 없이 동일 결과 반환 |
| R-3 | 문서 파싱 실패 처리 | 암호화된 PDF 업로드 | 구조화된 에러 응답 반환, 서버 중단 없음 |

---

## 7. 문서 생명주기 | Document Lifecycle

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| D-1 | 문서 업로드 및 인덱싱 | PDF/MD/TXT 각 1건 업로드 | 청크 생성 + 메타데이터 바인딩 + FAISS 추가 완료 |
| D-2 | 증분 인덱싱 | 기존 인덱스에 신규 문서 추가 | 전체 rebuild 없이 정상 추가 |
| D-3 | 문서 삭제 | Lab PI로 `/document/{id}` 호출 | 메타데이터 + 청크 + 벡터 참조 모두 제거 |
| D-4 | 전체 재빌드 | Admin으로 `/admin/rebuild-index` 호출 | cached 메타데이터 기반 rebuild 성공 |

---

## 8. 보안 | Security

| ID | 항목 | 검증 방법 | Pass 조건 |
| :--- | :--- | :--- | :--- |
| S-1 | 만료 토큰 거부 | 만료된 JWT로 요청 | HTTP 401 반환 |
| S-2 | 잘못된 토큰 거부 | 임의 문자열로 Authorization 헤더 설정 | HTTP 401 반환, 내부 정보 미노출 |
| S-3 | 최소 권한 원칙 | 각 역할로 허용되지 않은 엔드포인트 호출 | HTTP 403 반환 |

---

## 9. 에러 응답 형식 | Error Response Format

| ID | 항목 | Pass 조건 |
| :--- | :--- | :--- |
| E-1 | 에러 응답 구조화 | 모든 오류에 `request_id`, `status`, `error`, `reason` 필드 포함 |
| E-2 | request_id 포함 | 모든 응답(성공/실패)에 UUID 포함 |

---

## 평가 결과 기록 양식 | Score Sheet

| 카테고리 | 총 항목 | Pass | Fail | Partial | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 프라이버시 | 4 | | | | |
| 인용/근거 생성 | 4 | | | | |
| 안전 폴백 | 3 | | | | |
| 성능 | 3 | | | | |
| 메모리 효율성 | 2 | | | | |
| 신뢰성/복구 | 3 | | | | |
| 문서 생명주기 | 4 | | | | |
| 보안 | 3 | | | | |
| 에러 응답 | 2 | | | | |
| **합계** | **28** | | | | |

---

## Pass 기준 | Overall Pass Threshold

- **전체 통과:** Pass 25/28 이상 (89%), Fail 0건
- **조건부 통과:** Pass 22/28 이상, Fail 항목이 성능 카테고리에만 해당
- **미통과:** Fail이 프라이버시(P), 인용(C), 보안(S) 카테고리에 1건 이상 존재
