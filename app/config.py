from __future__ import annotations

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_index_dir: Path = base_dir / "data" / "index"
    faiss_index_path: Path = data_index_dir / "faiss.index"
    metadata_store_path: Path = data_index_dir / "metadata.json"

    # ── Generation Model ─────────────────────────────────────────────────────
    generation_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True   # CUDA 전용 NF4 4-bit 양자화; MPS는 항상 float16 (별도 분기, 이 플래그와 무관)
    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.1   # do_sample=True 일 때만 적용됨 (현재 미사용)
    generation_do_sample: bool = False    # True로 변경 시 위 temperature 활성화

    # ── Embedding Model ────────────────────────────────────────────────────
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.30          # cosine similarity (L2-normalized IP)
    retrieval_fallback_score_threshold: float = 0.20  # 2nd-pass 완화 임계값 (vocabulary mismatch 구제)
    retrieval_score_gap: float = 0.25                 # top-1 대비 최대 허용 score 차이 (cross-domain 누수 방지)
    citation_min_score: float = 0.65                  # citations 포함 최소 점수 (noise 필터)
    citation_injection_min_overlap: float = 0.15  # P16 post-hoc injection 최소 word-overlap (eval 최적값)

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # ── Auth ──────────────────────────────────────────────────────────────────
    # 기본값 없음 — .env 미설정 시 서버 기동 실패 (의도된 동작)
    # 생성: openssl rand -hex 32
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    # ── Admin credentials ───────────────────────────────────────────────────
    # 기본값 없음 — .env에 ADMIN_USERNAME / ADMIN_PASSWORD 필수 설정
    admin_username: str
    admin_password: str


    @model_validator(mode="after")
    def _validate_chunk_params(self) -> "Settings":
        """Ensure chunk_overlap is strictly less than chunk_size.

        Prevents chunker.py runtime errors caused by misconfigured overlap.
        Validated at server startup so misconfiguration fails early.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self


settings = Settings()
