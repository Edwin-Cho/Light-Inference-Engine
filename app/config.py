from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_processed_dir: Path = base_dir / "data" / "processed"
    data_index_dir: Path = base_dir / "data" / "index"
    faiss_index_path: Path = data_index_dir / "faiss.index"
    metadata_store_path: Path = data_index_dir / "metadata.json"

    # ── Generation Model ───────────────────────────────────────────────────
    generation_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    use_adapter: bool = False
    load_in_4bit: bool = True          # CUDA only; MPS에서 float16 자동 fallback
    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.1
    generation_do_sample: bool = False

    # ── Embedding Model ────────────────────────────────────────────────────
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.30   # cosine similarity (L2-normalized IP)
    retrieval_score_gap: float = 0.25          # top-1 대비 최대 허용 score 차이 (cross-domain 누수 방지)
    citation_min_score: float = 0.65           # citations 포함 최소 점수 (noise 필터)

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # ── Auth ───────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production-at-least-32-chars!!"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    # ── Admin credentials (dev default) ───────────────────────────────────
    admin_username: str = "admin"
    admin_password: str = "admin1234"


settings = Settings()
