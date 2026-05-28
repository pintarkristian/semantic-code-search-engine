"""Centralised, typed application settings backed by pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- server ---
    app_name: str = "semcode"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "pretty"  # "pretty" (dev) | "json" (prod)
    request_timeout_seconds: float = Field(30.0, gt=0)
    rate_limit_requests: int = Field(120, ge=0)
    rate_limit_window_seconds: int = Field(60, gt=0)

    # --- embedding model ---
    embedding_model_name: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    embedding_device: str = "cpu"  # "cpu" | "cuda" — override to "cuda" if GPU available
    batch_size: int = Field(64, gt=0)
    max_chunk_tokens: int = Field(512, gt=0)

    # --- paths ---
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/index.faiss")
    metadata_path: Path = Path("data/metadata.parquet")
    reranker_model_path: Path = Path("data/reranker")

    # --- retrieval ---
    top_k_retrieve: int = Field(50, gt=0)  # candidates fetched from each source before fusion
    top_k_return: int = Field(10, gt=0)  # results returned to the caller
    max_query_length: int = Field(512, gt=0)
    max_search_k: int = Field(100, gt=0)
    use_reranker: bool = False  # optional final learned re-ranking stage

    # --- fusion weights (must sum to 1.0 for RRF scaling to be meaningful) ---
    dense_weight: float = 0.7
    bm25_weight: float = 0.3

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, value: object) -> object:
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
