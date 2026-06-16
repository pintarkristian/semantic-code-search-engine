"""Centralised, typed application settings backed by pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Port = Annotated[int, Field(ge=1, le=65535)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- server ---
    app_name: str = "semcode"
    debug: bool = False
    host: str = "0.0.0.0"
    port: Port = 8000
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR
    log_format: str = "pretty"  # "pretty" (dev) | "json" (prod)
    request_timeout_seconds: PositiveFloat = 30.0
    rate_limit_requests: NonNegativeInt = 120
    rate_limit_window_seconds: PositiveInt = 60

    # --- embedding model ---
    embedding_model_name: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    embedding_device: str = "cpu"  # "cpu" | "cuda" — override to "cuda" if GPU available
    batch_size: PositiveInt = 64
    max_chunk_tokens: PositiveInt = 512

    # --- paths ---
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/index.faiss")
    metadata_path: Path = Path("data/metadata.parquet")
    reranker_model_path: Path = Path("data/reranker")

    # --- retrieval ---
    top_k_retrieve: PositiveInt = 50  # candidates fetched from each source before fusion
    top_k_return: PositiveInt = 10  # results returned to the caller
    max_query_length: PositiveInt = 512
    max_search_k: PositiveInt = 100
    use_reranker: bool = False  # optional final learned re-ranking stage

    # --- fusion weights (must sum to 1.0 for RRF scaling to be meaningful) ---
    dense_weight: float = 0.7
    bm25_weight: float = 0.3

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"release", "prod", "production"}:
            return False
        return value

    @field_validator("host", mode="before")
    @classmethod
    def _validate_host(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("host must be a string")
        if not value.strip():
            raise ValueError("host must contain non-whitespace text")
        return value.strip()

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of: {', '.join(sorted(allowed))}")
        return normalized

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"pretty", "json"}:
            raise ValueError("log_format must be 'pretty' or 'json'")
        return normalized

    @field_validator("embedding_device")
    @classmethod
    def _validate_embedding_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("embedding_device must be 'cpu' or 'cuda'")
        return normalized

    @model_validator(mode="after")
    def _validate_retrieval_weights(self) -> Settings:
        # These fields interact at runtime: invalid combinations can produce
        # empty or out-of-bounds rankings even though each value is valid alone.
        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.top_k_return > self.max_search_k:
            raise ValueError("top_k_return must be less than or equal to max_search_k")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
