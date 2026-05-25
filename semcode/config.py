"""Application settings (placeholder)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "semcode"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    model_name: str = "microsoft/codebert-base"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    index_path: str = "data/index.faiss"
    data_path: str = "data/"

    class Config:
        env_file = ".env"


settings = Settings()
