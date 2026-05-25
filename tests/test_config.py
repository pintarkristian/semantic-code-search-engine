"""Unit tests for semcode.config."""

from pathlib import Path

import pytest

from semcode.config import Settings, get_settings


def test_defaults() -> None:
    s = Settings()
    assert s.app_name == "semcode"
    assert s.embedding_device == "cpu"
    assert s.batch_size == 64
    assert s.max_chunk_tokens == 512
    assert s.top_k_retrieve == 50
    assert s.top_k_return == 10
    assert s.dense_weight == pytest.approx(0.7)
    assert s.bm25_weight == pytest.approx(0.3)
    assert isinstance(s.data_dir, Path)
    assert isinstance(s.faiss_index_path, Path)
    assert isinstance(s.metadata_path, Path)
    assert isinstance(s.reranker_model_path, Path)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "test-model")
    monkeypatch.setenv("BATCH_SIZE", "16")
    monkeypatch.setenv("TOP_K_RETRIEVE", "20")
    monkeypatch.setenv("DENSE_WEIGHT", "0.6")
    monkeypatch.setenv("BM25_WEIGHT", "0.4")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

    s = Settings()
    assert s.embedding_model_name == "test-model"
    assert s.batch_size == 16
    assert s.top_k_retrieve == 20
    assert s.dense_weight == pytest.approx(0.6)
    assert s.bm25_weight == pytest.approx(0.4)
    assert s.embedding_device == "cuda"


def test_path_fields_are_path_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "/tmp/mydata")
    monkeypatch.setenv("FAISS_INDEX_PATH", "/tmp/mydata/custom.faiss")
    s = Settings()
    assert s.data_dir == Path("/tmp/mydata")
    assert s.faiss_index_path == Path("/tmp/mydata/custom.faiss")


def test_get_settings_is_cached() -> None:
    # get_settings() is lru_cache(maxsize=1) — must return the same object
    a = get_settings()
    b = get_settings()
    assert a is b
