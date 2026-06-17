"""Unit tests for semcode.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_debug_release_alias_is_trimmed() -> None:
    assert Settings(debug=" prod ").debug is False


def test_path_fields_are_path_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "/tmp/mydata")
    monkeypatch.setenv("FAISS_INDEX_PATH", "/tmp/mydata/custom.faiss")
    s = Settings()
    assert s.data_dir == Path("/tmp/mydata")
    assert s.faiss_index_path == Path("/tmp/mydata/custom.faiss")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout_seconds", 0),
        ("rate_limit_requests", -1),
        ("rate_limit_window_seconds", 0),
        ("port", 0),
        ("batch_size", 0),
        ("max_chunk_tokens", 0),
        ("top_k_retrieve", 0),
        ("top_k_return", 0),
        ("max_query_length", 0),
        ("max_search_k", 0),
    ],
)
def test_operational_limits_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_rate_limit_zero_disables_limiter() -> None:
    assert Settings(rate_limit_requests=0).rate_limit_requests == 0


def test_request_timeout_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="request_timeout_seconds"):
        Settings(request_timeout_seconds=float("inf"))


def test_port_must_not_exceed_tcp_range() -> None:
    with pytest.raises(ValidationError):
        Settings(port=65536)


def test_host_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="host"):
        Settings(host="   ")


def test_host_must_be_string() -> None:
    with pytest.raises(ValidationError, match="host must be a string"):
        Settings(host=123)


def test_app_name_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="app_name"):
        Settings(app_name="   ")


def test_app_name_is_trimmed() -> None:
    assert Settings(app_name=" semcode-api ").app_name == "semcode-api"


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(log_level="verbose")


def test_log_level_must_be_string() -> None:
    with pytest.raises(ValidationError, match="log_level must be a string"):
        Settings(log_level=123)


def test_log_level_is_trimmed_and_normalized() -> None:
    assert Settings(log_level=" info ").log_level == "INFO"


def test_invalid_log_format_rejected() -> None:
    with pytest.raises(ValidationError, match="log_format"):
        Settings(log_format="plain")


def test_log_format_must_be_string() -> None:
    with pytest.raises(ValidationError, match="log_format must be a string"):
        Settings(log_format=123)


def test_log_format_is_trimmed_and_normalized() -> None:
    assert Settings(log_format=" JSON ").log_format == "json"


def test_invalid_embedding_device_rejected() -> None:
    with pytest.raises(ValidationError, match="embedding_device"):
        Settings(embedding_device="tpu")


def test_embedding_device_must_be_string() -> None:
    with pytest.raises(ValidationError, match="embedding_device must be a string"):
        Settings(embedding_device=123)


def test_embedding_device_is_trimmed_and_normalized() -> None:
    assert Settings(embedding_device=" CUDA ").embedding_device == "cuda"


def test_negative_retrieval_weight_rejected() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        Settings(dense_weight=-0.1)


def test_non_finite_retrieval_weight_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Settings(dense_weight=float("nan"))


def test_all_zero_retrieval_weights_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one retrieval weight"):
        Settings(dense_weight=0.0, bm25_weight=0.0)


def test_default_return_must_not_exceed_search_limit() -> None:
    with pytest.raises(ValidationError, match="top_k_return"):
        Settings(top_k_return=20, max_search_k=10)


def test_get_settings_is_cached() -> None:
    # get_settings() is lru_cache(maxsize=1) — must return the same object
    a = get_settings()
    b = get_settings()
    assert a is b
