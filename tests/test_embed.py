"""Tests for semcode.embed — Embedder and helpers."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from semcode.config import Settings
from semcode.embed import (
    Embedder,
    EmbeddingCache,
    chunk_to_text,
    content_hash_for_text,
    embed_dataframe,
    embed_dataframe_cached,
)

# ---------------------------------------------------------------------------
# Mock model — deterministic, content-addressed vectors, no network calls
# ---------------------------------------------------------------------------

_MOCK_DIM = 64


class _MockModel:
    """Fake SentenceTransformer that returns deterministic L2-normalised vectors."""

    def get_sentence_embedding_dimension(self) -> int:
        return _MOCK_DIM

    def encode(
        self,
        inputs: list[str] | str,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> np.ndarray:
        texts = inputs if isinstance(inputs, list) else [inputs]
        vecs = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)
        return vecs.astype(np.float32)

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        # Seed from SHA-256 of text so the same text always gets the same vector,
        # regardless of what batch it appears in — mirrors real model behaviour.
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(_MOCK_DIM).astype(np.float32)


class _BadDimensionModel(_MockModel):
    def get_sentence_embedding_dimension(self) -> int:
        return 0


class _NonIntegerDimensionModel(_MockModel):
    def get_sentence_embedding_dimension(self) -> str:
        return "bad"


class _BadShapeModel(_MockModel):
    def encode(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros(_MOCK_DIM, dtype=np.float32)


class _WrongWidthModel(_MockModel):
    def encode(self, inputs: list[str] | str, *args: Any, **kwargs: Any) -> np.ndarray:
        texts = inputs if isinstance(inputs, list) else [inputs]
        return np.zeros((len(texts), _MOCK_DIM + 1), dtype=np.float32)


class _NonFiniteModel(_MockModel):
    def encode(self, inputs: list[str] | str, *args: Any, **kwargs: Any) -> np.ndarray:
        texts = inputs if isinstance(inputs, list) else [inputs]
        out = np.zeros((len(texts), _MOCK_DIM), dtype=np.float32)
        out[0, 0] = np.nan
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_model_name="mock-model",
        embedding_device="cpu",
        batch_size=4,
        max_chunk_tokens=128,
        metadata_path=tmp_path / "metadata.parquet",
        data_dir=tmp_path,
    )


@pytest.fixture
def embedder(settings: Settings) -> Embedder:
    return Embedder(settings, _model=_MockModel())


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol_name": "validate_token",
                "symbol_type": "function",
                "docstring": '"""Return True if the token is valid."""',
                "code": "def validate_token(token: str) -> bool:\n    return bool(token)",
                "language": "python",
                "file_path": "auth.py",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "symbol_name": "TokenValidator",
                "symbol_type": "class",
                "docstring": '"""Validates bearer tokens."""',
                "code": "class TokenValidator:\n    pass",
                "language": "python",
                "file_path": "auth.py",
                "start_line": 5,
                "end_line": 6,
            },
            {
                "symbol_name": "formatDate",
                "symbol_type": "function",
                "docstring": "// Format a Date as ISO string",
                "code": "function formatDate(d) { return d.toISOString(); }",
                "language": "javascript",
                "file_path": "utils.js",
                "start_line": 1,
                "end_line": 1,
            },
        ]
    )


# ---------------------------------------------------------------------------
# chunk_to_text
# ---------------------------------------------------------------------------


def test_chunk_to_text_includes_symbol_name() -> None:
    row = {"symbol_name": "validate_token", "docstring": "", "code": "def validate_token(): pass"}
    text = chunk_to_text(row)
    assert "validate_token" in text


def test_chunk_to_text_includes_docstring() -> None:
    row = {"symbol_name": "foo", "docstring": "Return the bar value.", "code": "def foo(): pass"}
    text = chunk_to_text(row)
    assert "Return the bar value." in text


def test_chunk_to_text_includes_code() -> None:
    row = {"symbol_name": "foo", "docstring": "", "code": "def foo(): return 42"}
    text = chunk_to_text(row)
    assert "def foo(): return 42" in text


def test_chunk_to_text_skips_anonymous_name() -> None:
    row = {"symbol_name": "<anonymous>", "docstring": "", "code": "() => {}"}
    text = chunk_to_text(row)
    assert "<anonymous>" not in text
    assert "() => {}" in text


def test_chunk_to_text_skips_empty_docstring() -> None:
    row = {"symbol_name": "foo", "docstring": "", "code": "pass"}
    text = chunk_to_text(row)
    # No blank line from empty docstring
    assert "\n\n" not in text


def test_chunk_to_text_truncates() -> None:
    row = {"symbol_name": "x", "docstring": "", "code": "a" * 5000}
    text = chunk_to_text(row, max_chars=100)
    assert len(text) <= 100


def test_chunk_to_text_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_to_text({"code": "def ok(): pass"}, max_chars=0)


def test_chunk_to_text_rejects_non_integer_max_chars() -> None:
    with pytest.raises(TypeError, match="max_chars must be an integer"):
        chunk_to_text({"code": "def ok(): pass"}, max_chars="20")  # type: ignore[arg-type]


def test_content_hash_for_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="expects a string"):
        content_hash_for_text(123)  # type: ignore[arg-type]


def test_chunk_to_text_accepts_pandas_series(sample_df: pd.DataFrame) -> None:
    for _, row in sample_df.iterrows():
        text = chunk_to_text(row)
        assert isinstance(text, str)
        assert len(text) > 0


def test_chunk_to_text_skips_missing_dataframe_values() -> None:
    row = pd.Series({"symbol_name": pd.NA, "docstring": np.nan, "code": "def ok(): pass"})

    text = chunk_to_text(row)

    assert "nan" not in text.lower()
    assert "<NA>" not in text
    assert "def ok(): pass" in text


# ---------------------------------------------------------------------------
# Embedder — shape and dtype
# ---------------------------------------------------------------------------


def test_encode_output_shape(embedder: Embedder) -> None:
    texts = ["def foo(): pass", "class Bar: pass", "function baz() {}"]
    out = embedder.encode(texts)
    assert out.shape == (3, _MOCK_DIM)


def test_encode_output_dtype(embedder: Embedder) -> None:
    out = embedder.encode(["hello world"])
    assert out.dtype == np.float32


def test_encode_empty_list(embedder: Embedder) -> None:
    out = embedder.encode([])
    assert out.shape[0] == 0
    assert out.shape[1] == _MOCK_DIM


def test_encode_single_text(embedder: Embedder) -> None:
    out = embedder.encode(["def foo(): pass"])
    assert out.shape == (1, _MOCK_DIM)


def test_encode_rejects_non_string_items(embedder: Embedder) -> None:
    with pytest.raises(TypeError, match="list of strings"):
        embedder.encode(["ok", 123])  # type: ignore[list-item]


def test_encode_rejects_non_list_texts(embedder: Embedder) -> None:
    with pytest.raises(TypeError, match="list of strings"):
        embedder.encode("not a list")  # type: ignore[arg-type]


def test_encode_rejects_invalid_model_output_shape(settings: Settings) -> None:
    emb = Embedder(settings, _model=_BadShapeModel())

    with pytest.raises(ValueError, match="embedding matrix"):
        emb.encode(["hello"])


def test_encode_rejects_wrong_model_output_width(settings: Settings) -> None:
    emb = Embedder(settings, _model=_WrongWidthModel())

    with pytest.raises(ValueError, match="embedding dimension"):
        emb.encode(["hello"])


def test_encode_rejects_non_finite_model_output(settings: Settings) -> None:
    emb = Embedder(settings, _model=_NonFiniteModel())

    with pytest.raises(ValueError, match="NaN or infinite"):
        emb.encode(["hello"])


def test_dimension_property(embedder: Embedder) -> None:
    assert embedder.dimension == _MOCK_DIM


def test_dimension_property_rejects_invalid_model_dimension(settings: Settings) -> None:
    emb = Embedder(settings, _model=_BadDimensionModel())

    with pytest.raises(ValueError, match="dimension"):
        _ = emb.dimension


def test_dimension_property_rejects_non_integer_model_dimension(settings: Settings) -> None:
    emb = Embedder(settings, _model=_NonIntegerDimensionModel())

    with pytest.raises(ValueError, match="dimension must be an integer"):
        _ = emb.dimension


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------


def test_vectors_are_l2_normalised(embedder: Embedder) -> None:
    texts = ["def foo(): pass", "class Bar: pass", "SELECT * FROM users"]
    out = embedder.encode(texts)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)


def test_single_vector_is_unit_length(embedder: Embedder) -> None:
    out = embedder.encode(["a single embedding"])
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Batching consistency
# ---------------------------------------------------------------------------


def test_batched_equals_single_item_encoding(embedder: Embedder) -> None:
    texts = [
        "def validate(token): return bool(token)",
        "class Auth: pass",
        "function login(user, pw) {}",
        "public void connect() {}",
    ]
    # Encode all at once
    batch_out = embedder.encode(texts)
    # Encode one at a time
    single_outs = np.stack([embedder.encode([t])[0] for t in texts])
    np.testing.assert_allclose(batch_out, single_outs, atol=1e-6)


def test_same_text_same_vector(embedder: Embedder) -> None:
    text = "def add(a, b): return a + b"
    v1 = embedder.encode([text])[0]
    v2 = embedder.encode([text])[0]
    np.testing.assert_array_equal(v1, v2)


def test_different_texts_different_vectors(embedder: Embedder) -> None:
    v1 = embedder.encode(["def foo(): return 1"])[0]
    v2 = embedder.encode(["class Bar: pass"])[0]
    assert not np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# embed_dataframe
# ---------------------------------------------------------------------------


def test_embed_dataframe_shape(embedder: Embedder, sample_df: pd.DataFrame) -> None:
    out = embed_dataframe(sample_df, embedder=embedder)
    assert out.shape == (len(sample_df), _MOCK_DIM)


def test_embed_dataframe_dtype(embedder: Embedder, sample_df: pd.DataFrame) -> None:
    out = embed_dataframe(sample_df, embedder=embedder)
    assert out.dtype == np.float32


def test_embed_dataframe_vectors_normalised(embedder: Embedder, sample_df: pd.DataFrame) -> None:
    out = embed_dataframe(sample_df, embedder=embedder)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(sample_df)), atol=1e-5)


def test_embed_dataframe_empty(embedder: Embedder) -> None:
    empty_df = pd.DataFrame(
        columns=[
            "symbol_name",
            "docstring",
            "code",
            "language",
            "file_path",
            "start_line",
            "end_line",
        ]
    )
    out = embed_dataframe(empty_df, embedder=embedder)
    assert out.shape[0] == 0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_cuda_falls_back_to_cpu_when_unavailable(settings: Settings) -> None:
    import torch

    settings_cuda = Settings(
        embedding_model_name="mock-model",
        embedding_device="cuda",
        batch_size=4,
    )
    emb = Embedder(settings_cuda, _model=_MockModel())
    # When cuda is unavailable, _resolve_device() returns "cpu"
    if not torch.cuda.is_available():
        assert emb._resolve_device() == "cpu"


def test_cpu_device_stays_cpu(settings: Settings) -> None:
    emb = Embedder(settings, _model=_MockModel())
    assert emb._resolve_device() == "cpu"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_encode_truncates_to_max_chars(settings: Settings) -> None:
    # max_chunk_tokens=128 → max_chars=512
    long_text = "x" * 10_000
    emb = Embedder(settings, _model=_MockModel())
    # Just verify encode doesn't crash with very long input
    out = emb.encode([long_text])
    assert out.shape == (1, _MOCK_DIM)


def test_embedding_cache_ignores_malformed_payload(settings: Settings) -> None:
    cache_path = settings.data_dir / "embedding_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(["not", "a", "cache"], f)

    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    assert cache.get("missing") is None


def test_embedding_cache_rejects_invalid_dimension(settings: Settings) -> None:
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=0)


def test_embedding_cache_rejects_non_integer_dimension(settings: Settings) -> None:
    with pytest.raises(ValueError, match="dimension must be an integer"):
        EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension="bad")  # type: ignore[arg-type]


def test_embedding_cache_rejects_non_string_model_name(settings: Settings) -> None:
    with pytest.raises(TypeError, match="model_name must be a string"):
        EmbeddingCache(settings, model_name=123, dimension=_MOCK_DIM)  # type: ignore[arg-type]


def test_embedding_cache_rejects_blank_model_name(settings: Settings) -> None:
    with pytest.raises(ValueError, match="model_name"):
        EmbeddingCache(settings, model_name="   ", dimension=_MOCK_DIM)


def test_embedding_cache_ignores_invalid_dimension_metadata(settings: Settings) -> None:
    cache_path = settings.data_dir / "embedding_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "model_name": settings.embedding_model_name,
                "dimension": "not-an-int",
                "vectors": {"stale": np.zeros(_MOCK_DIM, dtype=np.float32)},
            },
            f,
        )

    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    assert cache.get("stale") is None


def test_embedding_cache_ignores_malformed_vectors(settings: Settings) -> None:
    cache_path = settings.data_dir / "embedding_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "model_name": settings.embedding_model_name,
                "dimension": _MOCK_DIM,
                "vectors": ["not", "a", "mapping"],
            },
            f,
        )

    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    assert cache.get("not") is None


def test_embedding_cache_skips_wrong_dimension_vectors(settings: Settings) -> None:
    cache_path = settings.data_dir / "embedding_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "model_name": settings.embedding_model_name,
                "dimension": _MOCK_DIM,
                "vectors": {
                    "bad": np.zeros(_MOCK_DIM + 1, dtype=np.float32),
                    "good": np.zeros(_MOCK_DIM, dtype=np.float32),
                },
            },
            f,
        )

    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    assert cache.get("bad") is None
    assert cache.get("good") is not None


def test_embedding_cache_rejects_wrong_dimension_on_set(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    with pytest.raises(ValueError, match="cached vector dimension"):
        cache.set("bad", np.zeros(_MOCK_DIM + 1, dtype=np.float32))


def test_embedding_cache_rejects_non_finite_vector_on_set(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)
    vector = np.zeros(_MOCK_DIM, dtype=np.float32)
    vector[0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        cache.set("bad", vector)


def test_embedding_cache_rejects_non_string_set_key(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    with pytest.raises(TypeError, match="content_hash must be a string"):
        cache.set(123, np.zeros(_MOCK_DIM, dtype=np.float32))  # type: ignore[arg-type]


def test_embedding_cache_rejects_blank_set_key(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    with pytest.raises(ValueError, match="content_hash"):
        cache.set("   ", np.zeros(_MOCK_DIM, dtype=np.float32))


def test_embedding_cache_rejects_non_string_get_key(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    with pytest.raises(TypeError, match="content_hash must be a string"):
        cache.get(123)  # type: ignore[arg-type]


def test_embedding_cache_rejects_blank_get_key(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)

    with pytest.raises(ValueError, match="content_hash"):
        cache.get("   ")


def test_embedding_cache_copies_vectors_on_set(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)
    vector = np.zeros(_MOCK_DIM, dtype=np.float32)
    cache.set("stable", vector)
    vector[0] = 1.0

    assert cache.get("stable")[0] == 0.0


def test_embedding_cache_returns_vector_copies(settings: Settings) -> None:
    cache = EmbeddingCache(settings, model_name=settings.embedding_model_name, dimension=_MOCK_DIM)
    cache.set("stable", np.zeros(_MOCK_DIM, dtype=np.float32))
    cached = cache.get("stable")
    cached[0] = 1.0

    assert cache.get("stable")[0] == 0.0


def test_cached_embedding_counts_duplicate_content_as_hits(settings: Settings) -> None:
    df = pd.DataFrame(
        [
            {"symbol_name": "same", "docstring": "", "code": "def same(): return 1"},
            {"symbol_name": "same", "docstring": "", "code": "def same(): return 1"},
        ]
    )
    _, stats = embed_dataframe_cached(df, embedder=Embedder(settings, _model=_MockModel()))

    assert stats["chunks_embedded"] == 1
    assert stats["cache_hits"] == 1


# ---------------------------------------------------------------------------
# Slow tests — real model (skipped unless --slow is passed)
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.mark.slow
def test_real_model_smoke() -> None:
    """Smoke test: embed fixture repo chunks with a real tiny model.

    Downloads ~90 MB on first run; subsequent runs use the local cache.
    Run with: pytest --slow -k test_real_model_smoke
    """
    from semcode.ingest import CodeIngestor

    settings = Settings(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
        batch_size=8,
        max_chunk_tokens=256,
    )
    df = CodeIngestor(FIXTURE_REPO, settings).ingest()
    emb = Embedder(settings)

    vectors = embed_dataframe(df, embedder=emb)

    assert vectors.shape == (len(df), emb.dimension)
    assert vectors.dtype == np.float32
    # All vectors should be unit-length
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(df)), atol=1e-4)
