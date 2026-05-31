"""PyTorch embedding pipeline via sentence-transformers.

Public surface:
    chunk_to_text(row)          — build a single embedding input string from a chunk row
    Embedder                    — lazy singleton model wrapper
    embed_dataframe(df)         — map an ingest DataFrame to a float32 vector matrix
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from semcode.config import Settings, get_settings
from semcode.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

_EMBEDDING_CACHE_FILENAME = "embedding_cache.pkl"


def embedding_cache_path(settings: Settings) -> Path:
    """Return the persistent embedding cache path for settings."""
    return settings.data_dir / _EMBEDDING_CACHE_FILENAME


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"nan", "<NA>", "NaT", "None"}:
        return ""
    return text


def chunk_to_text(row: Any, max_chars: int = 2048) -> str:
    """Combine symbol_name + docstring + code into a single embedding input string.

    The combined text is truncated to max_chars before being passed to the model
    (the model's tokeniser will also truncate, but this avoids sending very large
    strings over the wire in future distributed setups).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    parts: list[str] = []

    name = _text_value(
        row.get("symbol_name", "") if hasattr(row, "get") else getattr(row, "symbol_name", "")
    )
    if name and name != "<anonymous>":
        parts.append(f"symbol: {name}")

    doc = _text_value(
        row.get("docstring", "") if hasattr(row, "get") else getattr(row, "docstring", "")
    )
    if doc:
        parts.append(doc)

    code = _text_value(row.get("code", "") if hasattr(row, "get") else getattr(row, "code", ""))
    if code:
        parts.append(code)

    text = "\n".join(parts)
    return text[:max_chars]


def content_hash_for_text(text: str) -> str:
    """Return a stable SHA-256 content hash for an embedding input string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash_for_row(row: Any, max_chars: int = 2048) -> str:
    """Return the content hash for the exact text representation embedded."""
    return content_hash_for_text(chunk_to_text(row, max_chars=max_chars))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Lazy singleton wrapper around a SentenceTransformer model.

    The model is downloaded and loaded on the first call to .encode() or
    .dimension — not at construction time, so importing the module is free.

    Args:
        settings: application settings (defaults to get_settings()).
        _model:   pre-built model instance (used in tests to inject a mock).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        _model: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model = _model  # None means lazy-load on first use

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        cfg = self.settings.embedding_device
        if cfg == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cfg

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            device = self._resolve_device()
            log.info(
                "loading embedding model",
                model=self.settings.embedding_model_name,
                device=device,
            )
            self._model = SentenceTransformer(self.settings.embedding_model_name, device=device)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        dimension = int(self._load_model().get_sentence_embedding_dimension())
        if dimension <= 0:
            raise ValueError("embedding model dimension must be positive")
        return dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings to an L2-normalised float32 matrix.

        Args:
            texts: list of strings to encode.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32,
            with each row L2-normalised to unit length.
        """
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._load_model()
        max_chars = self.settings.max_chunk_tokens * 4
        truncated = [t[:max_chars] for t in texts]

        with torch.no_grad():
            vectors = model.encode(
                truncated,
                batch_size=self.settings.batch_size,
                show_progress_bar=len(texts) >= self.settings.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        out = np.asarray(vectors, dtype=np.float32)
        if out.ndim != 2 or out.shape[0] != len(texts):
            raise ValueError(
                f"Expected embedding matrix with {len(texts)} rows, got shape {out.shape}"
            )
        expected_dim = self.dimension
        if out.shape[1] != expected_dim:
            raise ValueError(f"Expected embedding dimension {expected_dim}, got {out.shape[1]}")
        return out


# ---------------------------------------------------------------------------
# Persistent embedding cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Persistent content-hash keyed embedding cache under settings.data_dir."""

    def __init__(self, settings: Settings, *, model_name: str, dimension: int) -> None:
        self.settings = settings
        self.model_name = model_name
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.path = embedding_cache_path(settings)
        self._vectors: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
            log.warning("ignoring unreadable embedding cache", path=str(self.path), error=str(exc))
            return

        if not isinstance(payload, dict):
            log.warning(
                "ignoring malformed embedding cache",
                path=str(self.path),
                payload_type=type(payload).__name__,
            )
            return

        if (
            payload.get("model_name") != self.model_name
            or int(payload.get("dimension", -1)) != self.dimension
        ):
            log.info(
                "ignoring embedding cache for different model or dimension",
                path=str(self.path),
                cached_model=payload.get("model_name"),
                cached_dimension=payload.get("dimension"),
                model=self.model_name,
                dimension=self.dimension,
            )
            return

        vectors = payload.get("vectors", {})
        if not isinstance(vectors, dict):
            log.warning(
                "ignoring embedding cache with malformed vectors",
                path=str(self.path),
                vectors_type=type(vectors).__name__,
            )
            return

        loaded: dict[str, np.ndarray] = {}
        skipped = 0
        for key, value in vectors.items():
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (self.dimension,):
                skipped += 1
                continue
            loaded[str(key)] = vector
        self._vectors = loaded
        log.info("loaded embedding cache", path=str(self.path), entries=len(self._vectors))
        if skipped:
            log.warning(
                "ignored embedding cache entries with invalid dimensions",
                path=str(self.path),
                skipped=skipped,
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "vectors": self._vectors,
        }
        with open(self.path, "wb") as f:
            pickle.dump(payload, f)
        log.info("saved embedding cache", path=str(self.path), entries=len(self._vectors))

    def clear(self) -> None:
        self._vectors.clear()
        if self.path.exists():
            self.path.unlink()

    def get(self, content_hash: str) -> np.ndarray | None:
        return self._vectors.get(content_hash)

    def set(self, content_hash: str, vector: np.ndarray) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise ValueError(
                f"Expected cached vector dimension {self.dimension}, got shape {arr.shape}"
            )
        self._vectors[content_hash] = arr


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------


def embed_dataframe(
    df: pd.DataFrame,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> np.ndarray:
    """Encode every row in an ingest DataFrame to a dense vector matrix.

    Args:
        df:       DataFrame with columns produced by semcode.ingest.
        embedder: optional pre-configured Embedder (useful for tests / reuse).
        settings: used to build an Embedder if embedder is not supplied.

    Returns:
        np.ndarray of shape (len(df), dimension), dtype float32.
    """
    emb = embedder or Embedder(settings)
    max_chars = emb.settings.max_chunk_tokens * 4
    texts = [chunk_to_text(row, max_chars=max_chars) for _, row in df.iterrows()]
    return emb.encode(texts)


def ensure_content_hashes(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Return a copy of df with a content_hash column for embedding inputs."""
    out = df.copy()
    max_chars = settings.max_chunk_tokens * 4
    out["content_hash"] = [
        content_hash_for_row(row, max_chars=max_chars) for _, row in out.iterrows()
    ]
    return out


def embed_dataframe_cached(
    df: pd.DataFrame,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
    reset_cache: bool = False,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Encode a DataFrame using the persistent content-hash embedding cache."""
    import time

    emb = embedder or Embedder(settings)
    cfg = emb.settings
    dimension = emb.dimension
    cache = EmbeddingCache(
        cfg,
        model_name=cfg.embedding_model_name,
        dimension=dimension,
    )
    if reset_cache:
        cache.clear()

    df_hashed = ensure_content_hashes(df, cfg)
    hashes = [str(value) for value in df_hashed["content_hash"].tolist()]

    start = time.perf_counter()
    missing_text_by_hash: dict[str, str] = {}
    max_chars = cfg.max_chunk_tokens * 4
    for content_hash, (_, row) in zip(hashes, df_hashed.iterrows()):
        if cache.get(content_hash) is None and content_hash not in missing_text_by_hash:
            missing_text_by_hash[content_hash] = chunk_to_text(row, max_chars=max_chars)

    missing_hashes = list(missing_text_by_hash)
    if missing_hashes:
        new_vectors = emb.encode([missing_text_by_hash[h] for h in missing_hashes])
        for content_hash, vector in zip(missing_hashes, new_vectors):
            cache.set(content_hash, vector)

    if hashes:
        cached_vectors = []
        for content_hash in hashes:
            cached = cache.get(content_hash)
            if cached is None:
                raise RuntimeError(f"Missing cached vector for content hash {content_hash}")
            cached_vectors.append(cached)
        vectors = np.stack(cached_vectors).astype(np.float32)
    else:
        vectors = np.zeros((0, dimension), dtype=np.float32)

    cache.save()
    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    stats: dict[str, int | float] = {
        "chunks": len(hashes),
        "chunks_embedded": len(missing_hashes),
        "cache_hits": len(hashes) - len(missing_hashes),
        "cache_entries": len(cache._vectors),
        "elapsed_ms": elapsed_ms,
    }
    log.info(
        "embedding cache stats",
        chunks=stats["chunks"],
        chunks_embedded=stats["chunks_embedded"],
        cache_hits=stats["cache_hits"],
        elapsed_ms=stats["elapsed_ms"],
    )
    return vectors, stats
