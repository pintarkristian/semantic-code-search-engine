"""PyTorch embedding pipeline via sentence-transformers.

Public surface:
    chunk_to_text(row)          — build a single embedding input string from a chunk row
    Embedder                    — lazy singleton model wrapper
    embed_dataframe(df)         — map an ingest DataFrame to a float32 vector matrix
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from semcode.config import Settings, get_settings
from semcode.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def chunk_to_text(row: Any, max_chars: int = 2048) -> str:
    """Combine symbol_name + docstring + code into a single embedding input string.

    The combined text is truncated to max_chars before being passed to the model
    (the model's tokeniser will also truncate, but this avoids sending very large
    strings over the wire in future distributed setups).
    """
    parts: list[str] = []

    name = str(row.get("symbol_name", "") if hasattr(row, "get") else getattr(row, "symbol_name", "")).strip()
    if name and name != "<anonymous>":
        parts.append(f"symbol: {name}")

    doc = str(row.get("docstring", "") if hasattr(row, "get") else getattr(row, "docstring", "")).strip()
    if doc:
        parts.append(doc)

    code = str(row.get("code", "") if hasattr(row, "get") else getattr(row, "code", "")).strip()
    if code:
        parts.append(code)

    text = "\n".join(parts)
    return text[:max_chars]


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
            self._model = SentenceTransformer(
                self.settings.embedding_model_name, device=device
            )
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        return int(self._load_model().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings to an L2-normalised float32 matrix.

        Args:
            texts: list of strings to encode.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32,
            with each row L2-normalised to unit length.
        """
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

        return np.asarray(vectors, dtype=np.float32)


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
