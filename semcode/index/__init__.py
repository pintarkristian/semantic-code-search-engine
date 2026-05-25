"""FAISS vector store and end-to-end indexing pipeline.

Public surface:
    ManifestMismatchError — raised when a saved index conflicts with current settings
    VectorStore           — build / save / load / search a FAISS index
    IndexingPipeline      — orchestrate ingest → embed → FAISS build → persist
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# faiss-cpu and PyTorch each ship an OpenMP runtime (libomp140 vs libiomp5md).
# On Windows, loading both in the same process triggers a hard abort unless this
# flag is set before faiss is imported. setdefault preserves any existing override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import pandas as pd

from semcode.config import Settings, get_settings
from semcode.embed import Embedder, embed_dataframe
from semcode.ingest import CodeIngestor
from semcode.logging import get_logger

log = get_logger(__name__)

# IVF search-probe count — higher = more accurate, slower
_IVF_NPROBE: int = 8
# Default corpus size at which to switch FlatIP → IVF
_IVF_THRESHOLD_DEFAULT: int = 10_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestMismatchError(ValueError):
    """Raised when a saved index manifest conflicts with current settings."""


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """FAISS-backed vector store with manifest-guarded persistence.

    Invariant: row *i* in the FAISS index == row *i* in the metadata DataFrame.
    Callers must ensure this by always building the index from the same DataFrame
    ordering that was used to write the parquet (the IndexingPipeline guarantees this).

    Args:
        settings:      application settings (defaults to get_settings()).
        ivf_threshold: number of vectors above which an IVFFlat index is used
                       instead of IndexFlatIP. Defaults to 10 000.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ivf_threshold: int = _IVF_THRESHOLD_DEFAULT,
    ) -> None:
        self.settings = settings or get_settings()
        self.ivf_threshold = ivf_threshold
        self._index: faiss.Index | None = None
        self._manifest: dict | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray) -> None:
        """Build a FAISS index from an L2-normalised float32 matrix.

        For n < ivf_threshold: IndexFlatIP (exact, no training needed).
        For n >= ivf_threshold: IndexIVFFlat (approximate, trained).

        Args:
            vectors: shape (n, dim), dtype float32, rows L2-normalised.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, dim = vectors.shape

        if n == 0:
            self._index = faiss.IndexFlatIP(max(dim, 1))
            index_type = "flat"
        elif n >= self.ivf_threshold:
            self._index = self._build_ivf(vectors, dim)
            index_type = "ivf"
        else:
            idx = faiss.IndexFlatIP(dim)
            idx.add(vectors)
            self._index = idx
            index_type = "flat"

        self._manifest = {
            "model_name": self.settings.embedding_model_name,
            "dimension": dim,
            "chunk_count": n,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "index_type": index_type,
        }
        log.info("built FAISS index", type=index_type, n=n, dim=dim)

    @staticmethod
    def _build_ivf(vectors: np.ndarray, dim: int) -> faiss.IndexIVFFlat:
        n = len(vectors)
        # Start with sqrt(n), then back off until FAISS won't warn about
        # insufficient training points (rule of thumb: 39 × nlist).
        nlist = min(max(4, int(n ** 0.5)), 1024)
        while nlist > 1 and 39 * nlist > n:
            nlist = max(1, nlist // 2)

        quantiser = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vectors)
        ivf.add(vectors)
        ivf.nprobe = _IVF_NPROBE
        return ivf

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the FAISS index and manifest JSON to settings paths."""
        if self._index is None or self._manifest is None:
            raise RuntimeError("Call build() before save().")

        index_path = self.settings.faiss_index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

        manifest_path = index_path.with_suffix(".json")
        manifest_path.write_text(json.dumps(self._manifest, indent=2))
        log.info("saved index", path=str(index_path))

    def load(self, *, expected_dim: int | None = None) -> None:
        """Load the FAISS index and validate the manifest.

        Args:
            expected_dim: if provided, also verify the stored dimension matches.
                          The IndexingPipeline passes embedder.dimension here.

        Raises:
            FileNotFoundError:     index or manifest file is missing.
            ManifestMismatchError: model name or (optionally) dimension mismatch.
        """
        index_path = self.settings.faiss_index_path
        manifest_path = index_path.with_suffix(".json")

        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {index_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        self._validate_manifest(manifest, expected_dim=expected_dim)

        self._index = faiss.read_index(str(index_path))
        if isinstance(self._index, faiss.IndexIVFFlat):
            self._index.nprobe = _IVF_NPROBE
        self._manifest = manifest
        log.info(
            "loaded index",
            type=manifest.get("index_type"),
            chunks=manifest.get("chunk_count"),
        )

    def _validate_manifest(
        self, manifest: dict, *, expected_dim: int | None = None
    ) -> None:
        saved_model = manifest.get("model_name", "")
        current_model = self.settings.embedding_model_name
        if saved_model != current_model:
            raise ManifestMismatchError(
                f"Index was built with model '{saved_model}' but settings specify "
                f"'{current_model}'. Re-index with --rebuild."
            )
        if expected_dim is not None:
            saved_dim = manifest.get("dimension")
            if saved_dim != expected_dim:
                raise ManifestMismatchError(
                    f"Index dimension {saved_dim} does not match "
                    f"embedder dimension {expected_dim}. Re-index with --rebuild."
                )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return the k nearest neighbours of query_vec.

        Args:
            query_vec: 1-D or (1, dim) float32 array.
            k:         maximum number of results.

        Returns:
            List of (row_index, score) tuples sorted by score descending.
            row_index corresponds directly to the integer row position in
            the metadata DataFrame saved during indexing.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]

        k_clamped = min(k, max(self._index.ntotal, 1))
        scores, indices = self._index.search(vec, k_clamped)

        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0  # FAISS returns -1 for unfilled slots
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> dict | None:
        return self._manifest

    @property
    def ntotal(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)


# ---------------------------------------------------------------------------
# IndexingPipeline
# ---------------------------------------------------------------------------

class IndexingPipeline:
    """Orchestrate ingest → embed → FAISS build → persist.

    Artifacts produced (all under settings.data_dir):
        metadata_path           — chunk metadata as parquet (written by ingest)
        faiss_index_path        — FAISS index binary
        faiss_index_path.json   — manifest JSON

    Args:
        settings:      application settings (defaults to get_settings()).
        embedder:      pre-built Embedder; if None, one is created lazily.
        ivf_threshold: forwarded to VectorStore (default 10 000).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        ivf_threshold: int = _IVF_THRESHOLD_DEFAULT,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder  # None = build lazily on first run
        self.ivf_threshold = ivf_threshold

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    def run(self, repo_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
        """Execute the full pipeline and return (DataFrame, vectors).

        The return values are provided for callers that want to inspect results;
        all artifacts are also written to disk as a side-effect.
        """
        # 1. Ingest
        log.info("pipeline: ingesting repo", repo=str(repo_path))
        df = CodeIngestor(repo_path, self.settings).ingest()
        log.info("pipeline: ingested", chunks=len(df))

        # 2. Embed
        log.info("pipeline: embedding chunks")
        vectors = embed_dataframe(df, embedder=self.embedder)
        log.info("pipeline: embedded", shape=list(vectors.shape))

        # 3. Build + persist FAISS index
        store = VectorStore(self.settings, ivf_threshold=self.ivf_threshold)
        store.build(vectors)
        store.save()

        log.info(
            "pipeline: complete",
            chunks=len(df),
            dimension=int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] else 0,
            index_type=store.manifest.get("index_type") if store.manifest else "n/a",
        )
        return df, vectors
