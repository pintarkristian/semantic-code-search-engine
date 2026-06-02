"""FAISS vector store and end-to-end indexing pipeline.

Public surface:
    ManifestMismatchError — raised when a saved index conflicts with current settings
    VectorStore           — build / save / load / search a FAISS index
    IndexingPipeline      — orchestrate ingest → embed → FAISS build → persist
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

# faiss-cpu and PyTorch each ship an OpenMP runtime (libomp140 vs libiomp5md).
# On Windows, loading both in the same process triggers a hard abort unless this
# flag is set before faiss is imported. setdefault preserves any existing override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import pandas as pd

from semcode.config import Settings, get_settings
from semcode.embed import Embedder, embed_dataframe_cached, ensure_content_hashes
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
        if ivf_threshold <= 0:
            raise ValueError("ivf_threshold must be positive")
        self.ivf_threshold = ivf_threshold
        self._index: faiss.Index | None = None
        self._manifest: dict | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, *, ids: np.ndarray | None = None) -> None:
        """Build a FAISS index from an L2-normalised float32 matrix.

        For n < ivf_threshold: IndexFlatIP (exact, no training needed).
        For n >= ivf_threshold: IndexIVFFlat (approximate, trained).

        Args:
            vectors: shape (n, dim), dtype float32, rows L2-normalised.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("FAISS vectors must contain only finite values")
        n, dim = vectors.shape
        if dim <= 0:
            raise ValueError("FAISS vector dimension must be positive")
        ids = None if ids is None else np.ascontiguousarray(ids, dtype=np.int64)
        if ids is not None and len(ids) != n:
            raise ValueError(f"Expected {n} ids, got {len(ids)}")
        if ids is not None and len(np.unique(ids)) != len(ids):
            raise ValueError("FAISS ids must be unique")

        if n == 0:
            base = faiss.IndexFlatIP(max(dim, 1))
            index_type = "flat"
        elif n >= self.ivf_threshold:
            base = self._build_ivf(vectors, dim)
            index_type = "ivf"
        else:
            base = faiss.IndexFlatIP(dim)
            index_type = "flat"

        if ids is not None:
            idx = faiss.IndexIDMap2(base)
            if n:
                idx.add_with_ids(vectors, ids)
            self._index = idx
        else:
            if n:
                base.add(vectors)
            self._index = base

        self._manifest = {
            "model_name": self.settings.embedding_model_name,
            "dimension": dim,
            "chunk_count": n,
            "built_at": datetime.now(UTC).isoformat(),
            "index_type": index_type,
            "id_mapped": ids is not None,
        }
        log.info("built FAISS index", type=index_type, n=n, dim=dim)

    @staticmethod
    def _build_ivf(vectors: np.ndarray, dim: int) -> faiss.IndexIVFFlat:
        n = len(vectors)
        # Start with sqrt(n), then back off until FAISS won't warn about
        # insufficient training points (rule of thumb: 39 × nlist).
        nlist = min(max(4, int(n**0.5)), 1024)
        while nlist > 1 and 39 * nlist > n:
            nlist = max(1, nlist // 2)

        quantiser = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vectors)
        ivf.nprobe = _IVF_NPROBE
        return ivf

    def update(
        self,
        *,
        remove_ids: np.ndarray,
        add_vectors: np.ndarray,
        add_ids: np.ndarray,
    ) -> None:
        """Incrementally remove and add vectors by stable FAISS IDs."""
        if self._index is None or self._manifest is None:
            raise RuntimeError("Call load() or build() before update().")
        # Incremental updates address FAISS vectors by explicit IDs, not by row
        # position. A plain Flat/IVF index cannot safely remove and replace rows.
        if self._manifest.get("id_mapped") is not True:
            raise RuntimeError("Incremental update requires an ID-mapped FAISS index.")

        remove_ids = np.ascontiguousarray(remove_ids, dtype=np.int64)
        add_vectors = np.ascontiguousarray(add_vectors, dtype=np.float32)
        add_ids = np.ascontiguousarray(add_ids, dtype=np.int64)

        if add_vectors.ndim != 2:
            raise ValueError(f"Expected 2-D add_vectors, got shape {add_vectors.shape}")
        if not np.isfinite(add_vectors).all():
            raise ValueError("FAISS add_vectors must contain only finite values")
        if len(np.unique(remove_ids)) != len(remove_ids):
            raise ValueError("FAISS remove ids must be unique")
        expected_dim = int(self._index.d)
        if add_vectors.shape[1] != expected_dim:
            raise ValueError(
                f"Expected add_vectors dimension {expected_dim}, got {add_vectors.shape[1]}"
            )
        if len(add_vectors) != len(add_ids):
            raise ValueError(f"Expected {len(add_vectors)} add ids, got {len(add_ids)}")
        if len(np.unique(add_ids)) != len(add_ids):
            raise ValueError("FAISS add ids must be unique")

        removed = 0
        if len(remove_ids):
            removed = int(self._index.remove_ids(remove_ids))
        if len(add_ids):
            self._index.add_with_ids(add_vectors, add_ids)

        self._manifest.update(
            {
                "chunk_count": int(self._index.ntotal),
                "built_at": datetime.now(UTC).isoformat(),
            }
        )
        log.info("updated FAISS index", removed=removed, added=len(add_ids), n=self.ntotal)

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

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestMismatchError(f"Index manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ManifestMismatchError("Index manifest must be a JSON object.")
        self._validate_manifest(manifest, expected_dim=expected_dim)

        self._index = faiss.read_index(str(index_path))
        if isinstance(self._index, faiss.IndexIVFFlat):
            self._index.nprobe = _IVF_NPROBE
        elif isinstance(self._index, faiss.IndexIDMap):
            inner = faiss.downcast_index(self._index.index)
            if isinstance(inner, faiss.IndexIVFFlat):
                inner.nprobe = _IVF_NPROBE
        self._manifest = manifest
        log.info(
            "loaded index",
            type=manifest.get("index_type"),
            chunks=manifest.get("chunk_count"),
        )

    def _validate_manifest(self, manifest: dict, *, expected_dim: int | None = None) -> None:
        saved_model = manifest.get("model_name", "")
        current_model = self.settings.embedding_model_name
        if saved_model != current_model:
            raise ManifestMismatchError(
                f"Index was built with model '{saved_model}' but settings specify "
                f"'{current_model}'. Re-index with --rebuild."
            )
        saved_dim = manifest.get("dimension")
        if not isinstance(saved_dim, int) or saved_dim <= 0:
            raise ManifestMismatchError("Index manifest dimension must be a positive integer.")
        saved_chunks = manifest.get("chunk_count")
        if not isinstance(saved_chunks, int) or saved_chunks < 0:
            raise ManifestMismatchError("Index manifest chunk_count must be a non-negative integer.")
        if expected_dim is not None:
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
        if k <= 0:
            raise ValueError("k must be positive")

        vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]
        if vec.ndim != 2 or vec.shape[0] != 1:
            raise ValueError(f"Expected one query vector, got shape {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError("query vector must contain only finite values")
        expected_dim = int(self._index.d)
        if vec.shape[1] != expected_dim:
            raise ValueError(f"Expected query dimension {expected_dim}, got {vec.shape[1]}")

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
        if ivf_threshold <= 0:
            raise ValueError("ivf_threshold must be positive")
        self.ivf_threshold = ivf_threshold
        self.last_stats: dict[str, int | float | bool] = {}

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    def run(self, repo_path: Path, *, rebuild: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
        """Execute the indexing pipeline and return (DataFrame, vectors).

        The return values are provided for callers that want to inspect results;
        all artifacts are also written to disk as a side-effect.
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

        import time

        pipeline_start = time.perf_counter()

        old_df = self._load_existing_metadata() if not rebuild else None

        # 1. Ingest
        log.info("pipeline: ingesting repo", repo=str(repo_path))
        df = CodeIngestor(repo_path, self.settings).ingest(persist=False)
        df = ensure_content_hashes(df, self.settings)
        log.info("pipeline: ingested", chunks=len(df))

        dimension = self.embedder.dimension
        manifest_changed = self._manifest_changed(expected_dim=dimension)
        legacy_index = self._legacy_unmapped_index()
        index_artifacts_missing = self._index_artifacts_missing()
        full_rebuild = bool(
            rebuild or manifest_changed or legacy_index or index_artifacts_missing or old_df is None
        )
        if manifest_changed and not rebuild:
            log.info("pipeline: manifest changed, forcing full rebuild")
        if legacy_index and not rebuild:
            log.info("pipeline: legacy unmapped index, forcing one-time full rebuild")
        if index_artifacts_missing and not rebuild:
            log.info("pipeline: index artifacts missing, forcing full rebuild")

        if old_df is not None:
            old_df = ensure_content_hashes(old_df, self.settings)
        df = self._assign_vector_ids(df, None if full_rebuild else old_df)
        diff_stats = self._diff_metadata(None if full_rebuild else old_df, df)
        log.info(
            "pipeline: metadata diff",
            added=diff_stats["added"],
            updated=diff_stats["updated"],
            removed=diff_stats["removed"],
            unchanged=diff_stats["unchanged"],
            full_rebuild=full_rebuild,
        )

        # 2. Embed
        log.info("pipeline: embedding chunks")
        vectors, embedding_stats = embed_dataframe_cached(
            df,
            embedder=self.embedder,
            reset_cache=full_rebuild,
        )
        log.info(
            "pipeline: embedded",
            shape=list(vectors.shape),
            chunks_embedded=embedding_stats["chunks_embedded"],
            cache_hits=embedding_stats["cache_hits"],
            elapsed_ms=embedding_stats["elapsed_ms"],
        )

        # 3. Build + persist FAISS index
        store = VectorStore(self.settings, ivf_threshold=self.ivf_threshold)
        vector_ids = df["vector_id"].to_numpy(dtype=np.int64)
        if full_rebuild:
            store.build(vectors, ids=vector_ids)
        else:
            index_diff = self._index_diff(old_df, df)
            changed_or_added = index_diff["changed_or_added"]
            add_mask = df["chunk_id"].isin(changed_or_added).to_numpy()
            store.load(expected_dim=dimension)
            store.update(
                remove_ids=np.asarray(index_diff["remove_ids"], dtype=np.int64),
                add_vectors=vectors[add_mask],
                add_ids=vector_ids[add_mask],
            )
        store.save()

        # 4. Build + persist BM25 corpus (import here to avoid module-level cycle)
        from semcode.search._bm25 import BM25Retriever, bm25_corpus_path  # noqa: PLC0415

        bm25_path = bm25_corpus_path(self.settings.faiss_index_path)
        previous_corpus = None
        if not full_rebuild and bm25_path.exists():
            previous_corpus = BM25Retriever.load(bm25_path).corpus
        bm25 = BM25Retriever.from_incremental_dataframe(
            df,
            previous_df=None if full_rebuild else old_df,
            previous_corpus=previous_corpus,
        )
        bm25.save(bm25_path)

        self.settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.settings.metadata_path, index=False)
        log.info("wrote metadata", path=str(self.settings.metadata_path), rows=len(df))

        elapsed_ms = round((time.perf_counter() - pipeline_start) * 1000, 3)
        self.last_stats = {
            **diff_stats,
            **embedding_stats,
            "embedding_elapsed_ms": embedding_stats["elapsed_ms"],
            "full_rebuild": full_rebuild,
            "elapsed_ms": elapsed_ms,
        }

        log.info(
            "pipeline: complete",
            chunks=len(df),
            dimension=int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] else 0,
            index_type=store.manifest.get("index_type") if store.manifest else "n/a",
            chunks_embedded=embedding_stats["chunks_embedded"],
            cache_hits=embedding_stats["cache_hits"],
            elapsed_ms=elapsed_ms,
        )
        return df, vectors

    def _load_existing_metadata(self) -> pd.DataFrame | None:
        if not self.settings.metadata_path.exists():
            return None
        return pd.read_parquet(self.settings.metadata_path)

    def _index_artifacts_missing(self) -> bool:
        return not (
            self.settings.faiss_index_path.exists()
            and self.settings.faiss_index_path.with_suffix(".json").exists()
        )

    def _manifest_changed(self, *, expected_dim: int) -> bool:
        manifest_path = self.settings.faiss_index_path.with_suffix(".json")
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return (
            manifest.get("model_name") != self.settings.embedding_model_name
            or manifest.get("dimension") != expected_dim
        )

    def _legacy_unmapped_index(self) -> bool:
        manifest_path = self.settings.faiss_index_path.with_suffix(".json")
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return manifest.get("id_mapped") is not True

    @staticmethod
    def _diff_metadata(
        old_df: pd.DataFrame | None,
        new_df: pd.DataFrame,
    ) -> dict[str, int]:
        if old_df is None:
            return {
                "added": int(len(new_df)),
                "updated": 0,
                "removed": 0,
                "unchanged": 0,
            }

        old_by_id = {
            str(row["chunk_id"]): str(row.get("content_hash", "")) for _, row in old_df.iterrows()
        }
        new_by_id = {
            str(row["chunk_id"]): str(row.get("content_hash", "")) for _, row in new_df.iterrows()
        }

        old_ids = set(old_by_id)
        new_ids = set(new_by_id)
        shared = old_ids & new_ids
        unchanged = sum(1 for chunk_id in shared if old_by_id[chunk_id] == new_by_id[chunk_id])
        updated = len(shared) - unchanged
        return {
            "added": len(new_ids - old_ids),
            "updated": updated,
            "removed": len(old_ids - new_ids),
            "unchanged": unchanged,
        }

    @staticmethod
    def _assign_vector_ids(
        df: pd.DataFrame,
        old_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        IndexingPipeline._validate_unique_chunk_ids(df, name="new metadata")
        out = df.copy()
        if old_df is None or "vector_id" not in old_df.columns:
            out["vector_id"] = list(range(len(out)))
            return out

        IndexingPipeline._validate_unique_chunk_ids(old_df, name="previous metadata")
        if old_df["vector_id"].duplicated().any():
            raise ValueError("previous metadata vector_id values must be unique")
        old_ids = {str(row["chunk_id"]): int(row["vector_id"]) for _, row in old_df.iterrows()}
        next_id = max(old_ids.values(), default=-1) + 1
        vector_ids: list[int] = []
        for _, row in out.iterrows():
            chunk_id = str(row["chunk_id"])
            if chunk_id in old_ids:
                vector_ids.append(old_ids[chunk_id])
            else:
                vector_ids.append(next_id)
                next_id += 1
        out["vector_id"] = vector_ids
        return out

    @staticmethod
    def _validate_unique_chunk_ids(df: pd.DataFrame, *, name: str) -> None:
        if "chunk_id" in df.columns and df["chunk_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} chunk_id values must be unique")

    @staticmethod
    def _index_diff(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> dict[str, list]:
        if old_df is None:
            return {
                "remove_ids": [],
                "changed_or_added": [str(value) for value in new_df["chunk_id"].tolist()],
            }

        old_by_chunk = {
            str(row["chunk_id"]): (
                str(row.get("content_hash", "")),
                int(row["vector_id"]),
            )
            for _, row in old_df.iterrows()
        }
        new_by_chunk = {
            str(row["chunk_id"]): str(row.get("content_hash", "")) for _, row in new_df.iterrows()
        }
        changed_or_added: list[str] = []
        remove_ids: list[int] = []

        for chunk_id, (old_hash, vector_id) in old_by_chunk.items():
            new_hash = new_by_chunk.get(chunk_id)
            if new_hash is None or new_hash != old_hash:
                remove_ids.append(vector_id)

        for _, row in new_df.iterrows():
            chunk_id = str(row["chunk_id"])
            previous = old_by_chunk.get(chunk_id)
            if previous is None or previous[0] != str(row.get("content_hash", "")):
                changed_or_added.append(chunk_id)

        return {"remove_ids": remove_ids, "changed_or_added": changed_or_added}
