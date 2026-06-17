"""Dense + BM25 hybrid search with Reciprocal Rank Fusion.

Public surface:
    SearchResult   — pydantic result model (dense / bm25 / fused scores)
    Searcher       — load once, search many times (RRF fusion)
    format_results — terminal renderer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

from semcode.config import Settings, get_settings
from semcode.embed import Embedder
from semcode.index import VectorStore
from semcode.logging import get_logger
from semcode.search._bm25 import BM25Retriever, bm25_corpus_path

if TYPE_CHECKING:
    from semcode.rerank import ReRanker

log = get_logger(__name__)

_SNIPPET_LINES = 6
_RRF_K: int = 60  # standard RRF smoothing constant


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One ranked search result.

    ``score`` equals ``fused_score`` and is kept for backward compatibility
    with formatters and callers that were written before hybrid retrieval.
    The individual ``dense_score`` and ``bm25_score`` are raw values (cosine
    similarity and BM25 score respectively); they are zero when a document was
    not retrieved by that source.  All three are available for the M7 re-ranker.
    """

    rank: int
    score: float  # == fused_score (backward-compat alias)
    rerank_score: float | None = None
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    chunk_id: str = ""
    file_path: str
    symbol_name: str
    symbol_type: str
    language: str
    start_line: int
    end_line: int
    snippet: str


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    dense_hits: list[tuple[int, float]],
    bm25_hits: list[tuple[int, float]],
    *,
    dense_weight: float,
    bm25_weight: float,
    k: int = _RRF_K,
) -> list[tuple[int, float, float, float]]:
    """Weighted RRF over dense and BM25 hit lists.

    Args:
        dense_hits: [(row_idx, cosine_score), …] sorted by score descending.
        bm25_hits:  [(row_idx, bm25_score),   …] sorted by score descending.
        dense_weight: weight applied to dense RRF contributions.
        bm25_weight:  weight applied to BM25 RRF contributions.
        k: RRF smoothing constant (default 60).

    Returns:
        [(row_idx, dense_score, bm25_score, fused_score), …] sorted by
        fused_score descending.  Documents appearing in only one list get
        a zero contribution from the missing source.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("RRF weights must be non-negative")
    if dense_weight + bm25_weight <= 0:
        raise ValueError("At least one RRF weight must be positive")

    dense_map: dict[int, tuple[int, float]] = {
        row_idx: (rank + 1, score) for rank, (row_idx, score) in enumerate(dense_hits)
    }
    bm25_map: dict[int, tuple[int, float]] = {
        row_idx: (rank + 1, score) for rank, (row_idx, score) in enumerate(bm25_hits)
    }

    fused: list[tuple[int, float, float, float]] = []
    for row_idx in set(dense_map) | set(bm25_map):
        d_rank, d_score = dense_map.get(row_idx, (0, 0.0))
        b_rank, b_score = bm25_map.get(row_idx, (0, 0.0))
        d_rrf = dense_weight / (k + d_rank) if d_rank > 0 else 0.0
        b_rrf = bm25_weight / (k + b_rank) if b_rank > 0 else 0.0
        fused.append((row_idx, d_score, b_score, d_rrf + b_rrf))

    fused.sort(key=lambda x: x[3], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


class Searcher:
    """Load VectorStore + BM25 + metadata once; answer many search queries.

    ``search()`` runs dense FAISS retrieval and BM25 retrieval independently,
    then fuses the two ranked lists with weighted Reciprocal Rank Fusion.

    Args:
        settings: application settings (defaults to get_settings()).
        embedder: pre-built Embedder; if None, one is created lazily.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._store: VectorStore | None = None
        self._bm25: BM25Retriever | None = None
        self._meta: pd.DataFrame | None = None
        self._doc_id_to_pos: dict[int, int] = {}
        self._reranker: ReRanker | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    def _ensure_loaded(self) -> None:
        if self._store is not None:
            return

        meta_path = self.settings.metadata_path
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata at {meta_path}. Run `semcode index <repo>` first."
            )

        store = VectorStore(self.settings)
        store.load(expected_dim=self.embedder.dimension)

        meta = pd.read_parquet(meta_path)
        required_columns = {
            "chunk_id",
            "file_path",
            "symbol_name",
            "symbol_type",
            "language",
            "start_line",
            "end_line",
            "code",
        }
        missing_columns = sorted(required_columns - set(meta.columns))
        if missing_columns:
            raise ValueError(f"Metadata is missing required columns: {missing_columns}")
        if meta["chunk_id"].astype(str).duplicated().any():
            raise ValueError("Metadata chunk_id values must be unique.")
        for column in ("file_path", "symbol_name", "symbol_type", "language"):
            if meta[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Metadata {column} values must contain non-whitespace text.")
        try:
            start_lines = meta["start_line"].astype(int)
            end_lines = meta["end_line"].astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("Metadata line spans must be integers.") from exc
        if (start_lines < 1).any() or (end_lines < start_lines).any():
            raise ValueError("Metadata line spans must be 1-indexed and ordered.")

        bm25_path = bm25_corpus_path(self.settings.faiss_index_path)
        if bm25_path.exists():
            bm25 = BM25Retriever.load(bm25_path)
        else:
            log.warning("BM25 corpus not found, rebuilding from metadata", path=str(bm25_path))
            bm25 = BM25Retriever.from_dataframe(meta)
            bm25.save(bm25_path)

        self._store = store
        self._meta = meta
        if "vector_id" in meta.columns:
            if meta["vector_id"].duplicated().any():
                raise ValueError("Metadata vector_id values must be unique.")
            try:
                # BM25 and FAISS results are joined back through vector_id, so
                # validate metadata IDs before searches start returning rows.
                self._doc_id_to_pos = {
                    int(doc_id): int(pos) for pos, doc_id in enumerate(meta["vector_id"].tolist())
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("Metadata vector_id values must be integers.") from exc
        else:
            self._doc_id_to_pos = {int(pos): int(pos) for pos in range(len(meta))}
        missing_bm25_ids = sorted(set(bm25.doc_ids) - set(self._doc_id_to_pos))
        if missing_bm25_ids:
            raise ValueError(
                f"BM25 corpus contains document IDs not present in metadata: {missing_bm25_ids[:5]}"
            )
        self._bm25 = bm25
        log.info("searcher ready", chunks=len(meta), ntotal=store.ntotal)

    def candidates(self, query: str, k: int | None = None) -> pd.DataFrame:
        """Return hybrid candidates with scores and metadata, sorted by fused score."""
        query = query.strip()
        if not query:
            raise ValueError("query must contain non-whitespace text")
        if len(query) > self.settings.max_query_length:
            raise ValueError(
                f"query must be at most {self.settings.max_query_length} characters"
            )
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k is not None and k <= 0:
            raise ValueError("k must be positive")
        if k is not None and k > self.settings.max_search_k:
            raise ValueError(f"k must be less than or equal to {self.settings.max_search_k}")
        self._ensure_loaded()
        if self._store is None or self._bm25 is None or self._meta is None:
            raise RuntimeError("Searcher failed to load index artifacts.")

        # Candidate callers may ask for more rows than the default retrieval
        # pool. Fetch enough from each source before fusion to honor that limit.
        retrieve_n = max(self.settings.top_k_retrieve, k or 0)

        query_vec: np.ndarray = self.embedder.encode([query])[0]
        dense_hits = self._store.search(query_vec, retrieve_n)
        bm25_hits = self._bm25.search(query, retrieve_n)

        fused = _reciprocal_rank_fusion(
            dense_hits,
            bm25_hits,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        if k is not None:
            fused = fused[:k]

        rows: list[dict] = []
        for doc_id, d_score, b_score, f_score in fused:
            row_pos = self._doc_id_to_pos.get(int(doc_id))
            if row_pos is None:
                continue
            row = self._meta.iloc[row_pos].to_dict()
            row.update(
                {
                    "row_idx": int(row_pos),
                    "doc_id": int(doc_id),
                    "dense_score": float(d_score),
                    "bm25_score": float(b_score),
                    "fused_score": float(f_score),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _maybe_rerank(
        self, query: str, candidates: pd.DataFrame, use_reranker: bool
    ) -> pd.DataFrame:
        if not use_reranker or candidates.empty:
            return candidates
        if self._reranker is None:
            from semcode.rerank import ReRanker  # noqa: PLC0415

            self._reranker = ReRanker(self.settings)
        if not self._reranker.available:
            return candidates
        return self._reranker.rerank(query, candidates)

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        use_reranker: bool | None = None,
    ) -> list[SearchResult]:
        """Embed query, retrieve from dense + BM25, fuse with RRF, return top-k.

        Args:
            query: natural-language search query.
            k:     number of results; defaults to settings.top_k_return.

        Returns:
            List of SearchResult sorted by fused_score descending.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must contain non-whitespace text")
        # Match API validation for direct Python callers and avoid embedding
        # unexpectedly large request bodies.
        if len(query) > self.settings.max_query_length:
            raise ValueError(
                f"query must be at most {self.settings.max_query_length} characters"
            )
        k = k if k is not None else self.settings.top_k_return
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        # Keep direct Python callers under the same bound as the HTTP surface.
        if k > self.settings.max_search_k:
            raise ValueError(f"k must be less than or equal to {self.settings.max_search_k}")
        reranker_enabled = self.settings.use_reranker if use_reranker is None else use_reranker
        candidates = self.candidates(query)
        ranked = self._maybe_rerank(query, candidates, reranker_enabled)

        results: list[SearchResult] = []
        for rank, (_, row) in enumerate(ranked.head(k).iterrows(), start=1):
            f_score = float(row["fused_score"])
            rerank_score = row.get("rerank_score")
            result_score = (
                float(rerank_score)
                if rerank_score is not None and pd.notna(rerank_score)
                else f_score
            )
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(result_score, 6),
                    rerank_score=(
                        round(float(rerank_score), 6)
                        if rerank_score is not None and pd.notna(rerank_score)
                        else None
                    ),
                    dense_score=round(float(row["dense_score"]), 4),
                    bm25_score=round(float(row["bm25_score"]), 4),
                    fused_score=round(f_score, 6),
                    chunk_id=str(row["chunk_id"]),
                    file_path=str(row["file_path"]),
                    symbol_name=str(row["symbol_name"]),
                    symbol_type=str(row["symbol_type"]),
                    language=str(row["language"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    snippet=_make_snippet(str(row["code"]), _SNIPPET_LINES),
                )
            )

        log.info("search complete", query=query, k=k, hits=len(results))
        return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _make_snippet(code: str, max_lines: int) -> str:
    """Return the first max_lines non-empty lines of code."""
    if not isinstance(max_lines, int):
        raise TypeError("max_lines must be an integer")
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    lines = code.splitlines()
    kept: list[str] = []
    for line in lines:
        if not kept and not line.strip():
            continue
        if len(kept) >= max_lines:
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def format_results(results: list[SearchResult], query: str = "", verbose: bool = False) -> str:
    """Render a list of SearchResult as a terminal-friendly string.

    Args:
        results: ranked results from Searcher.search().
        query:   original query string (shown in header when non-empty).
        verbose: when True, also show dense_score and bm25_score per result.
    """
    if not results:
        return "No results found."

    lines: list[str] = []
    if query:
        lines.append(f"Results for: {query!r}\n")

    for r in results:
        if verbose:
            score_str = f"score={r.score:.4f}  dense={r.dense_score:.4f}  bm25={r.bm25_score:.4f}"
            if r.rerank_score is not None:
                # In reranked output, score is the learned score; fused remains
                # useful as retrieval context when debugging ranking changes.
                score_str += f"  fused={r.fused_score:.4f}"
        else:
            score_str = f"score={r.score:.4f}"
        header = (
            f"#{r.rank}  {score_str}  "
            f"{r.file_path}:{r.start_line}  "
            f"({r.symbol_name} · {r.language})"
        )
        lines.append(header)
        for snippet_line in r.snippet.splitlines():
            lines.append(f"    {snippet_line}")
        lines.append("")

    return "\n".join(lines).rstrip()
