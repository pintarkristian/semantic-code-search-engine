"""Dense semantic search over a FAISS index.

Public surface:
    SearchResult   — pydantic result model
    Searcher       — load once, search many times
    format_results — terminal renderer
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from semcode.config import Settings, get_settings
from semcode.embed import Embedder, chunk_to_text
from semcode.index import VectorStore
from semcode.logging import get_logger

log = get_logger(__name__)

_SNIPPET_LINES = 6  # max lines shown per result in terminal output


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    rank: int
    score: float
    file_path: str
    symbol_name: str
    symbol_type: str
    language: str
    start_line: int
    end_line: int
    snippet: str


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class Searcher:
    """Load VectorStore + metadata once; answer many search queries.

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
        self._meta: pd.DataFrame | None = None

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

        self._meta = pd.read_parquet(meta_path)
        self._store = store
        log.info("searcher ready", chunks=len(self._meta), ntotal=store.ntotal)

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Embed the query, search FAISS, join to metadata, return ranked results.

        Args:
            query: natural-language search query.
            k:     number of results; defaults to settings.top_k_return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        self._ensure_loaded()

        k = k if k is not None else self.settings.top_k_return

        query_vec: np.ndarray = self.embedder.encode([query])[0]  # (dim,)
        hits = self._store.search(query_vec, k)  # [(row_idx, score), ...]

        results: list[SearchResult] = []
        for rank, (row_idx, score) in enumerate(hits, start=1):
            row = self._meta.iloc[row_idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(float(score), 4),
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
    lines = code.splitlines()
    kept: list[str] = []
    for line in lines:
        if len(kept) >= max_lines:
            break
        kept.append(line)
    # Strip trailing blank lines
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def format_results(results: list[SearchResult], query: str = "") -> str:
    """Render a list of SearchResult as a terminal-friendly string.

    Each hit shows:
        #rank  score  file_path:start_line  (symbol_name · language)
        snippet (indented)
    """
    if not results:
        return "No results found."

    lines: list[str] = []
    if query:
        lines.append(f"Results for: {query!r}\n")

    for r in results:
        header = (
            f"#{r.rank}  score={r.score:.4f}  "
            f"{r.file_path}:{r.start_line}  "
            f"({r.symbol_name} · {r.language})"
        )
        lines.append(header)
        for snippet_line in r.snippet.splitlines():
            lines.append(f"    {snippet_line}")
        lines.append("")

    return "\n".join(lines).rstrip()
