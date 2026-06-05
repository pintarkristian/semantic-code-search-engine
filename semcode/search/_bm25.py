"""BM25 retriever with identifier-aware tokenization.

Public surface:
    tokenize(text)         — split camelCase/snake_case, lowercase, filter short tokens
    BM25Retriever          — BM25Okapi wrapper with save/load and DataFrame builder
    bm25_corpus_path(path) — derive BM25 corpus file path from a FAISS index path
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from semcode.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


def bm25_corpus_path(faiss_index_path: Path) -> Path:
    """Derive the BM25 corpus pickle path from the FAISS index path.

    Example: data/index.faiss -> data/index_bm25.pkl
    """
    return faiss_index_path.with_name(faiss_index_path.stem + "_bm25.pkl")


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokenization.

    Splits camelCase and snake_case into subwords, lowercases everything,
    and filters tokens shorter than 2 characters.

    Examples:
        "formatDate"     -> ["format", "date"]
        "validate_token" -> ["validate", "token"]
        "XMLParser"      -> ["xml", "parser"]
        "QueryBuilder"   -> ["query", "builder"]
    """
    # camelCase: insert space before uppercase that follows a lowercase letter
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # ALL-CAPS prefix: insert space before uppercase+lowercase run (e.g. XMLParser -> XML Parser)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # Split on anything that is not a letter or digit (handles _, ., -, whitespace, …)
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """BM25Okapi retriever over a tokenized code corpus.

    The corpus is a list of token lists that maps 1:1 to the metadata DataFrame
    rows (and thus to the FAISS index rows).  Only the corpus (list of lists) is
    persisted to disk; the BM25Okapi object is rebuilt on load.

    Args:
        corpus: list of token-lists, one per document.
    """

    def __init__(self, corpus: list[list[str]], *, doc_ids: list[int] | None = None) -> None:
        # BM25Okapi expects a sequence of token sequences. Validate this before
        # construction so malformed persisted corpora fail with our error text.
        for doc in corpus:
            if not isinstance(doc, list):
                raise ValueError("BM25 corpus documents must be token lists")
            if any(not isinstance(token, str) for token in doc):
                raise ValueError("BM25 corpus tokens must be strings")
        self._corpus = corpus
        self._doc_ids = doc_ids or list(range(len(corpus)))
        if len(self._doc_ids) != len(corpus):
            raise ValueError(f"Expected {len(corpus)} doc_ids, got {len(self._doc_ids)}")
        self._bm25: BM25Okapi | None = BM25Okapi(corpus) if corpus else None

    @property
    def corpus(self) -> list[list[str]]:
        """Tokenized documents in metadata row order."""
        return self._corpus

    @property
    def doc_ids(self) -> list[int]:
        """Document IDs aligned to the tokenized corpus."""
        return self._doc_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (row_idx, bm25_score) pairs sorted by score descending.

        Documents with a BM25 score of zero are excluded.
        """
        if k <= 0:
            return []
        if self._bm25 is None or not self._corpus:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        top_k = min(k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(self._doc_ids[int(i)]), float(scores[i])) for i in indices if scores[i] > 0.0]

    # ------------------------------------------------------------------
    # Build from DataFrame
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> BM25Retriever:
        """Build a BM25Retriever by tokenizing each row's text representation.

        Uses the same ``chunk_to_text`` as the embedding pipeline so the token
        space is consistent with what the model sees.
        """
        from semcode.embed import chunk_to_text  # local import avoids top-level cycle

        corpus = [tokenize(chunk_to_text(row)) for _, row in df.iterrows()]
        doc_ids = (
            [int(value) for value in df["vector_id"].tolist()]
            if "vector_id" in df.columns
            else list(range(len(corpus)))
        )
        return cls(corpus, doc_ids=doc_ids)

    @classmethod
    def from_incremental_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        previous_df: pd.DataFrame | None = None,
        previous_corpus: list[list[str]] | None = None,
    ) -> BM25Retriever:
        """Build a corpus reusing unchanged token lists from a previous corpus."""
        from semcode.embed import chunk_to_text  # local import avoids top-level cycle

        previous_by_id: dict[str, tuple[str, list[str]]] = {}
        if previous_df is not None and previous_corpus is not None:
            for (_, row), tokens in zip(previous_df.iterrows(), previous_corpus):
                previous_by_id[str(row["chunk_id"])] = (
                    str(row.get("content_hash", "")),
                    tokens,
                )

        corpus: list[list[str]] = []
        doc_ids: list[int] = []
        reused = 0
        for _, row in df.iterrows():
            doc_ids.append(int(row.get("vector_id", len(doc_ids))))
            previous = previous_by_id.get(str(row["chunk_id"]))
            if previous is not None and previous[0] == str(row.get("content_hash", "")):
                corpus.append(previous[1])
                reused += 1
            else:
                corpus.append(tokenize(chunk_to_text(row)))

        log.info("built BM25 corpus", docs=len(corpus), reused=reused)
        return cls(corpus, doc_ids=doc_ids)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the tokenized corpus to disk as a pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "doc_ids": self._doc_ids}, f)
        log.info("saved BM25 corpus", path=str(path), docs=len(self._corpus))

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load a previously saved tokenized corpus and reconstruct BM25Okapi."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            # Current payloads store explicit FAISS IDs; older payloads were
            # plain corpus lists and are handled by the compatibility branch.
            if "corpus" not in payload:
                raise ValueError("BM25 corpus payload is missing 'corpus'")
            corpus: list[list[str]] = payload["corpus"]
            if not isinstance(corpus, list):
                raise ValueError("BM25 corpus payload must contain a list corpus")
            doc_ids = [int(value) for value in payload.get("doc_ids", range(len(corpus)))]
        else:
            corpus = payload
            doc_ids = list(range(len(corpus)))
        if not isinstance(corpus, list):
            raise ValueError("BM25 corpus payload must be a list or object")
        log.info("loaded BM25 corpus", path=str(path), docs=len(corpus))
        return cls(corpus, doc_ids=doc_ids)
