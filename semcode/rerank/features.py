"""Feature engineering for the learned re-ranker."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from semcode.search._bm25 import tokenize

LANGUAGES: tuple[str, ...] = ("go", "java", "javascript", "python", "typescript")

FEATURE_COLUMNS: list[str] = [
    "dense_score",
    "bm25_score",
    "fused_score",
    "symbol_token_overlap",
    "query_code_len_ratio",
    "query_tokens_in_docstring",
    "lang_go",
    "lang_java",
    "lang_javascript",
    "lang_python",
    "lang_typescript",
    "lang_other",
]


def _tokens(text: object) -> set[str]:
    return set(tokenize(str(text or "")))


def _numeric_feature(row: pd.Series, column: str) -> float:
    value = float(row.get(column, 0.0) or 0.0)
    if not math.isfinite(value):
        raise ValueError(f"{column} must be finite")
    return value


def build_features(query: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """Build a stable numeric feature matrix for ``query`` and candidate chunks.

    ``candidates`` is expected to contain score columns from hybrid search plus
    metadata columns from ingest. Missing optional columns are treated as empty
    strings or zero scores so the function can also be used in focused tests.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must contain non-whitespace text")
    query_tokens = _tokens(query)
    query_len = max(len(query_tokens), 1)

    rows: list[dict[str, float]] = []
    for _, row in candidates.iterrows():
        symbol_tokens = _tokens(row.get("symbol_name", ""))
        code_tokens = _tokens(row.get("code", ""))
        doc_tokens = _tokens(row.get("docstring", ""))
        language = str(row.get("language", "") or "").lower()

        symbol_overlap = len(query_tokens & symbol_tokens)
        docstring_hit = 1.0 if query_tokens and bool(query_tokens & doc_tokens) else 0.0
        code_len = max(len(code_tokens), 1)

        features: dict[str, float] = {
            # Score features come from retrieval artifacts. Validate them here
            # before a bad index or test double can poison model training.
            "dense_score": _numeric_feature(row, "dense_score"),
            "bm25_score": _numeric_feature(row, "bm25_score"),
            "fused_score": _numeric_feature(row, "fused_score"),
            "symbol_token_overlap": float(symbol_overlap),
            "query_code_len_ratio": float(query_len / code_len),
            "query_tokens_in_docstring": docstring_hit,
        }
        for lang in LANGUAGES:
            features[f"lang_{lang}"] = 1.0 if language == lang else 0.0
        features["lang_other"] = 0.0 if language in LANGUAGES else 1.0
        rows.append(features)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype="float32")


def add_labels(
    query: str,
    candidates: pd.DataFrame,
    relevant_chunk_ids: Iterable[str],
) -> pd.DataFrame:
    """Return feature rows plus label and lightweight candidate metadata."""
    if "chunk_id" not in candidates.columns:
        raise ValueError("candidates must include a chunk_id column")
    if isinstance(relevant_chunk_ids, str):
        raise TypeError("relevant_chunk_ids must be an iterable of chunk ID strings, not a string")
    normalized_relevant = [str(chunk_id).strip() for chunk_id in relevant_chunk_ids]
    if any(not chunk_id for chunk_id in normalized_relevant):
        raise ValueError("relevant_chunk_ids must contain non-whitespace text")
    if len(set(normalized_relevant)) != len(normalized_relevant):
        raise ValueError("relevant_chunk_ids must be unique")
    relevant = set(normalized_relevant)
    features = build_features(query, candidates)
    labeled = features.copy()
    labeled.insert(0, "label", candidates["chunk_id"].astype(str).isin(relevant).astype("float32"))
    labeled.insert(0, "query", query)

    for column in ("chunk_id", "symbol_name", "file_path", "language"):
        if column in candidates.columns:
            labeled[column] = candidates[column].astype(str).to_numpy()

    return labeled
