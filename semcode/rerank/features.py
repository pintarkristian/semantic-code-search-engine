"""Feature engineering for the learned re-ranker."""

from __future__ import annotations

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
            "dense_score": float(row.get("dense_score", 0.0) or 0.0),
            "bm25_score": float(row.get("bm25_score", 0.0) or 0.0),
            "fused_score": float(row.get("fused_score", 0.0) or 0.0),
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
    relevant = {str(chunk_id) for chunk_id in relevant_chunk_ids}
    features = build_features(query, candidates)
    labeled = features.copy()
    labeled.insert(0, "label", candidates["chunk_id"].astype(str).isin(relevant).astype("float32"))
    labeled.insert(0, "query", query)

    for column in ("chunk_id", "symbol_name", "file_path", "language"):
        if column in candidates.columns:
            labeled[column] = candidates[column].astype(str).to_numpy()

    return labeled
