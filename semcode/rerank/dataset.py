"""Build supervised training rows for the learned re-ranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from semcode.config import Settings, get_settings
from semcode.logging import get_logger
from semcode.rerank.features import add_labels
from semcode.search import Searcher

log = get_logger(__name__)


def load_labels(path: Path) -> dict[str, list[str]]:
    """Load query -> relevant chunk_id labels from JSON.

    Supported shapes:
      {"query text": ["chunk_id", ...]}
      [{"query": "query text", "relevant_chunk_ids": ["chunk_id", ...]}, ...]
    """
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # Labels are user-authored input. Re-raise with the path so CLI/API
        # callers can point users at the exact file that needs correction.
        raise ValueError(f"Failed to parse labels JSON at {path}: {exc.msg}") from exc
    if isinstance(raw, dict):
        labels: dict[str, list[str]] = {}
        for query, chunk_ids in raw.items():
            query_text = _label_query_text(query)
            if query_text in labels:
                raise ValueError(f"Duplicate label query: {query_text!r}")
            labels[query_text] = _label_chunk_ids(
                chunk_ids,
                message="Label values must be strings or lists of chunk IDs.",
            )
        return labels

    if isinstance(raw, list):
        labels = {}
        for item in raw:
            if not isinstance(item, dict) or "query" not in item:
                raise ValueError("Label list entries must contain a 'query' field.")
            query_text = _label_query_text(item["query"])
            if query_text in labels:
                raise ValueError(f"Duplicate label query: {query_text!r}")
            chunk_ids = item.get("relevant_chunk_ids", item.get("relevant", []))
            labels[query_text] = _label_chunk_ids(
                chunk_ids,
                message="Label entry relevant IDs must be a string or list.",
            )
        return labels

    raise ValueError("Labels JSON must be an object or list.")


def _label_query_text(value: object) -> str:
    # Labels are keyed by the exact query text used for candidate retrieval, so
    # normalize once here instead of letting dict/list label formats diverge.
    if not isinstance(value, str):
        raise ValueError("Label queries must be strings.")
    query = str(value).strip()
    if not query:
        raise ValueError("Label queries must contain non-whitespace text.")
    return query


def _label_chunk_ids(value: object, *, message: str) -> list[str]:
    if isinstance(value, str):
        raw_chunk_ids = [value]
    elif isinstance(value, list):
        raw_chunk_ids = value
    else:
        raise ValueError(message)
    chunk_ids = [str(chunk_id).strip() for chunk_id in raw_chunk_ids]
    if not chunk_ids:
        raise ValueError("Label entries must include at least one relevant chunk ID.")
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("Label chunk IDs must contain non-whitespace text.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Label chunk IDs must be unique per query.")
    return chunk_ids


def build_reranker_dataset(
    labels_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    *,
    candidates_per_query: int | None = None,
    negatives_per_query: int = 8,
) -> pd.DataFrame:
    """Retrieve hybrid candidates, label positives/negatives, and save parquet."""
    settings = settings or get_settings()
    if candidates_per_query is not None and candidates_per_query <= 0:
        raise ValueError("candidates_per_query must be positive")
    if negatives_per_query < 0:
        raise ValueError("negatives_per_query must be non-negative")
    labels = load_labels(labels_path)
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")

    searcher = Searcher(settings)
    frames: list[pd.DataFrame] = []
    candidate_limit = candidates_per_query or settings.top_k_retrieve

    for query, relevant_chunk_ids in labels.items():
        candidates = searcher.candidates(query, k=candidate_limit)
        if candidates.empty:
            log.warning("no candidates for reranker query", query=query)
            continue

        labeled = add_labels(query, candidates, relevant_chunk_ids)
        positives = labeled[labeled["label"] == 1.0]
        negatives = labeled[labeled["label"] == 0.0].head(negatives_per_query)
        if positives.empty:
            log.warning("labeled positive not retrieved", query=query, labels=relevant_chunk_ids)
        frames.append(pd.concat([positives, negatives], ignore_index=True))

    if not frames:
        raise ValueError("No reranker rows were generated.")

    dataset = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    log.info(
        "wrote reranker dataset",
        path=str(output_path),
        rows=len(dataset),
        positives=int(dataset["label"].sum()),
    )
    return dataset
