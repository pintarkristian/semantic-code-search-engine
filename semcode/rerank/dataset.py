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
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        labels: dict[str, list[str]] = {}
        for query, chunk_ids in raw.items():
            if isinstance(chunk_ids, str):
                labels[str(query)] = [chunk_ids]
            else:
                labels[str(query)] = [str(chunk_id) for chunk_id in chunk_ids]
        return labels

    if isinstance(raw, list):
        labels = {}
        for item in raw:
            if not isinstance(item, dict) or "query" not in item:
                raise ValueError("Label list entries must contain a 'query' field.")
            chunk_ids = item.get("relevant_chunk_ids", item.get("relevant", []))
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            labels[str(item["query"])] = [str(chunk_id) for chunk_id in chunk_ids]
        return labels

    raise ValueError("Labels JSON must be an object or list.")


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
