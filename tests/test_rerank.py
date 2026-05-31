"""Tests for semcode.rerank feature engineering, model IO, and search integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from semcode.config import Settings
from semcode.embed import Embedder
from semcode.index import IndexingPipeline
from semcode.rerank import (
    FEATURE_COLUMNS,
    ReRanker,
    add_labels,
    build_features,
    build_model,
    build_reranker_dataset,
    load_labels,
    train_reranker_model,
)
from semcode.search import Searcher
from tests.conftest import MockSentenceTransformer

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_model_name="test-model",
        data_dir=tmp_path,
        faiss_index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.parquet",
        reranker_model_path=tmp_path / "reranker",
        batch_size=8,
        top_k_retrieve=12,
        top_k_return=5,
    )


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "a",
                "dense_score": 0.8,
                "bm25_score": 2.0,
                "fused_score": 0.04,
                "symbol_name": "validate_token",
                "language": "python",
                "code": "def validate_token(token): return bool(token)",
                "docstring": "Validate a JWT token.",
            },
            {
                "chunk_id": "b",
                "dense_score": 0.2,
                "bm25_score": 0.0,
                "fused_score": 0.01,
                "symbol_name": "formatDate",
                "language": "javascript",
                "code": "function formatDate(date) { return date.toISOString(); }",
                "docstring": "Format a Date object.",
            },
        ]
    )


def test_feature_builder_columns_are_stable() -> None:
    features = build_features("validate JWT token", _candidate_frame())
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (2, len(FEATURE_COLUMNS))
    assert features["lang_python"].tolist() == [1.0, 0.0]
    assert features["query_tokens_in_docstring"].tolist() == [1.0, 0.0]


def test_add_labels_requires_chunk_id() -> None:
    candidates = _candidate_frame().drop(columns=["chunk_id"])

    with pytest.raises(ValueError, match="chunk_id"):
        add_labels("validate token", candidates, ["a"])


def test_load_labels_rejects_invalid_object_values(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"query": 123}), encoding="utf-8")

    with pytest.raises(ValueError, match="strings or lists"):
        load_labels(labels_path)


def test_load_labels_rejects_invalid_list_entry_values(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps([{"query": "validate token", "relevant_chunk_ids": 123}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="string or list"):
        load_labels(labels_path)


def test_load_labels_rejects_blank_queries(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"   ": ["chunk-1"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Label queries"):
        load_labels(labels_path)


def test_build_dataset_rejects_invalid_sampling_options(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"validate token": ["chunk-1"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="candidates_per_query"):
        build_reranker_dataset(labels_path, tmp_path / "dataset.parquet", candidates_per_query=0)

    with pytest.raises(ValueError, match="negatives_per_query"):
        build_reranker_dataset(labels_path, tmp_path / "dataset.parquet", negatives_per_query=-1)


def test_train_model_rejects_invalid_training_options(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing.parquet"

    with pytest.raises(ValueError, match="epochs"):
        train_reranker_model(dataset_path, epochs=0)

    with pytest.raises(ValueError, match="batch_size"):
        train_reranker_model(dataset_path, batch_size=0)


def test_build_model_rejects_invalid_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        build_model(0)


def test_reranker_score_rejects_blank_query(tmp_path: Path) -> None:
    reranker = ReRanker(_settings(tmp_path))

    with pytest.raises(ValueError, match="non-whitespace"):
        reranker.score("   ", _candidate_frame())


def test_reranker_score_falls_back_on_wrong_score_count(tmp_path: Path) -> None:
    class _BadSignature:
        def __call__(self, features: Any) -> dict[str, np.ndarray]:
            return {"scores": np.asarray([[0.5]], dtype="float32")}

    reranker = ReRanker(_settings(tmp_path))
    reranker._available = True
    reranker._model = object()
    reranker._signature = _BadSignature()

    scores = reranker.score("validate token", _candidate_frame())

    np.testing.assert_allclose(scores, _candidate_frame()["fused_score"].to_numpy("float32"))


def test_trained_toy_model_loads_and_scores_in_probability_range(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    features = build_features("validate token", _candidate_frame())
    dataset = features.copy()
    dataset.insert(0, "label", [1.0, 0.0])
    dataset_path = tmp_path / "toy.parquet"
    dataset.to_parquet(dataset_path, index=False)

    train_reranker_model(dataset_path, settings, epochs=2, batch_size=2)
    scores = ReRanker(settings).score("validate token", _candidate_frame())

    assert len(scores) == 2
    assert all(0.0 <= float(score) <= 1.0 for score in scores)


def test_reranking_changes_fixture_order_vs_fused_baseline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    embedder = Embedder(settings, _model=MockSentenceTransformer())
    IndexingPipeline(settings, embedder=embedder).run(FIXTURE_REPO)

    searcher = Searcher(settings, embedder=embedder)
    baseline = [r.chunk_id for r in searcher.search("function", k=5, use_reranker=False)]

    _export_inverse_fused_model(settings.reranker_model_path)
    reranked = [r.chunk_id for r in searcher.search("function", k=5, use_reranker=True)]

    assert baseline != reranked


def _export_inverse_fused_model(model_path: Path) -> None:
    import numpy as np
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(len(FEATURE_COLUMNS),), name="features")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="relevance")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    weights = np.zeros((len(FEATURE_COLUMNS), 1), dtype="float32")
    weights[FEATURE_COLUMNS.index("fused_score"), 0] = -200.0
    model.layers[-1].set_weights([weights, np.array([0.0], dtype="float32")])
    model.export(str(model_path))
