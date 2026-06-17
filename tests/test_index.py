"""Tests for semcode.index — VectorStore and IndexingPipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from semcode.config import Settings
from semcode.embed import Embedder
from semcode.index import (
    IndexingPipeline,
    ManifestMismatchError,
    VectorStore,
)
from tests.conftest import MOCK_DIM, MockSentenceTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _unit_vectors(n: int, dim: int = MOCK_DIM, seed: int = 0) -> np.ndarray:
    """Return n random L2-normalised float32 vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _settings(tmp_path: Path, model_name: str = "test-model") -> Settings:
    return Settings(
        embedding_model_name=model_name,
        data_dir=tmp_path,
        faiss_index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.parquet",
        batch_size=8,
    )


def _mock_embedder(settings: Settings) -> Embedder:
    return Embedder(settings, _model=MockSentenceTransformer())


class CountingSentenceTransformer(MockSentenceTransformer):
    def __init__(self) -> None:
        self.encoded_chunks = 0

    def encode(
        self,
        inputs: list[str] | str,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = inputs if isinstance(inputs, list) else [inputs]
        self.encoded_chunks += len(texts)
        return super().encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )


def _copy_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, repo)
    return repo


# ---------------------------------------------------------------------------
# VectorStore — build
# ---------------------------------------------------------------------------


def test_build_flat_index(tmp_path: Path) -> None:
    vecs = _unit_vectors(50)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    assert store.ntotal == 50


def test_vector_store_rejects_invalid_ivf_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ivf_threshold"):
        VectorStore(_settings(tmp_path), ivf_threshold=0)


def test_build_sets_manifest(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    assert store.manifest is not None
    assert store.manifest["chunk_count"] == 10
    assert store.manifest["dimension"] == MOCK_DIM
    assert store.manifest["model_name"] == "test-model"
    assert store.manifest["index_type"] == "flat"


def test_build_ivf_above_threshold(tmp_path: Path) -> None:
    # Use ivf_threshold=20 so a small corpus triggers IVF
    vecs = _unit_vectors(100)
    store = VectorStore(_settings(tmp_path), ivf_threshold=20)
    store.build(vecs)
    assert store.manifest is not None
    assert store.manifest["index_type"] == "ivf"
    assert store.ntotal == 100


def test_build_empty_corpus(tmp_path: Path) -> None:
    vecs = np.zeros((0, MOCK_DIM), dtype=np.float32)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    assert store.ntotal == 0
    assert store.manifest is not None
    assert store.manifest["chunk_count"] == 0


def test_build_rejects_zero_width_vectors(tmp_path: Path) -> None:
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="dimension"):
        store.build(np.zeros((0, 0), dtype=np.float32))


def test_build_rejects_1d_array(tmp_path: Path) -> None:
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="2-D"):
        store.build(np.zeros(MOCK_DIM, dtype=np.float32))


def test_build_rejects_non_finite_vectors(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    vecs[0, 0] = np.nan
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="finite"):
        store.build(vecs)


def test_build_rejects_duplicate_faiss_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="ids must be unique"):
        store.build(vecs, ids=np.asarray([1, 1, 2], dtype=np.int64))


def test_build_rejects_non_integer_faiss_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="ids must be integers"):
        store.build(vecs, ids=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_build_rejects_non_1d_faiss_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(ValueError, match="1-D FAISS ids"):
        store.build(vecs, ids=np.asarray([[1], [2], [3]], dtype=np.int64))


# ---------------------------------------------------------------------------
# VectorStore — search
# ---------------------------------------------------------------------------


def test_nearest_neighbour_is_self(tmp_path: Path) -> None:
    """A vector's nearest neighbour in the index must be itself."""
    vecs = _unit_vectors(30)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)

    for i in range(len(vecs)):
        results = store.search(vecs[i], k=1)
        assert len(results) == 1
        row_idx, score = results[0]
        assert row_idx == i, f"Expected self at index {i}, got {row_idx}"
        assert score == pytest.approx(1.0, abs=1e-4)


def test_search_returns_k_results(tmp_path: Path) -> None:
    vecs = _unit_vectors(20)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    results = store.search(vecs[0], k=5)
    assert len(results) == 5


def test_search_k_clamped_to_ntotal(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    results = store.search(vecs[0], k=100)
    assert len(results) == 3


def test_search_rejects_non_positive_k(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    with pytest.raises(ValueError, match="k must be positive"):
        store.search(vecs[0], k=0)


def test_search_scores_descending(tmp_path: Path) -> None:
    vecs = _unit_vectors(15)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    results = store.search(vecs[0], k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_row_indices(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    results = store.search(vecs[0], k=3)
    for idx, _ in results:
        assert 0 <= idx < 10


def test_search_2d_query_accepted(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    query = vecs[0:1]  # shape (1, dim)
    results = store.search(query, k=1)
    assert results[0][0] == 0


def test_search_rejects_multiple_query_vectors(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    with pytest.raises(ValueError, match="one query vector"):
        store.search(vecs[:2], k=1)


def test_search_rejects_wrong_query_dimension(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    with pytest.raises(ValueError, match="query dimension"):
        store.search(np.zeros(MOCK_DIM + 1, dtype=np.float32), k=1)


def test_search_rejects_non_finite_query_vector(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    query = vecs[0].copy()
    query[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        store.search(query, k=1)


def test_search_before_build_raises(tmp_path: Path) -> None:
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="build"):
        store.search(np.zeros(MOCK_DIM, dtype=np.float32), k=1)


def test_search_empty_index_returns_empty(tmp_path: Path) -> None:
    vecs = np.zeros((0, MOCK_DIM), dtype=np.float32)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    results = store.search(np.zeros(MOCK_DIM, dtype=np.float32), k=5)
    assert results == []


# ---------------------------------------------------------------------------
# VectorStore — save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_index_file(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    store.save()
    assert (tmp_path / "index.faiss").exists()


def test_save_creates_manifest_file(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    store.save()
    assert (tmp_path / "index.json").exists()


def test_roundtrip_preserves_search_results(tmp_path: Path) -> None:
    """Search results must be identical before and after save/load."""
    vecs = _unit_vectors(25)
    s = _settings(tmp_path)

    store = VectorStore(s)
    store.build(vecs)
    before = store.search(vecs[3], k=5)
    store.save()

    store2 = VectorStore(s)
    store2.load()
    after = store2.search(vecs[3], k=5)

    assert before == after


def test_roundtrip_ntotal(tmp_path: Path) -> None:
    vecs = _unit_vectors(18)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()

    store2 = VectorStore(s)
    store2.load()
    assert store2.ntotal == 18


def test_roundtrip_nn_is_self_after_load(tmp_path: Path) -> None:
    vecs = _unit_vectors(20)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()

    store2 = VectorStore(s)
    store2.load()
    for i in range(len(vecs)):
        results = store2.search(vecs[i], k=1)
        assert results[0][0] == i


def test_roundtrip_ivf_preserves_results(tmp_path: Path) -> None:
    vecs = _unit_vectors(100)
    s = _settings(tmp_path)
    store = VectorStore(s, ivf_threshold=20)
    store.build(vecs)
    before = store.search(vecs[0], k=3)
    store.save()

    store2 = VectorStore(s, ivf_threshold=20)
    store2.load()
    after = store2.search(vecs[0], k=3)
    assert before == after


# ---------------------------------------------------------------------------
# VectorStore — manifest validation
# ---------------------------------------------------------------------------


def test_load_missing_index_raises(tmp_path: Path) -> None:
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.load()


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    # Write index but no manifest
    vecs = _unit_vectors(5)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()
    (tmp_path / "index.json").unlink()
    store2 = VectorStore(s)
    with pytest.raises(FileNotFoundError):
        store2.load()


def test_model_name_mismatch_raises(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    s_build = _settings(tmp_path, model_name="model-A")
    store = VectorStore(s_build)
    store.build(vecs)
    store.save()

    s_load = _settings(tmp_path, model_name="model-B")
    store2 = VectorStore(s_load)
    with pytest.raises(ManifestMismatchError, match="model"):
        store2.load()


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    vecs = _unit_vectors(10, dim=MOCK_DIM)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()

    store2 = VectorStore(s)
    with pytest.raises(ManifestMismatchError, match="dimension"):
        store2.load(expected_dim=MOCK_DIM + 1)


def test_load_rejects_invalid_manifest_dimension(tmp_path: Path) -> None:
    vecs = _unit_vectors(5)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()
    manifest_path = s.faiss_index_path.with_suffix(".json")
    manifest = manifest_path.read_text(encoding="utf-8").replace(f'"dimension": {MOCK_DIM}', '"dimension": 0')
    manifest_path.write_text(manifest, encoding="utf-8")

    store2 = VectorStore(s)
    with pytest.raises(ManifestMismatchError, match="positive integer"):
        store2.load()


def test_load_rejects_invalid_manifest_chunk_count(tmp_path: Path) -> None:
    vecs = _unit_vectors(5)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()
    manifest_path = s.faiss_index_path.with_suffix(".json")
    manifest = manifest_path.read_text(encoding="utf-8").replace('"chunk_count": 5', '"chunk_count": -1')
    manifest_path.write_text(manifest, encoding="utf-8")

    store2 = VectorStore(s)
    with pytest.raises(ManifestMismatchError, match="chunk_count"):
        store2.load()


def test_load_rejects_non_object_manifest(tmp_path: Path) -> None:
    vecs = _unit_vectors(5)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()
    s.faiss_index_path.with_suffix(".json").write_text("[]", encoding="utf-8")

    store2 = VectorStore(s)
    with pytest.raises(ManifestMismatchError, match="JSON object"):
        store2.load()


def test_load_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    vecs = _unit_vectors(5)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()
    s.faiss_index_path.with_suffix(".json").write_text("{not-json", encoding="utf-8")

    store2 = VectorStore(s)
    with pytest.raises(ManifestMismatchError, match="not valid JSON"):
        store2.load()


def test_correct_expected_dim_loads_fine(tmp_path: Path) -> None:
    vecs = _unit_vectors(10)
    s = _settings(tmp_path)
    store = VectorStore(s)
    store.build(vecs)
    store.save()

    store2 = VectorStore(s)
    store2.load(expected_dim=MOCK_DIM)  # should not raise
    assert store2.ntotal == 10


def test_save_before_build_raises(tmp_path: Path) -> None:
    store = VectorStore(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="build"):
        store.save()


# ---------------------------------------------------------------------------
# VectorStore — incremental update
# ---------------------------------------------------------------------------


def test_update_rejects_duplicate_add_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="add ids must be unique"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=_unit_vectors(2, seed=1),
            add_ids=np.asarray([3, 3], dtype=np.int64),
        )


def test_update_rejects_non_integer_add_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="add ids must be integers"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=_unit_vectors(1, seed=1),
            add_ids=np.asarray([3.5], dtype=np.float32),
        )


def test_update_rejects_non_1d_add_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="1-D FAISS add ids"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=_unit_vectors(1, seed=1),
            add_ids=np.asarray([[3]], dtype=np.int64),
        )


def test_update_rejects_duplicate_remove_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="remove ids must be unique"):
        store.update(
            remove_ids=np.asarray([1, 1], dtype=np.int64),
            add_vectors=np.zeros((0, MOCK_DIM), dtype=np.float32),
            add_ids=np.asarray([], dtype=np.int64),
        )


def test_update_rejects_non_integer_remove_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="remove ids must be integers"):
        store.update(
            remove_ids=np.asarray([1.5], dtype=np.float32),
            add_vectors=np.zeros((0, MOCK_DIM), dtype=np.float32),
            add_ids=np.asarray([], dtype=np.int64),
        )


def test_update_rejects_non_1d_remove_ids(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="1-D FAISS remove ids"):
        store.update(
            remove_ids=np.asarray([[1]], dtype=np.int64),
            add_vectors=np.zeros((0, MOCK_DIM), dtype=np.float32),
            add_ids=np.asarray([], dtype=np.int64),
        )


def test_update_rejects_wrong_add_vector_dimension(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="add_vectors dimension"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=np.zeros((1, MOCK_DIM + 1), dtype=np.float32),
            add_ids=np.asarray([3], dtype=np.int64),
        )


def test_update_rejects_non_finite_add_vectors(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs, ids=np.arange(3, dtype=np.int64))
    add_vectors = _unit_vectors(1, seed=1)
    add_vectors[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=add_vectors,
            add_ids=np.asarray([3], dtype=np.int64),
        )


def test_update_requires_id_mapped_index(tmp_path: Path) -> None:
    vecs = _unit_vectors(3)
    store = VectorStore(_settings(tmp_path))
    store.build(vecs)
    with pytest.raises(RuntimeError, match="ID-mapped"):
        store.update(
            remove_ids=np.asarray([], dtype=np.int64),
            add_vectors=_unit_vectors(1, seed=1),
            add_ids=np.asarray([3], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# IndexingPipeline
# ---------------------------------------------------------------------------


def test_pipeline_produces_parquet(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    pipeline.run(FIXTURE_REPO)
    assert s.metadata_path.exists()


def test_pipeline_rejects_invalid_ivf_threshold(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    with pytest.raises(ValueError, match="ivf_threshold"):
        IndexingPipeline(s, embedder=_mock_embedder(s), ivf_threshold=0)


def test_assign_vector_ids_rejects_duplicate_new_chunk_ids() -> None:
    df = pd.DataFrame({"chunk_id": ["same", "same"]})

    with pytest.raises(ValueError, match="chunk_id"):
        IndexingPipeline._assign_vector_ids(df, None)


def test_assign_vector_ids_rejects_duplicate_old_chunk_ids() -> None:
    df = pd.DataFrame({"chunk_id": ["new"]})
    old_df = pd.DataFrame({"chunk_id": ["same", "same"], "vector_id": [0, 1]})

    with pytest.raises(ValueError, match="previous metadata"):
        IndexingPipeline._assign_vector_ids(df, old_df)


def test_assign_vector_ids_rejects_duplicate_old_vector_ids() -> None:
    df = pd.DataFrame({"chunk_id": ["new"]})
    old_df = pd.DataFrame({"chunk_id": ["a", "b"], "vector_id": [0, 0]})

    with pytest.raises(ValueError, match="vector_id"):
        IndexingPipeline._assign_vector_ids(df, old_df)


def test_pipeline_produces_faiss_index(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    pipeline.run(FIXTURE_REPO)
    assert s.faiss_index_path.exists()


def test_pipeline_produces_manifest(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    pipeline.run(FIXTURE_REPO)
    assert s.faiss_index_path.with_suffix(".json").exists()


def test_pipeline_returns_matching_df_and_vectors(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    emb = _mock_embedder(s)
    pipeline = IndexingPipeline(s, embedder=emb)
    df, vectors = pipeline.run(FIXTURE_REPO)

    assert len(df) == vectors.shape[0]
    assert vectors.shape[1] == MOCK_DIM
    assert vectors.dtype == np.float32


def test_pipeline_vectors_are_unit_length(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    _, vectors = pipeline.run(FIXTURE_REPO)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-4)


def test_pipeline_index_loadable(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    df, vectors = pipeline.run(FIXTURE_REPO)

    store = VectorStore(s)
    store.load(expected_dim=MOCK_DIM)
    assert store.ntotal == len(df)


def test_pipeline_nn_is_self_after_full_run(tmp_path: Path) -> None:
    """End-to-end: the nearest neighbour of any chunk vector must be itself."""
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    df, vectors = pipeline.run(FIXTURE_REPO)

    store = VectorStore(s)
    store.load()

    misses = 0
    for i in range(len(vectors)):
        results = store.search(vectors[i], k=1)
        if not results or results[0][0] != i:
            misses += 1
    assert misses == 0, f"{misses}/{len(vectors)} vectors did not self-retrieve"


def test_pipeline_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    df, vectors = pipeline.run(repo)
    assert len(df) == 0
    assert vectors.shape[0] == 0
    # All artifacts are still written
    assert s.metadata_path.exists()
    assert s.faiss_index_path.exists()
    assert s.faiss_index_path.with_suffix(".json").exists()


def test_pipeline_rejects_missing_repo_path(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    with pytest.raises(FileNotFoundError, match="Repository path does not exist"):
        pipeline.run(tmp_path / "missing")


def test_pipeline_rejects_file_repo_path(tmp_path: Path) -> None:
    repo_file = tmp_path / "not-a-repo.py"
    repo_file.write_text("def f(): pass\n")
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    with pytest.raises(NotADirectoryError, match="not a directory"):
        pipeline.run(repo_file)


def test_pipeline_unchanged_reindex_embeds_zero_chunks(tmp_path: Path) -> None:
    repo = _copy_fixture_repo(tmp_path)
    s = _settings(tmp_path)
    model = CountingSentenceTransformer()
    pipeline = IndexingPipeline(s, embedder=Embedder(s, _model=model))

    df, _ = pipeline.run(repo)
    first_count = model.encoded_chunks
    assert first_count == len(df)

    df2, _ = pipeline.run(repo)

    assert len(df2) == len(df)
    assert model.encoded_chunks == first_count
    assert pipeline.last_stats["chunks_embedded"] == 0
    assert pipeline.last_stats["cache_hits"] == len(df2)


def test_pipeline_single_function_edit_reembeds_only_changed_chunk(tmp_path: Path) -> None:
    repo = _copy_fixture_repo(tmp_path)
    s = _settings(tmp_path)
    model = CountingSentenceTransformer()
    pipeline = IndexingPipeline(s, embedder=Embedder(s, _model=model))

    pipeline.run(repo)
    first_count = model.encoded_chunks
    auth_path = repo / "auth.py"
    auth_path.write_text(
        auth_path.read_text(newline="").replace(
            "return len(segments) == 3",
            "return len(segments) == 3 and segments[0] != ''",
        ),
        newline="",
    )

    pipeline.run(repo)

    assert model.encoded_chunks == first_count + 1
    assert pipeline.last_stats["updated"] == 1
    assert pipeline.last_stats["chunks_embedded"] == 1


def test_pipeline_deleted_file_removed_from_search_results(tmp_path: Path) -> None:
    from semcode.search import Searcher

    repo = _copy_fixture_repo(tmp_path)
    s = _settings(tmp_path)
    embedder = Embedder(s, _model=MockSentenceTransformer())
    pipeline = IndexingPipeline(s, embedder=embedder)
    pipeline.run(repo)

    before = Searcher(s, embedder=embedder).search(
        "format date ISO string", k=10, use_reranker=False
    )
    assert any(result.file_path == "utils.js" for result in before)

    (repo / "utils.js").unlink()
    pipeline.run(repo)

    after = Searcher(s, embedder=embedder).search(
        "format date ISO string", k=10, use_reranker=False
    )
    assert all(result.file_path != "utils.js" for result in after)
    assert pipeline.last_stats["removed"] > 0


def test_pipeline_rebuilds_when_faiss_artifact_missing(tmp_path: Path) -> None:
    repo = _copy_fixture_repo(tmp_path)
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    pipeline.run(repo)

    s.faiss_index_path.unlink()
    df, vectors = pipeline.run(repo)

    assert s.faiss_index_path.exists()
    assert pipeline.last_stats["full_rebuild"] is True
    assert len(df) == vectors.shape[0]


def test_pipeline_rebuilds_when_manifest_artifact_missing(tmp_path: Path) -> None:
    repo = _copy_fixture_repo(tmp_path)
    s = _settings(tmp_path)
    pipeline = IndexingPipeline(s, embedder=_mock_embedder(s))
    pipeline.run(repo)

    s.faiss_index_path.with_suffix(".json").unlink()
    df, vectors = pipeline.run(repo)

    assert s.faiss_index_path.with_suffix(".json").exists()
    assert pipeline.last_stats["full_rebuild"] is True
    assert len(df) == vectors.shape[0]
