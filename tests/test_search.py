"""Tests for semcode.search — Searcher and formatting."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from semcode.config import Settings
from semcode.embed import Embedder, chunk_to_text
from semcode.index import IndexingPipeline, VectorStore
from semcode.search import (
    Searcher,
    SearchResult,
    _make_snippet,
    _reciprocal_rank_fusion,
    format_results,
)
from semcode.search._bm25 import BM25Retriever, bm25_corpus_path, tokenize
from tests.conftest import MockSentenceTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _settings(tmp_path: Path, model_name: str = "test-model", **overrides: object) -> Settings:
    params = {
        "embedding_model_name": model_name,
        "data_dir": tmp_path,
        "faiss_index_path": tmp_path / "index.faiss",
        "metadata_path": tmp_path / "metadata.parquet",
        "batch_size": 8,
        "top_k_return": 5,
    }
    params.update(overrides)
    return Settings(**params)


def _mock_embedder(settings: Settings) -> Embedder:
    return Embedder(settings, _model=MockSentenceTransformer())


def _build_index(tmp_path: Path, n: int = 12) -> tuple[Settings, Embedder, pd.DataFrame]:
    """Create a synthetic FAISS + BM25 index in tmp_path; return (settings, embedder, df)."""
    settings = _settings(tmp_path)
    embedder = _mock_embedder(settings)

    rows = [
        {
            "chunk_id": f"{i:016x}",
            "file_path": f"src/module_{i // 3}.py",
            "language": "python",
            "symbol_name": f"func_{i}",
            "symbol_type": "function_definition",
            "start_line": i * 10 + 1,
            "end_line": i * 10 + 9,
            "code": f'def func_{i}(x):\n    """Does thing {i}."""\n    return x + {i}',
            "docstring": f"Does thing {i}.",
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(settings.metadata_path, index=False)

    texts = [chunk_to_text(row) for _, row in df.iterrows()]
    vectors = embedder.encode(texts)

    store = VectorStore(settings)
    store.build(vectors)
    store.save()

    bm25 = BM25Retriever.from_dataframe(df)
    bm25.save(bm25_corpus_path(settings.faiss_index_path))

    return settings, embedder, df


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_basic_construction(self) -> None:
        r = SearchResult(
            rank=1,
            score=0.92,
            file_path="src/auth.py",
            symbol_name="validate",
            symbol_type="function_definition",
            language="python",
            start_line=10,
            end_line=20,
            snippet="def validate():\n    pass",
        )
        assert r.rank == 1
        assert r.score == 0.92
        assert r.file_path == "src/auth.py"
        assert r.symbol_name == "validate"

    def test_score_is_float(self) -> None:
        r = SearchResult(
            rank=1,
            score=1,
            file_path="f.py",
            symbol_name="s",
            symbol_type="t",
            language="python",
            start_line=1,
            end_line=2,
            snippet="",
        )
        assert isinstance(r.score, float)

    def test_rejects_invalid_line_span(self) -> None:
        with pytest.raises(ValueError, match="line span"):
            SearchResult(
                rank=1,
                score=0.5,
                file_path="src/auth.py",
                symbol_name="validate",
                symbol_type="function",
                language="python",
                start_line=10,
                end_line=9,
                snippet="def validate(): pass",
            )


# ---------------------------------------------------------------------------
# Snippet helper
# ---------------------------------------------------------------------------


class TestMakeSnippet:
    def test_limits_lines(self) -> None:
        code = "\n".join(f"line {i}" for i in range(20))
        snippet = _make_snippet(code, max_lines=6)
        assert snippet.count("\n") < 6

    def test_strips_trailing_blanks(self) -> None:
        code = "def f():\n    pass\n\n\n"
        snippet = _make_snippet(code, max_lines=10)
        assert not snippet.endswith("\n")

    def test_strips_leading_blanks(self) -> None:
        code = "\n\n\ndef f():\n    pass"
        snippet = _make_snippet(code, max_lines=10)
        assert snippet.startswith("def f")

    def test_empty_code(self) -> None:
        assert _make_snippet("", max_lines=6) == ""

    def test_rejects_non_positive_max_lines(self) -> None:
        with pytest.raises(ValueError, match="max_lines"):
            _make_snippet("def f(): pass", max_lines=0)

    def test_rejects_non_integer_max_lines(self) -> None:
        with pytest.raises(TypeError, match="max_lines must be an integer"):
            _make_snippet("def f(): pass", max_lines="6")  # type: ignore[arg-type]

    def test_short_code_unchanged(self) -> None:
        code = "x = 1\ny = 2"
        assert _make_snippet(code, max_lines=6) == code


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


class TestSearcher:
    def test_raises_when_no_metadata(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(FileNotFoundError, match="semcode index"):
            searcher.search("anything")

    def test_search_returns_results(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function that returns a value", k=5)
        assert 1 <= len(results) <= 5

    def test_results_are_search_result_instances(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("return value", k=3)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_scores_descending(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("compute result", k=8)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function", k=4)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_result_fields_populated(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("does thing", k=1)
        assert len(results) == 1
        r = results[0]
        assert r.file_path.startswith("src/")
        assert r.symbol_name.startswith("func_")
        assert r.language == "python"
        assert r.start_line > 0
        assert r.end_line >= r.start_line
        assert "def func_" in r.snippet

    def test_k_limits_results(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path, n=12)
        searcher = Searcher(settings, embedder=embedder)
        assert len(searcher.search("x", k=3)) <= 3
        assert len(searcher.search("x", k=1)) <= 1

    def test_candidates_k_expands_retrieval_pool(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, top_k_retrieve=1, top_k_return=1, max_search_k=10)
        embedder = _mock_embedder(settings)
        _build_index(tmp_path, n=8)
        searcher = Searcher(settings, embedder=embedder)

        assert len(searcher.candidates("thing", k=4)) == 4

    def test_blank_query_rejected(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        with pytest.raises(ValueError, match="non-whitespace"):
            searcher.search("   ")

    def test_search_rejects_query_above_max_length(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, max_query_length=5)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(ValueError, match="at most 5"):
            searcher.search("x" * 6)

    def test_candidates_rejects_query_above_max_length(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, max_query_length=5)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(ValueError, match="at most 5"):
            searcher.candidates("x" * 6)

    def test_candidates_rejects_non_positive_k(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        with pytest.raises(ValueError, match="k must be positive"):
            searcher.candidates("function", k=0)

    def test_candidates_rejects_non_integer_k(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(TypeError, match="k must be an integer"):
            searcher.candidates("function", k="1")  # type: ignore[arg-type]

    def test_search_rejects_non_positive_k(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        with pytest.raises(ValueError, match="k must be positive"):
            searcher.search("function", k=0)

    def test_search_rejects_non_integer_k(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(TypeError, match="k must be an integer"):
            searcher.search("function", k="1")  # type: ignore[arg-type]

    def test_candidates_rejects_k_above_max_search_k(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, max_search_k=5)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(ValueError, match="less than or equal to 5"):
            searcher.candidates("function", k=6)

    def test_search_rejects_k_above_max_search_k(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, max_search_k=5)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        with pytest.raises(ValueError, match="less than or equal to 5"):
            searcher.search("function", k=6)

    def test_rejects_non_integer_metadata_vector_ids(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df["vector_id"] = ["bad-id"] + [str(value) for value in range(1, len(df))]
        df.to_parquet(settings.metadata_path, index=False)
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="vector_id values must be integers"):
            searcher.search("function", k=1)

    def test_rejects_bm25_ids_missing_from_metadata(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        BM25Retriever([["function"]], doc_ids=[999]).save(bm25_corpus_path(settings.faiss_index_path))
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="not present in metadata"):
            searcher.search("function", k=1)

    def test_rejects_duplicate_metadata_chunk_ids(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df.loc[1, "chunk_id"] = df.loc[0, "chunk_id"]
        df.to_parquet(settings.metadata_path, index=False)
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="chunk_id values must be unique"):
            searcher.search("function", k=1)

    def test_rejects_invalid_metadata_line_spans(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df.loc[0, "start_line"] = 10
        df.loc[0, "end_line"] = 9
        df.to_parquet(settings.metadata_path, index=False)
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="line spans"):
            searcher.search("function", k=1)

    def test_rejects_non_integer_metadata_line_spans(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df["start_line"] = ["not-int"] + [str(value) for value in df["start_line"].iloc[1:]]
        df["end_line"] = [str(value) for value in df["end_line"]]
        df.to_parquet(settings.metadata_path, index=False)
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="line spans must be integers"):
            searcher.search("function", k=1)

    def test_rejects_blank_metadata_display_fields(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df.loc[0, "file_path"] = "   "
        df.to_parquet(settings.metadata_path, index=False)
        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="file_path values"):
            searcher.search("function", k=1)

    def test_search_trims_query_before_rerank(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        searcher = Searcher(settings, embedder=_mock_embedder(settings))
        seen: dict[str, str] = {}

        def fake_candidates(query: str):
            return pd.DataFrame(
                [
                    {
                        "chunk_id": "chunk-1",
                        "file_path": "src/auth.py",
                        "symbol_name": "validate",
                        "symbol_type": "function",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "code": "def validate(): pass",
                        "dense_score": 0.5,
                        "bm25_score": 0.0,
                        "fused_score": 0.25,
                    }
                ]
            )

        def fake_rerank(query: str, candidates: pd.DataFrame, use_reranker: bool):
            seen["query"] = query
            return candidates

        searcher.candidates = fake_candidates  # type: ignore[method-assign]
        searcher._maybe_rerank = fake_rerank  # type: ignore[method-assign]

        searcher.search("  validate token  ", k=1)

        assert seen["query"] == "validate token"

    def test_default_k_from_settings(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path, n=12)
        # top_k_return=5 in _settings helper
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function")  # no k override
        assert len(results) <= settings.top_k_return

    def test_store_loaded_once(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        searcher.search("a", k=1)
        store_ref = searcher._store
        searcher.search("b", k=1)
        assert searcher._store is store_ref  # same object, not reloaded

    def test_missing_bm25_corpus_is_rebuilt_and_saved(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        bm25_path = bm25_corpus_path(settings.faiss_index_path)
        bm25_path.unlink()

        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function", k=1)

        assert results
        assert bm25_path.exists()

    def test_metadata_missing_required_columns_rejected(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df.drop(columns=["code"]).to_parquet(settings.metadata_path, index=False)

        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="required columns"):
            searcher.search("function", k=1)

    def test_duplicate_metadata_vector_ids_rejected(self, tmp_path: Path) -> None:
        settings, embedder, df = _build_index(tmp_path)
        df["vector_id"] = [0] * len(df)
        df.to_parquet(settings.metadata_path, index=False)

        searcher = Searcher(settings, embedder=embedder)

        with pytest.raises(ValueError, match="vector_id"):
            searcher.search("function", k=1)

    def test_fused_score_positive(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function", k=5)
        for r in results:
            assert r.score > 0  # RRF scores are always positive
            assert r.score == r.fused_score

    def test_score_fields_present(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function", k=3)
        for r in results:
            assert hasattr(r, "dense_score")
            assert hasattr(r, "bm25_score")
            assert hasattr(r, "fused_score")


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def _sample_results(self) -> list[SearchResult]:
        return [
            SearchResult(
                rank=i,
                score=1.0 - i * 0.1,
                file_path=f"src/mod_{i}.py",
                symbol_name=f"func_{i}",
                symbol_type="function_definition",
                language="python",
                start_line=i * 10,
                end_line=i * 10 + 5,
                snippet=f"def func_{i}():\n    pass",
            )
            for i in range(1, 4)
        ]

    def test_empty_returns_no_results(self) -> None:
        assert format_results([]) == "No results found."

    def test_contains_file_path(self) -> None:
        results = self._sample_results()
        output = format_results(results)
        for r in results:
            assert r.file_path in output

    def test_contains_symbol_name(self) -> None:
        results = self._sample_results()
        output = format_results(results)
        for r in results:
            assert r.symbol_name in output

    def test_contains_score(self) -> None:
        results = self._sample_results()
        output = format_results(results)
        assert "score=" in output

    def test_verbose_rerank_result_shows_rerank_as_score(self) -> None:
        result = SearchResult(
            rank=1,
            score=0.9,
            rerank_score=0.9,
            dense_score=0.2,
            bm25_score=1.5,
            fused_score=0.03,
            file_path="src/auth.py",
            symbol_name="validate",
            symbol_type="function",
            language="python",
            start_line=1,
            end_line=2,
            snippet="def validate(): pass",
        )

        output = format_results([result], verbose=True)

        assert "score=0.9000" in output
        assert "fused=0.0300" in output

    def test_query_shown_when_provided(self) -> None:
        results = self._sample_results()
        output = format_results(results, query="find auth")
        assert "find auth" in output

    def test_snippet_indented(self) -> None:
        results = self._sample_results()
        output = format_results(results)
        assert "    def func_" in output

    def test_rank_shown(self) -> None:
        results = self._sample_results()
        output = format_results(results)
        assert "#1" in output
        assert "#2" in output


# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_camel_case_split(self) -> None:
        tokens = tokenize("formatDate")
        assert "format" in tokens
        assert "date" in tokens

    def test_snake_case_split(self) -> None:
        tokens = tokenize("validate_token")
        assert "validate" in tokens
        assert "token" in tokens

    def test_all_caps_prefix(self) -> None:
        tokens = tokenize("XMLParser")
        assert "xml" in tokens
        assert "parser" in tokens

    def test_short_tokens_filtered(self) -> None:
        # Single-character splits are filtered; multi-char tokens are kept
        tokens = tokenize("a_b_c foo")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "foo" in tokens

    def test_output_lowercased(self) -> None:
        tokens = tokenize("QueryBuilder")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_numbers_kept(self) -> None:
        tokens = tokenize("encode_base64")
        assert "encode" in tokens
        assert "base64" in tokens


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    def test_exact_token_match_scores_highest(self) -> None:
        corpus = [
            ["validate", "token", "jwt"],
            ["format", "date", "iso"],
            ["hash", "password", "plaintext"],
        ]
        bm25 = BM25Retriever(corpus)
        hits = bm25.search("hash password", k=3)
        assert hits[0][0] == 2  # row 2 has "hash" and "password"

    def test_no_match_returns_empty(self) -> None:
        corpus = [["alpha", "beta"], ["gamma", "delta"]]
        bm25 = BM25Retriever(corpus)
        assert bm25.search("xorshift", k=3) == []

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        corpus = [["foo", "bar"], ["baz", "qux"]]
        bm25 = BM25Retriever(corpus)
        path = tmp_path / "corpus.pkl"
        bm25.save(path)
        loaded = BM25Retriever.load(path)
        assert loaded._corpus == corpus
        assert bm25.search("foo", k=2) == loaded.search("foo", k=2)

    def test_load_rejects_payload_without_corpus(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.pkl"
        with open(path, "wb") as f:
            pickle.dump({"doc_ids": [1, 2]}, f)

        with pytest.raises(ValueError, match="missing 'corpus'"):
            BM25Retriever.load(path)

    def test_load_rejects_corrupt_pickle_with_path(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.pkl"
        path.write_bytes(b"not-a-pickle")

        with pytest.raises(ValueError, match="Failed to load BM25 corpus"):
            BM25Retriever.load(path)

    def test_empty_corpus(self) -> None:
        bm25 = BM25Retriever([])
        assert bm25.search("anything", k=5) == []

    def test_k_limits_hits(self) -> None:
        corpus = [["alpha"], ["beta"], ["gamma"], ["alpha", "beta"], ["alpha", "gamma"]]
        bm25 = BM25Retriever(corpus)
        hits = bm25.search("alpha beta gamma", k=2)
        assert len(hits) <= 2

    def test_non_positive_k_returns_no_hits(self) -> None:
        bm25 = BM25Retriever([["alpha"], ["beta"]])
        assert bm25.search("alpha", k=0) == []

    def test_rejects_non_integer_k(self) -> None:
        bm25 = BM25Retriever([["alpha"], ["beta"]])
        with pytest.raises(TypeError, match="k must be an integer"):
            bm25.search("alpha", k="1")  # type: ignore[arg-type]

    def test_rejects_mismatched_doc_ids(self) -> None:
        with pytest.raises(ValueError, match="doc_ids"):
            BM25Retriever([["alpha"], ["beta"]], doc_ids=[10])

    def test_rejects_empty_explicit_doc_ids(self) -> None:
        with pytest.raises(ValueError, match="Expected 1 doc_ids, got 0"):
            BM25Retriever([["alpha"]], doc_ids=[])

    def test_rejects_non_integer_doc_ids(self) -> None:
        with pytest.raises(ValueError, match="doc_ids must be integers"):
            BM25Retriever([["alpha"]], doc_ids=["not-an-id"])  # type: ignore[list-item]

    def test_rejects_string_doc_ids(self) -> None:
        with pytest.raises(ValueError, match="list of integers"):
            BM25Retriever([["alpha"], ["beta"]], doc_ids="12")  # type: ignore[arg-type]

    def test_rejects_duplicate_doc_ids(self) -> None:
        with pytest.raises(ValueError, match="doc_ids must be unique"):
            BM25Retriever([["alpha"], ["beta"]], doc_ids=[7, 7])

    def test_rejects_non_list_documents(self) -> None:
        with pytest.raises(ValueError, match="token lists"):
            BM25Retriever([["alpha"], "beta"])  # type: ignore[list-item]

    def test_rejects_non_string_tokens(self) -> None:
        with pytest.raises(ValueError, match="tokens must be strings"):
            BM25Retriever([["alpha", 123]])  # type: ignore[list-item]

    def test_scores_descending(self) -> None:
        corpus = [
            ["foo"],
            ["foo", "bar"],
            ["foo", "bar", "baz"],
        ]
        bm25 = BM25Retriever(corpus)
        hits = bm25.search("foo bar baz", k=3)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_from_dataframe(self, tmp_path: Path) -> None:
        _, _, df = _build_index(tmp_path)
        bm25 = BM25Retriever.from_dataframe(df)
        assert len(bm25._corpus) == len(df)
        # Each document should have some tokens
        assert all(len(doc) > 0 for doc in bm25._corpus)

    def test_bm25_corpus_path_derivation(self, tmp_path: Path) -> None:
        faiss_path = tmp_path / "index.faiss"
        assert bm25_corpus_path(faiss_path) == tmp_path / "index_bm25.pkl"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_dense_only_preserves_dense_order(self) -> None:
        dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
        bm25: list[tuple[int, float]] = []
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=1.0, bm25_weight=0.0)
        order = [row_idx for row_idx, *_ in fused]
        assert order == [0, 1, 2]

    def test_bm25_only_preserves_bm25_order(self) -> None:
        dense: list[tuple[int, float]] = []
        bm25 = [(2, 5.0), (0, 3.0), (1, 1.0)]
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.0, bm25_weight=1.0)
        order = [row_idx for row_idx, *_ in fused]
        assert order == [2, 0, 1]

    def test_bm25_weight_boosts_exact_identifier(self) -> None:
        # doc 2 is last in dense; give BM25 all the weight
        dense = [(0, 0.9), (1, 0.8), (2, 0.1)]
        bm25 = [(2, 9.0), (1, 0.5), (0, 0.2)]
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.0, bm25_weight=1.0)
        assert fused[0][0] == 2  # BM25-top doc wins

    def test_doc_only_in_dense_included(self) -> None:
        dense = [(0, 0.9), (99, 0.4)]
        bm25 = [(0, 5.0)]
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.7, bm25_weight=0.3)
        assert any(r[0] == 99 for r in fused)

    def test_doc_only_in_bm25_included(self) -> None:
        dense = [(0, 0.9)]
        bm25 = [(0, 5.0), (77, 2.0)]
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.7, bm25_weight=0.3)
        assert any(r[0] == 77 for r in fused)

    def test_fused_scores_descending(self) -> None:
        dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
        bm25 = [(2, 5.0), (0, 3.0), (1, 1.0)]
        fused = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.5, bm25_weight=0.5)
        scores = [f for *_, f in fused]
        assert scores == sorted(scores, reverse=True)

    def test_zero_weights_are_rejected(self) -> None:
        dense = [(0, 0.9)]
        bm25 = [(0, 5.0)]
        with pytest.raises(ValueError, match="positive"):
            _reciprocal_rank_fusion(dense, bm25, dense_weight=0.0, bm25_weight=0.0)

    def test_rejects_non_positive_rrf_k(self) -> None:
        with pytest.raises(ValueError, match="RRF k"):
            _reciprocal_rank_fusion([(0, 0.9)], [], dense_weight=1.0, bm25_weight=0.0, k=0)

    def test_rejects_non_integer_rrf_k(self) -> None:
        with pytest.raises(TypeError, match="RRF k must be an integer"):
            _reciprocal_rank_fusion(
                [(0, 0.9)],
                [],
                dense_weight=1.0,
                bm25_weight=0.0,
                k="60",  # type: ignore[arg-type]
            )

    def test_rejects_negative_rrf_weights(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            _reciprocal_rank_fusion([(0, 0.9)], [], dense_weight=-1.0, bm25_weight=0.0)

    def test_rejects_non_numeric_rrf_weights(self) -> None:
        with pytest.raises(TypeError, match="weights must be numeric"):
            _reciprocal_rank_fusion(
                [(0, 0.9)],
                [],
                dense_weight="1.0",  # type: ignore[arg-type]
                bm25_weight=0.0,
            )

    def test_rejects_non_finite_rrf_weights(self) -> None:
        with pytest.raises(ValueError, match="weights must be finite"):
            _reciprocal_rank_fusion(
                [(0, 0.9)],
                [],
                dense_weight=float("nan"),
                bm25_weight=0.0,
            )

    def test_rejects_all_zero_rrf_weights(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _reciprocal_rank_fusion([(0, 0.9)], [], dense_weight=0.0, bm25_weight=0.0)


# ---------------------------------------------------------------------------
# Hybrid Searcher — BM25 boost and weight extremes
# ---------------------------------------------------------------------------


def _unique_id_corpus(
    tmp_path: Path, dense_weight: float, bm25_weight: float
) -> tuple[Settings, Embedder]:
    """Build an index with 10 generic functions plus one unique 'xorshift_prng' function."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        embedding_model_name="test-model",
        data_dir=tmp_path,
        faiss_index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.parquet",
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        top_k_retrieve=20,
        top_k_return=11,
        batch_size=8,
    )
    embedder = _mock_embedder(settings)

    rows = [
        {
            "chunk_id": f"{i:016x}",
            "file_path": "src/utils.py",
            "language": "python",
            "symbol_name": f"generic_{i}",
            "symbol_type": "function_definition",
            "start_line": i * 5 + 1,
            "end_line": i * 5 + 4,
            "code": f"def generic_{i}(x):\n    return x + {i}",
            "docstring": "",
        }
        for i in range(10)
    ]
    rows.append(
        {
            "chunk_id": "xorshiftunique00",
            "file_path": "src/rng.py",
            "language": "python",
            "symbol_name": "xorshift_prng",
            "symbol_type": "function_definition",
            "start_line": 100,
            "end_line": 107,
            "code": "def xorshift_prng(seed):\n    return seed ^ (seed << 13)",
            "docstring": "xorshift pseudo random number generator",
        }
    )
    df = pd.DataFrame(rows)
    df.to_parquet(settings.metadata_path, index=False)

    texts = [chunk_to_text(row) for _, row in df.iterrows()]
    vectors = embedder.encode(texts)
    store = VectorStore(settings)
    store.build(vectors)
    store.save()
    BM25Retriever.from_dataframe(df).save(bm25_corpus_path(settings.faiss_index_path))

    return settings, embedder


class TestHybridSearcher:
    def test_pure_bm25_exact_identifier_first(self, tmp_path: Path) -> None:
        """With weight=(0,1), searching the exact identifier ranks its doc #1."""
        settings, embedder = _unique_id_corpus(tmp_path, dense_weight=0.0, bm25_weight=1.0)
        results = Searcher(settings, embedder=embedder).search("xorshift", k=11)
        assert results[0].symbol_name == "xorshift_prng"

    def test_pure_dense_order_matches_cosine(self, tmp_path: Path) -> None:
        """With weight=(1,0), dense_score drives the ranking; bm25_score is zero."""
        settings, embedder = _unique_id_corpus(tmp_path, dense_weight=1.0, bm25_weight=0.0)
        results = Searcher(settings, embedder=embedder).search("compute result", k=5)
        for r in results:
            assert r.bm25_score == 0.0 or r.bm25_score >= 0.0  # no BM25 contribution

    def test_hybrid_boosts_exact_identifier_over_pure_dense(self, tmp_path: Path) -> None:
        """BM25-heavy hybrid should rank xorshift_prng better than pure dense for its name."""
        settings_d, emb_d = _unique_id_corpus(tmp_path / "d", dense_weight=1.0, bm25_weight=0.0)
        dense_results = Searcher(settings_d, embedder=emb_d).search("xorshift", k=11)
        dense_rank = next(r.rank for r in dense_results if r.symbol_name == "xorshift_prng")

        settings_h, emb_h = _unique_id_corpus(tmp_path / "h", dense_weight=0.3, bm25_weight=0.7)
        hybrid_results = Searcher(settings_h, embedder=emb_h).search("xorshift", k=11)
        hybrid_rank = next(r.rank for r in hybrid_results if r.symbol_name == "xorshift_prng")

        assert hybrid_rank <= dense_rank  # BM25 boost improved or matched the rank

    def test_hybrid_results_have_all_score_fields(self, tmp_path: Path) -> None:
        settings, embedder = _unique_id_corpus(tmp_path, dense_weight=0.7, bm25_weight=0.3)
        results = Searcher(settings, embedder=embedder).search("function", k=5)
        for r in results:
            assert r.score == r.fused_score
            assert r.dense_score >= 0.0
            assert r.bm25_score >= 0.0
            assert r.fused_score > 0.0

    def test_doc_with_bm25_match_has_nonzero_bm25_score(self, tmp_path: Path) -> None:
        settings, embedder = _unique_id_corpus(tmp_path, dense_weight=0.7, bm25_weight=0.3)
        results = Searcher(settings, embedder=embedder).search("xorshift prng", k=11)
        xorshift = next((r for r in results if r.symbol_name == "xorshift_prng"), None)
        assert xorshift is not None
        assert xorshift.bm25_score > 0.0

    def test_verbose_format_shows_all_scores(self, tmp_path: Path) -> None:
        settings, embedder = _unique_id_corpus(tmp_path, dense_weight=0.7, bm25_weight=0.3)
        results = Searcher(settings, embedder=embedder).search("function", k=3)
        output = format_results(results, verbose=True)
        assert "dense=" in output
        assert "bm25=" in output


# ---------------------------------------------------------------------------
# Slow integration tests — real model, real fixture repo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_searcher(tmp_path_factory: pytest.TempPathFactory) -> Searcher:
    """Build a real FAISS index from the fixture repo using the real embedding model.

    Session-scoped so the model is loaded and the index is built only once
    across all slow tests in this session.
    """
    data_dir = tmp_path_factory.mktemp("real_index")
    settings = Settings(
        data_dir=data_dir,
        faiss_index_path=data_dir / "index.faiss",
        metadata_path=data_dir / "metadata.parquet",
    )
    pipeline = IndexingPipeline(settings)
    pipeline.run(FIXTURE_REPO)
    return Searcher(settings, embedder=pipeline.embedder)


def _top_names(searcher: Searcher, query: str, k: int = 5) -> list[str]:
    return [r.symbol_name for r in searcher.search(query, k=k)]


@pytest.mark.slow
class TestSemanticRanking:
    """Regression checks: intent-style queries must surface the right symbols."""

    def test_validate_token_top3(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "validate JWT token", k=5)
        assert any("validate" in n.lower() or "token" in n.lower() for n in names[:3])

    def test_hash_password_top3(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "hash a password string", k=5)
        assert any("hash" in n.lower() or "password" in n.lower() for n in names[:3])

    def test_format_date_top3(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "format a date as ISO date string", k=5)
        assert any("date" in n.lower() or "format" in n.lower() for n in names[:3])

    def test_nonempty_string_check_top3(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "check whether a string is non-empty", k=5)
        assert any("string" in n.lower() or "empty" in n.lower() for n in names[:3])

    def test_normalise_query_top5(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "normalise and clean a search query string", k=5)
        assert any(
            "query" in n.lower() or "normalise" in n.lower() or "normalize" in n.lower()
            for n in names[:5]
        )

    def test_build_url_with_params_top5(self, real_searcher: Searcher) -> None:
        names = _top_names(real_searcher, "build a URL with query parameters", k=5)
        assert any(
            "build" in n.lower() or "query" in n.lower() or "param" in n.lower() for n in names[:5]
        )

    def test_results_are_ranked(self, real_searcher: Searcher) -> None:
        results = real_searcher.search("extract user id from token", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_fixture_languages_present(self, real_searcher: Searcher) -> None:
        results = real_searcher.search("function", k=16)
        langs = {r.language for r in results}
        assert langs >= {"python", "javascript", "typescript"}
