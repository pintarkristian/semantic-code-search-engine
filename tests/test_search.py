"""Tests for semcode.search — Searcher and formatting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semcode.config import Settings
from semcode.embed import Embedder, chunk_to_text
from semcode.index import VectorStore
from semcode.search import SearchResult, Searcher, _make_snippet, format_results
from tests.conftest import MOCK_DIM, MockSentenceTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _settings(tmp_path: Path, model_name: str = "test-model") -> Settings:
    return Settings(
        embedding_model_name=model_name,
        data_dir=tmp_path,
        faiss_index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.parquet",
        batch_size=8,
        top_k_return=5,
    )


def _mock_embedder(settings: Settings) -> Embedder:
    return Embedder(settings, _model=MockSentenceTransformer())


def _build_index(tmp_path: Path, n: int = 12) -> tuple[Settings, Embedder, pd.DataFrame]:
    """Create a synthetic index in tmp_path and return (settings, embedder, df)."""
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
            "code": f"def func_{i}(x):\n    \"\"\"Does thing {i}.\"\"\"\n    return x + {i}",
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
            rank=1, score=1, file_path="f.py", symbol_name="s",
            symbol_type="t", language="python", start_line=1, end_line=2,
            snippet="",
        )
        assert isinstance(r.score, float)


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

    def test_empty_code(self) -> None:
        assert _make_snippet("", max_lines=6) == ""

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

    def test_score_range(self, tmp_path: Path) -> None:
        settings, embedder, _ = _build_index(tmp_path)
        searcher = Searcher(settings, embedder=embedder)
        results = searcher.search("function", k=5)
        for r in results:
            assert -1.01 <= r.score <= 1.01  # cosine similarity bounds


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
