"""Unit tests for the semcode CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from semcode.__main__ import app
from semcode.config import get_settings

runner = CliRunner()


class _FakePipeline:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.last_stats = {
            "chunks_embedded": 0,
            "cache_hits": 0,
            "embedding_elapsed_ms": 0.0,
            "elapsed_ms": 0.0,
        }

    def run(self, repo_path: Path, *, rebuild: bool = False):
        import numpy as np
        import pandas as pd

        return pd.DataFrame(), np.zeros((0, 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "serve", "train-reranker"):
        assert cmd in result.output


def test_index_runs_ingestor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("semcode.index.IndexingPipeline", _FakePipeline)
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 0
    assert "ingested" in result.output


def test_index_rebuild_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("semcode.index.IndexingPipeline", _FakePipeline)
    result = runner.invoke(app, ["index", str(tmp_path), "--rebuild"])
    assert result.exit_code == 0
    assert "ingested" in result.output


def test_search_help_shows_options() -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "--k" in result.output
    assert "--reranker" in result.output


def test_search_k_option_is_recognized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # --k must not produce "No such option" regardless of index state
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "metadata.parquet"))
    get_settings.cache_clear()
    result = runner.invoke(app, ["search", "find auth handler", "--k", "5"])
    assert "No such option" not in (result.output or "")


def test_search_validation_error_is_user_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSearcher:
        def __init__(self, settings) -> None:
            self.settings = settings

        def search(self, query: str, k: int, use_reranker: bool):
            raise ValueError("query must contain non-whitespace text")

    monkeypatch.setattr("semcode.search.Searcher", _FakeSearcher)
    result = runner.invoke(app, ["search", "   "])
    assert result.exit_code == 1
    assert "[semcode] error: query must contain non-whitespace text" in result.output


def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert "serving API" in result.output
    assert calls
    assert calls[0][0] == ("semcode.api:create_app",)
    assert calls[0][1]["factory"] is True


def test_serve_host_port_override(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert result.exit_code == 0
    assert "127.0.0.1" in result.output
    assert "9000" in result.output
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 9000


def test_train_reranker_stub(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text("{}")
    result = runner.invoke(app, ["train-reranker", str(labels)])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output
