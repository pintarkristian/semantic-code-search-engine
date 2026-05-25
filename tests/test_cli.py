"""Unit tests for the semcode CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from semcode.__main__ import app

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "serve", "train-reranker"):
        assert cmd in result.output


def test_index_runs_ingestor(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 0
    assert "ingested" in result.output


def test_index_rebuild_flag(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path), "--rebuild"])
    assert result.exit_code == 0
    assert "ingested" in result.output


def test_search_help_shows_options() -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "--k" in result.output
    assert "--reranker" in result.output


def test_search_k_option_is_recognized() -> None:
    # --k must not produce "No such option" regardless of index state
    result = runner.invoke(app, ["search", "find auth handler", "--k", "5"])
    assert "No such option" not in (result.output or "")


def test_serve_stub() -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_serve_host_port_override() -> None:
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert result.exit_code == 0
    assert "127.0.0.1" in result.output
    assert "9000" in result.output


def test_train_reranker_stub(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text("{}")
    result = runner.invoke(app, ["train-reranker", str(labels)])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output
