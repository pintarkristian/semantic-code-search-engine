"""Tests for semcode.ingest — CodeIngestor and AST chunking."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from semcode.config import Settings
from semcode.ingest import CodeIngestor, ingest
from semcode.ingest._ast import EXT_TO_LANG, extract_chunks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _settings(tmp_path: Path) -> Settings:
    """Return a Settings instance with metadata_path inside tmp_path."""
    return Settings(
        metadata_path=tmp_path / "metadata.parquet",
        data_dir=tmp_path,
    )


@pytest.fixture
def ingestor(tmp_path: Path) -> CodeIngestor:
    return CodeIngestor(FIXTURE_REPO, _settings(tmp_path))


@pytest.fixture
def df(ingestor: CodeIngestor) -> pd.DataFrame:
    return ingestor.ingest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_dataframe_columns(df: pd.DataFrame) -> None:
    expected = {
        "chunk_id",
        "file_path",
        "language",
        "symbol_name",
        "symbol_type",
        "start_line",
        "end_line",
        "code",
        "docstring",
    }
    assert expected.issubset(set(df.columns))


def test_dataframe_not_empty(df: pd.DataFrame) -> None:
    assert len(df) > 0


def test_chunk_ids_are_unique(df: pd.DataFrame) -> None:
    assert df["chunk_id"].nunique() == len(df)


def test_chunk_ids_are_hex(df: pd.DataFrame) -> None:
    assert all(len(cid) == 16 for cid in df["chunk_id"])
    assert all(set(cid) <= set("0123456789abcdef") for cid in df["chunk_id"])


def test_chunk_ids_are_stable_across_repo_locations(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    shutil.copytree(FIXTURE_REPO, repo_a)
    shutil.copytree(FIXTURE_REPO, repo_b)

    df_a = CodeIngestor(repo_a, _settings(tmp_path / "a")).ingest(persist=False)
    df_b = CodeIngestor(repo_b, _settings(tmp_path / "b")).ingest(persist=False)

    ids_a = df_a.sort_values(["file_path", "symbol_name", "start_line"])["chunk_id"].tolist()
    ids_b = df_b.sort_values(["file_path", "symbol_name", "start_line"])["chunk_id"].tolist()
    assert ids_a == ids_b


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_python_language_detected(df: pd.DataFrame) -> None:
    assert "python" in df["language"].values


def test_uppercase_source_extension_detected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "APP.PY").write_text("def app():\n    return True\n")

    df = CodeIngestor(repo, _settings(tmp_path)).ingest()

    assert len(df) == 1
    assert df.iloc[0]["language"] == "python"


def test_javascript_language_detected(df: pd.DataFrame) -> None:
    assert "javascript" in df["language"].values


def test_typescript_language_detected(df: pd.DataFrame) -> None:
    assert "typescript" in df["language"].values


def test_file_paths_are_relative(df: pd.DataFrame) -> None:
    for fp in df["file_path"]:
        assert not Path(fp).is_absolute(), f"Expected relative path, got: {fp}"


# ---------------------------------------------------------------------------
# Python symbol extraction
# ---------------------------------------------------------------------------


def test_python_extracts_top_level_functions(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    names = set(py["symbol_name"])
    assert "validate_token" in names
    assert "hash_password" in names


def test_python_extracts_class(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    assert "TokenValidator" in set(py["symbol_name"])


def test_python_extracts_methods(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    names = set(py["symbol_name"])
    assert "validate" in names
    assert "extract_user_id" in names


def test_python_symbol_types(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    class_rows = py[py["symbol_name"] == "TokenValidator"]
    assert (class_rows["symbol_type"] == "class").all()
    fn_rows = py[py["symbol_name"] == "validate_token"]
    assert (fn_rows["symbol_type"] == "function").all()


def test_python_line_spans_are_positive(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    assert (py["start_line"] >= 1).all()
    assert (py["end_line"] >= py["start_line"]).all()


def test_python_line_span_order(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    validate = py[py["symbol_name"] == "validate_token"].iloc[0]
    hash_pw = py[py["symbol_name"] == "hash_password"].iloc[0]
    # validate_token is defined before hash_password in auth.py
    assert validate["start_line"] < hash_pw["start_line"]


def test_python_code_contains_def(df: pd.DataFrame) -> None:
    py = df[(df["language"] == "python") & (df["symbol_type"] == "function")]
    assert all("def " in code for code in py["code"])


def test_python_docstring_extracted(df: pd.DataFrame) -> None:
    py = df[df["language"] == "python"]
    validator_class = py[py["symbol_name"] == "TokenValidator"].iloc[0]
    assert "Validates bearer tokens" in validator_class["docstring"]


# ---------------------------------------------------------------------------
# JavaScript symbol extraction
# ---------------------------------------------------------------------------


def test_js_extracts_functions(df: pd.DataFrame) -> None:
    js = df[df["language"] == "javascript"]
    names = set(js["symbol_name"])
    assert "formatDate" in names
    assert "isNonEmptyString" in names


def test_js_extracts_class(df: pd.DataFrame) -> None:
    js = df[df["language"] == "javascript"]
    assert "QueryBuilder" in set(js["symbol_name"])


def test_js_extracts_methods(df: pd.DataFrame) -> None:
    js = df[df["language"] == "javascript"]
    assert "build" in set(js["symbol_name"])


def test_js_line_spans_valid(df: pd.DataFrame) -> None:
    js = df[df["language"] == "javascript"]
    assert (js["start_line"] >= 1).all()
    assert (js["end_line"] >= js["start_line"]).all()


# ---------------------------------------------------------------------------
# TypeScript symbol extraction
# ---------------------------------------------------------------------------


def test_ts_extracts_class(df: pd.DataFrame) -> None:
    ts = df[df["language"] == "typescript"]
    assert "SearchResultFormatter" in set(ts["symbol_name"])


def test_ts_extracts_function(df: pd.DataFrame) -> None:
    ts = df[df["language"] == "typescript"]
    assert "normaliseQuery" in set(ts["symbol_name"])


def test_ts_extracts_methods(df: pd.DataFrame) -> None:
    ts = df[df["language"] == "typescript"]
    assert "format" in set(ts["symbol_name"])


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------


def test_parquet_written(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    CodeIngestor(FIXTURE_REPO, s).ingest()
    assert s.metadata_path.exists()


def test_parquet_round_trips(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    df_written = CodeIngestor(FIXTURE_REPO, s).ingest()
    df_read = pd.read_parquet(s.metadata_path)
    assert len(df_written) == len(df_read)
    assert set(df_written.columns) == set(df_read.columns)


# ---------------------------------------------------------------------------
# Edge cases — graceful failure
# ---------------------------------------------------------------------------


def test_empty_file_no_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "empty.py").write_bytes(b"")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_whitespace_only_file_no_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "blank.py").write_text("   \n\n   \n")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert len(df) == 0


def test_syntax_error_no_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bad.py").write_bytes(b"def (: pass\n@@@ broken !!!")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    # tree-sitter always produces a partial tree; the sliding-window fallback fires
    assert isinstance(df, pd.DataFrame)


def test_binary_looking_file_no_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "binary.py").write_bytes(bytes(range(256)))
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert isinstance(df, pd.DataFrame)


def test_non_source_files_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.txt").write_text("some notes")
    (repo / "data.csv").write_text("a,b,c")
    (repo / "logic.py").write_text("def foo(): pass\n")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert all(fp.endswith(".py") for fp in df["file_path"])


# ---------------------------------------------------------------------------
# Skip dirs
# ---------------------------------------------------------------------------


def test_node_modules_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nm = repo / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("function vendored() {}")
    (repo / "app.js").write_text("function appFunc() {}")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert not any("node_modules" in fp for fp in df["file_path"])
    assert any("app.js" in fp for fp in df["file_path"])


def test_pycache_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pc = repo / "__pycache__"
    pc.mkdir(parents=True)
    (pc / "cached.py").write_text("def cached(): pass\n")
    (repo / "real.py").write_text("def real(): pass\n")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert not any("__pycache__" in fp for fp in df["file_path"])


def test_venv_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    venv = repo / "venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "helper.py").write_text("def vendored(): pass\n")
    (repo / "main.py").write_text("def main(): pass\n")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert not any("venv" in fp for fp in df["file_path"])


# ---------------------------------------------------------------------------
# .gitignore support
# ---------------------------------------------------------------------------


def test_gitignore_respected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("ignored/\n")
    ignored = repo / "ignored"
    ignored.mkdir()
    (ignored / "secret.py").write_text("def secret(): pass\n")
    (repo / "public.py").write_text("def public_fn(): pass\n")
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    assert not any("secret" in fp for fp in df["file_path"])
    assert any("public.py" in fp for fp in df["file_path"])


def test_unreadable_gitignore_does_not_block_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("ignored.py\n")
    (repo / "public.py").write_text("def public_fn(): pass\n")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == ".gitignore":
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    df = CodeIngestor(repo, _settings(tmp_path)).ingest()

    assert any("public.py" in fp for fp in df["file_path"])


# ---------------------------------------------------------------------------
# Sliding-window fallback
# ---------------------------------------------------------------------------


def test_sliding_window_fires_when_no_ast_chunks(tmp_path: Path) -> None:
    # A Go file with no top-level functions/methods — just package + import
    repo = tmp_path / "repo"
    repo.mkdir()
    content = 'package main\n\nimport "fmt"\n\nvar x = 1\n'
    (repo / "empty_go.go").write_text(content)
    df = CodeIngestor(repo, _settings(tmp_path)).ingest()
    if len(df) > 0:
        assert (df["symbol_type"] == "chunk").all()


def test_sliding_window_symbol_names(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    # Write 120 lines of non-parseable Go to force sliding window
    lines = ["package main"] + [f"// line {i}" for i in range(119)]
    (repo / "comments.go").write_text("\n".join(lines))
    df = CodeIngestor(repo, _settings(tmp_path), window_lines=50, window_stride=25).ingest()
    for _, row in df.iterrows():
        if row["symbol_type"] == "chunk":
            assert row["symbol_name"].startswith("lines_")


def test_ingestor_rejects_non_positive_window_lines(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="window_lines must be positive"):
        CodeIngestor(repo, _settings(tmp_path), window_lines=0)


def test_ingestor_rejects_non_positive_window_stride(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="window_stride must be positive"):
        CodeIngestor(repo, _settings(tmp_path), window_stride=0)


def test_ingestor_rejects_missing_repo_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Repository path does not exist"):
        CodeIngestor(tmp_path / "missing", _settings(tmp_path))


def test_ingestor_rejects_file_repo_path(tmp_path: Path) -> None:
    repo_file = tmp_path / "repo.py"
    repo_file.write_text("def f(): pass\n")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        CodeIngestor(repo_file, _settings(tmp_path))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def test_ingest_function_returns_dataframe(tmp_path: Path) -> None:
    df = ingest(FIXTURE_REPO, _settings(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# AST unit tests (language-level)
# ---------------------------------------------------------------------------


def test_extract_chunks_python_basic() -> None:
    src = b"def foo(x):\n    return x + 1\n"
    chunks = extract_chunks("python", src)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "foo"
    assert chunks[0].symbol_type == "function"
    assert chunks[0].start_line == 1


def test_extract_chunks_python_class_with_methods() -> None:
    src = b"class Bar:\n    def baz(self):\n        pass\n"
    chunks = extract_chunks("python", src)
    names = {c.symbol_name for c in chunks}
    assert "Bar" in names
    assert "baz" in names


def test_extract_chunks_js_function() -> None:
    src = b"function greet(name) { return 'hi ' + name; }\n"
    chunks = extract_chunks("javascript", src)
    assert any(c.symbol_name == "greet" for c in chunks)


def test_extract_chunks_go_function() -> None:
    src = b"package main\nfunc Add(a, b int) int { return a + b }\n"
    chunks = extract_chunks("go", src)
    assert any(c.symbol_name == "Add" for c in chunks)


def test_extract_chunks_java_class() -> None:
    src = b"public class Greeter { public void greet() {} }\n"
    chunks = extract_chunks("java", src)
    names = {c.symbol_name for c in chunks}
    assert "Greeter" in names
    assert "greet" in names


def test_extract_chunks_unknown_language_returns_empty() -> None:
    assert extract_chunks("cobol", b"some code") == []


def test_ext_to_lang_coverage() -> None:
    assert EXT_TO_LANG[".py"] == "python"
    assert EXT_TO_LANG[".ts"] == "typescript"
    assert EXT_TO_LANG[".go"] == "go"
    assert EXT_TO_LANG[".java"] == "java"
