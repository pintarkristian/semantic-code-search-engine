"""Code ingestion pipeline: repo walking, AST chunking, parquet persistence.

Public surface:
    CodeIngestor   — main class; call .ingest() to get a DataFrame
    ingest()       — convenience function wrapping CodeIngestor
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pathspec

from semcode.config import Settings, get_settings
from semcode.embed import content_hash_for_row
from semcode.ingest._ast import EXT_TO_LANG, AstChunk, extract_chunks
from semcode.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        "vendor",
        ".idea",
        ".vscode",
        ".tox",
        "eggs",
        ".eggs",
        "htmlcov",
        "site-packages",
    }
)

_PARQUET_COLUMNS = [
    "chunk_id",
    "file_path",
    "language",
    "symbol_name",
    "symbol_type",
    "start_line",
    "end_line",
    "code",
    "docstring",
    "content_hash",
]

# Sliding-window defaults (lines); override via subclass if needed
_WINDOW_LINES: int = 50
_WINDOW_STRIDE: int = 25


# ---------------------------------------------------------------------------
# CodeIngestor
# ---------------------------------------------------------------------------


class CodeIngestor:
    """Walk a repository and return a pandas DataFrame of code chunks.

    Each row is one chunk (function, method, class, or sliding-window fallback)
    with a stable chunk_id hash so downstream pipelines can detect unchanged chunks.
    """

    def __init__(
        self,
        repo_path: Path,
        settings: Settings | None = None,
        *,
        window_lines: int = _WINDOW_LINES,
        window_stride: int = _WINDOW_STRIDE,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.settings = settings or get_settings()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_path}")
        if window_lines <= 0:
            raise ValueError("window_lines must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        self.window_lines = window_lines
        self.window_stride = window_stride
        self._gitignore = self._load_gitignore()

    # ------------------------------------------------------------------
    # .gitignore support
    # ------------------------------------------------------------------

    def _load_gitignore(self) -> pathspec.PathSpec | None:
        gi_path = self.repo_path / ".gitignore"
        if gi_path.exists():
            try:
                lines = gi_path.read_text(encoding="utf-8", errors="replace").splitlines()
                return pathspec.PathSpec.from_lines("gitignore", lines)
            except OSError:
                pass
        return None

    def _is_gitignored(self, rel: Path) -> bool:
        if self._gitignore is None:
            return False
        return bool(self._gitignore.match_file(str(rel).replace("\\", "/")))

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    def _iter_source_files(self) -> Iterator[Path]:
        for path in sorted(self.repo_path.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.repo_path)
            # Skip any path whose components include a vendored/build dir
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if self._is_gitignored(rel):
                continue
            if path.suffix in EXT_TO_LANG:
                yield path

    # ------------------------------------------------------------------
    # Chunk extraction
    # ------------------------------------------------------------------

    def _make_chunk_id(
        self, rel_path: str, symbol_name: str, start_line: int, end_line: int
    ) -> str:
        key = f"{rel_path}:{symbol_name}:{start_line}:{end_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _chunk_file(self, file_path: Path) -> list[dict]:
        rel = file_path.relative_to(self.repo_path)
        rel_str = str(rel).replace("\\", "/")
        language = EXT_TO_LANG[file_path.suffix]

        try:
            source_bytes = file_path.read_bytes()
        except OSError as exc:
            log.warning("cannot read file", path=str(file_path), error=str(exc))
            return []

        if not source_bytes.strip():
            return []  # empty or whitespace-only file

        ast_chunks = extract_chunks(language, source_bytes)

        if ast_chunks:
            return [self._record(rel_str, language, c) for c in ast_chunks]
        # Fallback: sliding window over raw lines
        return self._sliding_window(source_bytes, rel_str, language)

    def _record(self, rel_str: str, language: str, chunk: AstChunk) -> dict:
        code = chunk.text.decode("utf-8", errors="replace")
        record = {
            "chunk_id": self._make_chunk_id(
                rel_str, chunk.symbol_name, chunk.start_line, chunk.end_line
            ),
            "file_path": rel_str,
            "language": language,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "code": code,
            "docstring": chunk.docstring,
        }
        record["content_hash"] = content_hash_for_row(
            record,
            max_chars=self.settings.max_chunk_tokens * 4,
        )
        return record

    def _sliding_window(self, source_bytes: bytes, rel_str: str, language: str) -> list[dict]:
        lines = source_bytes.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return []
        records: list[dict] = []
        start = 0
        while start < len(lines):
            end = min(start + self.window_lines, len(lines))
            start_line = start + 1  # 1-indexed
            end_line = end
            symbol_name = f"lines_{start_line}_{end_line}"
            record = {
                "chunk_id": self._make_chunk_id(rel_str, symbol_name, start_line, end_line),
                "file_path": rel_str,
                "language": language,
                "symbol_name": symbol_name,
                "symbol_type": "chunk",
                "start_line": start_line,
                "end_line": end_line,
                "code": "\n".join(lines[start:end]),
                "docstring": "",
            }
            record["content_hash"] = content_hash_for_row(
                record,
                max_chars=self.settings.max_chunk_tokens * 4,
            )
            records.append(record)
            if end == len(lines):
                break
            start += self.window_stride
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, *, persist: bool = True) -> pd.DataFrame:
        """Walk the repo, extract chunks, optionally persist to parquet, return DataFrame."""
        records: list[dict] = []
        files = list(self._iter_source_files())
        log.info("ingesting repo", repo=str(self.repo_path), file_count=len(files))

        for file_path in files:
            chunks = self._chunk_file(file_path)
            log.debug("chunked file", path=str(file_path), chunks=len(chunks))
            records.extend(chunks)

        df = pd.DataFrame(records, columns=_PARQUET_COLUMNS)

        if persist:
            out = self.settings.metadata_path
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            log.info("wrote metadata", path=str(out), rows=len(df))

        return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def ingest(repo_path: Path, settings: Settings | None = None) -> pd.DataFrame:
    """Ingest a repository and return a chunk DataFrame."""
    return CodeIngestor(repo_path, settings).ingest()
