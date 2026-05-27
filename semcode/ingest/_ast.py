"""Tree-sitter AST parsing helpers — one parser instance per language.

Each public function operates on raw bytes and returns plain data structures
so the CodeIngestor can stay ignorant of tree-sitter internals.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

# Extension → language name
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
}

# language name → (language-capsule factory, chunk node types)
_LANG_CONFIG: dict[str, tuple] = {
    "python": (
        tspython.language,
        frozenset({"function_definition", "class_definition"}),
    ),
    "javascript": (
        tsjs.language,
        frozenset({"function_declaration", "class_declaration", "method_definition"}),
    ),
    "typescript": (
        tsts.language_typescript,
        frozenset({"function_declaration", "class_declaration", "method_definition"}),
    ),
    "go": (
        tsgo.language,
        frozenset({"function_declaration", "method_declaration"}),
    ),
    "java": (
        tsjava.language,
        frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "method_declaration",
                "constructor_declaration",
            }
        ),
    ),
}

# AST node type → canonical symbol type string
_NODE_SYMBOL_TYPE: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "class_definition": "class",
    "class_declaration": "class",
    "interface_declaration": "class",
    "method_definition": "method",
    "method_declaration": "method",
    "constructor_declaration": "method",
}

# Comment-line prefixes per language for leading-comment extraction
_COMMENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "python": ("#",),
    "javascript": ("//", "/*", " *", "*/"),
    "typescript": ("//", "/*", " *", "*/"),
    "go": ("//", "/*"),
    "java": ("//", "/*", " *", "*", "*/"),
}


# ---------------------------------------------------------------------------
# Singleton parsers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=len(_LANG_CONFIG))
def _get_parser(language: str) -> Parser:
    lang_fn, _ = _LANG_CONFIG[language]
    return Parser(Language(lang_fn()))


def _get_chunk_types(language: str) -> frozenset[str]:
    _, types = _LANG_CONFIG[language]
    return types


# ---------------------------------------------------------------------------
# Name / type extraction
# ---------------------------------------------------------------------------


def _symbol_name(node: Node) -> str:
    """Return the identifier name for a chunk node."""
    for child in node.children:
        if child.type in (
            "identifier",
            "field_identifier",
            "property_identifier",
            "type_identifier",
        ):
            if child.text:
                return child.text.decode("utf-8", errors="replace")
    return "<anonymous>"


def _symbol_type(node: Node) -> str:
    return _NODE_SYMBOL_TYPE.get(node.type, "function")


# ---------------------------------------------------------------------------
# Docstring / leading-comment extraction
# ---------------------------------------------------------------------------


def _python_docstring(node: Node) -> str:
    """Return the first string literal inside a Python function/class body."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for expr in stmt.children:
                        if expr.type == "string" and expr.text:
                            return expr.text.decode("utf-8", errors="replace")
            break
    return ""


def _leading_comment(source_lines: list[str], start_line: int, language: str) -> str:
    """Return comment lines immediately above start_line (1-indexed)."""
    prefixes = _COMMENT_PREFIXES.get(language, ())
    collected: list[str] = []
    i = start_line - 2  # 0-indexed line just above the symbol
    while i >= 0:
        stripped = source_lines[i].strip()
        if stripped and any(stripped.startswith(p) for p in prefixes):
            collected.insert(0, source_lines[i].rstrip())
            i -= 1
        else:
            break
    return "\n".join(collected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AstChunk(NamedTuple):
    symbol_name: str
    symbol_type: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    text: bytes
    docstring: str


def extract_chunks(language: str, source_bytes: bytes) -> list[AstChunk]:
    """Parse source_bytes and return AST-derived chunks for the given language.

    Never raises — on parse failure returns an empty list.
    """
    if language not in _LANG_CONFIG:
        return []
    try:
        parser = _get_parser(language)
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    source_lines = source_bytes.decode("utf-8", errors="replace").splitlines()
    chunk_types = _get_chunk_types(language)
    out: list[AstChunk] = []
    _walk(tree.root_node, chunk_types, language, source_lines, out)
    return out


def _walk(
    node: Node,
    chunk_types: frozenset[str],
    language: str,
    source_lines: list[str],
    out: list[AstChunk],
) -> None:
    if node.type in chunk_types and node.text is not None:
        name = _symbol_name(node)
        sym_type = _symbol_type(node)
        start = node.start_point.row + 1  # tree-sitter rows are 0-indexed
        end = node.end_point.row + 1

        if language == "python":
            doc = _python_docstring(node)
        else:
            doc = _leading_comment(source_lines, start, language)

        out.append(
            AstChunk(
                symbol_name=name,
                symbol_type=sym_type,
                start_line=start,
                end_line=end,
                text=node.text,
                docstring=doc,
            )
        )
        # Always recurse so nested symbols (methods inside classes) are captured too.

    for child in node.children:
        _walk(child, chunk_types, language, source_lines, out)
