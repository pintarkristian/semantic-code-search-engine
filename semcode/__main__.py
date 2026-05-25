"""Entry point: python -m semcode <subcommand>."""

from __future__ import annotations

from pathlib import Path

import typer

from semcode.config import get_settings
from semcode.logging import configure_logging, get_logger

app = typer.Typer(
    name="semcode",
    help="Semantic code search engine — search your codebase by intent.",
    no_args_is_help=True,
)


def _setup() -> None:
    s = get_settings()
    configure_logging(log_level=s.log_level, log_format=s.log_format)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to the repository to index."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force a full rebuild of the index."),
) -> None:
    """Walk a repository, parse source files, embed chunks, and build the search index."""
    _setup()
    log = get_logger(__name__)
    log.info("index stub called", repo_path=str(repo_path), rebuild=rebuild)
    typer.echo(f"[semcode] indexing {repo_path} (rebuild={rebuild}) — not yet implemented (M2–M4)")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    k: int = typer.Option(10, "--k", "-k", help="Number of results to return."),
    use_reranker: bool = typer.Option(True, "--reranker/--no-reranker", help="Apply TF re-ranker."),
) -> None:
    """Search the current index using a natural-language query."""
    _setup()
    log = get_logger(__name__)
    log.info("search stub called", query=query, k=k, use_reranker=use_reranker)
    typer.echo(f"[semcode] search: '{query}' (k={k}) — not yet implemented (M5)")


@app.command()
def serve(
    host: str = typer.Option(None, "--host", help="Bind host (overrides config)."),
    port: int = typer.Option(None, "--port", "-p", help="Bind port (overrides config)."),
) -> None:
    """Start the FastAPI server."""
    _setup()
    s = get_settings()
    log = get_logger(__name__)
    effective_host = host or s.host
    effective_port = port or s.port
    log.info("serve stub called", host=effective_host, port=effective_port)
    typer.echo(f"[semcode] serve {effective_host}:{effective_port} — not yet implemented (M8)")


@app.command(name="train-reranker")
def train_reranker(
    labels_path: Path = typer.Argument(..., help="Path to labels JSON file."),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Training epochs."),
) -> None:
    """Build training data and train the TensorFlow re-ranker model."""
    _setup()
    log = get_logger(__name__)
    log.info("train-reranker stub called", labels_path=str(labels_path), epochs=epochs)
    typer.echo(f"[semcode] train-reranker epochs={epochs} — not yet implemented (M7)")


if __name__ == "__main__":
    app()
