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
    rebuild: bool = typer.Option(
        False,
        "--rebuild/--no-rebuild",
        help="Force a full rebuild instead of the default incremental update.",
    ),
) -> None:
    """Walk a repository, parse source files, embed chunks, and build the search index."""
    _setup()
    log = get_logger(__name__)
    if not repo_path.exists():
        typer.echo(f"[semcode] error: {repo_path} does not exist", err=True)
        raise typer.Exit(1)
    log.info("starting pipeline", repo_path=str(repo_path), rebuild=rebuild)
    from semcode.index import IndexingPipeline

    s = get_settings()
    pipeline = IndexingPipeline(s)
    df, vectors = pipeline.run(repo_path, rebuild=rebuild)
    stats = pipeline.last_stats
    typer.echo(
        f"[semcode] ingested and indexed {len(df)} chunks  "
        f"dim={vectors.shape[1] if len(df) else 0}  "
        f"embedded={stats.get('chunks_embedded', 0)}  "
        f"cache_hits={stats.get('cache_hits', 0)}  "
        f"embedding_ms={stats.get('embedding_elapsed_ms', 0)}  "
        f"elapsed_ms={stats.get('elapsed_ms', 0)}  "
        f"-> {s.faiss_index_path}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    k: int = typer.Option(10, "--k", "-k", help="Number of results to return."),
    use_reranker: bool = typer.Option(True, "--reranker/--no-reranker", help="Apply TF re-ranker."),
) -> None:
    """Search the current index using a natural-language query."""
    _setup()
    log = get_logger(__name__)
    log.info("search called", query=query, k=k, use_reranker=use_reranker)
    from semcode.search import Searcher, format_results

    s = get_settings()
    searcher = Searcher(s)
    try:
        results = searcher.search(query, k=k, use_reranker=use_reranker)
    except FileNotFoundError as exc:
        typer.echo(f"[semcode] error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(format_results(results, query=query))


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
    log.info("starting API server", host=effective_host, port=effective_port)
    typer.echo(f"[semcode] serving API at http://{effective_host}:{effective_port}")
    import uvicorn

    uvicorn.run("semcode.api:create_app", host=effective_host, port=effective_port, factory=True)


@app.command(name="train-reranker")
def train_reranker(
    labels_path: Path = typer.Argument(..., help="Path to labels JSON file."),
    dataset_path: Path = typer.Option(
        Path("data/reranker_dataset.parquet"),
        "--dataset",
        help="Where to write/read the generated feature parquet.",
    ),
    epochs: int = typer.Option(20, "--epochs", "-e", help="Training epochs."),
    negatives_per_query: int = typer.Option(
        8,
        "--negatives-per-query",
        help="Number of fused non-relevant candidates to keep per query.",
    ),
) -> None:
    """Build training data and train the TensorFlow re-ranker model."""
    _setup()
    log = get_logger(__name__)
    if not labels_path.exists():
        typer.echo(f"[semcode] error: {labels_path} does not exist", err=True)
        raise typer.Exit(1)

    from semcode.rerank import build_reranker_dataset, load_labels, train_reranker_model

    labels = load_labels(labels_path)
    if not labels:
        typer.echo(
            "[semcode] train-reranker skipped: no labels supplied; not yet implemented for empty labels."
        )
        return

    s = get_settings()
    try:
        dataset = build_reranker_dataset(
            labels_path,
            dataset_path,
            s,
            negatives_per_query=negatives_per_query,
        )
        metrics = train_reranker_model(dataset_path, s, epochs=epochs)
    except Exception as exc:
        log.error("train-reranker failed", error=str(exc))
        typer.echo(f"[semcode] error: {exc}", err=True)
        raise typer.Exit(1)

    latest = {name: values[-1] for name, values in metrics.items() if values}
    typer.echo(
        f"[semcode] trained reranker rows={len(dataset)} "
        f"model={s.reranker_model_path} metrics={latest}"
    )


if __name__ == "__main__":
    app()
