"""Generate a reranker training parquet from labels and an existing semcode index."""

from __future__ import annotations

from pathlib import Path

import typer

from semcode.config import get_settings
from semcode.logging import configure_logging
from semcode.rerank import build_reranker_dataset

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    labels_path: Path = typer.Argument(..., help="JSON mapping query text to relevant chunk_ids."),
    output_path: Path = typer.Option(
        Path("data/reranker_dataset.parquet"),
        "--output",
        "-o",
        help="Parquet path to write.",
    ),
    candidates_per_query: int | None = typer.Option(
        None,
        "--candidates-per-query",
        help="Hybrid candidates to retrieve before labeling. Defaults to settings.top_k_retrieve.",
    ),
    negatives_per_query: int = typer.Option(
        8,
        "--negatives-per-query",
        help="Non-relevant candidates to keep per query.",
    ),
) -> None:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    dataset = build_reranker_dataset(
        labels_path,
        output_path,
        settings,
        candidates_per_query=candidates_per_query,
        negatives_per_query=negatives_per_query,
    )
    typer.echo(f"[semcode] wrote {len(dataset)} reranker rows -> {output_path}")


if __name__ == "__main__":
    app()
