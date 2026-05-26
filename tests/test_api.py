"""Tests for the FastAPI application."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from semcode.api import create_app
from semcode.config import Settings
from semcode.embed import Embedder
from semcode.index import IndexingPipeline
from semcode.search import Searcher
from tests.conftest import MockSentenceTransformer

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_model_name="test-model",
        data_dir=tmp_path,
        faiss_index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.parquet",
        reranker_model_path=tmp_path / "reranker",
        batch_size=8,
        top_k_retrieve=12,
        top_k_return=5,
    )


@asynccontextmanager
async def _client(app) -> AsyncIterator[httpx.AsyncClient]:  # type: ignore[no-untyped-def]
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def _preindexed_app(tmp_path: Path):
    settings = _settings(tmp_path)
    embedder = Embedder(settings, _model=MockSentenceTransformer())
    IndexingPipeline(settings, embedder=embedder).run(FIXTURE_REPO)
    searcher = Searcher(settings, embedder=embedder)
    return create_app(settings, searcher=searcher), settings


@pytest.mark.asyncio
async def test_health_without_index(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with _client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is False
    assert body["model_name"] == "test-model"
    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_health_with_index(tmp_path: Path) -> None:
    app, _ = _preindexed_app(tmp_path)
    async with _client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["index_loaded"] is True


@pytest.mark.asyncio
async def test_search_returns_ranked_results_against_fixture(tmp_path: Path) -> None:
    app, _ = _preindexed_app(tmp_path)
    async with _client(app) as client:
        response = await client.get("/search", params={"q": "validate JWT token", "k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "validate JWT token"
    assert body["latency_ms"] >= 0.0
    assert 1 <= len(body["results"]) <= 5
    assert [result["rank"] for result in body["results"]] == list(range(1, len(body["results"]) + 1))
    assert any("validate" in result["symbol_name"].lower() for result in body["results"])


@pytest.mark.asyncio
async def test_bad_search_input_returns_422(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with _client(app) as client:
        response = await client.get("/search", params={"q": "", "k": 0})

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "validation_error"


@pytest.mark.asyncio
async def test_delete_index_clears_artifacts(tmp_path: Path) -> None:
    app, settings = _preindexed_app(tmp_path)
    async with _client(app) as client:
        response = await client.delete("/index")
        health = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["index_loaded"] is False
    assert health.json()["index_loaded"] is False
    assert not settings.metadata_path.exists()
    assert not settings.faiss_index_path.exists()
    assert not settings.faiss_index_path.with_suffix(".json").exists()
