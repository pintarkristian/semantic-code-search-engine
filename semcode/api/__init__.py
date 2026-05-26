"""FastAPI application for semcode."""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from semcode.config import Settings, get_settings
from semcode.embed import embedding_cache_path
from semcode.index import IndexingPipeline
from semcode.logging import get_logger
from semcode.search import SearchResult, Searcher
from semcode.search._bm25 import bm25_corpus_path

log = get_logger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class HealthResponse(BaseModel):
    """Service health and active model/index state."""

    status: str = Field(..., examples=["ok"])
    index_loaded: bool
    model_name: str


class IndexRequest(BaseModel):
    """Request body for indexing a repository."""

    repo_path: Path = Field(..., description="Path to the repository to index.")
    rebuild: bool = Field(False, description="Force a full rebuild of persisted artifacts.")


class IndexJobResponse(BaseModel):
    """Created background indexing job."""

    job_id: str
    status: JobStatus


class IndexStatusResponse(BaseModel):
    """Current indexing job state."""

    job_id: str
    status: JobStatus
    repo_path: str
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    chunks: int | None = None
    dimension: int | None = None


class SearchResponse(BaseModel):
    """Ranked search results plus query latency."""

    query: str
    latency_ms: float
    results: list[SearchResult]


class StatsResponse(BaseModel):
    """Index statistics derived from persisted metadata and manifest."""

    index_loaded: bool
    chunk_count: int
    language_breakdown: dict[str, int]
    model_name: str
    index_info: dict


class DeleteIndexResponse(BaseModel):
    """Artifacts removed by DELETE /index."""

    deleted: list[str]
    index_loaded: bool


def create_app(
    settings: Settings | None = None,
    *,
    searcher: Searcher | None = None,
    autoload_index: bool = True,
) -> FastAPI:
    """Create a configured FastAPI application."""
    app_settings = settings or get_settings()
    jobs: dict[str, dict] = {}
    jobs_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        app.state.jobs = jobs
        app.state.jobs_lock = jobs_lock
        app.state.searcher = searcher or Searcher(app_settings)
        app.state.index_loaded = False
        if autoload_index and _index_artifacts_exist(app_settings):
            app.state.index_loaded = _load_searcher(app)
        yield

    app = FastAPI(
        title="semcode API",
        version="0.1.0",
        description="Semantic code search over indexed repositories.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _install_request_id_middleware(app)
    _install_exception_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Return basic service health and index state."""
        return HealthResponse(
            status="ok",
            index_loaded=bool(app.state.index_loaded),
            model_name=app_settings.embedding_model_name,
        )

    @app.post("/index", response_model=IndexJobResponse, status_code=202, tags=["index"])
    async def index_repo(
        payload: IndexRequest,
        background_tasks: BackgroundTasks,
    ) -> IndexJobResponse:
        """Start repository indexing in a FastAPI background task."""
        if not payload.repo_path.exists() or not payload.repo_path.is_dir():
            raise HTTPException(status_code=422, detail="repo_path must be an existing directory")

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "repo_path": str(payload.repo_path),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "chunks": None,
            "dimension": None,
        }
        with jobs_lock:
            jobs[job_id] = job

        background_tasks.add_task(_run_index_job, app, job_id, payload.repo_path, payload.rebuild)
        return IndexJobResponse(job_id=job_id, status="queued")

    @app.get("/index/status/{job_id}", response_model=IndexStatusResponse, tags=["index"])
    async def index_status(job_id: str) -> IndexStatusResponse:
        """Return the current state of a background indexing job."""
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="index job not found")
            return IndexStatusResponse(**job)

    @app.get("/search", response_model=SearchResponse, tags=["search"])
    async def search(
        q: str = Query(..., min_length=1, description="Natural-language code search query."),
        k: int = Query(10, ge=1, le=100, description="Number of ranked results to return."),
        use_reranker: bool = Query(False, description="Apply the optional learned reranker."),
    ) -> SearchResponse:
        """Search the loaded index and return ranked code chunks."""
        start = time.perf_counter()
        try:
            results = app.state.searcher.search(q, k=k, use_reranker=use_reranker)
            app.state.index_loaded = True
        except FileNotFoundError as exc:
            app.state.index_loaded = False
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return SearchResponse(
            query=q,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            results=results,
        )

    @app.get("/stats", response_model=StatsResponse, tags=["index"])
    async def stats() -> StatsResponse:
        """Return persisted index statistics."""
        chunk_count = 0
        language_breakdown: dict[str, int] = {}
        if app_settings.metadata_path.exists():
            df = pd.read_parquet(app_settings.metadata_path)
            chunk_count = int(len(df))
            if "language" in df.columns:
                language_breakdown = {
                    str(lang): int(count)
                    for lang, count in df["language"].value_counts().sort_index().items()
                }

        return StatsResponse(
            index_loaded=bool(app.state.index_loaded),
            chunk_count=chunk_count,
            language_breakdown=language_breakdown,
            model_name=app_settings.embedding_model_name,
            index_info=_read_index_manifest(app_settings),
        )

    @app.delete("/index", response_model=DeleteIndexResponse, tags=["index"])
    async def delete_index() -> DeleteIndexResponse:
        """Delete persisted index artifacts and reset the app-scoped searcher."""
        deleted: list[str] = []
        for path in _artifact_paths(app_settings):
            if path.is_dir():
                shutil.rmtree(path)
                deleted.append(str(path))
            elif path.exists():
                path.unlink()
                deleted.append(str(path))

        app.state.searcher = Searcher(app_settings)
        app.state.index_loaded = False
        return DeleteIndexResponse(deleted=deleted, index_loaded=False)

    return app


def _install_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _json_error(request, 422, "validation_error", exc.errors())

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _json_error(request, exc.status_code, str(exc.detail), None)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled API error", error=str(exc))
        return _json_error(request, 500, "internal_server_error", None)


def _json_error(
    request: Request,
    status_code: int,
    message: str,
    details: object | None,
) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "details": details,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    return JSONResponse(status_code=status_code, content=body)


def _run_index_job(app: FastAPI, job_id: str, repo_path: Path, rebuild: bool) -> None:
    settings: Settings = app.state.settings
    jobs: dict[str, dict] = app.state.jobs
    jobs_lock: Lock = app.state.jobs_lock

    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.time()

    try:
        pipeline = IndexingPipeline(settings)
        df, vectors = pipeline.run(repo_path, rebuild=rebuild)
        app.state.searcher = Searcher(settings)
        app.state.index_loaded = _load_searcher(app)
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "succeeded",
                    "finished_at": time.time(),
                    "chunks": int(len(df)),
                    "dimension": int(vectors.shape[1]) if vectors.ndim == 2 and len(df) else 0,
                }
            )
    except Exception as exc:  # pragma: no cover - exercised through integration behavior
        app.state.index_loaded = False
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "finished_at": time.time(),
                    "error": str(exc),
                }
            )
        log.exception("index job failed", job_id=job_id, error=str(exc))


def _load_searcher(app: FastAPI) -> bool:
    try:
        app.state.searcher._ensure_loaded()
        return True
    except FileNotFoundError:
        return False


def _index_artifacts_exist(settings: Settings) -> bool:
    return settings.metadata_path.exists() and settings.faiss_index_path.exists()


def _read_index_manifest(settings: Settings) -> dict:
    manifest_path = settings.faiss_index_path.with_suffix(".json")
    if not manifest_path.exists():
        return {}
    import json

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _artifact_paths(settings: Settings) -> list[Path]:
    return [
        settings.metadata_path,
        settings.faiss_index_path,
        settings.faiss_index_path.with_suffix(".json"),
        bm25_corpus_path(settings.faiss_index_path),
        embedding_cache_path(settings),
    ]


def _delete_artifacts(settings: Settings) -> None:
    for path in _artifact_paths(settings):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


app = create_app()
