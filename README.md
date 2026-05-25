# Semantic Code Search Engine

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/CI-pending-lightgrey)
![Status](https://img.shields.io/badge/status-in%20development-orange)

> **Search your codebase by what you mean, not what you typed.**

> ⚠️ **This project is currently in active development.** 

---

## The Problem

Every developer knows the frustration: you vaguely remember a function that "validates a JWT
and extracts the user role", but `grep -r "jwt"` returns 400 matches and `ctrl+shift+F` in the
IDE gives you token noise. You end up manually skimming files for ten minutes to find twenty
lines of code.

This is the fundamental limitation of **lexical search** — it matches characters, not concepts.
It has no understanding of what code *does*, only what it literally *says*. If the function you
are looking for is called `_check_bearer`, a keyword search for "JWT" will not find it.

**semcode** addresses this with semantic search. You describe your *intent* in plain English and
the engine finds the code that satisfies it, regardless of the specific identifiers used. Ask
*"where does the app check user permissions before writing to the database?"* and the engine
returns the relevant code chunks ranked by meaning, not by string overlap.

The target audience is developers working on large, unfamiliar codebases — onboarding to a new
team, doing security reviews, or navigating a legacy monorepo where the naming conventions were
invented by someone who left three years ago.

---

## Planned Architecture

The system is divided into two pipelines that share the same embedding model.

```
                      ┌──────────────────────────────┐
   target repo  ──────▶  ingest (tree-sitter AST)     │
                      │  → pandas metadata (parquet)  │
                      └───────────────┬──────────────┘
                                      │ code chunks
                      ┌───────────────▼──────────────┐
                      │  embed (PyTorch + HuggingFace) │
                      │  code-aware transformer        │
                      └───────┬───────────────────────┘
                              │ dense vectors
               ┌──────────────▼──────────┐  ┌─────────────┐
               │  FAISS (dense index)     │  │  BM25       │
               │  IndexFlatIP / IVF       │  │  (sparse)   │
               └──────────────┬──────────┘  └──────┬──────┘
                              │                     │
   query  ────────────────────┼─────────────────────┤
   (natural language)         │   Reciprocal         │
                              └──── Rank Fusion ─────┘
                                         │ fused candidates
                      ┌──────────────────▼──────────────┐
                      │  re-rank (TensorFlow/Keras MLP)  │
                      │  trained on retrieval features   │
                      └──────────────────┬──────────────┘
                                         │ ranked results
                      ┌──────────────────▼──────────────┐
                      │  FastAPI  (/index /search        │
                      │  /health /stats /metrics)        │
                      └─────────────────────────────────┘
```

### Component Overview

| Stage | Module | What it will do |
|---|---|---|
| **Ingest** | `semcode.ingest` | Walk a repo respecting `.gitignore`, parse source files with tree-sitter, extract function/class/method chunks with metadata (file, line span, symbol name, docstring), fall back to sliding-window chunking for unparseable files. Persist a pandas DataFrame to parquet. |
| **Embed** | `semcode.embed` | Load a code-aware transformer (e.g. CodeBERT) via PyTorch + sentence-transformers. Batch-encode chunks to L2-normalised dense vectors. Singleton model load; tqdm progress for large corpora. |
| **Index** | `semcode.index` | Build a FAISS `IndexFlatIP` (or IVF above a size threshold) from the embeddings. Persist index + metadata + a manifest JSON (model name, dimension, chunk count, built-at). Refuse to load if model/dimension changed. Also build a BM25 index from identifier-aware tokenised chunk text. |
| **Search** | `semcode.search` | Encode the query with the same embedder. Retrieve top-K from FAISS and top-K from BM25 independently, then fuse the two ranked lists with Reciprocal Rank Fusion weighted by configurable `dense_weight` / `bm25_weight`. Return `SearchResult` pydantic objects carrying per-source scores. |
| **Rerank** | `semcode.rerank` | A small TensorFlow/Keras MLP that re-scores fused candidates from engineered features (dense score, BM25 score, symbol-name token overlap, length ratio, language one-hot, docstring match). Has its own training loop and is saved in SavedModel format. Falls back gracefully to the fused score when no model is present. |
| **API** | `semcode.api` | FastAPI app with CORS, request-id middleware, and global JSON error handlers. Loads the searcher once at startup. Exposes all planned endpoints. Runs indexing as a FastAPI `BackgroundTask`. |

---

## Why PyTorch AND TensorFlow?

In most production systems you standardise on one ML framework. This project deliberately uses
both, but gives each a **distinct, defensible role** rather than duplicating work.

### PyTorch — bi-encoder embeddings

PyTorch handles the heavy lifting in `semcode.embed` and all query encoding. The reason is
straightforward: the best pretrained code models — CodeBERT, GraphCodeBERT, StarEncoder — are
PyTorch-native and distributed through Hugging Face Transformers. The `sentence-transformers`
library, which provides clean pooling, normalisation, and batching on top of those models, is
also PyTorch-native.

The bi-encoder architecture is the right choice here because it encodes each chunk *once* at
index time and stores the resulting vector. At query time, only the query needs encoding — the
stored vectors are fetched directly. This makes FAISS lookup sub-millisecond regardless of index
size, which is essential for an interactive developer tool.

PyTorch's dynamic computation graph also makes it straightforward to fine-tune the embedding
model on a domain-specific code corpus in a later milestone without rewriting infrastructure.

### TensorFlow — learned re-ranker

TensorFlow handles `semcode.rerank`. The re-ranker is a different kind of ML component: it does
not produce embeddings. Instead, it receives a set of already-retrieved candidate chunks and
re-scores each one using *engineered features* — the FAISS similarity score, the BM25 score, the
Reciprocal Rank Fusion score, exact symbol-name token overlap, query-to-code length ratio,
language, and whether query tokens appear in the docstring.

This is a small Keras MLP trained on (query, candidate, relevant: bool) triplets. It has its own
dataset construction script, its own training loop with early stopping and AUC/accuracy logging,
and is saved in TF SavedModel format so it can be served independently via TF Serving if needed.

Using TensorFlow here is a deliberate architectural boundary: the re-ranker is a separable,
trainable component with a clean feature interface. It can be hot-swapped, retrained on new
labelled data, or replaced with a different TF model without touching the PyTorch embedding
pipeline.

**pandas** sits between both stages — it manages chunk metadata, drives the feature engineering
for the re-ranker, and stores the training dataset.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Web framework | FastAPI + Uvicorn | 0.115 / 0.32 |
| Settings | Pydantic v2 + pydantic-settings | 2.10 / 2.6 |
| CLI | Typer | planned (M1) |
| AST parsing | tree-sitter + tree-sitter-languages | 0.23 / 1.10 |
| Embeddings | PyTorch + Transformers + sentence-transformers | 2.5 / 4.47 / 3.3 |
| Dense index | FAISS (CPU) | 1.9 |
| Sparse index | BM25 (rank-bm25) | 0.2 |
| Re-ranking | TensorFlow / Keras | 2.18 |
| Data wrangling | pandas + pyarrow | 2.2 / 18.1 |
| Observability | structlog + Prometheus client | 24.4 / planned M12 |
| Linting | Ruff + Black | 0.8 / 24.10 |
| Type checking | mypy (strict) | 1.13 |
| Testing | pytest + pytest-asyncio + httpx | 8.3 / 0.24 / 0.28 |

---

## Quickstart

> These instructions describe the intended workflow. The API is not yet implemented — only
> the package stubs exist. Instructions will be updated as milestones complete.

### Local (venv)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/semcode.git
cd semcode

# 2. Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set your preferred model, data paths, etc.

# 5. Index a repository (planned — M4)
python -m semcode index /path/to/your/repo

# 6. Search (planned — M5)
python -m semcode search "validate JWT and return user claims"

# 7. Start the API server (planned — M8)
make run
# → http://localhost:8000/docs
```

### Docker (planned — M11)

```bash
# Build and start
docker compose up --build

# The API will be available at http://localhost:8000
# The ./data directory is volume-mounted for index persistence
```

---

## Planned API Endpoints

These endpoints are designed and documented here; they will be implemented in M8.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + readiness. Reports whether an index is loaded and which model is active. |
| `POST` | `/index` | Index a repository. Body: `{"repo_path": "/path/to/repo", "rebuild": false}`. Runs as a background task; returns a job ID. |
| `GET` | `/index/status/{job_id}` | Poll indexing job status and progress. |
| `GET` | `/search` | Search by natural-language query. Params: `q`, `k` (default 10), `use_reranker` (default true). |
| `GET` | `/stats` | Index statistics: chunk count, language breakdown, model name, index built-at timestamp. |
| `GET` | `/metrics` | Prometheus metrics: request counts, latency histograms, index size. (M12) |
| `GET` | `/version` | App version, model name, and index manifest. (M12) |
| `DELETE` | `/index` | Clear all persisted index artifacts. |

### Example search response (planned)

```bash
curl "http://localhost:8000/search?q=validate+JWT+and+extract+user+role&k=3"
```

```json
{
  "query": "validate JWT and extract user role",
  "results": [
    {
      "file": "auth/middleware.py",
      "symbol_name": "_check_bearer",
      "symbol_type": "function",
      "language": "python",
      "start_line": 42,
      "end_line": 68,
      "score": 0.94,
      "dense_score": 0.91,
      "bm25_score": 0.73,
      "snippet": "def _check_bearer(token: str) -> UserClaims:\n    ..."
    }
  ],
  "total_candidates": 120,
  "latency_ms": 38
}
```

---

## Development Roadmap

The project is built in thirteen milestones. Each milestone is a self-contained unit that
can be completed in a single session and leaves the repo in a working, tested state.

- [x] **M0** — Project scaffold: package structure, pyproject.toml, requirements, Makefile, README *(current — done)*
- [ ] **M1** — Configuration, logging & CLI skeleton: pydantic-settings `Settings`, structlog JSON/pretty logging, Typer CLI with `index`, `search`, `serve`, `train-reranker` stub subcommands
- [ ] **M2** — Code ingestion & AST chunking: `semcode.ingest` repo walker with `.gitignore` support, tree-sitter parsing for Python/JS/TS/Go/Java, function/class/method chunk extraction, sliding-window fallback, parquet output
- [ ] **M3** — Embedding pipeline: `semcode.embed` with PyTorch + sentence-transformers, batched L2-normalised encoding, singleton model load, `embed_dataframe()` helper
- [ ] **M4** — Vector store & full indexing pipeline: `semcode.index` FAISS build/save/load with manifest validation, `IndexingPipeline` orchestrating ingest → embed → index, end-to-end `python -m semcode index <repo>` command
- [ ] **M5** — Dense semantic search: `semcode.search` query encoding, FAISS ANN lookup, metadata join, `SearchResult` pydantic model, terminal result renderer, `search` CLI subcommand
- [ ] **M6** — Hybrid retrieval (BM25 + RRF): BM25 index over identifier-aware tokenised chunks, Reciprocal Rank Fusion with configurable weights, per-source scores on every result
- [ ] **M7** — TensorFlow learned re-ranker: `semcode.rerank` feature engineering with pandas, `scripts/build_reranker_dataset.py`, Keras MLP training loop (AUC/accuracy logging, early stopping), SavedModel persistence, optional inference stage in `Searcher`
- [ ] **M8** — FastAPI server: `semcode.api` with lifespan model loading, CORS, request-id middleware, background indexing tasks, all planned endpoints, `serve` CLI subcommand, async integration tests
- [ ] **M9** — Caching & incremental indexing: content-hash embedding cache, diff-based incremental index update (add/remove/change chunks), `rebuild` flag on CLI and API, timing logs
- [ ] **M10** — Testing, coverage & CI: comprehensive unit + integration tests, coverage threshold (≥75% on fast suite), GitHub Actions workflow (lint → typecheck → test), `CONTRIBUTING.md`, updated README badges
- [ ] **M11** — Dockerization: multi-stage Dockerfile (non-root user, slim runtime), `.dockerignore`, `docker-compose.yml` with optional Qdrant profile, `docker-build` / `docker-up` / `docker-down` Makefile targets
- [ ] **M12** — Observability & production hardening: Prometheus `/metrics`, `/version` endpoint, liveness vs readiness distinction, input validation limits, rate limiting (slowapi), graceful startup with missing-artifact reporting, `ARCHITECTURE.md`, `examples/` directory with sample queries

**Release plan:** tag `v0.1.0` after M8 (first working server) and `v1.0.0` after M12 (production-ready).

---

## Stretch Goals (post-M12)

- **Evaluation harness** — build a labelled query set and report Recall@k / MRR / nDCG before vs after the TF re-ranker, turning "it feels better" into a measurable result.
- **VS Code extension** or a small React UI hitting the FastAPI backend.
- **Multi-repo / workspace indexing** with per-repo filters on search.
- **Fine-tuning pipeline** — contrastive training on CodeSearchNet to adapt the bi-encoder to a specific codebase.

---

## Limitations

These limitations apply to the planned design and will inform implementation decisions:

- **CPU-only by default.** FAISS and model inference will run on CPU. GPU support requires manual
  CUDA setup and swapping `faiss-cpu` for `faiss-gpu`.
- **No incremental indexing until M9.** Re-indexing a large repository from scratch can take
  several minutes depending on corpus size and hardware.
- **Single-repo indexing until post-M12.** The initial design indexes one repository at a time.
- **English queries only.** The bi-encoder models targeted are trained on English natural language;
  non-English queries will produce degraded results.
- **Large model downloads.** First run will download approximately 500 MB of transformer weights.
  Ensure network access and sufficient disk space before indexing.
- **TensorFlow install size.** TensorFlow adds ~600 MB to the environment. A future iteration may
  evaluate ONNX Runtime as a lighter alternative for the re-ranker inference path.
- **Re-ranker requires labelled data.** The TensorFlow MLP needs (query → relevant chunk) label
  pairs to train on. The fixture label set shipped with the repo is small; quality on a new
  codebase will depend on the effort put into labelling.
