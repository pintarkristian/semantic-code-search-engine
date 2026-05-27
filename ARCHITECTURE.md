# semcode Architecture

semcode is a local semantic code search service built around a simple request lifecycle:
ingest source files, embed chunks, persist retrieval indexes, then serve hybrid search over HTTP.

## Indexing Lifecycle

1. A user starts indexing with `python -m semcode index <repo>` or `POST /index`.
2. `CodeIngestor` walks the repository, respects `.gitignore`, and extracts AST-backed chunks.
3. Each chunk is normalized into embedding text and keyed by a content hash.
4. `IndexingPipeline` loads prior metadata when present and computes added, removed, changed, and
   unchanged chunks.
5. New or changed content hashes are embedded; unchanged hashes reuse vectors from the cache under
   `settings.data_dir`.
6. FAISS stores dense vectors by stable `vector_id`, while BM25 stores tokenized chunk text by the
   same document IDs.
7. Metadata, manifest, FAISS index, BM25 corpus, and embedding cache are persisted under `data/`.

## Search Lifecycle

1. `GET /search?q=...` validates query length and `k` bounds.
2. Middleware assigns a request ID, checks rate limits, enforces request timeout, and records
   request metrics.
3. `Searcher` lazily loads metadata, FAISS, and BM25 if not already loaded.
4. The query is embedded once.
5. Dense FAISS hits and sparse BM25 hits are retrieved independently.
6. Reciprocal Rank Fusion combines the two ranked lists using configured weights.
7. If enabled and available, the TensorFlow reranker scores fused candidates.
8. The API returns ranked `SearchResult` objects and records search latency.

## Runtime Endpoints

- `/health` reports liveness and readiness. The service can be `up` while not `ready` if no index
  artifacts are mounted.
- `/version` reports app version, embedding model, readiness, and the persisted index manifest.
- `/metrics` exposes Prometheus counters, histograms, and index-size gauges.
- `/stats` summarizes persisted metadata.

## Operational Notes

The default vector backend is FAISS. Qdrant is available as an optional Docker Compose profile for
future experiments, but the application path remains FAISS-first so local usage and tests do not
require an external service.

The default fast test suite uses deterministic mock embedders. Slow tests that rely on real model
downloads are marked with `@pytest.mark.slow`.
