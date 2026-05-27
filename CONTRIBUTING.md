# Contributing

Thanks for helping improve semcode. Keep changes small, tested, and focused on one behavior at a
time.

## Local Setup

Use Python 3.11 for parity with CI.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Fast Checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m black --check .
python -m mypy semcode
python -m pytest -q
```

The default pytest run is the fast suite. Slow tests that download or run real embedding models are
marked with `@pytest.mark.slow` and are skipped unless explicitly enabled:

```bash
python -m pytest --slow
```

CI sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`; fast tests must use mocks or fixtures
rather than downloading model weights.

## Coverage

The fast suite must keep coverage at or above the threshold configured in `pyproject.toml`.
Add focused tests for new behavior, especially around ingest, embedding, indexing, search, reranking,
and API paths.

## Pull Requests

- Include a short description of the behavior change.
- Mention any test gaps or intentionally skipped slow checks.
- Do not commit generated index artifacts, model weights, local caches, or virtual environments.
