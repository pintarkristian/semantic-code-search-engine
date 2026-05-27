"""Shared pytest configuration, helpers, and fixtures."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Slow-test gate
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests that download and use real ML models.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="pass --slow to run model-download tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Shared mock sentence-transformer model
# ---------------------------------------------------------------------------

MOCK_DIM = 64


class MockSentenceTransformer:
    """Fake SentenceTransformer: deterministic, content-addressed, no network calls.

    The same text always produces the same vector regardless of batch context,
    mirroring real model behaviour and making batching-consistency tests meaningful.
    """

    def get_sentence_embedding_dimension(self) -> int:
        return MOCK_DIM

    def encode(
        self,
        inputs: list[str] | str,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> np.ndarray:
        texts = inputs if isinstance(inputs, list) else [inputs]
        vecs = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)
        return vecs.astype(np.float32)

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(MOCK_DIM).astype(np.float32)
