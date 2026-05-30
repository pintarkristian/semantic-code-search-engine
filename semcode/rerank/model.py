"""TensorFlow/Keras model training and inference for re-ranking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semcode.config import Settings, get_settings
from semcode.logging import get_logger
from semcode.rerank.features import FEATURE_COLUMNS, build_features

log = get_logger(__name__)


def _import_tf() -> Any:
    import tensorflow as tf  # noqa: PLC0415

    return tf


def _split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [column for column in [*FEATURE_COLUMNS, "label"] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df.empty:
        raise ValueError("Cannot train reranker on an empty dataset.")

    x = df[FEATURE_COLUMNS].astype("float32").to_numpy()
    y = df["label"].astype("float32").to_numpy()

    rng = np.random.default_rng(42)
    pos_idx = rng.permutation(np.flatnonzero(y == 1.0))
    neg_idx = rng.permutation(np.flatnonzero(y == 0.0))
    if len(pos_idx) >= 3 and len(neg_idx) >= 3:
        pos_val = max(1, int(round(len(pos_idx) * 0.2)))
        neg_val = max(1, int(round(len(neg_idx) * 0.2)))
        val_idx = np.concatenate([pos_idx[:pos_val], neg_idx[:neg_val]])
        train_idx = np.concatenate([pos_idx[pos_val:], neg_idx[neg_val:]])
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
    else:
        order = rng.permutation(len(df))
        val_idx = np.array([], dtype=int)
        train_idx = order
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_model(input_dim: int) -> Any:
    """Create the compact Keras MLP used by the reranker."""
    tf = _import_tf()
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Normalization(name="feature_normalization")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="relevance")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def train_reranker_model(
    dataset_path: Path,
    settings: Settings | None = None,
    *,
    epochs: int = 20,
    batch_size: int = 16,
) -> dict[str, list[float]]:
    """Train and export the reranker SavedModel to ``settings.reranker_model_path``."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    settings = settings or get_settings()
    df = pd.read_parquet(dataset_path)
    x_train, y_train, x_val, y_val = _split_xy(df)
    if not np.any(y_train == 1.0) or not np.any(y_train == 0.0):
        raise ValueError("Reranker training needs at least one positive and one negative label.")

    tf = _import_tf()
    tf.keras.utils.set_random_seed(42)
    model = build_model(input_dim=len(FEATURE_COLUMNS))
    norm = model.get_layer("feature_normalization")
    norm.adapt(x_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc" if len(x_val) else "auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]
    validation_data = (x_val, y_val) if len(x_val) else None
    history = model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=_class_weight(y_train),
        verbose=0,
    )

    model_path = settings.reranker_model_path
    if model_path.exists():
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(model_path))
    (model_path / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2),
        encoding="utf-8",
    )

    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    log.info(
        "trained reranker", path=str(model_path), metrics={k: v[-1] for k, v in metrics.items()}
    )
    return metrics


def _class_weight(y: np.ndarray) -> dict[int, float]:
    positives = float(np.sum(y == 1.0))
    negatives = float(np.sum(y == 0.0))
    total = max(positives + negatives, 1.0)
    return {
        0: total / max(2.0 * negatives, 1.0),
        1: total / max(2.0 * positives, 1.0),
    }


class ReRanker:
    """Load a SavedModel and produce relevance probabilities for candidates."""

    def __init__(self, settings: Settings | None = None, *, model_path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_path = model_path or self.settings.reranker_model_path
        self._model: Any | None = None
        self._signature: Any | None = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = (self.model_path / "saved_model.pb").exists()
        return self._available

    def _ensure_loaded(self) -> bool:
        if not self.available:
            return False
        if self._model is None:
            tf = _import_tf()
            self._model = tf.saved_model.load(str(self.model_path))
            self._signature = self._model.signatures["serving_default"]
            log.info("loaded reranker", path=str(self.model_path))
        return True

    def score(self, query: str, candidates: pd.DataFrame) -> np.ndarray:
        """Return a score in [0, 1] for each candidate, or fused scores on fallback."""
        fallback = candidates.get("fused_score", pd.Series([0.0] * len(candidates))).to_numpy(
            dtype="float32"
        )
        if candidates.empty or not self._ensure_loaded():
            return fallback

        try:
            tf = _import_tf()
            features = build_features(query, candidates).to_numpy(dtype="float32")
            if self._signature is None:
                return fallback
            outputs = self._signature(tf.constant(features))
            tensor = next(iter(outputs.values()))
            scores = np.asarray(tensor).reshape(-1).astype("float32")
            return np.clip(scores, 0.0, 1.0)
        except Exception as exc:  # pragma: no cover - defensive production fallback
            log.warning("reranker failed; falling back to fused score", error=str(exc))
            return fallback

    def rerank(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return candidates sorted by learned relevance descending."""
        if candidates.empty:
            return candidates
        ranked = candidates.copy()
        ranked["rerank_score"] = self.score(query, ranked)
        return ranked.sort_values("rerank_score", ascending=False, kind="mergesort").reset_index(
            drop=True
        )
