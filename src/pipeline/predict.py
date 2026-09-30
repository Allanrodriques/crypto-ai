"""Inference: load a trained artifact and score new candles.

This module is the seam the future prediction API will call.  It is written now,
in V1, so that adding a REST endpoint later is a thin wrapper over
:func:`predict_latest` rather than a rewrite of the ML layer.

Contract
--------
* reads a model bundle written by :mod:`src.pipeline.train`
  (``models/<name>.joblib``) which carries the fitted estimator **and** the exact
  feature column order it was trained on
* recomputes features with the same :class:`~src.features.feature_engineering.FeatureEngineer`
  configuration, so a serving path and a training path can never diverge
* returns ``probability_up`` plus a human-readable direction, and never a
  guaranteed-return claim
* performs no I/O to any exchange and places no orders
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.features.feature_engineering import FeatureEngineer
from src.models import MODEL_REGISTRY
from src.utils import format_timestamp, get_logger, set_global_seed, utc_now

logger = get_logger("pipeline.predict")


@dataclass
class Prediction:
    """One scored candle."""

    timestamp: pd.Timestamp
    probability_up: float
    prediction: str  # "UP" | "DOWN"
    confidence: float
    threshold: float

    def describe(self, symbol: str = "") -> str:
        head = f"{symbol} " if symbol else ""
        return (
            f"{head}Prediction: {self.prediction}  "
            f"Probability: {self.probability_up:.2f}  "
            f"(threshold {self.threshold:.2f}, {self.timestamp:%Y-%m-%d %H:%M UTC})"
        )


@dataclass
class ModelBundle:
    """A loaded artifact: estimator, feature order and training provenance."""

    model: Any
    model_name: str
    feature_columns: list[str]
    symbol: str
    interval: str
    horizon_candles: int
    threshold: float
    random_state: int
    trained_at: str
    path: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ModelBundle":
        bundle = joblib.load(path)
        required = {"model", "model_name", "feature_columns", "trained_at"}
        missing = required.difference(bundle)
        if missing:
            raise ValueError(f"Model artifact {path} is missing key(s): {sorted(missing)}")
        return cls(
            model=bundle["model"],
            model_name=bundle["model_name"],
            feature_columns=list(bundle["feature_columns"]),
            symbol=bundle.get("symbol", ""),
            interval=bundle.get("interval", "1h"),
            horizon_candles=int(bundle.get("horizon_candles", 1)),
            threshold=float(bundle.get("threshold", 0.0)),
            random_state=int(bundle.get("random_state", 42)),
            trained_at=bundle["trained_at"],
            path=Path(path),
        )


def load_model(config: Config, name: str | None = None) -> ModelBundle:
    """Load a trained model, defaulting to the selection recorded in metadata."""
    from src.pipeline.train import MODEL_FILENAMES
    from src.utils import load_json

    if name is None:
        metadata_path = config.paths.models_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No model metadata at {metadata_path}. Run scripts/train_model.py first."
            )
        name = load_json(metadata_path)["selected_model"]

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")

    path = config.paths.models_dir / MODEL_FILENAMES.get(name, f"{name}.joblib")
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run scripts/train_model.py first.")
    return ModelBundle.load(path)


def predict_from_klines(
    klines: pd.DataFrame,
    bundle: ModelBundle,
    config: Config,
    *,
    decision_threshold: float | None = None,
) -> pd.DataFrame:
    """Score every supplied candle.

    The feature matrix is rebuilt from raw klines and reordered to the bundle's
    training column order, so a mismatch surfaces as an error instead of a silent
    misalignment of feature positions.
    """
    if klines.empty:
        raise ValueError("Cannot predict on an empty kline frame")

    features = FeatureEngineer(config.features).build(klines)
    if features.empty:
        raise ValueError(
            "Feature matrix is empty: the supplied history is shorter than the "
            f"warm-up period of {FeatureEngineer(config.features).required_history} candles."
        )
    missing = [c for c in bundle.feature_columns if c not in features.columns]
    if missing:
        raise ValueError(f"Feature(s) missing at inference time: {missing}")

    X = features[bundle.feature_columns].astype("float64")
    proba = np.asarray(bundle.model.predict_proba(X))[:, 1]

    threshold = decision_threshold if decision_threshold is not None else config.backtest.get(
        "probability_threshold", 0.60
    )
    predicted = (proba >= float(threshold)).astype(int)
    return pd.DataFrame(
        {
            "probability_up": proba,
            "prediction": np.where(predicted == 1, "UP", "DOWN"),
            "confidence": np.maximum(proba, 1.0 - proba),
            "threshold": float(threshold),
        },
        index=features.index,
    )


def predict_latest(
    klines: pd.DataFrame,
    bundle: ModelBundle,
    config: Config,
    *,
    decision_threshold: float | None = None,
) -> Prediction:
    """Score the most recent candle and return a single :class:`Prediction`.

    Note that the newest candle returned by Binance may still be forming.  Pass
    fully-closed candles only (the downloader already enforces this).
    """
    set_global_seed(config.random_state)
    scored = predict_from_klines(klines, bundle, config, decision_threshold=decision_threshold)
    if scored.empty:
        raise ValueError("No scorable candles were produced")
    row = scored.iloc[-1]
    return Prediction(
        timestamp=scored.index[-1],
        probability_up=float(row["probability_up"]),
        prediction=str(row["prediction"]),
        confidence=float(row["confidence"]),
        threshold=float(row["threshold"]),
    )


def predict_and_save(
    config: Config,
    klines: pd.DataFrame,
    *,
    name: str | None = None,
    filename: str | None = None,
) -> tuple[Path, Prediction]:
    """Score a frame, persist the predictions, and return the latest prediction."""
    bundle = load_model(config, name)
    scored = predict_from_klines(klines, bundle, config)
    scored.insert(0, "symbol", config.symbol)
    scored.insert(0, "interval", config.interval)

    stem = filename or f"{config.symbol}_{config.interval}_{bundle.model_name}_predictions"
    path = config.paths.predictions_dir / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=True)
    logger.info("Wrote %d prediction(s) -> %s", len(scored), path)

    latest = Prediction(
        timestamp=scored.index[-1],
        probability_up=float(scored["probability_up"].iloc[-1]),
        prediction=str(scored["prediction"].iloc[-1]),
        confidence=float(scored["confidence"].iloc[-1]),
        threshold=float(scored["threshold"].iloc[-1]),
    )
    logger.info("Latest: %s", latest.describe(config.symbol))
    return path, latest
