"""Dataset construction for V2 experiments.

V1's :class:`~src.dataset.dataset_builder.DatasetBuilder` hard-wires the V1
feature set (it calls ``FeatureEngineer`` internally), which is exactly right for
the immutable baseline but cannot express "technical + sentiment" or a
three-class target.  This module assembles a dataset from an *already built*
feature matrix, reusing V1's :class:`~src.dataset.dataset_builder.Dataset` and
:class:`~src.dataset.dataset_builder.DataSplit` containers so that V1 training,
evaluation and backtesting code works unchanged on V2 datasets.

What is reused, deliberately
----------------------------
* chronological, never-shuffled splitting with horizon purging
* the ``Dataset``/``DataSplit`` containers and ``assert_no_leakage``
* the rule that ``features[t]`` uses ``<= t`` and only ``target`` looks forward

What is new
-----------
* any combination of feature groups
* configurable target: horizon, threshold, and binary vs three-class
* dataset versioning with a content hash (requirement: reproducibility)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.dataset.dataset_builder import (
    SPLIT_NAMES,
    DataSplit,
    Dataset,
    LeakageError,
)
from src.utils import format_timestamp, get_logger, interval_to_milliseconds, save_json

logger = get_logger("dataset.factory")

#: Label codes for the three-class formulation.
DOWN, NEUTRAL, UP = 0, 1, 2
CLASS_NAMES_3 = ("DOWN", "NEUTRAL", "UP")


class TargetSpecError(ValueError):
    """Raised for an invalid target specification."""


@dataclass(frozen=True)
class TargetSpec:
    """Configurable prediction target.

    Parameters
    ----------
    horizon_candles:
        How far ahead the return is measured.
    threshold:
        Absolute return that separates a directional move from a flat one.
    mode:
        ``"binary"`` -> ``1`` when ``future_return >= threshold`` else ``0``.
        ``"three_class"`` -> ``2`` (UP) above ``+threshold``, ``0`` (DOWN)
        below ``-threshold``, ``1`` (NEUTRAL) in between.  Thresholds are
        applied symmetrically.
    """

    horizon_candles: int
    threshold: float
    mode: str = "binary"

    def __post_init__(self) -> None:
        if self.horizon_candles < 1:
            raise TargetSpecError("horizon_candles must be >= 1")
        if self.threshold < 0:
            raise TargetSpecError("threshold must be >= 0")
        if self.mode not in {"binary", "three_class"}:
            raise TargetSpecError(f"mode must be 'binary' or 'three_class', got {self.mode!r}")

    @property
    def n_classes(self) -> int:
        return 2 if self.mode == "binary" else 3

    @property
    def positive_class(self) -> int:
        return 1

    def describe(self) -> str:
        if self.mode == "binary":
            return (
                f"binary: target[t] = 1 if close[t + {self.horizon_candles}] / close[t] - 1 "
                f">= {self.threshold} else 0"
            )
        return (
            f"three_class: UP if return > +{self.threshold}, DOWN if < -{self.threshold}, "
            f"NEUTRAL otherwise, over {self.horizon_candles} candle(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "horizon_candles": self.horizon_candles,
            "threshold": self.threshold,
            "n_classes": self.n_classes,
            "definition": self.describe(),
        }


@dataclass(frozen=True)
class SplitSpec:
    """Chronological split ratios with horizon purging."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise TargetSpecError(f"split ratios must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "chronological (never shuffled)",
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
        }


def build_target(
    close: pd.Series, spec: TargetSpec, *, interval: str = "1h"
) -> pd.DataFrame:
    """Build ``future_return`` and ``target`` from a close series.

    The only forward-looking computation in the project.  Rows whose label
    cannot be resolved (the final ``horizon`` bars) are left as ``NaN`` and
    dropped by the caller rather than being back-filled.
    """
    step = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
    forward_close = close.shift(-spec.horizon_candles)
    future_return = (forward_close / close) - 1.0

    if spec.mode == "binary":
        target = (future_return >= spec.threshold).astype("float64")
    else:
        target = np.where(
            future_return > spec.threshold,
            float(UP),
            np.where(future_return < -spec.threshold, float(DOWN), float(NEUTRAL)),
        )
        target = pd.Series(target, index=close.index, dtype="float64")

    target[forward_close.isna()] = np.nan
    return pd.DataFrame({"future_return": future_return, "target": target}, index=close.index)


def build_dataset(
    features: pd.DataFrame,
    market: pd.DataFrame,
    target_spec: TargetSpec,
    split_spec: SplitSpec,
    *,
    symbol: str,
    interval: str,
    feature_groups: Sequence[str],
    data_sources: Sequence[str],
    drop_incomplete: bool = True,
) -> Dataset:
    """Assemble a :class:`Dataset` from a pre-computed feature matrix.

    Parameters
    ----------
    features:
        Feature frame, already aligned and causal, UTC indexed.
    market:
        Raw OHLCV on the same index (used by the backtest and plots).
    target_spec:
        Horizon / threshold / class formulation.
    split_spec:
        Chronological ratios.
    feature_groups, data_sources:
        Recorded in metadata for experiment provenance.
    """
    if features.empty:
        raise ValueError("Cannot build a dataset from an empty feature matrix")

    features = features.sort_index()
    labels = build_target(market["close"], target_spec, interval=interval).loc[features.index]
    market = market.loc[features.index]

    complete = features.notna().all(axis=1)
    if drop_incomplete:
        keep = complete & labels["target"].notna()
    else:
        keep = labels["target"].notna()
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "dropping %d row(s): %d incomplete feature row(s), %d unresolvable label(s)",
            dropped, int((~complete).sum()), int(labels["target"].isna().sum()),
        )

    frame = features.loc[keep].copy()
    frame["future_return"] = labels.loc[keep, "future_return"]
    frame["target"] = labels.loc[keep, "target"].astype("int8")
    for column in ("open", "high", "low", "close", "volume"):
        if column in market.columns:
            frame[column] = market.loc[keep, column]
    frame.index.name = "timestamp"

    feature_columns = list(features.columns)
    splits = _slice(frame, feature_columns, split_spec, purge=target_spec.horizon_candles)

    dataset = Dataset(
        symbol=symbol,
        interval=interval,
        horizon_candles=target_spec.horizon_candles,
        threshold=target_spec.threshold,
        feature_columns=feature_columns,
        frame=frame,
        splits=splits,
        metadata={
            "symbol": symbol,
            "interval": interval,
            "rows": int(len(frame)),
            "n_rows": int(len(frame)),
            "range_start": format_timestamp(frame.index.min()),
            "range_end": format_timestamp(frame.index.max()),
            "target": target_spec.to_dict(),
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            # Hash of the *names and order* only, independent of the values.  Two
            # experiments are comparable on a metric only if they were built from
            # the same feature list, so this pins that separately from
            # `dataset_hash`, which also covers the numbers.
            "feature_manifest_hash": feature_manifest_hash(feature_columns),
            "feature_groups": list(feature_groups),
            "data_sources": list(data_sources),
            "rows_dropped_in_build": int(dropped),
            # Per-split row counts and date bounds.  The *test* bounds in
            # particular matter for the report: a headline table that labels the
            # full dataset span as the evaluation period is wrong, because the
            # score was measured on the last 15% only.
            "split": {
                **split_spec.to_dict(),
                "purge_candles": target_spec.horizon_candles,
                "bounds": {
                    name: {
                        "n": int(len(s)),
                        "start": format_timestamp(s.index.min()) if len(s) else None,
                        "end": format_timestamp(s.index.max()) if len(s) else None,
                    }
                    for name, s in splits.items()
                },
            },
        },
        horizon_ms=interval_to_milliseconds(interval) * target_spec.horizon_candles,
    )
    dataset.assert_no_leakage()
    return dataset


def _slice(
    frame: pd.DataFrame, feature_columns: list[str], spec: SplitSpec, *, purge: int
) -> dict[str, DataSplit]:
    """Chronological, purged slicing (same contract as V1)."""
    n = len(frame)
    n_train = int(n * spec.train_ratio)
    n_valid = int(n * spec.validation_ratio)
    bounds = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_valid),
        "test": (n_train + n_valid, n),
    }
    splits: dict[str, DataSplit] = {}
    for order, name in enumerate(SPLIT_NAMES):
        lo, hi = bounds[name]
        if order < len(SPLIT_NAMES) - 1:
            hi = max(lo, hi - purge)
        block = frame.iloc[lo:hi]
        if block.empty:
            logger.warning("split %r is empty after purging", name)
            continue
        splits[name] = DataSplit(
            name=name,
            X=block[feature_columns].astype("float64"),
            y=block["target"].astype("int8"),
            frame=block,
        )
    return splits


# --------------------------------------------------------------------------- versioning

def feature_manifest_hash(feature_columns: Sequence[str]) -> str:
    """Hash of the feature *names* and their order, ignoring their values.

    Comparability between two experiments is a property of the columns they
    were trained on, not of the numbers in them: two runs on different feature
    lists must never be differenced against each other even if their rows
    happen to align.
    """
    hasher = hashlib.sha256()
    hasher.update(str(len(feature_columns)).encode())
    for name in feature_columns:
        hasher.update(b"\x00")
        hasher.update(name.encode())
    return hasher.hexdigest()[:16]


def dataset_hash(frame: pd.DataFrame, feature_columns: Sequence[str]) -> str:
    """Content hash of the *feature matrix*, so identical data yields one version.

    Hashes the index and the feature values only - not the label - because the
    same feature matrix legitimately appears under several target definitions
    (a 6h and a 24h label over identical history), and those are different
    experiments rather than different data.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(feature_columns).encode())
    values = frame[list(feature_columns)].to_numpy(dtype="float64")
    hasher.update(str(values.shape).encode())
    hasher.update(np.ascontiguousarray(values).tobytes())
    hasher.update(frame.index.astype("int64").to_numpy().tobytes())
    return hasher.hexdigest()[:16]


@dataclass
class DatasetVersion:
    """Immutable description of one dataset build, for reproducibility."""

    version: str
    created_at: str
    symbol: str
    interval: str
    data_sources: list[str]
    feature_groups: list[str]
    target: dict[str, Any]
    date_range: dict[str, str]
    row_count: int
    feature_count: int
    hash: str
    path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.version,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "timeframe": self.interval,
            "data_sources": self.data_sources,
            "feature_groups": self.feature_groups,
            "target": self.target,
            "date_range": self.date_range,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "hash": self.hash,
            "path": self.path,
            **({"extra": self.extra} if self.extra else {}),
        }


def version_dataset(
    dataset: Dataset,
    *,
    data_sources: Sequence[str],
    feature_groups: Sequence[str],
    out_dir: Path,
    name_hint: str = "dataset",
) -> DatasetVersion:
    """Compute a version record, write the parquet and its metadata sidecar."""
    content_hash = dataset_hash(dataset.frame, dataset.feature_columns)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"{name_hint}_v{created}_{content_hash[:8]}"

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{version}.parquet"
    dataset.frame.to_parquet(parquet_path)

    record = DatasetVersion(
        version=version,
        created_at=created,
        symbol=dataset.symbol,
        interval=dataset.interval,
        data_sources=list(data_sources),
        feature_groups=list(feature_groups),
        target=dataset.metadata.get("target", {}),
        date_range={
            "start": str(dataset.frame.index.min()),
            "end": str(dataset.frame.index[-1]),
        },
        row_count=int(len(dataset.frame)),
        feature_count=len(dataset.feature_columns),
        hash=content_hash,
        path=str(parquet_path),
    )
    save_json({**record.to_dict(), "metadata": dataset.metadata}, out_dir / f"{version}.meta.json")
    logger.info("dataset version %s (%d rows, %d features)", version, record.row_count, record.feature_count)
    return record
