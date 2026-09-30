"""Build the leakage-free ML dataset: features, label, and chronological splits.

This module is the **only** place in the project permitted to look at future
prices, and it does so for exactly one purpose: constructing the label.

Feature/label separation
------------------------
============================  ==========================================================
:mod:`src.features`          ``feature[t]`` uses candles ``<= t``.  Pinned by
                             ``tests/test_features.py``.
**This module**              ``target[t]`` uses ``close[t + horizon]``.  That is
                             the label; a label is *supposed* to be resolved in
                             the future.
============================  ==========================================================

Splitting
---------
Rows are ordered by time and cut by ratio.  The series is never shuffled, and
``train_test_split`` is never called.  Additionally every non-final split has its
last ``purge_candles`` rows removed, because a training label at ``t`` resolves
against ``close[t + horizon]`` — without purging, a training label would be
partly priced from inside the *next* (untouched) period.  This is the standard
purging/embargo step from financial-ML practice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from src.config import Config
from src.features.feature_engineering import FeatureEngineer
from src.utils import format_timestamp, get_logger, save_json, timestamp_to_utc

logger = get_logger("dataset")

SPLIT_NAMES = ("train", "validation", "test")

#: Raw OHLCV columns carried alongside the feature matrix so downstream code
#: (backtest, plots) never has to re-read the raw file.
MARKET_COLUMNS = ("open", "high", "low", "close", "volume")


class LeakageError(RuntimeError):
    """Raised when a constructed split violates a chronological guarantee."""


# --------------------------------------------------------------------------- split

@dataclass
class DataSplit:
    """One chronologically contiguous block of the dataset."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame  # full row view: features + label + market columns

    def __len__(self) -> int:
        return len(self.X)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def start(self) -> pd.Timestamp | None:
        return self.frame.index.min() if len(self.frame) else None

    @property
    def end(self) -> pd.Timestamp | None:
        return self.frame.index.max() if len(self.frame) else None

    def class_distribution(self) -> dict[str, Any]:
        counts = self.y.value_counts().sort_index()
        total = int(counts.sum())
        return {
            "n": total,
            "n_up": int(counts.get(1, 0)),
            "n_down": int(counts.get(0, 0)),
            "share_up": float(counts.get(1, 0) / total) if total else None,
            "imbalance_ratio": float(counts.get(0, 0) / counts.get(1, 1)) if counts.get(1, 0) else None,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": len(self),
            "start": format_timestamp(self.start),
            "end": format_timestamp(self.end),
            "class_distribution": self.class_distribution(),
        }


@dataclass
class Dataset:
    """The complete assembled dataset plus its chronological splits."""

    symbol: str
    interval: str
    horizon_candles: int
    threshold: float
    feature_columns: list[str]
    frame: pd.DataFrame  # features + future_return + target + market columns
    splits: dict[str, DataSplit]
    metadata: dict[str, Any] = field(default_factory=dict)
    horizon_ms: int = 3_600_000  # milliseconds spanned by the label horizon

    # ------------------------------------------------------------ accessors

    @property
    def X(self) -> pd.DataFrame:
        return self.splits["train"].X

    @property
    def y(self) -> pd.Series:
        return self.splits["train"].y

    def split(self, name: str) -> DataSplit:
        if name not in self.splits:
            raise KeyError(f"Unknown split {name!r}; available: {sorted(self.splits)}")
        return self.splits[name]

    def __iter__(self) -> Iterator[DataSplit]:
        for name in SPLIT_NAMES:
            if name in self.splits:
                yield self.splits[name]

    # ------------------------------------------------------------ integrity

    def assert_no_leakage(self) -> "Dataset":
        """Verify every chronological invariant this module promises.

        Raises :class:`LeakageError` on any violation.  Called at construction
        time, so an unsafe dataset cannot reach training.
        """
        present = [n for n in SPLIT_NAMES if n in self.splits and len(self.splits[n])]
        previous: DataSplit | None = None
        for name in present:
            current = self.splits[name]
            if not current.index.is_monotonic_increasing or not current.index.is_unique:
                raise LeakageError(f"Split {name!r} is not a sorted, unique time series")
            if previous is not None:
                if current.index.min() <= previous.index.max():
                    raise LeakageError(
                        f"Split {name!r} starts at {format_timestamp(current.index.min())}, "
                        f"which is not strictly after split {previous.name!r} "
                        f"ends at {format_timestamp(previous.index.max())}"
                    )
                # No training label may resolve against a price inside a later split.
                train_last_row = previous.index[-1]
                label_end = train_last_row + pd.Timedelta(milliseconds=self.horizon_ms)
                if label_end > current.index.min():
                    raise LeakageError(
                        f"Label of {previous.name!r} row {format_timestamp(train_last_row)} resolves at "
                        f"{format_timestamp(label_end)}, inside split {name!r}. Purge is too small."
                    )
            previous = current

        if "target" not in self.frame.columns:
            raise LeakageError("Dataset frame has no 'target' column")
        if self.frame.index.has_duplicates:
            raise LeakageError("Dataset frame contains duplicate timestamps")
        return self

    # --------------------------------------------------------------- persist

    def save(self, directory: str | os.PathLike[str], stem: str | None = None) -> dict[str, Path]:
        """Persist the dataset frame and a metadata sidecar."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.symbol}_{self.interval}_dataset"
        frame_path = out / f"{stem}.parquet"
        meta_path = out / f"{stem}.meta.json"

        self.frame.to_parquet(frame_path, index=True)
        save_json(self.metadata, meta_path)
        logger.info("Saved dataset frame (%d rows) -> %s", len(self.frame), frame_path)
        return {"frame": frame_path, "metadata": meta_path}

    @classmethod
    def load(cls, path: str | os.PathLike[str], config: Config) -> "Dataset":
        """Rebuild a :class:`Dataset` from a persisted frame using ``config``.

        The splits are recomputed with the *current* config rather than trusted
        from disk, so a stale frame can never silently change the experiment.
        """
        from src.features.feature_engineering import FEATURE_DOCS
        from src.utils import load_json

        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")

        meta_path = Path(path).with_suffix(".meta.json")
        metadata: dict[str, Any] = load_json(meta_path) if meta_path.exists() else {}
        feature_columns = list(metadata.get("feature_columns")) or [c for c in FEATURE_DOCS if c in frame.columns]

        builder = DatasetBuilder(config)
        splits = builder._slice_splits(frame, feature_columns, purge_candles=builder.purge_candles)
        return cls(
            symbol=str(metadata.get("symbol", config.symbol)),
            interval=str(metadata.get("interval", config.interval)),
            horizon_candles=builder.horizon,
            threshold=builder.threshold,
            feature_columns=feature_columns,
            frame=frame,
            splits=splits,
            metadata=metadata,
            horizon_ms=builder._horizon_ms,
        )


# --------------------------------------------------------------------------- builder

class DatasetBuilder:
    """Assemble features + label + chronological splits for one series."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.symbol = config.symbol
        self.interval = config.interval
        self.horizon = config.horizon_candles
        self.threshold = config.threshold
        self.feature_config = config.features
        self._horizon_ms = _horizon_ms(self.interval, self.horizon)

        split_cfg = config.split
        self.train_ratio = float(split_cfg["train_ratio"])
        self.validation_ratio = float(split_cfg["validation_ratio"])
        self.test_ratio = float(split_cfg["test_ratio"])
        self.purge_candles = int(
            split_cfg.get("purge_candles") if split_cfg.get("purge_candles") is not None else self.horizon
        )

    # ---------------------------------------------------------------- label

    def build_target(self, close: pd.Series) -> pd.DataFrame:
        """Construct the label columns.

        ``future_return`` and ``target`` are the ONLY forward-looking quantities
        in the project.  The final ``horizon`` rows have no resolvable label and
        are returned as ``NaN`` so the caller drops them explicitly rather than
        inventing a value.
        """
        close = close.astype("float64")
        forward_close = close.shift(-self.horizon)
        future_return = (forward_close / close) - 1.0
        target = (future_return >= self.threshold).astype("float64")
        target[forward_close.isna()] = np.nan  # unresolvable -> explicitly missing
        return pd.DataFrame({"future_return": future_return, "target": target}, index=close.index)

    # ------------------------------------------------------------- assembly

    def build(self, klines: pd.DataFrame, *, engineer: FeatureEngineer | None = None) -> Dataset:
        """Build the full dataset from a validated OHLCV frame."""
        if klines.empty:
            raise ValueError("Cannot build a dataset from an empty kline frame")

        engineer = engineer or FeatureEngineer(self.feature_config)
        features = engineer.build(klines)
        if features.empty:
            raise ValueError("Feature matrix is empty; the history is shorter than the warm-up period")

        market = klines.loc[features.index, [c for c in MARKET_COLUMNS if c in klines.columns]]
        labels = self.build_target(klines["close"]).loc[features.index]

        # Only rows with a resolvable label and a warm feature vector survive.
        keep = features.notna().all(axis=1) & labels["target"].notna()
        dropped = int((~keep).sum())
        if dropped:
            logger.info(
                "Dropping %d row(s): %d unresolvable label(s) (last %d candle(s)) + warm-up NaNs",
                dropped, int(labels["target"].isna().sum()), self.horizon,
            )

        frame = features.loc[keep].copy()
        frame["future_return"] = labels.loc[keep, "future_return"]
        frame["target"] = labels.loc[keep, "target"].astype("int8")
        for col in market.columns:
            frame[col] = market.loc[keep, col]
        frame.index.name = "timestamp"

        feature_columns = list(features.columns)
        splits = self._slice_splits(frame, feature_columns, purge_candles=self.purge_candles)

        metadata = self._metadata(frame, splits, feature_columns, warmup=engineer.required_history, dropped=dropped)
        dataset = Dataset(
            symbol=self.symbol,
            interval=self.interval,
            horizon_candles=self.horizon,
            threshold=self.threshold,
            feature_columns=feature_columns,
            frame=frame,
            splits=splits,
            metadata=metadata,
            horizon_ms=self._horizon_ms,
        )
        dataset.assert_no_leakage()

        logger.info(
            "Dataset: %d rows, %d features, horizon=%d, threshold=%.4f | %s",
            len(frame), len(feature_columns), self.horizon, self.threshold,
            " | ".join(f"{s.name}={len(s)}" for s in splits.values()),
        )
        return dataset

    def _slice_splits(
        self, frame: pd.DataFrame, feature_columns: list[str], *, purge_candles: int
    ) -> dict[str, DataSplit]:
        """Cut the frame into chronological, purged train/validation/test blocks."""
        n = len(frame)
        n_train = int(n * self.train_ratio)
        n_valid = int(n * self.validation_ratio)
        # Leave the remainder to test so the ratios always sum to the full set.
        n_test = n - n_train - n_valid

        bounds = {
            "train": (0, n_train),
            "validation": (n_train, n_train + n_valid),
            "test": (n_train + n_valid, n),
        }

        splits: dict[str, DataSplit] = {}
        for order, name in enumerate(SPLIT_NAMES):
            lo, hi = bounds[name]
            if order < len(SPLIT_NAMES) - 1:
                # Purge the tail: those labels resolve inside the *next* period.
                hi = max(lo, hi - purge_candles)
            block = frame.iloc[lo:hi]
            if block.empty:
                logger.warning("Split %r is empty after purging", name)
                continue
            splits[name] = DataSplit(
                name=name,
                X=block[feature_columns].astype("float64"),
                y=block["target"].astype("int8"),
                frame=block,
            )
        return splits

    def _metadata(
        self,
        frame: pd.DataFrame,
        splits: dict[str, DataSplit],
        feature_columns: list[str],
        *,
        warmup: int,
        dropped: int,
    ) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": int(len(frame)),
            "range_start": format_timestamp(frame.index.min()),
            "range_end": format_timestamp(frame.index.max()),
            "target": {
                "type": "binary_classification",
                "horizon_candles": self.horizon,
                "horizon_hours": self.horizon * _interval_hours(self.interval),
                "threshold": self.threshold,
                "definition": (
                    f"target[t] = 1 if close[t + {self.horizon}] / close[t] - 1 >= {self.threshold} else 0"
                ),
            },
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "feature_warmup_candles": int(warmup),
            "rows_dropped_in_build": int(dropped),
            "split": {
                "method": "chronological (never shuffled)",
                "train_ratio": self.train_ratio,
                "validation_ratio": self.validation_ratio,
                "test_ratio": self.test_ratio,
                "purge_candles": self.purge_candles,
                "purge_rationale": (
                    f"The last {self.purge_candles} row(s) of each non-final split are removed so that no "
                    f"training/validation label resolves against a close inside a later split."
                ),
                "blocks": {name: split.summary() for name, split in splits.items()},
            },
        }


# --------------------------------------------------------------------------- helpers

def _horizon_ms(interval: str, horizon: int) -> int:
    from src.utils import interval_to_milliseconds

    return interval_to_milliseconds(interval) * horizon


def _interval_hours(interval: str) -> float:
    from src.utils import interval_to_milliseconds

    return interval_to_milliseconds(interval) / 3_600_000


# --------------------------------------------------------------------------- entry point

def build_dataset(
    config: Config,
    klines: pd.DataFrame,
    *,
    engineer: FeatureEngineer | None = None,
) -> Dataset:
    """Build, persist and return the dataset for ``klines``."""
    dataset = DatasetBuilder(config).build(klines, engineer=engineer)
    dataset.save(config.paths.processed_dir)
    return dataset
