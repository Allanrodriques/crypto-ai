"""Time-series cross-validation with an explicit purge gap.

Why not ``TimeSeriesSplit``?
----------------------------
``sklearn.model_selection.TimeSeriesSplit`` respects ordering but has no notion
of a *label horizon*.  With ``horizon_candles = 6`` the label of the last
training row resolves against ``close[t + 6]``, which may sit inside the
validation fold.  Training on it lets the model see validation-period prices —
a real leak that inflates out-of-sample scores.

:class:`PurgedExpandingWindowSplit` therefore removes ``gap`` rows from the tail
of every training fold before handing it to a model.  ``gap`` defaults to the
target horizon (set in ``config/config.yaml`` as ``cv.gap``).

Splits are also *expanding*: fold ``k`` trains on everything before it, so later
folds see more history, and no fold ever trains on data from its own future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import format_timestamp, get_logger

logger = get_logger("evaluation.cv")


class InsufficientDataError(ValueError):
    """Raised when the training block cannot supply the requested number of folds."""


@dataclass(frozen=True)
class Fold:
    """One train/validation fold expressed as positional indices into a frame."""

    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None

    @property
    def n_train(self) -> int:
        return int(self.train_index.size)

    @property
    def n_validation(self) -> int:
        return int(self.validation_index.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "train_range": [format_timestamp(self.train_start), format_timestamp(self.train_end)],
            "validation_range": [
                format_timestamp(self.validation_start),
                format_timestamp(self.validation_end),
            ],
        }


class PurgedExpandingWindowSplit:
    """Expanding-window splitter with a purge gap between train and validation.

    Parameters
    ----------
    n_splits:
        Number of validation folds.
    gap:
        Rows removed from the end of each training fold.  Defaults to the label
        horizon so no training label resolves against a validation-period close.
    min_train_size:
        Smallest permitted training fold.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.date_range("2022-01-01", periods=100, freq="1h", tz="UTC")
    >>> splitter = PurgedExpandingWindowSplit(n_splits=3, gap=2)
    >>> folds = list(splitter.split(idx))
    >>> len(folds)
    3
    >>> bool((folds[0].train_index.max() < folds[0].validation_index.min()))
    True
    """

    def __init__(self, n_splits: int = 5, *, gap: int = 0, min_train_size: int = 100) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        self.n_splits = int(n_splits)
        self.gap = int(gap)
        self.min_train_size = int(min_train_size)

    def split(self, index: Sequence[pd.Timestamp] | pd.Index) -> Iterator[Fold]:
        """Yield folds over a chronologically sorted index.

        Validation blocks tile the tail of the series; training blocks expand
        backwards from each validation block's start.
        """
        idx = pd.DatetimeIndex(index)
        if not idx.is_monotonic_increasing:
            raise ValueError("PurgedExpandingWindowSplit requires a chronologically sorted index")
        n = len(idx)
        if n < self.min_train_size + self.n_splits * (self.gap + 1):
            raise InsufficientDataError(
                f"Not enough rows for {self.n_splits} purged folds: need at least "
                f"{self.min_train_size + self.n_splits * (self.gap + 1)}, got {n}. "
                "Reduce cv.n_splits, cv.min_train_size, or download more history."
            )

        # Reserve room for the purged gap and one validation row per fold.
        first_validation = self.min_train_size + self.gap
        remaining = n - first_validation
        if remaining < self.n_splits:
            raise InsufficientDataError(
                f"Only {remaining} row(s) available for validation after a {first_validation}-row "
                f"train prefix; need at least {self.n_splits}."
            )

        # Uneven remainder is spread across the earliest folds so later folds
        # never receive a validation block of zero length.
        base, extra = divmod(remaining, self.n_splits)
        sizes = [base + (1 if i < extra else 0) for i in range(self.n_splits)]

        cursor = first_validation
        for fold_no, size in enumerate(sizes, start=1):
            val_lo, val_hi = cursor, cursor + size
            train_hi = val_lo - self.gap  # <- the purge
            if train_hi < self.min_train_size:
                raise InsufficientDataError(f"Fold {fold_no} would train on fewer than min_train_size rows")
            yield Fold(
                fold=fold_no,
                train_index=np.arange(0, train_hi),
                validation_index=np.arange(val_lo, val_hi),
                train_start=idx[0],
                train_end=idx[train_hi - 1],
                validation_start=idx[val_lo],
                validation_end=idx[val_hi - 1],
            )
            cursor = val_hi

    def get_n_splits(self) -> int:
        return self.n_splits

    def describe(self, index: Sequence[pd.Timestamp] | pd.Index) -> list[dict[str, Any]]:
        return [fold.to_dict() for fold in self.split(index)]


def cross_validate(
    estimator_factory: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splitter: PurgedExpandingWindowSplit,
    threshold: float = 0.5,
    with_calibration: bool = True,
) -> dict[str, Any]:
    """Run purged expanding-window CV and aggregate out-of-fold metrics.

    Every fold builds a *fresh* estimator from ``estimator_factory`` and fits it
    on that fold's training rows only.  The factory is a function, not an
    instance, precisely so state cannot leak between folds.

    Returns
    -------
    dict
        ``folds`` (per-fold metrics), ``aggregate`` (mean/std of the headline
        metrics) and ``oof_probability`` (out-of-fold ``probability_up``,
        aligned to the validation rows only).
    """
    from sklearn.base import clone

    from src.evaluation.metrics import calibration_metrics, classification_metrics

    per_fold: list[dict[str, Any]] = []
    oof_index: list[pd.Timestamp] = []
    oof_probability: list[float] = []
    oof_actual: list[int] = []

    for fold in splitter.split(X.index):
        X_tr, y_tr = X.iloc[fold.train_index], y.iloc[fold.train_index]
        X_va, y_va = X.iloc[fold.validation_index], y.iloc[fold.validation_index]

        if y_tr.nunique() < 2:
            logger.warning("Fold %d has a single-class training block; skipping", fold.fold)
            continue

        estimator = clone(estimator_factory) if hasattr(estimator_factory, "get_params") else estimator_factory()
        estimator.fit(X_tr, y_tr)
        proba = np.asarray(estimator.predict_proba(X_va))[:, 1]

        entry = fold.to_dict()
        entry["metrics"] = classification_metrics(y_va, proba, threshold=threshold)
        if with_calibration:
            entry["calibration"] = calibration_metrics(y_va, proba)
        per_fold.append(entry)

        oof_index.extend(X_va.index.tolist())
        oof_probability.extend(proba.tolist())
        oof_actual.extend(np.asarray(y_va).astype(int).tolist())

    if not per_fold:
        raise InsufficientDataError("No fold produced a usable training set")

    keys = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    aggregate: dict[str, Any] = {"n_folds": len(per_fold), "gap": splitter.gap, "strategy": "expanding_window_purged"}
    for key in keys:
        values = [f["metrics"].get(key) for f in per_fold if f["metrics"].get(key) is not None]
        aggregate[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }

    oof = pd.DataFrame(
        {"probability_up": oof_probability, "target": oof_actual},
        index=pd.DatetimeIndex(oof_index, name="timestamp"),
    )
    return {"folds": per_fold, "aggregate": aggregate, "oof": oof}
