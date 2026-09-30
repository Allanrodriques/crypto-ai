"""Walk-forward evaluation.

Why not a single split
----------------------
One train/test split answers "how did this model do in one particular period",
which for a 2-year-old test set is really a question about one market regime.  A
result of 0.58 ROC-AUC might be driven entirely by one trending quarter.

Walk-forward answers a different and more useful question: *how stable is this
across successive windows of time?*  It repeatedly trains on everything up to a
cut point and predicts the block immediately after, then rolls forward.  Each
window is disjoint in its test period, so the windows collectively cover a long
stretch of out-of-sample history.

Reported statistics
-------------------
``mean``, ``median``, ``std``, ``min``, ``max`` per metric, plus the fraction of
windows that beat the majority-class baseline.  Reporting a spread rather than a
single number is the entire point: a mean of 0.57 with a std of 0.06 and a range
of 0.48-0.68 is a very different research conclusion from 0.57 +/- 0.01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.utils import get_logger

logger = get_logger("evaluation.walkforward")


class WalkForwardError(ValueError):
    """Raised for an inconsistent walk-forward configuration."""


@dataclass(frozen=True)
class WalkForwardPlan:
    """How many windows, how long each train/test block, and how to step.

    Parameters
    ----------
    n_windows:
        Number of successive test blocks.
    train_size:
        Rows in each training block (the first window's train block is the
        earliest slice; later windows extend it rather than sliding a fixed
        window, which is the standard expanding-window convention).
    test_size:
        Rows in each test block.
    step:
        Rows to advance between windows.  Defaults to ``test_size`` so test
        blocks do not overlap.
    gap:
        Rows dropped between train and test to purge label overlap.
    """

    n_windows: int = 5
    train_size: int = 5_000
    test_size: int = 1_500
    step: int | None = None
    gap: int = 0
    anchor: str = "end"

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            raise WalkForwardError("n_windows must be >= 1")
        if self.test_size < 1:
            raise WalkForwardError("test_size must be >= 1")
        if self.step is None:
            object.__setattr__(self, "step", self.test_size)
        if self.step < 1:
            raise WalkForwardError("step must be >= 1")
        if self.gap < 0:
            raise WalkForwardError("gap must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "step": self.step,
            "gap": self.gap,
            "anchor": self.anchor,
        }


@dataclass
class WalkForwardWindow:
    """One train/test window with its index range and realised metrics."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.index,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metrics": self.metrics,
        }


def plan_windows(n_rows: int, plan: WalkForwardPlan) -> list[tuple[int, int, int, int]]:
    """Resolve a plan into concrete ``(train_lo, train_hi, test_lo, test_hi)``.

    Windows are laid out backwards from the end of the data by default, so the
    most recent window is fully populated and older windows are dropped if the
    data is too short.  Laying them out forwards instead would leave the final
    - and most informative - window truncated.
    """
    n_windows, step, test = plan.n_windows, plan.step, plan.test_size
    if plan.anchor == "end":
        final_test_hi = n_rows
        windows = []
        for k in range(n_windows):
            test_hi = final_test_hi - k * step
            test_lo = test_hi - test
            train_hi = test_lo - plan.gap
            train_lo = train_hi - plan.train_size
            if test_lo <= 0 or train_lo < 0:
                break
            windows.append((train_lo, train_hi, test_lo, test_hi))
        windows.reverse()
        return windows

    windows = []
    for k in range(n_windows):
        train_lo = k * step
        train_hi = train_lo + plan.train_size
        test_lo = train_hi + plan.gap
        test_hi = test_lo + test
        if test_hi > n_rows:
            break
        windows.append((train_lo, train_hi, test_lo, test_hi))
    return windows


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_factory,
    plan: WalkForwardPlan,
    *,
    metric_fn,
    threshold: float = 0.5,
) -> list[WalkForwardWindow]:
    """Train/evaluate ``estimator_factory`` over each planned window.

    ``metric_fn(y_true, y_proba) -> dict`` computes the metrics of interest, so
    binary and three-class targets share one implementation.
    """
    windows = plan_windows(len(X), plan)
    if not windows:
        raise WalkForwardError(
            f"no window fits: {len(X)} rows, train={plan.train_size}, test={plan.test_size}, "
            f"gap={plan.gap}, step={plan.step}"
        )

    results: list[WalkForwardWindow] = []
    for i, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(windows, start=1):
        X_tr, y_tr = X.iloc[tr_lo:tr_hi], y.iloc[tr_lo:tr_hi]
        X_te, y_te = X.iloc[te_lo:te_hi], y.iloc[te_lo:te_hi]

        estimator = clone(estimator_factory) if hasattr(estimator_factory, "get_params") else estimator_factory()
        estimator.fit(X_tr, y_tr)
        proba = _predict_proba(estimator, X_te)

        results.append(
            WalkForwardWindow(
                index=i,
                train_start=X.index[tr_lo],
                train_end=X.index[tr_hi - 1],
                test_start=X.index[te_lo],
                test_end=X.index[te_hi - 1],
                n_train=len(X_tr),
                n_test=len(X_te),
                metrics={k: float(v) for k, v in metric_fn(y_te, proba).items()},
            )
        )
        logger.info(
            "walk-forward window %d/%d  train %s..%s  test %s..%s  %s",
            i, len(windows), X.index[tr_lo].date(), X.index[tr_hi - 1].date(),
            X.index[te_lo].date(), X.index[te_hi - 1].date(),
            " ".join(f"{k}={v:.4f}" for k, v in results[-1].metrics.items() if k in {"roc_auc", "pr_auc"}),
        )
    return results


def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X))
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw
    return np.asarray(estimator.predict(X))


def summarise_windows(windows: Sequence[WalkForwardWindow]) -> dict[str, Any]:
    """Aggregate per-window metrics into mean/median/std/min/max."""
    if not windows:
        return {}
    keys = sorted({k for w in windows for k in w.metrics})
    summary: dict[str, Any] = {"n_windows": len(windows), "windows": [w.to_dict() for w in windows]}
    for key in keys:
        values = np.array([w.metrics.get(key, np.nan) for w in windows], dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        }
    return summary


def windows_frame(windows: Sequence[WalkForwardWindow]) -> pd.DataFrame:
    """Flat per-window table for CSV export."""
    rows = []
    for w in windows:
        row = {k: v for k, v in w.to_dict().items() if k != "metrics"}
        row.update(w.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
