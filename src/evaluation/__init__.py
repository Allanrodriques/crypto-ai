"""Evaluation layer: metrics, probability calibration, time-series CV, backtest, plots.

The layer answers two questions that are deliberately kept apart:

* **Did the model predict well?**  -> :mod:`src.evaluation.metrics`
* **Would a fixed rule on its probability have made money?**  ->
  :mod:`src.evaluation.backtest`

A model can score well and lose money after costs; a losing strategy can still
have a well-ranked ROC.  Every report keeps the two in separate sections.
"""

from src.evaluation.backtest import (
    BacktestConfigError,
    BacktestResult,
    BacktestRules,
    Trade,
    run_backtest,
    threshold_sensitivity,
)
from src.evaluation.metrics import (
    calibration_metrics,
    class_distribution,
    classification_metrics,
    expected_calibration_error,
    headline_metrics,
    reliability_curve,
    summarise_metric_table,
)
from src.evaluation.time_series_cv import (
    Fold,
    InsufficientDataError,
    PurgedExpandingWindowSplit,
    cross_validate,
)

__all__ = [
    "BacktestConfigError",
    "BacktestResult",
    "BacktestRules",
    "Fold",
    "InsufficientDataError",
    "PurgedExpandingWindowSplit",
    "Trade",
    "calibration_metrics",
    "class_distribution",
    "classification_metrics",
    "cross_validate",
    "expected_calibration_error",
    "headline_metrics",
    "reliability_curve",
    "run_backtest",
    "summarise_metric_table",
    "threshold_sensitivity",
]
