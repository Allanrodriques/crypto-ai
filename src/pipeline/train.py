"""Training orchestration: fit, validate, select, persist.

Order of operations (this ordering is the whole point of the module)
-------------------------------------------------------------------
1. Fit every model on the **training block only**.
2. Score on the **validation block** (out-of-sample, still not the test set).
3. Select the best model on **validation** metrics.
4. Refit the selected model on **train + validation**.
5. Touch the **test block exactly once**, for the final report.

Step 4 before step 5 is deliberate: the test set is untouched while the model
that will be measured on it is being decided, and it is touched only after that
decision is frozen.  The test period is never used for tuning, threshold
selection or early stopping.

Artifacts written
-----------------
``models/<name>.joblib``          trained estimator
``models/<name>_cv.json``         per-fold and aggregate time-series CV results
``models/model_metadata.json``    full reproducibility record
``models/model_registry.json``    index of what was trained
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.config import Config
from src.dataset.dataset_builder import Dataset
from src.evaluation.metrics import (
    calibration_metrics,
    classification_metrics,
    headline_metrics,
    reliability_curve,
)
from src.evaluation.time_series_cv import (
    InsufficientDataError,
    PurgedExpandingWindowSplit,
    cross_validate,
)
from src.features.feature_engineering import FEATURE_DOCS
from src.models import ALL_MODELS, build_model, model_params
from src.utils import format_timestamp, get_logger, save_json, set_global_seed, utc_now

logger = get_logger("pipeline.train")

#: Metrics the model-selection step is allowed to optimise, and in what order.
#: ROC-AUC leads because it is threshold-independent and robust to the class
#: imbalance that a "did price rise by 0.5%" target almost always develops.
SELECTION_METRICS = ("roc_auc", "pr_auc", "f1", "accuracy")

MODEL_FILENAMES = {
    "majority_baseline": "majority_baseline.joblib",
    "momentum_baseline": "momentum_baseline.joblib",
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}


@dataclass
class TrainedModel:
    """One fitted model plus everything needed to reproduce and report it."""

    name: str
    estimator: Any
    feature_columns: list[str]
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    validation_calibration: dict[str, Any] = field(default_factory=dict)
    cv_results: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    fitted_on: str = "train"
    #: Validation probabilities from the *train-only* fit, captured before any
    #: refit.  Threshold diagnostics must use these, never the refit estimator's
    #: in-sample validation scores.
    validation_probabilities: np.ndarray | None = None

    def validation_score(self, metric: str) -> float:
        value = self.validation_metrics.get(metric)
        return float(value) if value is not None else float("-inf")


@dataclass
class TrainingResult:
    """Everything :mod:`src.pipeline.evaluate` needs after a training run."""

    dataset: Dataset
    models: dict[str, TrainedModel]
    selected_name: str
    selected: TrainedModel
    config: Config
    metadata: dict[str, Any]

    @property
    def final_estimator(self) -> Any:
        return self.selected.estimator


# --------------------------------------------------------------------------- entry point

def train_models(
    config: Config,
    dataset: Dataset,
    model_names: Sequence[str] = ALL_MODELS,
    *,
    run_cv: bool = True,
    select_on: str = "validation",
    refit_on_train_plus_validation: bool = True,
) -> TrainingResult:
    """Train, cross-validate and select models.

    Parameters
    ----------
    config:
        Active configuration.
    dataset:
        Dataset produced by :class:`~src.dataset.dataset_builder.DatasetBuilder`.
    model_names:
        Which registered models to train.
    run_cv:
        Run purged expanding-window CV on the training block.
    refit_on_train_plus_validation:
        Refit the selected model on train+validation before the final test
        evaluation.  Disable to inspect validation-fitted models.
    """
    set_global_seed(config.random_state)
    threshold = config.backtest.get("probability_threshold", 0.60)

    if "train" not in dataset.splits:
        raise ValueError("Dataset has no training split")

    trained: dict[str, TrainedModel] = {}
    for name in model_names:
        logger.info("Training %s", name)
        trained[name] = _train_one(config, dataset, name, threshold, run_cv=run_cv)

    selected_name, selected = _select_model(trained, model_names, select_on)
    logger.info("Selected %r on %s metrics (pre-refit)", selected_name, select_on)

    # Freeze the choice, then refit on everything except the test block.
    if refit_on_train_plus_validation and "validation" in dataset.splits:
        fit_blocks = ["train", "validation"]
        X_fit = pd.concat([dataset.split(b).X for b in fit_blocks])
        y_fit = pd.concat([dataset.split(b).y for b in fit_blocks])
        logger.info("Refitting %r on train+validation (%d rows)", selected_name, len(X_fit))
        estimator = _fresh_estimator(config, dataset, selected_name)
        estimator.fit(X_fit, y_fit)
        selected.estimator = estimator
        selected.fitted_on = "+".join(fit_blocks)
        # Re-persist, otherwise the artifact on disk stays the train-only model
        # and inference would silently use a weaker estimator than the report
        # describes.  The train-only validation probabilities captured during
        # _train_one are what the threshold diagnostic uses, so overwriting the
        # artifact does not make that diagnostic in-sample.
        _dump_artifact(config, dataset, selected_name, estimator, selected.fitted_on, selected.path)

    metadata = _metadata(config, dataset, trained, selected_name, threshold)
    save_json(metadata, config.paths.models_dir / "model_metadata.json")
    save_json(
        {
            "selected_model": selected_name,
            "selection_metric_order": list(SELECTION_METRICS),
            "models": {
                name: {
                    "artifact": MODEL_FILENAMES.get(name, f"{name}.joblib"),
                    "validation": headline_metrics(t.validation_metrics),
                    "cv_aggregate": t.cv_results.get("aggregate"),
                    "fitted_on": t.fitted_on,
                }
                for name, t in trained.items()
            },
        },
        config.paths.models_dir / "model_registry.json",
    )
    return TrainingResult(
        dataset=dataset,
        models=trained,
        selected_name=selected_name,
        selected=selected,
        config=config,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- internals

def _fresh_estimator(config: Config, dataset: Dataset, name: str) -> Any:
    return build_model(name, config, dataset.feature_columns)


def _dump_artifact(
    config: Config, dataset: Dataset, name: str, estimator: Any, fitted_on: str, path: Path | None = None
) -> Path:
    """Persist one trained estimator plus the metadata needed to reuse it."""
    artifact = path or (config.paths.models_dir / MODEL_FILENAMES.get(name, f"{name}.joblib"))
    joblib.dump(
        {
            "model": estimator,
            "model_name": name,
            "feature_columns": list(dataset.feature_columns),
            "symbol": dataset.symbol,
            "interval": dataset.interval,
            "horizon_candles": dataset.horizon_candles,
            "threshold": dataset.threshold,
            "random_state": config.random_state,
            "fitted_on": fitted_on,
            "trained_at": utc_now().isoformat(),
        },
        artifact,
    )
    logger.info("Saved %s -> %s", name, artifact)
    return artifact


def _train_one(
    config: Config, dataset: Dataset, name: str, threshold: float, *, run_cv: bool
) -> TrainedModel:
    """Fit one model on train, score on validation, and optionally cross-validate."""
    train = dataset.split("train")
    estimator = _fresh_estimator(config, dataset, name)
    estimator.fit(train.X, train.y)

    metrics: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    validation_proba: np.ndarray | None = None
    if "validation" in dataset.splits:
        validation = dataset.split("validation")
        proba = np.asarray(estimator.predict_proba(validation.X))[:, 1]
        validation_proba = proba
        metrics = classification_metrics(validation.y, proba, threshold=0.5)
        calibration = calibration_metrics(validation.y, proba)

    cv_results: dict[str, Any] = {}
    if run_cv:
        cv_results = _run_cv(config, dataset, name, threshold)

    artifact = _dump_artifact(config, dataset, name, estimator, "train")

    return TrainedModel(
        name=name,
        estimator=estimator,
        feature_columns=list(dataset.feature_columns),
        validation_metrics=metrics,
        validation_calibration=calibration,
        cv_results=cv_results,
        path=artifact,
        fitted_on="train",
        validation_probabilities=validation_proba,
    )


def _run_cv(config: Config, dataset: Dataset, name: str, threshold: float) -> dict[str, Any]:
    """Purged expanding-window CV on the training block only."""
    train = dataset.split("train")
    gap = config.cv.get("gap")
    gap = config.horizon_candles if gap is None else int(gap)
    splitter = PurgedExpandingWindowSplit(
        n_splits=int(config.cv["n_splits"]),
        gap=gap,
        min_train_size=int(config.cv.get("min_train_size", 1000)),
    )
    try:
        factory = _fresh_estimator(config, dataset, name)
        results = cross_validate(factory, train.X, train.y, splitter=splitter, threshold=threshold)
    except InsufficientDataError as exc:
        logger.warning("Skipping CV for %s: %s", name, exc)
        return {"skipped": True, "reason": str(exc)}

    payload = {"folds": results["folds"], "aggregate": results["aggregate"], "splitter": splitter.describe(train.X.index)}
    save_json(payload, config.paths.models_dir / f"{name}_cv.json")

    curve = reliability_curve(results["oof"]["target"], results["oof"]["probability_up"])
    curve.to_csv(config.paths.metrics_dir / f"{name}_cv_calibration.csv", index=False)
    results["oof"].to_parquet(config.paths.metrics_dir / f"{name}_cv_oof_predictions.parquet")
    return payload


def _select_model(
    trained: Mapping[str, TrainedModel], model_names: Sequence[str], select_on: str
) -> tuple[str, TrainedModel]:
    """Pick a model by validation metric, in :data:`SELECTION_METRICS` order.

    Baselines are eligible to win.  If ``majority_baseline`` ranks first, that is
    the honest result and the report says so.
    """
    candidates = [n for n in model_names if n in trained and trained[n].validation_metrics]
    if not candidates:
        raise ValueError("No model produced validation metrics; cannot select")

    def key(name: str) -> tuple[float, ...]:
        return tuple(-trained[name].validation_score(metric) for metric in SELECTION_METRICS)

    ranked = sorted(candidates, key=key)
    for name in ranked:
        logger.info("  %-22s %s", name, headline_metrics(trained[name].validation_metrics))
    best = ranked[0]
    logger.info(
        "Selected %r by validation %s. Baselines are eligible to win; if one did, "
        "the more complex models found nothing durable.",
        best, SELECTION_METRICS[0],
    )
    return best, trained[best]


# --------------------------------------------------------------------------- metadata

def _metadata(
    config: Config,
    dataset: Dataset,
    trained: Mapping[str, TrainedModel],
    selected_name: str,
    decision_threshold: float,
) -> dict[str, Any]:
    """The full reproducibility record stored next to the artifacts."""
    splits = dataset.splits
    return {
        "trained_at": utc_now().isoformat(),
        "selected_model": selected_name,
        "selection_metrics": list(SELECTION_METRICS),
        "decision_threshold": decision_threshold,
        "decision_threshold_provenance": "config backtest.probability_threshold — fixed a priori, not tuned on test",
        "dataset": {
            "symbol": dataset.symbol,
            "interval": dataset.interval,
            "rows": int(len(dataset.frame)),
            "range_start": format_timestamp(dataset.frame.index.min()),
            "range_end": format_timestamp(dataset.frame.index.max()),
            "horizon_candles": dataset.horizon_candles,
            "target_threshold": dataset.threshold,
            "target_definition": dataset.metadata.get("target", {}).get("definition"),
            "n_features": len(dataset.feature_columns),
            "feature_columns": list(dataset.feature_columns),
            "feature_docs": {c: FEATURE_DOCS.get(c, "") for c in dataset.feature_columns},
            "split_method": "chronological (never shuffled)",
            "purge_candles": config.split.get("purge_candles") or config.horizon_candles,
            "evaluation_periods": {
                name: {
                    "start": format_timestamp(split.start),
                    "end": format_timestamp(split.end),
                    "rows": len(split),
                    "class_distribution": split.class_distribution(),
                }
                for name, split in splits.items()
            },
        },
        "models": {
            name: {
                "artifact": str(t.path) if t.path else None,
                "fitted_on": t.fitted_on,
                "parameters": model_params(t.estimator),
                "validation_metrics": headline_metrics(t.validation_metrics),
                "validation_class_distribution": t.validation_metrics.get("class_distribution"),
                "cv_aggregate": t.cv_results.get("aggregate"),
            }
            for name, t in trained.items()
        },
        "config_fingerprint": config.fingerprint(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "random_state": config.random_state,
        "disclaimers": [
            "Feature importance is not evidence of causality.",
            "probability_up is a model output, not a guaranteed probability of profit.",
            "The test period was used once, for final evaluation only; it was never tuned on.",
            "Backtest results are research simulations without slippage, funding or live execution.",
        ],
    }
