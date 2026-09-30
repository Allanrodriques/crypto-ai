"""Walk-forward evaluation harness: the keystone of the V3 pipeline.

This module owns the one thing that cannot be delegated - the *order* in which
data is allowed to touch a model.  Everything else in V3 is a leaf that consumes
its output.

The per-horizon loop
--------------------
Each horizon gets its own splitter, its own hyperparameter search and its own
set of models.  Nothing is ever shared across horizons: a 1d model is trained
and scored on 1d labels, a 180d model on 180d labels.  Reusing a short-horizon
model to "predict" a long one would silently change the estimand, so it is not
done anywhere in V3.

Inside one fold
---------------
1. Fit every candidate on **train** only.
2. Score candidates on **validation** only, and pick the winner per model
   family.  The test block is not touched and not even loaded.
3. Refit the winner on train+validation.  This is safe because the splitter
   already guarantees ``validation_end + horizon <= test_start``, so the
   combined set cannot reach into the test period.
4. Build the split-conformal band from the *validation* residuals of the
   train-only fit.  Those residuals are genuinely out-of-sample, which is what
   makes the resulting interval's coverage claim meaningful.
5. Fit the direction classifier on train+validation features against the sign of
   realised returns.  This is a genuinely separate model, not a thresholded
   regression, so the reported probability is not a relabelled point forecast.
6. Predict the test block and record it.

Steps 4 and 5 are why this harness owns the loop: conformal calibration and
probability calibration both have a strict "which split am I allowed to use"
requirement, and getting that wrong is the single easiest way to publish an
over-confident model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.utils import get_logger
from src.v3.horizons import Horizon
from src.v3.metrics import evaluate_regression
from src.v3.models import MODEL_NAMES, build_model, fit_model, predict_model
from src.v3.splits import PurgedWalkForwardSplitter, assert_no_overlap
from src.v3.uncertainty import (
    CalibratedDirectionModel,
    SplitConformal,
    empirical_probability_positive,
    fit_split_conformal,
    interval_summary,
)

logger = get_logger("v3.walkforward")


class WalkForwardError(RuntimeError):
    """Raised when a fold cannot be evaluated honestly."""


# Small, fixed grids.  Deliberately tiny: a large grid selected on a
# 3,000-row validation block is a bigger overfitting risk than it is a gain.
HYPERPARAMETER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "ridge": [{"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0}],
    "random_forest": [{"n_estimators": 300}],
    "xgboost": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    ],
}
BASELINES = ("zero", "mean", "trailing_mean")

# The direction classifier is a function of (features, sign of the realised
# return) - it does not depend on which regressor produced the point forecast.
# Fitting it once per fold rather than once per model is the difference between
# 6 fits and 30, which is what made an earlier version of this harness
# unbearably slow.  Method is 'sigmoid' by default because isotonic needs far
# more calibration rows than a 3-month block contains to avoid overfitting.
DIRECTION_METHOD = "sigmoid"
DIRECTION_CV = 3


@dataclass(frozen=True)
class FoldResult:
    """Everything one fold produced, for one horizon and one model."""

    horizon: str
    fold: int
    model: str
    n_train: int
    n_validation: int
    n_test: int
    params: dict[str, Any]
    validation_score: float
    test_score: float
    metrics: dict[str, float]
    conformal: SplitConformal | None
    interval: dict[str, float]
    seconds: float

    def to_row(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "fold": self.fold,
            "model": self.model,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "validation_score": self.validation_score,
            "test_score": self.test_score,
            "params": repr(self.params),
            "seconds": self.seconds,
            **{f"metric_{k}": v for k, v in self.metrics.items()},
            **{f"interval_{k}": v for k, v in self.interval.items()},
        }


@dataclass(frozen=True)
class HorizonResult:
    """Pooled out-of-sample results for one symbol and one horizon."""

    symbol: str
    horizon: str
    horizon_days: float
    feature_columns: list[str]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    interval_summary: pd.DataFrame
    geometry: dict[str, Any]
    coverage: dict[str, Any]
    seconds: float
    models: list[str] = field(default_factory=list)
    # Trailing series used to label market regimes for the analysis and report
    # modules, which cannot derive labels from predictions alone.  Deliberately
    # not part of the scored payload: it is a label, never an input, and holding
    # it here keeps ``src.v3.report`` from having to invent one from the target.
    regime_context: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def target_column(self) -> str:
        return f"future_return_{self.horizon}"

    def best_model(self, metric: str = "rmse") -> str:
        if self.pooled_metrics.empty:
            raise WalkForwardError(f"No pooled metrics for {self.symbol}/{self.horizon}")
        return str(self.pooled_metrics.sort_values(metric).index[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "n_predictions": int(len(self.predictions)),
            "n_features": len(self.feature_columns),
            "models": self.models,
            "seconds": self.seconds,
            "geometry": self.geometry,
            "coverage": self.coverage,
            "pooled_metrics": self.pooled_metrics.reset_index().to_dict("records"),
            "best_model": self.best_model(),
        }


@dataclass(frozen=True)
class V3RunResult:
    """The full run: every horizon, every symbol."""

    symbol: str
    horizons: list[str]
    feature_columns: list[str]
    results: list[HorizonResult]
    dataset: dict[str, Any]
    config: dict[str, Any]
    seconds: float

    def for_horizon(self, horizon: str) -> HorizonResult:
        for result in self.results:
            if result.horizon == horizon:
                return result
        raise WalkForwardError(f"Horizon {horizon!r} not present in run for {self.symbol}")

    def horizon_table(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            best = result.best_model()
            metrics = result.pooled_metrics.loc[best]
            rows.append(
                {
                    "symbol": result.symbol,
                    "horizon": result.horizon,
                    "horizon_days": result.horizon_days,
                    "best_model": best,
                    "n_predictions": len(result.predictions),
                    **{f"best_{k}": v for k, v in metrics.items()},
                }
            )
        return pd.DataFrame(rows)

    def model_table(self) -> pd.DataFrame:
        frames = []
        for result in self.results:
            frame = result.pooled_metrics.reset_index().rename(columns={"index": "model"})
            frame.insert(0, "horizon", result.horizon)
            frame.insert(0, "symbol", result.symbol)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- helpers
def _columns_for_models(models: Sequence[str], suffix: str) -> list[str]:
    return [f"{m}{suffix}" for m in models]


def _finite_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all(axis=1)


def _select_params(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], float, BaseEstimator]:
    """Grid-search on validation only.

    Returns ``(best_params, best_rmse, fitted_train_only_model)``.  The fitted
    model is returned rather than discarded because it is *already* the
    train-only fit the conformal band needs - refitting it would be a pure
    waste, and for a 300-tree forest that waste is measured in minutes per fold.
    """
    grid = HYPERPARAMETER_GRIDS.get(model_name)
    if not grid:
        model = build_model(model_name, seed=seed)
        return {}, float("nan"), fit_model(model, X_train, y_train)

    best_params: dict[str, Any] = grid[0]
    best_score = np.inf
    best_model: BaseEstimator | None = None
    for params in grid:
        try:
            model = fit_model(build_model(model_name, seed=seed, **params), X_train, y_train)
            predictions = predict_model(model, X_valid)
            mask = np.isfinite(y_valid) & np.isfinite(predictions)
            if not mask.any():
                continue
            score = float(np.sqrt(np.mean((y_valid[mask] - predictions[mask]) ** 2)))
        except Exception as exc:  # a bad hyperparameter should not kill the run
            logger.warning("grid point %s for %s failed: %s", params, model_name, exc)
            continue
        if score < best_score:
            best_score, best_params, best_model = score, params, model
    if best_model is None or not np.isfinite(best_score):
        raise WalkForwardError(
            f"No usable validation score for {model_name}; check the validation block is populated"
        )
    return best_params, best_score, best_model


# ------------------------------------------------------------------ one horizon
def run_horizon(
    frame: pd.DataFrame,
    horizon: Horizon,
    splitter: PurgedWalkForwardSplitter,
    feature_columns: Sequence[str],
    symbol: str = "",
    model_names: Sequence[str] = MODEL_NAMES,
    seed: int = 42,
    alpha: float = 0.1,
) -> HorizonResult:
    """Walk forward one horizon and pool the out-of-sample test predictions.

    The labelled rows are derived from the target column's own NaN mask rather
    than from a :class:`FutureReturnTargets` object, so this harness depends only
    on the frame that ``build_v3_dataset`` returns and not on that module's
    internals.
    """
    started = time.perf_counter()
    target_column = f"future_return_{horizon.label}"
    if target_column not in frame:
        raise WalkForwardError(f"Target column {target_column!r} missing from frame")

    features = [c for c in feature_columns if c in frame.columns]
    labelled = frame.loc[frame[target_column].notna()].copy()
    if labelled.empty:
        raise WalkForwardError(
            f"No labelled rows for horizon {horizon.label}; the target window exceeds the data"
        )

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, np.ndarray]] = {
        m: {"y": [], "pred": [], "lo": [], "hi": [], "prob_empirical": [], "prob_calibrated": []}
        for m in model_names
    }

    n_folds = 0
    for fold in splitter.split(labelled.index):
        assert_no_overlap(fold)
        n_folds += 1

        train = labelled.iloc[fold.train]
        valid = labelled.iloc[fold.validation]
        test = labelled.iloc[fold.test]

        X_train, y_train = train[features], train[target_column].to_numpy(dtype=float)
        X_valid, y_valid = valid[features], valid[target_column].to_numpy(dtype=float)
        X_test, y_test = test[features], test[target_column].to_numpy(dtype=float)

        # Refit set: train + validation, already proven disjoint from test.
        X_fit = pd.concat([X_train, X_valid], axis=0)
        y_fit = np.concatenate([y_train, y_valid])

        block = pd.DataFrame(
            {"target": y_test}, index=test.index, columns=["target"]
        )

        # Method B: a genuinely separate, calibrated direction model.  Fitted
        # ONCE per fold on train+validation, never on test, and independent of
        # which regressor is being scored.
        try:
            direction = CalibratedDirectionModel(
                method=DIRECTION_METHOD, random_state=seed
            ).fit(X_fit, (y_fit > 0).astype(int))
            direction_prob = direction.predict_proba_positive(X_test)
        except Exception as exc:
            logger.debug(
                "%s fold %d: direction calibration unavailable (%s)", horizon.label, fold.fold, exc
            )
            direction_prob = np.full(len(test), np.nan)
        block["prob_calibrated"] = direction_prob

        for model_name in model_names:
            fold_started = time.perf_counter()
            try:
                params, validation_score, train_only = _select_params(
                    model_name, X_train, y_train, X_valid, y_valid, seed
                )
                valid_pred = predict_model(train_only, X_valid)
            except WalkForwardError:
                if model_name in BASELINES:
                    params, validation_score = {}, float("nan")
                    try:
                        train_only = fit_model(build_model(model_name, seed=seed), X_train, y_train)
                        valid_pred = predict_model(train_only, X_valid)
                    except Exception as exc:
                        logger.warning(
                            "%s/%s fold %d: baseline fit failed (%s), skipping",
                            horizon.label, model_name, fold.fold, exc,
                        )
                        continue
                else:
                    logger.warning(
                        "%s/%s fold %d: no validation score, skipping model",
                        horizon.label,
                        model_name,
                        fold.fold,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "%s/%s fold %d: fit failed (%s), skipping", horizon.label, model_name, fold.fold, exc
                )
                continue

            # `train_only` is fitted on TRAIN only, so its validation residuals
            # are genuinely out-of-sample - which is what makes the conformal
            # band below an honest interval rather than a fitted one.
            residual_mask = np.isfinite(y_valid) & np.isfinite(valid_pred)
            conformal = None
            if residual_mask.sum() >= 20:
                conformal = fit_split_conformal(
                    y_valid[residual_mask], valid_pred[residual_mask], alpha=alpha
                )

            # --- refit on train+validation with the selected hyperparameters
            if model_name in BASELINES:
                final = build_model(model_name, seed=seed)
            else:
                final = build_model(model_name, seed=seed, **params)
            try:
                final = fit_model(final, X_fit, y_fit)
                test_pred = predict_model(final, X_test)
            except Exception as exc:
                logger.warning(
                    "%s/%s fold %d: refit failed (%s), skipping",
                    horizon.label,
                    model_name,
                    fold.fold,
                    exc,
                )
                continue

            block[f"{model_name}_pred"] = test_pred
            if conformal is not None:
                lower, upper = conformal.interval(test_pred)
                block[f"{model_name}_lo"] = lower
                block[f"{model_name}_hi"] = upper
            else:
                block[f"{model_name}_lo"] = np.nan
                block[f"{model_name}_hi"] = np.nan

            # Method A: empirical CDF of this model's own out-of-sample residual
            # distribution.  This one IS per model, because residuals are.
            if residual_mask.sum() >= 20:
                block[f"{model_name}_prob_empirical"] = empirical_probability_positive(
                    test_pred, y_valid[residual_mask] - valid_pred[residual_mask]
                )
            else:
                block[f"{model_name}_prob_empirical"] = np.full(len(test), np.nan)

            metrics = evaluate_regression(y_test, test_pred)
            summary = (
                interval_summary(y_test, test_pred, conformal)
                if conformal is not None
                else {"empirical_coverage": np.nan, "nominal_coverage": 1 - alpha,
                      "mean_interval_width": np.nan, "n": int(len(y_test))}
            )
            fold_rows.append(
                FoldResult(
                    horizon=horizon.label,
                    fold=fold.fold,
                    model=model_name,
                    n_train=int(fold.train.size),
                    n_validation=int(fold.validation.size),
                    n_test=int(fold.test.size),
                    params=params,
                    validation_score=validation_score,
                    test_score=float(metrics.get("rmse", np.nan)),
                    metrics=metrics,
                    conformal=conformal,
                    interval=summary,
                    seconds=time.perf_counter() - fold_started,
                ).to_row()
            )

            pooled[model_name]["y"].append(y_test)
            pooled[model_name]["pred"].append(test_pred)

        prediction_frames.append(block)

    if not fold_rows:
        raise WalkForwardError(
            f"Horizon {horizon.label}: no model produced a usable fold. "
            "Check the purge/embargo geometry against the labelled span."
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="last")]

    pooled_rows: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        arrays = pooled[model_name]
        if not arrays["y"]:
            continue
        y = np.concatenate(arrays["y"])
        p = np.concatenate(arrays["pred"])
        row = evaluate_regression(y, p)
        pred_col = f"{model_name}_pred"
        if pred_col in predictions:
            mask = predictions[pred_col].notna().to_numpy()
            y_pooled = predictions["target"].to_numpy()[mask]
            row["n"] = int(mask.sum())
            for suffix, key in (("_lo", "coverage_lo"), ("_hi", "coverage_hi")):
                cols = (f"{model_name}{suffix}", pred_col)
                if all(c in predictions for c in cols):
                    lo = predictions[f"{model_name}_lo"].to_numpy()[mask]
                    hi = predictions[f"{model_name}_hi"].to_numpy()[mask]
                    inside = (y_pooled >= lo) & (y_pooled <= hi)
                    row[key] = float(np.mean(inside)) if inside.any() else np.nan
            for suffix, key in (("_prob_empirical", "brier_empirical"),):
                col = f"{model_name}{suffix}"
                if col in predictions:
                    prob = predictions[col].to_numpy()[mask]
                    ok = np.isfinite(prob)
                    if ok.any():
                        row[key] = float(np.mean((prob[ok] - (y_pooled[ok] > 0).astype(float)) ** 2))
            # The direction model's probability is shared across models by
            # construction, so it is recorded once per horizon, not per model.
            if "prob_calibrated" in predictions:
                prob = predictions["prob_calibrated"].to_numpy()[mask]
                ok = np.isfinite(prob)
                if ok.any():
                    row["brier_calibrated"] = float(
                        np.mean((prob[ok] - (y_pooled[ok] > 0).astype(float)) ** 2)
                    )
        row.setdefault("n", int(len(y)))
        pooled_rows[model_name] = row

    pooled_metrics = pd.DataFrame.from_dict(pooled_rows, orient="index")
    pooled_metrics.index.name = "model"

    interval_summary_frame = (
        fold_metrics[fold_metrics["interval_empirical_coverage"].notna()]
        .groupby("model")[["interval_empirical_coverage", "interval_nominal_coverage",
                           "interval_mean_interval_width", "interval_n"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "interval_empirical_coverage": "mean_fold_coverage",
                "interval_nominal_coverage": "nominal_coverage",
                "interval_mean_interval_width": "mean_interval_width",
                "interval_n": "mean_calibration_rows",
            }
        )
    )

    coverage = {
        "labelled_rows": int(len(labelled)),
        "n_folds": n_folds,
        "n_features": len(features),
        "prediction_rows": int(len(predictions)),
        "label_span": [str(labelled.index[0]), str(labelled.index[-1])],
        "prediction_span": [str(predictions.index[0]), str(predictions.index[-1])],
    }

    return HorizonResult(
        symbol=symbol,
        horizon=horizon.label,
        horizon_days=horizon.days,
        feature_columns=features,
        predictions=predictions,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        interval_summary=interval_summary_frame,
        geometry=splitter.describe(labelled.index),
        coverage=coverage,
        seconds=time.perf_counter() - started,
        models=[m for m in model_names if m in pooled_rows],
    )


def run_walk_forward(
    dataset: Any,
    horizons: Sequence[Horizon],
    model_names: Sequence[str] = MODEL_NAMES,
    seed: int = 42,
    n_splits: int = 4,
    test_fraction: float = 0.08,
    validation_fraction: float = 0.08,
    embargo: str | None = "horizon",
    alpha: float = 0.1,
) -> V3RunResult:
    """Run every horizon end to end and return the pooled results."""
    started = time.perf_counter()
    frame = dataset.frame
    symbol = getattr(dataset, "symbol", "")
    results: list[HorizonResult] = []

    for horizon in horizons:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=n_splits,
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
            embargo=embargo,
        )
        logger.info(
            "walk-forward %s: purge=%s embargo=%s", horizon.label, splitter.purge, splitter.embargo_delta
        )
        results.append(
            run_horizon(
                frame=frame,
                horizon=horizon,
                splitter=splitter,
                feature_columns=dataset.feature_columns,
                symbol=symbol,
                model_names=model_names,
                seed=seed,
                alpha=alpha,
            )
        )

    return V3RunResult(
        symbol=symbol,
        horizons=[h.label for h in horizons],
        feature_columns=list(dataset.feature_columns),
        results=results,
        dataset=dataset.describe(),
        config={
            "models": list(model_names),
            "n_splits": n_splits,
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "embargo": embargo,
            "alpha": alpha,
            "seed": seed,
        },
        seconds=time.perf_counter() - started,
    )
