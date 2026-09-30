"""Run one experiment end to end and write every artefact it produces.

A single entry point, because partial runs are how leaked results get reported
---------------------------------------------------------------------------
Each experiment must execute the same sequence or its numbers are not
comparable to the baseline:

1. build the dataset from the spec's feature groups and target;
2. train on the training block, select on validation, **never on test**;
3. refit the selected model on train+validation;
4. score the held-out test block once;
5. roll forward over several time-disjoint windows for robustness;
6. break the test results down by market regime;
7. rank features by gain / permutation / SHAP;
8. simulate the fixed threshold on the test block, against buy-and-hold;
9. write metrics, predictions, trades, importances and a run manifest.

The test block is scored exactly once.  Anything that inspects test results and
then changes the model, features or threshold belongs in the validation block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset.factory import build_dataset
from src.dataset.multisource import ExperimentContext
from src.evaluation.backtest import BacktestRules, run_backtest
from src.evaluation.importance import importance_bundle
from src.evaluation.metrics import classification_metrics, positive_class_proba
from src.evaluation.regimes import (
    RegimeConfig,
    evaluate_by_regime,
    label_regimes,
    regime_distribution,
)
from src.evaluation.walkforward import (
    WalkForwardPlan,
    run_walk_forward,
    summarise_windows,
    windows_frame,
)
from src.experiments.spec import ExperimentSpec
from src.experiments.threshold import ThresholdSelection, select_threshold
from src.models import build_model, model_params
from src.pipeline.train import train_models
from src.utils import get_logger, save_json, set_global_seed, utc_now

logger = get_logger("experiments.runner")

#: Metrics a walk-forward window reports, kept to a small stable set.
WALKFORWARD_METRICS = ("roc_auc", "pr_auc", "f1", "brier")


@dataclass
class ExperimentResult:
    """Everything one experiment produced."""

    spec: ExperimentSpec
    dataset_metadata: dict[str, Any]
    metrics: dict[str, Any]
    walkforward: dict[str, Any]
    regimes: dict[str, Any]
    importance: dict[str, Any]
    backtest: dict[str, Any]
    artifacts: dict[str, str]
    output_dir: str
    warnings: list[str] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        value = self.metrics.get("n_features")
        return int(value) if value is not None else 0

    @property
    def dataset_rows(self) -> int:
        value = self.dataset_metadata.get("n_rows")
        return int(value) if value is not None else 0

    @property
    def feature_manifest_hash(self) -> str:
        """Content hash of the exact feature list this experiment trained on.

        Two runs are only comparable if their feature manifests match, so this
        hash is the identity of the experiment's inputs.
        """
        return str(self.dataset_metadata.get("feature_manifest_hash", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.spec.experiment_id,
            "name": self.spec.name,
            "hypothesis": self.spec.hypothesis,
            "feature_groups": self.spec.feature_groups,
            "n_features": self.metrics.get("n_features"),
            "spec": self.spec.to_dict(),
            "dataset": self.dataset_metadata,
            "metrics": self.metrics,
            "walkforward": self.walkforward,
            "regimes": self.regimes,
            "importance": {k: v for k, v in self.importance.items() if k != "caveat"},
            "backtest": self.backtest,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "interpretation_caveat": (
                "Metrics are measurements, not conclusions. A positive out-of-sample "
                "delta on one symbol and one period is a hypothesis for further testing, "
                "not evidence of a tradable edge."
            ),
        }


def _walkforward_metric_fn():
    def metric_fn(y_true, y_proba) -> dict[str, float]:
        full = classification_metrics(y_true, positive_class_proba(y_proba))
        return {k: full[k] for k in WALKFORWARD_METRICS if k in full and full[k] is not None}

    return metric_fn


def run_experiment(
    spec: ExperimentSpec,
    config,
    *,
    context: ExperimentContext,
    output_dir: str | Path,
    walkforward_plan: WalkForwardPlan | None = None,
    with_shap: bool = True,
    with_importance: bool = True,
    run_backtest_simulation: bool = True,
    enabled_features: dict[str, list[str]] | None = None,
    threshold_objective: str = "excess_vs_buy_hold",
    threshold_min_trades: int = 30,
) -> ExperimentResult:
    """Execute ``spec`` and persist its artefacts under ``output_dir``."""
    set_global_seed(int(getattr(config, "random_state", 42)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    artifacts: dict[str, str] = {}

    # The context already holds every aligned source, so an experiment only
    # computes the feature groups it actually names.
    feature_matrix = context.features(spec.feature_groups, enabled_features=enabled_features)
    missing = [s for s in spec.data_sources if s not in _loaded_sources(context)]
    if missing:
        warnings.append(f"requested sources absent from the shared context: {missing}")

    dataset = build_dataset(
        feature_matrix,
        context.spot,
        spec.target,
        spec.split,
        symbol=context.symbol,
        interval=context.interval,
        feature_groups=list(spec.feature_groups),
        data_sources=list(spec.data_sources),
    )

    if not len(dataset.frame):
        raise ValueError(
            f"{spec.experiment_id}: dataset is empty for groups {spec.feature_groups}"
        )
    if dataset.frame["target"].nunique() < 2:
        warnings.append("target has a single class in this feature set")
        logger.warning("%s: degenerate target", spec.experiment_id)

    # ---- train / select on validation / refit ------------------------------
    training = train_models(
        config,
        dataset,
        model_names=[spec.model_name],
        run_cv=bool(getattr(config, "run_cross_validation", False)),
    )
    selected = training.selected
    estimator = selected.estimator

    # ---- score the test block exactly once ---------------------------------
    test_split = dataset.splits["test"]
    test_proba = _predict_proba(estimator, test_split.X)
    pos_proba = positive_class_proba(test_proba)
    test_metrics = classification_metrics(test_split.y, pos_proba)
    train_metrics = classification_metrics(
        dataset.splits["train"].y,
        positive_class_proba(_predict_proba(estimator, dataset.splits["train"].X)),
    )

    metrics: dict[str, Any] = {
        "n_features": len(dataset.feature_columns),
        "feature_columns": dataset.feature_columns,
        "n_rows": int(len(dataset.frame)),
        "n_train": int(len(dataset.splits["train"])),
        "n_validation": int(len(dataset.splits["validation"])),
        "n_test": int(len(test_split)),
        "selected_model": spec.model_name,
        "selection_rule": "highest validation roc_auc; test block never used for selection",
        "validation_metrics": selected.validation_metrics,
        "test_metrics": test_metrics,
        "train_metrics_refit": train_metrics,
        "test_positive_rate": float(test_split.y.mean()),
        "model_params": model_params(estimator),
    }

    # ---- walk-forward robustness ------------------------------------------
    # Windows are laid out inside train+validation only.  Rolling forward over
    # the test block would report a number that has already been used once for
    # scoring, and would double-count those months in the spread.
    plan = walkforward_plan or _default_walkforward_plan(spec, len(dataset.frame))
    try:
        windows = run_walk_forward(
            _pooled_features(dataset),
            _pooled_labels(dataset),
            lambda: build_model(spec.model_name, config, dataset.feature_columns),
            plan,
            metric_fn=_walkforward_metric_fn(),
        )
        walkforward = summarise_windows(windows)
        windows_frame(windows).to_csv(out / "walkforward_windows.csv", index=False)
        artifacts["walkforward_windows"] = str(out / "walkforward_windows.csv")
    except Exception as exc:
        logger.warning("%s: walk-forward failed: %s", spec.experiment_id, exc)
        warnings.append(f"walk-forward failed: {exc}")
        walkforward = {"error": str(exc), "plan": plan.to_dict()}

    # ---- regime breakdown --------------------------------------------------
    close = context.spot.loc[dataset.frame.index, "close"]
    regimes = label_regimes(close, config=RegimeConfig(), interval=spec.interval)
    test_regime_metrics = evaluate_by_regime(
        test_split.y, pos_proba, regimes, classification_metrics, index=test_split.frame.index
    )
    regime_block = {
        "definitions": RegimeConfig().to_dict(),
        "test_distribution": regime_distribution(regimes.reindex(test_split.frame.index)),
        "test_metrics_by_regime": test_regime_metrics,
    }

    # ---- feature importance ------------------------------------------------
    if with_importance:
        try:
            importance = importance_bundle(
                estimator,
                test_split.X,
                test_split.y,
                with_shap=with_shap,
                random_state=int(getattr(config, "random_state", 42)),
            )
            pd.DataFrame(importance.get("permutation") or []).to_csv(
                out / "feature_importance_permutation.csv", index=False
            )
            pd.DataFrame(importance.get("gain") or []).to_csv(
                out / "feature_importance_gain.csv", index=False
            )
            if importance.get("shap"):
                pd.DataFrame(importance["shap"]).to_csv(out / "feature_importance_shap.csv", index=False)
        except Exception as exc:
            logger.warning("%s: importance failed: %s", spec.experiment_id, exc)
            warnings.append(f"importance failed: {exc}")
            importance = {"error": str(exc)}
    else:
        importance = {}

    # ---- threshold selection on validation, then the test backtest ---------
    if run_backtest_simulation:
        rules = _backtest_rules(config, dataset.horizon_candles, float(spec.backtest_threshold))

        # The probability cut is chosen on validation and frozen.  Doing this
        # per experiment with one identical procedure is what keeps a fixed 0.60
        # from silently producing zero trades for a well-calibrated tree model.
        # The cut must be chosen with probabilities from the TRAIN-ONLY fit.
        # `estimator` has been refit on train+validation, so scoring it on the
        # validation block would be in-sample: the threshold would be tuned
        # against a model that has already memorised those rows, and the
        # resulting test backtest would inherit that optimism.  The pre-refit
        # validation probabilities are captured for exactly this purpose.
        validation_split = dataset.splits["validation"]
        validation_frame = validation_split.frame.copy()
        if selected.validation_probabilities is None:
            raise RuntimeError(
                f"model {selected.name!r} has no pre-refit validation probabilities; "
                "a threshold cannot be selected without leaking validation into it"
            )
        validation_frame["probability_up"] = positive_class_proba(
            np.asarray(selected.validation_probabilities)
        )
        selection = select_threshold(
            validation_frame,
            rules,
            objective=threshold_objective,
            min_trades=threshold_min_trades,
        )
        warnings.extend(_threshold_warnings(selection))

        final_rules = BacktestRules(
            **{
                **rules.__dict__,
                "probability_threshold": selection.threshold,
            }
        ).validate()

        test_frame = test_split.frame.copy()
        test_frame["probability_up"] = pos_proba
        result = run_backtest(test_frame, final_rules)
        backtest = dict(result.metrics)
        backtest["threshold"] = final_rules.probability_threshold
        backtest["threshold_provenance"] = (
            f"selected on the validation block by {selection.objective}; "
            f"the test block never informed it"
        )
        backtest["threshold_selection"] = selection.to_dict()
        backtest["trades"] = result.trades.to_dict(orient="records")
        pd.DataFrame(selection.curve).to_csv(out / "threshold_curve_validation.csv", index=False)
        artifacts["threshold_curve"] = str(out / "threshold_curve_validation.csv")
        result.trades.to_csv(out / "trades.csv", index=False)
        pd.DataFrame(result.equity_curve).to_csv(out / "equity_curve.csv", index=False)
        artifacts["trades"] = str(out / "trades.csv")
        artifacts["equity_curve"] = str(out / "equity_curve.csv")
    else:
        backtest = {}

    # ---- predictions and manifest -----------------------------------------
    predictions = test_split.frame[["target", "future_return"]].copy() if "future_return" in test_split.frame else test_split.frame[["target"]].copy()
    predictions["probability_up"] = pos_proba
    applied_threshold = backtest.get("threshold", float(spec.backtest_threshold))
    predictions["predicted"] = (predictions["probability_up"] >= float(applied_threshold)).astype(int)
    predictions["decision_threshold"] = float(applied_threshold)
    predictions.to_csv(out / "test_predictions.csv")
    artifacts["predictions"] = str(out / "test_predictions.csv")

    result_block = ExperimentResult(
        spec=spec,
        dataset_metadata=dataset.metadata,
        metrics=metrics,
        walkforward=walkforward,
        regimes=regime_block,
        importance=importance,
        backtest={k: v for k, v in backtest.items() if k != "trades"},
        artifacts=artifacts,
        output_dir=str(out),
        warnings=warnings,
    )
    save_json(result_block.to_dict(), out / "metrics.json")
    artifacts["metrics"] = str(out / "metrics.json")
    logger.info(
        "%s done: %d features, test roc_auc=%s, walk-forward %s",
        spec.experiment_id, len(dataset.feature_columns),
        _fmt(metrics["test_metrics"].get("roc_auc")),
        _fmt(walkforward.get("roc_auc", {}).get("mean")),
    )
    return result_block


def _loaded_sources(context: ExperimentContext) -> set[str]:
    """Which of the three external sources the shared context actually carries."""
    available = {"binance_spot"}
    if getattr(context.aligned, "funding", None) is not None:
        available.add("binance_funding")
    if getattr(context.aligned, "futures", None) is not None:
        available.add("binance_futures")
    if getattr(context.aligned, "sentiment", None) is not None:
        available.add("fear_greed")
    return available


def _threshold_warnings(selection: ThresholdSelection) -> list[str]:
    """Surface a fragile or degenerate threshold choice as a run warning."""
    out: list[str] = []
    if selection.note:
        out.append(selection.note)
    if selection.stability.get("checked") and selection.stability.get("agrees") is False:
        picks = selection.stability.get("picks", {})
        out.append(
            f"validation halves chose thresholds {picks.get('first_half')} and "
            f"{picks.get('second_half')}; the choice is period-dependent"
        )
    if selection.n_trades < selection.min_trades:
        out.append(
            f"selected threshold produced only {selection.n_trades} validation trades "
            f"(floor {selection.min_trades})"
        )
    return out


def _backtest_rules(config, horizon_candles: int, threshold: float) -> BacktestRules:
    """Build backtest rules from config, with the experiment's fixed threshold.

    The threshold comes from the experiment spec, not from the config's
    tuned value and never from test results.
    """
    rules = BacktestRules.from_config(
        getattr(config, "backtest", {}) or {}, horizon_candles=horizon_candles
    )
    return BacktestRules(
        initial_capital=rules.initial_capital,
        probability_threshold=threshold,
        transaction_cost_bps=rules.transaction_cost_bps,
        slippage_bps=rules.slippage_bps,
        allow_short=rules.allow_short,
        max_concurrent_positions=rules.max_concurrent_positions,
        exit_mode=rules.exit_mode,
        exit_probability=rules.exit_probability,
        risk_free_rate_annual=rules.risk_free_rate_annual,
        periods_per_year=rules.periods_per_year,
        price_column=rules.price_column,
        horizon_candles=rules.horizon_candles,
    ).validate()


def _pooled_features(dataset) -> pd.DataFrame:
    """Train+validation features, used only for walk-forward window planning."""
    return pd.concat([dataset.splits["train"].X, dataset.splits["validation"].X])


def _pooled_labels(dataset) -> pd.Series:
    return pd.concat([dataset.splits["train"].y, dataset.splits["validation"].y])


def _default_walkforward_plan(spec: ExperimentSpec, n_rows: int) -> WalkForwardPlan:
    """A plan that fits whatever data the experiment actually produced."""
    test = max(500, int(0.12 * n_rows))
    train = max(1_000, int(0.45 * n_rows))
    gap = spec.target.horizon_candles
    n_windows = 5
    while n_windows > 2 and 2 * train + (n_windows - 1) * test + gap > n_rows:
        n_windows -= 1
    return WalkForwardPlan(
        n_windows=n_windows, train_size=train, test_size=test, step=test, gap=gap
    )


def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X))
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw
    return np.asarray(estimator.predict(X))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
