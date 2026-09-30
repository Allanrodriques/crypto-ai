"""Final evaluation: test-set metrics, calibration, backtest, plots and reports.

This is the only module that reads the test split, and it reads it **once**, after
the model has already been chosen on validation.  Nothing in here feeds a
decision that changes the model: thresholds come from config, the model is
already frozen, and every artefact is a report.

Report layout
-------------
``reports/metrics/<model>_metrics.json``      per-model test metrics + class balance + calibration
``reports/metrics/<model>_predictions.csv``  per-candle test predictions
``reports/metrics/model_comparison.json``    every model side by side
``reports/metrics/summary.md``                the human-readable summary
``reports/backtests/<model>_backtest.json``  strategy metrics
``reports/backtests/<model>_trades.csv``     the trade log
``reports/plots/*.png``                      the ten research figures
``reports/feature_importance.csv``           XGBoost importance (top 20 highlighted)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation import plots
from src.evaluation.backtest import BacktestRules, run_backtest, threshold_sensitivity
from src.evaluation.metrics import (
    calibration_metrics,
    classification_metrics,
    class_distribution,
    headline_metrics,
    reliability_curve,
)
from src.pipeline.train import SELECTION_METRICS, TrainingResult
from src.utils import format_timestamp, get_logger, save_json, utc_now

logger = get_logger("pipeline.evaluate")

#: Thresholds probed by the validation-only sensitivity diagnostic.
_SENSITIVITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)

DISCLAIMER = (
    "Research output only. `probability_up` is a model score, not a guaranteed "
    "probability of profit. Past backtest performance does not imply future results."
)


@dataclass
class EvaluationResult:
    """Per-model evaluation outputs, keyed by model name."""

    config: Config
    selected_name: str
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    artefacts: list[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for name, payload in self.per_model.items():
            test = payload.get("test", {}).get("metrics", {})
            backtest = payload.get("backtest", {}).get("metrics", {})
            rows.append(
                {
                    "model": name,
                    "test_accuracy": test.get("accuracy"),
                    "test_f1": test.get("f1"),
                    "test_roc_auc": test.get("roc_auc"),
                    "test_pr_auc": test.get("pr_auc"),
                    "brier": payload.get("test", {}).get("calibration", {}).get("brier_score"),
                    "backtest_return": backtest.get("total_return"),
                    "buy_hold_return": backtest.get("buy_hold_return"),
                    "n_trades": backtest.get("n_trades"),
                    "sharpe": backtest.get("sharpe_ratio"),
                }
            )
        return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- entry point

def evaluate_models(
    config: Config,
    result: TrainingResult,
    *,
    write_plots: bool = True,
    backtest_baselines: bool = False,
) -> EvaluationResult:
    """Evaluate every trained model on the held-out test period.

    Parameters
    ----------
    backtest_baselines:
        Also backtest the naive baselines.  Off by default because the majority
        baseline never trades, which makes its strategy statistics meaningless.
    """
    dataset = result.dataset
    if "test" not in dataset.splits:
        raise ValueError("Dataset has no test split; cannot evaluate")

    test = dataset.split("test")
    decision_threshold = float(config.backtest.get("probability_threshold", 0.60))
    rules = BacktestRules.from_config(config.backtest, horizon_candles=dataset.horizon_candles)

    evaluation = EvaluationResult(config=config, selected_name=result.selected_name)
    per_model_metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    hard_predictions: dict[str, np.ndarray] = {}

    for name, trained in result.models.items():
        proba = np.asarray(trained.estimator.predict_proba(test.X))[:, 1]
        predictions[name] = proba
        hard = (proba >= 0.5).astype(int)
        hard_predictions[name] = hard

        payload: dict[str, Any] = {
            "model": name,
            "evaluated_at": utc_now().isoformat(),
            "test_period": {
                "start": format_timestamp(test.start),
                "end": format_timestamp(test.end),
                "rows": len(test),
            },
            "test": {
                "metrics": classification_metrics(test.y, proba, threshold=0.5),
                "calibration": calibration_metrics(test.y, proba),
            },
            "validation": {
                "metrics": trained.validation_metrics,
                "calibration": trained.validation_calibration,
            },
            "cv_aggregate": trained.cv_results.get("aggregate"),
        }

        should_backtest = name == result.selected_name or backtest_baselines
        if should_backtest:
            payload["backtest"] = _run_and_store_backtest(
                config, dataset, test, proba, rules, name, evaluation, result.models
            )

        path = config.paths.metrics_dir / f"{name}_metrics.json"
        save_json(payload, path)
        evaluation.artefacts.append(str(path))

        curve = reliability_curve(test.y, proba)
        curve.to_csv(config.paths.metrics_dir / f"{name}_calibration.csv", index=False)

        pred_frame = pd.DataFrame(
            {
                "timestamp": test.frame.index,
                "probability_up": proba,
                "predicted_target": hard,
                "actual_target": test.frame["target"].to_numpy(),
                "future_return": test.frame["future_return"].to_numpy(),
                "close": test.frame["close"].to_numpy(),
            }
        )
        pred_path = config.paths.metrics_dir / f"{name}_predictions.csv"
        pred_frame.to_csv(pred_path, index=False)
        evaluation.artefacts.append(str(pred_path))

        per_model_metrics[name] = payload
        evaluation.per_model[name] = payload
        logger.info("Evaluated %-22s %s", name, headline_metrics(payload["test"]["metrics"]))

    _feature_importance(config, result, evaluation)

    comparison = {
        "generated_at": utc_now().isoformat(),
        "selected_model": result.selected_name,
        "selection_metrics": list(SELECTION_METRICS),
        "test_period": {
            "start": format_timestamp(test.start),
            "end": format_timestamp(test.end),
            "rows": len(test),
        },
        "class_distribution": class_distribution(test.y),
        "disclaimer": DISCLAIMER,
        "models": {
            name: {
                "test": headline_metrics(payload["test"]["metrics"]),
                "test_confusion_matrix": payload["test"]["metrics"]["confusion_matrix"],
                "calibration": payload["test"]["calibration"],
                "backtest": payload.get("backtest", {}).get("metrics"),
            }
            for name, payload in per_model_metrics.items()
        },
    }
    comparison_path = save_json(comparison, config.paths.metrics_dir / "model_comparison.json")
    evaluation.artefacts.append(str(comparison_path))

    if write_plots:
        evaluation.plots = _write_plots(
            config, dataset, result, test, predictions, hard_predictions, per_model_metrics
        )

    summary_path = _write_summary(config, result, evaluation)
    evaluation.artefacts.append(str(summary_path))

    return evaluation


# --------------------------------------------------------------------------- backtest

def _run_and_store_backtest(
    config: Config,
    dataset: Any,
    test: Any,
    proba: np.ndarray,
    rules: BacktestRules,
    name: str,
    evaluation: EvaluationResult,
    trained: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fixed-rule backtest on the test period and persist the artefacts."""
    frame = test.frame.copy()
    frame["probability_up"] = proba
    outcome = run_backtest(frame, rules, probability_column="probability_up", target_column="target")

    # Threshold sensitivity is a DIAGNOSTIC on validation, never a selection step,
    # and it is deliberately not run on the test period.
    sensitivity = _validation_threshold_sensitivity(config, dataset, rules, name, trained)

    payload = {
        "model": name,
        "generated_at": utc_now().isoformat(),
        "period": {"start": format_timestamp(test.start), "end": format_timestamp(test.end)},
        "rules": outcome.rules,
        "metrics": outcome.metrics,
        "threshold_sensitivity_on_validation": sensitivity,
        "threshold_sensitivity_note": (
            "Diagnostic only, computed on the validation period. The applied threshold comes from "
            "config and was NOT selected by scanning the test period."
        ),
        "interpretation": {
            "separation": (
                "Prediction performance (above) and strategy performance (here) are different "
                "questions. A model can rank outcomes well and still lose money after costs."
            ),
            "limitations": [
                "Fills are modelled at the open of the candle after the signal; no slippage model.",
                "Costs are a flat bps charge on entry and exit; no funding, borrow or market impact.",
                "Long-only, no leverage, no exchange connectivity, no live execution.",
                "Signals are independent, non-overlapping trades of a fixed horizon.",
            ],
        },
    }
    save_json(payload, config.paths.backtests_dir / f"{name}_backtest.json")
    if not outcome.trades.empty:
        outcome.trades.to_csv(config.paths.backtests_dir / f"{name}_trades.csv", index=False)
    outcome.equity_curve.to_csv(config.paths.backtests_dir / f"{name}_equity.csv")
    return payload


def _validation_threshold_sensitivity(
    config: Config, dataset: Any, rules: BacktestRules, name: str, trained: Mapping[str, Any] | None = None
) -> list[dict[str, Any]] | None:
    """Score the *validation* split at several thresholds, for inspection only.

    The probabilities come from the model as it was fitted **on the training block
    only**, captured during training.  The selected model is refit on
    train+validation before the test set is touched, so its validation scores are
    in-sample and would flatter it; the captured train-only probabilities are the
    honest basis for this diagnostic.

    Returns ``None`` when the diagnostic cannot run.  It must never influence the
    applied threshold, which comes from config.
    """
    import joblib

    from src.pipeline.train import MODEL_FILENAMES

    if "validation" not in dataset.splits:
        return None
    validation = dataset.split("validation")
    try:
        captured = None if trained is None else trained[name].validation_probabilities
        if captured is not None:
            proba = np.asarray(captured, dtype="float64")
        else:
            # Fall back to the persisted artifact, but only if it is still a
            # train-only fit; a refit artifact would make this in-sample.
            artifact = config.paths.models_dir / MODEL_FILENAMES.get(name, f"{name}.joblib")
            if not artifact.exists():
                return None
            bundle = joblib.load(artifact)
            if bundle.get("fitted_on", "train") != "train":
                logger.warning(
                    "Skipping threshold sensitivity for %s: no train-only validation "
                    "probabilities were captured and the persisted model is %s-fit",
                    name, bundle.get("fitted_on"),
                )
                return None
            proba = np.asarray(bundle["model"].predict_proba(validation.X))[:, 1]
        frame = validation.frame.copy()
        frame["probability_up"] = proba
        return threshold_sensitivity(frame, rules, _SENSITIVITY_THRESHOLDS).to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - a diagnostic must not break evaluation
        logger.warning("Threshold sensitivity diagnostic failed for %s: %s", name, exc)
        return None


# --------------------------------------------------------------------------- importance

def _feature_importance(config: Config, result: TrainingResult, evaluation: EvaluationResult) -> None:
    """Write XGBoost gain-based importance to CSV, with a causality caveat."""
    from src.models.xgboost_model import XGBModel

    trained = result.models.get("xgboost")
    if trained is None:
        logger.info("No xgboost model in this run; skipping feature importance")
        return

    estimator = trained.estimator
    model = dict(estimator.steps)["classifier"] if hasattr(estimator, "steps") else estimator
    if not isinstance(model, XGBModel):
        logger.warning("xgboost artifact is not an XGBModel; skipping importance")
        return

    frame = model.importance_frame(gain=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["is_top_20"] = frame["rank"] <= 20
    frame["interpretation"] = "model reliance, not causal effect"
    path = config.paths.reports_dir / "feature_importance.csv"
    frame.to_csv(path, index=False)
    evaluation.artefacts.append(str(path))
    logger.info("Wrote feature importance for %d features -> %s", len(frame), path)


# --------------------------------------------------------------------------- plots

def _write_plots(
    config: Config,
    dataset: Any,
    result: TrainingResult,
    test: Any,
    predictions: Mapping[str, np.ndarray],
    hard_predictions: Mapping[str, np.ndarray],
    per_model_metrics: Mapping[str, Any],
) -> list[str]:
    """Produce the ten research figures."""
    directory = config.paths.plots_dir
    selected = result.selected_name
    proba = predictions[selected]
    paths: list[Path] = []

    paths.append(plots.plot_price_history(dataset.frame, directory, symbol=dataset.symbol, interval=dataset.interval))
    paths.append(plots.plot_split_overview(dataset.frame, dataset.splits, directory, symbol=dataset.symbol))
    paths.append(
        plots.plot_feature_distributions(
            dataset.frame, dataset.splits, dataset.feature_columns, directory, top_n=8
        )
    )
    paths.append(
        plots.plot_confusion_matrix(
            test.y, hard_predictions[selected], directory,
            title=f"Confusion matrix — {selected} (test)", name=f"confusion_matrix_{selected}",
        )
    )
    paths.append(
        plots.plot_roc_curve(
            test.y, proba, directory,
            title=f"ROC — {selected} (test)", name=f"roc_curve_{selected}",
            comparison={n: p for n, p in predictions.items() if n in {"logistic_regression", "random_forest", "xgboost"}},
        )
    )
    paths.append(
        plots.plot_precision_recall_curve(
            test.y, proba, directory,
            title=f"Precision-Recall — {selected} (test)", name=f"precision_recall_curve_{selected}",
            comparison={n: p for n, p in predictions.items() if n in {"logistic_regression", "random_forest", "xgboost"}},
        )
    )

    prob_frame = test.frame.copy()
    prob_frame["probability_up"] = proba
    paths.append(plots.plot_probability_over_time(prob_frame, directory, threshold=0.5,
                                                  title=f"Predicted probability over time — {selected} (test)"))

    backtest_payload = per_model_metrics.get(selected, {}).get("backtest")
    if backtest_payload:
        equity = _equity_from_trades(config, selected, backtest_payload)
        if equity is not None:
            paths.append(plots.plot_equity_curve(equity, backtest_payload["metrics"], directory,
                                                 title=f"Equity curve — {selected} (test, net of costs)"))
            paths.append(plots.plot_drawdown_curve(equity, directory, title=f"Drawdown — {selected} (test)"))

    importance_csv = config.paths.reports_dir / "feature_importance.csv"
    if importance_csv.exists():
        importance = pd.read_csv(importance_csv)
        paths.append(plots.plot_feature_importance(importance, directory, top_n=20))

    logger.info("Wrote %d plot(s) to %s", len(paths), directory)
    return [str(p) for p in paths]


def _equity_from_trades(config: Config, name: str, payload: Mapping[str, Any]) -> pd.DataFrame | None:
    equity_path = config.paths.backtests_dir / f"{name}_equity.csv"
    if not equity_path.exists():
        return None
    frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame


# --------------------------------------------------------------------------- summary

def _write_summary(config: Config, result: TrainingResult, evaluation: EvaluationResult) -> Path:
    """Render the human-readable Markdown summary."""
    dataset = result.dataset
    selected = evaluation.per_model[evaluation.selected_name]
    test = dataset.split("test")
    train = dataset.split("train")
    lines: list[str] = []

    lines.append("# Evaluation summary\n")
    lines.append(f"> {DISCLAIMER}\n")
    lines.append("## Run\n")
    lines.append(f"- Generated: {utc_now().isoformat()}")
    lines.append(f"- Symbol / interval: **{dataset.symbol} {dataset.interval}**")
    lines.append(f"- Data range: {format_timestamp(dataset.frame.index.min())} -> {format_timestamp(dataset.frame.index.max())}")
    lines.append(f"- Target: `1` if `close[t+{dataset.horizon_candles}]/close[t] - 1 >= {dataset.threshold}` else `0`")
    lines.append(f"- Features: {len(dataset.feature_columns)} (all causal, no future data)")
    lines.append(f"- Split: chronological, purge = {config.split.get('purge_candles') or dataset.horizon_candles} candles, never shuffled")
    lines.append(f"- Random state: {config.random_state}\n")

    lines.append("## Periods and class balance\n")
    lines.append("| Period | Rows | Start | End | Share UP | Majority share |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: |")
    for name in ("train", "validation", "test"):
        if name not in dataset.splits:
            continue
        split = dataset.splits[name]
        dist = split.class_distribution()
        share = f"{dist['share_up']:.1%}" if dist.get("share_up") is not None else "n/a"
        majority = f"{max(dist['share_up'] or 0, dist['n_up'] and (1 - dist['share_up']) or 0):.1%}"
        lines.append(f"| {name} | {len(split):,} | {format_timestamp(split.start)[:10]} | "
                     f"{format_timestamp(split.end)[:10]} | {share} | {majority} |")
    lines.append("")

    lines.append("## Test-set performance (touched once, after model selection)\n")
    lines.append("| Model | Accuracy | F1 | ROC-AUC | PR-AUC | Brier | ECE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, payload in evaluation.per_model.items():
        metrics = payload["test"]["metrics"]
        cal = payload["test"]["calibration"]
        fmt = lambda v, spec=".4f": "n/a" if v is None else format(float(v), spec)  # noqa: E731
        lines.append(
            f"| {name} | {fmt(metrics['accuracy'])} | {fmt(metrics['f1'])} | {fmt(metrics['roc_auc'])} | "
            f"{fmt(metrics['pr_auc'])} | {fmt(cal['brier_score'])} | {fmt(cal['expected_calibration_error'])} |"
        )
    lines.append("")
    lines.append(f"Selected model: **{result.selected_name}** (chosen on validation "
                 f"{result.metadata['selection_metrics'][0]}, then refit on train+validation).\n")

    cm = selected["test"]["metrics"]["confusion_matrix"]
    lines.append("## Confusion matrix — selected model (test)\n")
    lines.append("| | predicted DOWN | predicted UP |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| **actual DOWN** | {cm['true_negative']:,} | {cm['false_positive']:,} |")
    lines.append(f"| **actual UP** | {cm['false_negative']:,} | {cm['true_positive']:,} |")
    lines.append("")

    backtest = selected.get("backtest", {}).get("metrics")
    if backtest:
        lines.append("## Strategy backtest (test, net of costs) — separate from model quality\n")
        lines.append(f"Rule: enter long when `probability_up >= "
                     f"{config.backtest.get('probability_threshold')}`; fill at the next candle's open; "
                     f"hold {dataset.horizon_candles} candles. Cost {config.backtest.get('transaction_cost_bps')} bps per side.\n")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, spec in (
            ("initial_capital", ",.2f"), ("final_capital", ",.2f"), ("total_return", "+.2%"),
            ("buy_hold_return", "+.2%"), ("excess_vs_buy_hold", "+.2%"), ("max_drawdown", ".2%"),
            ("n_trades", ",.0f"), ("win_rate", ".1%"), ("avg_trade_return", "+.4%"),
            ("sharpe_ratio", ".3f"), ("costs_paid", ",.2f"),
        ):
            value = backtest.get(key)
            lines.append(f"| {key} | {'n/a' if value is None else format(float(value), spec)} |")
        lines.append("")

    lines.append("## Interpretation limits\n")
    lines.append("- Feature importance reflects model reliance, not causality.")
    lines.append("- `probability_up` is a model score; calibration is measured against the observed base rate, and neither implies profit.")
    lines.append("- The test period was used once, for reporting. No threshold, hyper-parameter or feature set was tuned on it.")
    lines.append("- The backtest is a simulation: flat costs, no slippage, no funding, no market impact, no live execution.")
    lines.append("- A high accuracy on an imbalanced target can come from predicting one class; read it next to ROC-AUC and the class balance above.")

    text = "\n".join(lines) + "\n"
    path = config.paths.metrics_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote summary -> %s", path)
    return path
