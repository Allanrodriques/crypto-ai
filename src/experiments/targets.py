"""Compare target definitions, including a three-class DOWN / FLAT / UP label.

Why the target deserves its own experiment
------------------------------------------
Everything downstream inherits the target's definition, so a target that cannot
be traded invalidates the rest of the work no matter how good the features are.
Three things are checked here rather than assumed:

*Tradedability.*  A target is only interesting if a decision rule at some
threshold produces a usable number of trades.  A label that is 99% one class can
score a respectable accuracy while being impossible to trade.
*Consistency with the horizon.*  A threshold far from the asset's typical move
produces a label that is mostly noise; one far inside it produces a label so
balanced that any model appears skilled.
*Degenerate and three-class behaviour.*  A three-class target removes the
arbitrary knife-edge at ``+threshold`` and separates "went nowhere" from
"reversed", which is usually the more useful signal - at the cost of a harder
learning problem and a majority FLAT class that inflates naive accuracy.

The three-class path is evaluated with macro-averaged metrics, never accuracy
alone, because a model predicting FLAT everywhere beats accuracy whenever FLAT
dominates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.dataset.factory import build_dataset
from src.evaluation.bootstrap import paired_block_bootstrap_auc
from src.evaluation.metrics import (
    classification_metrics,
    multiclass_metrics,
    positive_class_proba,
)
from src.models import build_model
from src.utils import get_logger, save_json, set_global_seed

logger = get_logger("experiments.targets")

#: Three-class band as a fraction of the threshold: inside it, FLAT.
DEFAULT_FLAT_BAND = 0.5


@dataclass
class TargetVariant:
    """One target definition and what it produced."""

    name: str
    description: str
    mode: str
    horizon_candles: int
    threshold: float
    n_rows: int = 0
    class_distribution: dict[str, Any] = field(default_factory=dict)
    majority_share: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
    test_balanced_accuracy: float | None = None
    delta_roc_auc_vs_binary: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool = False
    #: A one-sided "worse" flag, so a clearly negative interval is not rendered
    #: the same as an interval that merely straddles zero.
    significantly_worse: bool = False
    #: "better" / "worse" / "indistinguishable" / "reference" / "not_compared".
    #: The last two matter: a variant that was never compared must not be
    #: rendered as though a paired test found it indistinguishable.
    verdict: str = "not_compared"
    walkforward_roc_auc_mean: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "horizon_candles": self.horizon_candles,
            "threshold": self.threshold,
            "n_rows": self.n_rows,
            "class_distribution": self.class_distribution,
            "majority_share": self.majority_share,
            "test_roc_auc": self.test_roc_auc,
            "test_pr_auc": self.test_pr_auc,
            "test_accuracy": self.test_accuracy,
            "test_macro_f1": self.test_macro_f1,
            "test_balanced_accuracy": self.test_balanced_accuracy,
            "delta_roc_auc_vs_binary": self.delta_roc_auc_vs_binary,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant_at_95": self.significant,
            "significantly_worse_at_95": self.significantly_worse,
            "verdict": self.verdict,
            "walkforward_roc_auc_mean": self.walkforward_roc_auc_mean,
            "notes": self.notes,
        }


def _fit(X_train, y_train, X_val, y_val, spec, config, feature_columns, seed, *, multiclass: bool):
    """Fit the model, selecting the XGBoost objective when the target has 3 classes.

    A single early-stopped internal split is used here rather than V1's full
    purged CV, because this module runs many target variants and CV on each would
    dominate the runtime.  That is a deliberate speed/robustness trade, and it is
    recorded in the report rather than hidden.
    """
    set_global_seed(seed)
    model = build_model(spec.model_name, config, feature_columns)

    if multiclass and hasattr(model, "params"):
        # The wrapper builds XGBClassifier inside fit(), so the objective has to be
        # set on the wrapper's params dict up front: the binary default aborts
        # natively (not with a Python exception) on a three-class label.
        # `params` is stored by reference so sklearn.clone() keeps working.
        patched = dict(getattr(model, "params", {}) or {})
        patched["objective"] = "multi:softprob"
        # `logloss` resolves to the *binary* row log-loss in libxgboost, which
        # also aborts natively against a 3-class label. `mlogloss` is the
        # multiclass form and is safe with multi:softprob.
        if str(patched.get("eval_metric", "logloss")).lower() == "logloss":
            patched["eval_metric"] = "mlogloss"
        model.params = patched

    try:
        model.fit(X_train, y_train)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def _evaluate(spec, config, context, *, horizon: int, threshold: float, mode: str,
              name: str, description: str, walkforward=None) -> dict[str, Any]:
    from src.dataset.factory import SplitSpec, TargetSpec

    matrix = context.features(spec.feature_groups)
    dataset = build_dataset(
        matrix,
        context.spot,
        TargetSpec(horizon_candles=horizon, threshold=threshold, mode=mode),
        spec.split,
        symbol=context.symbol,
        interval=context.interval,
        feature_groups=list(spec.feature_groups),
        data_sources=list(spec.data_sources),
    )
    if not len(dataset.frame):
        raise ValueError(f"{name}: empty dataset")

    train, val, test = dataset.splits["train"], dataset.splits["validation"], dataset.splits["test"]
    model = _fit(
        train.X, train.y, val.X, val.y, spec, config,
        dataset.feature_columns, int(getattr(config, "random_state", 42)),
        multiclass=(mode == "three_class"),
    )
    proba = _proba(model, test.X)

    out: dict[str, Any] = {
        "name": name, "description": description, "mode": mode,
        "horizon_candles": horizon, "threshold": threshold,
        "n_rows": int(len(dataset.frame)),
        "y": test.y, "proba": proba, "test": test,
        "n_features": len(dataset.feature_columns),
    }

    if mode == "three_class":
        metrics = multiclass_metrics(test.y, proba)
        out["test_accuracy"] = metrics["accuracy"]
        out["test_macro_f1"] = metrics["macro_f1"]
        out["test_balanced_accuracy"] = metrics["balanced_accuracy"]
        out["class_distribution"] = {
            k: v["share"] for k, v in metrics["per_class"].items()
        }
        out["test_roc_auc"] = None
        out["test_pr_auc"] = None
        out["notes"] = [
            "three-class: ROC-AUC is undefined, so macro-F1 and balanced accuracy are reported",
            "accuracy is reported only alongside macro-F1, because a majority-FLAT "
            "predictor can post high accuracy while being useless",
        ]
    else:
        pos = positive_class_proba(proba)
        metrics = classification_metrics(test.y, pos)
        out["test_roc_auc"] = metrics["roc_auc"]
        out["test_pr_auc"] = metrics["pr_auc"]
        out["test_accuracy"] = metrics["accuracy"]
        out["test_macro_f1"] = metrics["f1"]
        out["test_balanced_accuracy"] = metrics["balanced_accuracy"]
        out["class_distribution"] = {"1": float(test.y.mean()), "0": float(1 - test.y.mean())}
        out["notes"] = []

    shares = out["class_distribution"].values()
    out["majority_share"] = float(max(shares)) if shares else None

    if walkforward is not None:
        from src.evaluation.walkforward import run_walk_forward, summarise_windows

        X_pool = pd.concat([train.X, val.X])
        y_pool = pd.concat([train.y, val.y])
        metric_fn = (
            (lambda yt, yp: {k: v for k, v in multiclass_metrics(yt, yp).items()
                             if k in {"accuracy", "macro_f1", "balanced_accuracy"}})
            if mode == "three_class"
            else (lambda yt, yp: {k: classification_metrics(yt, positive_class_proba(yp)).get(k)
                                  for k in ("roc_auc", "pr_auc")})
        )
        windows = run_walk_forward(
            X_pool, y_pool,
            lambda: build_model(spec.model_name, config, dataset.feature_columns),
            walkforward, metric_fn=metric_fn,
        )
        summary = summarise_windows(windows)
        key = "macro_f1" if mode == "three_class" else "roc_auc"
        out["walkforward_roc_auc_mean"] = (summary.get(key) or {}).get("mean")
    return out


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    raw = np.asarray(model.decision_function(X))
    return np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw


def run_target_comparison(
    spec,
    config,
    *,
    context,
    output_dir: str | Path,
    variants: Sequence[dict[str, Any]] | None = None,
    walkforward=None,
    n_resamples: int = 500,
    block_size: int = 24,
) -> dict[str, Any]:
    """Evaluate several target definitions and write the comparison table."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = list(variants) if variants is not None else _default_variants(spec)
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for entry in plan:
        name = entry["name"]
        logger.info("target variant: %s", name)
        try:
            results.append(_evaluate(
                spec, config, context,
                horizon=entry.get("horizon_candles", spec.target.horizon_candles),
                threshold=entry.get("threshold", spec.target.threshold),
                mode=entry.get("mode", "binary"),
                name=name, description=entry.get("description", ""),
                walkforward=walkforward,
            ))
        except Exception as exc:
            logger.warning("target variant %s failed: %s", name, exc)
            errors[name] = str(exc)

    # Binary variants are only mutually comparable when they share test rows.
    binary = [r for r in results if r["mode"] != "three_class"]
    rows: list[TargetVariant] = []
    reference = next((r for r in binary if r["name"] == "binary_h6_t50bps"), binary[0] if binary else None)

    for got in results:
        row = TargetVariant(
            name=got["name"], description=got["description"], mode=got["mode"],
            horizon_candles=got["horizon_candles"], threshold=got["threshold"],
            n_rows=got["n_rows"], class_distribution=got["class_distribution"],
            majority_share=got["majority_share"],
            test_roc_auc=got["test_roc_auc"], test_pr_auc=got["test_pr_auc"],
            test_accuracy=got["test_accuracy"], test_macro_f1=got["test_macro_f1"],
            test_balanced_accuracy=got["test_balanced_accuracy"],
            walkforward_roc_auc_mean=got.get("walkforward_roc_auc_mean"),
            notes=list(got.get("notes", [])),
        )
        # Only same-mode, same-rows binary variants get an AUC delta.  A row
        # with no comparison is labelled as such rather than inheriting the
        # "indistinguishable" default, which would claim a paired test ran
        # when none did.
        if reference is not None and got["name"] == reference["name"]:
            row.verdict = "reference"
        elif (
            reference is not None
            and got["mode"] == reference["mode"]
            and got["name"] != reference["name"]
            and got["test_roc_auc"] is not None
            and reference["test_roc_auc"] is not None
        ):
            common = got["y"].index.intersection(reference["y"].index)
            if len(common) < 200:
                row.notes.append("too few shared test rows for a paired comparison")
                row.verdict = "not_compared"
            else:
                a = pd.Series(positive_class_proba(got["proba"]), index=got["y"].index).loc[common]
                b = pd.Series(positive_class_proba(reference["proba"]), index=reference["y"].index).loc[common]
                boot = paired_block_bootstrap_auc(
                    got["y"].loc[common], a.to_numpy(), b.to_numpy(),
                    n_resamples=n_resamples, block_size=block_size,
                    labels=(got["name"], reference["name"]),
                )
                row.delta_roc_auc_vs_binary = boot.point_delta
                row.ci_low, row.ci_high = boot.ci_low, boot.ci_high
                row.significant = boot.significant
                row.significantly_worse = boot.significantly_worse
                row.verdict = boot.verdict
        elif got["mode"] == "three_class" and reference is not None:
            row.notes.append(
                f"not directly comparable to {reference['name']}: a three-class label has no "
                f"single positive class, so its AUC delta is undefined by construction"
            )
        rows.append(row)

    table = pd.DataFrame([r.to_dict() for r in rows])
    table.to_csv(out / "target_comparison.csv", index=False)
    save_json(
        {
            "reference_variant": reference["name"] if reference else None,
            "variants": [r.to_dict() for r in rows],
            "failed_variants": errors,
            "caveat": (
                "Changing the target changes the question being asked, so accuracy and AUC are "
                "not comparable across rows. Only same-mode, same-rows variants carry a "
                "delta and an interval."
            ),
        },
        out / "target_comparison.json",
    )
    return {"table": table, "variants": rows, "results": results}


def _default_variants(spec) -> list[dict[str, Any]]:
    """Pre-registered target grid, anchored on the V1 definition."""
    base_h, base_t = spec.target.horizon_candles, spec.target.threshold
    return [
        {"name": "binary_h6_t50bps", "horizon_candles": 6, "threshold": base_t, "mode": "binary",
         "description": "V1 baseline: up if the 6-candle forward return exceeds +0.5%"},
        {"name": "binary_h6_t25bps", "horizon_candles": 6, "threshold": base_t / 2, "mode": "binary",
         "description": "tighter threshold, roughly twice the positive rate"},
        {"name": "binary_h6_t100bps", "horizon_candles": 6, "threshold": base_t * 2, "mode": "binary",
         "description": "wider threshold, rarer and more selective signals"},
        {"name": "binary_h12_t50bps", "horizon_candles": 12, "threshold": base_t, "mode": "binary",
         "description": "double the horizon at the same move size"},
        {"name": "binary_h24_t50bps", "horizon_candles": 24, "threshold": base_t, "mode": "binary",
         "description": "daily horizon: fewer, slower signals"},
        {"name": "three_class_h6_t50bps", "horizon_candles": 6, "threshold": base_t, "mode": "three_class",
         "description": "DOWN / FLAT / UP with a +/-0.25% flat band, so 'went nowhere' is "
                        "separated from 'reversed' instead of being labelled as down"},
    ]
