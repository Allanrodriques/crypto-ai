"""Classification metrics, probability-calibration diagnostics and class balance.

Two rules shape this module:

1. **Accuracy alone is not a result.**  On an imbalanced crypto target a
   majority-class predictor can post a 70%+ accuracy while being useless, so
   every report carries ROC-AUC, PR-AUC and the confusion matrix next to it.
2. **A predicted probability is a model output, not a promise.**  The
   calibration block reports Brier score, log loss, expected calibration error
   and a reliability curve so the number can be judged rather than trusted.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-15


# --------------------------------------------------------------------------- helpers

def _as_arrays(y_true: Any, y_pred_proba: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_pred_proba, dtype="float64").ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true has shape {y.shape} but probabilities have shape {p.shape}")
    if y.size == 0:
        raise ValueError("Cannot compute metrics on an empty sample")
    return y, p


def _safe(value: Any) -> Any:
    """Return ``None`` instead of NaN so the value serialises to strict JSON."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return f if np.isfinite(f) else None


def positive_class_proba(y_pred_proba: Any, n_classes: int = 2) -> np.ndarray:
    """Reduce a model's output to a 1-D positive-class probability.

    Tree ensembles return an ``(n, n_classes)`` matrix, linear models a bare
    probability, and wrappers vary between the two.  Mixing these up is a silent
    and very easy bug: a ``(n, 2)`` matrix has the right *length* relative to
    nothing at all, so it either crashes on a shape check or, worse, gets
    silently mis-shaped downstream.  This is the one place the conversion
    happens.
    """
    p = np.asarray(y_pred_proba, dtype="float64")
    if p.ndim == 1:
        return p
    if p.ndim != 2:
        raise ValueError(f"unexpected probability shape {p.shape}")
    if p.shape[1] == 1:
        return p[:, 0]
    if n_classes <= 2:
        # Column order follows the sorted label set, so the positive class is last.
        return p[:, 1]
    return p


def multiclass_metrics(
    y_true: Any,
    y_pred_proba: Any,
    *,
    labels: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Metrics for a three-class DOWN / FLAT / UP target.

    ``macro`` averaging is the headline number on purpose.  With accuracy alone a
    model that predicts FLAT for everything scores well whenever FLAT dominates,
    which it usually does at tight thresholds; macro-averaged scores and
    per-class recall expose that.
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_pred_proba, dtype="float64")
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true has shape {y.shape} but probabilities have shape {p.shape}")
    if y.size == 0:
        raise ValueError("Cannot compute metrics on an empty sample")
    # Row-normalise defensively: log_loss and argmax both assume a proper
    # distribution, and tree ensembles can leave a few ulps of drift.
    totals = p.sum(axis=1, keepdims=True)
    p = np.divide(p, totals, out=np.full_like(p, 1.0 / p.shape[1]), where=totals > 0)

    # Map each row's highest-probability column back to its label value, so the
    # predicted labels are 0/1/2 rather than column indices.
    y_hat = np.asarray(labels, dtype=int)[np.argmax(p, axis=1)]
    label_list = [int(v) for v in labels]

    per_class = {}
    for label in label_list:
        mask = y == label
        n = int(mask.sum())
        per_class[str(label)] = {
            "support": n,
            "share": float(n / y.size),
            "precision": _safe(precision_score(y == label, y_hat == label, zero_division=0)),
            "recall": _safe(recall_score(y == label, y_hat == label, zero_division=0)),
            "f1": _safe(f1_score(y == label, y_hat == label, zero_division=0)),
        }

    return {
        "n_samples": int(y.size),
        "n_classes": len(label_list),
        "labels": label_list,
        "accuracy": _safe(accuracy_score(y, y_hat)),
        "balanced_accuracy": _safe(balanced_accuracy_score(y, y_hat)),
        "macro_f1": _safe(f1_score(y, y_hat, average="macro", zero_division=0)),
        "weighted_f1": _safe(f1_score(y, y_hat, average="weighted", zero_division=0)),
        "log_loss": _safe(log_loss(y, p, labels=label_list)),
        "confusion_matrix": confusion_matrix(y, y_hat, labels=label_list).tolist(),
        "per_class": per_class,
        "predicted_class_share": {
            str(label): float(np.mean(y_hat == label)) for label in label_list
        },
    }


# --------------------------------------------------------------------------- core metrics

def classification_metrics(
    y_true: Any,
    y_pred_proba: Any,
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Full metric bundle for one (y, probability) pair.

    Parameters
    ----------
    y_true:
        Binary labels.
    y_pred_proba:
        Probability of the positive class, shape ``(n,)``.
    threshold:
        Decision cut for converting probability to a hard class.  Configured,
        never tuned on the test set.
    """
    y, p = _as_arrays(y_true, y_pred_proba)
    y_hat = (p >= threshold).astype(int)
    pos = (y == positive_label).astype(int)
    pred_pos = (y_hat == positive_label).astype(int)

    single_class = len(np.unique(y)) < 2
    single_prob = len(np.unique(p)) < 2

    tn, fp, fn, tp = confusion_matrix(pos, pred_pos, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "n_samples": int(y.size),
        "accuracy": _safe(accuracy_score(y, y_hat)),
        "balanced_accuracy": _safe(balanced_accuracy_score(y, y_hat)) if not single_class else None,
        "precision": _safe(precision_score(pos, pred_pos, zero_division=0)),
        "recall": _safe(recall_score(pos, pred_pos, zero_division=0)),
        "f1": _safe(f1_score(pos, pred_pos, zero_division=0)),
        "matthews_corrcoef": _safe(matthews_corrcoef(pos, pred_pos)),
        "specificity": _safe(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": None if single_class or single_prob else _safe(roc_auc_score(pos, p)),
        "pr_auc": None if single_class else _safe(average_precision_score(pos, p)),
        "roc_auc_note": None if not (single_class or single_prob) else "undefined: single class or constant probability",
        "pr_auc_note": None if not single_class else "undefined: single class in y_true",
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "layout": ["predicted_down", "predicted_up"],
            "columns": ["actual_down", "actual_up"],
        },
        "predicted_up_rate": _safe(float(pred_pos.mean())),
        "probability_stats": {
            "mean": _safe(p.mean()),
            "std": _safe(p.std(ddof=0)),
            "min": _safe(p.min()),
            "p05": _safe(np.quantile(p, 0.05)),
            "median": _safe(np.median(p)),
            "p95": _safe(np.quantile(p, 0.95)),
            "max": _safe(p.max()),
        },
        "class_distribution": class_distribution(y, positive_label=positive_label),
    }


def class_distribution(y_true: Any, *, positive_label: int = 1) -> dict[str, Any]:
    """Class balance of a label vector — reported for every split and every model.

    A target with, say, 32% positives means accuracy above 68% is achievable by
    predicting one class forever.  Reporting the distribution is what makes the
    accuracy number interpretable.
    """
    y = np.asarray(y_true).astype(int).ravel()
    if y.size == 0:
        return {"n": 0, "share_up": None, "imbalance_ratio": None}
    n_up = int((y == positive_label).sum())
    n_down = int(y.size - n_up)
    return {
        "n": int(y.size),
        "n_up": n_up,
        "n_down": n_down,
        "share_up": _safe(n_up / y.size),
        "share_down": _safe(n_down / y.size),
        "imbalance_ratio_down_over_up": _safe(n_down / n_up) if n_up else None,
        "majority_class_share": _safe(max(n_up, n_down) / y.size),
    }


# --------------------------------------------------------------------------- calibration

def reliability_curve(
    y_true: Any, y_pred_proba: Any, *, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Binned reliability (calibration) curve.

    Parameters
    ----------
    strategy:
        ``"quantile"`` (default) puts the same number of samples in every bin,
        which is the right choice when probabilities cluster near one value.
        ``"uniform"`` uses fixed-width bins, which is the textbook version.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_lower, bin_upper, n, mean_predicted, observed_frequency``.
        A perfectly calibrated model has ``mean_predicted == observed_frequency``.
    """
    y, p = _as_arrays(y_true, y_pred_proba)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0

    bin_index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_index == b
        count = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": count,
                "mean_predicted": float(p[mask].mean()) if count else None,
                "observed_frequency": float(y[mask].mean()) if count else None,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: Any, y_pred_proba: Any, *, n_bins: int = 10) -> float:
    """Sample-weighted ``|mean predicted - observed frequency|`` across bins.

    0.0 is perfect calibration, 0.5 is maximal error.  Interpret alongside the
    base rate: a model that always says 0.30 on a 30%-positive target scores 0.0.
    """
    curve = reliability_curve(y_true, y_pred_proba, n_bins=n_bins)
    curve = curve.dropna(subset=["mean_predicted", "observed_frequency"])
    total = float(curve["n"].sum())
    if total == 0:
        return float("nan")
    weights = curve["n"] / total
    gaps = (curve["mean_predicted"] - curve["observed_frequency"]).abs()
    return float((weights * gaps).sum())


def calibration_metrics(y_true: Any, y_pred_proba: Any, *, n_bins: int = 10) -> dict[str, Any]:
    """Probability-quality metrics for a model's ``probability_up`` output.

    Interpretation guard-rail: these measure how well the *model's own score*
    matches the observed base rate in this sample.  They are not a statement
    about future returns, and a well-calibrated model can still lose money.
    """
    y, p = _as_arrays(y_true, y_pred_proba)
    clipped = np.clip(p, EPS, 1 - EPS)
    single_class = len(np.unique(y)) < 2
    return {
        "brier_score": _safe(brier_score_loss(y, clipped)),
        "brier_skill_score": _safe(brier_skill_score(y, clipped)),
        "log_loss": None if single_class else _safe(log_loss(y, clipped, labels=[0, 1])),
        "expected_calibration_error": _safe(expected_calibration_error(y, p, n_bins=n_bins)),
        "base_rate_up": _safe(float(y.mean())),
        "mean_predicted_probability": _safe(float(p.mean())),
        "calibration_gap": _safe(float(p.mean() - y.mean())),
        "n_bins": int(n_bins),
        "note": (
            "Calibration describes the model's score against observed base rates on this sample. "
            "It is not a guarantee of profit."
        ),
    }


def brier_skill_score(y_true: Any, y_pred_proba: Any) -> float:
    """Brier score relative to always predicting the base rate (0 = no better)."""
    y, p = _as_arrays(y_true, y_pred_proba)
    base = float(y.mean())
    reference = float(np.mean((y - base) ** 2))
    if reference <= 0:
        return float("nan")
    return 1.0 - float(np.mean((y - np.clip(p, EPS, 1 - EPS)) ** 2)) / reference


# --------------------------------------------------------------------------- aggregation

def summarise_metric_table(results: dict[str, dict[str, Any]], keys: Sequence[str]) -> pd.DataFrame:
    """Turn ``{model_name: metrics}`` into a comparison table for the report."""
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        for key in keys:
            row[key] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def headline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """The small set of numbers quoted in the console summary."""
    return {
        key: metrics.get(key)
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
        )
    }
