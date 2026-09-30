"""Feature importance: gain, permutation, and optional SHAP.

Three different questions
-------------------------
``gain``
    How much did the model *use* this feature to split nodes?  Cheap and
    available for every tree model, but it is biased toward high-cardinality
    features and it is a property of the fitted model, not of the data.
``permutation``
    How much does performance *degrade* when this feature is shuffled?  This is
    a property of the model-plus-data combination and is generally the more
    trustworthy of the two.
``shap``
    Local, additive per-prediction attributions.  Adds real insight into
    *direction* (does a high value push the prediction up or down), which the
    other two cannot show.

None of these establish causality.  A feature can be important because it
leaks, because it proxies for something else, or because it genuinely helps.
They are reported as diagnostics, and the leakage tests in ``tests/`` are what
actually constrain causality.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils import get_logger

logger = get_logger("evaluation.importance")

#: Features evaluated per permutation repeat (cost control).
DEFAULT_PERMUTATION_REPEATS = 5


def gain_importance(estimator, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Extract gain-based importance from a fitted tree model.

    Returns an empty frame (rather than raising) for models without
    ``feature_importances_``, so a logistic run does not break the report.
    """
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "gain_importance", "rank"])
    values = np.asarray(importances, dtype="float64")
    total = float(values.sum()) or 1.0
    frame = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "gain_importance": values,
            "gain_share_pct": 100.0 * values / total,
        }
    )
    return frame.sort_values("gain_importance", ascending=False).reset_index(drop=True).assign(
        rank=lambda d: np.arange(1, len(d) + 1)
    )


def permutation_importance_frame(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "roc_auc",
    repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = 42,
    max_features: int | None = 40,
) -> pd.DataFrame:
    """Permutation importance, optionally limited to the top-N by gain.

    Permutation importance is O(features x repeats) full refits of the *score*
    only, which is affordable for a few dozen features but not for 105.  The
    default therefore measures the top 40 by gain and says so in the returned
    frame, rather than silently measuring everything and taking forever.
    """
    columns = list(X.columns)
    if max_features is not None and len(columns) > max_features:
        ranked = gain_importance(estimator, columns)
        if not ranked.empty:
            columns = ranked.head(max_features)["feature"].tolist()
        else:
            columns = columns[:max_features]

    result = permutation_importance(
        estimator,
        X[columns],
        y,
        scoring=scoring,
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "measured_features": len(columns),
        }
    )
    return frame.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def shap_importance(
    estimator,
    X: pd.DataFrame,
    *,
    max_rows: int = 2_000,
    feature_columns: Sequence[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """SHAP mean absolute value plus sign-consistency, if ``shap`` is installed.

    Returns ``None`` when the optional dependency is absent so the pipeline
    degrades gracefully rather than failing.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.info("shap not installed; skipping SHAP analysis")
        return None

    sample = X
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=random_state)

    explainer = shap.TreeExplainer(estimator)
    values = explainer(sample, check_additivity=False)
    matrix = getattr(values, "values", values)
    if matrix.ndim == 3:  # (rows, features, classes)
        matrix = matrix[:, :, -1]

    magnitude = np.abs(matrix)
    sign = np.sign(matrix)
    frame = pd.DataFrame(
        {
            "feature": list(sample.columns),
            "shap_mean_abs": magnitude.mean(axis=0),
            "shap_std": matrix.std(axis=0),
            "shap_positive_share": (sign > 0).mean(axis=0),
        }
    ).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["measured_rows"] = len(sample)
    return frame, {"rows_explained": int(len(sample)), "n_features": int(len(sample.columns))}


def importance_bundle(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "roc_auc",
    with_shap: bool = True,
    repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Gather all available importance measures for one fitted model."""
    out: dict[str, Any] = {}
    gain = gain_importance(estimator, list(X.columns))
    out["gain"] = gain.to_dict(orient="records")[:50] if not gain.empty else []

    try:
        permutation = permutation_importance_frame(
            estimator, X, y, scoring=scoring, repeats=repeats, random_state=random_state
        )
        out["permutation"] = permutation.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - scoring edge cases
        logger.info("permutation importance unavailable: %s", exc)
        out["permutation"] = []

    if with_shap:
        shap_result = shap_importance(estimator, X, random_state=random_state)
        if shap_result is not None:
            frame, meta = shap_result
            out["shap"] = frame.to_dict(orient="records")
            out["shap_meta"] = meta
        else:
            out["shap"] = []

    out["caveat"] = (
        "Importance measures model reliance, not causal effect. A feature can rank "
        "highly because it proxies for another variable or because of leakage; it "
        "does not cause price movement."
    )
    return out
