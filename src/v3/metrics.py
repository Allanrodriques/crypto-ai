"""Evaluation metrics for V3 forward-return regression.

V3 is a *regression* problem: for each horizon ``H`` the model predicts the
forward return ``close[t + H] / close[t] - 1``, not a label.  That changes what
an honest evaluation has to say, and this module exists to say it.

Three properties of the task drive the design.

1. **Targets overlap heavily.**  Two rows an hour apart at ``H = 30d`` share
   99.86% of their forward window, so the *number* of rows overstates the amount
   of independent evidence.  Point metrics (MAE/RMSE/R^2) are therefore reported
   next to rank and portfolio-style statistics (:func:`information_coefficient`,
   :func:`decile_analysis`, :func:`spread_analysis`), which is what a reader
   should actually trade on.  Nothing here corrects for overlap - that is the
   splitter's job - but a single ``r2`` alone is not a result.

2. **A predicted return is not a probability.**  Squashing a regression output
   through a sigmoid to get "probability up" produces a number that *looks*
   like it can be scored with a Brier score while measuring nothing.  So
   :func:`brier_score` and :func:`calibration_table` refuse inputs outside
   ``[0, 1]`` and refuse non-binary labels: they only accept probabilities that
   came from a genuine calibration step.

3. **Sign information is the tradable part.**  A model with a useless ``r2`` can
   still be informative if the *sign* of its output beats 50%.  That is
   measured separately, and both sides of the confusion are reported.

NaN policy (uniform across this module)
---------------------------------------
Every pairwise function takes a ``(y_true, y_pred)`` vector pair and applies one
rule: **keep a row only when both values are finite.**  A prediction that is
``NaN`` (unlabelled tail, missing feature) is not a data point, and pairing it
with a real target would fabricate a perfect miss.  The number of surviving rows
is always returned as ``n`` (or as the per-decile counts), so a filtered metric
can never be mistaken for one computed on the full sample.  Length mismatch is
a bug in the caller and raises :class:`MetricsError`; non-finite values are
expected data state and are filtered, never raised on.  When filtering leaves
nothing usable, every metric is ``NaN`` and the counts are ``0`` - functions do
not raise and do not return a plausible-looking zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("v3.metrics")

#: A forward return whose magnitude is below this is indistinguishable from
#: zero, so a percentage error against it is an arbitrary multiple of noise.
#: See :func:`regression_metrics` for how this bounds MAPE.
MAPE_FLOOR = 1e-6

#: Returned in place of a metric that is undefined for the surviving rows.
NAN = float("nan")


class MetricsError(ValueError):
    """Raised when inputs cannot be scored at all (bad shape, bad range).

    A subclass of :class:`ValueError` so callers can keep catching the built-in.
    """


# --------------------------------------------------------------------------- helpers

def _clean_pair(
    y_true: Any, y_pred: Any, *, name_true: str = "y_true", name_pred: str = "y_pred"
) -> tuple[np.ndarray, np.ndarray]:
    """Return the aligned finite ``(y_true, y_pred)`` pair.

    A row survives only if **both** values are finite; see the module docstring
    for why.  Shape disagreement raises :class:`MetricsError` - a mismatched
    length means the caller mis-paired two differently-sorted vectors, which is
    precisely the bug that produces a plausible-looking but wrong score.
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    if y.shape != p.shape:
        raise MetricsError(
            f"{name_true} has {y.size} rows but {name_pred} has {p.size}; "
            "the two must be aligned row-for-row before scoring"
        )
    keep = np.isfinite(y) & np.isfinite(p)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug("dropped %d row(s) with a non-finite value before scoring", dropped)
    return y[keep], p[keep]


def _clean_vector(values: Any, *, name: str = "values") -> np.ndarray:
    """Return the finite entries of a 1-D vector (``NaN``/``inf`` dropped)."""
    arr = np.asarray(values, dtype="float64").ravel()
    keep = np.isfinite(arr)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug("dropped %d non-finite entr(ies) from %s", dropped, name)
    return arr[keep]


def _mean(values: np.ndarray) -> float:
    """Mean of a possibly empty vector, or ``NaN`` (never 0.0 for "no data")."""
    return float(values.mean()) if values.size else NAN


def _std(values: np.ndarray) -> float:
    """Sample standard deviation (``ddof=1``).

    ``ddof=1`` rather than ``0`` on purpose: a single observation has no
    dispersion, and reporting ``0.0`` there would be a confident-looking number
    for "unknown".  It is ``NaN`` instead.  For ``n < 2`` the denominator is zero,
    so this returns ``NaN`` before the arithmetic can divide by it.
    """
    if values.size < 2:
        return NAN
    return float(np.std(values, ddof=1))


def _quantile(values: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile, or ``NaN`` for an empty vector."""
    return float(np.quantile(values, q)) if values.size else NAN


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination with an explicit zero-variance guard.

    ``1 - SS_res / SS_tot`` is undefined when ``y_true`` is constant (the
    denominator is 0).  ``sklearn`` papers over that with ``force_finite`` and
    reports ``1.0`` for a perfect fit to a constant target, which reads as a
    *great* result for a model that predicted nothing.  ``NaN`` is the honest
    answer: with no variance in the target there is nothing to explain.
    """
    if y.size < 2:
        return NAN
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:
        return NAN
    ss_res = float(np.sum((y - p) ** 2))
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------- regression

def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Point-forecast error for a forward-return prediction.

    Parameters
    ----------
    y_true, y_pred:
        Row-aligned vectors of realised and predicted forward returns.  Rows
        where either side is not finite are dropped (module NaN policy) and the
        survivors are reported as ``n``.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``mse``, ``r2``, ``n``, plus ``mape`` and
        ``mape_coverage`` (see below).  Every error metric is ``NaN`` when no row
        survives the filter.

    Notes
    -----
    **MAPE is a weak metric on returns and is reported only for comparability.**
    A return distribution straddles zero, so ``|y_true|`` approaches 0 while the
    error stays finite and the ratio explodes.  The guard is therefore twofold:
    rows with ``|y_true| < MAPE_FLOOR`` (1e-6, i.e. a return indistinguishable
    from flat) are excluded from the average, and ``mape_coverage`` states the
    fraction of rows that survived that exclusion.  ``mape`` is ``NaN`` when no
    row qualifies.  A low ``mape_coverage`` means the number is computed on a
    cherry-picked subset and should not be quoted.
    """
    y, p = _clean_pair(y_true, y_pred)
    if y.size == 0:
        logger.warning("regression_metrics received no finite (y_true, y_pred) rows")
        return {
            "mae": NAN, "rmse": NAN, "mse": NAN, "r2": NAN, "n": 0,
            "mape": NAN, "mape_coverage": NAN,
        }

    err = p - y
    mse = float(np.mean(err**2))
    usable = np.abs(y) >= MAPE_FLOOR
    coverage = float(usable.mean())

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": _r2(y, p),
        "n": int(y.size),
        "mape": float(np.mean(np.abs(err[usable] / y[usable]))) if usable.any() else NAN,
        "mape_coverage": coverage,
    }


def direction_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Does the *sign* of the prediction carry information?

    This is the tradable question, and it is separate from the point-forecast
    error: a model can be well calibrated in magnitude and useless in direction,
    or vice versa.

    Direction rules (both are deliberate, and both are conservative):

    * A prediction of exactly ``0.0`` is **never** correct.  Rounding to flat is
      what a model does when it has no view, and scoring it as half-right would
      let "predict nothing" post a positive number.  Zero predictions are
      therefore misses in ``direction_accuracy``.
    * A ``y_true`` of exactly ``0.0`` is not a hit for either side, and is
      counted in neither ``n_up`` nor ``n_down``.  Such rows still count in the
      denominator of ``direction_accuracy`` and in ``n``, because they were real
      observations the model failed to call.

    ``balanced_direction_accuracy`` is the mean of the two one-sided recalls, so
    a model that always answers "up" cannot score well on a market that is down
    more often than up.  It is ``NaN`` when one of the two classes is absent from
    ``y_true``, because a one-sided recall average over one class is not a
    balanced score.
    """
    y, p = _clean_pair(y_true, y_pred)
    if y.size == 0:
        logger.warning("direction_metrics received no finite (y_true, y_pred) rows")
        return {
            "direction_accuracy": NAN, "balanced_direction_accuracy": NAN,
            "n_up": 0, "n_down": 0, "n": 0,
        }

    up, down = p > 0.0, p < 0.0
    hits = (up & (y > 0.0)) | (down & (y < 0.0))

    true_up = y > 0.0
    true_down = y < 0.0
    n_up = int(true_up.sum())
    n_down = int(true_down.sum())

    if n_up and n_down:
        balanced = 0.5 * (float((up & true_up).sum()) / n_up + float((down & true_down).sum()) / n_down)
    else:
        balanced = NAN

    return {
        "direction_accuracy": float(hits.sum()) / float(y.size),
        "balanced_direction_accuracy": balanced,
        "n_up": n_up,
        "n_down": n_down,
        "n": int(y.size),
    }


def information_coefficient(y_true: Any, y_pred: Any) -> float:
    """Spearman rank correlation between prediction and realised return.

    Rank correlation rather than Pearson because the return distribution is
    heavy-tailed and the prediction is not calibrated to a Gaussian: IC measures
    ordering, which is the thing a rank-based portfolio actually consumes, and it
    is the metric that survives a monotone-but-nonlinear mapping.

    Returns ``NaN`` - never an exception - when either input is constant or fewer
    than 3 rows survive the NaN filter.  A constant input is a *reported* fact
    about the model, not a crash: ``sklearn``'s ``nan_efficiency_score`` and
    pandas' ``ConstantInputWarning`` both turn it into an error or a warning
    noise, and the reader needs to see "this model has no ranking power" in the
    results table.
    """
    y, p = _clean_pair(y_true, y_pred)
    if y.size < 3:
        return NAN
    if np.all(y == y[0]) or np.all(p == p[0]):
        logger.debug("information_coefficient is undefined for a constant input")
        return NAN

    # Checked before the call so the undefined case never reaches pandas, which
    # warns (and under -W error, aborts) on constant input.
    ic = pd.Series(p).corr(pd.Series(y), method="spearman")
    return float(ic) if ic is not None and np.isfinite(ic) else NAN


# --------------------------------------------------------------------------- ranking portfolios

_DECILE_COLUMNS = (
    "decile", "n", "n_deciles", "mean_actual_return", "median_actual_return", "std_actual_return"
)


def decile_analysis(y_true: Any, y_pred: Any, n_deciles: int = 5) -> pd.DataFrame:
    """Group rows by predicted rank and summarise what actually happened.

    This is the shape test for a model: if the prediction is informative, the
    mean realised return must rise monotonically across the buckets.  A flat or
    decreasing ladder means the model is adding magnitude, not information.

    Parameters
    ----------
    n_deciles:
        Requested bucket count.  With the default ``5`` these are quintiles; the
        name is kept from the finance convention.

    Returns
    -------
    pandas.DataFrame
        Columns ``decile, n, n_deciles, mean_actual_return,
        median_actual_return, std_actual_return``.  ``decile`` is **0-based**:
        ``0`` is the lowest-prediction bucket and the last row is the highest.
        ``n_deciles`` records the count actually used, which is smaller than the
        request when there are fewer rows than buckets.

    Notes
    -----
    Ties are the trap in this function.  ``qcut`` on raw predictions cannot split
    a tied group and raises, or - worse - silently drops duplicates and returns
    fewer buckets than requested, which makes a broken model look like one with
    fewer, larger buckets.  The fix is to rank first (which is why Spearman IC is
    defined for tied predictions at all) and cut the *ranks*: ranks are
    ``1..n`` and distinct, so every requested bucket is guaranteed at least one
    row.  Ties are broken by position in the input, so with a heavily-tied
    prediction the extreme buckets are an arbitrary subset of the tied group -
    which is itself the finding, not a defect of the grouping.
    """
    if n_deciles < 1:
        raise MetricsError(f"n_deciles must be >= 1, got {n_deciles}")

    y, p = _clean_pair(y_true, y_pred)
    if y.size == 0:
        logger.warning("decile_analysis received no finite (y_true, y_pred) rows")
        return pd.DataFrame({c: pd.Series(dtype=_decile_dtype(c)) for c in _DECILE_COLUMNS})

    used = int(min(n_deciles, y.size))
    frame = pd.DataFrame({"pred": p, "actual": y})

    if used < 2:
        # Nothing to cut: a single row is its own bucket, std is undefined.
        rows = [{
            "decile": 0, "n": int(y.size), "n_deciles": 1,
            "mean_actual_return": _mean(y), "median_actual_return": _median(y),
            "std_actual_return": NAN,
        }]
        return pd.DataFrame(rows, columns=list(_DECILE_COLUMNS))

    ranked = pd.Series(p).rank(method="first")
    frame["decile"] = pd.qcut(ranked, q=used, labels=False, duplicates="drop").astype(int)

    grouped = frame.groupby("decile")["actual"]
    table = pd.DataFrame(
        {
            "decile": np.arange(used),
            "n": grouped.size().reindex(range(used), fill_value=0).to_numpy(),
            "mean_actual_return": grouped.mean().reindex(range(used)).to_numpy(),
            "median_actual_return": grouped.median().reindex(range(used)).to_numpy(),
            "std_actual_return": grouped.std(ddof=1).reindex(range(used)).to_numpy(),
        }
    )
    table["n_deciles"] = int(table["decile"].max() + 1)
    return table[list(_DECILE_COLUMNS)]


def _decile_dtype(column: str) -> str:
    """Pandas dtype for an empty decile table's column."""
    return "float64" if column in {"mean_actual_return", "median_actual_return", "std_actual_return"} else "int64"


def _median(values: np.ndarray) -> float:
    """Median of a possibly empty vector."""
    return float(np.median(values)) if values.size else NAN


def spread_analysis(y_true: Any, y_pred: Any, tail_fraction: float = 0.2) -> dict[str, float]:
    """Long-minus-short return of the extreme prediction tails.

    The simplest portfolio a rank-based model can express: go long the highest
    ``tail_fraction`` of predictions, short the lowest, and report what the two
    legs actually realised.  Unlike MAE this is in the units that matter (a
    return), and unlike the IC it is directly interpretable as PnL before costs.

    Parameters
    ----------
    tail_fraction:
        Fraction of rows per leg, default ``0.2`` (the top and bottom quintiles).
        Each leg holds ``floor(n * tail_fraction)`` rows.  If that is zero - too
        few rows for a meaningful tail - every value is ``NaN`` and both counts
        are ``0`` rather than a one-row "spread" built from a single point.

    Notes
    -----
    Ties are resolved by input position (a stable sort), which keeps the result
    deterministic and reproducible.  A constant prediction gives a spread of
    exactly 0.0, correctly: two arbitrary halves of a constant score have the
    same expected return by construction.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise MetricsError(f"tail_fraction must be in (0, 1], got {tail_fraction}")

    y, p = _clean_pair(y_true, y_pred)
    if y.size == 0:
        logger.warning("spread_analysis received no finite (y_true, y_pred) rows")
        return {
            "long_mean_return": NAN, "short_mean_return": NAN, "spread": NAN,
            "n_long": 0, "n_short": 0,
        }

    per_leg = int(np.floor(y.size * tail_fraction))
    if per_leg < 1:
        logger.warning(
            "spread_analysis: %d rows cannot fill a %.2f tail; returning NaN", y.size, tail_fraction
        )
        return {
            "long_mean_return": NAN, "short_mean_return": NAN, "spread": NAN,
            "n_long": 0, "n_short": 0,
        }

    order = np.argsort(p, kind="stable")
    short_idx = order[:per_leg]
    long_idx = order[-per_leg:]
    long_mean = _mean(y[long_idx])
    short_mean = _mean(y[short_idx])
    spread = long_mean - short_mean if np.isfinite(long_mean) and np.isfinite(short_mean) else NAN

    return {
        "long_mean_return": long_mean,
        "short_mean_return": short_mean,
        "spread": spread,
        "n_long": int(long_idx.size),
        "n_short": int(short_idx.size),
    }


# --------------------------------------------------------------------------- aggregate

def evaluate_regression(y_true: Any, y_pred: Any, n_deciles: int = 5) -> dict[str, float]:
    """One flat bundle of headline V3 metrics for a single (y, prediction) pair.

    Every key comes from the functions above, so the NaN and zero-prediction
    policies are exactly the ones documented there - this is a convenience for
    the per-horizon report table, not a second implementation of the maths.

    Returns
    -------
    dict[str, float]
        :func:`regression_metrics` + :func:`direction_metrics` plus
        ``spearman_ic`` (rank correlation) and ``long_short_spread`` (the
        top-minus-bottom quintile realised return).  Read the last two next to
        ``r2``, never instead of it.
    """
    bundle: dict[str, float] = {}
    bundle.update(regression_metrics(y_true, y_pred))
    bundle.update(direction_metrics(y_true, y_pred))
    spread = spread_analysis(y_true, y_pred)
    bundle["spearman_ic"] = information_coefficient(y_true, y_pred)
    bundle["long_short_spread"] = spread["spread"]
    logger.info(
        "regression bundle: n=%s r2=%.4f dir_acc=%.4f ic=%.4f spread=%.6f",
        bundle.get("n"), bundle["r2"], bundle["direction_accuracy"],
        bundle["spearman_ic"], bundle["long_short_spread"],
    )
    return bundle


# --------------------------------------------------------------------------- distributions

def distribution_stats(values: Any) -> dict[str, float]:
    """Shape summary of a return or prediction distribution.

    The quantiles are the point of this function.  A forward return distribution
    whose median is negative and whose p95 is a few percent is the context every
    regression metric needs: a model can look acceptable in ``R^2`` purely
    because returns are mean-reverting noise, and only the spread of the target
    says how much signal there was available in the first place.

    Non-finite entries are dropped (module NaN policy) and ``n`` is the count that
    survived, so ``n`` smaller than the input length is visible rather than
    implied.  All statistics are ``NaN`` when nothing survives.  ``std`` is the
    sample standard deviation (``ddof=1``), so a single value reports ``NaN``.
    """
    arr = _clean_vector(values, name="distribution_stats")
    if arr.size == 0:
        logger.warning("distribution_stats received no finite values")
        return {
            "n": 0, "mean": NAN, "std": NAN, "median": NAN,
            "p05": NAN, "p25": NAN, "p75": NAN, "p95": NAN,
            "min": NAN, "max": NAN, "positive_fraction": NAN,
        }

    return {
        "n": int(arr.size),
        "mean": _mean(arr),
        "std": _std(arr),
        "median": _median(arr),
        "p05": _quantile(arr, 0.05),
        "p25": _quantile(arr, 0.25),
        "p75": _quantile(arr, 0.75),
        "p95": _quantile(arr, 0.95),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "positive_fraction": float((arr > 0.0).mean()),
    }


# --------------------------------------------------------------------------- calibration

def _validated_probability_pair(prob: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    """Align a probability/label pair and enforce the domain of a probability.

    The range check is the important part.  Squeezing a regression output into
    ``[0, 1]`` with a clip or a sigmoid is the standard way a return model gets
    scored with a Brier score that measures nothing; refusing the input is what
    keeps that mistake loud.
    """
    p = np.asarray(prob, dtype="float64").ravel()
    y = np.asarray(labels, dtype="float64").ravel()
    if p.shape != y.shape:
        raise MetricsError(
            f"prob has {p.size} rows but labels has {y.size}; "
            "the two must be aligned row-for-row before scoring"
        )
    # Validated before the NaN filter, but only over the rows that can actually
    # be scored: an all-NaN input has no range to check, and `nanmin` on it warns.
    live = p[np.isfinite(p)]
    if live.size:
        if live.min() < 0.0 or live.max() > 1.0:
            raise MetricsError(
                "prob must lie in [0, 1]; a regression output is not a probability. "
                "Calibrate the score first, then score the calibrated probability."
            )
        bad_labels = np.setdiff1d(np.unique(y[np.isfinite(y)]), np.array([0.0, 1.0]))
        if bad_labels.size:
            raise MetricsError(f"labels must be binary 0/1, found {bad_labels.tolist()}")
    keep = np.isfinite(p) & np.isfinite(y)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug("dropped %d row(s) with a non-finite probability/label", dropped)
    return p[keep], y[keep]


def brier_score(prob: Any, labels: Any) -> float:
    """Mean squared error of a *genuine* probability against a binary outcome.

    ``mean((prob - labels) ** 2)``.  Both inputs must be real probabilities: a
    ``prob`` outside ``[0, 1]`` and a non-binary ``labels`` are both rejected with
    :class:`MetricsError`, because a Brier score computed on a rescaled return
    looks meaningful and means nothing.

    Rows where either side is not finite are dropped.  ``NaN`` is returned when
    nothing survives.
    """
    p, y = _validated_probability_pair(prob, labels)
    if p.size == 0:
        logger.warning("brier_score received no finite (prob, labels) rows")
        return NAN
    return float(np.mean((p - y) ** 2))


def calibration_table(prob: Any, labels: Any, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability table over ``[0, 1]``.

    A model is calibrated when ``mean_predicted == observed_frequency`` in every
    populated bin.  Equal-width bins are used deliberately (unlike V1's
    quantile-binned curve) because the question here is "what does the model
    claim when it says 0.2?", which only has a meaning on a fixed grid.  A
    quantile-binned curve is always straight by construction and therefore cannot
    answer it.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_lower, bin_upper, n, mean_predicted, observed_frequency``,
        always exactly ``n_bins`` rows: empty bins are kept with ``n == 0`` and
        ``NaN`` in both aggregates.  A dropped empty bin would hide the fact that
        the model never predicts in that range, which is itself a calibration
        finding.
    """
    if n_bins < 1:
        raise MetricsError(f"n_bins must be >= 1, got {n_bins}")

    p, y = _validated_probability_pair(prob, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for b in range(n_bins):
        lower, upper = float(edges[b]), float(edges[b + 1])
        # Left-closed / right-open, with the last bin closed on the right so a
        # probability of exactly 1.0 is counted rather than falling off the end.
        if b == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "n": count,
                "mean_predicted": float(p[mask].mean()) if count else NAN,
                "observed_frequency": float(y[mask].mean()) if count else NAN,
            }
        )
    return pd.DataFrame(rows)
