"""Regime, distribution, interval, calibration and ablation reporting for V3.

The pooled metrics that :mod:`src.v3.walkforward` produces answer one question -
"how good is this model on average?".  For a forward-return model that question
is not sufficient, and this module exists to say why in tables rather than prose.

Why an average is the wrong headline
-------------------------------------
A crypto return model is only ever consumed through a *decision* made in a
*specific* market state, and those two facts break the average from both sides:

* **The edge is not where the data is.**  Crypto forward returns are small and
  persistent relative to their noise in calm periods and large and mean-reverting
  in stressed ones.  A model that only works when volatility is high can post a
  respectable pooled ``r2`` and be untradeable for 80% of the calendar, and an
  average hides that completely.  So :func:`evaluate_by_regime` is not a
  diagnostic extra - for a long horizon it is the first table to read.
* **The claim needs an error bar.**  A 90% conformal band measured on 200 pooled
  test points, or a 62% direction hit rate on 300 rows, is a single binomial
  draw.  Every proportion here therefore travels with a Wilson interval
  (:func:`src.v3.uncertainty.wilson_interval`) rather than as a bare number.
* **A number can be structurally meaningless.**  An always-0.5 probability
  scores a respectable Brier on a coin-flip market, and its "calibration slope"
  is undefined rather than zero.  :func:`calibration_report` detects that
  degeneracy explicitly instead of printing a plausible number.

Two deliberate constraints on the definitions
---------------------------------------------
1. **Regimes are quantiles of a trailing feature, never of the target.**  Both
   :func:`label_volatility_regimes` and :func:`label_trend_regimes` rank the
   series they are given and cut the *ranks*, so each bucket holds roughly the
   same number of rows and the label assigned to a value depends on the value
   alone - reordering the input cannot change it.  Cut points on the realised
   target would make every conditional statistic in the report a tautology.
2. **Nothing here imports the walk-forward harness.**  :func:`run_ablation`
   takes a ``fit_predict(features, seed) -> y_pred`` callable from the caller.
   An ablation study needs the harness to refit, and the harness lives downstream
   of the metrics this module reports; importing it would close the loop and make
   the reporting layer impossible to test without training a model.

This module is a leaf.  It imports only from :mod:`src.v3.metrics`,
:mod:`src.v3.uncertainty` and :mod:`src.utils`, so the maths it reports is
exactly the maths those modules define, and V1/V2 stay frozen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger
from src.v3.metrics import (
    brier_score,
    calibration_table,
    decile_analysis,
    distribution_stats,
    information_coefficient,
    regression_metrics,
    spread_analysis,
)
from src.v3.uncertainty import wilson_interval

logger = get_logger("v3.analysis")

NAN = float("nan")


class AnalysisError(ValueError):
    """Raised when the inputs cannot be analysed at all.

    A :class:`ValueError` subclass, matching :class:`src.v3.metrics.MetricsError`:
    every failure mode here is a caller passing something the report cannot
    honestly present, and the right response is to refuse rather than to emit a
    table of ``NaN`` that looks like a result.
    """


# --------------------------------------------------------------------- definitions

#: Column-name fragments that identify a dispersion measure.  Volatility is
#: non-negative by construction, so any column carrying one of these is treated
#: as a volatility source by :func:`regime_label`.
VOLATILITY_TOKENS: tuple[str, ...] = ("vol", "atr", "range", "sigma", "stdev", "std", "ddof")

#: Column-name fragments that identify a *signed* state measure.  Momentum, RSI
#: and z-scores are included because they are all trend states wearing different
#: units, and quartileing a signed series is what makes a trend regime readable.
TREND_TOKENS: tuple[str, ...] = (
    "trend", "momentum", "slope", "rsi", "zscore", "z_score", "signed", "return",
)


@dataclass(frozen=True)
class RegimeDefinition:
    """A documented quantile-bin rule for turning a series into regime labels.

    Attributes
    ----------
    name:
        Identifier of the rule itself (``'volatility_quartile'``), used in logs
        and report filenames rather than as a row label - the row label is
        :attr:`labels`, so a report never mixes a rule name with a bucket name.
    column:
        The feature the rule is defined on.  Nothing in this module computes the
        feature; the caller supplies the already-trailing column, and recording
        its name here is what makes the report self-describing.
    bins:
        The **interior** quantile cut points in ``(0, 1)``, so ``len(labels) ==
        len(bins) + 1``.  A value whose quantile position equals a cut point falls
        in the *lower* bucket, matching :func:`pandas.qcut`.
    labels:
        One label per bucket, in ascending value order: ``labels[0]`` is the
        lowest bucket.  For a volatility rule that is the calmest bucket; for a
        trend rule it is the most bearish.
    """

    name: str
    column: str
    bins: tuple[float, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise AnalysisError("RegimeDefinition.name must be non-empty")
        if not self.column:
            raise AnalysisError("RegimeDefinition.column must be non-empty")
        if len(self.labels) != len(self.bins) + 1:
            raise AnalysisError(
                f"Regime {self.name!r} has {len(self.bins)} interior bin(s) and "
                f"{len(self.labels)} label(s); there must be exactly one more label than bin"
            )
        if not self.labels[0] or len(set(self.labels)) != len(self.labels):
            raise AnalysisError(f"Regime {self.name!r} labels must be non-empty and unique")

    # ------------------------------------------------------------------ labelling
    def label(self, values: Any) -> pd.Series:
        """Apply this rule to ``values`` and return a labelled :class:`~pandas.Series`.

        Buckets are cut on **ranks**, not on the values themselves.  That buys
        three properties the report depends on:

        * *Balance* - every bucket holds about the same number of rows, so a
          regime comparison is not secretly a comparison of two sample sizes.  A
          fixed-threshold cut (the ``vol > 0.03`` rule) would put 97% of crypto
          rows in one bucket and make every other bucket empty.
        * *Order invariance* - the label is a function of the value and the
          sample's rank distribution only, so shuffling the input cannot move a
          value between buckets.  Ties share their average rank, so a tied block
          is never split across two regimes.
        * *No leakage* - only the unconditional distribution of the column is
          used, and the caller supplies a trailing value for each timestamp.  The
          target is never read.

        A non-finite input becomes a ``NaN`` label rather than being sorted into
        a bucket: an unknown state is not the lowest state.
        """
        numeric = _as_numeric(values, name=self.column)
        finite = np.isfinite(numeric.to_numpy(dtype="float64"))
        if not finite.any():
            logger.warning("regime %s received no finite values; all labels are NaN", self.name)
            return pd.Series(np.nan, index=numeric.index, dtype="object", name=self.column)

        # `rank(pct=True)` is in (0, 1] and is computed on finite values only, so
        # the NaN rows are absent from the ranking rather than being placed at an
        # arbitrary position.  Assignment is positional, not `.loc`-based, so a
        # duplicated index cannot make the labels land on the wrong rows.
        quantiles = numeric[finite].rank(method="average", pct=True).to_numpy(dtype=float)
        # `right=True` puts a value exactly on a cut point in the lower bucket,
        # which is what qcut does and what makes the edge case reproducible.
        bucket = np.digitize(quantiles, np.asarray(self.bins, dtype=float), right=True)
        labels = np.full(len(numeric), np.nan, dtype="object")
        labels[finite] = [self.labels[i] for i in bucket]
        logger.debug(
            "regime %s: %d row(s) over %d bucket(s) from column %r",
            self.name, int(finite.sum()), len(self.labels), self.column,
        )
        return pd.Series(labels, index=numeric.index, name=self.column)

    def describe(self) -> str:
        """One-line human description, for the report header."""
        edges = ", ".join(f"{e:g}" for e in self.bins)
        return (
            f"{self.name} on {self.column}: rank cut at [{edges}] -> "
            + " < ".join(self.labels)
        )


def volatility_regime(n_bins: int = 4, column: str = "realised_vol") -> RegimeDefinition:
    """The volatility regime rule: ``n_bins`` equal-count buckets, calmest first.

    Quartiles by default.  Volatility is the single most useful conditioning
    variable for a forward-return model, and equal-count buckets are what make
    the conditional table readable: a calm bucket that holds 3% of rows says
    nothing, while a calm bucket holding 25% of them answers "does the edge
    survive when the market is quiet?".
    """
    return _build_definition("volatility", "vol", column, n_bins)


def trend_regime(n_bins: int = 4, column: str = "trend") -> RegimeDefinition:
    """The trend regime rule: ``n_bins`` equal-count buckets, most bearish first.

    Quartiles rather than a bull/bear/sideways trichotomy because the buckets are
    then the same size as the volatility ones, so the two regime tables can be
    read against each other without the reader having to remember that one of them
    has an empty middle.
    """
    return _build_definition("trend", "trend", column, n_bins)


def _build_definition(kind: str, token: str, column: str, n_bins: int) -> RegimeDefinition:
    """Assemble a :class:`RegimeDefinition` with even interior quantile edges."""
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 1:
        raise AnalysisError(f"n_bins must be an integer >= 1, got {n_bins!r}")
    bins = tuple(float(q) for q in np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    labels = tuple(f"{token}_q{i + 1}" for i in range(n_bins))
    return RegimeDefinition(
        name=f"{kind}_quantile_{n_bins}", column=column, bins=bins, labels=labels
    )


#: Documented default volatility rule - quartile of a trailing realised vol.
DEFAULT_VOLATILITY_REGIME: RegimeDefinition = volatility_regime()
#: Documented default trend rule - quartile of a trailing signed trend measure.
DEFAULT_TREND_REGIME: RegimeDefinition = trend_regime()


def label_volatility_regimes(
    values: Any, column: str = "realised_vol", n_bins: int = 4
) -> pd.Series:
    """Rank-quantile volatility regimes, labelled ``vol_q1`` (calmest) upward.

    Parameters
    ----------
    values:
        A trailing, non-negative dispersion series (realised vol, ATR, rolling
        std).  Anything non-numeric is coerced to ``NaN`` rather than raising, so
        a feature with a warm-up hole labels that hole as unknown.
    column:
        Name recorded on the returned series and used in the log line.  It is
        *not* looked up in ``values``: this function labels a series it is given.
    n_bins:
        Number of equal-count buckets.

    Returns
    -------
    pandas.Series
        Object labels aligned to the input index; ``NaN`` where the input was not
        finite.  The target column is never consulted, by design - see the module
        docstring.
    """
    return volatility_regime(n_bins=n_bins, column=column).label(values)


def label_trend_regimes(values: Any, column: str = "trend", n_bins: int = 4) -> pd.Series:
    """Rank-quantile trend regimes, labelled ``trend_q1`` (most bearish) upward.

    Same contract as :func:`label_volatility_regimes` on a *signed* series: the
    only difference is the label prefix and the reading order, so ``trend_q4`` is
    the strongest uptrend and ``trend_q1`` the strongest downtrend.
    """
    return trend_regime(n_bins=n_bins, column=column).label(values)


def regime_label(values: Any, column: str, n_bins: int = 4) -> pd.Series:
    """Label regimes, choosing the volatility or trend rule for ``column``.

    Dispatch order, and why it is in this order:

    1. **The column name**, matched against :data:`VOLATILITY_TOKENS` and
       :data:`TREND_TOKENS`.  A name is the only signal that is stable across
       datasets and available before the values are inspected.
    2. **The sign structure**, used only when the name says nothing.  A series
       with no negative values cannot be a trend measure, so it is treated as a
       dispersion measure; a two-sided series is treated as a trend measure.

    A name matching *both* vocabularies (``trend_vol_adjusted``) is ambiguous and
    raises :class:`AnalysisError` rather than being resolved by coin flip: the
    two rules produce labels that read identically in a report column and would be
    silently swapped.
    """
    if not isinstance(column, str) or not column.strip():
        raise AnalysisError("regime_label needs a non-empty column name to dispatch on")
    text = column.strip().lower()

    is_vol = any(token in text for token in VOLATILITY_TOKENS)
    is_trend = any(token in text for token in TREND_TOKENS)
    if is_vol and is_trend:
        raise AnalysisError(
            f"Column {column!r} matches both the volatility and trend vocabularies; "
            "name it after the quantity it holds so the regime rule is unambiguous"
        )
    if is_vol:
        return label_volatility_regimes(values, column=column, n_bins=n_bins)
    if is_trend:
        return label_trend_regimes(values, column=column, n_bins=n_bins)

    numeric = _as_numeric(values, name=column)
    live = numeric.dropna()
    if live.empty:
        logger.warning(
            "regime_label cannot infer a rule for %r from empty values; using the "
            "volatility rule", column,
        )
        return label_volatility_regimes(values, column=column, n_bins=n_bins)
    two_sided = bool((live < 0.0).any() and (live > 0.0).any())
    if two_sided:
        logger.debug("column %r has no regime token but is two-sided; treating as trend", column)
        return label_trend_regimes(values, column=column, n_bins=n_bins)
    logger.debug("column %r has no regime token but is non-negative; treating as volatility", column)
    return label_volatility_regimes(values, column=column, n_bins=n_bins)


# ------------------------------------------------------------------- per-regime

#: Columns of :func:`evaluate_by_regime`, in order.  ``std_actual_return`` is the
#: one column beyond the required set and is here for a reason: within-regime
#: dispersion is what says whether a regime is where the return is, and it is
#: ``NaN`` for a single-row regime (sample ``ddof=1``), which is the honest
#: "unknown" rather than a confident 0.0.
REGIME_COLUMNS: tuple[str, ...] = (
    "regime",
    "n",
    "rmse",
    "mae",
    "direction_accuracy",
    "mean_actual_return",
    "mean_predicted",
    "hit_rate_ci_lower",
    "hit_rate_ci_upper",
    "std_actual_return",
)


def evaluate_by_regime(
    y_true: Any, y_pred: Any, regimes: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Score one prediction per regime, with a Wilson interval on the hit rate.

    Parameters
    ----------
    y_true, y_pred:
        Row-aligned realised and predicted forward returns.  Rows where either
        side is not finite are dropped before the per-regime split, and the
        surviving count is the ``n`` reported for each regime - so a filtered
        metric can never be mistaken for one computed on the full sample.
    regimes:
        Labels aligned to ``y_true``'s index, as a :class:`~pandas.Series` or as a
        single-column :class:`~pandas.DataFrame`.  A frame with several columns
        uses the first non-numeric one, because a regime label is categorical by
        definition.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`REGIME_COLUMNS`, one row per regime in sorted label order.
        A one-row regime reports ``n == 1``, its error metrics, a hit rate of 0 or
        1, a valid (very wide) Wilson interval, and ``NaN`` dispersion - not an
        exception and not a fake ``0.0``.

    Notes
    -----
    Rows whose label is ``NaN`` are dropped and counted in a debug log rather than
    emitted as a pseudo-regime: "unlabelled" is a property of the feature's warm-up
    window, not a market state, and giving it a row in a regime table invites
    reading it as one.
    """
    truth = _as_series(y_true, name="y_true")
    point = _as_series(y_pred, name="y_pred")
    if len(truth) != len(point):
        raise AnalysisError(
            f"y_true has {len(truth)} rows but y_pred has {len(point)}; they must be aligned"
        )

    labels = _as_label_series(regimes, index=truth.index, n_rows=len(truth))
    truth_values = pd.to_numeric(truth, errors="coerce").to_numpy(dtype=float)
    point_values = pd.to_numeric(point, errors="coerce").to_numpy(dtype=float)

    scored = np.isfinite(truth_values) & np.isfinite(point_values)
    raw_labels = labels.to_numpy(dtype=object)
    # Presence is tested on the *raw* label: `str(None)` is the perfectly ordinary
    # string "None", and stringifying first would promote a missing label into a
    # regime literally called "None".
    present = [str(v) for v, keep in zip(raw_labels, scored) if keep and _is_label(v)]
    as_text = np.array([str(v) for v in raw_labels], dtype=object)
    unlabelled = int(scored.sum()) - len(present)
    if unlabelled:
        logger.debug("evaluate_by_regime dropped %d unlabelled row(s)", unlabelled)

    rows: list[dict[str, Any]] = []
    for regime in sorted(set(present)):
        mask = scored & (as_text == regime)
        y = truth_values[mask]
        p = point_values[mask]
        n = int(y.size)

        error = regression_metrics(y, p)
        direction = _direction_hits(y, p)
        hits = int(direction["hits"])
        lower, upper = wilson_interval(hits, n) if n > 0 else (NAN, NAN)
        rows.append({
            "regime": regime,
            "n": n,
            "rmse": error["rmse"],
            "mae": error["mae"],
            "direction_accuracy": direction["direction_accuracy"],
            "mean_actual_return": float(y.mean()) if n else NAN,
            "mean_predicted": float(p.mean()) if n else NAN,
            "hit_rate_ci_lower": lower,
            "hit_rate_ci_upper": upper,
            "std_actual_return": float(np.std(y, ddof=1)) if n > 1 else NAN,
        })

    frame = pd.DataFrame(rows, columns=list(REGIME_COLUMNS))
    if frame.empty:
        frame = frame.astype({c: "float64" for c in REGIME_COLUMNS if c != "regime"})
    logger.info(
        "evaluate_by_regime: %d regime(s) over %d scored row(s)", len(frame), int(scored.sum())
    )
    return frame


def _is_label(value: Any) -> bool:
    """True when a regime label is actually present (``NaN`` is not a regime)."""
    return bool(pd.notna(value))


def _direction_hits(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Direction hit count and rate under the :mod:`src.v3.metrics` policy.

    A prediction of exactly ``0.0`` is "no view" and is never a hit; a target of
    exactly ``0.0`` is a real observation the model failed to call, so it stays in
    the denominator.  The rule is duplicated here rather than imported from
    :func:`src.v3.metrics.direction_metrics` because that function returns a rate
    and the Wilson interval needs the raw ``(successes, n)`` pair.
    """
    n = int(y.size)
    if n == 0:
        return {"hits": 0, "n": 0, "direction_accuracy": NAN}
    up, down = p > 0.0, p < 0.0
    hits = int(((up & (y > 0.0)) | (down & (y < 0.0))).sum())
    return {"hits": hits, "n": n, "direction_accuracy": hits / n}


# ----------------------------------------------------------------- distributions

#: Columns of :func:`return_distribution_report`, in order.
DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "horizon", "column", "n", "mean", "std", "median", "p05", "p25", "p75", "p95",
    "min", "max", "positive_fraction", "skew", "excess_kurtosis",
)


def return_distribution_report(
    frame: pd.DataFrame,
    target_col: str = "target",
    pred_cols: Sequence[str] | None = None,
    horizons: Mapping[str, str] | Sequence[tuple[str, str]] | str | None = None,
) -> pd.DataFrame:
    """Shape of the target and of every prediction, per horizon.

    The context table for every other number in this module.  A forward-return
    model that looks adequate against a target whose ``p95`` is 0.4% and whose
    median is negative has not beaten anything, and the only way to see that is
    to print the target's own distribution next to the predictions'.  ``skew`` and
    ``excess_kurtosis`` are included because both distributions are heavy-tailed:
    a normal-error assumption underneath a RMSE is not conservative.

    Parameters
    ----------
    frame:
        Prediction frame in the :mod:`src.v3.walkforward` schema.
    target_col:
        Column holding the realised forward return.
    pred_cols:
        Point-forecast columns to summarise.  ``None`` means "every ``*_pred``
        column in the schema", which excludes the interval bounds and the two
        probability columns: those are not return distributions and averaging
        ``prob_calibrated`` into a return table would be a category error.
    horizons:
        ``{column: horizon_label}`` (or an equivalent sequence of pairs, or a
        single label for every column) so that a frame holding several stacked
        horizons is reported separately per horizon.  ``None`` labels every row
        ``'all'``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`DISTRIBUTION_COLUMNS`.  The target is always summarised,
        including when ``pred_cols`` is empty - a target-only report is the useful
        diagnostic, not a degenerate one.
    """
    if not isinstance(frame, pd.DataFrame):
        raise AnalysisError(f"frame must be a DataFrame, got {type(frame).__name__}")
    if target_col not in frame.columns:
        raise AnalysisError(
            f"target column {target_col!r} is not in the frame; columns present: "
            f"{list(frame.columns)[:12]}"
        )

    if pred_cols is None:
        columns = [c for c in frame.columns if c != target_col and c.endswith("_pred")]
    else:
        columns = list(pred_cols)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise AnalysisError(f"pred_cols not present in the frame: {missing}")

    labelled = [(target_col, _resolve_horizon(horizons, target_col, "target"))] + [
        (c, _resolve_horizon(horizons, c, _model_of(c))) for c in columns
    ]

    rows: list[dict[str, Any]] = []
    for column, horizon in labelled:
        stats_row = distribution_stats(frame[column].to_numpy(dtype=float))
        skew, kurtosis = _tail_shape(frame[column].to_numpy(dtype=float))
        rows.append({"horizon": horizon, "column": column, **stats_row, "skew": skew,
                     "excess_kurtosis": kurtosis})

    out = pd.DataFrame(rows, columns=list(DISTRIBUTION_COLUMNS))
    logger.info("return_distribution_report: %d column(s) summarised", len(out))
    return out


def _model_of(column: str) -> str:
    """Model name behind a ``{model}_pred`` column, falling back to the column."""
    return column[: -len("_pred")] if column.endswith("_pred") else column


def _resolve_horizon(
    horizons: Mapping[str, str] | Sequence[tuple[str, str]] | str | None,
    column: str,
    default: str,
) -> str:
    """Label for one column from the ``horizons`` argument of a report function."""
    if horizons is None:
        return "all"
    if isinstance(horizons, str):
        return horizons
    if isinstance(horizons, Mapping):
        mapping = dict(horizons)
    elif isinstance(horizons, Sequence):
        mapping = dict(horizons)
    else:
        raise AnalysisError(
            "horizons must be None, a label string, or a {column: label} mapping; "
            f"got {type(horizons).__name__}"
        )
    if column not in mapping:
        raise AnalysisError(
            f"horizons has no label for column {column!r}; every summarised column "
            f"must be labelled, got {sorted(mapping)}"
        )
    return str(mapping[column])


def _tail_shape(values: Any) -> tuple[float, float]:
    """``(skew, excess_kurtosis)`` from SciPy, or ``(NaN, NaN)`` when undefined.

    Three preconditions, because SciPy answers all three with a plausible number
    rather than an error: at least three observations, some dispersion at all
    (which a float-only series of repeated values can report as ``1e-17`` rather
    than ``0.0``), and moments that do not lose all their precision to
    cancellation.  A series that trips the last one raises a ``RuntimeWarning`` from
    inside SciPy, which is turned into ``NaN`` here - an unreportable shape is
    missing information, not a warning the reader has to interpret.
    """
    arr = np.asarray(values, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or float(np.std(arr)) <= 0.0:
        return NAN, NAN
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            return float(stats.skew(arr, bias=False)), float(stats.kurtosis(arr, bias=False))
        except RuntimeWarning as exc:
            logger.debug("distribution shape undefined for a near-constant series: %s", exc)
            return NAN, NAN


# --------------------------------------------------------------------- intervals

#: Columns of :func:`prediction_interval_report`, in order.
INTERVAL_COLUMNS: tuple[str, ...] = (
    "horizon", "model", "mean_fold_coverage", "nominal_coverage", "coverage_gap",
    "mean_interval_width", "median_interval_width", "n_predictions",
    "coverage_ci_lower", "coverage_ci_upper",
)


def prediction_interval_report(result: Any) -> pd.DataFrame:
    """What the split-conformal band actually did, per model.

    ``nominal_coverage`` is the guarantee the band claims; ``mean_fold_coverage``
    is what the held-out folds delivered.  Both are printed together with the gap
    and a Wilson interval, because a coverage number without an error bar on 200
    pooled test points cannot distinguish a well-calibrated band from a lucky one -
    and because printing only the nominal figure is what makes conformal coverage
    unfalsifiable.

    Parameters
    ----------
    result:
        A :class:`src.v3.walkforward.HorizonResult` (duck-typed: this module does
        not import the harness).

    Returns
    -------
    pandas.DataFrame
        Columns :data:`INTERVAL_COLUMNS`, one row per model.  ``n_predictions`` is
        the number of pooled rows that had a finite lower *and* upper bound, which
        is the ``n`` behind the pooled hit count in the Wilson interval.  When the
        harness recorded a per-fold summary it is preferred for
        ``mean_fold_coverage`` (folds are the independent units); the pooled
        hit rate is the fallback, and the Wilson interval is always on the pooled
        count.
    """
    horizon = str(getattr(result, "horizon", "all"))
    predictions = _require_predictions(result)
    summary = _interval_summary_for(result)
    models = _models_of(result, predictions)

    rows: list[dict[str, Any]] = []
    for model in models:
        pred_col = f"{model}_pred"
        lo_col, hi_col = f"{model}_lo", f"{model}_hi"
        have_bounds = pred_col in predictions and lo_col in predictions and hi_col in predictions

        hits = 0
        n_scored = 0
        widths: list[float] = []
        if have_bounds:
            truth = _numeric_column(predictions, "target")
            lo = _numeric_column(predictions, lo_col)
            hi = _numeric_column(predictions, hi_col)
            usable = np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
            inside = usable & (truth >= lo) & (truth <= hi)
            hits = int(inside.sum())
            n_scored = int(usable.sum())
            widths = list(np.asarray(hi - lo, dtype=float)[usable])

        pooled_coverage = hits / n_scored if n_scored else NAN
        fold_coverage = _lookup(summary, model, "mean_fold_coverage", pooled_coverage)
        nominal = _lookup(summary, model, "nominal_coverage", NAN)
        measured = np.isfinite(fold_coverage) and np.isfinite(nominal)
        coverage_gap = fold_coverage - nominal if measured else NAN
        ci_lower, ci_upper = wilson_interval(hits, n_scored) if n_scored > 0 else (NAN, NAN)

        rows.append({
            "horizon": horizon,
            "model": model,
            "mean_fold_coverage": fold_coverage,
            "nominal_coverage": nominal,
            "coverage_gap": coverage_gap,
            "mean_interval_width": _lookup(
                summary, model, "mean_interval_width", float(np.mean(widths)) if widths else NAN
            ),
            "median_interval_width": float(np.median(widths)) if widths else NAN,
            "n_predictions": n_scored,
            "coverage_ci_lower": ci_lower,
            "coverage_ci_upper": ci_upper,
        })

    out = pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))
    logger.info("prediction_interval_report: %d model(s) on %s", len(out), horizon)
    return out


# ------------------------------------------------------------------- calibration

#: Columns of :func:`calibration_report`, in order.
CALIBRATION_COLUMNS: tuple[str, ...] = (
    "horizon", "model", "source", "n", "brier", "n_bins", "bin_coverage_mae",
    "calibration_slope", "reliability_low", "reliability_high", "degenerate",
)


def calibration_report(result: Any, n_bins: int = 10) -> pd.DataFrame:
    """Score both probability routes, per model, and flag the degenerate one.

    Two routes exist in V3 and they are not interchangeable:

    * ``empirical`` - the empirical CDF of *this model's own* out-of-sample
      residuals, ``{model}_prob_empirical``.  Per model, because residuals are.
    * ``calibrated`` - the shared direction classifier, ``prob_calibrated``.  One
      model per fold, fitted independently of whichever regressor is being scored,
      so its row is deliberately **identical for every model** in the table.  That
      repetition is the finding: the direction view carries no per-regressor
      information, and a reader who sees the same Brier on three rows should not
      read it as three independent confirmations.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CALIBRATION_COLUMNS`, one row per ``(model, source)``.

        ``bin_coverage_mae`` is the mean absolute gap between the mean predicted
        probability and the observed up-frequency across the *populated* bins of
        :func:`src.v3.metrics.calibration_table` - the scalar form of the
        reliability curve.  ``reliability_low`` and ``reliability_high`` are that
        same gap at the lowest and highest populated bins, because the ends of the
        curve are where a confident model is either right or badly wrong.
        ``calibration_slope`` is the OLS slope of the binary up/outcome on the
        probability: 1.0 is perfect, 0.0 means the probability carries no ranking
        information, and ``NaN`` means it is constant.

        ``degenerate`` is ``True`` when every probability in the row is identical
        (the classic always-0.5 output of a classifier that failed to fit or a
        fold with a single target class).  Such a row still has a computable Brier,
        which is exactly why it needs a flag: 0.25 looks like a result and means
        "this model said nothing".  A row with no scorable rows at all is flagged
        too, since every one of its statistics is ``NaN``.
    """
    horizon = str(getattr(result, "horizon", "all"))
    predictions = _require_predictions(result)
    models = _models_of(result, predictions)

    rows: list[dict[str, Any]] = []
    for model in models:
        empirical_col = f"{model}_prob_empirical"
        if empirical_col in predictions:
            rows.append({
                "horizon": horizon, "model": model, "source": "empirical",
                **_calibration_row(predictions, empirical_col, n_bins),
            })
        if "prob_calibrated" in predictions:
            rows.append({
                "horizon": horizon, "model": model, "source": "calibrated",
                **_calibration_row(predictions, "prob_calibrated", n_bins),
            })

    out = pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))
    if not out.empty:
        flagged = out.loc[out["degenerate"], ["model", "source"]]
        for model, source in flagged.itertuples(index=False):
            logger.warning(
                "calibration_report: %s/%s probabilities are constant; the Brier score "
                "on that row is not evidence of calibration", model, source,
            )
    return out


def _calibration_row(predictions: pd.DataFrame, column: str, n_bins: int) -> dict[str, Any]:
    """One ``(model, source)`` row of :func:`calibration_report`."""
    prob = _numeric_column(predictions, column)
    truth = _numeric_column(predictions, "target")

    finite = np.isfinite(prob) & np.isfinite(truth)
    n = int(finite.sum())
    p = prob[finite]
    y = (truth[finite] > 0.0).astype(float)

    out_of_range = int(((p < 0.0) | (p > 1.0)).sum())
    if out_of_range:
        raise AnalysisError(
            f"{column!r} holds {out_of_range} value(s) outside [0, 1]. A regression "
            "output is not a probability: calibrate it before scoring it with a Brier."
        )
    if n == 0:
        return {
            "n": 0, "brier": NAN, "n_bins": int(n_bins), "bin_coverage_mae": NAN,
            "calibration_slope": NAN, "reliability_low": NAN, "reliability_high": NAN,
            "degenerate": True,
        }

    degenerate = bool(np.ptp(p) <= 0.0)
    brier = brier_score(p, y)
    table = calibration_table(p, y, n_bins=n_bins)
    gap = (table["mean_predicted"] - table["observed_frequency"]).abs()
    populated = table["n"] > 0
    gaps = gap[populated].to_numpy(dtype=float)

    return {
        "n": n,
        "brier": brier,
        "n_bins": int(n_bins),
        "bin_coverage_mae": float(gaps.mean()) if gaps.size else NAN,
        "calibration_slope": _calibration_slope(p, y),
        "reliability_low": float(gaps[0]) if gaps.size else NAN,
        "reliability_high": float(gaps[-1]) if gaps.size else NAN,
        "degenerate": degenerate,
    }


def _calibration_slope(prob: np.ndarray, labels: np.ndarray) -> float:
    """OLS slope of the binary outcome on the probability.

    ``NaN`` for fewer than two rows or for a constant probability.  ``scipy``'s
    ``linregress`` raises on a zero-variance ``x`` and pandas' ``corr`` warns; a
    constant probability means the model expressed no view, and its slope is
    undefined - not 0.0, which would read as "the probability is uninformative"
    rather than "there is no probability".
    """
    if prob.size < 2:
        return NAN
    spread = float(np.ptp(prob))
    if spread <= 0.0:
        return NAN
    slope = np.polyfit(prob, labels, 1)[0]
    return float(slope) if np.isfinite(slope) else NAN


# ----------------------------------------------------------------------- deciles

#: Columns of :func:`decile_report`, in order.
DECILE_COLUMNS: tuple[str, ...] = (
    "horizon", "model", "n", "n_deciles", "top_decile_return", "bottom_decile_return",
    "long_short_spread", "spearman_ic",
)


def decile_report(result: Any, n_deciles: int = 10) -> pd.DataFrame:
    """The economically meaningful version of a prediction: what its tails earned.

    ``rmse`` says how far a forecast was from the truth; it does not say whether
    ranking by the forecast would have made money.  This table is the one that
    answers the second question: the mean realised return in the top and bottom
    buckets, the tradable long-minus-short spread, and the rank IC.

    ``top_decile_return`` and ``bottom_decile_return`` come from
    :func:`src.v3.metrics.decile_analysis` with ``n_deciles=10`` - the extreme 10%
    each side, where a rank edge is either visible or absent.  ``long_short_spread``
    comes from :func:`src.v3.metrics.spread_analysis`, whose quintile tails are the
    larger, lower-variance legs; the two deliberately use different widths, so
    read the spread as the tradable number and the decile pair as the shape test.
    On a sample too small to fill ten buckets ``n_deciles`` reports the count
    actually used, so a four-bucket "decile" cannot be quoted as a decile result.
    """
    horizon = str(getattr(result, "horizon", "all"))
    predictions = _require_predictions(result)
    truth = _numeric_column(predictions, "target")
    models = _models_of(result, predictions)

    rows: list[dict[str, Any]] = []
    for model in models:
        pred_col = f"{model}_pred"
        if pred_col not in predictions:
            continue
        point = _numeric_column(predictions, pred_col)
        keep = np.isfinite(truth) & np.isfinite(point)
        y, p = truth[keep], point[keep]

        ladder = decile_analysis(y, p, n_deciles=n_deciles)
        spread = spread_analysis(y, p)
        used = int(ladder["n_deciles"].iloc[0]) if not ladder.empty else 0
        means = ladder["mean_actual_return"].to_numpy(dtype=float) if used else np.array([])
        rows.append({
            "horizon": horizon,
            "model": model,
            "n": int(y.size),
            "n_deciles": used,
            "top_decile_return": float(means[-1]) if used else NAN,
            "bottom_decile_return": float(means[0]) if used else NAN,
            "long_short_spread": spread["spread"],
            "spearman_ic": information_coefficient(y, p),
        })

    out = pd.DataFrame(rows, columns=list(DECILE_COLUMNS))
    logger.info("decile_report: %d model(s) on %s", len(out), horizon)
    return out


# ---------------------------------------------------------------------- ablation

#: Columns of :func:`run_ablation`, in order.
ABLATION_COLUMNS: tuple[str, ...] = (
    "variant", "dropped", "n_features", "n", "rmse", "mae", "spearman_ic",
    "delta_rmse", "delta_mae", "delta_spearman_ic",
)


def run_ablation(
    base_result: Any,
    fit_predict: Callable[[Sequence[str], int], Any],
    feature_groups: Mapping[str, Sequence[str]] | Sequence[Any],
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-one-group-out: does each feature group still earn its place?

    For every group, the group is removed, the model is refitted on what is left
    and the variant is scored **against the full model refitted through the same
    code path**.  The full variant is itself a fit rather than
    ``base_result``'s reported number, because a delta measured between two
    different pipelines measures the pipelines, not the features.

    Parameters
    ----------
    base_result:
        A :class:`src.v3.walkforward.HorizonResult`-like object; only
        ``predictions['target']`` and ``feature_columns`` are read.
    fit_predict:
        ``fit_predict(features, seed) -> y_pred``, supplied by the caller.  It
        receives the *retained column names* and must return a prediction aligned
        to ``base_result.predictions.index``.  Injecting it is what keeps this
        module import-clean of :mod:`src.v3.walkforward` - the harness owns fitting
        and this layer owns reporting, and neither imports the other.
    feature_groups:
        ``{group_name: [columns]}``, or a sequence of objects exposing ``.name``
        and ``.features`` (the repository's ``FeatureGroup``), or a sequence of
        ``(name, columns)`` pairs.  Dropping a group drops exactly the columns it
        lists, whatever the dataset calls them.
    seed:
        Forwarded to ``fit_predict`` unchanged for every variant, so a
        non-deterministic fitter cannot make the comparison a coin flip.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`ABLATION_COLUMNS`, one row per variant: ``'full'`` first
        (with ``dropped is None`` and all deltas ``0.0``), then one ``'drop_<group>'``
        row each.  ``delta_*`` is ``variant - full``, so a **positive**
        ``delta_rmse`` means dropping the group hurt - the reading direction that
        matches "does it earn its place".  Every variant is fitted exactly once and
        scored on the same rows, so a delta is a paired comparison.
    """
    predictions = _require_predictions(base_result)
    if not callable(fit_predict):
        raise AnalysisError(
            "run_ablation needs a fit_predict(features, seed) callable; this module "
            "does not import the walk-forward harness, by design"
        )

    groups = _normalise_groups(feature_groups)
    all_features = [c for c in getattr(base_result, "feature_columns", []) or []]
    if not all_features:
        raise AnalysisError(
            "base_result.feature_columns is empty; an ablation needs the full feature "
            "list to drop groups from"
        )
    unknown = sorted({c for cols in groups.values() for c in cols} - set(all_features))
    if unknown:
        raise AnalysisError(
            f"feature_groups name columns that are not in feature_columns: {unknown[:8]}"
        )

    variants: list[tuple[str, str | None, list[str]]] = [("full", None, list(all_features))]
    for name, columns in groups.items():
        dropped = set(columns)
        variants.append((f"drop_{name}", name, [c for c in all_features if c not in dropped]))

    truth = _numeric_column(predictions, "target")
    rows: list[dict[str, Any]] = []
    for variant, dropped, features in variants:
        if dropped is not None and not features:
            logger.warning(
                "ablation variant %r has no features left; calling fit_predict anyway so "
                "every variant is fitted exactly once", variant,
            )
        # Exactly one call per variant.  A second call would make a stochastic
        # fitter's delta indistinguishable from its own seed noise.
        y_pred = fit_predict(list(features), int(seed))
        point = np.asarray(y_pred, dtype="float64").ravel()
        if point.size != truth.size:
            raise AnalysisError(
                f"fit_predict returned {point.size} predictions for the {variant!r} "
                f"variant but the result has {truth.size} rows; it must be aligned to "
                "base_result.predictions.index"
            )

        keep = np.isfinite(truth) & np.isfinite(point)
        error = regression_metrics(truth[keep], point[keep])
        rows.append({
            "variant": variant,
            "dropped": dropped,
            "n_features": len(features),
            "n": int(keep.sum()),
            "rmse": error["rmse"],
            "mae": error["mae"],
            "spearman_ic": information_coefficient(truth[keep], point[keep]),
        })

    out = pd.DataFrame(rows, columns=list(ABLATION_COLUMNS))
    # `rows[0]` is the full variant by construction, so every delta is measured
    # against one common reference and the full row's deltas are exactly 0.0.
    full = out.iloc[0]
    deltas = (("rmse", "delta_rmse"), ("mae", "delta_mae"), ("spearman_ic", "delta_spearman_ic"))
    for absolute, delta in deltas:
        out[delta] = out[absolute] - full[absolute]
    logger.info(
        "run_ablation: %d variant(s) over %d feature(s)", len(out), len(all_features)
    )
    return out


def _normalise_groups(
    feature_groups: Mapping[str, Sequence[str]] | Sequence[Any]
) -> dict[str, tuple[str, ...]]:
    """Accept a mapping, a sequence of pairs, or ``FeatureGroup``-like objects."""
    if isinstance(feature_groups, Mapping):
        pairs = list(feature_groups.items())
    else:
        pairs = []
        for entry in feature_groups:
            if isinstance(entry, tuple) and len(entry) == 2:
                pairs.append((str(entry[0]), tuple(entry[1])))
            elif hasattr(entry, "name") and hasattr(entry, "features"):
                pairs.append((str(entry.name), tuple(entry.features)))
            else:
                raise AnalysisError(
                    "feature_groups must be a {name: [columns]} mapping or a sequence of "
                    f"FeatureGroup-like objects; got an entry of type {type(entry).__name__}"
                )
    if not pairs:
        raise AnalysisError("feature_groups is empty; there is nothing to ablate")
    out: dict[str, tuple[str, ...]] = {}
    for name, columns in pairs:
        if name in out:
            raise AnalysisError(f"Duplicate feature group {name!r} in feature_groups")
        out[str(name)] = tuple(str(c) for c in columns)
    return out


# ---------------------------------------------------------------------- analyse

#: Keys of the frame dict returned by :func:`analyse`, in order.
ANALYSIS_REPORTS: tuple[str, ...] = (
    "regime_volatility", "regime_trend", "distribution", "intervals", "calibration", "deciles",
)


def analyse(
    result: Any,
    n_bins: int = 4,
    *,
    regime_values: pd.Series | None = None,
    trend_values: pd.Series | None = None,
    volatility_column: str = "realised_vol",
    trend_column: str = "trend",
) -> dict[str, pd.DataFrame]:
    """Every report in this module for one horizon result, in one call.

    Parameters
    ----------
    result:
        A :class:`src.v3.walkforward.HorizonResult` (duck-typed).
    n_bins:
        Regime bucket count.  Separate from the calibration bin count, which is
        fixed by :func:`calibration_report`'s equal-width grid - mixing the two
        would make ``n_bins`` mean two different things in one call.
    regime_values, trend_values:
        Optional trailing series to condition on.  A ``HorizonResult`` pools
        *predictions*, not features, so the conditioning column normally lives in
        the dataset frame the caller still holds; passing it in keeps this module
        from guessing.  When omitted, a column of that name on
        ``result.predictions`` is used if present.
    volatility_column, trend_column:
        Default column names to look for on ``result.predictions``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys :data:`ANALYSIS_REPORTS` - always all six, possibly empty, so a
        caller can index the dict without a ``.get`` dance.  The two regime frames
        are scored on the best pooled model (lowest ``rmse``), and carry a
        ``model`` column naming it: the regime question is "does the edge survive
        in this state", and that is a question about one tradeable model, not
        about all of them averaged together.

    Notes
    -----
    The regime frames come first in the dict because that is the reading order.
    For a long horizon the value of the model is almost entirely conditional -
    a pooled ``r2`` for 30d can rest on a handful of high-volatility months, and
    a table that shows only the pooled number would call that a result.
    """
    predictions = _require_predictions(result)
    horizon = str(getattr(result, "horizon", "all"))
    model = _primary_model(result, predictions)

    reports: dict[str, pd.DataFrame] = {}

    truth = pd.Series(_numeric_column(predictions, "target"), index=predictions.index, name="target")
    pred_col = f"{model}_pred"
    point = pd.Series(_numeric_column(predictions, pred_col), index=predictions.index, name=pred_col)

    for key, source, column, labeller in (
        ("regime_volatility", regime_values, volatility_column, label_volatility_regimes),
        ("regime_trend", trend_values, trend_column, label_trend_regimes),
    ):
        labels = _regime_labels(predictions, source, column, n_bins, labeller)
        if labels is None:
            reports[key] = _empty_regime_frame()
            continue
        frame = evaluate_by_regime(truth, point, labels)
        frame.insert(0, "model", model)
        reports[key] = frame

    reports["distribution"] = return_distribution_report(
        predictions, target_col="target", horizons={c: horizon for c in _distributable(predictions)}
    )
    reports["intervals"] = prediction_interval_report(result)
    reports["calibration"] = calibration_report(result)
    reports["deciles"] = decile_report(result)

    logger.info(
        "analyse: %s model=%s regimes=%s/%s",
        horizon, model, reports["regime_volatility"].shape[0], reports["regime_trend"].shape[0],
    )
    return {key: reports[key] for key in ANALYSIS_REPORTS}


# ------------------------------------------------------------------------ helpers

def _as_numeric(values: Any, name: str = "values") -> pd.Series:
    """Coerce to a float :class:`~pandas.Series`, turning junk into ``NaN``.

    A feature with a warm-up hole or a stray string is missing data, not an
    exception: a report that dies on one bad cell teaches the reader nothing about
    the model, while a report that silently coerces teaches them something false.
    So the coercion is permissive and the loss is logged and visible as ``NaN``
    rows in the output.
    """
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise AnalysisError(
                f"{name} was given a {values.shape[1]}-column frame; pass a Series or an array"
            )
        series = values.iloc[:, 0]
    else:
        array = np.asarray(values, dtype="float64")
        if array.ndim > 1:
            raise AnalysisError(f"{name} must be 1-D, got shape {array.shape}")
        series = pd.Series(array)

    numeric = pd.to_numeric(series, errors="coerce")
    alive = series.notna().to_numpy()
    broken = ~np.isfinite(numeric.to_numpy(dtype="float64")) & alive
    if broken.any():
        logger.debug("%s: %d value(s) were not finite numbers and became NaN", name, int(broken.sum()))
    return numeric.astype("float64")


def _as_series(values: Any, name: str) -> pd.Series:
    """Coerce to a 1-D :class:`~pandas.Series` keeping the original index."""
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise AnalysisError(
                f"{name} was given a {values.shape[1]}-column frame; pass a Series or an array"
            )
        return values.iloc[:, 0]
    array = np.asarray(values)
    if array.ndim > 1:
        raise AnalysisError(f"{name} must be 1-D, got shape {array.shape}")
    return pd.Series(array)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Column of a prediction frame as a float array; missing column -> all NaN."""
    if column not in frame.columns:
        return np.full(len(frame), NAN, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _as_label_series(regimes: pd.Series | pd.DataFrame, index: pd.Index, n_rows: int) -> pd.Series:
    """Extract the categorical regime labels aligned to ``index``."""
    if isinstance(regimes, pd.DataFrame):
        categorical = [c for c in regimes.columns if not pd.api.types.is_numeric_dtype(regimes[c])]
        if not categorical:
            raise AnalysisError(
                "regimes frame has no categorical column; pass a label Series or a frame "
                "whose first column holds the regime labels"
            )
        chosen = categorical[0]
        if len(regimes.columns) > 1 and chosen != regimes.columns[0]:
            logger.debug("regimes frame: using column %r, not the first column", chosen)
        series = regimes[chosen]
    elif isinstance(regimes, pd.Series):
        series = regimes
    else:
        array = np.asarray(regimes, dtype="object").ravel()
        if array.size != n_rows:
            raise AnalysisError(
                f"regimes has {array.size} label(s) but y_true has {n_rows} row(s); "
                "they must be aligned"
            )
        series = pd.Series(array, index=index)

    if not series.index.equals(index):
        series = series.reindex(index)
        if series.isna().all():
            raise AnalysisError(
                "regimes index does not line up with y_true; reindex the labels onto the "
                "prediction index before evaluating"
            )
    return series.astype(object)


def _require_predictions(result: Any) -> pd.DataFrame:
    """The prediction frame of a horizon result, or an explanation of its absence."""
    frame = getattr(result, "predictions", None)
    if not isinstance(frame, pd.DataFrame):
        raise AnalysisError(
            f"{type(result).__name__} has no predictions frame; every report in this "
            "module consumes src.v3.walkforward.HorizonResult.predictions"
        )
    return frame


def _models_of(result: Any, predictions: pd.DataFrame) -> list[str]:
    """Model names to report on, in the harness's own order where possible.

    Prefers ``result.models`` and falls back to discovering ``{model}_pred``
    columns.  The fallback matters: a result whose point forecasts are all NaN
    still has a model column, and dropping it would silently shrink the table.
    """
    declared = [str(m) for m in (getattr(result, "models", None) or [])]
    discovered = sorted({c[: -len("_pred")] for c in predictions.columns if c.endswith("_pred")})
    ordered = declared + [m for m in discovered if m not in declared]
    if not ordered:
        raise AnalysisError(
            "no model found: the frame has no '{model}_pred' column and the result "
            "declares no models"
        )
    return ordered


#: Models that emit a single constant for every row.  They are the benchmarks
#: a real model has to beat, and they win the pooled RMSE more often than anyone
#: expects, but a constant has no view to survive or fail in a regime: scoring
#: it by regime only describes how the market itself behaved.  So they are kept
#: out of the regime tables when a model with an actual view is available.
CONSTANT_MODELS = ("zero", "mean", "trailing_mean", "last", "naive")


def _primary_model(result: Any, predictions: pd.DataFrame) -> str:
    """The model the regime tables are scored on: best pooled ``rmse``.

    Constant benchmarks are skipped when any other model is present.  A regime
    breakdown answers "does the edge survive in this state", and a forecast that
    is 0.0 for every row has no edge to survive - its per-regime rows would
    describe the market, not the model.  Falling back to the best constant
    keeps the section populated when every available model is one.
    """
    declared = _models_of(result, predictions)
    candidates = [m for m in declared if m not in CONSTANT_MODELS] or list(declared)
    pooled = getattr(result, "pooled_metrics", None)
    if isinstance(pooled, pd.DataFrame) and not pooled.empty and "rmse" in pooled.columns:
        available = [m for m in candidates if m in pooled.index]
        if available:
            return str(pooled.loc[available].sort_values("rmse").index[0])
    return candidates[0]


def _interval_summary_for(result: Any) -> pd.DataFrame:
    """The harness's per-fold interval summary as a frame, or an empty one."""
    summary = getattr(result, "interval_summary", None)
    if isinstance(summary, pd.DataFrame):
        return summary
    if isinstance(summary, pd.Series):
        return summary.to_frame().T
    return pd.DataFrame()


def _lookup(summary: pd.DataFrame, model: str, column: str, default: float) -> float:
    """One value from the per-fold summary for ``model``, or ``default``.

    The harness returns that summary with ``model`` as a *column*
    (``groupby("model").mean().reset_index()``), but an index-keyed frame is an
    equally natural thing for a caller to hold, so both shapes are accepted rather
    than the report silently degrading to ``NaN`` on a present-but-reshaped input.
    """
    if summary.empty or column not in summary.columns:
        return float(default)

    if "model" in summary.columns:
        selected = summary.loc[summary["model"].astype(str) == model, column]
    elif model in summary.index:
        selected = summary.loc[model, column]
    else:
        return float(default)

    if isinstance(selected, pd.Series):
        if selected.empty:
            return float(default)
        selected = selected.iloc[0]
    try:
        number = float(selected)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _regime_labels(
    predictions: pd.DataFrame,
    source: pd.Series | None,
    column: str,
    n_bins: int,
    labeller: Callable[..., pd.Series],
) -> pd.Series | None:
    """Regime labels for the prediction index, or ``None`` when unavailable."""
    if source is not None:
        values = source.reindex(predictions.index) if isinstance(source, pd.Series) else source
    elif column in predictions.columns:
        values = predictions[column]
    else:
        logger.warning(
            "no %r column on result.predictions and none was passed in; the regime "
            "table will be empty. The conditioning feature lives in the dataset frame, "
            "not in the pooled predictions.", column,
        )
        return None
    labels = labeller(values, column=column, n_bins=n_bins)
    if labels.index.equals(predictions.index):
        return labels
    return labels.reindex(predictions.index)


def _distributable(predictions: pd.DataFrame) -> list[str]:
    """Columns that :func:`return_distribution_report` will summarise."""
    return [c for c in predictions.columns if c == "target" or c.endswith("_pred")]


def _empty_regime_frame() -> pd.DataFrame:
    """The documented regime columns, correctly typed, for the no-regimes case."""
    columns = ["model", *REGIME_COLUMNS]
    dtypes = {"model": "object", "regime": "object"}
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "float64")) for c in columns},
                        columns=columns)
