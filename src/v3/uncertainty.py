"""Uncertainty for the V3 forward-return regression: honest intervals and honest probabilities.

V3 predicts a *forward return* over horizons from 1d to 180d.  That output is a
number on a real line, and two things routinely get done to it that it cannot
support:

1. **A positive point forecast is reported as "70% chance of going up".**  It is
   not.  ``predicted_return = 0.004`` is a conditional mean estimate; reading a
   probability off it is a category error that survives into reports because the
   number happens to sit in ``[0, 1]``-looking territory.  This module therefore
   keeps the two questions strictly separate and offers *two* independent routes
   to a probability of going up, so they can be compared and scored rather than
   assumed to agree (see :func:`empirical_probability_positive` and
   :class:`CalibratedDirectionModel`).
2. **Coverage is asserted rather than measured.**  A ``+/- 3 sigma`` band around
   a bootstrapped residual is not a statement about future residuals, and
   nothing in the training log forces it to be checked.  Here the interval is
   the split-conformal interval of :func:`fit_split_conformal`, whose coverage
   guarantee holds out-of-sample by construction - and
   :func:`interval_summary` still *measures* it, so the guarantee is auditable
   rather than trusted.

Intervals are symmetric around the point forecast by construction
(``yhat +/- q``).  That is a property of the residual-absolute-value construction,
not a modelling claim: with heteroskedastic errors the *scale* of the band should
vary with the input, and a single global ``q`` is the honest, simple version of
that.  A mean/quantile-style asymmetric interval would need a model of the
conditional error scale, which this project does not have.

The probability of going up is deliberately provided two ways because they make
*different assumptions*:

* :func:`empirical_probability_positive` - Method A, distributional.  Treats the
  predictive error as a draw from the pooled empirical residual distribution.
  Needs no model, needs enough residuals, and inherits every bias of that
  distribution.
* :class:`CalibratedDirectionModel` - Method B, learned.  Fits a separate
  classifier to the binary direction label and calibrates it out-of-fold.  Needs
  a learnable signal; a well-calibrated constant is a *good* result when the
  signal is absent, and the calibration table is what shows that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from src.utils import get_logger

try:  # pragma: no cover - exercised by whichever branch the sibling module takes
    from src.v3.metrics import brier_score, calibration_table
except ImportError:  # noqa: BLE001
    # `src.v3.metrics` is written by a sibling module.  Importing it unguarded
    # would make this module - and every test of it - unimportable in the window
    # before that file lands.  The names stay referenced so the real
    # implementation is picked up the moment it exists; setting them to None keeps
    # the module importable on its own, and `calibration_report` raises rather
    # than substituting a local scoring function.  A stand-in Brier score would be
    # the worst outcome of the three: it would be the number that ends up in a
    # report, silently differing from the project's own definition.
    brier_score = None  # type: ignore[assignment]
    calibration_table = None  # type: ignore[assignment]

logger = get_logger("v3.uncertainty")

#: Calibration methods :class:`CalibratedDirectionModel` accepts.
CALIBRATION_METHODS = ("isotonic", "sigmoid")

#: Decimal places the conformal rank is snapped to before the ceiling is taken.
#: See :func:`_rank_ceiling` for why the snap is load-bearing.
_RANK_DECIMALS = 10


# --------------------------------------------------------------------------- conformal

def _rank_ceiling(value: float) -> int:
    """``ceil`` after snapping away binary-representation noise.

    Split out so the finite-sample correction has one definition and one test
    rather than being re-derived at each call site.  ``round`` here is not
    cosmetic: the exact value is an integer whenever ``alpha`` is a round
    fraction, and IEEE-754 evaluates those just *above* it.
    """
    return int(np.ceil(round(float(value), _RANK_DECIMALS)))


def conformal_quantile(abs_residuals: Any, alpha: float) -> float:
    """Split-conformal quantile of absolute residuals, with the finite-sample correction.

    The marginal coverage of a split-conformal interval is
    ``>= 1 - alpha`` in finite samples only if the threshold is the
    ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile of ``|residual|`` -
    the ``(n + 1)`` is what accounts for the test point not being in the
    calibration set, and dropping it (the "nice" ``(1 - alpha)`` quantile) makes
    the true coverage strictly below nominal for every finite ``n``.

    Two implementation details are not cosmetic:

    * **The quantile is taken on the sorted values directly.**  ``np.quantile``
      interpolates between order statistics, which would put the threshold
      between two observed residuals rather than on one of them, and the
      finite-sample argument only holds for a genuine order statistic.
    * **The rank is snapped before the ceiling.**  ``(n + 1) * (1 - alpha)`` is
      an integer whenever ``alpha`` is a round fraction, and in IEEE-754 those
      evaluate just *above* it - ``10 * 0.3`` is ``3.0000000000000004``, not
      ``3.0``.  A naive ``ceil`` then picks the next order statistic up, silently
      widening the interval, and for ``alpha`` near zero it selects an index past
      the end of the sample.  Rounding to :data:`_RANK_DECIMALS` recovers the
      mathematical value; the residual error there is ~1e-15, far below any gap
      a real ``alpha`` produces.

    Parameters
    ----------
    abs_residuals:
        Non-negative absolute residuals from the calibration block.  Only the
        magnitudes are used, and the sign of a residual carries no information
        for a symmetric band, so the caller may pass raw residuals too.
    alpha:
        Miscoverage level in ``(0, 1)``; ``0.1`` asks for a 90% interval.

    Returns
    -------
    float
        The order statistic ``sorted(|r|)[k - 1]`` with
        ``k = ceil((n + 1) * (1 - alpha))`` clamped to ``[1, n]``.
    """
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1); got {alpha!r}")

    values = np.asarray(abs_residuals, dtype="float64").ravel()
    if values.size == 0:
        raise ValueError("Cannot build a conformal quantile from an empty residual sample")
    if not np.all(np.isfinite(values)):
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Residuals must all be finite; {n_bad} of {values.size} are NaN or inf. "
            "Drop the affected calibration rows rather than imputing them."
        )

    ordered = np.sort(values)
    rank = _rank_ceiling((ordered.size + 1) * (1.0 - alpha_value))
    # `rank` is mathematically in [1, n+1].  n+1 only occurs for alpha so small
    # that the threshold is effectively infinite, and there the honest interval is
    # the widest one the data supports rather than an out-of-bounds index.
    index = int(min(max(rank, 1), ordered.size)) - 1
    return float(ordered[index])


@dataclass(frozen=True)
class SplitConformal:
    """A fitted symmetric split-conformal band around point forecasts.

    Immutable on purpose: the quantile is the product of one calibration block
    and passing it around between horizons, folds and reports is exactly how a
    band silently gets reused with the wrong ``alpha`` or the wrong residuals.
    """

    alpha: float
    quantile: float
    n_calibration: int

    def interval(self, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` bounds for a vector of point forecasts."""
        point = np.asarray(y_pred, dtype="float64").ravel()
        return point - self.quantile, point + self.quantile

    def interval_for_point(self, y_pred: float) -> tuple[float, float]:
        """Scalar convenience form of :meth:`interval`."""
        point = float(y_pred)
        return point - self.quantile, point + self.quantile

    def width(self) -> float:
        """Interval width.  Constant, because the band is ``yhat +/- q``."""
        return 2.0 * self.quantile

    def coverage(self, y_true: Any, y_pred: Any) -> float:
        """Share of observations that fall inside the band.

        NaN on an empty sample rather than 0.0 or 1.0: an unmeasured coverage is
        not perfect coverage, and reporting it as 0.0 is how a broken evaluation
        ends up looking like a well-calibrated model.  A non-finite target counts
        as *not covered* - an unlabelled row has not been shown to be inside
        anything, and counting it as covered would inflate the number.
        """
        truth = np.asarray(y_true, dtype="float64").ravel()
        point = np.asarray(y_pred, dtype="float64").ravel()
        if truth.size != point.size:
            raise ValueError(
                f"y_true has {truth.size} values but y_pred has {point.size}; coverage needs a matched pair"
            )
        if truth.size == 0:
            return float("nan")
        lower, upper = self.interval(point)
        inside = np.isfinite(truth) & (truth >= lower) & (truth <= upper)
        return float(inside.mean())


def fit_split_conformal(
    y_calib: Any, y_pred_calib: Any, alpha: float = 0.1
) -> SplitConformal:
    """Fit a conformal band on held-out calibration predictions.

    ``y_pred_calib`` must come from predictions the model did not train on.  The
    method assumes nothing about the error distribution - only that calibration
    and test residuals are exchangeable, which is the same assumption
    :mod:`src.v3.splits` earns with purging and embargoing rather than assuming.

    Parameters
    ----------
    y_calib:
        Realised targets on the calibration block, shape ``(n,)``.
    y_pred_calib:
        Model predictions for the same rows, in the same order.
    alpha:
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    SplitConformal
    """
    truth = np.asarray(y_calib, dtype="float64").ravel()
    point = np.asarray(y_pred_calib, dtype="float64").ravel()
    if truth.size != point.size:
        raise ValueError(
            f"y_calib has {truth.size} values but y_pred_calib has {point.size}; they must be paired"
        )
    if truth.size == 0:
        raise ValueError("Cannot fit a conformal band on an empty calibration block")
    if not (np.all(np.isfinite(truth)) and np.all(np.isfinite(point))):
        raise ValueError(
            "Calibration targets and predictions must all be finite. A NaN here is "
            "usually an unlabelled tail row; drop it rather than filling it in."
        )

    residuals = truth - point
    quantile = conformal_quantile(np.abs(residuals), alpha)
    return SplitConformal(alpha=float(alpha), quantile=quantile, n_calibration=int(truth.size))


# --------------------------------------------------------------------------- probability, method A

def empirical_probability_positive(y_pred: Any, residuals: Any) -> np.ndarray:
    """Method A: probability of a positive realised return from the residual distribution.

    If the realised value is ``y = yhat + e`` and the error ``e`` is drawn from
    the empirical distribution of observed residuals ``R``, then::

        P(y > 0 | yhat) = P(e > -yhat)

    which is estimated directly by counting the pooled residuals above
    ``-yhat``.  This is a **distributional assumption, not a guarantee**: the
    returned number is only a probability if a fresh error really is a draw from
    ``R``.

    Assumptions, stated plainly
    ---------------------------
    * **Pooling / exchangeability, and no regime awareness.**  One residual
      distribution is applied to every input.  If error scale grows with
      volatility - which it does in crypto - a single ``R`` is simultaneously too
      wide for calm periods and too narrow for violent ones, so the probability is
      over-confident exactly when it matters most.  Residuals from a calm month
      and a crash are treated as equally likely, which is false during precisely
      the periods where the answer is worth having.  The conformal band above has
      the same limitation, for the same reason.
    * **Symmetry about zero.**  The sign of a residual is treated as
      uninformative about the sign of the outcome beyond its magnitude, i.e.
      ``R`` is treated as symmetric.  On an exactly symmetric residual sample the
      estimator then satisfies ``p(-yhat) + p(yhat) = 1`` to the last bit, so a
      zero forecast gives 0.5.  The one exception is a ``yhat`` that coincides
      exactly with an observed residual, where the strict ``>`` costs one count.
      Real crypto residuals have fat tails and are not symmetric, and a skewed
      ``R`` biases the estimate in the direction of the skew.
    * **Continuity correction.**  ``(#{r > -yhat} + 0.5) / (n + 1)`` rather than
      ``#(r > -yhat) / n``.  The raw ratio can return exactly 0.0 or 1.0, which
      is a *finite-sample artefact* - the largest residual in the sample is not
      evidence that the outcome is impossible - and an exact 0 or 1 would be
      scored by the Brier score as maximal confidence.  The Laplace correction
      keeps every value strictly inside ``(0, 1)`` and stays consistent as
      ``n`` grows.

    Prefer :class:`CalibratedDirectionModel` when a direction signal exists; the
    two disagreeing is information, not a bug to be averaged away.

    Parameters
    ----------
    y_pred:
        Point forecasts, shape ``(n,)``.
    residuals:
        Signed residuals ``y_calib - y_pred_calib`` from held-out predictions.

    Returns
    -------
    np.ndarray
        Probabilities in ``[0, 1]``, non-decreasing in ``y_pred``, shape ``(n,)``.
    """
    point = np.asarray(y_pred, dtype="float64").ravel()
    spread = np.asarray(residuals, dtype="float64").ravel()
    if spread.size == 0:
        raise ValueError(
            "An empty residual sample carries no information about P(y > 0); "
            "at least one held-out residual is required"
        )
    if not np.all(np.isfinite(spread)):
        raise ValueError("Residuals must all be finite to form an empirical distribution")
    if not np.all(np.isfinite(point)):
        raise ValueError("y_pred must be finite to form a probability")

    ordered = np.sort(spread)
    n = ordered.size
    # `searchsorted(..., side="right")` counts residuals <= -yhat, so the count
    # above -yhat is n minus that.  Sorting the residuals once keeps this O(n log n)
    # overall rather than an O(n * m) scan per point.
    at_or_below = np.searchsorted(ordered, -point, side="right")
    count_above = n - at_or_below
    return np.clip((count_above + 0.5) / (n + 1.0), 0.0, 1.0)


# --------------------------------------------------------------------------- probability, method B

class _ClassifierTagOrder(ClassifierMixin, BaseEstimator):
    """Carrier for sklearn's classifier tags with the mixin on the left.

    ``BaseEstimator.__sklearn_tags__`` does not call ``super()``, so with the
    public ``(BaseEstimator, ClassifierMixin)`` base order the mixin's
    implementation is shadowed and the estimator advertises
    ``estimator_type=None``.  That silently changes downstream tool behaviour -
    ``check_cv`` stops stratifying, ``is_classifier`` returns False.  This empty
    carrier exists to hold the tags for the conventional order, so the requested
    base order can be kept without importing sklearn's private ``Tags`` classes.
    """


def _classifier_tags() -> Any:
    """Tags describing a ``ClassifierMixin``-first estimator.

    Taken from a throwaway instance rather than from a method, because
    ``ClassifierMixin.__sklearn_tags__`` reaches ``BaseEstimator`` through
    ``super()`` and the mixin position in ``type(self)``'s own MRO is what makes
    that lookup succeed.
    """
    return _ClassifierTagOrder().__sklearn_tags__()


class CalibratedDirectionModel(BaseEstimator, ClassifierMixin):
    """Method B: a separately fitted, out-of-fold calibrated classifier for ``P(y > 0)``.

    Why this is not ``predicted_return > 0``
    -----------------------------------------
    Thresholding a regressor at zero yields a *class*, never a probability, and
    its implied probability collapses to a step function: the model is 99%
    confident the market goes up and 1% confident it goes down either side of a
    boundary it cannot justify.  The two outputs also fail differently - the
    regression is trained to minimise a squared error on the magnitude, which
    says nothing about how often it gets the sign right.

    This estimator therefore fits a *classifier* on the binary direction label
    and then recalibrates it on out-of-fold predictions, so the number it emits
    is a probability that can be scored with a Brier score and inspected with a
    reliability table.

    Leakage
    -------
    ``fit`` runs an internal cross-validation to produce the calibration folds.
    **Those folds are carved out of the block passed to ``fit``, so that block
    must be the training/validation data only.**  Passing the test block here -
    which is the easy mistake, because the estimator looks self-contained -
    calibrates the model on the data it is then scored on, and the resulting
    Brier score is not an out-of-sample number.  Purging still applies within the
    fitting block, exactly as in :mod:`src.v3.splits`.

    Parameters
    ----------
    base_estimator:
        Classifier to calibrate.  ``None`` uses a shallow
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier`, deliberately
        not a deep one: the direction signal in this project is weak, and a
        flexible base model here mostly buys calibration noise.  The histogram
        variant is chosen over the textbook ``GradientBoostingClassifier`` for a
        concrete reason - on the real 84-feature, ~10k-row folds used by the V3
        harness the two are the same idea, but the histogram version is ~85x
        faster (0.8s vs 65s per fit), which is the difference between a
        runnable pipeline and an unrunnable one.
    method:
        ``'isotonic'`` or ``'sigmoid'``.  Isotonic is more flexible and needs
        more data per fold; sigmoid (Platt) is the safer default on small or
        noisy samples.  Validated in :meth:`fit`, not in ``__init__``, so that
        ``clone`` and grid search keep their contract.  Isotonic regression
        interpolates linearly between the observed out-of-fold scores and so
        reaches exactly ``0.0`` and ``1.0`` on a small sample - a hard ``1.0``
        is a genuine defect of a probability estimate, not a rounding detail.
        Sigmoid maps a bounded score and stays clear of both endpoints.
    n_bins:
        Bins for the calibration diagnostics on
        :meth:`calibration_report`.
    random_state:
        Seed for the default base estimator.

    Attributes
    ----------
    calibrated_:
        The fitted :class:`~sklearn.calibration.CalibratedClassifierCV`.
    n_folds_:
        Number of calibration folds actually used.
    n_imputed_:
        Non-finite feature cells replaced by 0.0, per :meth:`fit`.
    classes_:
        ``[0, 1]``.
    """

    def __init__(
        self,
        base_estimator: Any | None = None,
        method: str = "isotonic",
        n_bins: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.n_bins = n_bins
        self.random_state = random_state

    # ------------------------------------------------------------------ internals

    def _clean(self, X: Any, *, stage: str) -> tuple[np.ndarray, int]:
        """Coerce to a finite 2-D float array, recording what had to be replaced.

        A non-finite feature cell is imputed to 0.0, which is the neutral value
        for a mean-centred feature, and the count is logged and kept on the
        estimator so the substitution cannot pass unnoticed.  Rejecting the batch
        instead would be defensible, but in a pipeline a single NaN from an
        indicator that is undefined on the first bars of a window is a routine
        event, and a hard failure there tends to get "fixed" by dropping rows -
        which is a much larger and quieter intervention.
        """
        frame = np.asarray(X, dtype="float64")
        if frame.ndim == 1:
            frame = frame.reshape(-1, 1)
        if frame.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {frame.shape}")
        bad = ~np.isfinite(frame)
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning(
                "%s: imputing %d non-finite feature cells to 0.0 in a %s array",
                stage,
                n_bad,
                frame.shape,
            )
            frame = np.where(bad, 0.0, frame)
        return frame, n_bad

    def _resolve_base(self) -> Any:
        if self.base_estimator is not None:
            return self.base_estimator
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            random_state=self.random_state,
        )

    def _resolve_folds(self, n_min_class: int) -> int:
        """Calibration folds, bounded by the smallest class count.

        Stratified CV needs at least ``n_splits`` members of every class in each
        training fold, so a rare direction caps the fold count.  Three is the
        ceiling: the fold count is multiplied into every direction fit, and
        beyond three the extra folds buy a noisier per-fold calibration set
        rather than a better-calibrated one.
        """
        return int(max(2, min(3, n_min_class)))

    # ------------------------------------------------------------------ sklearn API

    def __sklearn_tags__(self) -> Any:  # noqa: D105
        return _classifier_tags()

    def fit(self, X: Any, y: Any) -> "CalibratedDirectionModel":
        """Fit the direction classifier and its out-of-fold calibration.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.  Non-finite cells
            are imputed to 0.0 and counted in ``n_imputed_``.
        y:
            Binary direction labels, ``0`` for a non-positive realised return
            and ``1`` for a positive one.  Both classes must be present: a
            single-class ``y`` means the direction is constant over the sample,
            and the only calibratable model in that case is a constant, which
            would be reported as a perfect probability without being one.
        """
        if self.method not in CALIBRATION_METHODS:
            raise ValueError(
                f"method must be one of {CALIBRATION_METHODS}; got {self.method!r}"
            )

        features, n_imputed = self._clean(X, stage="fit")
        labels = np.asarray(y).ravel()
        if labels.size != features.shape[0]:
            raise ValueError(
                f"X has {features.shape[0]} rows but y has {labels.size} values; they must be paired"
            )
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError(
                "y must be binary 0/1 (non-positive / positive realised return); "
                f"got values {np.unique(labels)[:5]}"
            )

        values, counts = np.unique(labels.astype(int), return_counts=True)
        if values.size < 2:
            raise ValueError(
                f"y contains a single class ({int(values[0])}); there is no direction to learn. "
                "Widen the horizon, or report the base rate, instead of fitting a model here."
            )

        n_folds = self._resolve_folds(int(counts.min()))
        self.calibrated_ = CalibratedClassifierCV(
            estimator=self._resolve_base(), method=self.method, cv=n_folds
        )
        self.calibrated_.fit(features, labels.astype(int))
        self.classes_ = np.array([0, 1])
        self.n_folds_ = n_folds
        self.n_imputed_ = n_imputed
        logger.info(
            "direction model fitted: method=%s folds=%d n=%d up_rate=%.3f imputed_cells=%d",
            self.method,
            n_folds,
            features.shape[0],
            float(np.mean(labels)),
            n_imputed,
        )
        return self

    def predict_proba_positive(self, X: Any) -> np.ndarray:
        """Calibrated ``P(realised return > 0)``, shape ``(len(X),)``."""
        check_is_fitted(self, "calibrated_")
        features, _ = self._clean(X, stage="predict")
        probabilities = self.calibrated_.predict_proba(features)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                f"Expected a 2-column probability matrix, got shape {probabilities.shape}"
            )
        return np.clip(probabilities[:, 1], 0.0, 1.0)

    def predict(self, X: Any) -> np.ndarray:
        """Hard direction at the 0.5 cut.  The threshold is fixed, not tuned on test."""
        return (self.predict_proba_positive(X) >= 0.5).astype(int)

    # ------------------------------------------------------------------ diagnostics

    def calibration_report(self, X: Any, y: Any) -> dict[str, Any]:
        """Brier score, base rate and a binned reliability table on a held-out block.

        Parameters
        ----------
        X, y:
            A block the model was *not* fitted on.  ``y`` is the binary direction
            label, and must contain both classes for a reliability table to mean
            anything.

        Returns
        -------
        dict
            ``brier_score``, ``base_rate_up``, ``mean_predicted_probability``,
            ``n`` and ``calibration_table`` (a :class:`pandas.DataFrame`).

        Raises
        ------
        ImportError
            If :mod:`src.v3.metrics` is unavailable.  A locally defined Brier score
            would be the wrong answer to give here, not a helpful fallback: it is
            the number that reaches the report.
        """
        if brier_score is None or calibration_table is None:  # pragma: no cover
            raise ImportError(
                "src.v3.metrics is not available; calibration_report needs its "
                "brier_score and calibration_table"
            )

        probabilities = self.predict_proba_positive(X)
        labels = np.asarray(y).ravel().astype(int)
        if labels.size != probabilities.size:
            raise ValueError(
                f"X has {probabilities.size} rows but y has {labels.size} values; they must be paired"
            )
        if np.unique(labels).size < 2:
            raise ValueError(
                "calibration_report needs both classes in y; a single-class block has no "
                "reliability curve to report"
            )

        return {
            "brier_score": float(brier_score(probabilities, labels)),
            "base_rate_up": float(labels.mean()),
            "mean_predicted_probability": float(probabilities.mean()),
            "n": int(labels.size),
            "calibration_table": calibration_table(
                probabilities, labels, n_bins=self.n_bins
            ),
        }


# --------------------------------------------------------------------------- reporting

def interval_summary(
    y_true: Any, y_pred: Any, conformal: SplitConformal
) -> dict[str, float]:
    """Measured interval performance, for a report or a horizon table.

    The nominal figure is the guarantee the band claims; the empirical figure is
    what actually happened on this block.  Printing only the nominal number is
    what makes conformal coverage unfalsifiable, so both travel together.

    Parameters
    ----------
    y_true, y_pred:
        A matched, held-out pair.
    conformal:
        The band fitted on the calibration block.

    Returns
    -------
    dict
        ``empirical_coverage``, ``nominal_coverage``, ``mean_interval_width``
        and ``n``.
    """
    truth = np.asarray(y_true, dtype="float64").ravel()
    point = np.asarray(y_pred, dtype="float64").ravel()
    if truth.size != point.size:
        raise ValueError(
            f"y_true has {truth.size} values but y_pred has {point.size}; they must be paired"
        )
    return {
        "empirical_coverage": conformal.coverage(truth, point),
        "nominal_coverage": 1.0 - float(conformal.alpha),
        "mean_interval_width": conformal.width(),
        "n": int(truth.size),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the textbook normal approximation because the coverage
    numbers that matter here are exactly where that approximation breaks: a
    measured 90% coverage on 200 test points, or an up-rate near 0.  The normal
    interval runs off the bottom of ``[0, 1]`` (negative width, silently
    discarded when someone clamps it) and has far too little width near the
    boundary.  The Wilson interval stays inside ``[0, 1]`` for every ``k``.

    Reference interval quality is a *binomial* claim and assumes independent
    Bernoulli draws.  For a coverage rate measured on overlapping 30d/180d
    labels that assumption is false - adjacent points share almost all of their
    outcome window - so the true interval is wider than the one returned here.
    It is a floor on the uncertainty, not the whole of it.

    Parameters
    ----------
    successes:
        Number of successes, ``0 <= successes <= n``.
    n:
        Number of trials, ``n > 0``.
    z:
        Normal quantile for the desired level; ``1.96`` is 95%.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` in ``[0, 1]``.
    """
    n_trials = int(n)
    n_successes = int(successes)
    if n_trials <= 0:
        raise ValueError(f"n must be positive; got {n!r}")
    if n_successes < 0 or n_successes > n_trials:
        raise ValueError(
            f"successes must lie in [0, {n_trials}]; got {successes!r}"
        )

    z_value = float(z)
    if not np.isfinite(z_value) or z_value <= 0.0:
        raise ValueError(f"z must be positive and finite; got {z!r}")

    rate = n_successes / n_trials
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / n_trials
    centre = rate + z_squared / (2.0 * n_trials)
    # Written as sqrt of a sum so that rate == 0 gives an exactly zero margin,
    # and the lower bound is then exactly 0.0 rather than 1e-18 off the boundary.
    margin = z_value * np.sqrt(
        rate * (1.0 - rate) / n_trials + z_squared / (4.0 * n_trials * n_trials)
    )
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))
