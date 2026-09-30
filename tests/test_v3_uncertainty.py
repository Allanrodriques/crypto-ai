"""Tests for V3 uncertainty: split-conformal intervals and the two probability routes.

The claims under test are structural rather than numeric wherever possible: the
finite-sample rank correction is checked against a hand-computed order statistic,
and the two probability routes are checked for the property that motivated them -
neither of them is ``predicted_return > 0`` wearing a decimal point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.v3.uncertainty import (
    CalibratedDirectionModel,
    SplitConformal,
    conformal_quantile,
    empirical_probability_positive,
    fit_split_conformal,
    interval_summary,
    wilson_interval,
)

try:  # The sibling module may not exist yet; these tests skip if it does not.
    from src.v3.metrics import brier_score, calibration_table
except ImportError:  # pragma: no cover
    brier_score = None
    calibration_table = None

needs_metrics = pytest.mark.skipif(
    brier_score is None, reason="src.v3.metrics is not importable"
)


def direction_problem(
    n: int = 1400, seed: int = 7, strength: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """A weak, learnable direction signal on four noise features.

    ``strength`` sets the logit coefficients.  The default is deliberately weak
    (roughly 65% accuracy) so that "beats a flat 0.5" is a real result rather
    than a trivially separable toy.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n, 4))
    logit = (
        strength * features[:, 0]
        - 0.6 * strength * features[:, 1]
        + 0.9 * rng.normal(0.0, 1.0, n)
    )
    return features, (logit > 0.0).astype(int)


def small_classifier() -> GradientBoostingClassifier:
    """A deliberately cheap base estimator, so the test suite stays quick."""
    return GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=0)


# --------------------------------------------------------------------------- conformal rank

def test_conformal_quantile_uses_the_finite_sample_correction() -> None:
    """k = ceil((n + 1) * (1 - alpha)), and nothing softer.

    n = 9, alpha = 0.1  ->  ceil(10 * 0.9) = 9  ->  the largest residual.
    Dropping the ``(n + 1)`` would give the 0.9 quantile of nine values, which is
    the eighth: a strictly narrower band whose coverage is below 0.9 in every
    finite sample.
    """
    residuals = np.arange(1.0, 10.0)  # 1..9
    assert conformal_quantile(residuals, 0.1) == 9.0

    # n = 9, alpha = 0.2 -> ceil(10 * 0.8) = 8 -> sorted(|r|)[7].
    assert conformal_quantile(residuals, 0.2) == 8.0

    # n = 9, alpha = 0.5 -> ceil(10 * 0.5) = 5 -> the median order statistic.
    assert conformal_quantile(residuals, 0.5) == 5.0


def test_conformal_quantile_differs_from_a_plain_quantile() -> None:
    """The correction is not cosmetic: it selects a wider order statistic.

    11 values, alpha = 0.1: k = ceil(12 * 0.9) = 11, the maximum.  A plain 0.9
    quantile of the same data interpolates to 9.0, i.e. a band two order
    statistics narrower.
    """
    residuals = np.arange(0.0, 11.0)  # 0..10
    assert conformal_quantile(residuals, 0.1) == 10.0
    assert float(np.quantile(residuals, 0.9)) == pytest.approx(9.0)


def test_conformal_quantile_survives_binary_rounding_of_the_rank() -> None:
    """Regression test for a ceiling pushed up by floating-point noise.

    ``(n + 1) * (1 - 0.7)`` is exactly 9 in real arithmetic, but IEEE-754
    evaluates ``30 * 0.3`` as ``9.000000000000002``.  A naive ``ceil`` then
    selects the tenth order statistic instead of the ninth - a silently wider
    interval - and for small alpha it can select an index past the end of the
    sample.  The rank must be snapped to the mathematical value.
    """
    residuals = np.arange(1.0, 30.0)  # 1..29
    assert math.ceil(30 * (1.0 - 0.7)) == 10  # what a naive implementation gets
    assert conformal_quantile(residuals, 0.7) == 9.0  # what the maths says
    # The other rounding trap: alpha = 0.95 makes the rank 1, not 2.
    assert conformal_quantile(np.arange(1.0, 20.0), 0.95) == 1.0


def test_conformal_quantile_is_order_independent() -> None:
    values = np.array([4.0, 1.0, 9.0, 2.0, 7.0, 3.0, 8.0, 5.0, 6.0])
    assert conformal_quantile(values, 0.1) == conformal_quantile(
        values[::-1].copy(), 0.1
    )


def test_conformal_quantile_rejects_unusable_input() -> None:
    residuals = np.arange(1.0, 10.0)
    for alpha in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            conformal_quantile(residuals, alpha)
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, float("nan"), 2.0], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, float("inf")], 0.1)


def test_conformal_quantile_returns_an_observed_order_statistic() -> None:
    """Interpolating between residuals would break the finite-sample argument."""
    rng = np.random.default_rng(3)
    values = np.abs(rng.normal(0.0, 1.0, 200))
    quantile = conformal_quantile(values, 0.1)
    assert np.isin(quantile, values)


# --------------------------------------------------------------------------- fitting and coverage

def test_fit_split_conformal_records_its_provenance() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    band = fit_split_conformal(truth, np.zeros(10), alpha=0.2)

    assert isinstance(band, SplitConformal)
    assert band.alpha == 0.2
    assert band.n_calibration == 10
    # ceil(11 * 0.8) = 9 -> the 9th smallest of 1..10.
    assert band.quantile == 9.0
    assert band.width() == 18.0


def test_fit_split_conformal_uses_signed_residuals() -> None:
    """The band is driven by |error|, so the sign of an error is irrelevant."""
    truth = np.array([10.0, -10.0, 2.0, -2.0, 5.0, -5.0, 1.0, -1.0, 8.0, -8.0])
    band = fit_split_conformal(truth, np.zeros(10), alpha=0.1)
    assert band.quantile == 10.0  # max |residual|


def test_fit_split_conformal_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_split_conformal(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        fit_split_conformal(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_split_conformal(np.array([1.0, np.nan]), np.zeros(2))


def test_coverage_is_near_nominal_on_iid_errors() -> None:
    """Honest coverage: the point of the whole exercise is to be able to check it.

    The regression is a constant zero, so the residual *is* the error and the
    question reduces to "does the fitted order statistic bracket 90% of fresh
    draws from the same law".
    """
    rng = np.random.default_rng(11)
    calibration = rng.normal(0.0, 1.0, 2_000)
    held_out = rng.normal(0.0, 1.0, 5_000)
    forecast = np.zeros(held_out.size)

    band = fit_split_conformal(calibration, np.zeros(calibration.size), alpha=0.1)
    assert band.quantile == pytest.approx(1.645, abs=0.06)

    measured = band.coverage(held_out, forecast)
    assert 0.87 <= measured <= 0.93
    assert measured >= 0.90 - 0.03


def test_band_width_tracks_the_residual_scale() -> None:
    """The band is driven by the residuals, so five times the error is five times the band.

    A band hardcoded to some assumed sigma would hold the first two assertions
    and fail the third: at a 5x error scale it would collapse to 26% coverage.
    Nothing here is tuned to ``alpha`` - only the residual sample is.
    """
    rng = np.random.default_rng(5)
    loud_calibration = rng.normal(0.0, 0.05, 2_000)
    quiet_calibration = rng.normal(0.0, 0.01, 2_000)
    loud_held_out = rng.normal(0.0, 0.05, 2_000)
    quiet_held_out = rng.normal(0.0, 0.01, 2_000)

    loud_band = fit_split_conformal(loud_calibration, np.zeros(2_000), alpha=0.1)
    quiet_band = fit_split_conformal(quiet_calibration, np.zeros(2_000), alpha=0.1)
    zeros = np.zeros(2_000)

    assert loud_band.quantile == pytest.approx(5.0 * quiet_band.quantile, rel=0.10)
    assert loud_band.width() == pytest.approx(5.0 * quiet_band.width(), rel=0.10)

    # The same nominal alpha is met at either scale...
    assert 0.87 <= loud_band.coverage(loud_held_out, zeros) <= 0.93
    assert 0.87 <= quiet_band.coverage(quiet_held_out, zeros) <= 0.93

    # ...and a band carried across a 5x change in error scale under-covers hard.
    # This is the honest reading of a stale band: coverage *below* nominal means
    # the interval is too narrow, not that the model improved.
    assert quiet_band.coverage(loud_held_out, zeros) < 0.40


def test_coverage_exceeds_nominal_when_the_band_is_wider_than_the_errors() -> None:
    """The literal higher-than-nominal case, from two routes to a stale-wide band.

    Coverage above nominal is not a good result: it says the band is wider than
    the current errors require, so a report quoting it as "90% coverage achieved"
    is quoting a number that has drifted for the wrong reason.  It is included
    here precisely so that the direction of the effect is pinned down, in
    contrast to the under-coverage case above.
    """
    rng = np.random.default_rng(5)
    volatile_calibration = rng.normal(0.0, 0.05, 2_000)
    calm_held_out = rng.normal(0.0, 0.01, 2_000)
    # Route 1: a band fitted during a volatile stretch, applied to a calm one.
    volatile_band = fit_split_conformal(volatile_calibration, np.zeros(2_000), alpha=0.1)
    assert volatile_band.coverage(calm_held_out, np.zeros(2_000)) > 0.99

    # Route 2: a *shifted* calibration block - a biased model, which inflates
    # |residual| and therefore the band even though the scale is unchanged.
    biased_calibration = rng.normal(0.30, 0.05, 2_000)
    centred_held_out = rng.normal(0.0, 0.05, 2_000)
    biased_band = fit_split_conformal(biased_calibration, np.zeros(2_000), alpha=0.1)
    assert biased_band.quantile > 0.25
    assert biased_band.coverage(centred_held_out, np.zeros(2_000)) > 0.95


def test_coverage_is_nan_on_an_empty_sample() -> None:
    band = fit_split_conformal(np.arange(1.0, 10.0), np.zeros(9), alpha=0.1)
    assert math.isnan(band.coverage(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        band.coverage(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------- interval shape

def test_interval_is_centred_on_the_point_forecast() -> None:
    band = fit_split_conformal(np.arange(1.0, 10.0), np.zeros(9), alpha=0.1)
    forecast = np.array([-0.05, 0.0, 0.02, 0.31])
    lower, upper = band.interval(forecast)

    assert np.allclose(lower + band.quantile, forecast)
    assert np.allclose(upper - band.quantile, forecast)
    assert np.allclose(upper - lower, band.width())
    assert np.allclose((lower + upper) / 2.0, forecast)

    point_lower, point_upper = band.interval_for_point(0.17)
    assert point_lower == pytest.approx(0.17 - band.quantile)
    assert point_upper == pytest.approx(0.17 + band.quantile)
    assert point_upper - point_lower == pytest.approx(band.width())


def test_interval_is_frozen() -> None:
    """The quantile must not be editable in place; bands get reused across horizons."""
    band = fit_split_conformal(np.arange(1.0, 10.0), np.zeros(9), alpha=0.1)
    with pytest.raises(Exception):
        band.quantile = 999.0  # type: ignore[misc]


# --------------------------------------------------------------------------- method A

def test_empirical_probability_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(2)
    residuals = rng.normal(0.0, 1.0, 500)
    forecast = np.linspace(-3.0, 3.0, 61)

    probability = empirical_probability_positive(forecast, residuals)
    assert probability.shape == forecast.shape
    assert np.all(np.diff(probability) >= 0.0)
    assert np.all(probability >= 0.0) and np.all(probability <= 1.0)


def test_symmetric_residuals_give_one_half_at_zero() -> None:
    residuals = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    probability = empirical_probability_positive([0.0], residuals)
    # (#{r > 0} + 0.5) / (n + 1) = (3 + 0.5) / 7 = 0.5.
    assert probability[0] == pytest.approx(0.5, abs=1e-12)

    # A slightly asymmetric sample stays close to 0.5 at zero, but is not
    # identical to it - the deviation is the sample's own skew.
    skewed = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    assert empirical_probability_positive([0.0], skewed)[0] == pytest.approx(0.5, abs=0.2)


def test_symmetric_residuals_give_complementary_probabilities() -> None:
    """p(-yhat) + p(yhat) = 1 is the signature of treating R as symmetric.

    Exact, not approximate, on an exactly symmetric sample: the two tails are
    mirror images, so the two tail counts sum to n and the two continuity
    corrections add up to 1.  The one exception is a ``yhat`` that lands exactly
    on an observed residual, where the strict ``>`` costs one count.
    """
    residuals = np.concatenate([-np.arange(1.0, 51.0), np.arange(1.0, 51.0)])
    forecast = np.array([-3.7, -1.5, 0.0, 0.25])
    assert np.allclose(
        empirical_probability_positive(forecast, residuals)
        + empirical_probability_positive(-forecast, residuals),
        1.0,
        atol=1e-12,
    )

    # 1.0 *is* an observed residual, so one count is lost from the total.
    tied = np.array([1.0])
    assert empirical_probability_positive(tied, residuals)[0] + empirical_probability_positive(
        -tied, residuals
    )[0] == pytest.approx(1.0 - 1.0 / 101.0, abs=1e-12)

    # A real (asymmetric) sample is still close; the deviation is its own skew.
    rng = np.random.default_rng(21)
    real = rng.normal(0.0, 1.0, 4_001)
    points = np.array([-2.0, -0.5, 0.25, 2.0])
    assert np.allclose(
        empirical_probability_positive(points, real)
        + empirical_probability_positive(-points, real),
        1.0,
        atol=0.05,
    )


def test_probability_is_not_a_rescaled_point_forecast() -> None:
    """The trap this module exists to prevent.

    A point regression output of 0.004 invites being reported as "70% chance of
    going up".  With a typical error scale the honest probability at that
    forecast is barely above a coin flip, and the estimator is saturating near
    the edges long before the forecast does.
    """
    residuals = np.linspace(-1.0, 1.0, 4_001)
    probability = empirical_probability_positive([0.004], residuals)
    assert probability[0] == pytest.approx(0.5, abs=0.05)

    # Nothing ever reaches 0 or 1: the continuity correction forbids it, because
    # the largest observed residual is not proof the outcome is impossible.
    extreme = empirical_probability_positive([1e3, -1e3], residuals)
    assert np.all(extreme > 0.0) and np.all(extreme < 1.0)
    assert extreme[0] > 0.999 and extreme[1] < 0.001


def test_empirical_probability_matches_a_hand_count() -> None:
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 4.0])  # n = 5
    # yhat = -0.5 -> #{r > 0.5} = 2 -> (2 + 0.5) / 6.
    assert empirical_probability_positive([-0.5], residuals)[0] == pytest.approx(2.5 / 6.0)
    # yhat = -4.5 -> #{r > 4.5} = 0 -> 0.5 / 6.
    assert empirical_probability_positive([-4.5], residuals)[0] == pytest.approx(0.5 / 6.0)
    # yhat = 10 -> #{r > -10} = 5 -> 5.5 / 6.
    assert empirical_probability_positive([10.0], residuals)[0] == pytest.approx(5.5 / 6.0)


def test_empirical_probability_requires_residuals() -> None:
    with pytest.raises(ValueError):
        empirical_probability_positive([0.1, 0.2], [])
    with pytest.raises(ValueError):
        empirical_probability_positive([0.1], [float("nan")])


# --------------------------------------------------------------------------- method B

@needs_metrics
def test_direction_model_beats_a_flat_probability() -> None:
    features, labels = direction_problem()
    train, test = slice(0, 1_000), slice(1_000, None)

    model = CalibratedDirectionModel(base_estimator=small_classifier(), method="sigmoid")
    model.fit(features[train], labels[train])
    probability = model.predict_proba_positive(features[test])

    assert probability.shape == (features[test].shape[0],)
    assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
    assert brier_score(probability, labels[test]) < brier_score(
        np.full(probability.shape, 0.5), labels[test]
    )


@needs_metrics
def test_direction_model_is_a_probability_not_a_thresholded_forecast() -> None:
    """What a thresholded regressor cannot produce: a graded distribution of scores.

    A threshold at zero of a point forecast emits exactly two values.  Both
    calibration methods emit hundreds, and sigmoid - mapping a bounded score -
    stays clear of both endpoints.  Isotonic interpolates between the observed
    out-of-fold scores and does reach exactly 0.0 and 1.0 on a sample this size,
    which is exactly why it is the wrong choice when data is thin.
    """
    features, labels = direction_problem()
    train, test = slice(0, 1_000), slice(1_000, None)
    rows = features[test].shape[0]

    for method in ("sigmoid", "isotonic"):
        model = CalibratedDirectionModel(
            base_estimator=small_classifier(), method=method
        ).fit(features[train], labels[train])
        probability = model.predict_proba_positive(features[test])

        assert len(np.unique(probability)) > 20
        assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
        if method == "sigmoid":
            assert np.all(probability > 0.0) and np.all(probability < 1.0)

        hard = model.predict(features[test])
        assert np.array_equal(hard, (probability >= 0.5).astype(int))
        assert set(np.unique(hard)) <= {0, 1}
        assert hard.shape == (rows,)


@needs_metrics
def test_both_routes_are_available_and_need_not_agree() -> None:
    """Two methods, two assumptions, same target - so they can be scored side by side."""
    features, labels = direction_problem()
    train, test = slice(0, 1_000), slice(1_000, None)

    # Method B: a learned, calibrated classifier on the direction label.
    model = CalibratedDirectionModel(base_estimator=small_classifier()).fit(
        features[train], labels[train]
    )
    learned = model.predict_proba_positive(features[test])

    # Method A: a distribution, with the regression supplied as a constant
    # forecast so the residual sample is exactly the direction label's error.
    forecast = np.zeros(features[test].shape[0])
    residuals = labels[test] - forecast
    empirical = empirical_probability_positive(forecast, residuals)

    for probability in (learned, empirical):
        assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
        assert brier_score(probability, labels[test]) >= 0.0
    assert not np.allclose(learned, empirical)


def test_direction_model_rejects_unusable_labels() -> None:
    features, labels = direction_problem(n=400)
    with pytest.raises(ValueError):
        CalibratedDirectionModel().fit(features, np.zeros(400))
    with pytest.raises(ValueError):
        CalibratedDirectionModel().fit(features, np.ones(400))
    with pytest.raises(ValueError):
        CalibratedDirectionModel().fit(features, np.full(400, 2))
    with pytest.raises(ValueError):
        CalibratedDirectionModel().fit(features, labels[:100])


def test_direction_model_rejects_an_unknown_calibration_method() -> None:
    features, labels = direction_problem(n=200)
    model = CalibratedDirectionModel(method="nonsense")
    # The constructor stays permissive so `clone` and grid search keep working;
    # the failure is reported where it can be understood.
    assert model.method == "nonsense"
    with pytest.raises(ValueError):
        model.fit(features, labels)


def test_direction_model_handles_non_finite_features() -> None:
    features, labels = direction_problem(n=600)
    dirty = features.copy()
    dirty[3, 0] = np.nan
    dirty[10, 2] = np.inf
    truth = labels.copy()
    truth[dirty[:, 0] > 1.0] = 1 - truth[dirty[:, 0] > 1.0]
    truth[np.isinf(dirty[:, 2])] = 1

    model = CalibratedDirectionModel(
        base_estimator=RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    )
    model.fit(dirty, truth)
    probability = model.predict_proba_positive(dirty)

    assert model.n_imputed_ == 2
    assert np.all(np.isfinite(probability))
    assert np.all(probability >= 0.0) and np.all(probability <= 1.0)


def test_direction_model_is_a_well_behaved_sklearn_estimator() -> None:
    model = CalibratedDirectionModel(method="isotonic", n_bins=5, random_state=1)
    assert model.n_bins == 5 and model.random_state == 1
    assert is_classifier(model)
    assert clone(model).get_params()["method"] == "isotonic"

    features, labels = direction_problem(n=400)
    fitted = model.fit(features, labels)
    assert list(fitted.classes_) == [0, 1]
    assert 2 <= fitted.n_folds_ <= 5
    # The in-sample score is the mixin's default, not something bespoke.
    assert 0.0 <= fitted.score(features, labels) <= 1.0


def test_direction_model_reports_calibration_diagnostics() -> None:
    if calibration_table is None:  # pragma: no cover - sibling module absent
        pytest.skip("src.v3.metrics is not importable")
    features, labels = direction_problem()
    train, test = slice(0, 1_000), slice(1_000, None)
    model = CalibratedDirectionModel(
        base_estimator=small_classifier(), n_bins=4
    ).fit(features[train], labels[train])

    report = model.calibration_report(features[test], labels[test])
    assert set(report) == {
        "brier_score",
        "base_rate_up",
        "mean_predicted_probability",
        "n",
        "calibration_table",
    }
    assert report["n"] == int(features[test].shape[0])
    assert report["brier_score"] == pytest.approx(
        brier_score(model.predict_proba_positive(features[test]), labels[test])
    )
    table = report["calibration_table"]
    for column in ("bin_lower", "bin_upper", "n", "mean_predicted", "observed_frequency"):
        assert column in table.columns


# --------------------------------------------------------------------------- reporting

def test_interval_summary_reports_measured_and_nominal_coverage() -> None:
    rng = np.random.default_rng(17)
    calibration = rng.normal(0.0, 1.0, 1_500)
    held_out = rng.normal(0.0, 1.0, 400)
    forecast = rng.normal(0.0, 0.3, 400)

    band = fit_split_conformal(calibration, np.zeros(1_500), alpha=0.05)
    summary = interval_summary(held_out, forecast, band)

    assert set(summary) == {
        "empirical_coverage",
        "nominal_coverage",
        "mean_interval_width",
        "n",
    }
    assert summary["nominal_coverage"] == 0.95
    assert summary["n"] == 400
    assert summary["mean_interval_width"] == 2.0 * band.quantile
    assert summary["empirical_coverage"] == band.coverage(held_out, forecast)
    assert 0.90 <= summary["empirical_coverage"] <= 1.0


def test_interval_summary_rejects_mismatched_inputs() -> None:
    band = fit_split_conformal(np.arange(1.0, 10.0), np.zeros(9), alpha=0.1)
    with pytest.raises(ValueError):
        interval_summary(np.zeros(3), np.zeros(4), band)


# --------------------------------------------------------------------------- Wilson

def test_wilson_interval_against_a_hand_computed_value() -> None:
    """50/100 at 95%: p = 0.5, z^2 = 3.8416, margin = 0.09986, denom = 1.038416."""
    lower, upper = wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038, abs=5e-5)
    assert upper == pytest.approx(0.5962, abs=5e-5)
    assert lower < 0.5 < upper

    # Monotone in the success count, and symmetric about p = 0.5 at p = 0.5.
    assert wilson_interval(60, 100)[0] > lower
    assert wilson_interval(40, 100)[1] < upper


def test_wilson_interval_stays_inside_the_unit_interval() -> None:
    """The normal approximation returns a negative lower bound at k = 0."""
    lower, upper = wilson_interval(0, 100)
    assert lower == 0.0
    assert upper == pytest.approx(0.0370, abs=1e-4)

    lower, upper = wilson_interval(100, 100)
    assert upper <= 1.0 and lower == pytest.approx(0.9630, abs=1e-4)
    for successes in (0, 1, 7, 50, 99, 100):
        low, high = wilson_interval(successes, 100)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_narrows_with_sample_size() -> None:
    wide = wilson_interval(5, 10)
    narrow = wilson_interval(500, 1_000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0]) > 0.0


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 10, z=0.0)
