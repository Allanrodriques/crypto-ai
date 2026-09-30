"""Tests for V3 forward-return regression metrics.

Everything here is synthetic, in-memory and deterministic: no IO, no fixtures, no
network.  The cases that matter are the awkward ones - constant predictions,
NaN pairs, tied predictions, a sample too small to fill a bucket - because those
are exactly where a metrics module produces a confident wrong number instead of
a NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v3.metrics import (
    MetricsError,
    brier_score,
    calibration_table,
    decile_analysis,
    direction_metrics,
    distribution_stats,
    evaluate_regression,
    information_coefficient,
    regression_metrics,
    spread_analysis,
)

BUNDLE_KEYS = {
    "mae", "rmse", "mse", "r2", "n", "mape", "mape_coverage",
    "direction_accuracy", "balanced_direction_accuracy", "n_up", "n_down",
    "spearman_ic", "long_short_spread",
}


# --------------------------------------------------------------------------- perfect fit

def test_perfect_prediction_scores_perfectly() -> None:
    y = np.array([0.10, -0.05, 0.02, 0.07, -0.11, 0.03])
    metrics = regression_metrics(y, y.copy())

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["n"] == 6


def test_perfect_prediction_direction_and_ic() -> None:
    y = np.array([0.10, -0.05, 0.02, 0.07, -0.11, 0.03])
    direction = direction_metrics(y, y.copy())
    assert direction["direction_accuracy"] == 1.0
    assert direction["balanced_direction_accuracy"] == 1.0
    assert direction["n_up"] == 4
    assert direction["n_down"] == 2
    assert direction["n"] == 6
    assert information_coefficient(y, y.copy()) == 1.0


# --------------------------------------------------------------------------- constant predictions

def test_constant_prediction_ic_is_nan_not_an_exception() -> None:
    y = np.array([-0.10, -0.05, 0.02, 0.07, 0.11, 0.03])
    ic = information_coefficient(y, np.zeros(6))
    assert isinstance(ic, float)
    assert np.isnan(ic)


def test_constant_prediction_r2_is_finite_and_negative() -> None:
    y = np.array([-0.10, -0.05, 0.02, 0.07, 0.11, 0.03])
    for constant in (0.0, 0.04, -0.4):
        metrics = regression_metrics(y, np.full(6, constant))
        assert np.isfinite(metrics["r2"]), f"r2 should be finite for prediction {constant}"
        assert metrics["r2"] < 0.0
        assert metrics["n"] == 6


def test_zero_prediction_is_always_a_miss() -> None:
    """A prediction of exactly 0.0 is 'no view' and must never score as correct."""
    y = np.array([0.10, -0.05, 0.02, 0.07, -0.11])
    direction = direction_metrics(y, np.zeros(5))
    assert direction["direction_accuracy"] == 0.0
    assert direction["balanced_direction_accuracy"] == 0.0
    assert direction["n_up"] == 3
    assert direction["n_down"] == 2


def test_constant_nonzero_prediction_only_hits_its_own_side() -> None:
    y = np.array([0.10, -0.05, 0.02, 0.07, -0.11])
    up = direction_metrics(y, np.full(5, 0.5))
    assert up["direction_accuracy"] == pytest.approx(3 / 5)
    # recall on the up class is 1.0, recall on the down class is 0.0
    assert up["balanced_direction_accuracy"] == pytest.approx(0.5)

    down = direction_metrics(y, np.full(5, -0.5))
    assert down["direction_accuracy"] == pytest.approx(2 / 5)
    # up recall 0.0, down recall 1.0
    assert down["balanced_direction_accuracy"] == pytest.approx(0.5)


def test_balanced_direction_is_nan_when_one_class_is_absent() -> None:
    y = np.array([0.10, 0.20, 0.30])
    direction = direction_metrics(y, y.copy())
    assert direction["direction_accuracy"] == 1.0
    assert np.isnan(direction["balanced_direction_accuracy"])


def test_exact_zero_target_is_counted_in_neither_direction_class() -> None:
    y = np.array([0.10, -0.05, 0.0])
    direction = direction_metrics(y, y.copy())
    assert direction["n_up"] == 1
    assert direction["n_down"] == 1
    assert direction["n"] == 3
    # The flat row is a real observation the model failed to call.
    assert direction["direction_accuracy"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- shape / NaN contracts

def test_length_mismatch_raises_value_error() -> None:
    y = np.array([0.1, 0.2, 0.3])
    p = np.array([0.1, 0.2])
    for call in (
        regression_metrics,
        direction_metrics,
        information_coefficient,
        decile_analysis,
        spread_analysis,
        evaluate_regression,
    ):
        with pytest.raises(ValueError):
            call(y, p)
    with pytest.raises(ValueError):
        brier_score(np.array([0.2, 0.3]), np.array([0, 1, 0]))


def test_nan_rows_are_filtered_and_the_count_is_visible() -> None:
    y = np.array([0.10, np.nan, -0.20, 0.30, np.inf])
    p = np.array([0.10, 0.50, np.nan, 0.25, 0.0])
    metrics = regression_metrics(y, p)
    # Two rows survive: (0.10, 0.10) and (0.30, 0.25).
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(0.025)
    assert regression_metrics(y, p)["n"] == 2
    assert direction_metrics(y, p)["n"] == 2


def test_nan_filtering_is_never_silently_wrong() -> None:
    """The surviving rows, not the input length, drive every number."""
    y = np.array([1.0, np.nan, 3.0, 5.0])
    p = np.array([1.0, 2.0, 3.0, 5.0])
    clean = regression_metrics(np.array([1.0, 3.0, 5.0]), np.array([1.0, 3.0, 5.0]))
    assert regression_metrics(y, p) == clean


def test_empty_after_filter_returns_nan_and_zero() -> None:
    y = np.array([np.nan, np.nan])
    p = np.array([0.1, 0.2])

    metrics = regression_metrics(y, p)
    assert metrics["n"] == 0
    for key in ("mae", "rmse", "mse", "r2", "mape"):
        assert np.isnan(metrics[key]), key

    direction = direction_metrics(y, p)
    assert direction["n"] == 0
    assert np.isnan(direction["direction_accuracy"])
    assert np.isnan(direction["balanced_direction_accuracy"])

    assert np.isnan(information_coefficient(y, p))

    spread = spread_analysis(y, p)
    assert spread["n_long"] == 0 and spread["n_short"] == 0
    assert np.isnan(spread["spread"])

    assert decile_analysis(y, p).empty
    assert np.isnan(distribution_stats(y)["mean"])
    assert np.isnan(brier_score(np.array([np.nan]), np.array([1.0])))
    assert int(calibration_table(np.array([np.nan]), np.array([1.0]), n_bins=3)["n"].sum()) == 0


def test_ic_needs_three_rows() -> None:
    assert np.isnan(information_coefficient([0.1, 0.2], [0.15, 0.25]))
    assert information_coefficient([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 1.0


# --------------------------------------------------------------------------- deciles

def test_decile_analysis_is_monotone_for_an_informative_prediction() -> None:
    rng = np.random.default_rng(7)
    n = 500
    pred = rng.normal(size=n)
    actual = 0.6 * pred + 0.3 * rng.normal(size=n)

    table = decile_analysis(actual, pred, n_deciles=5)
    assert list(table["decile"]) == [0, 1, 2, 3, 4]
    assert (table["n"] == 100).all()
    assert int(table["n"].sum()) == n
    assert (table["n_deciles"] == 5).all()

    means = table["mean_actual_return"].to_numpy()
    assert np.all(np.diff(means) > 0), means
    assert means[-1] > means[0]


def test_decile_analysis_handles_ties_without_empty_bins() -> None:
    """A tied prediction used to be the classic way to lose a bucket."""
    pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    actual = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.05, 0.06, 0.04, 0.07])

    table = decile_analysis(actual, pred, n_deciles=5)
    assert len(table) == 5
    assert (table["n"] > 0).all()
    assert int(table["n"].sum()) == 10


def test_decile_analysis_reduces_the_count_when_rows_are_scarce() -> None:
    pred = np.array([0.1, 0.2, 0.3])
    actual = np.array([0.11, 0.19, 0.31])
    table = decile_analysis(actual, pred, n_deciles=5)
    assert len(table) == 3
    assert (table["n_deciles"] == 3).all()
    assert int(table["n"].sum()) == 3


def test_decile_analysis_single_row_is_one_bucket_with_undefined_std() -> None:
    table = decile_analysis([0.05], [0.04], n_deciles=5)
    assert len(table) == 1
    assert int(table["n"].iloc[0]) == 1
    assert int(table["n_deciles"].iloc[0]) == 1
    assert np.isnan(table["std_actual_return"].iloc[0])
    assert table["mean_actual_return"].iloc[0] == pytest.approx(0.05)


def test_decile_analysis_columns() -> None:
    table = decile_analysis(np.arange(20.0), np.arange(20.0), n_deciles=4)
    assert list(table.columns) == [
        "decile", "n", "n_deciles", "mean_actual_return", "median_actual_return", "std_actual_return"
    ]
    with pytest.raises(ValueError):
        decile_analysis(np.arange(5.0), np.arange(5.0), n_deciles=0)


# --------------------------------------------------------------------------- spread

def test_spread_analysis_matches_hand_computation() -> None:
    # 10 rows, tail_fraction 0.2 -> 2 rows per leg.
    pred = np.arange(10.0)
    actual = pred + 10.0  # long leg (rows 8, 9) realises 18.5; short (0, 1) realises 10.5

    spread = spread_analysis(actual, pred, tail_fraction=0.2)
    assert spread["n_long"] == 2
    assert spread["n_short"] == 2
    assert spread["long_mean_return"] == pytest.approx(18.5)
    assert spread["short_mean_return"] == pytest.approx(10.5)
    assert spread["spread"] == pytest.approx(8.0)


def test_spread_analysis_zero_spread_for_a_constant_prediction() -> None:
    actual = np.array([0.01, -0.02, 0.03, 0.04, -0.05])
    spread = spread_analysis(actual, np.zeros(5), tail_fraction=0.4)
    assert spread["n_long"] == 2
    assert spread["n_short"] == 2
    assert spread["spread"] == pytest.approx(0.0)


def test_spread_analysis_refuses_to_build_a_tail_from_too_few_rows() -> None:
    spread = spread_analysis(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), tail_fraction=0.2)
    assert spread["n_long"] == 0
    assert spread["n_short"] == 0
    assert np.isnan(spread["spread"])
    assert np.isnan(spread["long_mean_return"])
    with pytest.raises(ValueError):
        spread_analysis(np.arange(5.0), np.arange(5.0), tail_fraction=0.0)


# --------------------------------------------------------------------------- calibration

def test_brier_score_matches_hand_computation() -> None:
    # (0.9-1)^2 + (0.2-0)^2 + (0.7-0)^2 + (0.1-1)^2 = 0.01 + 0.04 + 0.49 + 0.81 = 1.35
    prob = np.array([0.9, 0.2, 0.7, 0.1])
    labels = np.array([1, 0, 0, 1])
    assert brier_score(prob, labels) == pytest.approx(1.35 / 4)


def test_brier_score_rejects_a_regression_output_as_a_probability() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([0.05, -0.10, 0.30]), np.array([1, 0, 1]))
    with pytest.raises(ValueError):
        brier_score(np.array([0.5, 0.5]), np.array([1, 2]))


def test_calibration_table_shape_and_counts() -> None:
    prob = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    labels = np.array([1, 0, 1, 0, 1])
    table = calibration_table(prob, labels, n_bins=4)

    assert list(table.columns) == [
        "bin_lower", "bin_upper", "n", "mean_predicted", "observed_frequency"
    ]
    assert len(table) == 4
    assert int(table["n"].sum()) == len(prob)
    assert table["bin_lower"].iloc[0] == 0.0
    assert table["bin_upper"].iloc[-1] == 1.0
    # prob == 1.0 must land in the last bin, not fall off the end.
    assert int(table["n"].iloc[-1]) == 2


def test_calibration_table_keeps_empty_bins() -> None:
    prob = np.array([0.1, 0.2, 0.3, 0.9])
    labels = np.array([1, 0, 1, 0])
    table = calibration_table(prob, labels, n_bins=4).set_index("bin_lower")

    assert int(table.loc[0.0, "n"]) == 2
    assert table.loc[0.0, "mean_predicted"] == pytest.approx(0.15)
    assert table.loc[0.0, "observed_frequency"] == pytest.approx(0.5)
    assert int(table.loc[0.5, "n"]) == 0
    assert np.isnan(table.loc[0.5, "observed_frequency"])
    assert np.isnan(table.loc[0.5, "mean_predicted"])


def test_calibration_table_default_bins_and_validation() -> None:
    table = calibration_table(np.linspace(0.0, 1.0, 100), np.zeros(100))
    assert len(table) == 10
    assert int(table["n"].sum()) == 100
    with pytest.raises(ValueError):
        calibration_table(np.array([0.5]), np.array([1]), n_bins=0)


# --------------------------------------------------------------------------- distributions & bundle

def test_distribution_stats_on_a_known_series() -> None:
    stats = distribution_stats(np.arange(10.0))
    assert stats["n"] == 10
    assert stats["mean"] == pytest.approx(4.5)
    assert stats["std"] == pytest.approx(3.02765035)
    assert stats["median"] == pytest.approx(4.5)
    assert stats["min"] == 0.0
    assert stats["max"] == 9.0
    assert stats["p05"] == pytest.approx(0.45)
    assert stats["p25"] == pytest.approx(2.25)
    assert stats["p75"] == pytest.approx(6.75)
    assert stats["p95"] == pytest.approx(8.55)
    assert stats["positive_fraction"] == pytest.approx(0.9)


def test_distribution_stats_drops_non_finite_and_reports_n() -> None:
    stats = distribution_stats(np.array([0.0, np.nan, 2.0, np.inf, -2.0]))
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["min"] == -2.0
    assert stats["max"] == 2.0
    assert stats["positive_fraction"] == pytest.approx(1 / 3)


def test_distribution_stats_std_is_undefined_for_one_value() -> None:
    assert np.isnan(distribution_stats([0.03])["std"])
    assert distribution_stats([0.03])["n"] == 1


def test_evaluate_regression_bundle() -> None:
    rng = np.random.default_rng(11)
    pred = rng.normal(size=200)
    actual = 0.8 * pred + 0.2 * rng.normal(size=200)

    bundle = evaluate_regression(actual, pred)
    assert BUNDLE_KEYS.issubset(bundle)
    assert bundle["n"] == 200
    assert bundle["r2"] > 0.0
    assert bundle["direction_accuracy"] > 0.5
    assert bundle["spearman_ic"] > 0.0
    assert bundle["long_short_spread"] > 0.0
    assert bundle["spearman_ic"] <= 1.0

    # The bundle must agree with its parts, not re-implement them.
    assert bundle["spearman_ic"] == information_coefficient(actual, pred)
    assert bundle["long_short_spread"] == spread_analysis(actual, pred)["spread"]
    assert bundle["mae"] == regression_metrics(actual, pred)["mae"]


def test_evaluate_regression_on_degenerate_input() -> None:
    bundle = evaluate_regression([np.nan, np.nan], [0.1, 0.2])
    assert bundle["n"] == 0
    assert np.isnan(bundle["r2"])
    assert np.isnan(bundle["spearman_ic"])
    assert np.isnan(bundle["long_short_spread"])


def test_mape_guard_excludes_flat_targets() -> None:
    """MAPE is reported with its coverage; a near-zero target is excluded, not divided by."""
    actual = np.array([0.0, 2.0, 4.0])
    pred = np.array([0.0, 3.0, 4.0])
    metrics = regression_metrics(actual, pred)
    assert metrics["mape_coverage"] == pytest.approx(2 / 3)
    assert metrics["mape"] == pytest.approx((1 / 2 + 0.0) / 2)

    degenerate = regression_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(degenerate["mape"])
    assert degenerate["mape_coverage"] == 0.0


def test_pandas_series_inputs_are_accepted() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    actual = pd.Series([0.01, -0.02, 0.03, 0.00, 0.05, -0.01], index=index)
    pred = pd.Series([0.02, -0.01, 0.02, 0.01, 0.04, -0.02], index=index)
    assert regression_metrics(actual, pred)["n"] == 6
    assert len(decile_analysis(actual, pred, n_deciles=3)) == 3
