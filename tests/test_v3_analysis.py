"""Tests for the V3 reporting layer: regimes, distributions, intervals, calibration.

Everything is synthetic, in-memory and deterministic.  The frames are shaped
exactly like ``HorizonResult.predictions`` - a sorted, unique, UTC ``DatetimeIndex``,
a realised ``target``, and per model a point forecast, split-conformal bounds, an
empirical probability, plus the one shared ``prob_calibrated`` column - so the
tests exercise the same alignment and column-name paths the pipeline will.

The claims under test are structural wherever a hand computation is not available:
a bucket count that is balanced *by construction*, a Wilson interval that contains
its own point estimate, a coverage gap that is exactly zero at nominal, and a
degenerate probability flagged rather than scored.  Those are the properties that
fail silently, and a silently wrong report is worse than no report.
"""

from __future__ import annotations

import types
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from src.v3.analysis import (
    ANALYSIS_REPORTS,
    AnalysisError,
    CALIBRATION_COLUMNS,
    DECILE_COLUMNS,
    DISTRIBUTION_COLUMNS,
    INTERVAL_COLUMNS,
    REGIME_COLUMNS,
    RegimeDefinition,
    analyse,
    calibration_report,
    decile_report,
    evaluate_by_regime,
    label_trend_regimes,
    label_volatility_regimes,
    prediction_interval_report,
    regime_label,
    return_distribution_report,
    run_ablation,
)
from src.v3.uncertainty import wilson_interval

N = 1200
MODELS = ("ridge", "xgboost", "random_forest")
FEATURE_COLUMNS = ("sma_20", "rsi_14", "atr_14", "funding_rate", "fear_greed")
FEATURE_GROUPS = {
    "technical": ["sma_20", "rsi_14", "atr_14"],
    "derivatives": ["funding_rate"],
    "context": ["fear_greed"],
}


def make_predictions(
    n: int = N, seed: int = 11, models: Sequence[str] = MODELS
) -> pd.DataFrame:
    """A prediction frame in the exact schema ``src.v3.walkforward`` produces.

    The data-generating process is not decorative: the return is a real function
    of a latent signal, each model's prediction shrinks that signal by a different
    factor and adds its own noise, and the target's dispersion scales with the
    latent volatility.  That is what makes the regime and decile assertions
    meaningful - a model that only works in high volatility is expressible here,
    and a model with no ranking power at all is too.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    vol = np.abs(rng.gamma(shape=2.0, scale=0.01, size=n))
    signal = 0.6 * np.sign(rng.normal(size=n)) * vol
    target = signal + rng.normal(0.0, 1.0, n) * vol

    frame = pd.DataFrame(
        {
            "target": target,
            # One shared direction model for the whole fold, as the harness fits
            # it: a monotone but non-linear function of the same latent signal, so
            # it is a real probability and not a thresholded point forecast.
            "prob_calibrated": np.clip(
                0.5 + 30.0 * signal / (40.0 * vol), 0.01, 0.99
            ),
        },
        index=index,
    )
    for model, shrink in zip(models, (0.6, 0.35, 0.1)):
        pred = shrink * target + (1.0 - shrink) * rng.normal(0.0, 0.02, n)
        half_width = 1.28 * vol
        frame[f"{model}_pred"] = pred
        frame[f"{model}_lo"] = pred - half_width
        frame[f"{model}_hi"] = pred + half_width
        frame[f"{model}_prob_empirical"] = np.clip(0.5 + 2.0 * (pred - target.mean()), 0.0, 1.0)

    # Trailing conditioning features, as a dataset frame would carry them.  They
    # are here so the regime tables have a source without a second object.
    frame["realised_vol"] = vol
    frame["trend"] = pd.Series(signal).rolling(48, min_periods=1).mean().to_numpy()
    return frame


def make_result(frame: pd.DataFrame | None = None, **overrides: Any) -> Any:
    """A ``HorizonResult``-like stub.

    A :class:`types.SimpleNamespace` rather than the real dataclass: importing
    ``src.v3.walkforward`` drags in the model factory, and these tests are about
    what the reporting layer does with a result, not about how one is produced.
    The attribute set mirrors the dataclass field-for-field.
    """
    frame = make_predictions() if frame is None else frame
    payload: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "horizon": "30d",
        "horizon_days": 30.0,
        "feature_columns": list(FEATURE_COLUMNS),
        "predictions": frame,
        "fold_metrics": pd.DataFrame(),
        "pooled_metrics": pd.DataFrame(
            {"rmse": [0.02, 0.05, 0.09], "mae": [0.015, 0.04, 0.07]},
            index=["ridge", "xgboost", "random_forest"],
        ),
        "interval_summary": pd.DataFrame(
            {
                "model": list(MODELS),
                "mean_fold_coverage": [0.9, 0.8, 0.75],
                "nominal_coverage": [0.9, 0.9, 0.9],
                "mean_interval_width": [0.05, 0.06, 0.07],
            }
        ),
        "geometry": {"purge": 720, "embargo": "horizon"},
        "coverage": {"n_folds": 4, "prediction_rows": int(len(frame))},
        "seconds": 12.5,
        "models": list(MODELS),
    }
    payload.update(overrides)
    return types.SimpleNamespace(**payload)


# ------------------------------------------------------------- regime labelling

def test_volatility_regimes_split_the_sample_into_equal_buckets() -> None:
    """Rank cuts balance the buckets; a fixed threshold would not."""
    values = pd.Series(np.abs(np.random.default_rng(3).gamma(2.0, 1.0, N)))

    labels = label_volatility_regimes(values)

    counts = labels.value_counts()
    assert set(counts.index) == {"vol_q1", "vol_q2", "vol_q3", "vol_q4"}
    assert counts.tolist() == [N // 4] * 4
    assert labels.notna().all()
    assert labels.name == "realised_vol"


def test_volatility_regimes_are_ordered_calmest_first() -> None:
    values = pd.Series(np.arange(100.0))

    labels = label_volatility_regimes(values)

    assert labels.iloc[0] == "vol_q1"
    assert labels.iloc[-1] == "vol_q4"
    # Monotone in the value, which is what makes 'q1' readable as 'calmest'.
    by_value = pd.DataFrame({"v": values, "label": labels}).sort_values("v")
    assert by_value["label"].is_monotonic_increasing


def test_trend_regimes_use_their_own_label_prefix() -> None:
    values = pd.Series(np.random.default_rng(5).normal(size=N))

    labels = label_trend_regimes(values, column="trend")

    assert set(labels.value_counts().index) == {"trend_q1", "trend_q2", "trend_q3", "trend_q4"}
    assert labels.name == "trend"
    assert labels.value_counts().tolist() == [N // 4] * 4


def test_a_missing_regime_value_becomes_a_missing_label() -> None:
    values = pd.Series(np.arange(200.0))
    values.iloc[[0, 7, 199]] = np.nan

    labels = label_volatility_regimes(values)

    assert labels.isna().sum() == 3
    assert labels.iloc[[0, 7, 199]].isna().all()
    # The missing rows are not silently sorted into the lowest bucket: the other
    # 197 rows still split into four near-equal buckets.
    assert max(labels.value_counts().tolist()) - min(labels.value_counts().tolist()) <= 1


def test_reordering_the_input_does_not_move_a_value_between_buckets() -> None:
    """The label is a function of the value, not of the row it happens to sit in."""
    values = pd.Series(np.random.default_rng(7).gamma(3.0, 1.0, 300))
    shuffled = values.sample(frac=1.0, random_state=17)

    straight = label_volatility_regimes(values)
    reordered = label_volatility_regimes(shuffled)

    moved = dict(zip(shuffled.to_numpy(), reordered.to_numpy()))
    assert all(
        moved[value] == straight.iloc[position]
        for position, value in enumerate(values.to_numpy())
    )


def test_a_heavily_tied_series_stays_in_one_bucket_per_tied_block() -> None:
    """Ties share an average rank, so a tied block is never split by a cut."""
    values = pd.Series([1.0] * 50 + [2.0] * 50 + [3.0] * 50 + [4.0] * 50)

    labels = label_volatility_regimes(values)

    assert labels.iloc[:50].nunique() == 1
    assert labels.iloc[50:100].nunique() == 1
    assert labels.nunique() == 4


def test_a_constant_column_still_produces_labelled_rows() -> None:
    """All-ties is degenerate but must not raise: the reader needs the label."""
    labels = label_volatility_regimes(pd.Series(np.full(50, 0.02)))

    assert labels.notna().all()
    assert labels.nunique() == 1


def test_regime_label_dispatches_on_the_column_name() -> None:
    values = pd.Series(np.random.default_rng(9).normal(size=200))
    positive = pd.Series(np.abs(values))

    assert label_volatility_regimes(values, column="atr_14").nunique() == 4
    assert regime_label(positive, column="realised_vol").nunique() == 4
    assert regime_label(values, column="trend_score").nunique() == 4


def test_regime_label_falls_back_to_the_sign_structure_of_an_unnamed_column() -> None:
    rng = np.random.default_rng(13)

    dispersion = regime_label(pd.Series(np.abs(rng.normal(size=200))), column="f_0042")
    signed = regime_label(pd.Series(rng.normal(size=200)), column="f_0042")

    assert dispersion.str.startswith("vol_q").all()
    assert signed.str.startswith("trend_q").all()
    assert dispersion.nunique() == 4
    assert signed.nunique() == 4


def test_regime_label_refuses_a_column_that_is_both() -> None:
    with pytest.raises(AnalysisError, match="both"):
        regime_label(pd.Series(np.arange(10.0)), column="trend_vol_adjusted")


def test_a_regime_definition_validates_its_own_shape() -> None:
    with pytest.raises(AnalysisError, match="one more label"):
        RegimeDefinition(name="bad", column="c", bins=(0.5,), labels=("only_one",))


# ------------------------------------------------------------- per-regime metrics

def test_evaluate_by_regime_counts_every_scored_row_exactly_once() -> None:
    frame = make_predictions()
    labels = label_volatility_regimes(frame["realised_vol"])

    table = evaluate_by_regime(frame["target"], frame["ridge_pred"], labels)

    assert list(table.columns) == list(REGIME_COLUMNS)
    assert table["regime"].tolist() == ["vol_q1", "vol_q2", "vol_q3", "vol_q4"]
    assert table["n"].tolist() == [N // 4] * 4
    assert int(table["n"].sum()) == N


def test_the_wilson_interval_brackets_the_reported_hit_rate() -> None:
    frame = make_predictions()
    labels = label_volatility_regimes(frame["realised_vol"])

    table = evaluate_by_regime(frame["target"], frame["ridge_pred"], labels)

    for row in table.itertuples(index=False):
        assert 0.0 <= row.hit_rate_ci_lower <= row.hit_rate_ci_upper <= 1.0
        assert row.hit_rate_ci_lower <= row.direction_accuracy <= row.hit_rate_ci_upper


def test_a_regimes_wilson_interval_matches_a_hand_computed_call() -> None:
    frame = make_predictions()
    labels = label_volatility_regimes(frame["realised_vol"])
    table = evaluate_by_regime(frame["target"], frame["ridge_pred"], labels).set_index("regime")

    y = frame["target"].to_numpy()
    p = frame["ridge_pred"].to_numpy()
    mask = labels.to_numpy() == "vol_q3"
    hits = int((((p > 0) & (y > 0)) | ((p < 0) & (y < 0)))[mask].sum())
    lower, upper = wilson_interval(hits, int(mask.sum()))

    row = table.loc["vol_q3"]
    assert int(row["n"]) == hits + int(mask.sum()) - hits
    assert row["hit_rate_ci_lower"] == pytest.approx(lower)
    assert row["hit_rate_ci_upper"] == pytest.approx(upper)
    assert row["direction_accuracy"] == pytest.approx(hits / row["n"])


def test_a_one_row_regime_reports_nan_dispersion_instead_of_crashing() -> None:
    y = pd.Series([0.10, -0.20, 0.30, 0.40])
    p = pd.Series([0.05, -0.10, 0.10, 0.20])
    labels = pd.Series(["busy", "calm", "calm", "calm"])

    table = evaluate_by_regime(y, p, labels).set_index("regime")

    singleton = table.loc["busy"]
    assert int(singleton["n"]) == 1
    assert singleton["rmse"] == pytest.approx(0.05)
    assert singleton["direction_accuracy"] == 1.0
    # Sample dispersion is undefined on one observation, and a wide Wilson
    # interval is still a valid one - neither is a reason to raise.
    assert np.isnan(singleton["std_actual_return"])
    assert 0.0 <= singleton["hit_rate_ci_lower"] <= 1.0
    assert singleton["hit_rate_ci_upper"] == pytest.approx(wilson_interval(1, 1)[1])
    assert int(table.loc["calm", "n"]) == 3


def test_unlabelled_rows_are_dropped_rather_than_reported_as_a_regime() -> None:
    y = pd.Series([0.10, -0.20, 0.30])
    p = pd.Series([0.05, -0.10, 0.10])
    labels = pd.Series(["calm", None, "calm"])

    table = evaluate_by_regime(y, p, labels)

    assert table["regime"].tolist() == ["calm"]
    assert int(table["n"].iloc[0]) == 2


def test_evaluate_by_regime_rejects_misaligned_inputs() -> None:
    with pytest.raises(AnalysisError, match="aligned"):
        evaluate_by_regime(pd.Series([0.1, 0.2]), pd.Series([0.1]), pd.Series(["a", "b"]))


# ------------------------------------------------------------ return distributions

def test_return_distribution_report_matches_hand_computed_statistics() -> None:
    values = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    index = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    frame = pd.DataFrame({"target": values}, index=index)

    row = return_distribution_report(frame).iloc[0]

    assert row["column"] == "target"
    assert int(row["n"]) == 10
    assert row["mean"] == pytest.approx(0.5)
    assert row["std"] == pytest.approx(3.0276503540974917)
    assert row["median"] == pytest.approx(0.5)
    assert row["p05"] == pytest.approx(-3.55)
    assert row["p25"] == pytest.approx(-1.75)
    assert row["p75"] == pytest.approx(2.75)
    assert row["p95"] == pytest.approx(4.55)
    assert row["min"] == -4.0
    assert row["max"] == 5.0
    # A target of exactly 0.0 is not a positive return, so 5 of 10 rows count.
    assert row["positive_fraction"] == pytest.approx(0.5)
    # The series is symmetric about its mean, so the skew is exactly zero - which
    # is the one shape statistic a hand can confirm without a table.
    assert row["skew"] == pytest.approx(0.0, abs=1e-12)
    assert row["excess_kurtosis"] == pytest.approx(-1.2, abs=1e-9)


def test_the_target_is_reported_even_with_no_prediction_columns() -> None:
    """A target-only report is the diagnostic, not a degenerate request."""
    frame = make_predictions()

    table = return_distribution_report(frame, pred_cols=[])

    assert list(table.columns) == list(DISTRIBUTION_COLUMNS)
    assert table["column"].tolist() == ["target"]
    assert int(table["n"].iloc[0]) == N


def test_prediction_columns_are_discovered_but_bounds_and_probabilities_are_not() -> None:
    frame = make_predictions()

    table = return_distribution_report(frame)

    # prob_calibrated, *_lo, *_hi and *_prob_empirical are not return
    # distributions; only the point forecasts and the target are.
    assert table["column"].tolist() == ["target", *[f"{m}_pred" for m in MODELS]]


def test_horizon_labels_come_from_the_mapping_and_must_cover_every_column() -> None:
    frame = make_predictions()

    table = return_distribution_report(
        frame, horizons={"target": "30d", "ridge_pred": "30d", "xgboost_pred": "3d",
                         "random_forest_pred": "3d"},
    )
    assert set(table["horizon"]) == {"30d", "3d"}
    assert table.loc[table["column"] == "target", "horizon"].iloc[0] == "30d"

    with pytest.raises(AnalysisError, match="no label for column"):
        return_distribution_report(frame, horizons={"target": "30d"})


def test_shape_statistics_are_nan_where_they_are_undefined() -> None:
    frame = pd.DataFrame({"target": [0.1, 0.1, 0.1, np.nan]}, index=range(4))

    row = return_distribution_report(frame).iloc[0]

    assert int(row["n"]) == 3
    assert row["skew"] != row["skew"]  # NaN: no dispersion, so no shape to report
    assert row["excess_kurtosis"] != row["excess_kurtosis"]


# ---------------------------------------------------------- prediction intervals

def test_the_coverage_gap_is_exactly_zero_when_coverage_is_nominal() -> None:
    frame = make_predictions()
    summary = pd.DataFrame(
        {
            "model": list(MODELS),
            "mean_fold_coverage": [0.9, 0.9, 0.9],
            "nominal_coverage": [0.9, 0.9, 0.9],
            "mean_interval_width": [0.05, 0.06, 0.07],
        }
    )

    table = prediction_interval_report(make_result(frame, interval_summary=summary))

    assert list(table.columns) == list(INTERVAL_COLUMNS)
    assert (table["coverage_gap"] == 0.0).all()
    assert table["mean_fold_coverage"].tolist() == [0.9, 0.9, 0.9]
    assert table["nominal_coverage"].tolist() == [0.9, 0.9, 0.9]


def test_a_band_that_undershoots_nominal_coverage_shows_a_negative_gap() -> None:
    frame = make_predictions()
    summary = pd.DataFrame(
        {"model": list(MODELS), "mean_fold_coverage": [0.8, 0.8, 0.8], "nominal_coverage": [0.9] * 3}
    )

    table = prediction_interval_report(make_result(frame, interval_summary=summary))

    assert table["coverage_gap"].tolist() == pytest.approx([-0.1, -0.1, -0.1])


def test_interval_widths_and_wilson_bounds_come_from_the_pooled_predictions() -> None:
    frame = make_predictions()

    table = prediction_interval_report(make_result(frame)).set_index("model")

    for model, fold_width in zip(MODELS, (0.05, 0.06, 0.07)):
        row = table.loc[model]
        width = (frame[f"{model}_hi"] - frame[f"{model}_lo"]).to_numpy()
        # The recorded fold summary is the reported mean; the median can only come
        # from the pooled predictions, and the gap between the two is the honest
        # statement that they are measured on different row sets.
        assert row["mean_interval_width"] == pytest.approx(fold_width)
        assert row["median_interval_width"] == pytest.approx(float(np.median(width)))
        assert int(row["n_predictions"]) == N
        assert 0.0 <= row["coverage_ci_lower"] <= row["coverage_ci_upper"] <= 1.0

    inside = (
        (frame["target"] >= frame["ridge_lo"]) & (frame["target"] <= frame["ridge_hi"])
    ).to_numpy()
    lower, upper = wilson_interval(int(inside.sum()), N)
    row = table.loc["ridge"]
    assert row["coverage_ci_lower"] == pytest.approx(lower)
    assert row["coverage_ci_upper"] == pytest.approx(upper)


def test_the_fold_summary_wins_over_the_pooled_width_when_it_is_present() -> None:
    frame = make_predictions()
    summary = pd.DataFrame(
        {"model": list(MODELS), "mean_fold_coverage": [0.9] * 3, "nominal_coverage": [0.9] * 3,
         "mean_interval_width": [1.0, 2.0, 3.0]}
    )

    table = prediction_interval_report(make_result(frame, interval_summary=summary))

    assert table["mean_interval_width"].tolist() == [1.0, 2.0, 3.0]


def test_pooled_coverage_is_the_fallback_when_no_fold_summary_exists() -> None:
    frame = make_predictions()

    table = prediction_interval_report(
        make_result(frame, interval_summary=pd.DataFrame())
    ).set_index("model")

    inside = (
        (frame["target"] >= frame["ridge_lo"]) & (frame["target"] <= frame["ridge_hi"])
    ).to_numpy()
    assert table.loc["ridge", "mean_fold_coverage"] == pytest.approx(inside.mean())
    assert table.loc["ridge", "mean_interval_width"] == pytest.approx(
        float((frame["ridge_hi"] - frame["ridge_lo"]).mean())
    )
    # No nominal figure is available, so the gap is undefined rather than zero.
    assert np.isnan(table.loc["ridge", "coverage_gap"])


# ----------------------------------------------------------------- calibration

def test_calibration_report_scores_both_probability_routes_for_every_model() -> None:
    frame = make_predictions()

    table = calibration_report(make_result(frame))

    assert list(table.columns) == list(CALIBRATION_COLUMNS)
    assert len(table) == 2 * len(MODELS)
    assert set(table["source"]) == {"empirical", "calibrated"}
    assert table.groupby("model")["source"].apply(set).eq({"empirical", "calibrated"}).all()
    assert not table["degenerate"].any()


def test_the_brier_score_is_the_hand_computed_mean_squared_error() -> None:
    frame = make_predictions()

    table = calibration_report(make_result(frame)).set_index(["model", "source"])

    prob = frame["ridge_prob_empirical"].to_numpy()
    labels = (frame["target"].to_numpy() > 0.0).astype(float)
    assert table.loc[("ridge", "empirical"), "brier"] == pytest.approx(
        float(np.mean((prob - labels) ** 2))
    )


def test_the_shared_calibrated_probability_appears_once_per_model() -> None:
    """It is one model per fold, so its row must be identical for every model."""
    frame = make_predictions()

    table = calibration_report(make_result(frame))
    shared = table[table["source"] == "calibrated"]

    assert len(shared) == len(MODELS)
    for column in ("n", "brier", "n_bins", "bin_coverage_mae", "calibration_slope", "degenerate"):
        assert shared[column].nunique() == 1, f"{column} varies across models"

    prob = frame["prob_calibrated"].to_numpy()
    labels = (frame["target"].to_numpy() > 0.0).astype(float)
    assert shared["brier"].iloc[0] == pytest.approx(float(np.mean((prob - labels) ** 2)))


def test_a_flat_probability_is_flagged_instead_of_being_scored() -> None:
    """All-0.5 has a computable Brier, which is exactly why it needs a flag."""
    frame = make_predictions()
    frame["prob_calibrated"] = 0.5

    table = calibration_report(make_result(frame))
    shared = table[table["source"] == "calibrated"]
    empirical = table[table["source"] == "empirical"]

    assert shared["degenerate"].all()
    # The per-model empirical route has a real spread and must not be flagged.
    assert not empirical["degenerate"].any()

    # A constant probability has no slope to report - undefined, not zero.
    assert shared["calibration_slope"].isna().all()
    assert np.isfinite(shared["brier"].iloc[0])
    assert empirical["calibration_slope"].notna().all()


def test_a_brier_score_outside_the_unit_interval_is_refused() -> None:
    frame = make_predictions()
    frame["ridge_prob_empirical"] = 1.7  # a regression output, not a probability

    with pytest.raises(AnalysisError, match=r"outside \[0, 1\]"):
        calibration_report(make_result(frame))


def test_reliability_reports_both_ends_of_the_populated_curve() -> None:
    frame = make_predictions()

    row = calibration_report(make_result(frame)).iloc[0]

    assert int(row["n_bins"]) == 10
    assert row["bin_coverage_mae"] >= 0.0
    assert 0.0 <= row["reliability_low"] <= 1.0
    assert 0.0 <= row["reliability_high"] <= 1.0


# --------------------------------------------------------------------- deciles

def test_the_top_decile_earns_at_least_as_much_as_the_bottom_decile() -> None:
    """A strictly monotone model: the tails must order by construction, not luck."""
    n = 1000
    index = pd.date_range("2024-03-01", periods=n, freq="1h", tz="UTC")
    pred = np.linspace(-0.05, 0.05, n)
    frame = pd.DataFrame({"target": 2.0 * pred, "ridge_pred": pred}, index=index)
    frame["ridge_lo"] = pred - 0.1
    frame["ridge_hi"] = pred + 0.1
    frame["ridge_prob_empirical"] = np.linspace(0.01, 0.99, n)

    table = decile_report(make_result(frame, models=["ridge"]))

    assert list(table.columns) == list(DECILE_COLUMNS)
    row = table.iloc[0]
    assert int(row["n_deciles"]) == 10
    assert row["top_decile_return"] > row["bottom_decile_return"]
    assert row["long_short_spread"] == pytest.approx(
        row["top_decile_return"] - row["bottom_decile_return"], rel=0.5
    )
    assert row["spearman_ic"] == pytest.approx(1.0)


def test_a_useless_model_reports_an_undefined_ic_rather_than_a_confident_zero() -> None:
    frame = make_predictions(n=400)

    table = decile_report(make_result(frame, models=["ridge"])).set_index("model")

    assert np.isfinite(table.loc["ridge", "spearman_ic"])
    constant = frame.assign(ridge_pred=0.0)
    flat = decile_report(make_result(constant, models=["ridge"]))
    assert np.isnan(flat["spearman_ic"].iloc[0])


# -------------------------------------------------------------------- ablation

def test_ablation_fits_every_variant_exactly_once() -> None:
    frame = make_predictions()
    result = make_result(frame)
    seen: list[tuple[str, ...]] = []

    def fit_predict(features: Sequence[str], seed: int) -> np.ndarray:
        seen.append(tuple(features))
        assert isinstance(seed, int)
        # A trivial "model": shrink the forecast when the strongest group is gone.
        return frame["ridge_pred"].to_numpy() * (1.0 if "sma_20" in features else 0.4)

    table = run_ablation(result, fit_predict, FEATURE_GROUPS)

    assert len(seen) == len(FEATURE_GROUPS) + 1
    assert seen[0] == FEATURE_COLUMNS
    assert seen[1] == tuple(c for c in FEATURE_COLUMNS if c not in FEATURE_GROUPS["technical"])
    assert len(seen) == len(table)


def test_ablation_returns_one_row_per_variant_with_a_dropped_column() -> None:
    frame = make_predictions()
    result = make_result(frame)
    calls: list[Sequence[str]] = []

    def fit_predict(features: Sequence[str], seed: int) -> np.ndarray:
        calls.append(features)
        return frame["ridge_pred"].to_numpy() * (1.0 if "sma_20" in features else 0.4)

    table = run_ablation(result, fit_predict, FEATURE_GROUPS).set_index("variant")

    assert set(table.index) == {"full", *(f"drop_{g}" for g in FEATURE_GROUPS)}
    assert "dropped" in table.columns
    assert table.loc["full", "dropped"] is None
    assert table.loc["drop_derivatives", "dropped"] == "derivatives"
    assert table.loc["full", "n_features"] == len(FEATURE_COLUMNS)
    assert table.loc["drop_derivatives", "n_features"] == len(FEATURE_COLUMNS) - 1
    # The deltas are against the full variant fitted through the same callable.
    assert table.loc["full", "delta_rmse"] == 0.0
    assert table.loc["drop_technical", "delta_rmse"] > 0.0
    assert table.loc["drop_context", "delta_rmse"] == 0.0
    assert table["n"].tolist() == [N] * (len(FEATURE_GROUPS) + 1)


def test_ablation_accepts_group_objects_and_rejects_anything_else() -> None:
    frame = make_predictions()
    result = make_result(frame)
    group = types.SimpleNamespace(name="technical", features=list(FEATURE_GROUPS["technical"]))
    calls: list[Sequence[str]] = []

    def fit_predict(features: Sequence[str], seed: int) -> np.ndarray:
        calls.append(features)
        return frame["ridge_pred"].to_numpy()

    table = run_ablation(result, fit_predict, [group])
    assert len(calls) == 2
    assert table["dropped"].tolist() == [None, "technical"]

    with pytest.raises(AnalysisError, match="FeatureGroup-like"):
        run_ablation(result, fit_predict, ["technical"])


def test_ablation_refuses_a_misaligned_prediction() -> None:
    result = make_result()

    with pytest.raises(AnalysisError, match="aligned"):
        run_ablation(result, lambda features, seed: np.zeros(5), FEATURE_GROUPS)


def test_ablation_needs_a_callable_because_it_never_imports_the_harness() -> None:
    with pytest.raises(AnalysisError, match="callable"):
        run_ablation(make_result(), "not-a-function", FEATURE_GROUPS)  # type: ignore[arg-type]


def test_ablation_rejects_a_group_naming_an_unknown_column() -> None:
    with pytest.raises(AnalysisError, match="not in feature_columns"):
        run_ablation(
            make_result(), lambda features, seed: np.zeros(N), {"ghost": ["not_a_feature"]}
        )


# --------------------------------------------------------------------- analyse

def test_analyse_returns_every_documented_report() -> None:
    frame = make_predictions()

    reports = analyse(make_result(frame))

    assert list(reports) == list(ANALYSIS_REPORTS)
    assert all(isinstance(v, pd.DataFrame) for v in reports.values())
    assert reports["regime_volatility"]["regime"].tolist() == [
        "vol_q1", "vol_q2", "vol_q3", "vol_q4",
    ]
    assert reports["regime_trend"]["regime"].str.startswith("trend_q").all()
    assert reports["regime_volatility"]["model"].tolist() == ["ridge"] * 4
    assert len(reports["intervals"]) == len(MODELS)
    assert len(reports["calibration"]) == 2 * len(MODELS)
    assert len(reports["deciles"]) == len(MODELS)
    assert reports["distribution"]["column"].tolist()[0] == "target"
    assert set(reports["distribution"]["horizon"]) == {"30d"}


def test_analyse_conditions_on_regimes_supplied_by_the_caller() -> None:
    """A HorizonResult pools predictions, not features, so the caller supplies the latter."""
    frame = make_predictions().drop(columns=["realised_vol", "trend"])
    vol = make_predictions()["realised_vol"]

    reports = analyse(make_result(frame), n_bins=2, regime_values=vol)

    assert reports["regime_volatility"]["regime"].tolist() == ["vol_q1", "vol_q2"]
    assert reports["regime_trend"].empty
    assert list(reports["regime_volatility"].columns) == ["model", *REGIME_COLUMNS]


def test_analyse_keeps_its_keys_when_there_is_no_regime_source() -> None:
    frame = make_predictions().drop(columns=["realised_vol", "trend"])

    reports = analyse(make_result(frame))

    assert list(reports["regime_volatility"].columns) == ["model", *REGIME_COLUMNS]
    assert reports["regime_volatility"].empty
    assert not reports["deciles"].empty
    assert not reports["intervals"].empty


def test_analyse_scores_the_regimes_on_the_best_pooled_model() -> None:
    frame = make_predictions()
    pooled = pd.DataFrame({"rmse": [0.09, 0.02, 0.05]}, index=["ridge", "xgboost", "random_forest"])

    reports = analyse(make_result(frame, pooled_metrics=pooled))

    assert set(reports["regime_volatility"]["model"]) == {"xgboost"}


def test_analyse_refuses_something_that_is_not_a_horizon_result() -> None:
    with pytest.raises(AnalysisError, match="predictions frame"):
        analyse(types.SimpleNamespace(horizon="30d"))
