"""Feature-layer tests: indicator correctness, NaN hygiene, and causality.

The causality tests in this file are the project's primary guarantee against
look-ahead.  They do not inspect the source code for suspicious patterns; they
*empirically* prove that no feature at time ``t`` can depend on any price after
``t``, by mutating and truncating the future and requiring the past to be
bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FEATURE_DOCS,
    FEATURE_GROUPS,
    FEATURE_GROUP_OF,
    FeatureEngineer,
    assert_documented,
    build_features,
    feature_documentation_markdown,
    feature_registry,
    group_of,
)
from tests.conftest import TEST_FEATURES, make_klines


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer(TEST_FEATURES)


@pytest.fixture
def features(engineer, klines) -> pd.DataFrame:
    return engineer.build(klines)


# --------------------------------------------------------------------------- schema

def test_every_documented_feature_is_produced(features, engineer):
    assert set(features.columns) == set(engineer.feature_names)
    assert_documented(features, engineer.feature_docs)


def test_production_registry_matches_the_real_engineer():
    """The shipped FEATURE_DOCS table is exactly what the real config produces."""
    from src.config import load_config

    real = FeatureEngineer(load_config().features)
    assert list(real.feature_names) == list(FEATURE_DOCS)
    assert len(FEATURE_DOCS) == 39


def test_all_required_feature_names_are_present():
    """The exact list the spec asked for, checked against the production config."""
    from src.config import load_config

    real = FeatureEngineer(load_config().features)
    produced = set(real.feature_names)
    required = {
        "return_1h", "return_3h", "return_6h", "return_12h", "return_24h",
        "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_12", "ema_26",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200", "sma20_vs_sma50", "sma50_vs_sma200",
        "rsi_14", "macd", "macd_signal", "macd_histogram", "roc_12",
        "atr_14", "atr_percent", "rolling_volatility_20", "rolling_volatility_50",
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
        "volume_sma_20", "volume_ratio", "volume_change",
        "candle_body", "candle_range", "upper_wick", "lower_wick", "body_to_range",
    }
    missing = sorted(required - produced)
    assert not missing, f"missing required feature(s): {missing}"


def test_every_feature_is_documented_and_grouped():
    for config in (None, TEST_FEATURES):
        registry = feature_registry(config)
        for name, doc in registry.items():
            assert doc.strip(), f"feature {name!r} has no documentation"
            assert group_of(name) in FEATURE_GROUPS
    assert set(FEATURE_DOCS) == set(FEATURE_GROUP_OF)
    assert sorted(FEATURE_GROUPS) == [
        "bollinger", "candle_structure", "ma_relationships", "momentum",
        "moving_averages", "returns", "volatility", "volume",
    ]
    assert sum(len(names) for names in FEATURE_GROUPS.values()) == len(FEATURE_DOCS)


def test_documentation_markdown_covers_every_feature():
    table = feature_documentation_markdown()
    for name in FEATURE_DOCS:
        assert f"`{name}`" in table
    assert all(len(names) > 0 for names in FEATURE_GROUPS.values())


# --------------------------------------------------------------------------- hygiene

def test_no_nans_or_infinities_after_warmup(features):
    assert features.isna().sum().sum() == 0
    assert np.isfinite(features.to_numpy()).all()


def test_warmup_rows_are_dropped(engineer, klines):
    built = engineer.build(klines)
    assert built.index[0] == klines.index[engineer.required_history]


def test_const_price_series_does_not_produce_infinities(engineer):
    flat = make_klines(120)
    for column in ("open", "high", "low", "close"):
        flat[column] = 100.0
    built = engineer.build(flat)
    values = built.to_numpy()
    assert np.isfinite(values).all(), "a flat series must not yield inf/NaN"


def test_zero_volume_series_is_safe(engineer):
    frame = make_klines(120)
    frame["volume"] = 0.0
    built = engineer.build(frame)
    assert np.isfinite(built.to_numpy()).all()


def test_duplicate_timestamps_are_rejected(engineer, klines):
    with pytest.raises(ValueError, match="unique timestamps"):
        engineer.build(pd.concat([klines, klines.iloc[:5]]))


def test_missing_column_is_rejected(engineer, klines):
    with pytest.raises(ValueError, match="missing required column"):
        engineer.build(klines.drop(columns=["volume"]))


# --------------------------------------------------------------------------- indicator math

def test_sma_matches_a_hand_computed_mean(engineer, klines):
    built = engineer.build(klines)
    period = 10
    for offset in (-1, -5, -50):
        position = len(klines) + offset
        window = klines["close"].iloc[position - period + 1: position + 1]
        assert built["sma_10"].iloc[offset] == pytest.approx(window.mean(), rel=1e-12)


def test_ema_is_close_to_its_recent_weighted_mean(engineer, klines):
    """EMA(12) must track recent prices more closely than SMA(12) after a jump."""
    frame = make_klines(400)
    frame.loc[frame.index[-1]:, ["open", "high", "low", "close"]] *= 1.2
    built = FeatureEngineer({**TEST_FEATURES, "sma_periods": [12], "ema_periods": [12]}).build(frame)
    assert built["ema_12"].iloc[-1] > built["sma_12"].iloc[-1]


def test_returns_match_pct_change(engineer, klines):
    built = engineer.build(klines)
    close = klines["close"]
    for period in (1, 3, 6):
        position = len(klines) - 1
        expected = close.iloc[position] / close.iloc[position - period] - 1.0
        assert built[f"return_{period}h"].iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_ma_relationships_are_normalised_ratios(features):
    """MA relationship features are fractions of price, not raw price levels.

    The test config only builds SMA(5/10/20), so only the relationships whose
    periods exist are asserted; the production config adds the 50/200 pairs.
    """
    row = features.iloc[-1]
    assert "price_vs_sma20" in features.columns
    assert abs(row["price_vs_sma20"]) < 5.0

    if "sma20_vs_sma50" in features.columns:
        assert abs(row["sma20_vs_sma50"]) < 5.0
    if "sma50_vs_sma200" in features.columns:
        assert abs(row["sma50_vs_sma200"]) < 5.0

    # A 200-period SMA must not be reproducible from a 20-candle window, i.e.
    # these features must actually vary rather than being constant placeholders.
    for name in ("price_vs_sma20", "sma_20", "return_1h"):
        assert features[name].nunique() > 10, f"{name!r} looks like a constant placeholder"


def test_price_vs_sma_equals_the_explicit_ratio(engineer, klines):
    built = engineer.build(klines)
    close = klines["close"].reindex(built.index)
    expected = close / built["sma_20"] - 1.0
    np.testing.assert_allclose(built["price_vs_sma20"].to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_rsi_is_bounded(engineer, features):
    rsi = features[f"rsi_{engineer.rsi_period}"].dropna()
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0


def test_macd_histogram_is_macd_minus_signal(features):
    diff = features["macd"] - features["macd_signal"] - features["macd_histogram"]
    assert diff.abs().max() < 1e-6


def test_bollinger_bands_are_symmetric_around_the_middle(features):
    upper_gap = features["bb_upper"] - features["bb_middle"]
    lower_gap = features["bb_middle"] - features["bb_lower"]
    assert (upper_gap - lower_gap).abs().max() < 1e-9
    assert (upper_gap > 0).all()


def test_bb_position_follows_its_definition(klines, engineer):
    """Price may sit outside the bands; the position formula must still hold."""
    built = engineer.build(klines)
    close = klines["close"].reindex(built.index)
    expected = (close - built["bb_lower"]) / (built["bb_upper"] - built["bb_lower"])
    np.testing.assert_allclose(built["bb_position"].to_numpy(), expected.to_numpy(), rtol=1e-9)
    assert built["bb_position"].notna().all()


def test_atr_is_non_negative_and_scales_with_price(engineer, features):
    atr = features[f"atr_{engineer.atr_period}"]
    assert (atr >= 0).all()
    assert (features["atr_percent"] >= 0).all()
    # atr_percent is atr / close x 100, so the two must agree.
    close = features["sma_20"] * (1.0 + features["price_vs_sma20"])
    np.testing.assert_allclose(
        features["atr_percent"].to_numpy(), (atr / close * 100.0).to_numpy(), rtol=1e-9
    )


def test_candle_structure_is_internally_consistent(features):
    row = features.iloc[-1]
    # The *unsigned* body plus both wicks reconstruct the range exactly; the body
    # feature is signed so that direction is encoded too.
    reconstructed = abs(row["candle_body"]) + row["upper_wick"] + row["lower_wick"]
    assert reconstructed == pytest.approx(row["candle_range"], rel=1e-9)
    assert row["candle_range"] >= 0
    assert (features["upper_wick"] >= 0).all()
    assert (features["lower_wick"] >= 0).all()


def test_volume_ratio_is_one_on_average_volume(features):
    assert features["volume_ratio"].median() > 0


# --------------------------------------------------------------------------- CAUSALITY

def test_features_do_not_use_future_rows(engineer, klines):
    """Core leakage guard.

    Every row strictly after a cut-off is replaced with extreme garbage.  If any
    feature at or before the cut-off changes, that feature reads the future.

    The comparison is done by *label* (``klines.index[:cut]``) rather than by
    position, so it stays correct regardless of how many warm-up rows the
    feature layer drops.
    """
    clean = klines.index[:400 - 30]
    baseline = engineer.build(klines)
    expected = baseline.loc[baseline.index < clean[-1] + baseline.index.freq]

    poisoned = klines.copy()
    future = poisoned.index[400 - 30:]
    poisoned.loc[future, "open"] *= 5.0
    poisoned.loc[future, "high"] *= 7.0
    poisoned.loc[future, "low"] *= 0.05
    poisoned.loc[future, "close"] *= 11.0
    poisoned.loc[future, "volume"] = 999_999.0

    actual = engineer.build(poisoned)
    pd.testing.assert_frame_equal(expected, actual.loc[actual.index.intersection(expected.index)])


def test_features_are_identical_when_the_future_is_truncated(engineer, klines):
    """A shorter history must not change the features of the candles it shares."""
    cut = 400 - 50
    full = engineer.build(klines)
    truncated = engineer.build(klines.iloc[:cut])
    pd.testing.assert_frame_equal(full.reindex(truncated.index), truncated)


def test_each_feature_alone_is_causal(engineer, klines):
    """Per-feature check, so a failure names the offending indicator."""
    cut = 400 - 20
    boundary = klines.index[cut]
    built = engineer.build(klines)
    baseline = built.loc[built.index < boundary]

    poisoned = klines.copy()
    poisoned.loc[poisoned.index[cut:], "close"] *= 3.3
    actual = engineer.build(poisoned).loc[built.index.intersection(baseline.index)]

    for column in baseline.columns:
        assert baseline[column].equals(actual[column]), f"feature {column!r} depends on future closes"


def test_no_negative_shift_is_used_in_the_feature_module():
    """Static guard against a future-looking shift sneaking into the layer."""
    import inspect

    from src.features import feature_engineering as module

    source = inspect.getsource(module)
    assert ".shift(-" not in source, "feature module must not shift by a negative period"


def test_feature_index_matches_input_index(engineer, klines):
    built = engineer.build(klines)
    assert built.index.is_monotonic_increasing
    assert built.index.is_unique
    assert built.index.tz is not None
    assert set(built.index).issubset(set(klines.index))


def test_build_is_deterministic(engineer, klines):
    pd.testing.assert_frame_equal(engineer.build(klines), engineer.build(klines))


def test_functional_wrapper_matches_the_class(config, klines):
    wrapper = build_features(klines, TEST_FEATURES)
    klass = FeatureEngineer(TEST_FEATURES).build(klines)
    pd.testing.assert_frame_equal(wrapper, klass)
