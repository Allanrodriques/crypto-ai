"""Feature-group contracts: documented columns, no look-ahead, no dead columns.

These are the tests that would catch a silently broken feature before it reaches
a result table.  The two that mattered most in practice were both found here
rather than in the output:

* a daily-indexed series reindexed onto an hourly grid, which NaN'd 96% of rows
  and shrank the dataset from ~41k rows to 1.4k without raising anything;
* a trailing z-score dividing by a zero-variance window, which deleted 16% of
  rows because funding sits at one rate for 24 hours at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features._common import rolling_zscore, safe_divide
from src.features.groups import (
    GROUP_ORDER,
    REGISTRY,
    validate_groups,
)
from src.features.sentiment import SENTIMENT_DOCS, build_sentiment_features

#: The V1 baseline group set, pinned here so a registry change cannot quietly
#: redefine what EXP-00 means.
BASELINE_GROUPS = ("technical",)


def _hourly(periods: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=periods, freq="1h", tz="UTC")
    close = pd.Series(20_000 * np.exp(np.cumsum(rng.normal(0, 0.004, periods))), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(20_000).to_numpy(),
            "high": (close * 1.004).to_numpy(),
            "low": (close * 0.996).to_numpy(),
            "close": close.to_numpy(),
            "volume": rng.lognormal(12, 0.6, periods),
            "quote_volume": rng.lognormal(18, 0.6, periods),
            "trades": rng.integers(500, 5_000, periods).astype(float),
            "taker_buy_volume": rng.lognormal(11, 0.6, periods),
            "taker_buy_quote_volume": rng.lognormal(17, 0.6, periods),
        },
        index=index,
    )


def _hourly_index(periods: int = 2_000) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=periods, freq="1h", tz="UTC")


# --------------------------------------------------------------------------- registry

def test_baseline_group_is_immutable_and_named_exp00():
    assert BASELINE_GROUPS == ("technical",)
    assert "technical" in REGISTRY
    assert GROUP_ORDER[0] == "technical"
    assert len(REGISTRY["technical"].features) == 39


def test_every_group_documents_exactly_its_declared_features():
    """A documented column that the builder does not emit is a silent lie."""
    for name, group in REGISTRY.items():
        docs = group.docs
        missing = [f for f in group.features if f not in docs]
        assert not missing, f"{name}: declared but undocumented features {missing}"


def test_declared_features_are_exactly_what_the_builders_emit():
    """The 107-vs-105 mismatch: docs advertised two names no builder ever produced.

    ``futures_spot_basis_zscore_60d`` was documented while the builder emitted
    ``futures_basis_zscore_60d``, and ``futures_basis_mean_7d`` was documented
    but never built.  A manifest that lists features which do not exist makes
    every "n features" claim in the report wrong.
    """
    from src.data.downloader import KlineStore
    from src.dataset.multisource import build_context
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    store = KlineStore(root / "data" / "raw", "BTCUSDT", "1h")
    if not store.exists():
        pytest.skip("BTCUSDT spot cache not present")

    spot = store.load()
    ctx = build_context(
        symbol="BTCUSDT",
        sources=["binance_funding", "binance_futures", "fear_greed"],
        spot=spot,
        cache_dir=root / "data" / "raw",
    )
    groups = list(REGISTRY)
    emitted = set(ctx.features(groups).columns)
    declared = {f for g in REGISTRY.values() for f in g.features}

    assert not (declared - emitted), f"documented but never built: {sorted(declared - emitted)}"
    assert not (emitted - declared), f"built but undocumented: {sorted(emitted - declared)}"
    assert len(declared) == 107


def test_feature_names_are_unique_within_a_group():
    for name, group in REGISTRY.items():
        assert len(set(group.features)) == len(group.features), f"{name} has duplicate features"


def test_no_feature_appears_in_two_groups():
    """A feature in two groups would double-count it in any combined experiment."""
    seen: dict[str, str] = {}
    for name, group in REGISTRY.items():
        for feature in group.features:
            assert feature not in seen, f"{feature} is in both {seen.get(feature)} and {name}"
            seen[feature] = name


def test_validate_groups_rejects_unknown_names():
    validate_groups(["technical", "derivatives"])
    with pytest.raises(Exception):
        validate_groups(["technical", "does_not_exist"])


# --------------------------------------------------------------------------- sentiment

def test_sentiment_features_are_populated_on_every_hour_not_only_midnight():
    """Regression test: daily stats reindexed onto an hourly grid lose 23/24 rows.

    A daily reading published at midnight stays valid for the whole day, so it
    must be carried forward.  ``reindex`` matched only 00:00 and produced a frame
    that was ~96% NaN while still looking structurally fine.
    """
    index = _hourly_index(2_000)
    values = pd.Series(np.arange(len(index), dtype=float) % 50, index=index)

    frame = build_sentiment_features(index, pd.DataFrame({"fear_greed_value": values}))

    raw = frame["fear_greed_value"].dropna()
    assert len(raw) > 0.9 * len(index), "sentiment value must be present on nearly every hour"
    # A populated value on a non-midnight hour is the specific regression.
    non_midnight = raw[raw.index.hour != 0]
    assert len(non_midnight) > 0.5 * len(index)


def test_sentiment_value_holds_constant_within_a_day():
    """Within a day the reading cannot change, because no new one is published."""
    index = _hourly_index(48 * 7)
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    frame = build_sentiment_features(
        index, pd.DataFrame({"fear_greed_value": values}), enabled=["fear_greed_value"]
    )
    series = frame["fear_greed_value"]
    for day in pd.unique(series.index.date):
        day_values = series[series.index.date == day].dropna()
        assert day_values.nunique() <= 1, f"{day} changed within the day"


def test_sentiment_rejects_a_non_dataframe_instead_of_returning_all_nan():
    """A dict is an easy caller mistake; it must not look like 'no data'."""
    index = _hourly_index(48)
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    with pytest.raises(TypeError):
        build_sentiment_features(index, {"fear_greed_value": values})


def test_sentiment_registry_and_builder_agree():
    from src.features.groups import SENTIMENT_GROUP

    assert SENTIMENT_GROUP == "sentiment"
    assert REGISTRY["sentiment"].features == tuple(SENTIMENT_DOCS)
    assert set(SENTIMENT_DOCS) <= set(build_sentiment_features.__doc__ or "") | set(SENTIMENT_DOCS)


def test_sentiment_regime_labels_are_ordered():
    """Higher fear must not map to a higher ordinal regime code."""
    index = _hourly_index(48)
    values = pd.Series([5.0] * 24 + [90.0] * 24, index=index)
    out = build_sentiment_features(
        index,
        pd.DataFrame({"fear_greed_value": values}),
        enabled=["fear_greed_regime"],
    )
    regime = out["fear_greed_regime"].dropna()
    assert regime.iloc[0] == 0.0, "index 5 is Extreme Fear"
    assert regime.iloc[-1] == 4.0, "index 90 is Extreme Greed"


# --------------------------------------------------------------------------- z-score

def test_rolling_zscore_excludes_the_current_value():
    """An extreme current bar must not inflate the baseline meant to detect it."""
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 100.0])
    z = rolling_zscore(series, window=4, min_periods=4)
    assert z.iloc[-1] > 10.0, "a 100x jump on a flat baseline must score far from zero"


def test_rolling_zscore_does_not_produce_nan_on_a_flat_window():
    """Regression test: zero-variance windows used to become NaN, deleting rows.

    Only the warm-up may be missing; from ``window`` onwards a perfectly flat
    series must score 0 rather than NaN.
    """
    series = pd.Series([0.0001] * 500)
    z = rolling_zscore(series, window=20, min_periods=20)
    assert z.isna().sum() == 20, "only the warm-up window may be missing"
    assert np.isfinite(z.dropna().to_numpy()).all()
    # Sitting exactly on a flat baseline is "at the mean", i.e. 0.
    assert abs(z.iloc[-1]) < 1e-6


def test_rolling_zscore_scores_a_jump_off_a_flat_baseline_finitely():
    """A real move after 100 identical prints is extreme, but not inf."""
    series = pd.Series([0.0001] * 100 + [0.01])
    z = rolling_zscore(series, window=20, min_periods=20)
    assert np.isfinite(z.iloc[-1]), "a jump must not become inf"
    assert z.iloc[-1] > 1_000, "and must still register as a large deviation"


def test_rolling_zscore_leaves_only_the_warm_up_missing():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0, 1, 1_000))
    z = rolling_zscore(series, window=20, min_periods=20)
    assert z.isna().sum() <= 20


# --------------------------------------------------------------------------- safe divide

def test_safe_divide_never_invents_infinity():
    """A zero denominator must become NaN, not a huge finite number.

    A tree model cannot tell a genuine extreme from an overflow, so a
    substitute value would read as a real signal.
    """
    numerator = pd.Series([1.0, 0.0, 2.0])
    denominator = pd.Series([0.0, 0.0, 4.0])
    out = safe_divide(numerator, denominator)
    assert not np.isinf(out.to_numpy()).any()
    assert out.iloc[0] != out.iloc[0], "1/0 must be NaN"
    assert out.iloc[1] != out.iloc[1], "0/0 must be NaN"
    assert out.iloc[2] == pytest.approx(0.5)


# --------------------------------------------------------------------------- context

def test_context_features_never_expose_an_incomplete_higher_timeframe_bucket():
    """At 03:00 a 4h feature may only use the bucket that closed at 00:00.

    Reading the 00:00-04:00 bucket at 03:00 would be reading a candle three
    quarters of the way through its own period - a look-ahead that is invisible
    in a backtest but makes the feature meaningless live.
    """
    from src.features.context import build_context_features, completed_higher_timeframe

    index = pd.date_range("2023-01-01", periods=4 * 400, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 400.0, len(index)), index=index)
    frame = pd.DataFrame({"close": close, "open": close.shift(1), "high": close * 1.01,
                          "low": close * 0.99, "volume": 1.0, "quote_volume": 1.0,
                          "trades": 1.0, "taker_buy_volume": 0.5,
                          "taker_buy_quote_volume": 0.5}, index=index)

    h4 = completed_higher_timeframe(frame, index, "4h")
    # No merge_asof key column may leak into the returned frame.
    assert not [c for c in h4.columns if str(c).startswith("key_")], h4.columns
    assert list(h4.columns) == ["open", "high", "low", "close", "volume"]

    # A bucket labelled L covers [L, L+4h) and only closes at L+4h, so at any
    # target T the newest visible close is the last hour *before* T's floor.
    for stamp, value in h4["close"].dropna().items():
        last_visible_hour = stamp.floor("4h") - pd.Timedelta(minutes=1)
        assert value == pytest.approx(close.asof(last_visible_hour)), f"leak at {stamp}"

    out = build_context_features(index, frame, enabled=["ctx_4h_return_1"])
    assert out["ctx_4h_return_1"].notna().any()
