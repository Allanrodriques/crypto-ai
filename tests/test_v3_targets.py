"""Tests for V3 multi-horizon forward-return targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v3.horizons import Horizon, build_horizon_ladder, parse_horizon
from src.v3.targets import FutureReturnTargets, build_future_returns, overlaps_boundary


def hourly_close(n: int = 6000, start: str = "2022-01-01") -> pd.Series:
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    drift = np.linspace(0.0, 5.0, n)
    return pd.Series(100.0 * np.exp(drift), index=index, name="close")


def test_horizon_parsing() -> None:
    assert parse_horizon("1d") == pd.Timedelta(days=1)
    assert parse_horizon("30d") == pd.Timedelta(days=30)
    assert parse_horizon("180d") == pd.Timedelta(days=180)
    with pytest.raises(ValueError):
        parse_horizon("7")


def test_nominal_bars_uses_bars_per_day() -> None:
    """Regression test: 1d on an hourly grid is 24 bars, not 576.

    ``bars_per_unit`` is bars per *day*.  Converting via hours multiplied a 1d
    horizon by 24 twice, which would have been fed straight into the purge width.
    """
    ladder = build_horizon_ladder({"horizons": ["1d", "7d", "30d", "180d"]})
    expected = {"1d": 24, "7d": 168, "30d": 720, "180d": 4320}
    for horizon in ladder:
        assert horizon.nominal_bars == expected[horizon.label]
        assert horizon.nominal_bars == round(horizon.hours)


def test_targets_are_exact_forward_returns() -> None:
    close = hourly_close(1000)
    ladder = build_horizon_ladder({"horizons": ["1d"]})
    targets = build_future_returns(close, ladder)

    # shift(-24) is the exact price 24 hours later; the final 24 are NaN.
    expected = (close.shift(-24) / close) - 1.0
    pd.testing.assert_series_equal(
        targets.frame["future_return_1d"], expected, check_names=False
    )


def test_no_labels_fabricated_past_end_of_data() -> None:
    """Regression test for an ffill that invented a flat tail.

    The original implementation reindexed onto the union of observed and
    target timestamps, then ffilled.  Every timestamp past the last real candle
    inherited the final close, producing a perfectly flat series of zero
    returns for the whole unlabelable tail - which would have made the 180d
    "distribution" a spike at exactly zero.
    """
    close = hourly_close(1000)
    ladder = build_horizon_ladder({"horizons": ["1d", "180d"]})
    targets = build_future_returns(close, ladder)

    assert int(targets.frame["future_return_1d"].isna().sum()) == 24
    # 180d exceeds the 1000h series, so nothing is labelable at all.
    assert int(targets.frame["future_return_180d"].isna().sum()) == 1000
    assert targets.frame["future_return_180d"].notna().sum() == 0

    labelled = targets.frame["future_return_1d"].dropna()
    assert labelled.std() > 0, "tail rows must be NaN, not a constant zero series"


def test_missing_candles_use_asof_close() -> None:
    """A gap in the price series must not shift labels or invent rows."""
    index = pd.date_range("2022-01-01", periods=1000, freq="1h", tz="UTC")
    gapped = index.delete(index.slice_indexer("2022-01-10 05:00", "2022-01-10 09:00"))
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(len(gapped), 0.0005))), index=gapped)

    ladder = build_horizon_ladder({"horizons": ["1d"]})
    targets = build_future_returns(close, ladder)

    assert len(targets.frame) == len(gapped)
    assert targets.frame.index.equals(gapped)
    offsets = targets.realised_offset_hours["1d"].dropna().unique()
    assert list(offsets) == [24.0]


def test_realised_offsets_reported_per_horizon() -> None:
    close = hourly_close(6000)
    ladder = build_horizon_ladder({"horizons": ["1d", "7d", "30d"]})
    targets = build_future_returns(close, ladder)

    for label, hours in (("1d", 24.0), ("7d", 168.0), ("30d", 720.0)):
        observed = targets.realised_offset_hours[label].dropna().unique()
        assert list(observed) == [hours]


def test_coverage_table_counts() -> None:
    close = hourly_close(6000)
    ladder = build_horizon_ladder({"horizons": ["1d", "30d"]})
    targets = build_future_returns(close, ladder)
    coverage = targets.coverage_table().set_index("horizon")

    assert int(coverage.loc["1d", "n_total"]) == 6000
    assert int(coverage.loc["1d", "n_labelled"]) == 6000 - 24
    assert int(coverage.loc["30d", "n_labelled"]) == 6000 - 720


def test_overlaps_boundary_matches_purge_definition() -> None:
    index = pd.date_range("2022-01-01", periods=5000, freq="1h", tz="UTC")
    horizon = Horizon.from_label("30d", bars_per_unit=24)
    boundary = index[2000]

    overlapping = overlaps_boundary(index, horizon, boundary)
    # A row at t overlaps if t + H >= boundary, i.e. t >= boundary - H.
    expected = index[index + horizon.delta >= boundary]
    assert list(overlapping) == list(expected)


def test_gap_in_future_price_keeps_row_unlabelled() -> None:
    index = pd.date_range("2022-01-01", periods=6000, freq="1h", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.linspace(0, 5, len(index))), index=index)
    # Drop the last day so 1d labels must be NaN at the tail.
    close = close.iloc[:-24]

    ladder = build_horizon_ladder({"horizons": ["1d"]})
    targets = build_future_returns(close, ladder)
    assert int(targets.frame["future_return_1d"].isna().sum()) == 24


def test_targets_frame_columns_follow_horizon_order() -> None:
    """The ladder is sorted by duration, so column order is canonical."""
    close = hourly_close(6000)
    ladder = build_horizon_ladder({"horizons": ["30d", "1d", "7d"]})
    targets = build_future_returns(close, ladder)

    assert [h.label for h in ladder] == ["1d", "7d", "30d"]
    assert [c for c in targets.frame.columns if c.startswith("future_return_")] == [
        "future_return_1d",
        "future_return_7d",
        "future_return_30d",
    ]


def test_labelled_mask_selects_valid_rows() -> None:
    close = hourly_close(1000)
    ladder = build_horizon_ladder({"horizons": ["1d"]})
    targets: FutureReturnTargets = build_future_returns(close, ladder)

    mask = targets.labelled_mask("1d")
    assert int(mask.sum()) == 976
    assert targets.frame.index[mask].equals(targets.frame.index[:-24])
