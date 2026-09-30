"""Tests for the V3 strategy layer.

Everything is synthetic, in-memory and deterministic: a 3000-row UTC hourly
prediction frame on one model, no IO, no network, no pipeline.

The centre of gravity is :func:`test_overlapping_mode_overstates_by_the_horizon`.
A constant positive target makes the double-counting exactly computable - the
honest book earns ``(1+r) ** (n/H)`` and the naive one earns ``(1+r) ** n`` - so
the ratio of their log returns is the horizon in bars, to the digit.  That test
is the reason the module exists; everything else covers the ordinary edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v3.horizons import Horizon
from src.v3.strategy import (
    MODES,
    NON_OVERLAPPING,
    OVERLAPPING,
    OVERLAPPING_WARNING,
    RESULT_COLUMNS,
    CostModel,
    StrategyError,
    StrategyResult,
    bootstrap_sharpe,
    compare_to_baselines,
    horizon_label,
    non_overlapping_mask,
    positions_from_predictions,
    run_strategy,
    sweep_strategies,
)

N_ROWS = 3_000
HORIZON_HOURS = 24
MODEL = "xgboost"
HOURLY = Horizon.from_label("1d", bars_per_unit=24)


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=N_ROWS, freq="1h", tz="UTC", name="timestamp")


def _frame(
    index: pd.DatetimeIndex,
    seed: int,
    *,
    model: str = MODEL,
    signal: np.ndarray | None = None,
    target: np.ndarray | None = None,
    second_model: bool = False,
) -> pd.DataFrame:
    """A prediction frame matching ``HorizonResult.predictions``' schema.

    Every frame is built from its own seeded generator rather than a shared one,
    so no test's result depends on which tests happened to run before it.
    """
    generator = np.random.default_rng(seed)
    if signal is None:
        signal = generator.normal(0.0002, 0.01, len(index))
    if target is None:
        target = generator.normal(0.0005, 0.02, len(index))
    frame = pd.DataFrame(
        {
            "target": target,
            f"{model}_pred": signal,
            f"{model}_lo": np.asarray(signal) - 0.01,
            f"{model}_hi": np.asarray(signal) + 0.01,
            f"{model}_prob_empirical": np.clip(0.5 + np.asarray(signal) * 5.0, 0.0, 1.0),
            "prob_calibrated": np.clip(0.5 + np.asarray(signal) * 5.0, 0.0, 1.0),
        },
        index=index,
    )
    if second_model:
        other = generator.normal(0.0, 0.01, len(index))
        frame[f"{model}_2_pred"] = other
        frame[f"{model}_2_lo"] = other - 0.01
        frame[f"{model}_2_hi"] = other + 0.01
    return frame


@pytest.fixture(scope="module")
def predictions(index) -> pd.DataFrame:
    return _frame(index, 20240601)


@pytest.fixture(scope="module")
def flat_predictions(index) -> pd.DataFrame:
    """Every prediction negative: a long/flat book that never opens a position."""
    return _frame(index, 1, signal=np.full(N_ROWS, -0.01))


@pytest.fixture(scope="module")
def long_predictions(index) -> pd.DataFrame:
    """Every prediction positive and every target positive: a clean long book."""
    return _frame(index, 2, signal=np.full(N_ROWS, 0.01), target=np.full(N_ROWS, 0.01))


# --------------------------------------------------------------------------- positions

def test_threshold_is_strictly_greater_than_zero() -> None:
    pred = pd.Series([-0.02, -0.001, 0.0, 0.001, 0.02], index=pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"))

    position = positions_from_predictions(pred)

    # A prediction of exactly 0.0 is "no view", not a bullish one.
    assert list(position) == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert position.name == "position"
    assert position.index.equals(pred.index)


def test_a_positive_threshold_needs_a_bigger_prediction() -> None:
    pred = pd.Series([0.001, 0.01, 0.05], index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"))

    assert list(positions_from_predictions(pred, threshold=0.01)) == [0.0, 0.0, 1.0]
    assert list(positions_from_predictions(pred, threshold=0.0)) == [1.0, 1.0, 1.0]


def test_all_negative_predictions_never_open_a_position() -> None:
    pred = pd.Series(np.full(50, -0.3), index=pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC"))

    position = positions_from_predictions(pred)

    assert len(position) == 50
    assert position.sum() == 0.0


def test_all_positive_predictions_are_always_long() -> None:
    pred = pd.Series(np.full(50, 0.3), index=pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC"))

    position = positions_from_predictions(pred)

    assert (position == 1.0).all()


def test_non_finite_predictions_are_flat_not_missing() -> None:
    pred = pd.Series([np.nan, 0.01, -np.inf, 0.02], index=pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"))

    position = positions_from_predictions(pred)

    assert list(position) == [0.0, 1.0, 0.0, 1.0]


def test_probability_path_is_inclusive_at_the_threshold() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    frame = pd.DataFrame({"pred": [0.0, 0.0, 0.0, 0.0], "prob_calibrated": [0.49, 0.50, 0.51, np.nan]}, index=index)

    position = positions_from_predictions(frame, prob_col="prob_calibrated", prob_threshold=0.5)

    # 0.50 is a genuine calibrated statement, so the cut is inclusive here -
    # the opposite of the strictly-greater rule on a raw return prediction.
    assert list(position) == [0.0, 1.0, 1.0, 0.0]


def test_probability_path_ignores_the_return_column() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    frame = pd.DataFrame({"xgboost_pred": [-0.5, 0.5], "prob_calibrated": [0.9, 0.1]}, index=index)

    position = positions_from_predictions(frame, prob_col="prob_calibrated")

    assert list(position) == [1.0, 0.0]


def test_prob_col_on_a_series_is_refused_loudly() -> None:
    pred = pd.Series([0.1, 0.2], index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))

    with pytest.raises(StrategyError, match="no columns"):
        positions_from_predictions(pred, prob_col="prob_calibrated")


def test_a_missing_probability_column_names_what_is_available() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    frame = pd.DataFrame({"xgboost_pred": [0.1, 0.2]}, index=index)

    with pytest.raises(StrategyError, match="available"):
        positions_from_predictions(frame, prob_col="prob_calibrated")


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_an_impossible_probability_threshold_is_rejected(bad: float) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    frame = pd.DataFrame({"prob_calibrated": [0.5, 0.6]}, index=index)

    with pytest.raises(StrategyError, match=r"prob_threshold"):
        positions_from_predictions(frame, prob_col="prob_calibrated", prob_threshold=bad)


# --------------------------------------------------------------------------- mask

def test_mask_selects_exactly_ceil_n_over_k_entries(index) -> None:
    mask = non_overlapping_mask(index, HOURLY)

    assert mask.sum() == int(np.ceil(N_ROWS / HORIZON_HOURS))


def test_mask_is_the_positional_stride_on_a_regular_grid(index) -> None:
    mask = non_overlapping_mask(index, HOURLY)

    expected = index[::HORIZON_HOURS]
    assert list(index[mask.to_numpy()]) == list(expected)
    assert len(expected) == mask.sum()


def test_mask_keeps_one_full_horizon_between_entries(index) -> None:
    entries = index[non_overlapping_mask(index, HOURLY).to_numpy()]

    gaps = np.diff(entries.asi8)
    assert (gaps == HORIZON_HOURS * 3_600_000_000_000).all()


@pytest.mark.parametrize("horizon_hours", [1, 6, 24, 168])
def test_mask_size_tracks_the_horizon(index, horizon_hours: int) -> None:
    mask = non_overlapping_mask(index, pd.Timedelta(hours=horizon_hours))

    assert mask.sum() == int(np.ceil(N_ROWS / horizon_hours))


def test_mask_accepts_a_horizon_string_and_a_bare_hour_count(index) -> None:
    by_object = non_overlapping_mask(index, HOURLY)
    by_string = non_overlapping_mask(index, "1d")
    by_hours = non_overlapping_mask(index, 24)

    assert by_object.equals(by_string)
    assert by_object.equals(by_hours)


def test_mask_uses_wall_clock_not_a_configured_bar_count() -> None:
    """A stride would need ``bars_per_unit``, which the grid can contradict.

    A 1d ``Horizon`` built with ``bars_per_unit=24`` (the hourly default) applied
    to a 4h grid gives a stride of 24 rows = 96 hours, a quarter of the trades
    the horizon allows.  The wall-clock rule trades every day.
    """
    four_hourly = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")

    entries = four_hourly[non_overlapping_mask(four_hourly, HOURLY).to_numpy()]

    assert len(entries) == 7
    assert all((b - a) == pd.Timedelta(days=1) for a, b in zip(entries[:-1], entries[1:]))
    # What `Horizon.nominal_bars` would have produced: 40 / 24 rows.
    assert len(four_hourly[:: HOURLY.nominal_bars]) == 2


def test_mask_survives_a_missing_candle() -> None:
    """A hole in the grid must not leave two trades sharing part of a window."""
    hours = list(pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC"))
    gapped = pd.DatetimeIndex(hours[:6] + hours[8:])

    entries = gapped[non_overlapping_mask(gapped, pd.Timedelta(hours=6)).to_numpy()]

    assert all((b - a) >= pd.Timedelta(hours=6) for a, b in zip(entries[:-1], entries[1:]))
    # The hole pushes the 12:00 entry out, so only two trades fit in 11 hours.
    assert len(entries) == 2


def test_mask_on_an_empty_index_is_empty(index) -> None:
    empty = index[:0]

    assert non_overlapping_mask(empty, HOURLY).empty


def test_mask_rejects_a_non_datetime_index() -> None:
    with pytest.raises(StrategyError, match="DatetimeIndex"):
        non_overlapping_mask(pd.RangeIndex(10), HOURLY)


def test_mask_rejects_an_unparseable_horizon(index) -> None:
    with pytest.raises(Exception):
        non_overlapping_mask(index, "one month")


# --------------------------------------------------------------------------- run_strategy, non-overlapping

def test_non_overlapping_trades_exactly_the_masked_rows(long_predictions) -> None:
    result = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    assert result.mode == NON_OVERLAPPING
    assert result.realisable is True
    assert result.note == ""
    assert result.n_trades == int(np.ceil(N_ROWS / HORIZON_HOURS))
    assert result.n_trades == non_overlapping_mask(long_predictions.index, HOURLY).sum()
    assert result.exposure == 1.0


def test_hit_rate_is_a_probability_bracketed_by_its_wilson_bounds(predictions) -> None:
    result = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    assert 0.0 <= result.hit_rate <= 1.0
    assert result.hit_rate_ci_lower <= result.hit_rate <= result.hit_rate_ci_upper
    assert 0.0 <= result.hit_rate_ci_lower <= result.hit_rate_ci_upper <= 1.0
    assert result.hit_rate_ci_upper - result.hit_rate_ci_lower > 0.0


def test_a_list_of_probability_columns_is_refused_not_averaged(index) -> None:
    """Averaging two models' calibrated probabilities yields an uncalibrated number."""
    frame = _frame(index, 29)
    frame["prob_calibrated"] = 0.9
    frame["other"] = 0.1

    with pytest.raises(StrategyError, match="single column name"):
        run_strategy(frame, HOURLY, MODEL, prob_col=["prob_calibrated", "other"])

    # Naming one of them is the supported way, and it must still work.
    assert run_strategy(frame, HOURLY, MODEL, prob_col="prob_calibrated", prob_threshold=0.5).n_trades > 0


def test_a_sequence_prob_column_never_reaches_pandas_as_a_hash_key(index) -> None:
    """Regression: a list used to raise ``TypeError: unhashable type: 'list'``."""
    frame = _frame(index, 30)
    frame["prob_calibrated"] = 0.9

    for bad in (["prob_calibrated"], ("prob_calibrated",), {"prob_calibrated"}, np.array(["prob_calibrated"])):
        with pytest.raises(StrategyError, match="single column name"):
            positions_from_predictions(frame[["prob_calibrated"]], prob_col=bad)
        with pytest.raises(StrategyError, match="single column name"):
            run_strategy(frame, HOURLY, MODEL, prob_col=bad)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("1d", "1d"),
        (pd.Timedelta(days=1), "1d"),
        (24, "1d"),
        (24.0, "1d"),
        (pd.Timedelta(hours=12), "12h"),
        (pd.Timedelta(minutes=90), "90m"),
    ],
)
def test_horizon_labels_are_normalised_however_the_horizon_was_spelled(given, expected: str) -> None:
    """The label lands in a report column, so a Timedelta must not print as '1 days 00:00:00'."""
    assert horizon_label(given) == expected


def test_the_result_carries_a_normalised_label_whatever_the_caller_spelled(index) -> None:
    frame = _frame(index, 31)

    spelled_out = run_strategy(frame, "7d", MODEL)
    as_timedelta = run_strategy(frame, pd.Timedelta(days=7), MODEL)
    as_hours = run_strategy(frame, 168, MODEL)

    assert spelled_out.horizon == as_timedelta.horizon == as_hours.horizon == "7d"
    # The *scored* grid must be identical too, not just the label.
    assert np.allclose(spelled_out.equity.to_numpy(), as_timedelta.equity.to_numpy())
    assert np.allclose(spelled_out.equity.to_numpy(), as_hours.equity.to_numpy())
    assert spelled_out.n_trades == as_timedelta.n_trades == as_hours.n_trades


def test_a_duplicate_timestamp_is_refused_even_when_it_looks_harmless(predictions) -> None:
    """Two rows for one timestamp would let the same trade be counted twice."""
    doubled = pd.concat([predictions.iloc[:5], predictions.iloc[:5]])

    with pytest.raises(StrategyError, match="duplicate timestamps"):
        run_strategy(doubled, HOURLY, MODEL)


def test_an_unsorted_index_is_sorted_rather_than_traded_backwards(predictions) -> None:
    shuffled = predictions.sample(frac=1.0, random_state=7)

    assert run_strategy(shuffled, HOURLY, MODEL).equity.index.is_monotonic_increasing


def test_a_frame_shorter_than_the_horizon_still_scores_one_trade(index) -> None:
    """A 5-hour frame against a 30-day horizon has one window, not zero."""
    frame = _frame(index[:5], 32)

    result = run_strategy(frame, "30d", MODEL)

    assert result.n_trades == 1
    assert result.exposure == 1.0
    assert result.years == pytest.approx(30.0 / 365.25)


@pytest.mark.parametrize(
    ("equity", "expected"),
    [
        ([1.1, 1.2, 1.3], 0.0),            # never off its peak
        ([1.2, 1.1, 0.9, 1.0], -0.25),      # the worst point is 0.9 against a 1.2 peak
        ([0.9, 1.1], -0.1),                 # a loss on the very first trade still counts
        ([1.0, 1.0], 0.0),                  # a flat book had a drawdown of zero, and it is zero
        (list(np.cumprod(np.full(60, 0.5))), -1.0),  # wiped out
    ],
)
def test_max_drawdown_is_signed_and_its_magnitude_is_real(equity: list[float], expected: float) -> None:
    """``max_drawdown`` must be ``<= 0`` *and* carry the magnitude, not just clamp to 0."""
    from src.v3.strategy import _max_drawdown

    measured = _max_drawdown(np.asarray(equity, dtype="float64"))

    assert measured <= 0.0
    assert measured == pytest.approx(expected, abs=1e-9)


def test_max_drawdown_is_never_positive(predictions) -> None:
    for mode in MODES:
        result = run_strategy(predictions, HOURLY, MODEL, mode=mode)
        assert result.max_drawdown <= 0.0, mode


def test_a_strategy_with_a_real_drawdown_reports_it_and_gets_a_calmar(index) -> None:
    """A drawdown that is silently reported as 0.0 would make Calmar undefined."""
    target = np.concatenate([np.full(200, 0.05), np.full(200, -0.20), np.full(200, 0.05)])
    frame = _frame(index[:600], 26, signal=np.ones(600), target=target)

    result = run_strategy(frame, "1d", MODEL)

    assert result.max_drawdown < -0.1
    assert np.isfinite(result.calmar)
    assert result.calmar == pytest.approx(result.cagr / abs(result.max_drawdown))


def test_a_monotone_winner_has_a_zero_drawdown_and_no_calmar(long_predictions) -> None:
    result = run_strategy(long_predictions, HOURLY, MODEL)

    assert result.max_drawdown == 0.0
    assert np.isnan(result.calmar)
    assert result.cagr > 0.0


def test_cagr_is_positive_only_when_the_total_return_is(index) -> None:
    for shift in (0, 1):
        frame = _frame(
            index,
            99 + shift,
            signal=np.full(N_ROWS, -0.01) if shift else None,
        )
        result = run_strategy(frame, HOURLY, MODEL, mode=NON_OVERLAPPING)
        if result.n_trades == 0:
            # No book, no CAGR - and certainly not a flat 0.0.
            assert np.isnan(result.total_return) and np.isnan(result.cagr)
        elif result.total_return > 0.0:
            assert result.cagr > 0.0
        elif result.total_return < 0.0:
            assert result.cagr < 0.0
        else:
            assert result.cagr == pytest.approx(0.0)


def test_years_is_the_wall_clock_the_book_was_committed(predictions) -> None:
    result = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    assert result.years == pytest.approx(1.0 * result.n_trades / 365.25)


def test_sharpe_is_annualised_by_the_horizon_not_the_hour(index) -> None:
    """The factor is exactly ``sqrt(periods_per_year)`` for the horizon.

    A naive hourly annualisation would multiply by ``sqrt(8760)``.  The signal
    here is long at every rebalance, so the book pays one opening fill and never
    closes: the per-period return series is the raw target with the fee taken off
    the first entry, which the test can reconstruct without reimplementing
    anything.
    """
    generator = np.random.default_rng(28)
    frame = _frame(
        index,
        27,
        signal=np.full(N_ROWS, 0.01),
        target=generator.normal(0.0005, 0.02, N_ROWS),
    )
    opening_fill = CostModel().one_way_bps() / 10_000.0

    for label, days in (("1d", 1.0), ("3d", 3.0), ("7d", 7.0)):
        result = run_strategy(frame, label, MODEL)
        grid = non_overlapping_mask(frame.index, label).to_numpy()
        net = frame.loc[grid, "target"].to_numpy(dtype="float64").copy()
        net[0] -= opening_fill
        per_period = float(np.mean(net) / np.std(net, ddof=1))

        assert result.sharpe == pytest.approx(per_period * np.sqrt(365.25 / days), rel=1e-9)
        # Not the hourly factor, which would be sqrt(8760) instead.
        assert result.sharpe != pytest.approx(per_period * np.sqrt(8760.0), rel=1e-6)


def test_overlapping_mode_uses_the_same_horizon_factor(long_predictions) -> None:
    """Both modes annualise identically, so the two rows are comparable."""
    horizon = "3d"
    honest = run_strategy(long_predictions, horizon, MODEL, mode=NON_OVERLAPPING)
    naive = run_strategy(long_predictions, horizon, MODEL, mode=OVERLAPPING)

    assert np.isfinite(honest.sharpe) and np.isfinite(naive.sharpe)
    # years is the one thing that legitimately differs between the modes.
    assert honest.years == pytest.approx(3.0 * honest.n_trades / 365.25)
    assert naive.years == pytest.approx((N_ROWS - 1) / 24.0 / 365.25)


def test_equity_curve_is_non_overlapping_and_starts_at_one(long_predictions) -> None:
    result = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    assert len(result.equity) == result.n_trades
    assert result.equity.index[0] == long_predictions.index[0]
    assert result.equity.iloc[0] == pytest.approx(1.01 - CostModel().one_way_bps() / 10_000.0)
    assert result.equity.is_monotonic_increasing


def test_a_continuously_held_position_pays_one_fill_for_the_whole_run(long_predictions) -> None:
    """``1 -> 1`` must be free, however many rebalances it spans."""
    result = run_strategy(long_predictions, HOURLY, MODEL, cost=CostModel(fee_bps=20.0, slippage_bps=10.0))

    assert result.n_trades == 125
    assert result.turnover == 1.0
    assert result.cost_drag == pytest.approx(30.0 / 10_000.0)


def test_the_short_leg_turns_the_book_over_much_more(index) -> None:
    """A centred signal engages the short side, so the book flips constantly."""
    frame = _frame(index, 20, signal=np.random.default_rng(21).normal(0.0, 0.01, N_ROWS))

    long_only = run_strategy(frame, HOURLY, MODEL, allow_short=False)
    both = run_strategy(frame, HOURLY, MODEL, allow_short=True)

    # Every one of the 125 rebalances is now a position rather than a coin flip.
    assert both.n_trades > long_only.n_trades
    assert both.turnover > long_only.turnover
    assert both.cost_drag > long_only.cost_drag

    # `n_trades` counts invested *periods* - in non-overlapping mode every one of
    # them is a complete hold, so 63 of them.  `turnover` counts actual fills,
    # which is fewer: 30 entries and 30 exits, because the book holds through the
    # gaps between them for free.
    grid = non_overlapping_mask(frame.index, HOURLY).to_numpy()
    long_series = positions_from_predictions(frame.loc[grid, f"{MODEL}_pred"]).to_numpy(dtype="float64")
    pairs = list(zip([0.0] + list(long_series), long_series))
    entries = sum(1 for previous, current in pairs if current and not previous)
    exits = sum(1 for previous, current in pairs if previous and not current)
    assert long_only.n_trades == int((long_series != 0.0).sum())
    assert entries < long_only.n_trades
    assert long_only.turnover == pytest.approx(float(entries + exits))
    assert long_series.min() == 0.0


def test_a_short_book_earns_on_a_falling_market(index) -> None:
    negative = _frame(
        index,
        np.random.default_rng(7),
        signal=np.full(N_ROWS, -0.02),
        target=np.full(N_ROWS, -0.01),
    )

    result = run_strategy(negative, HOURLY, MODEL, allow_short=True)

    assert result.n_trades == int(np.ceil(N_ROWS / HORIZON_HOURS))
    assert result.exposure == 1.0
    assert result.total_return > 0.0
    assert result.max_drawdown == 0.0


def test_long_flat_refuses_to_short_the_same_book(index) -> None:
    negative = _frame(
        index,
        np.random.default_rng(7),
        signal=np.full(N_ROWS, -0.02),
        target=np.full(N_ROWS, -0.01),
    )

    result = run_strategy(negative, HOURLY, MODEL, allow_short=False)

    assert result.n_trades == 0
    assert np.isnan(result.total_return)


def test_unknown_mode_is_refused(predictions) -> None:
    with pytest.raises(StrategyError, match="mode must be one of"):
        run_strategy(predictions, HOURLY, MODEL, mode="hourly")


def test_a_missing_model_column_lists_what_exists(predictions) -> None:
    with pytest.raises(StrategyError, match=r"lightgbm.*available"):
        run_strategy(predictions, HOURLY, "lightgbm")


def test_duplicate_timestamps_are_refused(predictions) -> None:
    doubled = pd.concat([predictions, predictions.iloc[:10]])

    with pytest.raises(StrategyError, match="duplicate timestamps"):
        run_strategy(doubled, HOURLY, MODEL)


# --------------------------------------------------------------------------- THE overstatement test

def test_overlapping_mode_overstates_by_the_horizon(long_predictions) -> None:
    """The anti-pattern, pinned with an exactly computable ratio.

    Every target is the same positive number and the book is always long, so the
    honest compounded return is ``(1 + r) ** (n / k)`` and the naive one is
    ``(1 + r) ** n``.  Their **log** returns therefore differ by exactly the
    horizon in bars - 24 here - whatever the cost assumption, because the cost
    enters both as the same additive subtraction from each period's gross.
    """
    cost = CostModel(fee_bps=0.0, slippage_bps=0.0)
    honest = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING, cost=cost)
    naive = run_strategy(long_predictions, HOURLY, MODEL, mode=OVERLAPPING, cost=cost)

    assert naive.n_trades == N_ROWS
    assert honest.n_trades == int(np.ceil(N_ROWS / HORIZON_HOURS))

    ratio = np.log1p(naive.total_return) / np.log1p(honest.total_return)
    assert ratio == pytest.approx(HORIZON_HOURS, rel=0.02)

    # In levels the gap is not a factor of 24, it is a factor of 24 *compounded*
    # 125 times over - the same 30 days of P&L counted 3000 times.
    assert naive.total_return > 100.0 * honest.total_return
    assert naive.total_return > 1e6
    assert honest.total_return == pytest.approx(1.01**125 - 1.0, rel=1e-9)


def test_overlapping_mode_is_flagged_as_not_realisable(long_predictions) -> None:
    honest = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)
    naive = run_strategy(long_predictions, HOURLY, MODEL, mode=OVERLAPPING)

    assert honest.realisable is True and honest.note == ""
    assert naive.realisable is False
    assert naive.note == OVERLAPPING_WARNING
    assert "NOT REALISABLE" in naive.note

    flattened = naive.to_dict()
    assert flattened["realisable"] == 0
    assert flattened["note"] == OVERLAPPING_WARNING
    assert flattened["mode"] == OVERLAPPING


def test_overlapping_equity_is_as_long_as_the_sample(long_predictions) -> None:
    honest = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)
    naive = run_strategy(long_predictions, HOURLY, MODEL, mode=OVERLAPPING)

    assert len(naive.equity) == N_ROWS
    assert len(honest.equity) == int(np.ceil(N_ROWS / HORIZON_HOURS))


def test_the_overstatement_survives_default_costs(long_predictions) -> None:
    honest = run_strategy(long_predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)
    naive = run_strategy(long_predictions, HOURLY, MODEL, mode=OVERLAPPING)

    ratio = np.log1p(naive.total_return) / np.log1p(honest.total_return)
    assert ratio == pytest.approx(HORIZON_HOURS, rel=0.05)


def test_a_longer_horizon_overstates_more(index) -> None:
    """The inflation factor is the horizon in bars, so it grows with the horizon."""
    frame = _frame(index, 3, signal=np.full(N_ROWS, 0.01), target=np.full(N_ROWS, 0.01))
    ratios = {}
    for label in ("1d", "3d", "7d"):
        horizon = Horizon.from_label(label, bars_per_unit=24)
        honest = run_strategy(frame, horizon, MODEL, mode=NON_OVERLAPPING, cost=CostModel(0.0, 0.0))
        naive = run_strategy(frame, horizon, MODEL, mode=OVERLAPPING, cost=CostModel(0.0, 0.0))
        assert naive.total_return > honest.total_return
        ratios[label] = np.log1p(naive.total_return) / np.log1p(honest.total_return)

    # 24 / 72 / 168 bars, exact for 1d and within one entry's rounding for the
    # others (3000 rows is not a whole number of 3d or 7d windows).
    assert ratios["1d"] == pytest.approx(24, rel=0.02)
    assert ratios["3d"] == pytest.approx(72, rel=0.02)
    assert ratios["7d"] == pytest.approx(168, rel=0.02)
    assert ratios["1d"] < ratios["3d"] < ratios["7d"]


# --------------------------------------------------------------------------- costs

def test_doubling_the_fee_strictly_reduces_the_total_return(long_predictions) -> None:
    cheap = run_strategy(long_predictions, HOURLY, MODEL, cost=CostModel(fee_bps=2.5, slippage_bps=1.0))
    dear = run_strategy(long_predictions, HOURLY, MODEL, cost=CostModel(fee_bps=5.0, slippage_bps=2.0))

    assert dear.total_return < cheap.total_return
    assert dear.cost_drag > cheap.cost_drag
    assert dear.n_trades == cheap.n_trades


def test_holding_a_position_costs_nothing_after_the_entry(long_predictions) -> None:
    """``1 -> 1`` must be free: 125 consecutive long trades cost one round trip."""
    cost = CostModel(fee_bps=10.0, slippage_bps=5.0)
    result = run_strategy(long_predictions, HOURLY, MODEL, cost=cost)

    assert result.turnover == 1.0
    assert result.cost_drag == pytest.approx(cost.one_way_bps() / 10_000.0)
    assert result.cost_drag < 2.0 * cost.round_trip_bps() / 10_000.0


def test_a_single_trade_and_a_hundred_trades_cost_the_same_one_way_fill(index) -> None:
    cost = CostModel(fee_bps=8.0, slippage_bps=3.0)
    one = _frame(index, 4, signal=np.full(N_ROWS, 0.01), target=np.full(N_ROWS, 0.01))
    long_horizon = run_strategy(one, "1d", MODEL, cost=cost)

    # Every row long, but with a horizon longer than the sample, so exactly one
    # trade is ever opened and then held.
    whole_sample = run_strategy(one, Horizon.from_label("180d", bars_per_unit=24), MODEL, cost=cost)

    assert whole_sample.n_trades == 1
    assert whole_sample.turnover == 1.0
    assert whole_sample.cost_drag == pytest.approx(cost.one_way_bps() / 10_000.0)
    assert long_horizon.cost_drag == pytest.approx(cost.one_way_bps() / 10_000.0)


def test_a_strategy_that_never_trades_pays_exactly_nothing(flat_predictions) -> None:
    result = run_strategy(flat_predictions, HOURLY, MODEL, cost=CostModel(fee_bps=50.0, slippage_bps=50.0))

    assert result.n_trades == 0
    assert result.turnover == 0.0
    assert result.cost_drag == 0.0


def test_round_trip_and_one_way_costs_are_what_they_claim() -> None:
    cost = CostModel(fee_bps=5.0, slippage_bps=2.0)

    assert cost.one_way_bps() == 7.0
    assert cost.round_trip_bps() == 14.0
    assert cost.to_dict() == {
        "fee_bps": 5.0,
        "slippage_bps": 2.0,
        "one_way_bps": 7.0,
        "round_trip_bps": 14.0,
    }


def test_a_closed_trade_costs_exactly_one_round_trip(index) -> None:
    """Entry and exit together must cost exactly ``round_trip_bps()`` and no more."""
    cost = CostModel(fee_bps=5.0, slippage_bps=2.0)
    signal = np.where(np.arange(N_ROWS) < HORIZON_HOURS, 0.01, -0.01)
    frame = _frame(index, 5, signal=signal, target=np.full(N_ROWS, 0.001))

    result = run_strategy(frame, "1d", MODEL, cost=cost)

    # One long entry, then flat at every later rebalance, so the book is a
    # single 24h round trip and nothing else.
    assert result.n_trades == 1
    assert result.turnover == pytest.approx(2.0)
    assert result.cost_drag == pytest.approx(cost.round_trip_bps() / 10_000.0)


def test_negative_costs_are_refused() -> None:
    with pytest.raises(StrategyError, match="fee_bps"):
        CostModel(fee_bps=-1.0)


# --------------------------------------------------------------------------- degenerate inputs

def test_zero_trades_gives_nan_metrics_and_no_exception(flat_predictions) -> None:
    result = run_strategy(flat_predictions, HOURLY, MODEL)

    assert result.n_trades == 0
    for name in (
        "total_return", "cagr", "sharpe", "sortino", "calmar",
        "hit_rate", "hit_rate_ci_lower", "hit_rate_ci_upper",
    ):
        assert np.isnan(getattr(result, name)), name
    # These three are genuinely zero rather than unknown.
    assert result.exposure == 0.0
    assert result.turnover == 0.0
    assert result.cost_drag == 0.0
    assert result.max_drawdown == 0.0


def test_zero_variance_returns_gives_a_nan_sharpe_and_no_exception(long_predictions) -> None:
    """A constant target has no dispersion, so there is no Sharpe to quote."""
    result = run_strategy(long_predictions, HOURLY, MODEL, cost=CostModel(0.0, 0.0))

    per_period = result.equity.pct_change().dropna()
    assert per_period.nunique() == 1
    assert per_period.iloc[0] == pytest.approx(0.01)
    assert per_period.std(ddof=1) == 0.0
    assert np.isnan(result.sharpe)
    assert np.isnan(result.sortino)
    assert np.isnan(result.calmar)
    assert result.hit_rate == 1.0
    assert result.total_return == pytest.approx(1.01**125 - 1.0, rel=1e-9)


def test_a_single_usable_row_is_not_an_error(index) -> None:
    frame = _frame(index, 6).iloc[:24].copy()
    frame["target"] = np.nan
    frame.iloc[0, frame.columns.get_loc("target")] = 0.02
    frame.iloc[0, frame.columns.get_loc(f"{MODEL}_pred")] = 0.05

    result = run_strategy(frame, HOURLY, MODEL)

    assert result.n_trades == 1
    assert result.hit_rate in (0.0, 1.0)
    assert result.hit_rate_ci_lower <= result.hit_rate <= result.hit_rate_ci_upper
    # One observation has no dispersion, so there is no risk-adjusted figure.
    assert np.isnan(result.sharpe)
    assert np.isnan(result.sortino)
    assert result.max_drawdown <= 0.0


def test_an_all_nan_target_column_raises_rather_than_inventing_a_return(predictions) -> None:
    broken = predictions.copy()
    broken["target"] = np.nan

    result = run_strategy(broken, HOURLY, MODEL)

    assert result.n_trades == 0
    assert np.isnan(result.total_return)


def test_an_unlabelled_tail_leaves_the_equity_curve_short_of_the_grid(index) -> None:
    """The last H rows have no realised window yet, so they cannot be scored."""
    tail = HORIZON_HOURS * 5
    truncated = _frame(index, 22).iloc[: N_ROWS - tail].copy()

    result = run_strategy(truncated, HOURLY, MODEL)

    assert len(result.equity) == int(np.ceil((N_ROWS - tail) / HORIZON_HOURS))
    assert result.equity.index[-1] <= truncated.index[-1]


def test_a_missing_prediction_is_a_flat_decision_and_still_costs(index) -> None:
    """A NaN signal closes the book; dropping the row would hide that close."""
    frame = _frame(index, 23, signal=np.full(N_ROWS, 0.01))
    frame.iloc[HORIZON_HOURS * 10, frame.columns.get_loc(f"{MODEL}_pred")] = np.nan

    result = run_strategy(frame, HOURLY, MODEL)

    # One entry, one close at the missing prediction, one re-entry after it.
    assert result.n_trades == int(np.ceil(N_ROWS / HORIZON_HOURS)) - 1
    assert result.turnover == 3.0
    assert result.cost_drag == pytest.approx(3.0 * CostModel().one_way_bps() / 10_000.0)


def test_a_missing_probability_column_is_refused_by_run_strategy(predictions) -> None:
    with pytest.raises(StrategyError, match="prob_col"):
        run_strategy(predictions, HOURLY, MODEL, prob_col="nope")


def test_run_strategy_can_trade_the_calibrated_probability(index) -> None:
    """The probability path must not fall back to ``prob > 0`` on the long side."""
    frame = _frame(index, 24)
    # Every probability is above zero, so a ``prob > threshold`` bug would be
    # long 125 times out of 125.
    frame["prob_calibrated"] = np.where(np.arange(N_ROWS) < HORIZON_HOURS * 10, 0.95, 0.05)
    grid = int(np.ceil(N_ROWS / HORIZON_HOURS))

    result = run_strategy(frame, HOURLY, MODEL, prob_col="prob_calibrated", prob_threshold=0.6)

    assert result.n_trades == 10
    assert result.turnover == 2.0
    assert result.exposure == pytest.approx(10 / grid)
    # The return prediction would have said something completely different.
    on_prediction = run_strategy(frame, HOURLY, MODEL)
    assert on_prediction.n_trades != result.n_trades


def test_the_short_side_of_the_probability_path_is_symmetric(index) -> None:
    frame = _frame(index, 25)
    frame["prob_calibrated"] = np.where(np.arange(N_ROWS) < HORIZON_HOURS * 10, 0.95, 0.05)

    result = run_strategy(frame, HOURLY, MODEL, prob_col="prob_calibrated", prob_threshold=0.6, allow_short=True)

    # Long for the first ten rebalances, short for the remaining 115, so the
    # book is never flat - and crosses exactly once, from +1 to -1.
    assert result.n_trades == int(np.ceil(N_ROWS / HORIZON_HOURS))
    assert result.exposure == 1.0
    assert result.turnover == 3.0  # one open, plus a close and an open to flip


# --------------------------------------------------------------------------- sweep

def test_sweep_returns_one_row_per_model_and_mode(index) -> None:
    frame = _frame(index, 8, second_model=True)
    models = [MODEL, f"{MODEL}_2"]

    table = sweep_strategies(frame, HOURLY, models)

    assert len(table) == len(models) * len(MODES)
    assert set(zip(table["model"], table["mode"])) == {(m, mode) for m in models for mode in MODES}


def test_sweep_flattens_every_result_field(index) -> None:
    table = sweep_strategies(_frame(index, 9), HOURLY, [MODEL])

    assert len(table) == len(MODES)
    assert "equity" not in table.columns
    # Every result column made it into the table, and nothing else did.
    assert list(table.columns) == list(RESULT_COLUMNS)


def test_sweep_marks_the_overlapping_rows_and_leaves_the_others_alone(index) -> None:
    frame = _frame(index, 10, signal=np.full(N_ROWS, 0.01), target=np.full(N_ROWS, 0.01))

    table = sweep_strategies(frame, HOURLY, [MODEL])

    honest = table[table["mode"] == NON_OVERLAPPING].iloc[0]
    naive = table[table["mode"] == OVERLAPPING].iloc[0]

    assert honest["realisable"] == 1
    assert honest["note"] == ""
    assert naive["realisable"] == 0
    assert naive["note"] == OVERLAPPING_WARNING
    assert naive["total_return"] > honest["total_return"]
    assert np.log1p(naive["total_return"]) / np.log1p(honest["total_return"]) == pytest.approx(
        HORIZON_HOURS, rel=0.02
    )


def test_sweep_forwards_kwargs_and_can_be_restricted_to_one_mode(index) -> None:
    table = sweep_strategies(
        _frame(index, 11), HOURLY, [MODEL], modes=[NON_OVERLAPPING], threshold=0.05
    )

    assert list(table["mode"]) == [NON_OVERLAPPING]


def test_an_empty_model_list_still_returns_the_columns(index) -> None:
    table = sweep_strategies(_frame(index, 12), HOURLY, [])

    assert table.empty
    assert "total_return" in table.columns


def test_sweep_rejects_an_unknown_mode(index) -> None:
    with pytest.raises(StrategyError, match="unknown mode"):
        sweep_strategies(_frame(index, 13), HOURLY, [MODEL], modes=["daily"])


# --------------------------------------------------------------------------- baselines

def test_compare_to_baselines_returns_the_three_documented_rows(predictions) -> None:
    result = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    table = compare_to_baselines(result, predictions, HOURLY, MODEL)

    assert list(table.index) == ["model", "always_long", "buy_and_hold"]
    for column in ("total_return", "sharpe", "max_drawdown", "hit_rate"):
        assert column in table.columns
        assert len(table[column]) == 3


def test_the_baselines_agree_with_each_other(predictions) -> None:
    """A continuous long book and the price path are the same trade."""
    result = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    table = compare_to_baselines(result, predictions, HOURLY, MODEL)

    assert table.loc["always_long", "total_return"] == pytest.approx(table.loc["buy_and_hold", "total_return"])
    assert table.loc["always_long", "n_trades"] == int(np.ceil(N_ROWS / HORIZON_HOURS))


def test_buy_and_hold_is_not_double_counted(predictions) -> None:
    """The baseline must compound the price path once, not once per row."""
    cost = CostModel(fee_bps=5.0, slippage_bps=2.0)
    result = run_strategy(predictions, HOURLY, MODEL, cost=cost)

    table = compare_to_baselines(result, predictions, HOURLY, MODEL, cost=cost)

    mask = non_overlapping_mask(predictions.index, HOURLY).to_numpy()
    windowed = predictions.loc[mask, "target"].to_numpy()
    # One opening fill on the first window, then the price path itself.
    expected = (1.0 + windowed[0] - cost.one_way_bps() / 10_000.0) * np.prod(1.0 + windowed[1:]) - 1.0
    naive_per_row = float(np.prod(1.0 + predictions["target"].to_numpy()) - 1.0)

    assert table.loc["buy_and_hold", "total_return"] == pytest.approx(expected, rel=1e-9)
    assert table.loc["buy_and_hold", "total_return"] != pytest.approx(naive_per_row, rel=1e-6)
    # 3000 hourly windows would cover the same 125 days 24 times over.
    assert len(windowed) == int(np.ceil(N_ROWS / HORIZON_HOURS))


def test_the_model_row_carries_the_result_and_its_warning(predictions) -> None:
    honest = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)
    naive = run_strategy(predictions, HOURLY, MODEL, mode=OVERLAPPING)

    honest_table = compare_to_baselines(honest, predictions, HOURLY, MODEL)
    naive_table = compare_to_baselines(naive, predictions, HOURLY, MODEL)

    assert honest_table.loc["model", "total_return"] == pytest.approx(honest.total_return)
    assert honest_table.loc["model", "realisable"] == 1
    # The baselines stay on the non-overlapping grid even when the model row is
    # not, so the table can never endorse an overlapping total.
    assert naive_table.loc["model", "realisable"] == 0
    assert naive_table.loc["model", "note"] == OVERLAPPING_WARNING
    assert naive_table.loc["buy_and_hold", "total_return"] == pytest.approx(
        honest_table.loc["buy_and_hold", "total_return"]
    )


def test_baselines_refuse_a_non_result(predictions) -> None:
    with pytest.raises(StrategyError, match="StrategyResult"):
        compare_to_baselines({"total_return": 1.0}, predictions, HOURLY, MODEL)


# --------------------------------------------------------------------------- bootstrap

def test_bootstrap_sharpe_returns_the_four_documented_keys() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.001, 0.02, 1_500)

    result = bootstrap_sharpe(returns, n_bootstrap=300, seed=3)

    assert set(result) == {"sharpe", "ci_lower", "ci_upper", "p_sharpe_gt_zero"}
    assert all(isinstance(v, float) for v in result.values())


def test_the_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0015, 0.02, 2_000)

    result = bootstrap_sharpe(returns, n_bootstrap=400, seed=7)

    assert result["ci_lower"] < result["sharpe"] < result["ci_upper"]
    assert 0.0 <= result["p_sharpe_gt_zero"] <= 1.0
    # A clearly positive mean return should be positive in most resamples.
    assert result["p_sharpe_gt_zero"] > 0.9


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    returns = np.random.default_rng(13).normal(0.001, 0.02, 800)

    first = bootstrap_sharpe(returns, n_bootstrap=200, seed=5)
    second = bootstrap_sharpe(returns, n_bootstrap=200, seed=5)
    third = bootstrap_sharpe(returns, n_bootstrap=200, seed=6)

    assert first == second
    assert first["sharpe"] == third["sharpe"]
    assert first["ci_lower"] != third["ci_lower"]


def test_bootstrap_of_a_negative_series_finds_a_negative_sharpe() -> None:
    returns = np.random.default_rng(14).normal(-0.002, 0.02, 1_200)

    result = bootstrap_sharpe(returns, n_bootstrap=300, seed=2)

    assert result["sharpe"] < 0.0
    assert result["p_sharpe_gt_zero"] < 0.1


def test_resampling_single_observations_would_manufacture_precision() -> None:
    """The point of blocks: an iid resample of a dependent series is too narrow.

    An AR(1) with ``rho = 0.95`` has a mean whose true sampling distribution is
    roughly 39x wider in variance than the iid one assumes.  Resampling rows
    individually would report a fraction of the real uncertainty, and it would
    look rigorous while doing it.
    """
    generator = np.random.default_rng(15)
    rho, size = 0.95, 2_000
    innovations = np.zeros(size)
    for step in range(1, size):
        innovations[step] = rho * innovations[step - 1] + generator.normal(0.0, np.sqrt(1.0 - rho**2))
    returns = innovations / np.std(innovations) * 0.02 + 0.0005

    iid = bootstrap_sharpe(returns, n_bootstrap=400, seed=4, block_size=1)
    blocked = bootstrap_sharpe(returns, n_bootstrap=400, seed=4, block_size=100)

    iid_width = iid["ci_upper"] - iid["ci_lower"]
    blocked_width = blocked["ci_upper"] - blocked["ci_lower"]
    assert blocked_width > 2.0 * iid_width
    # The point estimate does not depend on the resampling scheme.
    assert iid["sharpe"] == pytest.approx(blocked["sharpe"])


@pytest.mark.parametrize("bad", [0, 1, 99])
def test_bootstrap_guards_its_resample_count(bad: int) -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be >= 100"):
        bootstrap_sharpe(np.random.default_rng(16).normal(0.0, 0.01, 200), n_bootstrap=bad)


def test_bootstrap_accepts_a_series() -> None:
    series = pd.Series(np.random.default_rng(17).normal(0.001, 0.02, 500))

    result = bootstrap_sharpe(series, n_bootstrap=200, seed=1)

    assert np.isfinite(result["sharpe"])


def test_bootstrap_of_a_constant_series_is_nan_not_zero() -> None:
    result = bootstrap_sharpe(np.full(500, 0.01), n_bootstrap=200, seed=1)

    assert all(np.isnan(v) for v in result.values())


def test_bootstrap_of_an_empty_series_is_nan_not_an_exception() -> None:
    result = bootstrap_sharpe(np.array([]), n_bootstrap=200, seed=1)

    assert all(np.isnan(v) for v in result.values())


def test_bootstrap_ignores_non_finite_entries() -> None:
    clean = np.random.default_rng(18).normal(0.001, 0.02, 600)
    dirty = clean.copy()
    dirty[::10] = np.nan

    # The same 540 observations, in the same order, just reached two ways.
    assert bootstrap_sharpe(dirty, n_bootstrap=200, seed=1) == bootstrap_sharpe(
        np.delete(clean, np.arange(0, 600, 10)), n_bootstrap=200, seed=1
    )
    # And it is not the same as scoring all 600.
    assert bootstrap_sharpe(dirty, n_bootstrap=200, seed=1)["sharpe"] != pytest.approx(
        bootstrap_sharpe(clean, n_bootstrap=200, seed=1)["sharpe"]
    )


# --------------------------------------------------------------------------- result plumbing

def test_result_to_dict_is_flat_and_numeric(predictions) -> None:
    result = run_strategy(predictions, HOURLY, MODEL, mode=NON_OVERLAPPING)

    payload = result.to_dict()

    assert "equity" not in payload
    assert isinstance(payload["horizon"], str)
    assert isinstance(payload["model"], str)
    assert isinstance(payload["n_trades"], int)
    assert isinstance(payload["realisable"], int)
    assert set(payload) == set(result.to_dict())
    assert len(payload) == len(RESULT_COLUMNS) == 19


def test_strategy_result_is_frozen(long_predictions) -> None:
    result = run_strategy(long_predictions, HOURLY, MODEL)

    with pytest.raises(Exception):
        result.total_return = 0.0  # type: ignore[misc]


def test_cost_model_is_frozen() -> None:
    cost = CostModel()

    with pytest.raises(Exception):
        cost.fee_bps = 1.0  # type: ignore[misc]
