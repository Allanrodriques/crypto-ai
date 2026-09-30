"""Backtest tests: execution timing, cost accounting, and non-overlap.

The property that matters most here is that a signal computed from candle ``t``
is filled at the *open of candle t+1*.  Filling at the close of ``t`` would let
the simulation trade on information it could not have had when the signal
produced.  Several tests below pin that timing down against hand-computed
arithmetic rather than against the implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    BacktestConfigError,
    BacktestRules,
    run_backtest,
    threshold_sensitivity,
)

HORIZON = 6
COST_BPS = 10.0
COST = COST_BPS / 10_000.0


def make_prices(
    opens: list[float],
    closes: list[float],
    probabilities: list[float],
    *,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a minimal backtest frame with an hourly UTC index."""
    assert len(opens) == len(closes) == len(probabilities)
    index = pd.date_range(start, periods=len(opens), freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": np.asarray(opens, dtype="float64"),
            "close": np.asarray(closes, dtype="float64"),
            "probability_up": np.asarray(probabilities, dtype="float64"),
            "target": np.zeros(len(opens), dtype="int8"),
        },
        index=index,
    )


def rules(**overrides) -> BacktestRules:
    base = dict(
        initial_capital=10_000.0,
        probability_threshold=0.60,
        transaction_cost_bps=COST_BPS,
        horizon_candles=HORIZON,
        periods_per_year=24 * 365,
    )
    base.update(overrides)
    return BacktestRules(**base).validate()


@pytest.fixture
def one_trade() -> pd.DataFrame:
    """12 candles, one signal at bar 1, flat price until the exit.

    Signal at bar 1 -> entry at the open of bar 2 (101.0) -> exit at the close of
    bar 7 (the 6th held candle).
    """
    opens = [100.0] * 12
    closes = [100.0] * 12
    probabilities = [0.0] * 12
    opens[2] = 101.0          # entry fill
    closes[7] = 110.0         # exit fill
    probabilities[1] = 0.90   # the only signal
    return make_prices(opens, closes, probabilities)


# --------------------------------------------------------------------------- timing

def test_entry_is_the_open_after_the_signal_bar(one_trade):
    result = run_backtest(one_trade, rules())
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["signal_time"] == one_trade.index[1]
    assert trade["entry_time"] == one_trade.index[2], "must fill at the NEXT bar's open"
    assert trade["entry_price"] == pytest.approx(101.0)


def test_a_signal_never_fills_at_its_own_close(one_trade):
    """Regression guard: filling at close[t] is the classic look-ahead bug."""
    result = run_backtest(one_trade, rules())
    trade = result.trades.iloc[0]
    # close of the signal bar is 100.0; the entry must not be that price.
    assert trade["entry_price"] != pytest.approx(one_trade["close"].iloc[1])
    assert trade["entry_time"] > trade["signal_time"]


def test_exit_is_the_close_after_holding_the_horizon(one_trade):
    result = run_backtest(one_trade, rules())
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == one_trade.index[2]
    assert trade["exit_time"] == one_trade.index[7]
    assert trade["hold_candles"] == HORIZON
    assert trade["exit_price"] == pytest.approx(110.0)


def test_trades_never_overlap():
    """A signal inside an open position must be ignored, not stacked."""
    opens = [100.0] * 40
    closes = [100.0] * 40
    probabilities = [0.0] * 40
    opens[2] = 101.0
    # Bars 1, 3, 4, 5 all signal.  Only bar 1 can open a position, because the
    # position is still open until bar 7.
    for i in (1, 3, 4, 5):
        probabilities[i] = 0.95
    result = run_backtest(make_prices(opens, closes, probabilities), rules())
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["signal_time"] == probabilities_frame_time(opens, probabilities)


def probabilities_frame_time(opens, probabilities) -> pd.Timestamp:
    index = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return index[int(np.argmax(np.asarray(probabilities) >= 0.60))]


def test_no_trade_when_the_signal_is_the_final_bar():
    frame = make_prices([100.0] * 10, [100.0] * 10, [0.0] * 9 + [0.99])
    result = run_backtest(frame, rules())
    assert len(result.trades) == 0
    assert result.equity_curve.notna().all().all(), "equity must still be defined while flat"


def test_signals_below_the_threshold_are_ignored():
    frame = make_prices(
        [100.0] * 12, [100.0] * 12, [0.0, 0.59, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert len(run_backtest(frame, rules(probability_threshold=0.60)).trades) == 0
    # Lower the bar and the same bar 1 now trades.
    assert len(run_backtest(frame, rules(probability_threshold=0.50)).trades) == 1


# --------------------------------------------------------------------------- costs

def test_costs_reduce_the_net_return(one_trade):
    result = run_backtest(one_trade, rules())
    trade = result.trades.iloc[0]
    expected_gross = 110.0 / 101.0 - 1.0
    expected_net = (110.0 / 101.0) * (1.0 - COST) * (1.0 - COST) - 1.0
    assert trade["gross_return"] == pytest.approx(expected_gross)
    assert trade["net_return"] == pytest.approx(expected_net)
    assert trade["net_return"] < trade["gross_return"]
    assert trade["costs_paid"] > 0


def test_a_winning_trade_still_loses_money_when_costs_exceed_the_move():
    """A +0.1% gross move is a net loss at 10 bps each way (0.2% round trip)."""
    opens = [100.0] * 12
    closes = [100.0] * 12
    opens[2], closes[7] = 100.0, 100.1
    probabilities = [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = run_backtest(make_prices(opens, closes, probabilities), rules())
    trade = result.trades.iloc[0]
    assert trade["gross_return"] > 0
    assert trade["net_return"] < 0
    assert trade["pnl"] < 0
    assert result.metrics["total_pnl"] < 0


def test_zero_cost_makes_the_net_return_equal_the_gross_return(one_trade):
    result = run_backtest(one_trade, rules(transaction_cost_bps=0.0))
    trade = result.trades.iloc[0]
    assert trade["net_return"] == pytest.approx(trade["gross_return"])
    assert trade["costs_paid"] == pytest.approx(0.0)


def test_higher_costs_never_improve_a_fixed_price_path(one_trade):
    cheap = run_backtest(one_trade, rules(transaction_cost_bps=0.0)).metrics["final_capital"]
    dear = run_backtest(one_trade, rules(transaction_cost_bps=50.0)).metrics["final_capital"]
    assert dear < cheap


def test_final_capital_equals_initial_plus_total_pnl(one_trade):
    result = run_backtest(one_trade, rules())
    metrics = result.metrics
    assert metrics["final_capital"] == pytest.approx(
        metrics["initial_capital"] + result.trades["pnl"].sum()
    )
    assert metrics["total_pnl"] == pytest.approx(result.trades["pnl"].sum())


def test_single_trade_pnl_matches_hand_computed_cash_flows(one_trade):
    """Full hand calculation: fees, sizing and P&L, independent of the engine."""
    result = run_backtest(one_trade, rules())
    cash = 10_000.0
    entry_price, exit_price, rate = 101.0, 110.0, COST

    units = cash / (entry_price * (1.0 + rate))
    cash_in = units * entry_price * (1.0 + rate)      # entry notional + fee
    cash_out = units * exit_price * (1.0 - rate)      # exit notional - fee
    assert units == pytest.approx(result.trades.iloc[0]["units"])
    assert cash_out - cash_in == pytest.approx(result.trades.iloc[0]["pnl"])
    assert result.metrics["final_capital"] == pytest.approx(cash_out)


# --------------------------------------------------------------------------- equity

def test_equity_is_marked_to_market_while_holding(one_trade):
    result = run_backtest(one_trade, rules())
    curve = result.equity_curve
    assert curve["in_position"].iloc[2:8].eq(1).all()
    assert curve["in_position"].sum() == HORIZON
    assert curve["equity"].notna().all()
    assert curve.index.is_monotonic_increasing


def test_equity_never_dips_below_zero_while_long_only():
    opens = [100.0] * 30
    closes = [100.0] * 30
    probabilities = [0.0] * 30
    opens[2] = 100.0
    closes[7] = 40.0  # a -60% exit
    probabilities[1] = 0.9
    result = run_backtest(make_prices(opens, closes, probabilities), rules())
    assert (result.equity_curve["equity"] >= 0).all()
    assert result.metrics["max_drawdown"] > 0


def test_buy_and_hold_comparison_uses_the_same_window(one_trade):
    result = run_backtest(one_trade, rules())
    closes = one_trade["close"].to_numpy()
    expected = closes[-1] / closes[0] - 1.0
    assert result.metrics["buy_hold_return"] == pytest.approx(expected)
    assert result.metrics["buy_hold_return_net_of_costs"] == pytest.approx(
        (closes[-1] / closes[0]) * (1 - COST) * (1 - COST) - 1.0
    )


def test_exposure_and_trade_counts_are_consistent():
    opens = [100.0] * 60
    closes = [100.0] * 60
    probabilities = [0.0] * 60
    for i in range(0, 58, HORIZON + 2):
        probabilities[i] = 0.9
    result = run_backtest(make_prices(opens, closes, probabilities), rules())
    n_trades = len(result.trades)
    assert result.metrics["n_trades"] == n_trades
    held = int(result.equity_curve["in_position"].sum())
    assert result.metrics["time_in_market"] == pytest.approx(held / 60)
    # The final trade may be clipped by the end of the sample, never extended.
    assert 0 < held <= n_trades * HORIZON
    assert (result.trades["hold_candles"] <= HORIZON).all()
    if n_trades:
        assert result.metrics["win_rate"] == pytest.approx(0.0)  # flat prices -> net loss


# --------------------------------------------------------------------------- rules

@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"probability_threshold": 0.0}, "strictly between 0 and 1"),
        ({"probability_threshold": 1.0}, "strictly between 0 and 1"),
        ({"transaction_cost_bps": -1.0}, ">= 0"),
        ({"initial_capital": 0.0}, "must be > 0"),
        ({"allow_short": True}, "long-only"),
        ({"max_concurrent_positions": 2}, "max_concurrent_positions"),
        ({"exit_mode": "stop_loss"}, "Unsupported exit_mode"),
        ({"exit_probability": 1.5}, "exit_probability"),
    ],
)
def test_invalid_rules_are_rejected(overrides, match):
    with pytest.raises(BacktestConfigError, match=match):
        BacktestRules(**{**rules().__dict__, **overrides}).validate()


def test_missing_columns_are_rejected(one_trade):
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(one_trade.drop(columns=["open"]), rules())
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(one_trade.drop(columns=["close"]), rules())


def test_empty_frame_is_rejected():
    empty = make_prices([], [], [])
    with pytest.raises(ValueError, match="empty frame"):
        run_backtest(empty, rules())


# --------------------------------------------------------------------------- misc

def test_backtest_is_deterministic(one_trade):
    a = run_backtest(one_trade, rules())
    b = run_backtest(one_trade, rules())
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.equity_curve, b.equity_curve)
    assert a.metrics == b.metrics


def test_early_exit_triggers_when_configured():
    opens = [100.0] * 20
    closes = [100.0] * 20
    probabilities = [0.0] * 20
    opens[2] = 100.0
    closes[3] = 105.0
    probabilities[1] = 0.95   # entry signal
    probabilities[3] = 0.95   # still bullish, no exit yet
    probabilities[4] = 0.05   # collapse -> exit at bar 4
    closes[4] = 105.0         # the exit is filled at the close of bar 4
    frame = make_prices(opens, closes, probabilities)
    result = run_backtest(frame, rules(exit_probability=0.30))
    trade = result.trades.iloc[0]
    assert trade["early_exit"]
    assert trade["exit_time"] == frame.index[4]
    assert trade["hold_candles"] < HORIZON
    assert trade["exit_price"] == pytest.approx(frame["close"].iloc[4])


def test_threshold_sensitivity_reports_every_threshold_tested(one_trade):
    """The gate is ``signal < threshold``, so a 0.90 signal still trades at 0.90."""
    thresholds = [0.5, 0.6, 0.7, 0.90, 0.95]
    table = threshold_sensitivity(one_trade, rules(), thresholds)
    assert set(table.columns) >= {"probability_threshold", "n_trades", "total_return"}
    assert table["probability_threshold"].tolist() == thresholds
    # 0.90 is inclusive, so only the 0.95 cut suppresses the trade.
    assert table["n_trades"].tolist() == [1, 1, 1, 1, 0]


def test_a_perfect_signal_on_a_rising_market_beats_a_falling_one():
    opens = [100.0] * 20
    closes = [100.0] * 20
    probabilities = [0.0] * 20
    opens[2] = 100.0
    closes[7] = 120.0
    probabilities[1] = 0.95
    up = run_backtest(make_prices(opens, closes, probabilities), rules())

    closes_down = [100.0] * 20
    closes_down[7] = 80.0
    down = run_backtest(make_prices(opens, closes_down, probabilities), rules())
    assert up.metrics["total_return"] > down.metrics["total_return"]
    assert up.metrics["win_rate"] == 1.0
    assert down.metrics["win_rate"] == 0.0
