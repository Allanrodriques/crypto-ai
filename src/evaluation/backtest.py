"""Research backtest for the model's ``probability_up`` output.

This is a **research** backtest, not a trading engine: no leverage, no
exchange connectivity, no order routing, no live execution.  Its purpose is to
answer "would a fixed, pre-declared rule based on this model's probability have
been profitable on the held-out period, after costs?"

Timing rules (the part that decides whether results are honest)
----------------------------------------------------------------
======================  =========================================================
Signal                  computed at the **close of candle t** from features that
                        only use candles ``<= t`` (guaranteed by
                        :mod:`src.features`).
Entry                   at the **open of candle t+1**.  Never at the close of
                        ``t`` — that close is only known *after* the candle ends,
                        so filling there would be look-ahead.
Exit                    at the **close of candle t + horizon**, holding exactly
                        ``horizon`` candles, or earlier if
                        ``exit_probability`` triggers.
Concurrency             ``max_concurrent_positions = 1``.  Entries are therefore
                        non-overlapping and no candle's return is counted twice.
======================  =========================================================

The entry price (``open[t+1]``) deliberately differs from the label's reference
price (``close[t]``).  That is intentional: the label is a clean statement of
"did price rise over the next H candles", while the backtest asks the harder
question of whether the *executable* entry would have profited.

Threshold provenance
--------------------
``probability_threshold`` is read from ``config/config.yaml`` and applied
unchanged.  It is never selected by scanning test-set results.  The
:func:`threshold_sensitivity` helper exists for inspection on the **validation**
period only, and its output is labelled as a diagnostic, not a selection step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("evaluation.backtest")

DEFAULT_PRICE_COLUMN = "close"


class BacktestConfigError(ValueError):
    """Raised for backtest settings this implementation deliberately refuses."""


# --------------------------------------------------------------------------- config

@dataclass
class BacktestRules:
    """Fully-resolved, config-driven backtest rules."""

    initial_capital: float = 10_000.0
    probability_threshold: float = 0.60
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 0.0
    allow_short: bool = False
    max_concurrent_positions: int = 1
    exit_mode: str = "horizon"
    exit_probability: float | None = None
    risk_free_rate_annual: float = 0.0
    periods_per_year: int = 8760
    price_column: str = "close"
    horizon_candles: int = 6

    @property
    def cost_rate(self) -> float:
        """One-way exchange fee as a fraction (10 bps -> 0.001)."""
        return float(self.transaction_cost_bps) / 10_000.0

    @property
    def slippage_rate(self) -> float:
        """One-way slippage as a fraction, applied to the *fill price*.

        Slippage is kept separate from the fee because they are physically
        different costs.  The fee is a rate charged on notional and scales with
        price; slippage is the price concession paid to get filled, and it
        silently scales with *volatility* - a fixed 5 bps is optimistic in a
        volatile market and pessimistic in a quiet one.  Conflating them, as
        ``transaction_cost_bps`` alone did, hides which of the two is actually
        destroying the strategy.
        """
        return float(self.slippage_bps) / 10_000.0

    @property
    def one_way_total_rate(self) -> float:
        """Combined one-way fee + slippage, for quick comparisons."""
        return self.cost_rate + self.slippage_rate

    @classmethod
    def from_config(
        cls, backtest_config: Mapping[str, Any], *, horizon_candles: int
    ) -> "BacktestRules":
        cfg = dict(backtest_config or {})
        rules = cls(
            initial_capital=float(cfg.get("initial_capital", 10_000.0)),
            probability_threshold=float(cfg.get("probability_threshold", 0.60)),
            transaction_cost_bps=float(cfg.get("transaction_cost_bps", 10.0)),
            slippage_bps=float(cfg.get("slippage_bps", 0.0)),
            allow_short=bool(cfg.get("allow_short", False)),
            max_concurrent_positions=int(cfg.get("max_concurrent_positions", 1)),
            exit_mode=str(cfg.get("exit_mode", "horizon")),
            exit_probability=cfg.get("exit_probability"),
            risk_free_rate_annual=float(cfg.get("risk_free_rate_annual", 0.0)),
            periods_per_year=int(cfg.get("periods_per_year", 8760)),
            price_column=str(cfg.get("price_column", DEFAULT_PRICE_COLUMN)),
            horizon_candles=int(horizon_candles),
        )
        rules.validate()
        return rules

    def validate(self) -> "BacktestRules":
        if not 0 < self.probability_threshold < 1:
            raise BacktestConfigError("probability_threshold must be strictly between 0 and 1")
        if self.transaction_cost_bps < 0:
            raise BacktestConfigError("transaction_cost_bps must be >= 0")
        if self.slippage_bps < 0:
            raise BacktestConfigError("slippage_bps must be >= 0")
        if self.slippage_bps >= 5_000:
            raise BacktestConfigError("slippage_bps must be < 5000 (i.e. < 50% per side)")
        if self.initial_capital <= 0:
            raise BacktestConfigError("initial_capital must be > 0")
        if self.allow_short:
            raise BacktestConfigError(
                "This V1 backtest is long-only. Set backtest.allow_short: false. "
                "Short logic would need a borrow/borrow-cost model that is out of scope."
            )
        if self.max_concurrent_positions != 1:
            raise BacktestConfigError(
                "Only max_concurrent_positions: 1 is supported: overlapping entries would "
                "double-count the same candles' return. Increase exit_mode/horizon instead."
            )
        if self.exit_mode not in {"horizon"}:
            raise BacktestConfigError(f"Unsupported exit_mode {self.exit_mode!r}; only 'horizon' is implemented")
        if self.exit_probability is not None and not 0 < float(self.exit_probability) < 1:
            raise BacktestConfigError("backtest.exit_probability must be strictly between 0 and 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["one_way_fee_rate"] = self.cost_rate
        payload["one_way_slippage_rate"] = self.slippage_rate
        payload["one_way_total_rate"] = self.one_way_total_rate
        payload["threshold_provenance"] = "config file — fixed a priori, not tuned on the test period"
        return payload


# --------------------------------------------------------------------------- result

@dataclass
class Trade:
    """One completed round trip."""

    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    raw_entry_price: float
    raw_exit_price: float
    entry_price: float
    exit_price: float
    units: float
    gross_return: float
    net_return: float
    pnl: float
    fees_paid: float
    slippage_cost: float
    costs_paid: float
    hold_candles: int
    probability_up: float
    early_exit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Equity curve, trades and the full metric set for one backtest run."""

    rules: dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, Any]
    prediction_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": self.rules,
            "metrics": self.metrics,
            "prediction_metrics": self.prediction_metrics,
            "n_trades": int(len(self.trades)),
            "trades": self.trades.to_dict(orient="records"),
        }


# --------------------------------------------------------------------------- engine

def run_backtest(
    frame: pd.DataFrame,
    rules: BacktestRules,
    *,
    probability_column: str = "probability_up",
    target_column: str = "target",
) -> BacktestResult:
    """Simulate the fixed rule over ``frame``.

    Parameters
    ----------
    frame:
        UTC-indexed frame containing ``probability_up``, ``open``, ``close`` and
        ``target``.  Must be chronologically sorted and gap-free.
    rules:
        Resolved backtest rules.

    Returns
    -------
    BacktestResult
    """
    rules.validate()
    required = {probability_column, "open", rules.price_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame is missing column(s): {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot backtest an empty frame")

    proba = frame[probability_column].to_numpy(dtype="float64")
    open_ = frame["open"].to_numpy(dtype="float64")
    close = frame[rules.price_column].to_numpy(dtype="float64")
    times = frame.index
    n = len(frame)
    cost_rate = rules.cost_rate
    slip = rules.slippage_rate
    horizon = max(1, int(rules.horizon_candles))

    cash = float(rules.initial_capital)
    units = 0.0
    trades: list[Trade] = []

    equity = np.full(n, np.nan)
    position_flag = np.zeros(n, dtype=int)
    peak = float(rules.initial_capital)
    max_dd = 0.0

    i = 0
    while i < n:
        # Mark-to-market while flat.
        if units == 0.0:
            equity[i] = cash
        else:
            equity[i] = cash + units * close[i]
        if equity[i] > peak:
            peak = equity[i]
        max_dd = max(max_dd, (peak - equity[i]) / peak if peak > 0 else 0.0)

        signal = proba[i]
        if not np.isfinite(signal) or signal < rules.probability_threshold:
            i += 1
            continue

        # Signal at close of t -> fill at the OPEN of t+1.
        entry_i = i + 1
        if entry_i >= n:
            break

        raw_entry = float(open_[entry_i])
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            i += 1
            continue
        # Fill worse than the quoted open by the configured slippage.
        entry_price = raw_entry * (1.0 + slip)

        # Hold `horizon` candles: enter at open of t+1, exit at close of t+horizon.
        exit_i = min(entry_i + horizon - 1, n - 1)
        early_exit = False
        if rules.exit_probability is not None:
            for j in range(entry_i + 1, exit_i + 1):
                if np.isfinite(proba[j]) and proba[j] < float(rules.exit_probability):
                    exit_i = j
                    early_exit = True
                    break

        raw_exit = float(close[exit_i])
        if not np.isfinite(raw_exit) or raw_exit <= 0:
            i = exit_i + 1
            continue
        # Exit fills worse than the quoted close.
        exit_price = raw_exit * (1.0 - slip)

        # All-in sizing: buy `units` with all available cash, paying the entry fee.
        position_units = cash / (entry_price * (1.0 + cost_rate))
        entry_fee = position_units * entry_price * cost_rate
        cash -= position_units * entry_price + entry_fee
        units = position_units

        for j in range(entry_i, exit_i + 1):
            position_flag[j] = 1
            value = cash + units * close[j]
            equity[j] = value
            if value > peak:
                peak = value
            max_dd = max(max_dd, (peak - value) / peak if peak > 0 else 0.0)

        exit_fee = position_units * exit_price * cost_rate
        cash += position_units * exit_price - exit_fee

        gross_return = raw_exit / raw_entry - 1.0
        net_return = (exit_price / entry_price) * (1.0 - cost_rate) * (1.0 - cost_rate) - 1.0
        # Slippage's cash impact versus trading at the quoted prices.
        slippage_cost = (position_units * (entry_price - raw_entry)) + (
            position_units * (raw_exit - exit_price)
        )
        # Cash out minus cash in, both including the fee already charged.
        cash_out = position_units * exit_price * (1.0 - cost_rate)
        cash_in = position_units * entry_price * (1.0 + cost_rate)
        trades.append(
            Trade(
                signal_time=times[i],
                entry_time=times[entry_i],
                exit_time=times[exit_i],
                raw_entry_price=raw_entry,
                raw_exit_price=raw_exit,
                entry_price=entry_price,
                exit_price=exit_price,
                units=position_units,
                gross_return=gross_return,
                net_return=net_return,
                pnl=cash_out - cash_in,
                fees_paid=entry_fee + exit_fee,
                slippage_cost=slippage_cost,
                costs_paid=entry_fee + exit_fee + slippage_cost,
                hold_candles=int(exit_i - entry_i + 1),
                probability_up=float(signal),
                early_exit=early_exit,
            )
        )

        # The position is closed.  `units` must be cleared, otherwise every later
        # bar is marked as if the sold units were still held, and equity (and
        # therefore every metric derived from it) double-counts the trade.
        units = 0.0
        i = exit_i + 1  # resume after the exit: entries never overlap

    equity = pd.Series(equity, index=times, name="equity")
    equity = equity.ffill().bfill()
    equity_curve = pd.DataFrame(
        {
            "equity": equity,
            "close": close,
            "probability_up": proba,
            "in_position": position_flag,
        },
        index=times,
    )
    if target_column in frame.columns:
        equity_curve["target"] = frame[target_column].to_numpy()

    trades_frame = pd.DataFrame([t.to_dict() for t in trades])
    metrics = compute_backtest_metrics(
        equity=equity,
        trades=trades_frame,
        close=close,
        rules=rules,
        position_flag=position_flag,
        max_drawdown=max_dd,
    )
    return BacktestResult(
        rules=rules.to_dict(),
        trades=trades_frame,
        equity_curve=equity_curve,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- metrics

def compute_backtest_metrics(
    *,
    equity: pd.Series,
    trades: pd.DataFrame,
    close: np.ndarray,
    rules: BacktestRules,
    position_flag: np.ndarray,
    max_drawdown: float | None = None,
) -> dict[str, Any]:
    """Aggregate performance statistics for the strategy.

    The buy-and-hold comparison is the reference that keeps the ML model's
    apparent edge honest: a strategy that trails simply holding BTC has not
    earned its complexity.
    """
    initial = float(rules.initial_capital)
    final = float(equity.iloc[-1]) if len(equity) else initial
    total_return = final / initial - 1.0 if initial else 0.0

    returns = equity.pct_change().dropna()
    ann_factor = float(np.sqrt(max(rules.periods_per_year, 1)))
    per_bar_rf = (1.0 + rules.risk_free_rate_annual) ** (1.0 / max(rules.periods_per_year, 1)) - 1.0
    excess = returns - per_bar_rf
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(excess.mean() / volatility * ann_factor) if volatility > 0 else None

    drawdown = _drawdown_series(equity)
    if max_drawdown is None:
        max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    # Buy & hold over exactly the same window, for a like-for-like comparison.
    bh_gross = float(close[-1] / close[0] - 1.0) if len(close) > 1 and close[0] else None
    cost, slip = rules.cost_rate, rules.slippage_rate
    bh_net = (
        float((close[-1] / close[0]) * (1 - slip) * (1 - cost) * (1 - slip) * (1 - cost) - 1.0)
        if bh_gross is not None
        else None
    )

    n_trades = int(len(trades))
    wins = int((trades["net_return"] > 0).sum()) if n_trades else 0

    # Gross profit / gross loss.  Win rate alone is misleading: a 40% win-rate
    # book with 4:1 reward:risk beats a 60% book that averages winners and losers
    # at parity, and only the ratio of the two shows it.
    gross_profit = float(trades.loc[trades["net_return"] > 0, "pnl"].sum()) if n_trades else 0.0
    gross_loss = float(-trades.loc[trades["net_return"] < 0, "pnl"].sum()) if n_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    payoff = (
        float(trades.loc[trades["net_return"] > 0, "net_return"].mean())
        / abs(float(trades.loc[trades["net_return"] < 0, "net_return"].mean()))
        if n_trades and gross_loss > 0
        else None
    )

    # Annualised return from the actual elapsed time, not from bar count.
    n_bars = int(len(equity))
    years = n_bars / max(rules.periods_per_year, 1) if n_bars else 0.0
    cagr = ((final / initial) ** (1.0 / years) - 1.0) if years > 0 and initial > 0 and final > 0 else None

    return {
        # --- capital -----------------------------------------------------------
        "initial_capital": initial,
        "final_capital": final,
        "total_return": total_return,
        "cagr": cagr,
        "total_pnl": final - initial,
        "costs_paid": float(trades["costs_paid"].sum()) if n_trades else 0.0,
        "fees_paid": float(trades["fees_paid"].sum()) if n_trades else 0.0,
        "slippage_cost_paid": float(trades["slippage_cost"].sum()) if n_trades else 0.0,
        "transaction_cost_bps_per_side": rules.transaction_cost_bps,
        "slippage_bps_per_side": rules.slippage_bps,
        # --- benchmark ---------------------------------------------------------
        "buy_hold_return": bh_gross,
        "buy_hold_return_net_of_costs": bh_net,
        "buy_hold_final_capital": initial * (1.0 + bh_net) if bh_net is not None else None,
        "excess_vs_buy_hold": (total_return - bh_net) if bh_net is not None else None,
        # --- risk --------------------------------------------------------------
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown) * 100.0,
        "annualised_volatility": volatility * ann_factor if volatility else 0.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": _sortino(returns, per_bar_rf, ann_factor),
        "calmar_ratio": (
            float(cagr / max_drawdown) if cagr is not None and max_drawdown > 0 else None
        ),
        # --- trades ------------------------------------------------------------
        "n_trades": n_trades,
        "win_rate": wins / n_trades if n_trades else None,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_trade_return": float(trades["net_return"].mean()) if n_trades else None,
        "median_trade_return": float(trades["net_return"].median()) if n_trades else None,
        "best_trade_return": float(trades["net_return"].max()) if n_trades else None,
        "worst_trade_return": float(trades["net_return"].min()) if n_trades else None,
        "avg_hold_candles": float(trades["hold_candles"].mean()) if n_trades else None,
        "total_costs_bps_equivalent": float(
            (trades["costs_paid"].sum() / max(final, 1e-9)) * 10_000
        ) if n_trades else 0.0,
        # --- exposure ----------------------------------------------------------
        "time_in_market": float(position_flag.mean()) if position_flag.size else None,
        "n_bars": n_bars,
    }


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak."""
    if equity.empty:
        return pd.Series(dtype="float64")
    running_peak = equity.cummax()
    return (running_peak - equity) / running_peak


def _sortino(returns: pd.Series, per_bar_rf: float, ann_factor: float) -> float | None:
    """Sortino ratio: excess return over *downside* deviation only."""
    if len(returns) < 2:
        return None
    excess = returns - per_bar_rf
    downside = excess[excess < 0]
    if len(downside) < 2:
        return None
    deviation = float(np.sqrt((downside**2).mean()))
    if deviation <= 0:
        return None
    return float(excess.mean() / deviation * ann_factor)


# --------------------------------------------------------------------------- diagnostics

def threshold_sensitivity(
    frame: pd.DataFrame,
    rules: BacktestRules,
    thresholds: Sequence[float],
    *,
    probability_column: str = "probability_up",
) -> pd.DataFrame:
    """Re-run the backtest across several probability thresholds.

    .. warning::
       **Diagnostic only.**  This function exists so a threshold's sensitivity
       can be inspected on the *validation* period before it is fixed in the
       config.  Running it against the test period and keeping the best result
       would be test-set tuning, which this project does not do.  The pipeline
       calls it for the validation split only and labels the output as such.
    """
    rows = []
    for threshold in thresholds:
        local = BacktestRules(**{**rules.__dict__, "probability_threshold": float(threshold)})
        local.validate()
        result = run_backtest(frame, local, probability_column=probability_column)
        rows.append(
            {
                "probability_threshold": float(threshold),
                "total_return": result.metrics["total_return"],
                "buy_hold_return": result.metrics["buy_hold_return"],
                "n_trades": result.metrics["n_trades"],
                "win_rate": result.metrics["win_rate"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "max_drawdown": result.metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)
