"""Turn V3 forward-return predictions into a cost-aware, tradable return series.

Why this module exists
----------------------
Everything upstream in V3 produces a *forecast*, and a forecast is not a return.
The gap between the two is where a research pipeline most easily tells itself a
flattering story, and for this project the gap has one specific, large trap.

**The trap: overlapping forward windows.**
V3's target at row ``t`` is the realised return from ``t`` to ``t + H``, with
``H`` between 24h and 4320h.  A walk-forward prediction made *at* ``t`` describes
exactly that same window, so a position opened at ``t`` earns precisely
``target[t]`` - there is no extra lag to apply, and adding one would silently
throw away a real edge.  But consecutive rows' windows share almost all of their
outcomes: at ``H = 30d``, the window starting one hour later overlaps 99.86% of
the window starting now.  Summing ``target * position`` across every hourly row
therefore counts the same thirty days of P&L roughly 720 times.  A model with a
2% real edge at ``H = 30d`` "earns" 1400% under that arithmetic, and the number
is not a small exaggeration - it is a different quantity entirely, one that no
account can ever realise because the capital is committed to 720 simultaneous
positions on the same single asset.

So this module makes two modes, and refuses to let them be confused:

``non_overlapping`` (the default)
    Only every ``k``-th row is traded, where ``k`` is the horizon in bars.  One
    position is open at a time, each trade's window contains every other trade's
    window in no part, and the compounded return is a series an account could
    actually have produced.  **Only this mode's ``total_return`` and ``cagr``
    are realisable performance numbers.**

``overlapping``
    Every row is scored.  It is kept because it is the right *diagnostic* - it
    shows how much of the signal is available at every timestamp, which is what
    a continuously-rebalanced portfolio would want - and because deleting it
    entirely would make the anti-pattern easy to rediscover.  But its totals are
    marked ``realisable=False``, carry a warning string in
    :attr:`StrategyResult.note`, log a :mod:`logging` warning on every call, and
    print that warning into every flattened table.  :data:`OVERLAPPING_WARNING`
    is the single source of that text so no caller can paraphrase it away.

Anti-pattern, in one line: ``overlapping.total_return / non_overlapping.total_return``
is approximately ``H`` in bars, and every factor of ``H`` above the honest
figure is double-counted P&L.

Timing and annualisation
------------------------
Every Sharpe-style statistic here annualises by the **horizon**, not by the hour.
A naive hourly annualisation (``sqrt(8760)``) applied to the per-hour P&L of a
180-day horizon divides by 180 and then multiplies by 94 - producing a number
roughly 17000x the real risk-adjusted return, with no visible error anywhere in
the code that produced it.  The per-period Sharpe is therefore scaled by
``sqrt(365.25 / horizon_days)``: the number of independent opportunities per
year, which for a non-overlapping 30-day book is about twelve.  The same factor
is used in both modes so the two are comparable; note that in ``overlapping``
mode the "periods" are not independent, so even the horizon-scaled Sharpe is
optimistic there - another reason the mode is not realisable.

Cost accounting
---------------
Costs are charged **per position change**, never per row, because the thing that
costs money is crossing the spread and paying a fee, not holding.  A position
that stays at 1 for a hundred periods pays exactly what a position that stays at
1 for one period pays: one opening fill.  The charge is
``one_way_bps x |change in position|`` where
``one_way_bps = fee_bps + slippage_bps``, so a complete round trip (open then
close) costs exactly ``CostModel.round_trip_bps()`` and the name means what it
says.  See the ``run_strategy`` docstring for why the alternative reading - one
full round trip per position change - would double-charge every trade.

An exit is charged to the **last period of the hold it closes**, not to the
period after it.  In ``non_overlapping`` mode that keeps a flat period flat
instead of showing a cost-only loss out of nowhere, and it does not change the
total.

The cost of a short is symmetric.  A flip from ``+1`` to ``-1`` is a change of
2.0, correctly charged as a close plus an open.

NaN policy
----------
Every metric that cannot be computed returns ``float('nan')`` - never an
exception, and never a confident ``0.0``.  The distinction matters: a strategy
with no trades and a strategy with zero risk are different facts, and returning
``0.0`` for both would report "no losses" for a strategy that never existed.
:func:`run_strategy` therefore returns ``NaN`` for ``total_return``, ``cagr``,
``sharpe``, ``sortino``, ``calmar``, ``hit_rate`` and both Wilson bounds when
there are no trades, and when the trade returns have no dispersion.

Three quantities are the deliberate exceptions, because zero is the *truth*
about them rather than a stand-in for "unknown": ``exposure`` (no trades means
no time in the market), ``turnover`` and ``cost_drag`` (no trades means no
trading), and ``max_drawdown`` (a flat equity curve has a real, well-defined
drawdown of zero).  ``max_drawdown`` is always reported with ``n_trades``
alongside it for exactly this reason.

``n_trades`` versus ``turnover``
-------------------------------
These count different things and conflating them is how a cost estimate ends up
either ten times too small or ten times too large.

``n_trades`` is the number of **invested periods** - rebalances that carried a
non-zero position.  In non-overlapping mode each of those periods is a complete
hold: opened at ``t``, closed at ``t + H``, so 125 invested periods is 125
round trips of *exposure* whether or not any of them needed an order.

``turnover`` is the number of **actual fills**, the sum of ``|change in
position|``.  It is smaller, because two consecutive invested periods are one
continuous hold rather than a close followed by an open: a book that is long
across 63 rebalances but only enters and exits 30 times each has 63 invested
periods and 60 fills.  Costs are charged on ``turnover``; ``years`` and CAGR
use ``n_trades``, because wall-clock commitment is a property of being in the
market, not of how often you crossed the spread.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger
from src.v3.horizons import Horizon, parse_horizon
from src.v3.uncertainty import wilson_interval

logger = get_logger("v3.strategy")

__all__ = [
    "CostModel",
    "StrategyError",
    "StrategyResult",
    "bootstrap_sharpe",
    "compare_to_baselines",
    "non_overlapping_mask",
    "positions_from_predictions",
    "run_strategy",
    "sweep_strategies",
    "MODES",
    "NON_OVERLAPPING",
    "OVERLAPPING",
    "OVERLAPPING_WARNING",
]

#: Default mode.  Deliberately the only mode whose ``total_return`` means
#: "this is what the account would have made".
NON_OVERLAPPING = "non_overlapping"

#: Diagnostic mode.  Realisable totals are ``NaN``-adjacent by construction: see
#: :data:`OVERLAPPING_WARNING`.
OVERLAPPING = "overlapping"

#: Both modes, in report order: the honest one first.
MODES: tuple[str, ...] = (NON_OVERLAPPING, OVERLAPPING)

#: Stamped into :attr:`StrategyResult.note`, into every flattened sweep row, and
#: logged as a warning whenever ``mode='overlapping'`` runs.  Kept as one string
#: so no caller can quietly soften it.
OVERLAPPING_WARNING = (
    "NOT REALISABLE: mode='overlapping' scores every hourly row, whose forward "
    "windows overlap by ~99.9%, so the same P&L is summed about H times. Read "
    "mode='non_overlapping' for any achievable-performance claim."
)

#: Julian year.  Used for every wall-clock conversion here so that a 365-day
#: convention never silently disagrees with a 365.25 one.
DAYS_PER_YEAR = 365.25

#: Basis points per unit return.
BPS = 10_000.0

#: Column holding the realised forward return in a ``HorizonResult.predictions``
#: frame.
TARGET_COLUMN = "target"

#: Returned in place of a metric that is undefined for the surviving rows.
NAN = float("nan")


class StrategyError(ValueError):
    """Raised for inputs this module refuses to guess about.

    A :class:`ValueError` subclass so callers can keep catching the built-in, and
    deliberately loud: every case it covers is one where a silent default would
    produce a plausible-looking but wrong return series.
    """


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class CostModel:
    """Per-side trading frictions, in basis points of notional.

    Fee and slippage are separate because they are physically different costs.
    The fee is a rate charged on notional and scales with price; slippage is the
    concession paid to get filled and scales with *volatility*, so a fixed
    2 bps is optimistic in a violent market and pessimistic in a quiet one.
    Conflating them hides which one is actually destroying the strategy.

    Defaults are Binance-spot taker levels (5 bps fee) plus a small slippage
    allowance.  They are applied to notional, not to equity, and are charged on
    position changes only - see the module docstring.
    """

    fee_bps: float = 5.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        for name in ("fee_bps", "slippage_bps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise StrategyError(f"{name} must be a finite non-negative number, got {value!r}")

    def one_way_bps(self) -> float:
        """Cost of a single fill: one fee plus one slippage concession."""
        return float(self.fee_bps) + float(self.slippage_bps)

    def round_trip_bps(self) -> float:
        """Cost of one complete round trip: an opening fill and a closing fill.

        This is the number a single trade pays end to end, which is why it is
        twice :meth:`one_way_bps` and not the whole per-trade charge on its own.
        """
        return 2.0 * self.one_way_bps()

    def to_dict(self) -> dict[str, float]:
        return {
            "fee_bps": float(self.fee_bps),
            "slippage_bps": float(self.slippage_bps),
            "one_way_bps": self.one_way_bps(),
            "round_trip_bps": self.round_trip_bps(),
        }


# --------------------------------------------------------------------------- result

@dataclass(frozen=True)
class StrategyResult:
    """One (horizon, model, mode) run: the equity series and its statistics.

    ``total_return`` and ``cagr`` are realisable performance claims **only when**
    :attr:`realisable` is ``True``.  :attr:`note` carries the reason when it is
    not, and is empty for a clean non-overlapping run.

    ``n_trades`` is invested periods and ``turnover`` is actual fills - two
    different counts that disagree whenever the book holds through consecutive
    periods.  See the module docstring; the short version is that costs are
    charged on ``turnover`` and ``years`` is built from ``n_trades``.

    Every metric that cannot be computed is ``NaN``, never a confident ``0.0``,
    with the documented exceptions of :attr:`exposure`, :attr:`turnover`,
    :attr:`cost_drag` and :attr:`max_drawdown`, for which zero is the truth.
    """

    horizon: str
    model: str
    mode: str
    n_trades: int
    exposure: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    hit_rate_ci_lower: float
    hit_rate_ci_upper: float
    turnover: float
    cost_drag: float
    equity: pd.Series
    years: float = NAN
    realisable: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, float | str | int]:
        """Flatten to scalars for a report table.  The equity curve is dropped.

        ``realisable`` is emitted as ``0``/``1`` rather than a bool so the whole
        row is numeric and sorts and filters like the rest of the table - and so
        that a report which forgets to look at it still has to choose a number.
        """
        return {
            "horizon": self.horizon,
            "model": self.model,
            "mode": self.mode,
            "realisable": int(self.realisable),
            "n_trades": int(self.n_trades),
            "exposure": float(self.exposure),
            "total_return": float(self.total_return),
            "cagr": float(self.cagr),
            "sharpe": float(self.sharpe),
            "sortino": float(self.sortino),
            "max_drawdown": float(self.max_drawdown),
            "calmar": float(self.calmar),
            "hit_rate": float(self.hit_rate),
            "hit_rate_ci_lower": float(self.hit_rate_ci_lower),
            "hit_rate_ci_upper": float(self.hit_rate_ci_upper),
            "turnover": float(self.turnover),
            "cost_drag": float(self.cost_drag),
            "years": float(self.years),
            "note": self.note,
        }


#: Column order for every flattened table this module produces, taken from the
#: dataclass so the two can never drift apart.  ``equity`` is excluded: a Series
#: does not belong in a one-row-per-run table.
RESULT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(StrategyResult) if f.name != "equity")


# --------------------------------------------------------------------------- horizon plumbing

def horizon_delta(horizon: Any) -> pd.Timedelta:
    """Coerce ``horizon`` to a :class:`pandas.Timedelta`.

    Accepts a :class:`src.v3.horizons.Horizon` (its ``.delta``), a string like
    ``"30d"`` (via :func:`src.v3.horizons.parse_horizon`), a
    :class:`pandas.Timedelta`, or a bare number of **hours**.  Anything else
    raises: a misread horizon produces a target column that is subtly wrong and
    every downstream number wrong with it, so this is not a place to be helpful.
    """
    if isinstance(horizon, Horizon):
        delta = horizon.delta
    elif isinstance(horizon, pd.Timedelta):
        delta = horizon
    elif isinstance(horizon, str):
        delta = parse_horizon(horizon)
    elif isinstance(horizon, (int, float, np.integer, np.floating)):
        delta = pd.Timedelta(hours=float(horizon))
    else:
        raise StrategyError(
            "horizon must be a Horizon, a string like '30d', a Timedelta, or a number of "
            f"hours; got {type(horizon).__name__}"
        )
    if not isinstance(delta, pd.Timedelta):
        raise StrategyError(f"horizon did not resolve to a Timedelta; got {delta!r}")
    if delta <= pd.Timedelta(0):
        raise StrategyError(f"horizon must be positive; got {delta}")
    return delta


def horizon_label(horizon: Any) -> str:
    """Short display label for ``horizon``, preferring a ``Horizon.label``.

    Coerced through :func:`horizon_delta` so every caller gets the *same* label
    for the same horizon regardless of how it was spelled.  A ``Timedelta`` is
    rendered as ``"1d"``/``"12h"`` rather than pandas' ``"1 days 00:00:00"``,
    because this string ends up in a report column and a report that prints
    ``1 days 00:00:00`` as a horizon is a report nobody trusts.
    """
    if isinstance(horizon, Horizon):
        return str(horizon.label)
    if isinstance(horizon, str):
        return str(horizon)
    delta = horizon_delta(horizon)
    if delta >= pd.Timedelta(days=1):
        days = delta / pd.Timedelta(days=1)
        if float(days).is_integer():
            return f"{int(days)}d"
    hours = delta / pd.Timedelta(hours=1)
    if float(hours).is_integer():
        return f"{int(hours)}h"
    minutes = delta / pd.Timedelta(minutes=1)
    if float(minutes).is_integer():
        return f"{int(minutes)}m"
    return str(delta)


def _periods_per_year(horizon_days: float) -> float:
    """Independent opportunities per year at this horizon.

    ``365.25 / horizon_days``: one trade every horizon.  This is the
    annualisation factor for every Sharpe-style statistic, and the inverse of the
    ``years`` used for CAGR, so the two can never disagree.
    """
    return DAYS_PER_YEAR / horizon_days


def non_overlapping_mask(index: pd.DatetimeIndex, horizon: Any) -> pd.Series:
    """Boolean mask selecting the entry row of each non-overlapping window.

    A row is an entry when it is the first row of the index, or when it lies at
    least one full horizon after the previous entry.

    Wall-clock greedy, not a positional ``::k`` stride, because a stride needs a
    bar frequency it cannot see.  ``k`` would have to come from
    ``Horizon.nominal_bars`` - ``days x bars_per_unit`` - and that is a
    *configuration* value: use a 1d ``Horizon`` built with ``bars_per_unit=24``
    on a 4h grid and the stride trades every 96 hours, a quarter as often as the
    horizon allows, with no error anywhere.  On a grid with missing candles -
    :mod:`src.v3.horizons` documents one in this project's own BTC series - the
    stride also drifts wide, because the ``k``-th surviving candle is further
    away in wall clock than ``k`` candles ought to be.  Both failure modes lose
    trades; neither would announce itself.

    On a regular grid of the configured frequency the two agree exactly: the
    mask is ``index[::k]`` with ``k = horizon_hours / bar_hours``, which has
    ``ceil(n / k)`` entries.

    Parameters
    ----------
    index:
        UTC :class:`pandas.DatetimeIndex`, ascending.
    horizon:
        A ``Horizon``, ``"30d"``, a ``Timedelta``, or a number of hours.

    Returns
    -------
    pandas.Series
        Boolean series on ``index``, named ``"is_entry"``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise StrategyError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if len(index) == 0:
        return pd.Series(dtype=bool, index=index, name="is_entry")

    step_ns = int(horizon_delta(horizon).value)
    stamps = index.asi8
    entries = np.zeros(len(index), dtype=bool)
    entries[0] = True
    last = stamps[0]
    for position in range(1, len(index)):
        if stamps[position] - last >= step_ns:
            entries[position] = True
            last = stamps[position]
    return pd.Series(entries, index=index, name="is_entry")


# --------------------------------------------------------------------------- positions

def _resolve_prob_column(frame: pd.DataFrame, prob_col: Any) -> str:
    """Validate a single probability-column name against ``frame``.

    A *sequence* of names is refused rather than combined.  Averaging two
    models' calibrated probabilities does not produce a calibrated probability,
    and taking the min or the max is a modelling decision this module has no
    business making silently - the alternatives all produce a plausible number
    and a wrong one.  Scoring several models is what :func:`sweep_strategies` is
    for.  Refusing here also keeps the failure a ``StrategyError`` instead of an
    unhashable-type ``TypeError`` from deep inside pandas.
    """
    if isinstance(prob_col, (list, tuple, set, pd.Index, np.ndarray)):
        raise StrategyError(
            f"prob_col must be a single column name, got a {type(prob_col).__name__} of "
            f"{len(prob_col)}; combining several models' probabilities into one threshold "
            "test is a modelling decision this module will not make silently. Run one "
            "model at a time via sweep_strategies, or pick the column to trade."
        )
    if not isinstance(prob_col, str):
        raise StrategyError(f"prob_col must be a string, got {type(prob_col).__name__}")
    if prob_col not in frame.columns:
        raise StrategyError(
            f"prob_col {prob_col!r} is not a column; available: {sorted(frame.columns)}"
        )
    return prob_col


def positions_from_predictions(
    pred: pd.Series | pd.DataFrame,
    threshold: float = 0.0,
    prob_col: str | None = None,
    prob_threshold: float = 0.5,
) -> pd.Series:
    """Turn a signal into a long/flat position series.

    Two mutually exclusive paths, because a regression output and a calibrated
    probability are not the same kind of number and conflating them is how a
    return model ends up "traded" on a rescaled number:

    * ``prob_col is None`` - long where ``pred > threshold``.  Strictly greater,
      so a prediction of exactly ``0.0`` is flat: a model with no view must not
      be given a position.
    * ``prob_col`` given - long where ``pred[prob_col] >= prob_threshold``.
      Inclusive, because a calibrated probability of exactly the threshold is a
      genuine statement.  ``pred`` must then be a :class:`~pandas.DataFrame`: a
      Series carries no column names, and guessing which of them was meant is
      exactly the kind of silent substitution that produces a wrong answer.

    Parameters
    ----------
    pred:
        Signal series, or a frame when ``prob_col`` is used.
    threshold:
        Return threshold for the long side.  Must be finite.
    prob_col:
        Name of the calibrated-probability column, or ``None``.  A single name
        only; a list is refused (see :func:`_resolve_prob_column`).
    prob_threshold:
        Probability at or above which to be long.  Must lie in ``[0, 1]``.

    Returns
    -------
    pandas.Series
        Float series of ``0.0``/``1.0`` on ``pred``'s index, named ``"position"``.
        Non-finite signals are flat rather than dropped, so the returned index
        always matches the input.
    """
    limit = float(threshold)
    if not np.isfinite(limit):
        raise StrategyError(f"threshold must be finite, got {threshold!r}")
    cut = float(prob_threshold)
    if not np.isfinite(cut) or not 0.0 <= cut <= 1.0:
        raise StrategyError(f"prob_threshold must lie in [0, 1], got {prob_threshold!r}")

    if prob_col is not None:
        if not isinstance(pred, pd.DataFrame):
            raise StrategyError(
                "prob_col was given but pred is a Series, which has no columns to select "
                "from; pass the prediction frame and name the column explicitly"
            )
        if isinstance(pred, pd.DataFrame):
            prob_col = _resolve_prob_column(pred, prob_col)
        values = pred[prob_col].to_numpy(dtype="float64")
        # Inclusive on the calibrated side; see the docstring.
        signal = np.where(np.isfinite(values) & (values >= cut), 1.0, 0.0)
        source = pred[prob_col]
    else:
        if isinstance(pred, pd.DataFrame):
            if pred.shape[1] != 1:
                raise StrategyError(
                    f"pred is a DataFrame with {pred.shape[1]} columns and prob_col is None; "
                    "name the column explicitly rather than relying on column order"
                )
            pred = pred.iloc[:, 0]
        if not isinstance(pred, pd.Series):
            pred = pd.Series(np.asarray(pred, dtype="float64").ravel())
        values = pred.to_numpy(dtype="float64")
        # Strictly greater: a prediction of exactly zero is no view, not a
        # bullish one.
        signal = np.where(np.isfinite(values) & (values > limit), 1.0, 0.0)
        source = pred

    return pd.Series(signal, index=source.index, name="position")


# --------------------------------------------------------------------------- simulation

def _prepare(predictions: pd.DataFrame, model: str, prob_col: str | None) -> tuple[pd.DataFrame, str, str | None]:
    """Validate the prediction frame and pick the signal column.

    Returns the sorted frame, the name of the column actually used, and that
    name if it is a probability column (``None`` otherwise).
    """
    if not isinstance(predictions, pd.DataFrame):
        raise StrategyError(f"predictions must be a DataFrame, got {type(predictions).__name__}")
    if len(predictions) == 0:
        raise StrategyError("predictions is empty; there is nothing to trade")
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise StrategyError(
            f"predictions must be indexed by a UTC DatetimeIndex, got {type(predictions.index).__name__}"
        )
    if TARGET_COLUMN not in predictions.columns:
        raise StrategyError(
            f"predictions is missing {TARGET_COLUMN!r}; available: {sorted(predictions.columns)}"
        )

    frame = predictions.sort_index()
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()[:3]
        raise StrategyError(
            "predictions index has duplicate timestamps (e.g. "
            f"{[str(d) for d in duplicates]}); each timestamp must describe exactly one "
            "prediction, otherwise a trade would be counted more than once"
        )

    if prob_col is not None:
        signal_col = _resolve_prob_column(frame, prob_col)
    else:
        signal_col = f"{model}_pred"
        if signal_col not in frame.columns:
            available = sorted(c for c in frame.columns if c.endswith("_pred"))
            raise StrategyError(
                f"no prediction column for model {model!r} (expected {signal_col!r}); "
                f"available *_pred columns: {available}"
            )
    return frame, signal_col, (str(prob_col) if prob_col is not None else None)


def _position_series(
    signal: pd.Series,
    threshold: float,
    prob_threshold: float,
    allow_short: bool,
    is_probability: bool,
) -> pd.Series:
    """Build the position series, adding the short leg only when asked for.

    Long/flat is the default because a short needs a borrow, a borrow cost and a
    locate, none of which this research backtest models.  Opting in with
    ``allow_short=True`` is an explicit statement that the caller has thought
    about it.

    ``is_probability`` selects the short rule, and it is passed in rather than
    inferred: a return prediction and a calibrated probability are both plain
    floats, so there is nothing in the values to tell them apart, and guessing
    from the range of ``prob_threshold`` would compare a 0.4% return forecast
    against a 50% probability threshold.  The long side goes through
    :func:`positions_from_predictions` either way, wrapped in a single-column
    frame, so there is one implementation of each rule and not two.
    """
    long_only = positions_from_predictions(
        signal.to_frame("signal"),
        threshold=threshold,
        prob_col="signal" if is_probability else None,
        prob_threshold=prob_threshold,
    )
    if not allow_short:
        return long_only

    values = signal.to_numpy(dtype="float64")
    if is_probability:
        # Calibrated-probability path: symmetric tails around 0.5.
        short_mask = np.isfinite(values) & (values <= 1.0 - float(prob_threshold))
    else:
        short_mask = np.isfinite(values) & (values < -float(threshold))
    # Long wins any overlap, which can only happen if prob_threshold < 0.5.
    combined = np.where(short_mask, -1.0, long_only.to_numpy(dtype="float64"))
    return pd.Series(combined, index=signal.index, name="position")


def _simulate(
    position: np.ndarray,
    target: np.ndarray,
    one_way_rate: float,
) -> dict[str, Any]:
    """Compound the position against the realised target, charging on changes.

    Cost placement is the part worth reading twice:

    * an **opening** fill is charged on the period the position is opened;
    * a **closing** fill is charged on the last period of the hold it closes,
      one step back from where the position change is observed.

    A flip from ``+1`` to ``-1`` is a change of 2.0 and is charged as a close
    plus an open, one unit to each of the two trades.  A position that is never
    closed inside the sample is charged only its opening fill, which is correct:
    the liquidation is not an observation.

    The unit is ``one_way_rate = (fee_bps + slippage_bps) / 10_000``, i.e. half a
    round trip, so a trade that is opened and closed costs exactly
    ``CostModel.round_trip_bps()``.  Charging a *full* round trip per position
    change would cost ``2 x round_trip_bps()`` per trade and quietly flatter
    every result in the study by a factor of two.
    """
    previous = np.concatenate(([0.0], position[:-1])) if position.size else position
    change = position - previous

    opening = np.clip(change, 0.0, None)
    closing = np.clip(-change, 0.0, None)
    closing_on_hold = np.concatenate(([0.0], closing[:-1])) if closing.size else closing

    cost = (opening + closing_on_hold) * one_way_rate
    gross = position * target
    net = gross - cost
    equity = np.cumprod(1.0 + net) if net.size else np.empty(0, dtype="float64")

    return {
        "net": net,
        "gross": gross,
        "cost": cost,
        "equity": equity,
        "turnover": float(np.abs(change).sum()) if change.size else 0.0,
        "cost_drag": float(cost.sum()) if cost.size else 0.0,
    }


# --------------------------------------------------------------------------- statistics

def _sharpe(returns: np.ndarray, annualisation: float) -> float:
    """Horizon-annualised Sharpe, or ``NaN`` for fewer than two or no dispersion.

    ``ddof=1`` so a single observation reports "unknown" rather than a Sharpe of
    infinity.
    """
    if returns.size < 2:
        return NAN
    deviation = float(np.std(returns, ddof=1))
    if not np.isfinite(deviation) or deviation <= 0.0:
        return NAN
    return float(np.mean(returns) / deviation * annualisation)


def _sortino(returns: np.ndarray, annualisation: float) -> float:
    """Horizon-annualised Sortino: excess return over *downside* deviation only.

    The denominator is the semi-deviation over the **whole** series,
    ``sqrt(mean(min(r, 0)^2))``, which is the standard definition.  (V1's
    ``src.evaluation.backtest._sortino`` averages over the losing observations
    only, which discards the fact that flat periods carry no risk either and so
    understates the denominator.)
    """
    if returns.size < 2:
        return NAN
    downside = np.minimum(returns, 0.0)
    deviation = float(np.sqrt(np.mean(downside**2)))
    if not np.isfinite(deviation) or deviation <= 0.0:
        return NAN
    return float(np.mean(returns) / deviation * annualisation)


def _max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough decline of the equity curve, as a **negative** number.

    Signed, not a magnitude: the convention here is that ``max_drawdown <= 0``
    always, so a report cannot print "max drawdown: 3.2%" beside a Sharpe that
    someone forgot to annualise.  ``calmar`` divides by the absolute value.

    The running peak is seeded with the starting capital of 1.0 so a curve that
    opens on a loss records that loss; without the seed the first drawdown would
    be invisible whenever the very first trade lost money.
    """
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0.0, (equity - peak) / peak, 0.0)
    drawdown = drawdown[np.isfinite(drawdown)]
    if drawdown.size == 0:
        return 0.0
    # `dd` is already <= 0 everywhere, so the minimum is the worst drawdown.
    return float(min(float(drawdown.min()), 0.0))


def _cagr(total_return: float, years: float) -> float:
    """Compound the total return over ``years``, or ``NaN`` when undefined.

    ``years <= 0`` (no trades, or a zero-length sample) and a total return at or
    below ``-100%`` (equity exhausted, so the log is undefined) both give
    ``NaN``.  A non-positive total return gives a non-positive CAGR, which is the
    honest sign rather than a clipped zero.
    """
    if not np.isfinite(total_return) or not np.isfinite(years) or years <= 0.0:
        return NAN
    if total_return <= -1.0:
        return NAN
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _hit_rate(net: np.ndarray, position: np.ndarray) -> tuple[float, float, float]:
    """Win rate of the invested periods with Wilson score bounds.

    The bounds come from :func:`src.v3.uncertainty.wilson_interval`, reused
    rather than re-derived: it is the vetted implementation in this package, and
    its own docstring carries the caveat that matters here - a binomial interval
    assumes independent trials, which non-overlapping windows only approximately
    are.  For a hit rate it is a reasonable floor on the uncertainty, not the
    whole of it.
    """
    invested = position != 0.0
    n_trades = int(invested.sum())
    if n_trades == 0:
        return NAN, NAN, NAN
    wins = int((net[invested] > 0.0).sum())
    rate = wins / n_trades
    lower, upper = wilson_interval(wins, n_trades)
    return float(rate), float(lower), float(upper)


# --------------------------------------------------------------------------- engine

def run_strategy(
    predictions: pd.DataFrame,
    horizon: Any,
    model: str,
    mode: str = NON_OVERLAPPING,
    cost: CostModel = CostModel(),
    threshold: float = 0.0,
    allow_short: bool = False,
    prob_col: str | None = None,
    prob_threshold: float = 0.5,
) -> StrategyResult:
    """Run one (horizon, model, mode) strategy over a prediction frame.

    Timing
    ------
    A prediction is made **at** row ``t`` and describes the return from ``t`` to
    ``t + H``, which is exactly what :attr:`HorizonResult.predictions`' ``target``
    column realises.  A position opened at ``t`` therefore earns ``target[t]``,
    with **no additional lag**.  Applying one - the habit carried over from the
    V1 backtest, where the label and the executable fill sat on different candles
    - would shift every trade forward by a bar and silently discard a real edge
    while looking like extra conservatism.

    Modes
    -----
    ``'non_overlapping'`` (default)
        Only the rows selected by :func:`non_overlapping_mask` are traded, so one
        position is open at a time and no window overlaps another.  ``total_return``
        and ``cagr`` are realisable.

    ``'overlapping'``
        Every row is scored.  **Its totals are not achievable performance and
        overstate the honest figure by roughly the horizon in bars.**  The result
        is returned with ``realisable=False`` and
        ``note=OVERLAPPING_WARNING``, and a warning is logged.  Use it to see
        how much signal is available continuously, never as a headline.

    Annualisation
    -------------
    ``years`` is ``horizon_days * n_trades / 365.25`` for the non-overlapping mode
    - the wall-clock time the book was actually committed - and the index span
    divided by ``365.25`` for the overlapping mode, whose rows are the only thing
    that covers the whole period.  CAGR uses those years; Sharpe and Sortino are
    scaled by ``sqrt(365.25 / horizon_days)``, the number of independent
    opportunities per year at this horizon.  A naive hourly annualisation applied
    to a 180-day horizon is off by a factor of roughly 17000 and looks entirely
    plausible in the code that produced it, so the factor is derived from the
    horizon and never from the bar count.

    Parameters
    ----------
    predictions:
        ``HorizonResult.predictions``-shaped frame: a sorted, unique UTC
        ``DatetimeIndex`` with a ``target`` column and ``f'{model}_pred'``.
    horizon:
        A ``Horizon``, ``"30d"``, a ``Timedelta``, or a number of hours.
    model:
        Model name; selects the ``f'{model}_pred'`` column.
    mode:
        ``'non_overlapping'`` or ``'overlapping'``.
    cost:
        Per-side frictions, charged on position changes only.
    threshold:
        Long-side threshold on the prediction.
    allow_short:
        Add the short leg.  ``False`` (long/flat) by default.
    prob_col:
        Trade a calibrated probability column instead of the return prediction.
    prob_threshold:
        Probability at or above which to be long.

    Returns
    -------
    StrategyResult
    """
    if mode not in MODES:
        raise StrategyError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if not isinstance(cost, CostModel):
        raise StrategyError(f"cost must be a CostModel, got {type(cost).__name__}")

    delta = horizon_delta(horizon)
    horizon_days = delta / pd.Timedelta(days=1)
    annualisation = float(np.sqrt(_periods_per_year(horizon_days)))
    frame, signal_col, probability_col = _prepare(predictions, model, prob_col)

    entries = (
        non_overlapping_mask(frame.index, delta).to_numpy()
        if mode == NON_OVERLAPPING
        else np.ones(len(frame), dtype=bool)
    )
    # Rows whose forward window is not yet complete arrive as NaN targets; they
    # are unscoreable, so they leave the equity curve entirely.  A NaN *signal*
    # is the opposite: `positions_from_predictions` reads it as a flat position,
    # which is a decision, and dropping the row would hide that close from the
    # turnover accounting.
    usable = entries & np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype="float64"))
    evaluated = frame.loc[usable]
    if len(evaluated) == 0:
        logger.warning(
            "run_strategy(%s, %s, mode=%s): no evaluable rows on the %s grid",
            model, horizon_label(horizon), mode, mode,
        )
        return _empty_result(horizon_label(horizon), model, mode)

    position = _position_series(
        evaluated[signal_col], threshold, prob_threshold, allow_short, probability_col is not None
    )
    book = _simulate(
        position.to_numpy(dtype="float64"),
        evaluated[TARGET_COLUMN].to_numpy(dtype="float64"),
        cost.one_way_bps() / BPS,
    )

    net = book["net"]
    equity = book["equity"]
    n_trades = int((position.to_numpy(dtype="float64") != 0.0).sum())
    exposure = float(n_trades / len(evaluated)) if len(evaluated) else 0.0

    span_days = (frame.index[-1] - frame.index[0]) / pd.Timedelta(days=1)
    years = (
        horizon_days * n_trades / DAYS_PER_YEAR
        if mode == NON_OVERLAPPING
        else float(span_days) / DAYS_PER_YEAR
    )

    if n_trades == 0:
        # Evaluable rows exist but the book never opened.  A compounded return of
        # exactly 1.0 is a real number, and reporting it as `total_return=0.0`
        # would tell the reader the strategy was riskless rather than absent.
        logger.info(
            "run_strategy(%s, %s, mode=%s): the signal never cleared the threshold, "
            "so there is no performance to report",
            model, horizon_label(horizon), mode,
        )
        return _empty_result(horizon_label(horizon), model, mode, equity=equity, index=evaluated.index)

    total_return = float(equity[-1] - 1.0)
    max_drawdown = _max_drawdown(equity)
    hit_rate, ci_lower, ci_upper = _hit_rate(net, position.to_numpy(dtype="float64"))
    cagr = _cagr(total_return, years)

    realisable = mode == NON_OVERLAPPING
    if not realisable:
        logger.warning(
            "%s (model=%s, horizon=%s, n_trades=%d)", OVERLAPPING_WARNING, model, horizon_label(horizon), n_trades
        )

    return StrategyResult(
        horizon=horizon_label(horizon),
        model=str(model),
        mode=mode,
        n_trades=n_trades,
        exposure=exposure,
        total_return=total_return,
        cagr=cagr,
        sharpe=_sharpe(net, annualisation),
        sortino=_sortino(net, annualisation),
        max_drawdown=max_drawdown,
        calmar=(float(cagr / abs(max_drawdown)) if np.isfinite(cagr) and max_drawdown < 0.0 else NAN),
        hit_rate=hit_rate,
        hit_rate_ci_lower=ci_lower,
        hit_rate_ci_upper=ci_upper,
        turnover=book["turnover"],
        cost_drag=book["cost_drag"],
        equity=pd.Series(equity, index=evaluated.index, name="equity"),
        years=float(years),
        realisable=realisable,
        note="" if realisable else OVERLAPPING_WARNING,
    )


def _empty_result(
    horizon: str,
    model: str,
    mode: str,
    equity: np.ndarray | None = None,
    index: pd.Index | None = None,
) -> StrategyResult:
    """A result for "nothing was traded", which is ``NaN``, not a flat 0.0.

    ``exposure``, ``turnover``, ``cost_drag`` and ``max_drawdown`` are ``0.0``
    because zero is the truth about them; everything that would be a *performance*
    claim is ``NaN`` so it can never be read as one.  The equity curve is
    preserved when there is one, flat at the starting capital, so a chart is
    still drawable - it just never moved.
    """
    curve = (
        pd.Series(dtype="float64", name="equity")
        if equity is None
        else pd.Series(equity, index=index, name="equity")
    )
    return StrategyResult(
        horizon=horizon,
        model=model,
        mode=mode,
        n_trades=0,
        exposure=0.0,
        total_return=NAN,
        cagr=NAN,
        sharpe=NAN,
        sortino=NAN,
        max_drawdown=0.0,
        calmar=NAN,
        hit_rate=NAN,
        hit_rate_ci_lower=NAN,
        hit_rate_ci_upper=NAN,
        turnover=0.0,
        cost_drag=0.0,
        equity=curve,
        years=0.0,
        realisable=mode == NON_OVERLAPPING,
        note="" if mode == NON_OVERLAPPING else OVERLAPPING_WARNING,
    )


# --------------------------------------------------------------------------- sweep

def sweep_strategies(
    predictions: pd.DataFrame,
    horizon: Any,
    models: Iterable[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """One flattened row per ``(model, mode)`` pair.

    Both modes are run for every model by default, and they belong in the same
    table precisely so the reader can see the gap between them: the
    ``overlapping`` row is stamped ``realisable=0`` with
    :data:`OVERLAPPING_WARNING` in its ``note``, and the ratio of the two
    ``total_return`` cells is the double-counting factor.

    Extra keyword arguments are forwarded to :func:`run_strategy`.  ``modes``
    (or a single ``mode``) may be passed to restrict the sweep; ``cost``,
    ``threshold``, ``allow_short``, ``prob_col`` and ``prob_threshold`` behave
    exactly as they do there.

    Returns
    -------
    pandas.DataFrame
        One row per pair, columns in :data:`RESULT_COLUMNS` order.  Empty (but
        correctly columned) when ``models`` is empty.
    """
    modes: Sequence[str]
    if "modes" in kwargs:
        modes = tuple(kwargs.pop("modes"))
    elif "mode" in kwargs:
        single = kwargs.pop("mode")
        modes = (single,) if isinstance(single, str) else tuple(single)
    else:
        modes = MODES
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise StrategyError(f"unknown mode(s) {unknown}; allowed: {list(MODES)}")

    rows = [
        run_strategy(predictions, horizon, model, mode=mode, **kwargs).to_dict()
        for model in models
        for mode in modes
    ]
    if not rows:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


# --------------------------------------------------------------------------- baselines

def _baseline_row(
    frame: pd.DataFrame,
    delta: pd.Timedelta,
    cost: CostModel,
    label: str,
    note: str,
) -> dict[str, Any]:
    """Statistics for a fixed all-long book on the non-overlapping grid.

    Both baselines are always evaluated on the non-overlapping grid, including
    when the model row under test came from ``overlapping`` mode.  Compounding
    ``target`` row by row would be the same double-counting the module exists to
    prevent, and a benchmark is exactly the number nobody would check.
    """
    horizon_days = delta / pd.Timedelta(days=1)
    annualisation = float(np.sqrt(_periods_per_year(horizon_days)))

    mask = non_overlapping_mask(frame.index, delta).to_numpy()
    target = frame[TARGET_COLUMN].to_numpy(dtype="float64")
    usable = mask & np.isfinite(target)
    if not usable.any():
        return {
            "name": label, "n_trades": 0, "total_return": NAN, "sharpe": NAN,
            "max_drawdown": 0.0, "hit_rate": NAN, "realisable": 1, "note": note,
        }

    index = frame.index[usable]
    position = np.ones(int(usable.sum()), dtype="float64")
    book = _simulate(position, target[usable], cost.one_way_bps() / BPS)
    net, equity = book["net"], book["equity"]
    n_trades = int(net.size)
    years = horizon_days * n_trades / DAYS_PER_YEAR
    total_return = float(equity[-1] - 1.0)
    hit_rate, _, _ = _hit_rate(net, position)

    return {
        "name": label,
        "n_trades": n_trades,
        "total_return": total_return,
        "sharpe": _sharpe(net, annualisation),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": hit_rate,
        "realisable": 1,
        "note": note,
    }


def compare_to_baselines(
    result: StrategyResult,
    predictions: pd.DataFrame,
    horizon: Any,
    model: str,
    cost: CostModel = CostModel(),
) -> pd.DataFrame:
    """The model against ``always_long`` and ``buy_and_hold``.

    A model that trails simply holding the asset has not earned its complexity,
    and the only way to know is to put the two on one table with identical cost
    assumptions and an identical evaluation grid.

    Rows
    ----
    ``'model'``
        ``result`` itself, carrying its own mode.  If that mode is
        ``overlapping`` the row is stamped ``realisable=0`` and repeats
        :data:`OVERLAPPING_WARNING`, so this table cannot be read as approving
        of an overlapping total.
    ``'always_long'`` and ``'buy_and_hold'``
        A continuously-held long position over the non-overlapping windows, i.e.
        the compounded path ``close[t_last + H] / close[t_first] - 1``.  The two
        coincide by construction - on that grid the position is never flat - and
        they are kept as separate rows because *a disagreement between them
        would mean the accounting is wrong*, which is worth a line in the table.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``name``, columns ``n_trades, total_return, sharpe,
        max_drawdown, hit_rate, realisable, note``.  Always exactly three rows.
    """
    if not isinstance(result, StrategyResult):
        raise StrategyError(f"result must be a StrategyResult, got {type(result).__name__}")
    delta = horizon_delta(horizon)
    frame, _, _ = _prepare(predictions, model, None)

    model_note = result.note or (
        f"mode={result.mode}" if result.mode == NON_OVERLAPPING else OVERLAPPING_WARNING
    )
    rows = [
        {
            "name": "model",
            "n_trades": int(result.n_trades),
            "total_return": float(result.total_return),
            "sharpe": float(result.sharpe),
            "max_drawdown": float(result.max_drawdown),
            "hit_rate": float(result.hit_rate),
            "realisable": int(result.realisable),
            "note": model_note,
        },
        _baseline_row(frame, delta, cost, "always_long", "continuous long over non-overlapping windows"),
        _baseline_row(frame, delta, cost, "buy_and_hold", "compounded price path over the same windows"),
    ]
    table = pd.DataFrame(rows).set_index("name")
    table.index.name = "name"
    return table


# --------------------------------------------------------------------------- uncertainty

def bootstrap_sharpe(
    returns: Any,
    n_bootstrap: int = 1_000,
    seed: int = 42,
    block_size: int | None = None,
) -> dict[str, float]:
    """Block-bootstrap the Sharpe ratio of a return series.

    A Sharpe ratio computed on autocorrelated series is a point estimate with no
    error bar around it, which is how a Sharpe of 0.9 gets quoted as if it were
    0.9 plus or minus nothing.  Resampling **individual** observations would not
    fix that - it would make the interval far *too narrow*, destroying the serial
    dependence and manufacturing precision that the data does not have.

    So the resampling unit is a block: :func:`bootstrap_sharpe` uses the
    stationary (Politis-Romano) bootstrap, where each new observation starts a
    fresh block with probability ``1 / block_size`` and block lengths are
    geometric with that same mean.  The mean is right where the variance is
    minimised, blocks are drawn circularly, and the resulting series is
    stationary - so the interval is honest about dependence without assuming the
    series is a fixed block pattern.

    The returned ``sharpe`` is the **per-period** ratio ``mean / std``.  The
    function is horizon-agnostic by design, so annualise it yourself with
    ``sqrt(periods_per_year)`` from the horizon - :func:`run_strategy` already
    does.  Quoting an unannualised number next to an annualised one is a units
    mistake that makes a 30-day book look identical to an hourly one.

    Default ``block_size`` is ``round(n ** (1/3))``, the standard rule of thumb
    for a stationary bootstrap with unknown dependence length.  For V3 that is
    almost certainly too short - overlapping forward windows at ``H = 30d``
    carry dependence over hundreds of bars - so pass ``block_size=horizon_bars``
    when the series is an overlapping per-row return stream.

    Returns
    -------
    dict[str, float]
        ``sharpe``, ``ci_lower`` (2.5th percentile), ``ci_upper`` (97.5th) and
        ``p_sharpe_gt_zero``.  All four are ``NaN`` when the series has fewer
        than two observations or no dispersion - never ``0.0``, which would read
        as "the edge is exactly zero" rather than "there is nothing to measure".

    Raises
    ------
    ValueError
        If ``n_bootstrap < 100``.  A percentile interval from fewer than 100
        resamples is not an interval, and the guard is a floor rather than a
        recommendation.
    """
    if int(n_bootstrap) < 100:
        raise ValueError(
            f"n_bootstrap must be >= 100 for a percentile interval to mean anything, "
            f"got {n_bootstrap!r}"
        )

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        values = returns.to_numpy(dtype="float64").ravel()
    else:
        values = np.asarray(returns, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)

    empty = {"sharpe": NAN, "ci_lower": NAN, "ci_upper": NAN, "p_sharpe_gt_zero": NAN}
    if n < 2:
        logger.warning("bootstrap_sharpe received %d usable observation(s); returning NaN", n)
        return dict(empty)

    dispersion = float(np.std(values, ddof=1))
    if not np.isfinite(dispersion) or dispersion <= 0.0:
        logger.warning("bootstrap_sharpe received a constant series; Sharpe is undefined")
        return dict(empty)

    span = max(1, int(round(n ** (1.0 / 3.0)))) if block_size is None else int(block_size)
    if span < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}")
    span = min(span, n)
    logger.debug(
        "bootstrap_sharpe: n=%d block_size=%d n_bootstrap=%d seed=%s", n, span, int(n_bootstrap), seed
    )

    point = float(np.mean(values) / dispersion)
    rng = np.random.default_rng(int(seed))
    start_probability = 1.0 / span

    # Resampled in a loop rather than as one (n_bootstrap, n) matrix: a 1000x40k
    # index matrix is 320 MB, and a report that allocates that to produce four
    # numbers is a report that will be killed on a laptop.
    replicates = np.empty(int(n_bootstrap), dtype="float64")
    offsets = np.arange(span)
    for replicate in range(int(n_bootstrap)):
        pieces: list[np.ndarray] = []
        collected = 0
        while collected < n:
            start = int(rng.integers(0, n))
            length = int(min(rng.geometric(start_probability), n - collected))
            pieces.append(values[(start + offsets[:length]) % n])
            collected += length
        sample = np.concatenate(pieces)
        sample_std = float(np.std(sample, ddof=1))
        replicates[replicate] = (
            float(np.mean(sample) / sample_std) if np.isfinite(sample_std) and sample_std > 0.0 else np.nan
        )

    usable = replicates[np.isfinite(replicates)]
    if usable.size == 0:
        logger.warning("bootstrap_sharpe produced no finite replicate; returning NaN")
        return dict(empty)

    return {
        "sharpe": point,
        "ci_lower": float(np.percentile(usable, 2.5)),
        "ci_upper": float(np.percentile(usable, 97.5)),
        "p_sharpe_gt_zero": float((usable > 0.0).mean()),
    }
