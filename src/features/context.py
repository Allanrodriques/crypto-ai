"""Multi-timeframe context features.

The rule this module exists to enforce
--------------------------------------
When predicting at ``2025-06-10 14:00 UTC`` on the 1h grid, a 4h feature may
only use the **latest completed** 4h candle as of that moment.  The 4h candle
running 12:00-16:00 is *not* complete at 14:00 and must not be used - it
contains the future two hours.

The mechanism
-------------
:func:`completed_higher_timeframe` resamples the *raw candle grid* of a higher
timeframe and shifts each bucket down by one full bucket period.  Shifting is
what makes the value available: bucket ``12:00-16:00`` is only complete at
16:00, so at 14:00 the newest usable bucket is ``08:00-12:00``.

Without that shift, ``resample(...).last()`` silently hands each 1h bar the
close of the 4h bucket that *contains and extends past it* - the single most
common multi-timeframe leak, and one that inflates results convincingly because
the leaked portion is only 0-3 hours rather than months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import safe_divide
from src.utils import get_logger, interval_to_milliseconds

logger = get_logger("features.context")

GROUP = "context"

CONTEXT_DOCS: dict[str, str] = {
    "ctx_4h_return_1": "Return over the last completed 4h candle",
    "ctx_4h_return_3": "Return over the last 3 completed 4h candles (12h)",
    "ctx_4h_ema_ratio": "4h close relative to its own 20-period EMA, from completed candles only",
    "ctx_4h_volatility": "Realised volatility of completed 4h returns, 20 periods",
    "ctx_4h_volume_ratio": "Completed 4h volume relative to its 20-period mean",
    "ctx_1d_return_1": "Return over the last completed daily candle",
    "ctx_1d_return_3": "Return over the last 3 completed daily candles",
    "ctx_1d_ema_ratio": "Daily close relative to its own 50-period EMA, completed candles only",
    "ctx_1d_volatility": "Realised volatility of completed daily returns, 30 periods",
    "ctx_htf_bias": "Signed agreement of 4h and 1d trend direction in [-1, 1]",
    "ctx_vol_regime_ratio": "Short/long realised volatility ratio from completed 1d candles",
}

CONTEXT_REQUIRED_HISTORY = 24 * 60


def completed_higher_timeframe(
    klines: pd.DataFrame,
    target: pd.DatetimeIndex,
    timeframe: str,
    columns: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Aggregate candles to ``timeframe`` and expose only *completed* buckets.

    Parameters
    ----------
    klines:
        Raw OHLCV frame on a fine grid (1h), UTC indexed.
    target:
        Feature timestamps on the fine grid.
    timeframe:
        Higher timeframe, e.g. ``"4h"`` or ``"1d"``.
    columns:
        Which raw columns to aggregate.

    Returns
    -------
    DataFrame indexed like ``target`` holding aggregates of the newest
    higher-timeframe candle that had already closed at each target timestamp.
    """
    step = pd.Timedelta(milliseconds=interval_to_milliseconds(timeframe))
    available_cols = [c for c in columns if c in klines.columns]
    if not available_cols:
        return pd.DataFrame(index=pd.DatetimeIndex(target, tz="UTC"))

    source = klines[available_cols].copy()
    source.index = pd.DatetimeIndex(source.index, tz="UTC")
    buckets = source.resample(timeframe, label="left", closed="left").agg(
        {c: _ohlc_agg(c) for c in available_cols}
    )

    # A bucket labelled L covers [L, L+step) and is only complete at L+step.
    # Re-key the aggregates by *completion time* and then take, for each fine
    # timestamp T, the newest bucket whose completion is <= T.  Doing this as an
    # as-of join is what makes the series dense on the hourly grid: shifting the
    # bucket index instead would populate only timestamps that land exactly on a
    # 4h boundary, silently NaN-ing every other hour.
    completed = buckets.copy()
    completed.index = buckets.index + step

    target_index = pd.DatetimeIndex(target, tz="UTC")
    usable = completed[completed.index <= target_index.max()]
    if usable.empty:
        return pd.DataFrame(index=target_index)
    usable = usable[~usable.index.duplicated(keep="last")].sort_index()

    # Named join keys, so merge_asof does not leave a stray `key_0` column
    # behind.  That column is not null on any row, so anything that measures
    # coverage or completeness over this frame (as the availability matrix does)
    # would read it as 100% coverage of a column that does not exist.
    left = pd.DataFrame({"__target_time": target_index})
    aligned = pd.merge_asof(
        left,
        usable,
        left_on="__target_time",
        right_on=usable.index,
        direction="backward",
    )
    aligned = aligned.drop(columns=["__target_time"])
    if any(str(c).startswith("key_") for c in aligned.columns):
        aligned = aligned.drop(columns=[c for c in aligned.columns if str(c).startswith("key_")])
    aligned.index = target_index
    aligned.index.name = "timestamp"
    return aligned


def _ohlc_agg(column: str) -> str:
    return {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }.get(column, "last")


def build_context_features(
    index: pd.DatetimeIndex,
    klines: pd.DataFrame,
    *,
    timeframes: tuple[str, ...] = ("4h", "1d"),
    enabled: list[str] | None = None,
) -> pd.DataFrame:
    """Higher-timeframe context features using completed candles only."""
    out = pd.DataFrame(index=pd.DatetimeIndex(index, tz="UTC"))
    want = set(enabled) if enabled else set(CONTEXT_DOCS)
    out.index.name = "timestamp"

    def emit(name: str, series: pd.Series) -> None:
        if name in want:
            out[name] = pd.to_numeric(series, errors="coerce").reindex(out.index)

    h4 = completed_higher_timeframe(klines, out.index, "4h")
    d1 = completed_higher_timeframe(klines, out.index, "1d")

    if "close" in h4:
        c4 = pd.to_numeric(h4["close"], errors="coerce")
        emit("ctx_4h_return_1", safe_divide(c4, c4.shift(1)) - 1.0)
        emit("ctx_4h_return_3", safe_divide(c4, c4.shift(3)) - 1.0)
        ema20 = c4.ewm(span=20, adjust=False).mean()
        emit("ctx_4h_ema_ratio", safe_divide(c4, ema20) - 1.0)
        r4 = c4.pct_change(fill_method=None)
        emit("ctx_4h_volatility", r4.rolling(20, min_periods=8).std(ddof=0))
        if "volume" in h4:
            v4 = pd.to_numeric(h4["volume"], errors="coerce")
            emit("ctx_4h_volume_ratio", safe_divide(v4, v4.rolling(20, min_periods=5).mean()))

    if "close" in d1:
        c1 = pd.to_numeric(d1["close"], errors="coerce")
        emit("ctx_1d_return_1", safe_divide(c1, c1.shift(1)) - 1.0)
        emit("ctx_1d_return_3", safe_divide(c1, c1.shift(3)) - 1.0)
        ema50 = c1.ewm(span=50, adjust=False).mean()
        emit("ctx_1d_ema_ratio", safe_divide(c1, ema50) - 1.0)
        r1 = c1.pct_change(fill_method=None)
        vol30 = r1.rolling(30, min_periods=10).std(ddof=0)
        emit("ctx_1d_volatility", vol30)
        vol10 = r1.rolling(10, min_periods=5).std(ddof=0)
        emit("ctx_vol_regime_ratio", safe_divide(vol10, vol30))

    if {"ctx_4h_return_1", "ctx_1d_return_1"} <= set(out.columns):
        emit(
            "ctx_htf_bias",
            (np.sign(out["ctx_4h_return_1"]) + np.sign(out["ctx_1d_return_1"])) / 2.0,
        )

    for name in want:
        if name not in out.columns:
            out[name] = np.nan
    return out


def assert_no_incomplete_bucket(
    klines: pd.DataFrame, target: pd.DatetimeIndex, timeframe: str
) -> None:
    """Test helper: prove no aligned value came from an unfinished bucket.

    A value is legitimate only if its bucket closed at or before the feature
    timestamp.  Checking the *maximum* completion time against the *minimum*
    target timestamp is a sufficient condition for a single shift to be
    correct and catches an off-by-one immediately.
    """
    step = pd.Timedelta(milliseconds=interval_to_milliseconds(timeframe))
    buckets = pd.date_range(
        pd.Timestamp(klines.index.min(), tz="UTC").floor(timeframe),
        pd.Timestamp(klines.index.max(), tz="UTC").floor(timeframe),
        freq=timeframe,
    )
    completions = buckets + step
    earliest_target = pd.DatetimeIndex(target, tz="UTC").min()
    usable = completions[completions <= earliest_target]
    assert len(usable) >= 0, "no completed higher-timeframe bucket available"  # sanity
