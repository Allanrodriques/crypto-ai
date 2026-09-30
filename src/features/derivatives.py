"""Derivatives / futures-positioning features (EXP-01).

Sources
-------
* funding rates - point-in-time settlements every 8h
* perpetual futures klines - price, volume
* spot klines - for the basis

What is deliberately absent
---------------------------
**Open interest.**  ``/futures/data/openInterestHist`` refuses ``startTime``
and serves only a rolling ~30 day window, so it cannot be joined to a
multi-year backtest.  Reconstructing it would mean inventing data, which this
project does not do.  Its absence is recorded in the availability matrix and
in the experiment report as a known limitation.

Note also that long/short account ratio and taker buy/sell ratio share the same
~30 day cap and are excluded for the same reason.

Every feature here is causal by construction: values are taken from the row at
or before ``t`` and only ever look backwards.  Funding is a point-in-time
event, so it is read at its own timestamp with no lag; see
:mod:`src.data.sources.funding` for why that is correct rather than lax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import rolling_zscore, safe_divide

GROUP = "derivatives"

#: Documentation for every feature this module can emit.
DERIVATIVES_DOCS: dict[str, str] = {
    "funding_rate": "Most recent settled 8h perpetual funding rate (point-in-time, no lag)",
    "funding_rate_change": "Change in the funding rate since the previous settlement",
    "funding_rate_mean_3": "Mean funding rate over the last 3 settlements (~24h)",
    "funding_rate_mean_10": "Mean funding rate over the last 10 settlements (~80h)",
    "funding_zscore_60d": "Funding rate z-score over a trailing 60d window of settlements",
    "funding_rate_zscore_20": "Funding rate z-score over the last 20 settlements (~7d)",
    "funding_cumulative_24h": "Sum of funding rates over the last 3 settlements",
    "funding_positive_share_60d": "Share of the last 60d settlements with positive funding",
    "funding_streak": "Consecutive settlements at the current sign, signed",
    "funding_rate_abs": "Absolute funding rate - the cost of carry regardless of side",
    "futures_price": "Perpetual futures close at t",
    "futures_volume": "Perpetual futures base-asset volume over candle t",
    "futures_volume_ratio": "Futures volume divided by its 20-period mean",
    "futures_price_return_1h": "1h return of the perpetual futures close",
    "futures_spot_basis": "futures_close / spot_close - 1 (annualised not attempted)",
    "futures_spot_basis_mean_24": "Mean basis over the last 24h",
    "futures_basis_zscore_60d": "Basis z-score over a trailing 60d window",
    "futures_spot_volume_ratio": "Futures volume / spot volume over the same candle",
    "futures_volume_change": "Change in futures volume over the previous candle",
    "futures_trades_ratio": "Futures trade count / spot trade count over the same candle",
    "futures_taker_buy_ratio": "Share of futures volume bought aggressively (taker buy)",
    "futures_basis_change_1h": "Change in the futures-spot basis over 1h",
    "futures_spot_basis_mean_7d": "Mean basis over the last 7d of hourly bars",
}

#: Longest trailing window any feature needs, in hourly candles.
DERIVATIVES_REQUIRED_HISTORY = 24 * 60 + 24


def build_derivatives_features(
    index: pd.DatetimeIndex,
    funding: pd.DataFrame,
    futures: pd.DataFrame,
    spot: pd.DataFrame,
    *,
    enabled: list[str] | None = None,
) -> pd.DataFrame:
    """Compute derivatives features on the hourly feature grid.

    Parameters
    ----------
    index:
        Feature timestamps (hourly, UTC).
    funding:
        Funding frame already aligned to ``index`` by
        :func:`src.alignment.availability.align_asof` (carries ``funding_rate``).
    futures:
        Futures frame already aligned to ``index``.
    spot:
        Spot OHLCV frame already aligned to ``index``.

    All inputs are expected to be *pre-aligned*, i.e. already safe to use at
    each row.  This function performs no forward filling of its own.
    """
    out = pd.DataFrame(index=pd.DatetimeIndex(index, tz="UTC"))
    want = set(enabled) if enabled else set(DERIVATIVES_DOCS)
    out.index.name = "timestamp"

    def emit(name: str, series: pd.Series) -> None:
        if name in want:
            out[name] = pd.to_numeric(series, errors="coerce").reindex(out.index)

    # ------------------------------------------------------------- funding
    funding_rate = _column(funding, "funding_rate", out.index)
    if any(n.startswith("funding_") for n in want):
        # 8h settlements on an hourly grid: 3 bars per settlement period.
        emit("funding_rate", funding_rate)
        emit("funding_rate_abs", funding_rate.abs())
        emit(
            "funding_rate_change",
            funding_rate - funding_rate.shift(3),
        )
        emit("funding_rate_mean_3", funding_rate.rolling(3, min_periods=1).mean())
        emit("funding_rate_mean_10", funding_rate.rolling(10, min_periods=1).mean())
        emit("funding_cumulative_24h", funding_rate.rolling(3, min_periods=1).sum())
        emit(
            "funding_rate_zscore_20",
            rolling_zscore(funding_rate, 20, min_periods=20),
        )
        emit("funding_zscore_60d", rolling_zscore(funding_rate, 60 * 3, min_periods=60))
        emit(
            "funding_positive_share_60d",
            (funding_rate > 0).rolling(60 * 3, min_periods=60).mean(),
        )
        emit("funding_streak", _signed_streak(funding_rate, 3))

    # ------------------------------------------------------------- futures
    fut_close = _column(futures, "close", out.index)
    fut_volume = _column(futures, "futures_volume", out.index)
    fut_trades = _column(futures, "trades", out.index)
    fut_taker_buy = _column(futures, "taker_buy_volume", out.index)

    emit("futures_price", fut_close)
    emit("futures_volume", fut_volume)
    emit("futures_volume_ratio", safe_divide(fut_volume, fut_volume.rolling(20, min_periods=20).mean()))
    emit("futures_volume_change", fut_volume - fut_volume.shift(1))
    emit(
        "futures_price_return_1h",
        safe_divide(fut_close, fut_close.shift(1)) - 1.0,
    )
    emit("futures_taker_buy_ratio", safe_divide(fut_taker_buy, fut_volume))

    # ---------------------------------------------------------------- basis
    spot_close = _column(spot, "close", out.index)
    spot_volume = _column(spot, "volume", out.index)
    spot_trades = _column(spot, "trades", out.index)
    basis = safe_divide(fut_close, spot_close) - 1.0
    emit("futures_spot_basis", basis)
    emit("futures_spot_basis_mean_24", basis.rolling(24, min_periods=6).mean())
    emit("futures_basis_zscore_60d", rolling_zscore(basis, 24 * 60, min_periods=24 * 60))
    emit("futures_basis_change_1h", basis - basis.shift(1))
    emit("futures_spot_volume_ratio", safe_divide(fut_volume, spot_volume))
    emit("futures_trades_ratio", safe_divide(fut_trades, spot_trades))
    emit("futures_spot_basis_mean_7d", basis.rolling(24 * 7, min_periods=24).mean())

    return out


# --------------------------------------------------------------------- helpers

def _column(frame: pd.DataFrame | None, name: str, index: pd.DatetimeIndex) -> pd.Series:
    if frame is None or name not in getattr(frame, "columns", []):
        return pd.Series(np.nan, index=index, dtype="float64")
    series = pd.to_numeric(frame[name], errors="coerce")
    series.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return series.reindex(index)


def _signed_streak(series: pd.Series, per_period: int) -> pd.Series:
    """Length of the current run of same-signed settlements, signed by that sign.

    Built with positional numpy arrays on purpose.  An earlier version built the
    run length as a Series with a RangeIndex and then re-labelled it with
    ``pd.Series(values, index=series.index)`` - but that constructor *aligns by
    label*, so against a DatetimeIndex it silently produced an all-NaN column.
    """
    values = series.to_numpy(dtype="float64")
    sign = np.sign(values)
    observed = ~np.isnan(values)

    change = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        # Only a genuine change of sign starts a new run; NaN gaps are skipped
        # rather than treated as a sign change.
        both = observed[1:] & observed[:-1]
        change[1:] = np.where(both, sign[1:] != sign[:-1], False)

    group = np.cumsum(change)
    run_index = np.zeros(len(values), dtype="int64")
    for value in np.unique(group):
        mask = group == value
        run_index[mask] = np.arange(1, int(mask.sum()) + 1)

    out = sign * run_index
    out[~observed] = np.nan
    return pd.Series(out, index=series.index)
