"""Market-microstructure features (EXP-03).

What is available historically
------------------------------
Binance's kline rows carry two microstructure observables that exist for the
*entire* history:

* ``trades`` - the number of trades in the candle
* ``taker_buy_volume`` / ``taker_buy_quote_volume`` - the volume that hit the
  **ask** (aggressive buyers) versus the remainder, which hit the **bid**
  (aggressive sellers)

Everything in this module is derived from those two fields plus price.  No extra
requests are needed, and coverage is identical to the OHLCV series, which makes
the micro/OHLCV comparison clean.

What is NOT available, and therefore NOT built
---------------------------------------------
**Order-book state** - bid/ask spread, depth imbalance, top-of-book size - has
no historical endpoint on Binance.  ``/api/v3/depth`` returns a *current*
snapshot only; it accepts no historical parameter.  Sampling it during a
backfill and attaching those readings to 2022 rows would be fabricating
microstructure that was never observed, so this project does not do it.

Consequently the spec's ``bid_ask_spread``, ``bid_volume``, ``ask_volume``,
``order_book_imbalance`` and the ``order_imbalance_{1,5,15}m`` family are
**absent by design**.  Their absence is recorded in
``reports/data_availability.csv`` and called out in the experiment report as a
known limitation, and it is the reason EXP-03 measures trade flow rather than
book shape.

Derived from trade flow
-----------------------
The aggressive buy share is a genuine order-flow imbalance: it is the fraction
of executed volume that crossed the spread upward versus downward.  Rolling it
over 1m/5m/15m-equivalent windows gives multi-scale flow imbalance without
inventing a book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import rolling_zscore, safe_divide

GROUP = "microstructure"

MICROSTRUCTURE_DOCS: dict[str, str] = {
    "trade_count": "Number of trades in candle t (from the kline row)",
    "trade_count_ratio": "trade_count divided by its 20-period mean",
    "trade_count_zscore_60d": "Trade count z-score over a trailing 60d window",
    "trade_intensity": "Trades per unit of quote volume (turnover-normalised activity)",
    "avg_trade_size": "Mean quote volume per trade - how large participants are trading",
    "avg_trade_size_ratio": "avg_trade_size relative to its 20-period mean",
    "taker_buy_volume": "Base-asset volume bought aggressively (taker buy)",
    "taker_buy_quote_volume": "Quote volume bought aggressively",
    "aggressive_buy_ratio": "Share of volume that crossed the spread upward",
    "aggressive_sell_ratio": "Share of volume that crossed the spread downward",
    "buy_sell_ratio": "Aggressive buy volume divided by aggressive sell volume",
    "volume_imbalance": "Signed volume imbalance in [-1, 1]",
    "order_imbalance_1h": "Volume imbalance over the trailing 1 hour",
    "order_imbalance_5h": "Volume imbalance over the trailing 5 hours",
    "order_imbalance_15h": "Volume imbalance over the trailing 15 hours",
    "order_imbalance_zscore_60d": "1h volume imbalance z-score over a trailing 60d window",
    "taker_buy_ratio_mean_5": "5-hour mean of the aggressive buy ratio",
    "taker_buy_ratio_mean_15": "15-hour mean of the aggressive buy ratio",
    "taker_buy_ratio_change_1h": "Change in the aggressive buy ratio over 1h",
    "quote_volume_zscore_60d": "Quote volume z-score over a trailing 60d window",
    "volume_per_trade_cv_20": "Coefficient of variation of avg trade size over 20 bars",
    "flow_price_divergence": "Sign agreement between order flow and price change",
}

MICROSTRUCTURE_REQUIRED_HISTORY = 24 * 60 + 24


def build_microstructure_features(
    index: pd.DatetimeIndex,
    klines: pd.DataFrame,
    *,
    enabled: list[str] | None = None,
) -> pd.DataFrame:
    """Compute trade-flow microstructure features on the hourly grid.

    ``klines`` must carry the raw exchange fields ``volume``,
    ``quote_volume``, ``trades`` and ``taker_buy_volume`` - which the V1
    downloader already persists - aligned to ``index``.
    """
    out = pd.DataFrame(index=pd.DatetimeIndex(index, tz="UTC"))
    want = set(enabled) if enabled else set(MICROSTRUCTURE_DOCS)
    out.index.name = "timestamp"

    def emit(name: str, series: pd.Series) -> None:
        if name in want:
            out[name] = pd.to_numeric(series, errors="coerce").reindex(out.index)

    trades = _col(klines, "trades", out.index)
    volume = _col(klines, "volume", out.index)
    quote_volume = _col(klines, "quote_volume", out.index)
    taker_buy = _col(klines, "taker_buy_volume", out.index)
    taker_buy_quote = _col(klines, "taker_buy_quote_volume", out.index)
    close = _col(klines, "close", out.index)

    if trades.notna().sum() == 0 and taker_buy.notna().sum() == 0:
        for name in want:
            out[name] = np.nan
        return out

    # Aggressive sell volume is the remainder: everything that did not cross
    # the spread upward crossed it downward.
    taker_sell = (volume - taker_buy).where(volume.notna() & taker_buy.notna())

    emit("trade_count", trades)
    emit("trade_count_ratio", safe_divide(trades, trades.rolling(20, min_periods=5).mean()))
    emit("trade_count_zscore_60d", rolling_zscore(trades, 24 * 60, min_periods=24 * 20))
    emit("trade_intensity", safe_divide(trades, quote_volume, eps=0.0))

    avg_size = safe_divide(quote_volume, trades)
    emit("avg_trade_size", avg_size)
    emit("avg_trade_size_ratio", safe_divide(avg_size, avg_size.rolling(20, min_periods=5).mean()))
    emit("volume_per_trade_cv_20", _rolling_cv(avg_size, 20))

    emit("taker_buy_volume", taker_buy)
    emit("taker_buy_quote_volume", taker_buy_quote)

    buy_ratio = safe_divide(taker_buy, volume)
    sell_ratio = safe_divide(taker_sell, volume)
    emit("aggressive_buy_ratio", buy_ratio)
    emit("aggressive_sell_ratio", sell_ratio)
    emit("buy_sell_ratio", safe_divide(taker_buy, taker_sell))

    signed = (taker_buy - taker_sell)
    imbalance = safe_divide(signed, volume)
    emit("volume_imbalance", imbalance)
    emit("order_imbalance_1h", imbalance)
    emit("order_imbalance_5h", _trailing_sum(signed, volume, 5))
    emit("order_imbalance_15h", _trailing_sum(signed, volume, 15))
    emit("order_imbalance_zscore_60d", rolling_zscore(imbalance, 24 * 60, min_periods=24 * 20))

    emit("taker_buy_ratio_mean_5", buy_ratio.rolling(5, min_periods=2).mean())
    emit("taker_buy_ratio_mean_15", buy_ratio.rolling(15, min_periods=5).mean())
    emit("taker_buy_ratio_change_1h", buy_ratio - buy_ratio.shift(1))
    emit("quote_volume_zscore_60d", rolling_zscore(quote_volume, 24 * 60, min_periods=24 * 20))
    emit("flow_price_divergence", _flow_price_divergence(imbalance, close))

    for name in want:
        if name not in out.columns:
            out[name] = np.nan
    return out


# --------------------------------------------------------------------- helpers

def _trailing_sum(numerator: pd.Series, denominator: pd.Series, window: int) -> pd.Series:
    """Sum of ``numerator`` over ``window`` divided by the summed denominator."""
    num = numerator.rolling(window, min_periods=max(2, window // 2)).sum()
    den = denominator.rolling(window, min_periods=max(2, window // 2)).sum()
    return safe_divide(num, den)


def _rolling_cv(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    std = series.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
    return safe_divide(std, mean)


def _flow_price_divergence(imbalance: pd.Series, close: pd.Series) -> pd.Series:
    """+1 flow and price agree, -1 they disagree, 0 when either is flat/unknown.

    Reported as a sign-agreement indicator rather than a magnitude, so a single
    extreme bar cannot dominate the feature.
    """
    price_change = close.pct_change(fill_method=None)
    flow_sign = np.sign(imbalance.to_numpy(dtype="float64"))
    price_sign = np.sign(price_change.to_numpy(dtype="float64"))
    out = np.where(
        (np.abs(flow_sign) > 0) & (np.abs(price_sign) > 0),
        flow_sign * price_sign,
        0.0,
    )
    series = pd.Series(out, index=imbalance.index)
    return series.where(imbalance.notna() & price_change.notna())


def _col(frame: pd.DataFrame | None, name: str, index: pd.DatetimeIndex) -> pd.Series:
    if frame is None or name not in getattr(frame, "columns", []):
        return pd.Series(np.nan, index=index, dtype="float64")
    series = pd.to_numeric(frame[name], errors="coerce")
    series.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return series.reindex(index)
