"""Sentiment features (EXP-02).

Source
------
Alternative.me Crypto Fear & Greed Index, daily at 00:00 UTC, available from
2018-02-01.

Design decision: no directional prior
------------------------------------
The source publishes a human label (``Extreme Fear`` ... ``Extreme Greed``) and
it is tempting to encode "high = bullish".  This project does not.  The
sentiment features below are all *symmetric transforms of the raw number* -
levels, changes, rolling statistics, and a regime label - with no sign
convention attached.  If sentiment carries no directional information, the
model should be free to find none; baking in a bullish assumption would
manufacture an edge that the data may not support.

The published label is retained only as a *categorical regime* feature
(``fear_greed_regime``, 0-4 as an ordinal) and as a bucket for
one-hot-style columns, never as an assumed direction.

Availability
------------
A reading for day D is aligned as knowable from D+1 00:00 UTC.  See
:mod:`src.data.sources.sentiment`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import rolling_zscore, safe_divide

GROUP = "sentiment"

SENTIMENT_DOCS: dict[str, str] = {
    "fear_greed_value": "Fear & Greed index level, 0-100, most recent reading",
    "fear_greed_change": "Change in the index since the previous daily reading",
    "fear_greed_change_3d": "Change in the index over the last 3 daily readings",
    "fear_greed_rolling_mean_7": "7-day rolling mean of the index",
    "fear_greed_rolling_mean_30": "30-day rolling mean of the index",
    "fear_greed_rolling_std_30": "30-day rolling standard deviation of the index",
    "fear_greed_zscore_90d": "Index z-score against a trailing 90-day window",
    "fear_greed_regime": "Ordinal regime bucket 0-4 (0=Extreme Fear .. 4=Extreme Greed)",
    "fear_greed_regime_change": "Change in the regime bucket versus the previous reading",
    "fear_greed_above_ma30": "Index minus its 30-day mean (signed, no scaling)",
    "fear_greed_range_30d": "Max-min of the index over 30 days",
    "fear_greed_days_since_change": "Daily readings since the index last moved by >= 5 points",
}

SENTIMENT_REQUIRED_HISTORY = 24 * 90 + 24

#: Boundaries of the published buckets, used only to reproduce the source's own
#: classification as an ordinal value.
_REGIME_EDGES = (25, 45, 55, 75)


def build_sentiment_features(
    index: pd.DatetimeIndex,
    sentiment: pd.DataFrame,
    *,
    enabled: list[str] | None = None,
) -> pd.DataFrame:
    """Compute sentiment features on the hourly grid.

    ``sentiment`` must already be aligned (as-of) to ``index`` by the alignment
    module, so the daily reading is carried across the hours it covers with a
    staleness guard already applied.
    """
    if sentiment is not None and not isinstance(sentiment, pd.DataFrame):
        # A dict or Series here would otherwise fall through `_column`'s
        # `getattr(frame, "columns", [])` check and yield an all-NaN frame that
        # looks like "the source had no data" instead of "the caller was wrong".
        raise TypeError(
            f"sentiment must be a pandas DataFrame with aligned columns, got "
            f"{type(sentiment).__name__}"
        )
    out = pd.DataFrame(index=pd.DatetimeIndex(index, tz="UTC"))
    want = set(enabled) if enabled else set(SENTIMENT_DOCS)
    out.index.name = "timestamp"

    value = _column(sentiment, "fear_greed_value", out.index)
    if value.notna().sum() == 0:
        for name in want:
            out[name] = np.nan
        return out

    # Readings repeat across the 24 hourly bars of a day; compress to the
    # distinct daily sequence so rolling statistics are over *days*, not hours.
    daily = value[value.notna()].resample("1D").last()
    daily = daily.dropna()

    # `daily` is indexed at midnight while the feature grid is hourly, so it must
    # be attached by as-of join, not by reindex: reindexing would match only the
    # 00:00 bar of each day and NaN the other 23, silently destroying ~96% of the
    # rows.  As-of "backward" is also the honest operation - it takes the value
    # that was in effect at T, using nothing published after T.
    def emit(name: str, series: pd.Series) -> None:
        if name not in want:
            return
        clean = series.dropna()
        if clean.empty:
            out[name] = np.nan
            return
        merged = pd.merge_asof(
            pd.DataFrame(index=out.index),
            clean.to_frame(name),
            left_on=out.index,
            right_on=clean.index,
            direction="backward",
        )
        out[name] = merged[name].to_numpy()

    emit("fear_greed_value", daily)
    emit("fear_greed_change", daily.diff())
    emit("fear_greed_change_3d", daily.diff(3))
    emit("fear_greed_rolling_mean_7", daily.rolling(7, min_periods=1).mean())
    emit("fear_greed_rolling_mean_30", daily.rolling(30, min_periods=3).mean())
    emit("fear_greed_rolling_std_30", daily.rolling(30, min_periods=5).std(ddof=0))
    emit("fear_greed_zscore_90d", rolling_zscore(daily, 90, min_periods=30))
    emit("fear_greed_above_ma30", daily - daily.rolling(30, min_periods=3).mean())
    emit("fear_greed_range_30d", daily.rolling(30, min_periods=5).max() - daily.rolling(30, min_periods=5).min())

    regime = pd.cut(daily, bins=[-np.inf, *_REGIME_EDGES, np.inf], labels=False).astype("float64")
    emit("fear_greed_regime", regime)
    emit("fear_greed_regime_change", regime.diff())
    emit("fear_greed_days_since_change", _days_since_move(daily, threshold=5.0))

    for name in want:
        if name not in out.columns:
            out[name] = np.nan
    return out


def _days_since_move(series: pd.Series, threshold: float) -> pd.Series:
    """Daily readings elapsed since the last move of at least ``threshold``."""
    moved = series.diff().abs() >= threshold
    groups = (~moved).cumsum()
    counts = moved.groupby(groups).cumsum()
    return counts.astype("float64")


def _column(frame: pd.DataFrame | None, name: str, index: pd.DatetimeIndex) -> pd.Series:
    if frame is None or name not in getattr(frame, "columns", []):
        return pd.Series(np.nan, index=index, dtype="float64")
    series = pd.to_numeric(frame[name], errors="coerce")
    series.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return series.reindex(index)
