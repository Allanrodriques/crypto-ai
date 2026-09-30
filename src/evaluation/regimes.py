"""Market-regime labelling and per-regime evaluation.

Regime definitions are explicit and documented
----------------------------------------------
"Was the market a bull or a bear period?" is not a question with an obvious
answer, so this module refuses to hard-code a subjective definition.  Every rule
below is stated as a formula, and each regime is derived from information
available *at or before* ``t`` so a regime label can be used as a conditioning
variable without itself becoming a leak.

* **Trend** (bull / bear / sideways) - sign of the trailing 90-day return
  (2,160 hourly bars) on a 200-day moving average:

  ``bull``   close > SMA(200d) and 90d return > +5%
  ``bear``   close < SMA(200d) and 90d return < -5%
  ``sideways`` otherwise

  The +/-5% band exists so that "flat but drifting" is not silently forced into
  a directional label, which would make the sideways bucket empty.

* **Volatility** (high / low) - trailing 30-day realised volatility of hourly
  log returns versus its own trailing 365-day median:

  ``high``  vol(30d) > median(vol(365d))
  ``low``   otherwise

Evaluation
----------
:func:`evaluate_by_regime` reports each regime's row count, class balance and
metrics, so a strategy whose edge lives entirely in one regime is visible
instead of being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger, interval_to_milliseconds

logger = get_logger("evaluation.regimes")

TREND_REGIMES = ("bull", "sideways", "bear")
VOL_REGIMES = ("high_vol", "low_vol")

DEFAULT_TREND_WINDOW = 24 * 90
DEFAULT_SMA_WINDOW = 24 * 200
DEFAULT_TREND_BAND = 0.05
DEFAULT_VOL_WINDOW = 24 * 30
DEFAULT_VOL_MEDIAN_WINDOW = 24 * 365


@dataclass(frozen=True)
class RegimeConfig:
    """Documented parameters for regime labelling."""

    trend_window: int = DEFAULT_TREND_WINDOW
    sma_window: int = DEFAULT_SMA_WINDOW
    trend_band: float = DEFAULT_TREND_BAND
    vol_window: int = DEFAULT_VOL_WINDOW
    vol_median_window: int = DEFAULT_VOL_MEDIAN_WINDOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_window_candles": self.trend_window,
            "sma_window_candles": self.sma_window,
            "trend_band": self.trend_band,
            "vol_window_candles": self.vol_window,
            "vol_median_window_candles": self.vol_median_window,
            "definitions": {
                "bull": f"close > SMA({self.sma_window}) and {self.trend_window}c return > +{self.trend_band}",
                "bear": f"close < SMA({self.sma_window}) and {self.trend_window}c return < -{self.trend_band}",
                "sideways": "otherwise",
                "high_vol": f"realised vol({self.vol_window}) > its trailing median({self.vol_median_window})",
                "low_vol": "otherwise",
            },
        }


def label_regimes(
    close: pd.Series,
    *,
    config: RegimeConfig | None = None,
    interval: str = "1h",
) -> pd.DataFrame:
    """Attach ``trend`` and ``volatility`` regime columns to a close series.

    Every input is trailing, so the label at ``t`` uses only data up to ``t``.
    """
    cfg = config or RegimeConfig()
    close = pd.to_numeric(close, errors="coerce").sort_index()

    sma = close.rolling(cfg.sma_window, min_periods=max(10, cfg.sma_window // 4)).mean()
    trend_return = close / close.shift(cfg.trend_window) - 1.0

    trend = pd.Series("sideways", index=close.index, dtype="object")
    bull = (close > sma) & (trend_return > cfg.trend_band)
    bear = (close < sma) & (trend_return < -cfg.trend_band)
    trend[bull.to_numpy()] = "bull"
    trend[bear.to_numpy()] = "bear"
    trend[(sma.isna()) | (trend_return.isna())] = "unknown"

    log_ret = np.log(close).diff()
    realised = log_ret.rolling(cfg.vol_window, min_periods=max(24, cfg.vol_window // 4)).std(ddof=0)
    median_vol = realised.rolling(cfg.vol_median_window, min_periods=max(24, cfg.vol_window)).median()
    vol = pd.Series("unknown", index=close.index, dtype="object")
    known = realised.notna() & median_vol.notna()
    vol[(realised > median_vol) & known] = "high_vol"
    vol[known.to_numpy() & (realised <= median_vol).to_numpy()] = "low_vol"

    out = pd.DataFrame({"trend_regime": trend, "vol_regime": vol}, index=close.index)
    out["close"] = close
    out["sma"] = sma
    out["trend_return"] = trend_return
    out["realised_vol"] = realised
    out["median_vol"] = median_vol
    return out


def regime_distribution(regimes: pd.DataFrame) -> dict[str, Any]:
    """Counts and shares for each regime, for the report."""
    out: dict[str, Any] = {}
    for column in ("trend_regime", "vol_regime"):
        counts = regimes[column].value_counts(dropna=False)
        total = int(counts.sum())
        out[column] = {
            str(key): {"n": int(value), "share_pct": round(100.0 * value / total, 2) if total else 0.0}
            for key, value in counts.items()
        }
    return out


def evaluate_by_regime(
    y_true: pd.Series,
    y_proba: np.ndarray,
    regimes: pd.DataFrame,
    metric_fn,
    *,
    index: pd.DatetimeIndex | None = None,
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute metrics separately per trend and volatility regime.

    ``y_proba`` may be a 1-D positive-class probability or a full probability
    matrix; the positive class is taken as the last column, matching the label
    coding UP=1 (binary) or UP=2 (three-class).
    """
    idx = index if index is not None else y_true.index
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        positive = proba[:, -1]
    else:
        positive = proba

    labels = regimes.reindex(idx)
    results: dict[str, Any] = {}
    for column, buckets in (("trend_regime", TREND_REGIMES), ("vol_regime", VOL_REGIMES)):
        per_regime: dict[str, Any] = {}
        for bucket in (*buckets, "unknown"):
            mask = (labels[column] == bucket).to_numpy()
            n = int(mask.sum())
            if n < 30:
                per_regime[bucket] = {"n": n, "metrics": None, "note": "too few rows to evaluate"}
                continue
            subset_true = y_true.to_numpy()[mask]
            subset_proba = proba[mask]
            try:
                metrics = metric_fn(subset_true, subset_proba)
            except ValueError as exc:
                per_regime[bucket] = {"n": n, "metrics": None, "note": str(exc)}
                continue
            share_up = float(np.mean(subset_true == np.max(subset_true)))
            # A sub-sample can be single-class, which makes ROC-AUC/balanced
            # accuracy undefined rather than zero.  Those arrive as None and are
            # passed through so the report shows "undefined" instead of a
            # misleading 0.0.  Nested blocks (the confusion matrix) are dropped
            # here because the flat per-regime table cannot hold them.
            scalar = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.integer, np.floating)) or v is None
            }
            per_regime[bucket] = {
                "n": n,
                "share_positive": share_up,
                "metrics": {k: (None if v is None else float(v)) for k, v in scalar.items()},
            }
        results[column] = per_regime
    return results
