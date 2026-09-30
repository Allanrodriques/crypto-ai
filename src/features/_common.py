"""Shared helpers for V2 feature modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Elementwise division that yields ``NaN`` instead of inf/NaN leakage.

    Used instead of ``.replace(0, np.nan)`` style patching so that a zero
    denominator cannot silently become a huge finite number that a tree model
    would treat as a genuine extreme.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den.where(den.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Trailing z-score computed against history that *excludes* the current bar.

    Two details matter here, both learned from real data rather than theory.

    *Excluding the current value.*  When a bar is an extreme outlier, including
    it inflates the very mean and standard deviation meant to detect it - the
    same self-defeating statistic as a MAD threshold on fat-tailed returns.

    *A floor on the denominator.*  Trailing windows are often perfectly flat -
    funding sits at one rate for 24 hours at a time, so many 20-bar windows have
    zero standard deviation.  Dividing by that yields inf/NaN, which silently
    deleted ~16% of rows in the real BTC run.  The denominator is therefore
    floored relative to the series' own scale: a value sitting exactly on a flat
    baseline scores 0, and a genuine jump from a flat baseline scores large but
    finite.

    The floor must stay strictly above :func:`safe_divide`'s ``eps``, otherwise
    the floor is itself treated as a zero denominator and the result is NaN
    anyway - which is the precise failure the floor exists to prevent.  The
    first version of this helper set both to 1e-12, so a constant series still
    scored all-NaN and the fix appeared to do nothing.
    """
    minimum = window if min_periods is None else min_periods
    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum).mean()
    std = history.rolling(window, min_periods=minimum).std(ddof=0)

    scale = float(np.nanstd(series.to_numpy(dtype="float64"))) if len(series) else 0.0
    floor = max(1e-3 * abs(scale), 1e-11)
    denominator = std.where(std > floor, floor)
    return safe_divide(series - mean, denominator)
