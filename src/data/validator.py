"""Data-quality checks for downloaded klines.

Design rule
-----------
**Nothing is silently repaired.**  Every check produces a named issue with a
count and a sample of offending timestamps; the caller decides whether to
continue.  A clean run and a broken run are both reported, and both are
distinguishable from the summary status.

Checks
------
======================  =========================================================
``duplicate_timestamps`` repeated candle opens (would double-count a period)
``missing_timestamps``  interior holes on the expected interval grid
``ordering``            index not strictly increasing
``ohlc_consistency``    high >= max(open, close) and low <= min(open, close)
``positive_prices``     open/high/low/close > 0 and finite
``non_negative_volume`` volume >= 0 and finite
``abnormal_gaps``       holes longer than ``validation.max_gap_candles``
``price_outliers``      robust z-score AND |log return| beyond an absolute floor
``zero_volume``         suspicious run of volume-less candles
``alignment``           candle opens not on the interval grid
``future_candles``      candles stamped after "now" (clock drift / bad host)
======================  =========================================================
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils import format_timestamp, get_logger, interval_to_milliseconds, save_json, timestamp_to_utc, utc_now

logger = get_logger("data.validator")

PRICE_COLUMNS = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised by :func:`validate_klines` when ``fail_on_error`` is set and checks fail."""


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    severity: str  # "error" | "warning"
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregated result of every check for one dataset."""

    symbol: str
    interval: str
    rows: int
    range_start: pd.Timestamp | None
    range_end: pd.Timestamp | None
    checks: list[CheckResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- roll-up

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no check produced an error."""
        return not self.errors

    def status(self) -> str:
        if self.errors:
            return "FAIL"
        return "WARN" if self.warnings else "PASS"

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status(),
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": self.rows,
            "range_start": format_timestamp(self.range_start),
            "range_end": format_timestamp(self.range_end),
            "summary": {
                "checks_run": len(self.checks),
                "checks_passed": sum(1 for c in self.checks if c.passed),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "stats": self.stats,
            "checks": [c.to_dict() for c in self.checks],
        }

    def save(self, path: str | os.PathLike[str]) -> str:
        save_json(self.to_dict(), path)
        return str(path)

    def summary(self) -> str:
        lines = [
            f"Validation {self.status()} — {self.symbol} {self.interval}: {self.rows:,} candles "
            f"({format_timestamp(self.range_start)} .. {format_timestamp(self.range_end)})",
        ]
        for check in self.checks:
            mark = "ok  " if check.passed else check.severity[:5].ljust(5)
            suffix = "" if check.passed else f"  <- {check.count} occurrence(s); e.g. {', '.join(check.examples[:3])}"
            lines.append(f"  [{mark}] {check.name}: {check.message}{suffix}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers

def _examples(index: Sequence[pd.Timestamp], limit: int = 5) -> list[str]:
    return [format_timestamp(t) for t in list(index)[:limit]]


def _expected_grid_count(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> int:
    step_ms = interval_to_milliseconds(interval)
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    if end_ms < start_ms:
        return 0
    return int((end_ms - start_ms) // step_ms) + 1


# --------------------------------------------------------------------------- checks

def check_duplicates(df: pd.DataFrame) -> CheckResult:
    """Repeated candle opens. A duplicate would double-weight one period."""
    if df.empty:
        return CheckResult("duplicate_timestamps", True, "error", "empty frame")
    dupes = df.index[df.index.duplicated(keep=False)].unique()
    return CheckResult(
        "duplicate_timestamps",
        passed=len(dupes) == 0,
        severity="error",
        message="candle opens must be unique",
        count=len(dupes),
        examples=_examples(dupes),
    )


def check_ordering(df: pd.DataFrame) -> CheckResult:
    """Index must be strictly increasing (chronological, no reshuffling)."""
    if df.empty:
        return CheckResult("ordering", True, "error", "empty frame")

    monotonic = bool(df.index.is_monotonic_increasing)
    duplicated = df.index[df.index.duplicated(keep="first")]
    if monotonic and duplicated.size == 0:
        return CheckResult("ordering", True, "error", "strictly increasing")

    # Rows that break monotonicity: each is <= its predecessor.
    breaks = np.flatnonzero(np.asarray(df.index[1:]) <= np.asarray(df.index[:-1]))
    offenders = df.index[breaks + 1] if breaks.size else duplicated
    return CheckResult(
        "ordering",
        passed=False,
        severity="error",
        message=(
            f"{int(breaks.size)} row(s) not after their predecessor"
            + (f"; {duplicated.size} duplicate timestamp(s)" if duplicated.size else "")
        ),
        count=int(breaks.size) + int(duplicated.size),
        examples=_examples(offenders),
    )


def check_missing(df: pd.DataFrame, interval: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> tuple[CheckResult, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Interior holes on the expected interval grid.

    Returns the check *and* the gap ranges so a caller (e.g. the downloader)
    can repair them.
    """
    if df.empty:
        return CheckResult("missing_timestamps", True, "error", "empty frame"), []

    from src.data.downloader import find_missing_ranges  # local import: avoids a cycle

    lo = timestamp_to_utc(start) if start is not None else df.index.min()
    hi = timestamp_to_utc(end) if end is not None else df.index.max()
    gaps = find_missing_ranges(df.index, lo, hi, interval)
    missing_total = sum(int((g[1] - g[0]) / pd.Timedelta(milliseconds=interval_to_milliseconds(interval))) + 1 for g in gaps)
    if missing_total == 0:
        return (
            CheckResult("missing_timestamps", True, "error",
                        f"no holes; {len(df)} candles on a contiguous {interval} grid"),
            [],
        )
    return (
        CheckResult(
            "missing_timestamps",
            passed=False,
            severity="error",
            message=f"{missing_total} candle(s) missing across {len(gaps)} gap(s)",
            count=missing_total,
            examples=[format_timestamp(g[0]) for g in gaps],
        ),
        gaps,
    )


def check_alignment(df: pd.DataFrame, interval: str) -> CheckResult:
    """Every candle open must sit exactly on the interval grid."""
    if df.empty:
        return CheckResult("interval_alignment", True, "error", "empty frame")
    step_ms = interval_to_milliseconds(interval)
    epoch_ms = df.index.view("int64") // 1_000_000
    off_grid = df.index[epoch_ms % step_ms != 0]
    return CheckResult(
        "interval_alignment",
        passed=len(off_grid) == 0,
        severity="error",
        message=f"candle opens must be multiples of {interval}",
        count=len(off_grid),
        examples=_examples(off_grid),
    )


def check_ohlc(df: pd.DataFrame) -> CheckResult:
    """``high >= max(open, close)`` and ``low <= min(open, close)``."""
    if df.empty:
        return CheckResult("ohlc_consistency", True, "error", "empty frame")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    open_ = df["open"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")

    with np.errstate(invalid="ignore"):
        bad_high = ~((high >= open_) & (high >= close))
        bad_low = ~((low <= open_) & (low <= close))

    mask = bad_high | bad_low
    offenders = df.index[mask]
    detail = []
    if bad_high.any():
        detail.append(f"{int(bad_high.sum())} high < open/close")
    if bad_low.any():
        detail.append(f"{int(bad_low.sum())} low > open/close")
    return CheckResult(
        "ohlc_consistency",
        passed=not mask.any(),
        severity="error",
        message=("high >= max(open, close) and low <= min(open, close)" + ("" if not detail else "; " + ", ".join(detail))),
        count=int(mask.sum()),
        examples=_examples(offenders),
    )


def check_nulls(df: pd.DataFrame) -> CheckResult:
    """No NaN / non-finite values in the OHLCV core."""
    if df.empty:
        return CheckResult("null_values", True, "error", "empty frame")
    counts = {col: int(df[col].isna().sum()) for col in (*PRICE_COLUMNS, "volume") if col in df.columns}
    bad = {k: v for k, v in counts.items() if v}
    nonfinite = int((~np.isfinite(df[list(PRICE_COLUMNS)].to_numpy(dtype="float64"))).any(axis=1).sum())
    examples = []
    if bad:
        mask = np.zeros(len(df), dtype=bool)
        for col in bad:
            mask |= df[col].isna().to_numpy()
        examples = _examples(df.index[mask])
    return CheckResult(
        "null_values",
        passed=not bad and nonfinite == 0,
        severity="error",
        message="OHLCV must be present and finite"
        + ("" if not bad else f"; NaNs: {bad}")
        + ("" if not nonfinite else f"; {nonfinite} row(s) with inf/NaN"),
        count=int(sum(bad.values())) + nonfinite,
        examples=examples,
    )


def check_positive_prices(df: pd.DataFrame) -> CheckResult:
    """Prices must be strictly positive — a zero or negative price is corrupt."""
    if df.empty:
        return CheckResult("positive_prices", True, "error", "empty frame")
    mask = np.zeros(len(df), dtype=bool)
    for col in PRICE_COLUMNS:
        mask |= (df[col].to_numpy(dtype="float64") <= 0)
    return CheckResult(
        "positive_prices",
        passed=not mask.any(),
        severity="error",
        message="open/high/low/close must all be > 0",
        count=int(mask.sum()),
        examples=_examples(df.index[mask]),
    )


def check_volume(df: pd.DataFrame) -> CheckResult:
    """Volume must be finite and non-negative."""
    if df.empty:
        return CheckResult("non_negative_volume", True, "error", "empty frame")
    vol = df["volume"].to_numpy(dtype="float64")
    mask = (vol < 0) | ~np.isfinite(vol)
    return CheckResult(
        "non_negative_volume",
        passed=not mask.any(),
        severity="error",
        message="volume must be >= 0 and finite",
        count=int(mask.sum()),
        examples=_examples(df.index[mask]),
    )


def check_abnormal_gaps(
    df: pd.DataFrame, interval: str, max_gap_candles: int, gaps: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> CheckResult:
    """Holes longer than the configured tolerance."""
    if not gaps:
        return CheckResult("abnormal_gaps", True, "warning", f"no gap longer than {max_gap_candles} candles")
    step = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
    wide = [g for g in gaps if int((g[1] - g[0]) / step) + 1 > max_gap_candles]
    return CheckResult(
        "abnormal_gaps",
        passed=not wide,
        severity="warning",
        message=f"{len(wide)} gap(s) exceed {max_gap_candles} consecutive missing candles",
        count=len(wide),
        examples=[format_timestamp(g[0]) for g in wide],
    )


def check_price_outliers(
    df: pd.DataFrame, zscore_threshold: float, max_abs_log_return: float = 0.35
) -> CheckResult:
    """Flag returns so extreme they are more likely corruption than market action.

    Two conditions must hold together:

    * a *median / MAD* robust z-score above ``zscore_threshold`` — robust, so one
      mega-move cannot inflate the very threshold meant to catch it; and
    * an absolute log return above ``max_abs_log_return``.

    The absolute floor matters.  Hourly crypto returns are fat-tailed and mostly
    tiny, so MAD is around 0.1%; a perfectly ordinary 7% hourly candle is then
    ~70 robust sigmas out.  Judged on the z-score alone this check cries wolf on
    normal volatility and gets ignored, which is worse than not having it.  The
    absolute floor keeps it pointed at moves that could not be a real fill.
    """
    if len(df) < 3:
        return CheckResult("price_outliers", True, "warning", "too few rows for outlier detection")
    close = df["close"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(close))
    median = np.nanmedian(log_ret)
    mad = np.nanmedian(np.abs(log_ret - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return CheckResult("price_outliers", True, "warning", "degenerate return distribution; skipped")
    magnitude = np.abs(log_ret - median)
    robust_z = magnitude / scale
    bad = (robust_z > zscore_threshold) & (magnitude > max_abs_log_return)

    typical = float(np.nanpercentile(magnitude, 99.9)) if magnitude.size else 0.0
    context = f"99.9th pct |move| = {typical:.4f}, floor = {max_abs_log_return:.4f}"
    if not bad.any():
        return CheckResult(
            "price_outliers", True, "warning", f"no extreme moves ({context})"
        )
    offenders = df.index[1:][bad]
    worst = float(np.nanmax(magnitude[bad]))
    return CheckResult(
        "price_outliers",
        passed=False,
        severity="warning",
        message=(
            f"{int(bad.sum())} candle(s) with |robust z| > {zscore_threshold} AND "
            f"|log return| > {max_abs_log_return} (largest |move| = {worst:.4f}; {context})"
        ),
        count=int(bad.sum()),
        examples=_examples(offenders),
    )


def check_zero_volume(df: pd.DataFrame, max_ratio: float) -> CheckResult:
    """A large share of volume-less candles usually means a bad feed."""
    if df.empty:
        return CheckResult("zero_volume", True, "warning", "empty frame")
    ratio = float((df["volume"].to_numpy(dtype="float64") == 0).mean())
    offenders = df.index[df["volume"].to_numpy(dtype="float64") == 0]
    return CheckResult(
        "zero_volume",
        passed=ratio <= max_ratio,
        severity="warning",
        message=f"zero-volume share {ratio:.4%} (tolerance {max_ratio:.2%})",
        count=int((df["volume"].to_numpy(dtype="float64") == 0).sum()),
        examples=_examples(offenders),
    )


def check_future_candles(df: pd.DataFrame, now: pd.Timestamp | None = None) -> CheckResult:
    """Guard against clock drift or a tampered host stamping candles in the future."""
    if df.empty:
        return CheckResult("future_candles", True, "error", "empty frame")
    now = timestamp_to_utc(now) if now is not None else pd.Timestamp(utc_now())
    offenders = df.index[df.index > now]
    return CheckResult(
        "future_candles",
        passed=len(offenders) == 0,
        severity="error",
        message=f"no candle may be stamped after {format_timestamp(now)}",
        count=len(offenders),
        examples=_examples(offenders),
    )


# --------------------------------------------------------------------------- entry point

def validate_klines(
    df: pd.DataFrame,
    interval: str,
    symbol: str = "",
    validation_config: dict[str, Any] | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    fail_on_error: bool = False,
    save_to: str | os.PathLike[str] | None = None,
    now: pd.Timestamp | None = None,
) -> ValidationReport:
    """Run every check and return a :class:`ValidationReport`.

    Parameters
    ----------
    df:
        Raw kline frame with a UTC ``DatetimeIndex``.
    interval:
        Binance interval string, e.g. ``"1h"``.
    validation_config:
        The ``validation:`` block from ``config/config.yaml``.
    fail_on_error:
        Raise :class:`DataValidationError` when any error-severity check fails.
    save_to:
        Optional path for the JSON report.
    """
    cfg = dict(validation_config or {})
    max_gap = int(cfg.get("max_gap_candles", 2))
    zscore = float(cfg.get("price_move_zscore_threshold", 20.0))
    max_move = float(cfg.get("price_move_max_log_return", 0.35))
    max_zero_ratio = float(cfg.get("max_zero_volume_ratio", 0.01))
    strict = fail_on_error or bool(cfg.get("fail_on_error", False))

    report = ValidationReport(
        symbol=symbol.upper(),
        interval=interval,
        rows=len(df),
        range_start=df.index.min() if not df.empty else None,
        range_end=df.index.max() if not df.empty else None,
    )

    if df.empty:
        report.add(CheckResult("non_empty", False, "error", "no data to validate", count=0))
        if strict:
            raise DataValidationError("Cannot validate an empty dataframe")
        if save_to:
            report.save(save_to)
        return report

    report.add(check_duplicates(df))
    report.add(check_ordering(df))
    report.add(check_alignment(df, interval))
    report.add(check_nulls(df))
    report.add(check_positive_prices(df))
    report.add(check_ohlc(df))
    report.add(check_volume(df))
    report.add(check_future_candles(df, now=now))

    missing_check, gaps = check_missing(df, interval, start=start, end=end)
    report.add(missing_check)
    report.add(check_abnormal_gaps(df, interval, max_gap, gaps))
    report.add(check_price_outliers(df, zscore, max_move))
    report.add(check_zero_volume(df, max_zero_ratio))

    close = df["close"].astype("float64")
    report.stats = {
        "expected_candles_on_grid": _expected_grid_count(df.index.min(), df.index.max(), interval),
        "min_close": float(close.min()),
        "max_close": float(close.max()),
        "mean_close": float(close.mean()),
        "total_volume": float(df["volume"].sum()),
        "zero_volume_candles": int((df["volume"].to_numpy(dtype="float64") == 0).sum()),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "gap_count": len(gaps),
        "missing_candles": missing_check.count,
        "first_return": float(close.pct_change(fill_method=None).dropna().iloc[0]) if len(close) > 1 else None,
        "total_return": float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else None,
    }

    if save_to:
        report.save(save_to)
    if strict and not report.ok:
        raise DataValidationError(
            f"Validation failed with {len(report.errors)} error(s): "
            + "; ".join(f"{c.name}({c.count})" for c in report.errors)
            + f"\n{report.summary()}"
        )
    return report
