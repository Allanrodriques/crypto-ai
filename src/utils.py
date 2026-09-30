"""Small shared helpers: logging, deterministic seeding, JSON/IO, interval parsing."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER_NAME = "crypto_ml"

_INTERVAL_UNITS = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


# --------------------------------------------------------------------------- logging

def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured logger.

    A single stream handler is attached to the root ``crypto_ml`` logger so
    library code never configures logging itself.
    """
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("CRYPTO_ML_LOGLEVEL", "INFO"))
        root.propagate = False
    return root.getChild(name) if name != LOGGER_NAME else root


# --------------------------------------------------------------------------- determinism

def set_global_seed(seed: int) -> None:
    """Seed every RNG the project can reach.

    Covers Python's ``random``, NumPy's legacy global generator, and the
    environment variable hash seed is *not* set (it must be set before the
    interpreter starts to have any effect).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- json / io

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars, arrays and datetimes."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            if isinstance(o, pd.Timestamp) and o.tzinfo is None:
                o = o.tz_localize("UTC")
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return super().default(o)


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` so the output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def save_json(payload: Mapping[str, Any], path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Write ``payload`` as strict JSON (non-finite floats become ``null``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(dict(payload)), handle, indent=indent, cls=_NumpyEncoder, sort_keys=False)
        handle.write("\n")
    return target


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON document written by :func:`save_json`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_to_utc(value: Any) -> pd.Timestamp:
    """Coerce ``value`` to a tz-aware UTC :class:`pandas.Timestamp`.

    Binance reports candle open times in milliseconds since the epoch.  A naive
    datetime is *assumed* to be UTC (that is the only correct interpretation for
    an exchange-provided epoch value), and an already-localised value is
    converted rather than reinterpreted.
    """
    if value is None:
        raise ValueError("Cannot convert None to a timestamp")
    stamp = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Heuristic on magnitude: Binance klines use milliseconds.
        unit = "ms" if abs(float(value)) > 1e11 else "s"
        stamp = pd.to_datetime(float(value), unit=unit, utc=True)
    elif stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


# --------------------------------------------------------------------------- intervals

def interval_to_milliseconds(interval: str) -> int:
    """Convert a Binance interval string (e.g. ``1h``, ``4h``, ``1d``) to milliseconds."""
    text = str(interval).strip().lower()
    if not text or not text[0].isdigit() or len(text) < 2:
        raise ValueError(f"Unsupported interval format: {interval!r}")
    step, unit = int(text[:-1]), text[-1]
    if unit not in _INTERVAL_UNITS or step < 1:
        raise ValueError(f"Unsupported interval: {interval!r}")
    return step * _INTERVAL_UNITS[unit]


def candles_per_year(interval: str) -> int:
    """Number of candles in a 365-day year for ``interval`` (8760 for 1h)."""
    ms_per_year = 365 * 86_400_000
    return int(round(ms_per_year / interval_to_milliseconds(interval)))


def parse_date(value: Any) -> pd.Timestamp | None:
    """Parse a config date (``"2022-01-01"`` / ISO / ``None``) into a UTC timestamp."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return timestamp_to_utc(int(value))
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def floored_to_interval(stamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Snap ``stamp`` down to the nearest candle boundary for ``interval``."""
    step_ms = interval_to_milliseconds(interval)
    epoch_ms = int(stamp.value // 1_000_000)
    return pd.to_datetime((epoch_ms // step_ms) * step_ms, unit="ms", utc=True)


def format_timestamp(stamp: Any) -> str:
    """Human-readable UTC string used in logs and reports."""
    if stamp is None:
        return "None"
    return timestamp_to_utc(stamp).strftime("%Y-%m-%d %H:%M:%S UTC")


def humanise_count(n: int) -> str:
    return f"{n:,}"
