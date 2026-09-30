"""Horizon definitions for V3 multi-horizon return forecasting.

A *horizon* is a wall-clock duration, not a row count.  The distinction matters
because the hourly grid is not perfectly continuous - this repository's own BTC
series is missing at least one candle (2023-03-24 13:00 UTC) - so a positional
offset of "7 days" would silently mean "7 *observed* candles" and would drift.

Everything downstream keys off :class:`Horizon`, which carries the parsed
:class:`pandas.Timedelta`, the equivalent number of hours at the configured bar
frequency, and a canonical short label used in filenames and report columns.

Horizons are configuration, never hard-coded: ``V3Config`` reads them from the
``v3.horizons`` list and the default lives in :data:`DEFAULT_HORIZONS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

#: The default horizon ladder from the V3 research specification.
DEFAULT_HORIZONS: tuple[str, ...] = ("1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d")

#: ``1d`` / ``30d`` / ``12h`` / ``90m`` - a count plus a unit suffix.
_HORIZON_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")

#: Multipliers to :class:`pandas.Timedelta` units, longest first so that a
#: greedy match cannot read ``"d"`` out of ``"day"`` style spellings.
_UNIT_DAYS = {
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "w": 7.0,
    "week": 7.0,
    "weeks": 7.0,
    "h": 1.0 / 24.0,
    "hr": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
}


class HorizonError(ValueError):
    """Raised for a horizon string that cannot be parsed."""


def parse_horizon(value: str) -> pd.Timedelta:
    """Parse ``"30d"`` into ``Timedelta('30 days')``.

    Supports day and hour units (and weeks as a convenience).  Anything else is
    rejected loudly rather than being coerced, because a silently misread
    horizon would produce a target column that is subtly wrong.
    """
    if not isinstance(value, str):
        raise HorizonError(f"Horizon must be a string like '30d', got {type(value).__name__}")
    match = _HORIZON_PATTERN.match(value)
    if not match:
        raise HorizonError(
            f"Cannot parse horizon {value!r}; expected a count and a unit, e.g. '7d' or '12h'"
        )
    count, unit = match.group(1), match.group(2).lower()
    if unit not in _UNIT_DAYS:
        raise HorizonError(
            f"Unsupported horizon unit {unit!r} in {value!r}; supported units: {sorted(_UNIT_DAYS)}"
        )
    return pd.Timedelta(days=float(count) * _UNIT_DAYS[unit])


@dataclass(frozen=True)
class Horizon:
    """One forecast horizon: a wall-clock duration plus derived quantities.

    Attributes
    ----------
    label:
        Canonical short label, exactly as configured (``"30d"``).  Used in column
        names, artifact filenames and report rows.
    delta:
        The wall-clock duration of the forward return window.
    bars_per_unit:
        How many candles of the base frequency fit in one day.  For a 1h grid
        this is 24.
    """

    label: str
    delta: pd.Timedelta
    bars_per_unit: int

    def __post_init__(self) -> None:
        if not self.label or not isinstance(self.label, str):
            raise HorizonError(f"Horizon label must be a non-empty string, got {self.label!r}")
        if self.delta <= pd.Timedelta(0):
            raise HorizonError(f"Horizon {self.label!r} must be positive, got {self.delta}")
        if self.bars_per_unit < 1:
            raise HorizonError(f"bars_per_unit must be >= 1, got {self.bars_per_unit}")

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_label(cls, label: str, bars_per_unit: int) -> "Horizon":
        return cls(label=str(label).strip(), delta=parse_horizon(label), bars_per_unit=int(bars_per_unit))

    @classmethod
    def from_config(cls, values: Iterable[Any], bars_per_unit: int) -> tuple["Horizon", ...]:
        """Build the configured horizon ladder, validated and de-duplicated.

        The ladder is sorted by duration so that "1d, 3d, 180d" and "180d, 1d, 3d"
        produce byte-identical reports, and so that "the longest horizon" is
        simply ``ladder[-1]``.
        """
        if isinstance(values, (str, bytes)):
            raise HorizonError(
                "v3.horizons must be a list of labels such as ['1d', '7d'], not a bare string"
            )
        seen: dict[str, Horizon] = {}
        for value in values or ():
            horizon = cls.from_label(value, bars_per_unit)
            if horizon.label in seen:
                # A duplicate is harmless but almost always a config typo; keeping
                # the first and warning-free would hide it, so refuse instead.
                raise HorizonError(f"Duplicate horizon {horizon.label!r} in v3.horizons")
            seen[horizon.label] = horizon
        if not seen:
            raise HorizonError("v3.horizons is empty; at least one horizon is required")
        ordered = tuple(sorted(seen.values(), key=lambda h: (h.delta, h.label)))
        durations = [h.delta for h in ordered]
        if len(set(durations)) != len(durations):
            raise HorizonError(
                "v3.horizons contains two labels with the same duration "
                f"({[h.label for h in ordered]}); each horizon must be distinct"
            )
        return ordered

    # ------------------------------------------------------------- accessors
    @property
    def hours(self) -> float:
        return self.delta / pd.Timedelta(hours=1)

    @property
    def nominal_bars(self) -> int:
        """Bars the horizon spans if the grid were perfectly continuous.

        This is the *nominal* count used for purge sizing and for reporting.  It
        is never used to slice the frame positionally - see
        :func:`src.v3.targets.build_future_returns` for the time-based lookup.

        ``bars_per_unit`` is bars *per day* (24 for an hourly grid), so the
        conversion is ``days * bars_per_unit``.  Multiplying by ``hours`` instead
        would inflate a 1d horizon to 24*24 = 576 bars, which would then be used
        as a purge width and silently drop far too much training data.
        """
        return int(round(self.days * self.bars_per_unit))

    @property
    def bars_per_day(self) -> int:
        """Alias for :attr:`bars_per_unit`, named for how config declares it."""
        return self.bars_per_unit

    @property
    def days(self) -> float:
        return self.delta / pd.Timedelta(days=1)

    def target_column(self, symbol: str | None = None) -> str:
        """Column name holding this horizon's future return."""
        if symbol:
            return f"{symbol}_future_return_{self.label}"
        return f"future_return_{self.label}"

    def artifact_stem(self, model_name: str) -> str:
        """Filename stem for this (model, horizon) pair, e.g. ``xgboost_30d``."""
        return f"{model_name}_{self.label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "delta": str(self.delta),
            "days": self.days,
            "hours": self.hours,
            "nominal_bars": self.nominal_bars,
            "target_column": self.target_column(),
        }

    def describe(self) -> str:
        return f"{self.label} ({self.delta}, {self.nominal_bars} bars)"


def build_horizon_ladder(mapping: Mapping[str, Any] | None) -> tuple[Horizon, ...]:
    """Read the horizon ladder out of a ``v3:`` config mapping.

    Reads ``v3.horizons`` (a list of labels) and ``v3.bars_per_day`` (the base
    frequency, defaulting to 24 for the 1h grid this project runs on).
    """
    cfg = dict(mapping or {})
    labels = cfg.get("horizons", DEFAULT_HORIZONS)
    bars_per_unit = int(cfg.get("bars_per_day", 24))
    return Horizon.from_config(labels, bars_per_unit)


def max_horizon(horizons: Sequence[Horizon]) -> Horizon:
    """The longest horizon, which sets the purge/embargo width."""
    if not horizons:
        raise HorizonError("Cannot take the max of an empty horizon ladder")
    return max(horizons, key=lambda h: (h.delta, h.label))
