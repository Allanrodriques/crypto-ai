"""Centralised timestamp-alignment module.

This is the single place where the project decides *what was knowable when*.
Every non-OHLCV data source is attached to a feature grid through this module
and nowhere else, so the no-look-ahead rule has exactly one implementation to
audit and test.

The five timestamps
------------------
The vocabulary used throughout (and written into experiment metadata):

``data timestamp``
    The moment the observation is *about* - e.g. the close of a candle.
``event timestamp``
    When the real-world event happened (for sentiment: when the reading was
    produced).
``availability timestamp``
    The earliest moment the observation could have been known.  This is the
    value that actually matters and it is always ``>= event timestamp``.
``feature timestamp``
    The ``t`` of the row being predicted.  Only observations with
    ``availability timestamp <= feature timestamp`` may be used.
``prediction timestamp``
    When the prediction is made in practice: the *next* candle boundary after
    ``t``, since the signal is acted on at ``open[t+1]``.

The alignment rule
------------------
For feature timestamp ``T`` the value used is the **most recent observation
whose availability timestamp is ``<= T``**.  Three consequences follow, and all
three are deliberate:

1. **No forward fill from the future.**  A source is never consulted for data
   published after ``T``, even when the source physically holds it.  This is the
   entire point of the module.
2. **No indefinite back-fill.**  Each contract carries ``max_age``.  If the
   newest usable observation is older than that, the aligned value is ``NaN``
   rather than a stale number carried forward forever.  A 3-year-old funding
   rate pretending to describe today's positioning is worse than a gap.
3. **No interpolation across unavailability.**  A missing observation yields
   ``NaN``; the feature is then dropped by the dataset builder, which shortens
   the usable history rather than inventing values.

When the true availability time is unknown, ``release_lag`` is set from the
source's *documented worst case* and the conservative (later) value is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger, timestamp_to_utc

logger = get_logger("alignment")

#: Timestamp vocabulary persisted into experiment metadata for traceability.
TIMESTAMP_ROLES = (
    "data_timestamp",
    "event_timestamp",
    "availability_timestamp",
    "feature_timestamp",
    "prediction_timestamp",
)


@dataclass(frozen=True)
class AvailabilityContract:
    """Declares *when* a source's observations became knowable.

    Parameters
    ----------
    name:
        Source identifier, e.g. ``"funding_rate"``.
    frequency:
        Native sampling interval, e.g. ``"8h"`` for funding.
    release_lag:
        Conservative delay added to the event timestamp to obtain the
        availability timestamp.  Defaults to one full native period, which is
        the safe assumption when a source does not state a publication time.
        Funding is paid at the *start* of its period and is known
        immediately, so it legitimately uses ``0``.
    max_age:
        Longest gap between the availability timestamp and the feature
        timestamp that still yields a value.  Beyond this the value is ``NaN``.
        ``None`` disables the staleness guard.
    intrabar:
        ``True`` when the observation covers a whole period and is only
        complete at the period's close (candles, daily sentiment readings).
        ``False`` when the value is a point-in-time event known at once
        (a funding payment, a trade).
    """
    name: str
    frequency: str
    release_lag: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    max_age: pd.Timedelta | None = None
    intrabar: bool = True

    def __post_init__(self) -> None:
        # Coerce so contracts can be written as `release_lag="1d"` in YAML and in
        # tests without every call site having to wrap the literal.  The dataclass
        # is frozen, so the coerced values go in through object.__setattr__.
        object.__setattr__(self, "release_lag", _as_timedelta(self.release_lag, "release_lag"))
        if self.max_age is not None:
            max_age = _as_timedelta(self.max_age, "max_age")
            if max_age < pd.Timedelta(0):
                raise ValueError(f"max_age must not be negative, got {max_age}")
            object.__setattr__(self, "max_age", max_age)

    def availability(self, event_times: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Availability timestamps for ``event_times`` under this contract."""
        index = pd.DatetimeIndex(pd.to_datetime(event_times, utc=True))
        return index + self.release_lag

    def describe(self) -> dict[str, object]:
        return {
            "source": self.name,
            "frequency": self.frequency,
            "release_lag": str(self.release_lag),
            "max_age": None if self.max_age is None else str(self.max_age),
            "intrabar": self.intrabar,
        }


class AlignmentError(ValueError):
    """Raised when an alignment would violate the no-look-ahead contract."""


def _as_timedelta(value: Any, label: str) -> pd.Timedelta:
    """Accept a Timedelta, a pandas offset string, or a number of seconds."""
    if isinstance(value, pd.Timedelta):
        return value
    try:
        return pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a Timedelta, a pandas offset string like '1d', or seconds; "
            f"got {value!r}"
        ) from exc


def align_asof(
    target: pd.Series | pd.DatetimeIndex,
    source: pd.DataFrame,
    contract: AvailabilityContract,
    *,
    value_columns: Sequence[str] | None = None,
    allow_reindex: bool = False,
) -> pd.DataFrame:
    """Attach source observations to a feature grid without look-ahead.

    Parameters
    ----------
    target:
        Feature timestamps ``T`` (the rows being predicted).
    source:
        Frame indexed by *event* timestamp, or carrying an explicit
        ``available_at`` column.  When ``available_at`` is absent the
        contract's ``release_lag`` is added to the index.
    contract:
        Availability rules for the source.
    value_columns:
        Columns to align.  Defaults to every column except the time columns.
    allow_reindex:
        Reserved for sources whose native grid is coarser than the feature
        grid.  It does **not** permit forward filling; the as-of join below
        still never reaches past ``T``.

    Returns
    -------
    DataFrame indexed like ``target`` with the aligned value columns.  Cells
    where nothing was available are ``NaN``.
    """
    if not isinstance(source.index, pd.DatetimeIndex):
        raise AlignmentError(
            f"{contract.name}: source must be indexed by a DatetimeIndex (event timestamps)"
        )

    source = source.sort_index()
    columns = list(value_columns) if value_columns else _value_columns(source, contract)

    if "available_at" in source.columns:
        availability = pd.DatetimeIndex(pd.to_datetime(source["available_at"], utc=True))
    else:
        availability = contract.availability(source.index)

    working = source[columns].copy()
    # Carry the *availability* time of the row that supplied each value.  Without
    # it the staleness test below has nothing to measure: after the as-of join the
    # frame is indexed by the target timestamps, so `target - aligned.index` is
    # identically zero and `max_age` silently never expires anything.
    working["__available_at"] = availability
    # The join key MUST be the availability time, never the event time.  Joining
    # on the event timestamp would let a source observation dated in the future
    # supply a value to a feature row that precedes its publication - the exact
    # look-ahead this module exists to prevent.
    working["__source_time"] = availability
    # Duplicate availability timestamps would make merge_asof's direction
    # requirement ambiguous; keep the last, which is the latest observed state.
    working = working[~working["__source_time"].duplicated(keep="last")].sort_values("__source_time")

    target_index = pd.DatetimeIndex(pd.to_datetime(target, utc=True))
    left = pd.DataFrame({"__target_time": target_index})
    aligned = pd.merge_asof(
        left,
        working,
        left_on="__target_time",
        right_on="__source_time",
        direction="backward",
        allow_exact_matches=True,
    )
    # merge_asof's key columns are bookkeeping, not data.  They are dropped
    # explicitly so a leftover `key_0` cannot be mistaken for an aligned value -
    # which would inflate any coverage statistic computed over the frame.
    aligned.index = target_index
    source_available = pd.DatetimeIndex(aligned["__available_at"])
    aligned = aligned.drop(columns=["__target_time", "__source_time", "__available_at"])
    aligned.attrs["source_available_at"] = source_available

    if contract.max_age is not None:
        age = target_index - source_available
        stale = np.asarray(age.isna() | (age > contract.max_age))
        if stale.any():
            logger.info(
                "%s: %d feature timestamp(s) exceed max_age=%s and are left as NaN "
                "(no stale carry-forward)",
                contract.name,
                int(stale.sum()),
                contract.max_age,
            )
            aligned.loc[stale, columns] = np.nan
            # Mask with pd.NaT rather than assigning np.datetime64("NaT"): the
            # latter coerces the index to tz-naive datetime64 and the audit that
            # consumes this provenance then fails to compare it with a tz-aware
            # feature grid.
            source_available = pd.DatetimeIndex(
                [pd.NaT if drop else stamp for stamp, drop in zip(source_available, stale)]
            )
            aligned.attrs["source_available_at"] = source_available

    aligned.attrs["max_age"] = contract.max_age
    aligned.attrs["contract"] = contract.name
    aligned.attrs["value_columns"] = list(columns)
    return aligned


def _value_columns(source: pd.DataFrame, contract: AvailabilityContract) -> list[str]:
    reserved = {"available_at", "event_timestamp", "availability_timestamp"}
    cols = [c for c in source.columns if c not in reserved]
    if not cols:
        raise AlignmentError(f"{contract.name}: no value columns to align")
    return cols


def prediction_timestamps(feature_times: pd.Series | pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    """When a prediction made at bar ``t`` can actually be acted on.

    The signal uses candle ``t`` and is filled at ``open[t+1]``, so the
    prediction timestamp is the *next* candle boundary.  Kept here so the
    backtest and the feature code cannot disagree about it.
    """
    from src.utils import interval_to_milliseconds

    step = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
    return pd.DatetimeIndex(pd.to_datetime(feature_times, utc=True)) + step


def audit_no_lookahead(
    aligned: pd.DataFrame,
    contract: AvailabilityContract,
    target: pd.Series | pd.DatetimeIndex,
    value_column: str,
) -> None:
    """Assert that no aligned value came from an observation published after ``T``.

    This check is only meaningful against the *recorded provenance*, not against
    the target index.  ``align_asof`` stashes the availability timestamp of the
    row that supplied each value in ``aligned.attrs['source_available_at']``;
    comparing that against the feature timestamp ``T`` is what actually proves
    causality.  Comparing ``T`` against itself proves nothing.

    Raises
    ------
    AlignmentError
        If the provenance is missing, or any supplied value was available later
        than the feature timestamp that used it.
    """
    target_index = pd.DatetimeIndex(pd.to_datetime(target, utc=True))
    if not target_index.equals(pd.DatetimeIndex(aligned.index)):
        raise AlignmentError(
            f"{contract.name}: aligned frame is not indexed like the target grid"
        )

    provenance = aligned.attrs.get("source_available_at")
    if provenance is None:
        raise AlignmentError(
            f"{contract.name}: no provenance recorded, so causality cannot be verified. "
            f"Re-align through align_asof(), which records source_available_at."
        )
    provenance = pd.DatetimeIndex(pd.to_datetime(pd.Series(provenance), utc=True))

    values = aligned[value_column]
    supplied = values.notna().to_numpy()
    if not supplied.any():
        return

    feature_times = target_index[supplied]
    source_times = provenance[supplied]
    # A missing source time on a populated cell means the value came from
    # nowhere traceable, which is itself a causality failure.
    unknown = pd.isna(source_times)
    if unknown.any():
        raise AlignmentError(
            f"{contract.name}: {int(unknown.sum())} populated value(s) have no recorded "
            f"source availability timestamp"
        )

    lookahead = source_times > feature_times
    if lookahead.any():
        first = int(np.flatnonzero(lookahead)[0])
        raise AlignmentError(
            f"{contract.name}: {int(lookahead.sum())} look-ahead value(s) in "
            f"{value_column!r}; earliest at {feature_times[first]} used a source available "
            f"at {source_times[first]}"
        )

    if contract.max_age is not None:
        age = feature_times - source_times
        stale = np.asarray(age > contract.max_age)
        if stale.any():
            first = int(np.flatnonzero(stale)[0])
            raise AlignmentError(
                f"{contract.name}: {int(stale.sum())} value(s) in {value_column!r} are older "
                f"than max_age={contract.max_age}; earliest at {feature_times[first]} used a "
                f"source from {source_times[first]}"
            )


def coverage_report(aligned: pd.DataFrame, total_rows: int) -> dict[str, object]:
    """Coverage of a source over the feature grid, for the availability matrix.

    Only the aligned *value* columns are considered.  Counting every column in
    the frame would let a bookkeeping column that is never null inflate coverage
    to a meaningless 100%.
    """
    if total_rows <= 0:
        return {"rows": 0, "covered": 0, "coverage_pct": 0.0}
    value_columns = aligned.attrs.get("value_columns") or [
        c for c in aligned.columns if not str(c).startswith("__") and not str(c).startswith("key_")
    ]
    values = aligned[list(value_columns)] if value_columns else aligned
    covered_mask = values.notna().any(axis=1)
    non_null = int(covered_mask.sum())
    first = aligned.index[covered_mask]
    return {
        "rows": int(total_rows),
        "covered": non_null,
        "coverage_pct": round(100.0 * non_null / total_rows, 4),
        "first_covered": None if first.empty else str(first[0]),
        "last_covered": None if first.empty else str(first[-1]),
        "value_columns": list(value_columns),
    }
