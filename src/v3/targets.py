"""Multi-horizon forward-return targets.

The label for horizon ``H`` at timestamp ``t`` is::

    future_return_H[t] = close[t + H] / close[t] - 1

Why this is not ``close.shift(-bars)``
--------------------------------------
The V1 target does exactly that, and it is only correct because V1 uses a
single 6-candle horizon on a grid that is dense enough for the error to be
invisible.  V3 spans 24 hours to 4320 hours, and on this project's own BTC data
the hourly grid is missing at least one candle (2023-03-24 13:00 UTC).  A
positional shift of -4320 rows is then *not* 180 days - it is 180 days plus or
minus whatever the accumulated gap drift happens to be, and the error is
invisible because it is smooth.

So the lookup here is time-based: for each timestamp ``t`` we ask for the close
at exactly ``t + H`` and, if that candle does not exist, we take the most recent
close *at or before* ``t + H``.  The fallback is deliberately backward-only:

* Taking the *next* candle instead would push the label window past ``t + H``,
  which leaks information from a later time than the horizon claims.
* Taking the next candle would also silently make the realised holding period
  variable in a way that is hard to reason about.

Rows with no close at or before ``t + H`` - i.e. the tail of the series - get
``NaN``.  That is not a defect to be patched; it is the honest statement that
the future has not happened yet.  Callers must drop those rows for the
corresponding horizon rather than imputing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.v3.horizons import Horizon


@dataclass(frozen=True)
class FutureReturnTargets:
    """Forward-return targets for a ladder of horizons.

    Attributes
    ----------
    frame:
        One column per horizon, named ``future_return_{label}``, sharing the
        input index exactly.
    horizon_index:
        For each horizon, the timestamp actually used as the label's end point.
        This is exposed so a caller can *verify* that the realised window
        matches the requested one, rather than having to trust it.
    realised_offset_hours:
        For each horizon, the realised distance in hours between the prediction
        timestamp and the label end point.  Equal to the requested horizon
        except where a candle was missing.
    requested_hours / end_index:
        The requested window and the final timestamp available in the data.
    """

    frame: pd.DataFrame
    horizon_index: Mapping[str, pd.DatetimeIndex]
    realised_offset_hours: Mapping[str, pd.Series]
    requested_hours: Mapping[str, float]
    end_index: pd.Timestamp

    def column_for(self, horizon: Horizon) -> str:
        return self.frame.columns[list(self.requested_hours).index(horizon.label)]

    def labelled_mask(self, horizon: Horizon | str) -> pd.Series:
        """Boolean mask of rows that have a *real* (non-fabricated) target."""
        label = horizon.label if isinstance(horizon, Horizon) else str(horizon)
        return self.frame[f"future_return_{label}"].notna()

    def n_labelled(self, horizon: Horizon | str) -> int:
        return int(self.labelled_mask(horizon).sum())

    def coverage_table(self) -> pd.DataFrame:
        """Per-horizon label coverage, for the report's data-availability table."""
        rows = []
        for label, series in self.frame.items():
            if not label.startswith("future_return_"):
                continue
            horizon_key = label.removeprefix("future_return_")
            requested = self.requested_hours.get(horizon_key)
            realised = self.realised_offset_hours.get(horizon_key)
            rows.append(
                {
                    "horizon": horizon_key,
                    "requested_hours": requested,
                    "realised_mean_hours": None if realised is None else float(realised.mean()),
                    "realised_min_hours": None if realised is None else float(realised.min()),
                    "n_total": int(series.size),
                    "n_labelled": int(series.notna().sum()),
                    "n_unlabelled_tail": int(series.isna().sum()),
                    "labelled_pct": float(100.0 * series.notna().mean()),
                    "last_labelled_timestamp": (
                        None if not series.notna().any() else str(series.notna().idxmax())
                    ),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "end_index": str(self.end_index),
            "horizons": {k: {"requested_hours": v} for k, v in self.requested_hours.items()},
        }


def _asof_close(
    close: pd.Series, target_index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """Close at or before each timestamp in ``target_index``.

    Returns ``(close_values, source_timestamps)``, both **indexed by
    ``target_index``** and the same length as it.  ``source_timestamps`` is NaT
    where no candle exists at or before the target - the unlabelable tail.

    ``reindex(method="ffill")`` on a union index is the clean way to express
    "most recent observation at or before T" without the off-by-one hazards of a
    hand-rolled ``searchsorted``.  The reindex back onto ``target_index`` is
    essential: the union index is longer than the input, so anything aligned
    positionally afterwards would silently mis-pair prices and labels.
    """
    close = pd.to_numeric(close, errors="coerce")

    # The target stamps are "t + H".  Looking them up against `close` with a
    # plain reindex+ffill is the classic way to fabricate labels: the union
    # index extends past the end of the real data, and ffill happily carries the
    # last known close across that gap, inventing a future price for every
    # unlabelable tail row.  `limit_area="inside"` is the fix - it restricts
    # ffill to gaps *between* observations and leaves everything past the last
    # real candle as NaN, which is the honest answer.
    extended = close.reindex(close.index.union(target_index))
    extended = extended.ffill(limit_area="inside")

    stamp_index = close.index.union(target_index)
    extended_stamp = pd.Series(stamp_index, index=stamp_index).ffill(limit_area="inside")

    # Drop any inferred `freq` so a reindex cannot expand back to a full
    # regular range and mis-pair every price with its label.
    plain_targets = pd.DatetimeIndex(np.asarray(target_index), tz=target_index.tz)
    return (
        extended.reindex(plain_targets),
        extended_stamp.reindex(plain_targets),
    )


def build_future_returns(
    close: pd.Series,
    horizons: Sequence[Horizon],
    *,
    price_column: str = "close",
) -> FutureReturnTargets:
    """Build one forward-return target column per horizon.

    Parameters
    ----------
    close:
        Price series indexed by a timezone-aware, sorted, unique ``DatetimeIndex``.
        Only the timestamps are used from the index; values come from this
        series.
    horizons:
        The configured horizon ladder, typically from
        :func:`src.v3.horizons.build_horizon_ladder`.
    price_column:
        Retained for symmetry with the V1 dataset schema and for error messages.

    Returns
    -------
    FutureReturnTargets
        Targets plus the bookkeeping needed to audit them.
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be a pandas Series, got {type(close).__name__}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError(f"close must be indexed by a DatetimeIndex, got {type(close.index).__name__}")
    if close.index.tz is None:
        raise ValueError("close index must be timezone-aware; use pd.to_datetime(..., utc=True)")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted ascending")
    if not close.index.is_unique:
        raise ValueError(
            "close index has duplicate timestamps; run the V1 validator before building V3 targets"
        )
    if not horizons:
        raise ValueError("At least one horizon is required")

    index = close.index
    end_index = index[-1]
    out: dict[str, pd.Series] = {}
    horizon_index: dict[str, pd.DatetimeIndex] = {}
    realised: dict[str, pd.Series] = {}
    requested: dict[str, float] = {}

    for horizon in horizons:
        delta = horizon.delta
        target_index = index + delta
        label = horizon.label
        requested[label] = float(horizon.hours)

        here = pd.to_numeric(close, errors="coerce")

        # One as-of pass per horizon, both results indexed by target_index.
        future_close, used_at = _asof_close(close, target_index)

        # A label is only honest if *both* ends of the window have a real price.
        # `used_at` is NaT exactly when no candle exists at or before t + H.
        #
        # NOTE: these are combined as raw arrays, never with `&` on the Series.
        # `here` is indexed by t and `future_close` by t + H, so a label-wise
        # `&` would align the two indexes and silently produce a frame the
        # length of their union before failing to broadcast.  Positional
        # combination is safe because both arrays derive from the same
        # monotonically increasing `index` in the same order.
        here_values = here.to_numpy(dtype="float64")
        future_values = future_close.to_numpy(dtype="float64")
        used_stamps = pd.DatetimeIndex(used_at.to_numpy())
        used_present = np.asarray(~used_stamps.isna())

        valid = (
            np.isfinite(here_values)
            & np.isfinite(future_values)
            & used_present
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, future_values / here_values, np.nan)
        series = pd.Series(ratio - 1.0, index=index, name=f"future_return_{label}")
        series = series.replace([np.inf, -np.inf], np.nan)
        out[f"future_return_{label}"] = series

        # Actual elapsed time between prediction and the candle used as the end.
        offsets = (pd.Series(used_at.to_numpy(), index=index) - index).dt.total_seconds() / 3600.0
        realised[label] = offsets.where(series.notna())
        horizon_index[label] = pd.DatetimeIndex(used_at.to_numpy()[series.notna().to_numpy()])

    frame = pd.DataFrame(out, index=index)
    frame.index.name = close.index.name or "timestamp"
    return FutureReturnTargets(
        frame=frame,
        horizon_index=horizon_index,
        realised_offset_hours=realised,
        requested_hours=requested,
        end_index=end_index,
    )


def overlaps_boundary(timestamps: pd.DatetimeIndex, horizon: Horizon, boundary: pd.Timestamp) -> pd.DatetimeIndex:
    """Timestamps whose label window reaches ``boundary`` or later.

    The label for ``t`` is a function of prices in ``[t, t + horizon]``.  So
    ``t`` is unsafe for training against any evaluation period beginning at
    ``boundary`` exactly when ``t + horizon >= boundary``.

    This is the single definition of "purged" used by both the splitter and the
    leakage tests, so the test and the production code cannot drift apart.
    """
    stamps = pd.DatetimeIndex(timestamps)
    if len(stamps) == 0:
        return stamps
    return stamps[stamps + horizon.delta >= boundary]
