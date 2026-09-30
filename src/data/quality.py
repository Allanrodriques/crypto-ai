"""Source catalog: contracts, coverage accounting and the availability matrix.

This module is where the project's honesty about data availability is encoded.

Two things live here:

1. :data:`SOURCE_CONTRACTS` - the availability contract for every source,
   including the ones that are **unavailable** and therefore not implemented.
   Recording an unavailable source explicitly is what stops it from being
   quietly reintroduced later by someone assuming the data "must exist
   somewhere".
2. :func:`build_availability_matrix` - the machine-readable
   ``reports/data_availability.csv`` and per-source quality report required by
   the specification, with coverage measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.alignment.availability import TIMESTAMP_ROLES, AvailabilityContract
from src.utils import get_logger, save_json

logger = get_logger("data.quality")

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class SourceSpec:
    """Everything known about one candidate data source."""

    key: str
    label: str
    endpoint: str
    frequency: str
    #: Conservative publication delay.  See each value's justification inline.
    release_lag: pd.Timedelta
    #: Longest staleness tolerated before the aligned value becomes NaN.
    max_age: pd.Timedelta | None
    #: True when the observation covers a whole period and completes at its close.
    intrabar: bool
    #: False => the source has no usable history and is not implemented.
    historically_available: bool
    earliest_observed: str | None = None
    notes: str = ""
    unavailable_reason: str = ""
    used_by: tuple[str, ...] = ()
    feature_group: str | None = None

    def contract(self) -> AvailabilityContract:
        return AvailabilityContract(
            name=self.key,
            frequency=self.frequency,
            release_lag=self.release_lag,
            max_age=self.max_age,
            intrabar=self.intrabar,
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "feature_group": self.feature_group or "",
            "source": self.key,
            "label": self.label,
            "endpoint": self.endpoint,
            "start_date": self.earliest_observed or "",
            "end_date": "",
            "frequency": self.frequency,
            "coverage": "",
            "historical_available": self.historically_available,
            "release_lag": str(self.release_lag),
            "max_age": "" if self.max_age is None else str(self.max_age),
            "notes": self.notes or self.unavailable_reason,
        }


#: Conservative release lags, each justified rather than assumed.
SOURCE_CONTRACTS: dict[str, SourceSpec] = {
    "binance_spot": SourceSpec(
        key="binance_spot",
        label="Binance spot klines",
        endpoint="https://data-api.binance.vision/api/v3/klines",
        frequency="1h",
        release_lag=pd.Timedelta(0),
        max_age=None,
        intrabar=True,
        historically_available=True,
        earliest_observed="2017-08-17 (Binance spot launch)",
        notes="Candle fields include trades and taker_buy_volume, so trade-flow "
        "microstructure is available at full spot history with no extra requests.",
        used_by=("all",),
    ),
    "binance_funding": SourceSpec(
        key="binance_funding",
        label="Binance USD-M perpetual funding rates",
        endpoint="https://fapi.binance.com/fapi/v1/fundingRate",
        frequency="8h",
        release_lag=pd.Timedelta(0),
        max_age=pd.Timedelta(hours=24),
        intrabar=False,
        historically_available=True,
        earliest_observed="2020-01-01 (verified; perpetual launch 2019-09)",
        notes=(
            "Funding is charged at the START of the period it is quoted for and published "
            "immediately, so it is a point-in-time event with no release lag. max_age=24h "
            "means a missed settlement becomes NaN rather than being carried forward."
        ),
        used_by=("EXP-01-DERIVATIVES", "EXP-04-COMBINED"),
        feature_group="derivatives",
    ),
    "binance_futures": SourceSpec(
        key="binance_futures",
        label="Binance USD-M perpetual futures klines",
        endpoint="https://fapi.binance.com/fapi/v1/klines",
        frequency="1h",
        release_lag=pd.Timedelta(0),
        max_age=None,
        intrabar=True,
        historically_available=True,
        earliest_observed="2019-09-24 (verified to at least 2020-01-01 hourly)",
        notes="Used for futures price/volume and for the futures-spot basis.",
        used_by=("EXP-01-DERIVATIVES", "EXP-04-COMBINED"),
        feature_group="derivatives",
    ),
    "fear_greed": SourceSpec(
        key="fear_greed",
        label="Crypto Fear & Greed Index (Alternative.me)",
        endpoint="https://api.alternative.me/fng/",
        frequency="1d",
        release_lag=DAY,
        max_age=pd.Timedelta(days=3),
        intrabar=True,
        historically_available=True,
        earliest_observed="2018-02-01 (verified: 3,158 daily observations retrieved)",
        notes=(
            "A reading for day D describes that day, so it is treated as knowable from "
            "D+1 00:00 UTC (release_lag = 1 day). max_age=3d avoids carrying a stale reading "
            "indefinitely. The API ignores start_date, so full history is fetched at once."
        ),
        used_by=("EXP-02-SENTIMENT", "EXP-04-COMBINED"),
        feature_group="sentiment",
    ),
    # ---------------------------------------------------------------- unavailable
    "binance_open_interest": SourceSpec(
        key="binance_open_interest",
        label="Binance futures open interest",
        endpoint="https://fapi.binance.com/futures/data/openInterestHist",
        frequency="1h",
        release_lag=pd.Timedelta(0),
        max_age=None,
        intrabar=True,
        historically_available=False,
        notes="",
        unavailable_reason=(
            "NOT AVAILABLE HISTORICALLY. The endpoint rejects startTime (-1130) and serves a "
            "rolling window of only ~30 days (verified: 500 hourly rows ~= 20.8 days, 31 daily "
            "rows). It cannot be joined to a multi-year backtest, so open_interest and its "
            "derivatives are excluded. This is the single most important missing positioning "
            "signal and is a genuine limitation of the study, not an implementation gap."
        ),
        feature_group="derivatives",
    ),
    "binance_long_short_ratio": SourceSpec(
        key="binance_long_short_ratio",
        label="Binance global long/short account ratio",
        endpoint="https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        frequency="1h",
        release_lag=pd.Timedelta(0),
        max_age=None,
        intrabar=True,
        historically_available=False,
        notes="",
        unavailable_reason=(
            "NOT AVAILABLE HISTORICALLY. Same ~30 day rolling window as open interest "
            "(verified: 500 hourly rows span ~3 weeks)."
        ),
        feature_group="derivatives",
    ),
    "binance_order_book": SourceSpec(
        key="binance_order_book",
        label="Binance order book depth",
        endpoint="https://api.binance.com/api/v3/depth",
        frequency="snapshot",
        release_lag=pd.Timedelta(0),
        max_age=None,
        intrabar=False,
        historically_available=False,
        notes="",
        unavailable_reason=(
            "NO HISTORY AT ALL. /api/v3/depth returns a current snapshot and accepts no "
            "historical parameter. Sampling it during a backfill and attaching those readings "
            "to 2022 rows would fabricate microstructure that was never observed, so "
            "bid_ask_spread, bid/ask volume, order_book_imbalance and the order_imbalance_1m/"
            "5m/15m family are absent by design. EXP-03 measures trade flow instead."
        ),
        feature_group="microstructure",
    ),
}


# --------------------------------------------------------------------------- quality

@dataclass
class SourceQuality:
    """Measured quality of one source over the feature grid."""

    key: str
    rows: int = 0
    covered: int = 0
    duplicate_timestamps: int = 0
    missing_values: int = 0
    first_observed: str | None = None
    last_observed: str | None = None
    first_covered: str | None = None
    last_covered: str | None = None
    notes: str = ""

    @property
    def coverage_pct(self) -> float:
        return round(100.0 * self.covered / self.rows, 4) if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.key,
            "rows": self.rows,
            "covered": self.covered,
            "coverage_pct": self.coverage_pct,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_values": self.missing_values,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "first_covered": self.first_covered,
            "last_covered": self.last_covered,
            "notes": self.notes,
        }


def measure_source(
    key: str,
    frame: pd.DataFrame | None,
    *,
    feature_index: pd.DatetimeIndex | None = None,
    aligned: pd.DataFrame | None = None,
) -> SourceQuality:
    """Measure a source's coverage and integrity.

    When ``aligned`` is supplied the coverage is measured against the *feature
    grid*, which is the number that actually determines how much history each
    experiment can use.  Raw row counts alone are misleading: a daily source
    can have 100% of its own rows present and still cover only 4% of the
    hourly feature grid.
    """
    spec = SOURCE_CONTRACTS.get(key)
    quality = SourceQuality(key=key, notes=(spec.unavailable_reason or spec.notes) if spec else "")

    if frame is not None and len(frame):
        quality.rows = len(frame)
        quality.duplicate_timestamps = int(frame.index.duplicated().sum())
        quality.missing_values = int(frame.isna().sum().sum())
        quality.first_observed = str(frame.index.min())
        quality.last_observed = str(frame.index.max())

    if aligned is not None and feature_index is not None:
        quality.rows = int(len(feature_index))
        usable = aligned.notna().any(axis=1)
        quality.covered = int(usable.sum())
        if usable.any():
            quality.first_covered = str(feature_index[usable][0])
            quality.last_covered = str(feature_index[usable][-1])
    return quality


def build_availability_matrix(
    measured: Mapping[str, SourceQuality] | None = None,
) -> pd.DataFrame:
    """Build ``reports/data_availability.csv``.

    Includes unavailable sources as explicit rows with
    ``historical_available = False`` and a reason, so the absence is visible
    rather than inferred from a missing row.
    """
    rows: list[dict[str, Any]] = []
    for key, spec in SOURCE_CONTRACTS.items():
        row = spec.to_row()
        quality = (measured or {}).get(key)
        if quality is not None and spec.historically_available:
            row["start_date"] = quality.first_covered or quality.first_observed or ""
            row["end_date"] = quality.last_covered or quality.last_observed or ""
            row["coverage"] = f"{quality.coverage_pct}%"
            row["row_count"] = quality.rows
            row["missing_values"] = quality.missing_values
            row["duplicate_timestamps"] = quality.duplicate_timestamps
        else:
            row["coverage"] = "0%"
            row["row_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_group", "source"]).reset_index(drop=True)


def write_quality_reports(
    measured: Mapping[str, SourceQuality],
    out_dir,
) -> dict[str, str]:
    """Write ``data_quality.json`` and ``data_availability.csv``."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_availability_matrix(measured)

    csv_path = out / "data_availability.csv"
    matrix.to_csv(csv_path, index=False)

    payload = {
        "timestamp_roles": list(TIMESTAMP_ROLES),
        "sources": {key: q.to_dict() for key, q in measured.items()},
        "unavailable_sources": {
            key: {
                "label": spec.label,
                "reason": spec.unavailable_reason,
                "endpoint": spec.endpoint,
            }
            for key, spec in SOURCE_CONTRACTS.items()
            if not spec.historically_available
        },
    }
    json_path = save_json(payload, out / "data_quality.json")
    logger.info(
        "wrote data availability matrix (%d sources, %d unavailable) -> %s",
        len(matrix), sum(1 for s in SOURCE_CONTRACTS.values() if not s.historically_available), csv_path,
    )
    return {"availability_csv": str(csv_path), "quality_json": str(json_path)}
