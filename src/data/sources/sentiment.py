"""Crypto Fear & Greed Index (Alternative.me) - the sentiment source.

Endpoint
--------
``GET https://api.alternative.me/fng/``

Historical availability
-----------------------
Verified empirically.  The index starts ``2018-02-01`` and
``?limit=0`` returns the complete series in a single response (3,158
observations at the time of writing).  The ``start_date`` query parameter is
**ignored** by the API - it always answers with the most recent page - so
history is obtained by requesting everything at once rather than by paging
backwards.  This module does exactly that and verifies the returned earliest
timestamp, so a silently truncated response is detected instead of being
treated as full coverage.

Frequency
---------
One reading per day at 00:00 UTC.  Coverage therefore lands on 1 of every 24
hourly candles; the alignment module as-of joins it onto the hourly grid and
each reading is reused for the day it describes, with a ``max_age`` guard.

Interpretation caveat
----------------------
The index publishes a *label* alongside the number (``Extreme Fear`` ...
``Extreme Greed``).  This project deliberately does **not** encode any of those
labels as a directional signal, and does not map high values to bullish.  Only
the numeric value and its own changes/rolling statistics become features, so
the model is free to learn whatever relationship exists - including none.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from src.data.transport import build_session, get_json
from src.utils import get_logger

logger = get_logger("data.sources.sentiment")

FEAR_GREED_URL = "https://api.alternative.me/fng/"

#: The index itself begins on this date; used as an expected-coverage bound.
FEAR_GREED_EPOCH = pd.Timestamp("2018-02-01", tz="UTC")

#: Published alongside the number.  Recorded for provenance only - never used
#: as a directional feature.
FEAR_GREED_CLASSES = (
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
)


@dataclass(frozen=True)
class SentimentSettings:
    limit: int = 0          # 0 == "all available observations"
    date_format: str = "us"
    expected_start: pd.Timestamp = FEAR_GREED_EPOCH


def _parse_date(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def fetch_fear_greed(
    settings: SentimentSettings | None = None,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download the full Fear & Greed history.

    Returns a frame indexed by ``timestamp`` (00:00 UTC daily) with
    ``fear_greed_value``, the published ``classification`` (provenance only)
    and ``available_at``.

    ``available_at`` is set to the *following* midnight: a daily reading
    describes the day, and treating it as knowable at 00:00 of its own day would
    hand the model a value describing hours it has not lived through.  This is
    the conservative assumption required by the alignment policy.
    """
    settings = settings or SentimentSettings()
    owns_session = session is None
    session = session or build_session()
    try:
        payload = get_json(
            session,
            FEAR_GREED_URL,
            params={"limit": settings.limit, "format": "json", "date_format": settings.date_format},
        )
    finally:
        if owns_session:
            session.close()

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        raise RuntimeError("Fear & Greed endpoint returned no data")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = frame["timestamp"].map(_parse_date)
    frame["fear_greed_value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = (
        frame[["timestamp", "fear_greed_value", "value_classification"]]
        .rename(columns={"value_classification": "classification"})
        .dropna(subset=["fear_greed_value"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    # Conservative: a reading for day D is usable from D+1 00:00 UTC.
    frame["available_at"] = frame.index + pd.Timedelta(days=1)

    earliest = frame.index[0]
    if earliest > settings.expected_start + pd.Timedelta(days=7):
        logger.warning(
            "Fear & Greed history starts %s but the index is documented to start %s; "
            "the response may have been truncated",
            earliest.date(),
            settings.expected_start.date(),
        )
    logger.info(
        "fear_greed: %d rows, %s -> %s (classes present: %s)",
        len(frame),
        earliest.date(),
        frame.index[-1].date(),
        ", ".join(sorted(set(frame["classification"].dropna()))),
    )
    return frame
