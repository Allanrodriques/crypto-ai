"""Binance USD-M perpetual funding rates.

Endpoint
--------
``GET https://fapi.binance.com/fapi/v1/fundingRate``

Historical availability
-----------------------
Funding has been published since the perpetual launch (2019-09).  The endpoint
accepts ``startTime`` and pages backwards/forwards without a hard date cap, so
full history is retrievable.  Verified empirically: the earliest row available
for ``BTCUSDT`` is ``2020-01-01 00:00 UTC`` for the start we request, and the
series is continuous thereafter.

Frequency
---------
Every 8 hours for most symbols (00:00, 08:00, 16:00 UTC).  A few symbols
changed interval historically; the frame keeps whatever cadence the exchange
actually returned rather than assuming 8h.

Availability semantics (important)
----------------------------------
A funding *payment* is charged at the **start** of the period it is quoted for,
and the settled rate is published immediately.  The rate for 00:00 is therefore
known at 00:00 - it is not a summary of the period that follows.  This is why
the funding contract in :mod:`src.alignment.availability` uses
``release_lag = 0`` and ``intrabar = False``: it is a point-in-time event, not a
period aggregate.

That distinction matters.  Treating funding like a candle and lagging it 8
hours would throw away real information; treating a candle as a point event
would leak.  Getting this backwards is exactly the class of bug the alignment
module exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
import requests

from src.data.transport import build_session, get_json
from src.utils import get_logger, interval_to_milliseconds, timestamp_to_utc

logger = get_logger("data.sources.funding")

FUNDING_HOSTS = (
    "https://fapi.binance.com",
    "https://fapi.binance.com",
)
FUNDING_PATH = "/fapi/v1/fundingRate"

#: The endpoint returns *at most* 500 funding records per call regardless of the
#: requested ``limit``.  Asking for 1000 is not an error, but it makes
#: ``len(rows) < page_limit`` true on a full page, so a forward-paging loop
#: treats the very first response as the end of history and silently keeps only
#: the most recent ~5 months.  Match the real cap so the completeness check below
#: means what it says.
FUNDING_PAGE_LIMIT = 500


@dataclass(frozen=True)
class FundingSettings:
    """Tuning for the funding downloader."""

    symbol: str = "BTCUSDT"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    hosts: tuple[str, ...] = FUNDING_HOSTS
    page_limit: int = FUNDING_PAGE_LIMIT
    pause_seconds: float = 0.4
    max_pages: int = 5_000


def _ms(stamp: pd.Timestamp | None) -> int | None:
    if stamp is None:
        return None
    return int(timestamp_to_utc(stamp).value // 1_000_000)


def _iter_pages(settings: FundingSettings, session: requests.Session) -> Iterator[list[dict]]:
    """Page *backwards* through funding history using ``endTime`` cursors.

    Direction matters.  These Binance history endpoints cap each response at a
    few hundred records and return them oldest-first.  Paging forward with a
    ``startTime`` cursor therefore begins at the *oldest allowed* record, walks
    towards the present, and stops after one page - which silently truncated
    funding to the most recent 5 months and left the derivatives experiment
    running on 9.6% coverage.  Anchoring at ``end`` and stepping ``endTime``
    backwards walks the full history instead.
    """
    start_bound = _ms(settings.start)
    end_cursor = _ms(settings.end)
    seen: set[int] = set()
    for _ in range(settings.max_pages):
        params: dict[str, object] = {
            "symbol": settings.symbol,
            "limit": settings.page_limit,
        }
        if end_cursor is not None:
            params["endTime"] = end_cursor
        if start_bound is not None:
            params["startTime"] = start_bound

        rows = None
        last_error: Exception | None = None
        for host in settings.hosts:
            try:
                rows = get_json(session, f"{host}{FUNDING_PATH}", params=params)
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                logger.debug("funding host %s failed: %s", host, exc)
        if rows is None:
            raise RuntimeError(f"All funding hosts failed: {last_error}")
        if not isinstance(rows, list) or not rows:
            return

        fresh = [r for r in rows if int(r["fundingTime"]) not in seen]
        for row in rows:
            seen.add(int(row["fundingTime"]))
        if fresh:
            yield fresh
        if len(rows) < settings.page_limit:
            return

        oldest = min(int(r["fundingTime"]) for r in rows)
        if start_bound is not None and oldest <= start_bound:
            return  # reached the requested start
        if end_cursor is not None and oldest >= end_cursor:
            return  # cursor failed to move; stop rather than loop forever
        end_cursor = oldest - 1
        if settings.pause_seconds:
            time.sleep(settings.pause_seconds)


def fetch_funding_rates(
    settings: FundingSettings | None = None,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download funding rates.

    Returns a frame indexed by ``timestamp`` (the funding event time) with
    ``funding_rate`` and the notional rate over 8h, plus ``available_at`` equal
    to ``timestamp`` because a settled funding rate is known immediately.
    """
    settings = settings or FundingSettings()
    owns_session = session is None
    session = session or build_session()
    try:
        chunks = list(_iter_pages(settings, session))
    finally:
        if owns_session:
            session.close()

    if not chunks:
        raise RuntimeError(
            f"No funding history returned for {settings.symbol}; the symbol may not "
            "list on USD-M futures"
        )

    rows = [r for page in chunks for r in page]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    # Upstream occasionally returns a stray millisecond offset (e.g.
    # 08:00:00.008).  Flooring to the 8h funding grid keeps the index on the
    # settlement boundary, which the as-of alignment relies on.
    frame["timestamp"] = frame["timestamp"].dt.floor("8h")
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = (
        frame[["timestamp", "funding_rate"]]
        .dropna(subset=["funding_rate"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    frame["funding_rate_8h"] = frame["funding_rate"]
    # A funding rate is a point-in-time settlement, available at once.
    frame["available_at"] = frame.index
    logger.info(
        "funding_rate %s: %d rows, %s -> %s",
        settings.symbol,
        len(frame),
        frame.index[0],
        frame.index[-1],
    )
    return frame


def funding_cadence(frame: pd.DataFrame) -> pd.Timedelta | None:
    """Observed modal spacing between funding events (for the availability matrix)."""
    if len(frame) < 3:
        return None
    deltas = pd.Series(frame.index).diff().dropna()
    if deltas.empty:
        return None
    return pd.Timedelta(deltas.mode().iloc[0])
