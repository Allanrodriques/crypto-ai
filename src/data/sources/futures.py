"""Binance USD-M perpetual futures klines (price, volume, basis).

Endpoint
--------
``GET https://fapi.binance.com/fapi/v1/klines``

Historical availability
-----------------------
Continuous from the perpetual launch.  Verified empirically: hourly candles
exist back to at least ``2020-01-01``, so futures coverage comfortably spans
the V1 spot window.

Why it matters
--------------
Two things come from this source:

* the **basis** - perpetual mark/close versus spot close - which is a direct
  read on futures positioning relative to spot, and
* futures **volume**, which differs structurally from spot volume.

The basis is a genuine derivative-positioning signal and is available over the
whole period.  Open interest, which would be the strongest positioning
signal, is *not* available historically (see :mod:`src.data.quality`) and is
therefore excluded rather than reconstructed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
import requests

from src.data.transport import build_session, get_json
from src.utils import get_logger, interval_to_milliseconds, timestamp_to_utc

logger = get_logger("data.sources.futures")

FUTURES_HOSTS = ("https://fapi.binance.com",)
FUTURES_KLINES_PATH = "/fapi/v1/klines"
FUTURES_PAGE_LIMIT = 1500

#: Column order of a futures kline row.  Same layout as the spot endpoint.
_FUTURES_FIELDS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
_NUMERIC = _FUTURES_FIELDS[1:6] + _FUTURES_FIELDS[7:11]


@dataclass(frozen=True)
class FuturesSettings:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    hosts: tuple[str, ...] = FUTURES_HOSTS
    page_limit: int = FUTURES_PAGE_LIMIT
    pause_seconds: float = 0.4
    max_pages: int = 5_000


def _ms(stamp: pd.Timestamp | None) -> int | None:
    if stamp is None:
        return None
    return int(timestamp_to_utc(stamp).value // 1_000_000)


def _iter_pages(settings: FuturesSettings, session: requests.Session) -> Iterator[list[list]]:
    """Page *backwards* through futures klines using ``endTime`` cursors.

    The klines endpoint returns at most 1500 candles per call.  A forward
    ``startTime`` cursor starts at the oldest allowed candle, walks towards the
    present, and stops after a single page - which silently limited futures
    history to ~62 days and left the derivatives group at 3.6% coverage.
    """
    step_ms = interval_to_milliseconds(settings.interval)
    start_bound = _ms(settings.start)
    end_cursor = _ms(settings.end)
    seen: set[int] = set()
    for _ in range(settings.max_pages):
        params: dict[str, object] = {
            "symbol": settings.symbol,
            "interval": settings.interval,
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
                rows = get_json(session, f"{host}{FUTURES_KLINES_PATH}", params=params)
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                logger.debug("futures host %s failed: %s", host, exc)
        if rows is None:
            raise RuntimeError(f"All futures hosts failed: {last_error}")
        if not isinstance(rows, list) or not rows:
            return

        fresh = [row for row in rows if int(row[0]) not in seen]
        for row in rows:
            seen.add(int(row[0]))
        if fresh:
            yield fresh
        if len(rows) < settings.page_limit:
            return

        oldest = min(int(row[0]) for row in rows)
        if start_bound is not None and oldest <= start_bound:
            return
        if end_cursor is not None and oldest >= end_cursor:
            return
        end_cursor = oldest - 1
        if settings.pause_seconds:
            time.sleep(settings.pause_seconds)


def fetch_futures_klines(
    settings: FuturesSettings | None = None,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download perpetual futures klines as a UTC-indexed OHLCV frame."""
    settings = settings or FuturesSettings()
    owns_session = session is None
    session = session or build_session()
    try:
        pages = list(_iter_pages(settings, session))
    finally:
        if owns_session:
            session.close()

    if not pages:
        raise RuntimeError(f"No futures klines returned for {settings.symbol}")

    raw = [row for page in pages for row in page]
    frame = pd.DataFrame(raw, columns=list(_FUTURES_FIELDS))
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in _NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    keep = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume"]
    out = (
        frame[keep]
        .dropna(subset=["open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    out = out.rename(columns={"volume": "futures_volume", "quote_volume": "futures_quote_volume"})
    logger.info(
        "futures klines %s %s: %d rows, %s -> %s",
        settings.symbol,
        settings.interval,
        len(out),
        out.index[0],
        out.index[-1],
    )
    return out
