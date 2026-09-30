"""Low-level Binance public market-data client.

Scope
-----
Only **public** endpoints are used:

* ``GET /api/v3/klines``       - historical candlesticks (no API key)
* ``GET /api/v3/exchangeInfo`` - validate symbol / interval support
* ``GET /api/v3/time``         - server clock, used to detect host drift

The primary host is ``https://data-api.binance.vision`` (market data only).  The
client transparently fails over to a secondary host if the primary is blocked
(HTTP 451 geo-restriction) or unavailable.

Reliability contract
--------------------
* automatic pagination over ``startTime``/``endTime`` windows
* exponential backoff with jitter on 418 / 429 / 5xx and on transport errors
* client-side token-bucket rate limiting derived from Binance's documented
  request weights (2 per kline request, 1200 weight/minute/IP)
* HTTP 400 fails fast (malformed request) instead of burning retries
* pagination guards against a non-advancing cursor, which would loop forever
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils import (
    format_timestamp,
    get_logger,
    interval_to_milliseconds,
    parse_date,
    timestamp_to_utc,
)

logger = get_logger("data.binance_client")

KLINES_PATH = "/api/v3/klines"
EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
TIME_PATH = "/api/v3/time"

#: Column order returned by ``/api/v3/klines``.  ``ignore`` (index 11) is
#: documented as "ignore" and is therefore dropped rather than stored.
KLINE_FIELDS = (
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

#: Columns retained after parsing.  ``ignore`` is dropped; the rest of the
#: metadata (quote volume, trade count, taker-buy volume) is small and useful.
KLINE_COLUMNS = (
    "timestamp",
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
)

#: Error codes that are worth retrying even though they arrive with HTTP 200/400.
_RETRYABLE_BINANCE_CODES = {-1003, -1015}


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns an error that is not worth retrying."""

    def __init__(self, message: str, *, status_code: int | None = None, code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class BinanceRateLimitError(BinanceAPIError):
    """Raised when the retry budget for 418/429 responses is exhausted."""


class BinanceUnavailableError(BinanceAPIError):
    """Raised when every configured host fails."""


# --------------------------------------------------------------------------- rate limiting

class RateLimiter:
    """Token-bucket limiter sized from Binance's documented request weights.

    The bucket refills continuously at ``limit_per_minute`` tokens per minute and
    starts full, so the first requests are not penalised.  ``consume`` blocks
    until enough tokens are available.
    """

    def __init__(
        self,
        limit_per_minute: int = 1200,
        weight_per_request: int = 2,
        safety_factor: float = 0.9,
        min_interval: float = 0.0,
    ) -> None:
        if limit_per_minute <= 0 or weight_per_request <= 0:
            raise ValueError("Rate limit and request weight must be positive")
        self.capacity = float(limit_per_minute) * safety_factor
        self.tokens = self.capacity
        self.refill_per_second = float(limit_per_minute) / 60.0
        self.weight_per_request = weight_per_request
        self.min_interval = max(0.0, float(min_interval))
        self._last_refill = time.monotonic()
        self._last_call = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self._last_refill = now

    def consume(self, weight: int | None = None) -> None:
        """Block until ``weight`` tokens are available, then spend them."""
        cost = float(self.weight_per_request if weight is None else weight)
        while True:
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                break
            time.sleep(max(0.05, (cost - self.tokens) / self.refill_per_second))
        if self.min_interval:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()


# --------------------------------------------------------------------------- client

@dataclass
class ClientSettings:
    """Connection tuning, normally populated from the ``data:`` config block."""

    base_url: str = "https://data-api.binance.vision"
    fallback_base_urls: Sequence[str] = ()
    page_size: int = 1000
    weight_per_request: int = 2
    weight_limit_per_minute: int = 1200
    request_timeout_seconds: float = 30.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    retry_backoff_max_seconds: float = 90.0
    retry_status_codes: Sequence[int] = (418, 429, 500, 502, 503, 504)
    pause_between_requests_seconds: float = 0.0

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> "ClientSettings":
        return cls(
            base_url=str(data_config.get("base_url", cls.base_url)),
            fallback_base_urls=tuple(data_config.get("fallback_base_urls", ()) or ()),
            page_size=int(data_config.get("page_size", cls.page_size)),
            weight_per_request=int(data_config.get("weight_per_request", cls.weight_per_request)),
            weight_limit_per_minute=int(data_config.get("weight_limit_per_minute", cls.weight_limit_per_minute)),
            request_timeout_seconds=float(data_config.get("request_timeout_seconds", cls.request_timeout_seconds)),
            max_retries=int(data_config.get("max_retries", cls.max_retries)),
            retry_backoff_seconds=float(data_config.get("retry_backoff_seconds", cls.retry_backoff_seconds)),
            retry_backoff_max_seconds=float(data_config.get("retry_backoff_max_seconds", cls.retry_backoff_max_seconds)),
            retry_status_codes=tuple(data_config.get("retry_status_codes", cls.retry_status_codes)),
            pause_between_requests_seconds=float(
                data_config.get("pause_between_requests_seconds", cls.pause_between_requests_seconds)
            ),
        )


class BinanceClient:
    """Paginating HTTP client for Binance public klines.

    Parameters
    ----------
    settings:
        Connection tuning.  Use :meth:`ClientSettings.from_config` to build it
        from the YAML config.
    session:
        Optional pre-built :class:`requests.Session` (injected by tests).
    """

    def __init__(self, settings: ClientSettings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or ClientSettings()
        self._owns_session = session is None
        self.session = session or self._build_session()
        self._hosts: list[str] = [self.settings.base_url, *self.settings.fallback_base_urls]
        self._active_host = self._hosts[0]
        self.rate_limiter = RateLimiter(
            limit_per_minute=self.settings.weight_limit_per_minute,
            weight_per_request=self.settings.weight_per_request,
            min_interval=self.settings.pause_between_requests_seconds,
        )
        self.stats: dict[str, int] = {"requests": 0, "retries": 0, "pages": 0, "rows": 0}

    # ------------------------------------------------------------- plumbing

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=0,  # retries are handled explicitly so backoff stays configurable
            connect=0,
            read=0,
            status=0,
            backoff_factor=0,
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": "crypto-ml-predictor/1.0 (research; public market data only)",
                "Accept": "application/json",
            }
        )
        return session

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _sleep_backoff(self, attempt: int, reason: str) -> float:
        """Exponential backoff with full jitter, capped at ``retry_backoff_max``."""
        ceiling = min(self.settings.retry_backoff_max_seconds, self.settings.retry_backoff_seconds * (2**attempt))
        delay = ceiling / 2.0 + _jitter(ceiling / 2.0)
        logger.warning("Binance request failed (%s) — retry %d/%d in %.1fs", reason, attempt + 1,
                       self.settings.max_retries, delay)
        time.sleep(delay)
        return delay

    def _rotate_host(self) -> str | None:
        """Switch to the next configured host.  Returns ``None`` when exhausted."""
        remaining = [h for h in self._hosts if h != self._active_host]
        if not remaining:
            return None
        self._active_host = remaining[0]
        logger.info("Switching Binance host to %s", self._active_host)
        return self._active_host

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET with retries, host failover and client-side rate limiting."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        host_exchanges = 0
        attempt = 0
        last_error: Exception | None = None

        while True:
            url = f"{self._active_host.rstrip('/')}{path}"
            try:
                self.rate_limiter.consume()
                self.stats["requests"] += 1
                response = self.session.get(url, params=params, timeout=self.settings.request_timeout_seconds)
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    self._sleep_backoff(attempt, type(exc).__name__)
                    attempt += 1
                    self.stats["retries"] += 1
                    continue
                host_exchanges += 1
                if not self._rotate_host() or host_exchanges > len(self._hosts):
                    raise BinanceUnavailableError(f"All Binance hosts failed: {last_error}") from exc
                attempt = 0
                continue

            status = response.status_code

            if status == 451 and not self._rotate_host():
                raise BinanceUnavailableError(
                    f"{self._active_host} returned HTTP 451 (geo-restricted). "
                    "Set data.base_url to https://data-api.binance.vision in config/config.yaml."
                )

            if status in self.settings.retry_status_codes or status >= 500:
                last_error = BinanceRateLimitError(f"HTTP {status} from {url}", status_code=status)
                if attempt < self.settings.max_retries:
                    self._sleep_backoff(attempt, f"HTTP {status}")
                    attempt += 1
                    self.stats["retries"] += 1
                    continue
                if not self._rotate_host():
                    raise BinanceRateLimitError(
                        f"Binance rate limit / server error persisted after "
                        f"{self.settings.max_retries} retries (last status {status}). "
                        "Wait ~60s before resuming; the downloader can resume from disk."
                    ) from last_error
                attempt = 0
                continue

            if status >= 400:
                detail = _extract_error(response)
                raise BinanceAPIError(
                    f"Binance rejected {url} with HTTP {status}: {detail}",
                    status_code=status,
                    code=detail.get("code") if isinstance(detail, dict) else None,
                )

            try:
                return response.json()
            except ValueError as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    self._sleep_backoff(attempt, "malformed JSON")
                    attempt += 1
                    self.stats["retries"] += 1
                    continue
                raise BinanceAPIError(f"Binance returned non-JSON content from {url}") from exc

    # ------------------------------------------------------------- endpoints

    def get_server_time(self) -> pd.Timestamp:
        """Return Binance's server time as a UTC timestamp."""
        payload = self._get(TIME_PATH)
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise BinanceAPIError(f"Unexpected /time payload: {payload!r}")
        return timestamp_to_utc(int(payload["serverTime"]))

    def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """Return exchange metadata, optionally filtered to one symbol."""
        params = {"symbol": symbol.upper()} if symbol else None
        payload = self._get(EXCHANGE_INFO_PATH, params)
        if not isinstance(payload, dict) or "symbols" not in payload:
            raise BinanceAPIError(f"Unexpected /exchangeInfo payload for symbol={symbol!r}")
        return payload

    def assert_symbol_and_interval(self, symbol: str, interval: str) -> None:
        """Fail fast if the pair does not exist or the interval is not listed.

        Binance's ``exchangeInfo`` response lists supported intervals under
        ``filters[].intervals``; when that is absent (older responses) the
        canonical interval set is used instead.
        """
        info = self.get_exchange_info(symbol.upper())
        symbols = info.get("symbols", [])
        if not symbols:
            raise BinanceAPIError(
                f"Symbol {symbol.upper()!r} is not listed on Binance. Check config data.symbol."
            )
        status = symbols[0].get("status", "TRADING")
        if status not in {"TRADING", "HALT", "BREAK"}:
            raise BinanceAPIError(f"Symbol {symbol.upper()!r} has unusable status {status!r}")

        supported: set[str] | None = None
        for flt in symbols[0].get("filters", []) or []:
            if "intervals" in flt:
                supported = set(flt["intervals"])
        if supported is not None and interval not in supported:
            raise BinanceAPIError(
                f"Interval {interval!r} is not supported for {symbol.upper()}. "
                f"Supported: {sorted(supported)}"
            )

    # --------------------------------------------------------------- klines

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | pd.Timestamp | None = None,
        end_time: datetime | pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        """Fetch a single page of klines.

        Returns the raw JSON rows.  Use :meth:`iter_klines` for full history.
        """
        limit = min(int(limit or self.settings.page_size), 1000)
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(timestamp_to_utc(start_time).value // 1_000_000)
        if end_time is not None:
            params["endTime"] = int(timestamp_to_utc(end_time).value // 1_000_000)

        payload = self._get(KLINES_PATH, params)
        if not isinstance(payload, list):
            raise BinanceAPIError(f"Unexpected klines payload: {payload!r}")
        self.stats["pages"] += 1
        self.stats["rows"] += len(payload)
        return payload

    def iter_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime | pd.Timestamp | None = None,
        end_time: datetime | pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield successive pages of parsed klines from ``start_time`` to ``end_time``.

        Pagination advances ``startTime`` to ``last_open_time + interval_ms`` so
        pages neither overlap nor skip candles.  A guard breaks the loop if a
        page fails to advance the cursor, which protects against an upstream
        pagination quirk turning into an infinite request loop.
        """
        step_ms = interval_to_milliseconds(interval)
        cursor = timestamp_to_utc(start_time) if start_time is not None else None
        end_stamp = timestamp_to_utc(end_time) if end_time is not None else None
        page_size = min(int(limit or self.settings.page_size), 1000)
        page_index = 0

        while True:
            rows = self.fetch_klines(symbol, interval, start_time=cursor, end_time=end_stamp, limit=page_size)
            if not rows:
                break

            frame = parse_klines(rows, symbol=symbol, interval=interval)
            if frame.empty:
                break
            page_index += 1
            yield frame

            last_open_ms = int(frame.index[-1].value // 1_000_000)
            next_cursor_ms = last_open_ms + step_ms

            if end_stamp is not None:
                end_ms = int(end_stamp.value // 1_000_000)
                if next_cursor_ms > end_ms:
                    break
            if len(rows) < page_size:
                break
            if next_cursor_ms <= (cursor.value // 1_000_000 if cursor is not None else 0):
                logger.error(
                    "Pagination cursor stalled at %s — aborting to avoid an infinite loop",
                    format_timestamp(next_cursor_ms),
                )
                break
            cursor = pd.to_datetime(next_cursor_ms, unit="ms", utc=True)

        logger.info("Pagination complete for %s %s: %d page(s)", symbol.upper(), interval, page_index)


# --------------------------------------------------------------------------- parsing

def parse_klines(rows: Sequence[Sequence[Any]], symbol: str = "", interval: str = "") -> pd.DataFrame:
    """Convert raw ``/api/v3/klines`` rows into a typed, UTC-indexed DataFrame.

    The returned frame has a tz-aware ``DatetimeIndex`` named ``timestamp`` plus
    the columns in :data:`KLINE_COLUMNS`.  Numeric fields are coerced to
    ``float64``/``int64``; unparseable values become ``NaN`` and are surfaced
    later by :mod:`src.data.validator` rather than being silently dropped.
    """
    if not rows:
        index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in KLINE_COLUMNS}, index=index)
        return empty

    df = pd.DataFrame(list(rows), columns=list(KLINE_FIELDS))
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

    # Epoch milliseconds -> tz-aware UTC.
    df["timestamp"] = [timestamp_to_utc(v) for v in df["open_time"]]
    df["close_time"] = [timestamp_to_utc(v) for v in df["close_time"]]

    # Keep the documented metadata, drop the "ignore" column.
    df = df.loc[:, list(KLINE_COLUMNS)]
    df.index = pd.DatetimeIndex(df["timestamp"], name="timestamp")
    df = df.drop(columns=["timestamp"])

    if symbol:
        df.attrs["symbol"] = symbol.upper()
    if interval:
        df.attrs["interval"] = interval
    return df


def _jitter(ceiling: float) -> float:
    import random as _random

    return _random.uniform(0.0, ceiling)


def _extract_error(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:300]


def _default_intervals() -> list[str]:
    return [
        "1s", "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    ]


def parse_config_date(value: Any) -> pd.Timestamp | None:
    """Re-exported for convenience so scripts only import from one module."""
    return parse_date(value)
