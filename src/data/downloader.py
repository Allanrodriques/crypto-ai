"""Historical kline downloader: persistence, resume, dedup and gap repair.

The downloader owns everything the client deliberately does not: what goes on
disk, how an interrupted run is resumed, and how overlapping or missing ranges
are reconciled.  The client only knows how to fetch one page of candles.

On-disk layout
--------------
``data/raw/<SYMBOL>_<interval>.parquet``
    UTC-indexed OHLCV plus a small set of useful Binance metadata columns.
``data/raw/<SYMBOL>_<interval>.meta.json``
    Provenance sidecar: requested range, realised range, row count, gaps, and
    the wall-clock time of the fetch.

Both files are written atomically (temp file + ``os.replace``) so a crash can
never leave a half-written parquet that would poison the next resume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data.binance_client import BinanceClient, ClientSettings, parse_klines
from src.utils import (
    format_timestamp,
    get_logger,
    interval_to_milliseconds,
    parse_date,
    save_json,
    timestamp_to_utc,
    utc_now,
)

logger = get_logger("data.downloader")

#: Metadata columns that survive round-tripping through the store.
METADATA_COLUMNS = ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


# --------------------------------------------------------------------------- store

class KlineStore:
    """Parquet-backed (CSV fallback) storage for one symbol/interval series."""

    def __init__(self, directory: str | os.PathLike[str], symbol: str, interval: str) -> None:
        self.directory = Path(directory)
        self.symbol = symbol.upper()
        self.interval = interval
        self.data_path = self.directory / f"{self.symbol}_{self.interval}.parquet"
        self.csv_path = self.directory / f"{self.symbol}_{self.interval}.csv"
        self.meta_path = self.directory / f"{self.symbol}_{self.interval}.meta.json"

    def exists(self) -> bool:
        return self.data_path.exists() or self.csv_path.exists()

    def path(self) -> Path:
        return self.data_path if self.data_path.exists() else self.csv_path

    def load(self) -> pd.DataFrame:
        """Read the stored series, or an empty frame if nothing is stored yet."""
        if self.data_path.exists():
            df = pd.read_parquet(self.data_path)
        elif self.csv_path.exists():
            df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        else:
            return empty_klines()
        return normalise_klines(df)

    def save(self, df: pd.DataFrame) -> Path:
        """Write ``df`` atomically, preferring parquet and falling back to CSV."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.data_path.with_suffix(".parquet.tmp").exists():
            self.data_path.with_suffix(".parquet.tmp").unlink()
        try:
            tmp = self.data_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=True)
            os.replace(tmp, self.data_path)
            return self.data_path
        except Exception as exc:  # pragma: no cover - only without pyarrow
            logger.warning("Parquet write failed (%s); falling back to CSV", exc)
            tmp = self.csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=True, float_format="%.10f")
            os.replace(tmp, self.csv_path)
            return self.csv_path


# --------------------------------------------------------------------------- helpers

def empty_klines() -> pd.DataFrame:
    """An empty, correctly-typed kline frame with a UTC DatetimeIndex."""
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "quote_volume": pd.Series(dtype="float64"),
            "trades": pd.Series(dtype="Int64"),
            "taker_buy_volume": pd.Series(dtype="float64"),
            "taker_buy_quote_volume": pd.Series(dtype="float64"),
        },
        index=index,
    )


def normalise_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a UTC ``timestamp`` index, sorted ascending, deduplicated."""
    if df.empty:
        return empty_klines()

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = [timestamp_to_utc(v) for v in df["timestamp"]]
            df = df.set_index("timestamp")
        else:
            raise ValueError("Kline frame has neither a DatetimeIndex nor a 'timestamp' column")

    df.index = pd.DatetimeIndex([timestamp_to_utc(t) for t in df.index], name="timestamp")
    df = df.sort_index(kind="mergesort")
    df = df[~df.index.duplicated(keep="first")]

    for col in (*PRICE_COLUMNS, "volume", *METADATA_COLUMNS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df


def drop_incomplete_candle(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove the still-forming candle, if present.

    Binance returns the in-progress candle as the final kline.  Its OHLCV values
    are not final, so keeping it would both corrupt indicator values and
    misrepresent what was knowable at that timestamp.
    """
    if df.empty:
        return df
    now = timestamp_to_utc(now) if now is not None else pd.Timestamp(utc_now())
    if "close_time" in df.columns:
        return df.loc[df["close_time"] <= now]
    step_ms = interval_to_milliseconds(interval)
    last_closed_open = now.value // 1_000_000 - (now.value // 1_000_000) % step_ms - step_ms
    return df.loc[df.index.view("int64") // 1_000_000 <= last_closed_open]


def find_missing_ranges(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ``(gap_start, gap_end)`` candle-open pairs missing from ``index``.

    ``gap_end`` is inclusive of the last missing candle.  Only interior holes
    between ``start`` and ``end`` are reported.
    """
    step_ms = interval_to_milliseconds(interval)
    start_ms = int(timestamp_to_utc(start).value // 1_000_000)
    end_ms = int(timestamp_to_utc(end).value // 1_000_000)
    if end_ms < start_ms:
        return []

    expected = np.arange((start_ms // step_ms) * step_ms, end_ms + 1, step_ms, dtype="int64")
    if expected.size == 0:
        return []
    present = set((index.view("int64") // 1_000_000).tolist())
    missing = np.array([ms for ms in expected.tolist() if ms not in present], dtype="int64")
    if missing.size == 0:
        return []

    # Collapse consecutive missing milliseconds into contiguous candle runs.
    breaks = np.flatnonzero(np.diff(missing) != step_ms)
    starts = np.concatenate(([missing[0]], missing[breaks + 1]))
    ends = np.concatenate((missing[breaks], [missing[-1]]))
    return [
        (pd.to_datetime(s, unit="ms", utc=True), pd.to_datetime(e, unit="ms", utc=True))
        for s, e in zip(starts, ends)
    ]


# --------------------------------------------------------------------------- result

@dataclass
class DownloadResult:
    """Outcome of one :meth:`KlineDownloader.download` call."""

    symbol: str
    interval: str
    path: Path
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    requests_made: int
    pages_fetched: int
    duplicate_rows_dropped: int
    incomplete_rows_dropped: int
    gaps_repaired: int
    resumed_from: pd.Timestamp | None
    downloaded_at: pd.Timestamp
    duration_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "path": str(self.path),
            "rows": self.rows,
            "range_start": format_timestamp(self.start),
            "range_end": format_timestamp(self.end),
            "downloaded_at": self.downloaded_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "requests_made": self.requests_made,
            "pages_fetched": self.pages_fetched,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "incomplete_rows_dropped": self.incomplete_rows_dropped,
            "gaps_repaired": self.gaps_repaired,
            "resumed_from": format_timestamp(self.resumed_from) if self.resumed_from is not None else None,
            **self.extra,
        }


# --------------------------------------------------------------------------- downloader

class KlineDownloader:
    """Fetch and persist a full historical kline series.

    Example
    -------
    >>> downloader = KlineDownloader.from_config(load_config())      # doctest: +SKIP
    >>> result = downloader.download()                               # doctest: +SKIP
    >>> result.rows > 0                                              # doctest: +SKIP
    True
    """

    def __init__(
        self,
        client: BinanceClient,
        store: KlineStore,
        symbol: str,
        interval: str,
        *,
        start_date: Any = None,
        end_date: Any = None,
        drop_incomplete: bool = True,
        resume: bool = True,
        repair_gaps: bool = True,
        save_metadata: bool = True,
        checkpoint: bool = True,
    ) -> None:
        self.client = client
        self.store = store
        self.symbol = symbol.upper()
        self.interval = interval
        self.start_date = parse_date(start_date)
        self.end_date = parse_date(end_date)
        self.drop_incomplete = drop_incomplete
        self.resume = resume
        self.repair_gaps = repair_gaps
        self.save_metadata = save_metadata
        self.checkpoint = checkpoint

    @classmethod
    def from_config(
        cls,
        config: Config,
        client: BinanceClient | None = None,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> "KlineDownloader":
        data = config.data
        store = KlineStore(config.paths.raw_dir, symbol or config.symbol, interval or config.interval)
        active_client = client or BinanceClient(ClientSettings.from_config(data))
        return cls(
            active_client,
            store,
            symbol or config.symbol,
            interval or config.interval,
            start_date=start_date if start_date is not None else data.get("start_date"),
            end_date=end_date if end_date is not None else data.get("end_date"),
            drop_incomplete=bool(data.get("drop_incomplete_candle", True)),
            resume=bool(data.get("resume", True)),
            save_metadata=bool(data.get("save_metadata", True)),
        )

    # ------------------------------------------------------------- fetching

    def _fetch_range(
        self, start: pd.Timestamp, end: pd.Timestamp | None, existing: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Fetch ``[start, end]`` page by page, checkpointing to disk as it goes.

        Checkpointing is what makes a long download resumable: every completed
        page is flushed, so an interrupt costs at most one page of work.  The
        checkpoint always unions the new pages with ``existing`` so previously
        downloaded candles are never clobbered.
        """
        collected: list[pd.DataFrame] = []
        for page in self.client.iter_klines(self.symbol, self.interval, start_time=start, end_time=end):
            collected.append(page)
            if self.checkpoint:
                self.store.save(self._merge(pd.concat(collected), existing))
        if not collected:
            return empty_klines()
        return self._merge(pd.concat(collected), None)

    def _merge(self, fresh: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
        """Union two frames, keeping the newest copy of any duplicated candle."""
        parts = [f for f in (existing, fresh) if f is not None and not f.empty]
        if not parts:
            return empty_klines()
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")]
        return normalise_klines(combined)

    # ------------------------------------------------------------- main API

    def download(self) -> DownloadResult:
        """Download the configured range, resuming and repairing as needed."""
        import time as _time

        started = _time.perf_counter()
        downloaded_at = pd.Timestamp(utc_now())
        requests_before = self.client.stats["requests"]
        pages_before = self.client.stats["pages"]

        logger.info(
            "Downloading %s %s from %s to %s (resume=%s)",
            self.symbol, self.interval,
            format_timestamp(self.start_date), format_timestamp(self.end_date), self.resume,
        )
        self.client.assert_symbol_and_interval(self.symbol, self.interval)

        stored = self.store.load() if (self.resume and self.store.exists()) else empty_klines()
        resumed_from: pd.Timestamp | None = stored.index.max() if not stored.empty else None
        stored_rows = len(stored)

        if not stored.empty:
            logger.info(
                "Found %d stored candles %s .. %s on disk",
                stored_rows, format_timestamp(stored.index.min()), format_timestamp(stored.index.max()),
            )

        step_ms = interval_to_milliseconds(self.interval)
        step = pd.Timedelta(milliseconds=step_ms)

        # The newest candle Binance will let us keep: the still-forming one is dropped.
        last_closed = downloaded_at.floor(self.interval) - step
        upper = last_closed if self.end_date is None else min(self.end_date, last_closed)

        if stored.empty:
            lower = self.start_date or pd.Timestamp("2017-01-01", tz="UTC")
            if upper is not None and lower > upper:
                raise ValueError(
                    f"Requested start {format_timestamp(lower)} is after the last fully closed "
                    f"candle {format_timestamp(upper)}."
                )
            logger.info("Fetching %s .. %s", format_timestamp(lower), format_timestamp(upper))
            stored = self._fetch_range(lower, upper, existing=None)
        else:
            stored_min, stored_max = stored.index.min(), stored.index.max()

            # 1) Backfill anything requested before the earliest stored candle.
            if self.start_date is not None and self.start_date < stored_min:
                backfill_end = min(stored_min - step, upper or stored_min - step)
                if self.start_date <= backfill_end:
                    logger.info("Backfilling %s .. %s", format_timestamp(self.start_date), format_timestamp(backfill_end))
                    stored = self._merge(self._fetch_range(self.start_date, backfill_end, existing=stored), stored)

            # 2) Forward-fill anything after the latest stored candle.
            stored_max = stored.index.max()
            if upper is not None and stored_max < upper:
                logger.info("Forward-filling %s .. %s", format_timestamp(stored_max + step), format_timestamp(upper))
                stored = self._merge(self._fetch_range(stored_max + step, upper, existing=stored), stored)
            else:
                logger.info("Stored data already reaches %s; nothing forward to fetch", format_timestamp(stored_max))

        # 3) Repair interior holes (e.g. a resume that started mid-series).
        gaps_repaired = 0
        if self.repair_gaps and not stored.empty:
            gaps = find_missing_ranges(stored.index, stored.index.min(), stored.index.max(), self.interval)
            for gap_start, gap_end in gaps:
                n_missing = int((gap_end - gap_start) / step) + 1
                logger.warning(
                    "Repairing %d-candle gap %s .. %s", n_missing,
                    format_timestamp(gap_start), format_timestamp(gap_end),
                )
                stored = self._merge(self._fetch_range(gap_start, gap_end, existing=stored), stored)
                gaps_repaired += 1

        # --- final reconciliation ------------------------------------------------
        before_dedup = len(stored)
        stored = normalise_klines(stored)
        duplicates_dropped = before_dedup - len(stored)

        incomplete_dropped = 0
        if self.drop_incomplete and not stored.empty:
            pre = len(stored)
            stored = drop_incomplete_candle(stored, self.interval, now=downloaded_at)
            incomplete_dropped = pre - len(stored)

        stored = stored.loc[stored.index >= (self.start_date or stored.index.min())] if not stored.empty else stored
        if self.end_date is not None and not stored.empty:
            stored = stored.loc[stored.index <= self.end_date]

        path = self.store.save(stored)
        duration = _time.perf_counter() - started

        result = DownloadResult(
            symbol=self.symbol,
            interval=self.interval,
            path=path,
            rows=len(stored),
            start=stored.index.min() if not stored.empty else None,
            end=stored.index.max() if not stored.empty else None,
            requests_made=self.client.stats["requests"] - requests_before,
            pages_fetched=self.client.stats["pages"] - pages_before,
            duplicate_rows_dropped=duplicates_dropped,
            incomplete_rows_dropped=incomplete_dropped,
            gaps_repaired=gaps_repaired,
            resumed_from=resumed_from,
            downloaded_at=downloaded_at,
            duration_seconds=duration,
            extra={
                "requested_start": format_timestamp(self.start_date),
                "requested_end": format_timestamp(self.end_date),
                "pre_resume_rows_on_disk": stored_rows,
                "client_retries": self.client.stats["retries"],
                "drop_incomplete_candle": self.drop_incomplete,
            },
        )

        if self.save_metadata:
            save_json(result.to_metadata(), self.store.meta_path)
            logger.info("Wrote metadata sidecar %s", self.store.meta_path)

        logger.info(
            "Saved %d candles to %s (%s .. %s) in %.1fs using %d request(s)",
            result.rows, path, format_timestamp(result.start), format_timestamp(result.end),
            duration, result.requests_made,
        )
        return result

    def load(self) -> pd.DataFrame:
        """Convenience accessor for the stored series."""
        return self.store.load()

    def close(self) -> None:
        self.client.close()
