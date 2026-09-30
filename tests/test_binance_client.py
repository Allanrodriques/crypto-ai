"""Data-layer tests: API parsing, pagination, dedup, ordering, invalid OHLC, gaps.

No network access: :class:`tests.conftest.FakeSession` replays recorded
``/api/v3/klines`` response shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.binance_client import (
    BinanceAPIError,
    BinanceClient,
    ClientSettings,
    RateLimiter,
    parse_klines,
)
from src.data.downloader import (
    KlineDownloader,
    KlineStore,
    drop_incomplete_candle,
    find_missing_ranges,
    normalise_klines,
)
from src.data.validator import (
    DataValidationError,
    check_alignment,
    check_duplicates,
    check_missing,
    check_ohlc,
    check_price_outliers,
    check_ordering,
    check_volume,
    validate_klines,
)
from tests.conftest import FakeSession, make_kline_rows, make_klines


# --------------------------------------------------------------------------- parsing

def test_parse_klines_columns_and_dtypes():
    rows = make_kline_rows(make_klines(5))
    frame = parse_klines(rows, symbol="btcusdt", interval="1h")

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None
    assert str(frame.index.tz) == "UTC"
    for column in ("open", "high", "low", "close", "volume"):
        assert column in frame.columns
        assert frame[column].dtype == "float64"
    # Metadata is kept, the "ignore" column is not.
    assert {"quote_volume", "trades", "taker_buy_volume"}.issubset(frame.columns)
    assert "ignore" not in frame.columns
    assert "open_time" not in frame.columns


def test_parse_klines_uses_utc_not_local_time():
    rows = make_kline_rows(make_klines(3))
    frame = parse_klines(rows)
    first_open_ms = rows[0][0]
    assert frame.index[0].value // 1_000_000 == first_open_ms
    assert frame.index[0].hour == pd.Timestamp(first_open_ms, unit="ms", tz="UTC").hour


def test_parse_klines_empty_payload():
    frame = parse_klines([])
    assert frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_parse_klines_handles_unparseable_numbers_as_nan():
    rows = make_kline_rows(make_klines(2))
    rows[0][1] = "not-a-number"
    frame = parse_klines(rows)
    assert pd.isna(frame["open"].iloc[0])


# --------------------------------------------------------------------------- client

def test_client_paginates_over_multiple_pages(klines):
    session = FakeSession(klines, page_cap=100)
    client = BinanceClient(ClientSettings(page_size=100), session=session)

    collected = [page for page in client.iter_klines("BTCUSDT", "1h")]

    assert len(collected) > 1, "expected more than one page"
    combined = pd.concat(collected)
    assert len(combined) == len(klines)
    assert combined.index.is_monotonic_increasing
    assert not combined.index.has_duplicates
    kline_calls = [c for c in session.calls if c["url"].endswith("/klines")]
    # One extra call: the terminating empty page that proves the end of history.
    assert len(kline_calls) == len(collected) + 1


def test_pagination_cursors_advance_without_overlap(klines):
    session = FakeSession(klines, page_cap=60)
    client = BinanceClient(ClientSettings(page_size=60), session=session)
    stamps = [int(c["params"]["startTime"]) for c in session.calls if c["url"].endswith("/klines")]
    # startTime strictly increases; a stalled cursor would spin forever.
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_client_respects_end_time(klines):
    session = FakeSession(klines, page_cap=1000)
    client = BinanceClient(ClientSettings(page_size=1000), session=session)
    end = klines.index[49] + pd.Timedelta(milliseconds=3_599_999)
    frame = pd.concat(list(client.iter_klines("BTCUSDT", "1h", end_time=end)))
    assert frame.index.max() <= end


def test_client_rejects_unknown_symbol():
    client = BinanceClient(ClientSettings(), session=FakeSession(make_klines(2)))
    with pytest.raises(BinanceAPIError, match="Invalid symbol|not listed"):
        client.fetch_klines("NOSUCHPAIR", "1h")


def test_client_rejects_unsupported_interval():
    session = FakeSession(make_klines(2), supported_intervals=("1m", "5m", "1d"))
    client = BinanceClient(ClientSettings(), session=session)
    with pytest.raises(BinanceAPIError):
        client.assert_symbol_and_interval("BTCUSDT", "1h")
    # A supported interval passes.
    client.assert_symbol_and_interval("BTCUSDT", "1d")


def test_rate_limiter_charges_configured_weight():
    limiter = RateLimiter(limit_per_minute=1200, weight_per_request=2)
    before = limiter.tokens
    limiter.consume()
    assert before - limiter.tokens == pytest.approx(2.0)


def test_rate_limiter_rejects_nonsense():
    with pytest.raises(ValueError):
        RateLimiter(limit_per_minute=0)


# --------------------------------------------------------------------------- store

def test_store_roundtrip_preserves_values(tmp_path, klines):
    store = KlineStore(tmp_path, "BTCUSDT", "1h")
    store.save(normalise_klines(klines))
    loaded = store.load()
    assert len(loaded) == len(klines)
    assert loaded.index.equals(klines.index)
    np.testing.assert_allclose(loaded["close"].to_numpy(), klines["close"].to_numpy())
    np.testing.assert_allclose(loaded["volume"].to_numpy(), klines["volume"].to_numpy())


# --------------------------------------------------------------------------- dedup / order

def test_duplicate_timestamps_are_removed_and_detected(klines):
    doubled = pd.concat([klines, klines.iloc[:20]])
    assert doubled.index.has_duplicates

    cleaned = normalise_klines(doubled)
    assert not cleaned.index.has_duplicates
    assert len(cleaned) == len(klines)

    report_check = check_duplicates(doubled)
    assert not report_check.passed
    assert report_check.count == 20


def test_timestamps_are_sorted_chronologically():
    shuffled = make_klines(30)
    scrambled = normalise_klines(shuffled.iloc[::-1])
    assert scrambled.index.is_monotonic_increasing
    np.testing.assert_allclose(  # sorting must reorder, never alter, the values
        scrambled["close"].to_numpy(), shuffled["close"].to_numpy()
    )


def test_ordering_check_flags_a_shuffled_frame(klines):
    assert check_ordering(klines).passed
    bad = klines.iloc[::-1]
    assert not check_ordering(bad).passed


def test_alignment_check_flags_off_grid_timestamps(klines):
    assert check_alignment(klines, "1h").passed
    off_grid = klines.copy()
    off_grid.index = off_grid.index + pd.Timedelta(minutes=7)
    assert not check_alignment(off_grid, "1h").passed


# --------------------------------------------------------------------------- invalid OHLCV

@pytest.mark.parametrize(
    "mutate, description",
    [
        (lambda d: d.assign(high=d["high"] * 0.5), "high below open/close"),
        (lambda d: d.assign(low=d["low"] * 2.0), "low above open/close"),
        (lambda d: d.assign(close=d["close"] * 0.0), "zero close"),
        (lambda d: d.assign(open=d["open"] * -1.0), "negative open"),
    ],
)
def test_invalid_ohlc_is_detected(klines, mutate, description):
    bad = mutate(klines.copy())
    result = check_ohlc(bad)
    assert not result.passed, f"expected failure for {description}"
    assert result.count > 0


def test_negative_volume_is_detected(klines):
    bad = klines.copy()
    bad.iloc[10, bad.columns.get_loc("volume")] = -5.0
    result = check_volume(bad)
    assert not result.passed
    assert result.count == 1


def test_validator_reports_all_error_types_in_one_pass(klines):
    corrupt = klines.copy()
    corrupt.iloc[5, corrupt.columns.get_loc("high")] = 1.0     # high < open
    corrupt.iloc[9, corrupt.columns.get_loc("volume")] = -1.0   # negative volume
    corrupt.iloc[11, corrupt.columns.get_loc("close")] = np.nan  # null

    report = validate_klines(corrupt, "1h", "BTCUSDT")
    failed = {c.name for c in report.errors}
    assert {"ohlc_consistency", "non_negative_volume", "null_values"} <= failed
    assert not report.ok
    assert report.status() == "FAIL"


def test_validator_can_raise_on_error(klines):
    corrupt = klines.copy()
    corrupt.iloc[5, corrupt.columns.get_loc("high")] = 1.0
    with pytest.raises(DataValidationError, match="Validation failed"):
        validate_klines(corrupt, "1h", "BTCUSDT", fail_on_error=True)


def test_clean_data_passes_validation(klines):
    report = validate_klines(klines, "1h", "BTCUSDT")
    assert report.ok
    assert report.status() == "PASS"
    assert all(c.passed for c in report.checks)


# --------------------------------------------------------------------------- missing candles

def test_missing_candles_are_detected():
    frame = make_klines(50)
    holed = frame.drop(frame.index[20:25])
    check, gaps = check_missing(holed, "1h", start=frame.index[0], end=frame.index[-1])

    assert not check.passed
    assert check.count == 5
    assert len(gaps) == 1
    assert gaps[0][0] == frame.index[20]
    assert gaps[0][1] == frame.index[24]


def test_contiguous_data_reports_no_gaps(klines):
    check, gaps = check_missing(klines, "1h", start=klines.index[0], end=klines.index[-1])
    assert check.passed
    assert gaps == []


def test_find_missing_ranges_collapses_runs():
    frame = make_klines(20)
    holed = frame.drop([frame.index[5], frame.index[6], frame.index[7], frame.index[15]])
    gaps = find_missing_ranges(holed.index, frame.index[0], frame.index[-1], "1h")
    assert len(gaps) == 2
    assert (gaps[0][1] - gaps[0][0]) / pd.Timedelta(hours=1) == 2
    assert gaps[1][0] == frame.index[15]


def test_missing_candles_make_validation_fail():
    frame = make_klines(60)
    holed = frame.drop(frame.index[25:30])
    report = validate_klines(holed, "1h", "BTCUSDT")
    assert not report.ok
    assert any(c.name == "missing_timestamps" for c in report.errors)


# --------------------------------------------------------------------------- incomplete candle

def test_incomplete_candle_is_dropped():
    frame = make_klines(10)
    now = frame.index[-1] + pd.Timedelta(minutes=30)  # last candle still forming
    dropped = drop_incomplete_candle(frame, "1h", now=now)
    assert len(dropped) == len(frame) - 1
    assert dropped.index[-1] == frame.index[-2]


def test_all_closed_candles_are_kept():
    frame = make_klines(10)
    now = frame.index[-1] + pd.Timedelta(hours=1)
    assert len(drop_incomplete_candle(frame, "1h", now=now)) == len(frame)


# --------------------------------------------------------------------------- resume

def test_downloader_is_idempotent(tmp_path, config, klines):
    session = FakeSession(klines)
    client = BinanceClient(ClientSettings(page_size=1000), session=session)
    store = KlineStore(tmp_path, "BTCUSDT", "1h")
    downloader = KlineDownloader(client, store, "BTCUSDT", "1h", start_date="2023-01-01", end_date=None)

    expected = len(klines)
    first = downloader.download()
    assert first.rows == expected

    second = downloader.download()
    assert second.resumed_from is not None
    assert second.rows == expected
    # The second run must not re-fetch history: it makes a single forward-probe
    # whose startTime is strictly after the newest stored candle.
    kline_calls = [c for c in session.calls if c["url"].endswith("/klines")]
    assert len(kline_calls) == 2
    assert kline_calls[1]["params"]["startTime"] > klines.index[-1].value // 1_000_000


def test_downloader_backfills_earlier_history(tmp_path, config, klines):
    store = KlineStore(tmp_path, "BTCUSDT", "1h")
    session = FakeSession(klines)
    client = BinanceClient(ClientSettings(page_size=1000), session=session)

    # First store only the later half.
    store.save(normalise_klines(klines.iloc[200:]))
    downloader = KlineDownloader(client, store, "BTCUSDT", "1h", start_date="2023-01-01", end_date=None)
    result = downloader.download()

    assert result.rows == len(klines)
    assert store.load().index.min() == klines.index[0]
    assert store.load().index.max() == klines.index[-1]


def test_download_respects_end_date(tmp_path, config, klines):
    store = KlineStore(tmp_path, "BTCUSDT", "1h")
    client = BinanceClient(ClientSettings(page_size=1000), session=FakeSession(klines))
    end = klines.index[100]
    downloader = KlineDownloader(client, store, "BTCUSDT", "1h", start_date="2023-01-01", end_date=end)
    result = downloader.download()

    assert result.rows == 101
    assert result.end == end


def test_download_writes_metadata_sidecar(tmp_path, config, klines):
    store = KlineStore(tmp_path, "BTCUSDT", "1h")
    client = BinanceClient(ClientSettings(page_size=1000), session=FakeSession(klines))
    downloader = KlineDownloader(
        client, store, "BTCUSDT", "1h", start_date="2023-01-01", end_date=None, save_metadata=True
    )
    result = downloader.download()
    assert store.meta_path.exists()

    import json

    payload = json.loads(store.meta_path.read_text())
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "1h"
    assert payload["rows"] == result.rows == len(klines)


# ---------------------------------------------------------------------------
# price_outliers calibration


def _with_jump(df, at, factor):
    """Rebuild OHLCV from a jump applied to `close` from row `at` onwards,
    keeping high/low consistent so only the outlier check is under test."""
    out = df.copy()
    open_ = out["open"].to_numpy(dtype="float64", copy=True)
    close = out["close"].to_numpy(dtype="float64", copy=True)
    close[at:] *= factor
    open_[at:] = close[at:]
    out["open"] = open_
    out["close"] = close
    out["high"] = np.maximum(open_, close) * 1.001
    out["low"] = np.minimum(open_, close) * 0.999
    return out


def test_outlier_check_ignores_ordinary_crypto_volatility():
    """A routine 7% hourly move must NOT be reported as data corruption.

    Hourly crypto returns are fat-tailed and mostly tiny, so the MAD-derived
    scale is around 0.1%.  Judged on the robust z-score alone, a normal 7% hour
    looks like ~70 sigma and the check cries wolf on every real rally.
    """
    df = _with_jump(make_klines(400, vol=0.003), 120, 1.07)

    result = check_price_outliers(df, zscore_threshold=20.0, max_abs_log_return=0.35)
    assert result.passed, result.message
    # Still quantified as an extreme move, just not called corrupt.
    assert "floor" in result.message and "99.9th" in result.message


def test_outlier_check_flags_moves_beyond_the_absolute_floor():
    df = _with_jump(make_klines(400, vol=0.003), 150, 1.9)  # +90% in one hour

    result = check_price_outliers(df, zscore_threshold=20.0, max_abs_log_return=0.35)
    assert not result.passed
    assert result.count == 1
    assert result.examples == [str(df.index[150]).replace("+00:00", " UTC")]


def test_outlier_check_needs_both_criteria_to_fire():
    """A huge robust z-score with a sub-floor move is not corruption."""
    df = _with_jump(make_klines(400, vol=0.003), 150, 1.10)  # |move| = 0.095

    result = check_price_outliers(df, zscore_threshold=20.0, max_abs_log_return=0.35)
    assert result.passed, result.message
