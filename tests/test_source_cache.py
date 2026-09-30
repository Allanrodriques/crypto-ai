"""Cache reuse must be validated against the grid the features will use.

Two failure modes are covered here, and both were live bugs:

1.  ``load_sources`` passed ``start=None, end=None`` to the loader, so every
    cache check was a no-op and a truncated download was accepted as if it were
    the full history.  That is how a 5-month funding file became a
    multi-year experiment at 9.6% coverage.
2.  ``--no-download`` was accepted by the CLI and then ignored, so an "offline"
    run silently hit the network.  An offline flag that lies is worse than no
    flag: it makes a run non-reproducible while claiming not to be.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset.multisource import (
    SourceUnavailableError,
    _cache_shortfall,
    _fetch_would_help,
    load_or_fetch,
    load_sources,
)


def _frame(start: str, end: str, freq: str = "1h", col: str = "close") -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({col: range(len(index))}, index=index)


def _write_cache(cache_dir: Path, source: str, symbol: str, frame: pd.DataFrame) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{source}_{symbol}.parquet"
    frame.to_parquet(path)
    (path.parent / f"{source}_{symbol}.meta.json").write_text(
        json.dumps(
            {
                "source": source,
                "symbol": symbol,
                "rows": int(len(frame)),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
                "requested_start": None,
                "requested_end": None,
            }
        )
    )
    return path


def test_cache_shortfall_flags_a_late_start():
    index = _frame("2023-01-01", "2023-01-10").index
    assert _cache_shortfall(index, pd.Timestamp("2022-01-01", tz="UTC"), None) is not None


def test_cache_shortfall_flags_an_early_end():
    index = _frame("2023-01-01", "2023-01-10").index
    assert _cache_shortfall(index, None, pd.Timestamp("2024-01-01", tz="UTC")) is not None


def test_cache_shortfall_is_silent_when_unbounded():
    index = _frame("2023-01-01", "2023-01-10").index
    assert _cache_shortfall(index, None, None) is None


def test_fetch_would_help_is_true_without_metadata():
    index = _frame("2023-01-01", "2023-01-10").index
    assert _fetch_would_help({}, index, pd.Timestamp("2020-01-01", tz="UTC"), None) is True


def test_fetch_would_help_is_false_when_the_same_request_already_fell_short():
    """A source that cannot reach back must not be re-downloaded every run."""
    index = _frame("2023-01-01", "2023-01-10").index
    meta = {
        "start": "2023-01-01 00:00:00+00:00",
        "end": "2023-01-10 00:00:00+00:00",
        "requested_start": "2020-01-01 00:00:00+00:00",
        "requested_end": None,
    }
    assert _fetch_would_help(meta, index, pd.Timestamp("2020-01-01", tz="UTC"), None) is False


def test_fetch_would_help_is_true_when_the_earlier_request_asked_for_less():
    """A cache fetched for a *shorter* window should still be extended."""
    index = _frame("2023-01-01", "2023-01-10").index
    meta = {
        "start": "2023-01-01 00:00:00+00:00",
        "end": "2023-01-10 00:00:00+00:00",
        "requested_start": "2022-06-01 00:00:00+00:00",
        "requested_end": None,
    }
    assert _fetch_would_help(meta, index, pd.Timestamp("2020-01-01", tz="UTC"), None) is True


def test_offline_run_accepts_a_complete_cache(tmp_path: Path, monkeypatch):
    """Offline must not mean broken: a complete cache is enough."""
    frame = _frame("2022-01-01", "2023-01-01")
    _write_cache(tmp_path, "binance_funding", "BTCUSDT", frame)

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("offline run hit the network")

    monkeypatch.setattr("src.dataset.multisource.fetch_funding_rates", explode)
    got = load_or_fetch(
        "binance_funding",
        symbol="BTCUSDT",
        start=pd.Timestamp("2022-01-01", tz="UTC"),
        end=pd.Timestamp("2023-01-01", tz="UTC"),
        cache_dir=tmp_path,
        allow_download=False,
    )
    assert len(got) == len(frame)


def test_offline_run_refuses_to_fetch_a_missing_cache(tmp_path: Path, monkeypatch):
    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("offline run hit the network")

    monkeypatch.setattr("src.dataset.multisource.fetch_funding_rates", explode)
    with pytest.raises(SourceUnavailableError, match="no-download"):
        load_or_fetch(
            "binance_funding",
            symbol="BTCUSDT",
            start=None,
            end=None,
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_offline_run_rejects_a_truncated_cache_instead_of_accepting_it(
    tmp_path: Path, monkeypatch
):
    """The 5-month-funding failure: offline must fail loudly, not quietly."""
    _write_cache(tmp_path, "binance_funding", "BTCUSDT", _frame("2023-01-01", "2023-06-01"))

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("offline run hit the network")

    monkeypatch.setattr("src.dataset.multisource.fetch_funding_rates", explode)
    with pytest.raises(SourceUnavailableError, match="no-download"):
        load_or_fetch(
            "binance_funding",
            symbol="BTCUSDT",
            start=pd.Timestamp("2022-01-01", tz="UTC"),
            end=pd.Timestamp("2023-01-01", tz="UTC"),
            cache_dir=tmp_path,
            allow_download=False,
        )


def test_load_sources_defaults_the_window_to_the_feature_grid(tmp_path: Path, monkeypatch):
    """The grid bounds are what the features need, so they are the contract."""
    spot = _frame("2022-01-01", "2023-01-01")
    _write_cache(tmp_path, "binance_funding", "BTCUSDT", _frame("2023-01-01", "2023-06-01"))

    seen: dict[str, object] = {}

    def fake_funding(settings, session=None):
        seen["start"] = settings.start
        seen["end"] = settings.end
        return _frame("2022-01-01", "2023-01-01")

    monkeypatch.setattr("src.dataset.multisource.fetch_funding_rates", fake_funding)
    load_sources(
        symbol="BTCUSDT",
        sources=["binance_funding"],
        spot=spot,
        cache_dir=tmp_path,
        session=None,
    )
    # The truncated cache must have been rejected, forcing a fetch, and that
    # fetch must have been asked for the grid's full span.
    assert pd.Timestamp(seen["start"]) == spot.index.min()
    assert pd.Timestamp(seen["end"]) == spot.index.max()
