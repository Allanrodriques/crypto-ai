"""Shared pytest fixtures and synthetic market-data builders.

The suite never touches the network.  Binance responses are reproduced as
recorded payloads and served through a fake session, so the tests are
deterministic and runnable offline.

Isolation
---------
The suite must never write inside the real project.  ``data/raw``,
``data/processed``, ``data/predictions``, ``models`` and ``reports`` hold the
frozen V1 baseline; a test that writes there silently replaces real research
artefacts with a fixture.  That is not hypothetical - it happened, and the
frozen ``data/processed/BTCUSDT_1h_dataset.parquet`` was found to be a 374-row
fixture instead of the 41k-row dataset.

Two mechanisms keep that from recurring:

* every test config is *rebased* onto a temporary root, so its output paths are
  genuinely absolute under ``tmp``; and
* :func:`project_dirs_unchanged` fingerprints those five directories for the
  whole session and fails the run if any of them changed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

#: Project directories that hold real, non-reproducible research output.  Tests
#: must treat all of them as read-only.
PROTECTED_DIRS = ("data/raw", "data/processed", "data/predictions", "models", "reports")


def _fingerprint(path: Path) -> str:
    """Hash a directory tree's relative paths, sizes and mtimes.

    Content hashing is deliberately avoided: these trees hold multi-megabyte
    parquet and image files, and any write at all - even one that ends up
    byte-identical - is the behaviour under test.
    """
    if not path.exists():
        return "<absent>"
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def project_dirs_unchanged():
    """Fail the session if any test wrote into the real project directories.

    This is the backstop behind the rebased config fixture: if a future test
    reaches the project paths by some other route, the run goes red instead of
    quietly destroying the baseline.
    """
    before = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}
    yield before
    after = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}
    changed = [name for name in PROTECTED_DIRS if before[name] != after[name]]
    if changed:
        pytest.fail(
            "tests modified protected project directories: "
            + ", ".join(changed)
            + ". Test configs must be rebased onto a temporary root "
            "(Config.rebase), never written to the real project."
        )


@pytest.fixture(scope="session")
def test_root(tmp_path_factory) -> Path:
    """A temporary root that every test config's output paths resolve under."""
    return tmp_path_factory.mktemp("crypto-ml-test")

#: Fast feature config so tests can run on a few hundred synthetic candles.
TEST_FEATURES = {
    "sma_periods": [5, 10, 20],
    "ema_periods": [5, 9],
    "rsi_period": 7,
    "atr_period": 7,
    "bollinger_period": 10,
    "bollinger_std": 2.0,
    "macd_fast": 5,
    "macd_slow": 9,
    "macd_signal": 4,
    "roc_period": 5,
    "return_periods": [1, 3, 6],
    "volume_sma_period": 10,
    "volume_lookback": 3,
    "volatility_lookbacks": [10, 20],
    "periods_per_year": 8760,
    "eps": 1e-12,
}


def make_klines(
    n: int = 400,
    *,
    start: str = "2023-01-01",
    freq: str = "1h",
    seed: int = 42,
    price: float = 20_000.0,
    drift: float = 0.0,
    vol: float = 0.003,
) -> pd.DataFrame:
    """Build a deterministic, realistic OHLCV frame with a UTC DatetimeIndex."""
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * (1.0 + rng.normal(0, vol / 2, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.uniform(50, 500, n)
    step = pd.Timedelta(freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": index + step - pd.Timedelta(milliseconds=1),
            "quote_volume": volume * close,
            "trades": pd.array(rng.integers(10, 900, n), dtype="Int64"),
            "taker_buy_volume": volume * 0.5,
            "taker_buy_quote_volume": volume * close * 0.5,
        },
        index=index,
    )


def make_kline_rows(klines: pd.DataFrame) -> list[list]:
    """Convert a frame back into raw ``/api/v3/klines`` JSON row shape."""
    step = pd.Timedelta("1h")
    rows = []
    for ts, row in klines.iterrows():
        rows.append(
            [
                int(ts.value // 1_000_000),
                f"{row['open']:.8f}",
                f"{row['high']:.8f}",
                f"{row['low']:.8f}",
                f"{row['close']:.8f}",
                f"{row['volume']:.8f}",
                int((ts + step - pd.Timedelta(milliseconds=1)).value // 1_000_000),
                f"{row['quote_volume']:.8f}",
                int(row["trades"]),
                f"{row['taker_buy_volume']:.8f}",
                f"{row['taker_buy_quote_volume']:.8f}",
                "0",
            ]
        )
    return rows


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)[:300]

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Serves canned kline pages, honouring ``startTime``/``limit`` like Binance.

    Also records every call so pagination and rate-limiting behaviour can be
    asserted without a network.  A symbol outside ``known_symbols`` is rejected
    with HTTP 400, and an interval outside ``supported_intervals`` is rejected by
    ``exchangeInfo``, mirroring the real API.
    """

    def __init__(
        self,
        klines: pd.DataFrame,
        *,
        page_cap: int | None = None,
        known_symbols: tuple[str, ...] = ("BTCUSDT",),
        supported_intervals: tuple[str, ...] = ("1m", "5m", "1h", "4h", "1d"),
    ) -> None:
        self.klines = klines
        self.page_cap = page_cap
        self.known_symbols = known_symbols
        self.supported_intervals = supported_intervals
        self.calls: list[dict] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params})

        if url.endswith("/api/v3/time"):
            return FakeResponse({"serverTime": int(pd.Timestamp("2024-06-01", tz="UTC").value // 1_000_000)})

        symbol = str(params.get("symbol", "BTCUSDT")).upper()
        interval = str(params.get("interval", "1h"))
        if symbol not in self.known_symbols:
            return FakeResponse({"code": -1121, "msg": "Invalid symbol."}, status_code=400)

        if url.endswith("/api/v3/exchangeInfo"):
            return FakeResponse(
                {
                    "symbols": [
                        {
                            "symbol": symbol,
                            "status": "TRADING",
                            "filters": [{"filterType": "INTERVAL", "intervals": list(self.supported_intervals)}],
                        }
                    ]
                }
            )

        if url.endswith("/api/v3/klines"):
            if interval not in self.supported_intervals:
                return FakeResponse({"code": -1121, "msg": "Invalid interval."}, status_code=400)
            start_ms = int(params.get("startTime", 0))
            limit = int(params.get("limit", 1000))
            if self.page_cap is not None:
                limit = min(limit, self.page_cap)
            end_ms = int(params.get("endTime", 2**62))
            stamps = self.klines.index.view("int64") // 1_000_000
            selection = self.klines[(stamps >= start_ms) & (stamps <= end_ms)]
            return FakeResponse(make_kline_rows(selection.iloc[:limit]))

        return FakeResponse({"code": -1121, "msg": "Invalid symbol."}, status_code=400)

    def close(self) -> None:
        return None


@pytest.fixture
def config(test_root):
    """A test config whose output paths really are inside the temp directory.

    This previously did ``object.__setattr__(cfg.paths, "root", ...)``, which
    did nothing: ``Paths`` had already resolved every entry to an absolute
    path under the project, so ``build_dataset`` kept saving to the real
    ``data/processed``.  ``rebase`` rebuilds the paths instead.
    """
    return load_config().rebase(test_root).with_overrides({"features": TEST_FEATURES})


@pytest.fixture
def klines() -> pd.DataFrame:
    return make_klines(400)


@pytest.fixture
def long_klines() -> pd.DataFrame:
    return make_klines(1200, seed=7)


@pytest.fixture
def fake_session(klines):
    return FakeSession(klines)
