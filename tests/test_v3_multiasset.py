"""Tests for the V3 cross-symbol module (``src.v3.multiasset``).

Everything here is offline and small.  Synthetic hourly klines are written under
``tmp_path`` and every config is rebased onto it, so no real project directory is
read for data or written at all; the walk-forward harness is **stubbed** wherever
it is not the thing under test, because a real cross-asset fit is hours of CPU
and its own correctness is already covered by ``test_v3_models`` /
``test_v3_splits``.

The properties worth protecting, in order of how badly they would fail:

1. a symbol that cannot be used is *reported*, never padded, never crashed on;
2. a symbol that is skipped is recorded with the reason - a missing row and a
   deliberate exclusion look identical in every other way;
3. horizons are compared only where at least two symbols can support them;
4. symbols are fitted on the *intersection* of their features, or the comparison
   is between different models;
5. one symbol's failure costs one row of the table, not the whole run;
6. ``rank_within_horizon`` is a dense ranking, and it is reported next to the
   target's own scale so "easier market" cannot be read as "better model";
7. the long panel is restricted to the shared feature set and carries no
   unlabelled rows.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.v3 import multiasset
from src.v3.multiasset import (
    COMPARISON_COLUMNS,
    DEFAULT_MIN_ROWS,
    HORIZON_SUPPORT_COLUMNS,
    CrossSymbolResult,
    SymbolProfile,
    align_panel,
    build_cross_symbol_panel,
    common_horizons,
    compare_across_symbols,
    profile_symbol,
    run_multiasset,
)
from src.v3.walkforward import HorizonResult, V3RunResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_CONFIG_PATH = "config/v3.yaml"

#: Spot-only selection.  It needs nothing but the kline file, so these tests do
#: not depend on which external caches happen to exist, and it builds in
#: milliseconds.
SPOT_ONLY = ["technical"]

#: Symbols of deliberately different lengths.  A horizon is a wall-clock window,
#: so the short one simply cannot label the long horizons - which is the
#: situation the module exists to report rather than paper over.
LONG_SYMBOL = "LONGUSDT"    # 1500 candles
MID_SYMBOL = "MIDUSDT"      # 900 candles
SHORT_SYMBOL = "SHORTUSDT"  # 800 candles - 30d is beyond its reach
MISSING_SYMBOL = "GHOSTUSDT"  # never written to disk
#: A second long symbol, so a horizon the short one cannot reach is still
#: comparable between two others.
LONG2_SYMBOL = "LONG2USDT"

N_LONG, N_MID, N_SHORT = 1500, 900, 800
#: Hourly rows the V1 sma_200 warm-up costs, and the price/volume/technical
#: block's own 4h/1d context warm-up on top of it.
WARMUP_TECHNICAL, WARMUP_ALL_BUCKETS = 200, 480
HOURS = {"1d": 24, "7d": 168, "30d": 720}


# --------------------------------------------------------------------------- fixtures

def make_klines(n: int, *, seed: int, start: str = "2023-01-01", price: float = 20_000.0) -> pd.DataFrame:
    """Deterministic OHLCV frame with the trade-flow columns microstructure needs."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    volume = rng.uniform(50.0, 500.0, n)
    trades = rng.integers(10, 900, n)
    taker = volume * rng.uniform(0.4, 0.6, n)
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, n)),
            "high": close * (1.0 + np.abs(rng.normal(0.0, 0.002, n))),
            "low": close * (1.0 - np.abs(rng.normal(0.0, 0.002, n))),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": trades,
            "taker_buy_volume": taker,
            "taker_buy_quote_volume": taker * close,
        },
        index=index,
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """The rebased raw directory, pre-populated with synthetic symbols."""
    directory = tmp_path / "data" / "raw"
    directory.mkdir(parents=True)
    for symbol, n, seed in (
        (LONG_SYMBOL, N_LONG, 1),
        (LONG2_SYMBOL, N_LONG, 2),
        (MID_SYMBOL, N_MID, 3),
        (SHORT_SYMBOL, N_SHORT, 4),
    ):
        make_klines(n, seed=seed).to_parquet(directory / f"{symbol}_1h.parquet")
    return directory


@pytest.fixture
def v3_config(tmp_path: Path, raw_dir: Path) -> Config:
    """``config/v3.yaml`` with every path re-rooted into ``tmp_path``."""
    config = Config.load(V3_CONFIG_PATH, root=PROJECT_ROOT).rebase(tmp_path)
    assert config.paths.raw_dir == raw_dir
    return config


def fake_profile(symbol: str, labelled: dict[str, int], **overrides) -> SymbolProfile:
    """A profile built by hand, for tests about the decision rather than the data."""
    fields: dict = {
        "symbol": symbol,
        "available_buckets": ("price", "technical"),
        "missing_buckets": (),
        "n_rows": 10_000,
        "n_features": 39,
        "date_start": "2023-01-01 00:00:00+00:00",
        "date_end": "2024-01-01 00:00:00+00:00",
        "labelled_counts": dict(labelled),
        "notes": (),
    }
    fields.update(overrides)
    return SymbolProfile(**fields)


def fake_run_result(
    symbol: str,
    horizon: str,
    *,
    rmse: float,
    target_std: float,
    horizon_days: float = 1.0,
    n_predictions: int = 200,
    n_features: int = 39,
    ic: float = 0.25,
) -> V3RunResult:
    """A walk-forward result with controlled metrics and no model fitted.

    ``rmse`` is the winner's error and the ``zero`` baseline is set to the target's
    own standard deviation, which is what predicting flat actually scores - so
    ``rmse_skill_vs_zero`` is meaningful in the tests instead of arbitrary.
    """
    index = pd.date_range("2024-01-01", periods=n_predictions, freq="1h", tz="UTC", name="timestamp")
    rng = np.random.default_rng(abs(hash((symbol, horizon))) % (2**32))
    target = rng.normal(0.0, target_std, n_predictions)
    predictions = pd.DataFrame(
        {"target": target, "zero_pred": np.zeros(n_predictions), "ridge_pred": target * 0.9},
        index=index,
    )
    pooled = pd.DataFrame(
        {
            "mae": [target_std, rmse * 0.8],
            "rmse": [target_std, rmse],
            "spearman_ic": [0.0, ic],
            "direction_accuracy": [0.5, 0.55],
            "n": [n_predictions, n_predictions],
        },
        index=pd.Index(["zero", "ridge"], name="model"),
    )
    features = [f"f{i:02d}" for i in range(n_features)]
    return V3RunResult(
        symbol=symbol,
        horizons=[horizon],
        feature_columns=features,
        results=[
            HorizonResult(
                symbol=symbol,
                horizon=horizon,
                horizon_days=horizon_days,
                feature_columns=features,
                predictions=predictions,
                fold_metrics=pd.DataFrame(),
                pooled_metrics=pooled,
                interval_summary=pd.DataFrame(),
                geometry={},
                coverage={},
                seconds=0.0,
                models=["zero", "ridge"],
            )
        ],
        dataset={},
        config={},
        seconds=0.0,
    )


# --------------------------------------------------------------------------- 1. profile_symbol

def test_profile_symbol_reports_the_documented_fields(v3_config: Config) -> None:
    profile = profile_symbol(
        v3_config, MID_SYMBOL, horizons=["1d", "7d", "30d"], enabled_features=SPOT_ONLY
    )

    assert profile.symbol == MID_SYMBOL
    assert profile.n_rows == N_MID - WARMUP_TECHNICAL
    assert profile.n_features == 39
    assert profile.date_start is not None and profile.date_end is not None
    assert profile.date_start < profile.date_end
    # 900 candles - 200 warm-up - the horizon's own window.
    assert profile.labelled_counts == {
        "1d": N_MID - WARMUP_TECHNICAL - HOURS["1d"],
        "7d": N_MID - WARMUP_TECHNICAL - HOURS["7d"],
        "30d": 0,
    }
    assert isinstance(profile.notes, tuple) and all(isinstance(n, str) for n in profile.notes)
    assert set(profile.to_dict()) == {
        "symbol", "available_buckets", "missing_buckets", "n_rows", "n_features",
        "date_start", "date_end", "labelled_counts", "notes",
    }

    # The spot-only selection spans three V3 buckets and needs no source, so
    # nothing is missing - the asymmetry shows up on a selection that does.
    assert profile.available_buckets == ("price", "technical", "volume")
    assert profile.missing_buckets == ()

    with pytest.raises(FrozenInstanceError):
        profile.n_rows = 0  # type: ignore[misc]


def test_profile_symbol_supports_uses_labelled_count_not_row_count(v3_config: Config) -> None:
    profile = profile_symbol(
        v3_config, MID_SYMBOL, horizons=["1d", "7d", "30d"], enabled_features=SPOT_ONLY
    )

    assert profile.supports("1d") is True
    assert profile.supports("7d") is True
    # 700 rows in the frame, zero labels at 30d: a horizon longer than the history
    # is unsupported no matter how many rows the file has.
    assert profile.supports("30d") is False
    assert profile.supports("7d", min_rows=DEFAULT_MIN_ROWS) is True
    assert profile.supports("7d", min_rows=10_000) is False
    # A horizon the ladder never had is unsupported, not an error.
    assert profile.supports("180d") is False
    assert any("30d" in note for note in profile.notes)


def test_profile_symbol_reports_a_missing_symbol_instead_of_raising(v3_config: Config) -> None:
    profile = profile_symbol(
        v3_config, MISSING_SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY
    )

    assert profile.symbol == MISSING_SYMBOL
    assert profile.n_rows == 0
    assert profile.n_features == 0
    assert profile.date_start is None and profile.date_end is None
    assert profile.available_buckets == ()
    assert profile.labelled_counts == {"1d": 0}
    assert profile.supports("1d") is False
    assert any("DataUnavailableError" in note for note in profile.notes)


# --------------------------------------------------------------------------- 2. the panel

def test_build_cross_symbol_panel_records_a_missing_symbol_and_profiles_the_rest(
    v3_config: Config,
) -> None:
    result = build_cross_symbol_panel(
        v3_config,
        [MID_SYMBOL, MISSING_SYMBOL, SHORT_SYMBOL],
        horizons=["1d", "7d"],
        enabled_features=SPOT_ONLY,
        min_rows=400,
    )

    assert [p.symbol for p in result.profiles] == [MID_SYMBOL, SHORT_SYMBOL]
    assert MISSING_SYMBOL not in {p.symbol for p in result.profiles}
    assert MISSING_SYMBOL in result.skipped
    assert f"{MISSING_SYMBOL}_1h.parquet" in result.skipped[MISSING_SYMBOL]

    # A symbol with no file still gets a row per horizon, so "not profiled" is a
    # visible state rather than a gap in the table.
    support = result.horizon_support
    assert list(support.columns) == list(HORIZON_SUPPORT_COLUMNS)
    ghost = support[support["symbol"] == MISSING_SYMBOL]
    assert len(ghost) == 2
    assert ghost["skipped_reason"].str.contains("DataUnavailableError").all()
    assert not ghost["supported"].any()

    # The panel profiles; it does not predict, so the comparison table is empty
    # but carries its schema.
    assert list(result.comparisons.columns) == list(COMPARISON_COLUMNS)
    assert result.comparisons.empty


# --------------------------------------------------------------------------- 3. horizon support

def test_common_horizons_needs_min_symbols() -> None:
    profiles = [
        fake_profile("AUSDT", {"1d": 900, "7d": 800, "30d": 700}),
        fake_profile("BUSDT", {"1d": 900, "7d": 800, "30d": 700}),
        fake_profile("CUSDT", {"1d": 900, "7d": 800, "30d": 12}),
    ]

    # 30d is in the ladder for all three, but only two clear the default bar.
    assert common_horizons(profiles) == ["1d", "7d", "30d"]
    assert common_horizons(profiles, min_symbols=3) == ["1d", "7d"]
    assert common_horizons(profiles, min_symbols=3, min_rows=10) == ["1d", "7d", "30d"]
    assert common_horizons(profiles, min_rows=800) == ["1d", "7d"]
    assert common_horizons(profiles, min_rows=850) == ["1d"]
    # One symbol is not a comparison, whatever it can do on its own.
    assert common_horizons(profiles, min_symbols=1) == ["1d", "7d", "30d"]
    assert common_horizons(profiles[:1]) == []
    # Sorted by duration, not alphabetically: 1d, 7d, 30d - not 1d, 30d, 7d.
    assert common_horizons(profiles) != ["1d", "30d", "7d"]


def test_supported_horizons_agrees_with_the_profiles_it_was_built_from(v3_config: Config) -> None:
    result = build_cross_symbol_panel(
        v3_config,
        [MID_SYMBOL, SHORT_SYMBOL],
        horizons=["1d", "7d"],
        enabled_features=SPOT_ONLY,
        min_rows=500,
    )

    # MID reaches 500 labelled rows on 7d, SHORT does not, so 7d is not a
    # comparison even though one symbol supports it comfortably.
    assert result.supported_horizons() == ["1d"]
    assert result.supported_horizons(min_symbols=2) == common_horizons(
        result.profiles, min_symbols=2, min_rows=500
    )
    assert result.supported_horizons(min_symbols=3) == []

    support = result.horizon_support.set_index(["symbol", "horizon"])["skipped_reason"]
    assert pd.isna(support[(MID_SYMBOL, "1d")])
    assert "min_rows" in support[(SHORT_SYMBOL, "7d")]
    assert "fewer than 2 symbols" in support[(MID_SYMBOL, "7d")]

    # No symbol was skipped: both are in the comparison, on 1d.
    assert result.skipped == {}


# --------------------------------------------------------------------------- 4. run_multiasset isolation

def test_run_multiasset_isolates_one_failing_symbol(v3_config: Config, monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], list[str]]] = []

    def stub(dataset, horizons, **kwargs):
        symbol = dataset.symbol
        calls.append((symbol, tuple(h.label for h in horizons), list(dataset.feature_columns)))
        if symbol == SHORT_SYMBOL:
            raise RuntimeError("synthetic harness failure")
        return fake_run_result(
            symbol, horizons[0].label, rmse=0.01, target_std=0.02, horizon_days=horizons[0].days
        )

    monkeypatch.setattr(multiasset, "run_walk_forward", stub)

    result = run_multiasset(
        v3_config,
        [MID_SYMBOL, SHORT_SYMBOL],
        horizons=["1d", "7d"],
        enabled_features=SPOT_ONLY,
        min_rows=400,
    )

    # The surviving symbol keeps every row it was entitled to...
    comparisons = result.comparisons
    assert set(comparisons["symbol"]) == {MID_SYMBOL}
    assert set(comparisons["horizon"]) == {"1d", "7d"}
    assert MID_SYMBOL not in result.skipped
    assert "synthetic harness failure" in result.skipped[SHORT_SYMBOL]
    assert "1d" in result.skipped[SHORT_SYMBOL]

    # ...and the failure is recorded per (symbol, horizon), not just per symbol.
    support = result.horizon_support.set_index(["symbol", "horizon"])
    assert bool(support.loc[(MID_SYMBOL, "1d"), "evaluated"]) is True
    assert pd.isna(support.loc[(MID_SYMBOL, "7d"), "skipped_reason"])
    short = support.loc[SHORT_SYMBOL]
    assert not short["evaluated"].any()
    assert short["skipped_reason"].str.contains("synthetic harness failure").all()

    # One walk-forward call per (symbol, horizon) - that is what isolates them,
    # and each call carries exactly one horizon.
    assert [(symbol, labels) for symbol, labels, _ in calls] == [
        (MID_SYMBOL, ("1d",)), (MID_SYMBOL, ("7d",)),
        (SHORT_SYMBOL, ("1d",)), (SHORT_SYMBOL, ("7d",)),
    ]

    # Like-for-like: every symbol was handed the same feature list, and it is the
    # intersection, exposed on the result.
    shared = result.shared_features
    assert shared
    assert {tuple(call[2]) for call in calls} == {shared}
    assert set(shared).issubset(set(calls[0][2]))


def test_run_multiasset_skips_a_symbol_that_cannot_reach_a_comparable_horizon(
    v3_config: Config, monkeypatch
) -> None:
    """The short symbol is dropped for 30d only - and only for 30d."""

    def stub(dataset, horizons, **kwargs):
        return fake_run_result(
            dataset.symbol, horizons[0].label,
            rmse=0.01, target_std=0.02, horizon_days=horizons[0].days,
        )

    monkeypatch.setattr(multiasset, "run_walk_forward", stub)

    result = run_multiasset(
        v3_config,
        [LONG_SYMBOL, LONG2_SYMBOL, SHORT_SYMBOL],
        horizons=["1d", "7d", "30d"],
        enabled_features=SPOT_ONLY,
        min_rows=300,
    )

    comparisons = result.comparisons
    # 30d is supported by the two long symbols, so it is a comparison - and the
    # short symbol is simply not part of it.
    assert set(comparisons[comparisons["horizon"] == "30d"]["symbol"]) == {LONG_SYMBOL, LONG2_SYMBOL}
    # The short symbol is only out of *that* horizon: 800 candles is still enough
    # to clear min_rows on 1d and 7d, so it stays in the comparison elsewhere.
    assert set(comparisons[comparisons["horizon"] == "1d"]["symbol"]) == {
        LONG_SYMBOL, LONG2_SYMBOL, SHORT_SYMBOL
    }
    assert set(comparisons[comparisons["horizon"] == "7d"]["symbol"]) == {
        LONG_SYMBOL, LONG2_SYMBOL, SHORT_SYMBOL
    }
    assert SHORT_SYMBOL not in result.skipped      # it did produce rows, just not on 30d

    support = result.horizon_support.set_index(["symbol", "horizon"])
    assert int(support.loc[(SHORT_SYMBOL, "30d"), "n_labelled"]) == 0
    assert not bool(support.loc[(SHORT_SYMBOL, "30d"), "supported"])
    assert "no labels for 30d" in support.loc[(SHORT_SYMBOL, "30d"), "skipped_reason"]
    assert bool(support.loc[(SHORT_SYMBOL, "7d"), "evaluated"]) is True
    assert set(result.supported_horizons()) == {"1d", "7d", "30d"}


def test_run_multiasset_never_runs_a_horizon_only_one_symbol_supports(
    v3_config: Config, monkeypatch
) -> None:
    seen: list[str] = []

    def stub(dataset, horizons, **kwargs):
        seen.extend(horizon.label for horizon in horizons)
        return fake_run_result(
            dataset.symbol, horizons[0].label, rmse=0.01, target_std=0.02
        )

    monkeypatch.setattr(multiasset, "run_walk_forward", stub)

    result = run_multiasset(
        v3_config,
        [MID_SYMBOL, SHORT_SYMBOL],
        horizons=["1d", "7d", "30d"],
        enabled_features=SPOT_ONLY,
        min_rows=500,
    )

    # Only MID reaches 500 labels on 7d and neither reaches 30d, so 7d is not a
    # cross-symbol comparison and must never reach the harness.
    assert "30d" not in seen
    assert set(seen) == {"1d"}
    assert set(result.comparisons["horizon"]) == {"1d"}
    support = result.horizon_support.set_index(["symbol", "horizon"])
    assert "fewer than 2 symbols" in support.loc[(MID_SYMBOL, "7d"), "skipped_reason"]
    assert "no labels for 30d" in support.loc[(MID_SYMBOL, "30d"), "skipped_reason"]


# --------------------------------------------------------------------------- 5. comparison

def test_compare_across_symbols_ranks_densely_within_each_horizon() -> None:
    results = {
        # Deliberately inverted: the quietest market has the lowest error, so it
        # wins on RMSE, and the loudest one is the only one that beats its own
        # zero benchmark by a real margin.
        "QUIETUSDT": fake_run_result("QUIETUSDT", "1d", rmse=0.0015, target_std=0.002),
        "MIDUSDT": fake_run_result("MIDUSDT", "1d", rmse=0.0100, target_std=0.020),
        "LOUDUSDT": fake_run_result("LOUDUSDT", "1d", rmse=0.0120, target_std=0.050),
        # A second horizon, with an exact tie between two symbols.
        "QUIETUSDT_7d": fake_run_result("QUIETUSDT", "7d", rmse=0.004, target_std=0.008, horizon_days=7.0),
        "MIDUSDT_7d": fake_run_result("MIDUSDT", "7d", rmse=0.004, target_std=0.008, horizon_days=7.0),
        "LOUDUSDT_7d": fake_run_result("LOUDUSDT", "7d", rmse=0.020, target_std=0.050, horizon_days=7.0),
    }
    # The result's own `.symbol` wins over the mapping key, so the 7d entries
    # collapse onto the same three symbols.
    table = compare_across_symbols(results)

    assert list(table.columns) == list(COMPARISON_COLUMNS)
    one_day = table[table["horizon"] == "1d"].set_index("symbol")
    assert one_day["rank_within_horizon"].to_dict() == {
        "QUIETUSDT": 1.0, "MIDUSDT": 2.0, "LOUDUSDT": 3.0,
    }
    # The easiest target ranks best on raw error...
    assert one_day.loc["QUIETUSDT", "rank_within_horizon"] == 1.0
    # ...which is exactly why the scale-free columns are reported next to it: on
    # skill against the flat benchmark the loudest market is the best of the three.
    assert (one_day["rmse_skill_vs_zero"] < 1.0).all()
    assert one_day["rmse_skill_vs_zero"].idxmax() == "LOUDUSDT"
    assert one_day.loc["QUIETUSDT", "target_std"] < one_day.loc["LOUDUSDT", "target_std"]
    # The `zero` row is set to the target's own dispersion, which is the error
    # predicting flat actually makes (up to sampling noise in the fixture).
    assert one_day["zero_rmse"].to_numpy() == pytest.approx(
        one_day["target_std"].to_numpy(), rel=0.2
    )

    # Dense, not the gap method: two indistinguishable symbols share rank 1 and the
    # next takes 2 - no integer is skipped for a rank nobody earned.
    seven = table[table["horizon"] == "7d"].set_index("symbol")["rank_within_horizon"].to_dict()
    assert seven == {"QUIETUSDT": 1.0, "MIDUSDT": 1.0, "LOUDUSDT": 2.0}
    for _, group in table.groupby("horizon"):
        # Dense: the *distinct* ranks are 1..k, so a tie shares a number without
        # leaving a hole behind it.
        distinct = sorted(set(group["rank_within_horizon"]))
        assert distinct == list(range(1, len(distinct) + 1))

    assert (table["best_model"] == "ridge").all()
    assert (table["n_predictions"] == 200).all()


def test_compare_across_symbols_on_an_empty_mapping_keeps_the_schema() -> None:
    table = compare_across_symbols({})
    assert table.empty
    assert list(table.columns) == list(COMPARISON_COLUMNS)


# --------------------------------------------------------------------------- 6. the like-for-like panel

def test_align_panel_returns_the_documented_long_format(v3_config: Config) -> None:
    panel = align_panel(v3_config, [LONG_SYMBOL, MID_SYMBOL], ["1d", "7d"])

    assert list(panel.columns[:2]) == ["symbol", "timestamp"]
    assert list(panel.columns[-2:]) == ["target", "horizon"]
    assert set(panel["symbol"]) == {LONG_SYMBOL, MID_SYMBOL}
    assert set(panel["horizon"]) == {"1d", "7d"}
    assert isinstance(panel["timestamp"].dtype, pd.DatetimeTZDtype)

    features = list(panel.columns[2:-2])
    assert features, "the panel must carry the shared feature block"
    # Every (symbol, horizon) block is present, and no block is padded: the row
    # counts are exactly the labelled counts, never more.
    counts = panel.groupby(["symbol", "horizon"]).size()
    assert counts[(LONG_SYMBOL, "1d")] == N_LONG - 480 - 24
    assert counts[(MID_SYMBOL, "7d")] == N_MID - 480 - 168
    assert panel["target"].notna().all()
    assert panel.groupby(["symbol", "timestamp", "horizon"]).size().max() == 1


def test_align_panel_restricts_to_the_intersection_of_feature_sets(
    v3_config: Config, monkeypatch
) -> None:
    """A bucket only one symbol has is dropped for *every* symbol, not half-used."""
    real_build = multiasset.build_v3_dataset

    def asymmetric_build(config, symbol, horizons=None, enabled_features=None, max_rows=None):
        # The long symbol also gets the higher-timeframe `context` group, exactly
        # as it would if only it had a usable cache for that source.
        selection = SPOT_ONLY if symbol == MID_SYMBOL else ["technical", "context"]
        return real_build(config, symbol, horizons=horizons,
                          enabled_features=selection, max_rows=max_rows)

    monkeypatch.setattr(multiasset, "build_v3_dataset", asymmetric_build)

    full = real_build(
        v3_config, LONG_SYMBOL, horizons=["1d"], enabled_features=["technical", "context"]
    )
    narrow = real_build(v3_config, MID_SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY)
    extra = sorted(set(full.feature_columns) - set(narrow.feature_columns))
    assert extra, "the fixture must give one symbol genuinely more features"

    panel = align_panel(v3_config, [LONG_SYMBOL, MID_SYMBOL], ["1d"])

    features = list(panel.columns[2:-2])
    assert not set(extra) & set(features), "features only one symbol has must be dropped"
    assert set(features) == set(narrow.feature_columns)
    assert set(panel["symbol"]) == {LONG_SYMBOL, MID_SYMBOL}


# --------------------------------------------------------------------------- 7. to_dict

def test_result_to_dict_is_json_friendly(v3_config: Config) -> None:
    result = build_cross_symbol_panel(
        v3_config, [MID_SYMBOL, LONG_SYMBOL, MISSING_SYMBOL], horizons=["1d"], enabled_features=SPOT_ONLY
    )
    payload = result.to_dict()

    assert payload["symbols"] == [MID_SYMBOL, LONG_SYMBOL]
    assert MISSING_SYMBOL in payload["skipped"]
    assert payload["supported_horizons"] == ["1d"]
    # horizon_support is sorted by horizon then symbol, and a symbol that could
    # not be profiled still gets a row explaining itself.
    assert [row["symbol"] for row in payload["horizon_support"]] == [
        MISSING_SYMBOL, LONG_SYMBOL, MID_SYMBOL
    ]
    assert [row["supported"] for row in payload["horizon_support"]] == [False, True, True]
    assert payload["comparisons"] == []
    assert isinstance(CrossSymbolResult((), {}, pd.DataFrame(), pd.DataFrame()).to_dict(), dict)
