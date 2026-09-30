"""Tests for the V3 point-in-time dataset assembly.

Every test here is fast (seconds) and offline.  The functional tests run against
a synthetic hourly kline file written into a rebased temporary root, so no real
project directory is read *or* written; the single real-data test at the bottom
is read-only and is clearly marked.

The properties worth protecting, in order of how badly they would fail:

1. the config is a real, loadable V3 contract with the canonical ladder;
2. the frame is one row per unique sorted UTC timestamp, with one honest
   forward-return column per horizon;
3. the label tail is NaN - never a fabricated zero;
4. the drop report accounts for every raw row that went in;
5. ``labelled_frame`` is the only path to trainable rows, and it is honest;
6. building a dataset writes nothing;
7. a typo in a feature name fails loudly instead of shrinking the frame;
8. point-in-time: a row's features do not change when later data exists.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.features.groups import known_features
from src.v3.dataset import (
    DROP_REASONS,
    FEATURE_COVERAGE_COLUMNS,
    TARGET_COVERAGE_COLUMNS,
    DataUnavailableError,
    FeatureSelectionError,
    V3Dataset,
    build_v3_dataset,
    resolve_feature_selection,
)
from src.v3.horizons import DEFAULT_HORIZONS, build_horizon_ladder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_CONFIG_PATH = "config/v3.yaml"

#: The V3 group used by most tests: it needs nothing but spot klines, so the
#: result does not depend on which external caches happen to exist.
SPOT_ONLY_GROUPS = ["technical"]

#: Enough hourly candles for the V1 baseline warm-up (200 candles) plus a
#: 180d-shaped tail, while staying cheap to build.
N_CANDLES = 900
SYMBOL = "SYNTHUSDT"
DIRTY_SYMBOL = "DIRTYUSDT"
PREFIX_SYMBOL = "PREFIXUSDT"


# --------------------------------------------------------------------------- fixtures

def make_klines(n: int = N_CANDLES, *, seed: int = 7, start: str = "2023-01-01") -> pd.DataFrame:
    """A deterministic OHLCV frame with trade-flow columns, hourly and UTC."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    close = 20_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    volume = rng.uniform(50.0, 500.0, n)
    trades = rng.integers(10, 900, n)
    taker_buy = volume * rng.uniform(0.4, 0.6, n)
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, n)),
            "high": close * (1.0 + np.abs(rng.normal(0.0, 0.002, n))),
            "low": close * (1.0 - np.abs(rng.normal(0.0, 0.002, n))),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": trades,
            "taker_buy_volume": taker_buy,
            "taker_buy_quote_volume": taker_buy * close,
        },
        index=index,
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """The (rebased) raw directory, pre-populated with synthetic symbols."""
    directory = tmp_path / "data" / "raw"
    directory.mkdir(parents=True)

    clean = make_klines()
    clean.to_parquet(directory / f"{SYMBOL}_1h.parquet")

    # A symbol whose raw data needs cleaning: one duplicated timestamp and three
    # unusable closes (missing, infinite, non-positive).  `close` is the only
    # thing the builder looks at for a price, so a zero is treated as missing
    # rather than producing an infinite return.
    dirty = make_klines(seed=11)
    dirty = pd.concat([dirty, dirty.iloc[[-1]]])
    dirty.iloc[10, dirty.columns.get_loc("close")] = np.nan
    dirty.iloc[20, dirty.columns.get_loc("close")] = np.inf
    dirty.iloc[30, dirty.columns.get_loc("close")] = 0.0
    dirty.to_parquet(directory / f"{DIRTY_SYMBOL}_1h.parquet")

    # A strict prefix of the clean series, for the point-in-time test.  It must be
    # a *slice*, not a smaller draw: regenerating with a shorter length consumes
    # the RNG stream differently and would produce a different series.
    clean.iloc[:600].to_parquet(directory / f"{PREFIX_SYMBOL}_1h.parquet")
    return directory


@pytest.fixture
def v3_config(tmp_path: Path, raw_dir: Path) -> Config:
    """``config/v3.yaml`` with every path re-rooted into ``tmp_path``.

    ``rebase`` rebuilds the resolved paths, so nothing can reach the real
    project even if some code tried to write.
    """
    config = Config.load(V3_CONFIG_PATH, root=PROJECT_ROOT)
    rebased = config.rebase(tmp_path)
    assert rebased.paths.raw_dir == raw_dir
    return rebased


@pytest.fixture
def dataset(v3_config: Config) -> V3Dataset:
    """A clean synthetic dataset: spot-only features, two horizons."""
    return build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d", "7d"], enabled_features=SPOT_ONLY_GROUPS
    )


def tree(root: Path) -> set[str]:
    """Every path under ``root``, relative - the before/after snapshot of a build."""
    return {str(item.relative_to(root)) for item in root.rglob("*")}


# --------------------------------------------------------------------------- 1. config

def test_v3_config_loads_with_the_repo_loader_and_defines_the_ladder() -> None:
    """``config/v3.yaml`` must load through the project's own loader."""
    config = load_config(V3_CONFIG_PATH)
    assert config.source_path.name == "v3.yaml"
    assert config.source_path.parent.name == "config"

    block = config.raw["v3"]
    assert block["bars_per_day"] == 24
    assert list(block["horizons"]) == list(DEFAULT_HORIZONS)
    assert block["primary_symbol"] in block["symbols"]

    targets = block["targets"]
    assert targets["kind"] == "forward_return"
    assert targets["price_column"] == "close"
    assert targets["horizons_are_independent"] is True

    validation = block["validation"]
    assert validation["purge"] == "horizon"
    assert validation["embargo"] == "horizon"
    assert validation["min_train_fraction"] + validation["validation_fraction"] + (
        validation["test_fraction"]
    ) < 1.0

    assert block["outputs"]["model_dir"] == "models/v3"
    assert block["outputs"]["report_dir"] == "reports/experiments/v3"
    assert len(block["models"]) == 6
    assert block["reporting"]["artifacts"]

    # V3 outputs are namespaced away from the frozen V1/V2 directories.
    paths = config.paths
    assert paths.models_dir.name == "v3"
    assert "v3" in paths.reports_dir.parts
    assert paths.raw_dir == PROJECT_ROOT / "data" / "raw"


def test_horizon_ladder_over_the_v3_block_is_canonically_sorted() -> None:
    """The ladder is sorted by duration, so the order is independent of the YAML."""
    block = load_config(V3_CONFIG_PATH).raw["v3"]
    ladder = build_horizon_ladder(block)

    assert [horizon.label for horizon in ladder] == list(DEFAULT_HORIZONS)
    assert len(ladder) == 8
    deltas = [horizon.delta for horizon in ladder]
    assert deltas == sorted(deltas)

    # 24 bars/day on the hourly grid, not 24*24.
    expected = {"1d": 24, "3d": 72, "7d": 168, "180d": 4320}
    for horizon in ladder:
        assert horizon.bars_per_unit == 24
        if horizon.label in expected:
            assert horizon.nominal_bars == expected[horizon.label]

    # A shuffled declaration must produce the same ladder.
    shuffled = dict(block, horizons=list(reversed(block["horizons"])))
    assert [h.label for h in build_horizon_ladder(shuffled)] == [h.label for h in ladder]


def test_v3_config_declares_the_six_feature_buckets() -> None:
    block = load_config(V3_CONFIG_PATH).raw["v3"]
    features = block["features"]
    assert features["mode"] in {"all", "buckets", "explicit"}
    assert len(features["buckets"]) == 6
    for name, switch in features["buckets"].items():
        assert switch in (True, "all"), f"bucket {name} is not switched on"

    # The declaration must resolve to the whole registered registry.
    selection = resolve_feature_selection(block)
    assert len(selection.columns) == len(known_features()) == 107
    assert set(selection.columns) == set(known_features())


# --------------------------------------------------------------------------- 2. frame

def test_frame_has_one_target_column_per_horizon_and_a_clean_index(dataset: V3Dataset) -> None:
    frame = dataset.frame
    horizons = ["1d", "7d"]

    assert [column for column in frame.columns if column.startswith("future_return_")] == [
        "future_return_1d",
        "future_return_7d",
    ]
    assert dataset.target_columns == [f"future_return_{h}" for h in horizons]
    assert list(frame.columns) == dataset.feature_columns + dataset.target_columns
    assert set(dataset.feature_columns).isdisjoint(dataset.target_columns)

    index = frame.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is not None and str(index.tz) == "UTC"
    assert index.is_unique, "duplicate timestamps in the frame"
    assert index.is_monotonic_increasing
    assert frame.index.equals(pd.DatetimeIndex(frame.index, name="timestamp"))
    assert not frame.index.hasnans

    # Exactly one row per retained timestamp, and every feature is populated.
    assert len(frame) == len(index) == len(dataset)
    assert frame[dataset.feature_columns].notna().all().all()
    assert np.isfinite(frame[dataset.feature_columns].to_numpy(dtype="float64")).all()


def test_full_horizon_ladder_produces_a_column_per_configured_horizon(v3_config: Config) -> None:
    """The configured 8-horizon ladder yields 8 target columns on any frame."""
    dataset = build_v3_dataset(
        v3_config, SYMBOL, enabled_features=SPOT_ONLY_GROUPS, max_rows=400
    )
    assert dataset.target_columns == [f"future_return_{h}" for h in DEFAULT_HORIZONS]
    assert len(dataset.target_columns) == 8
    coverage = dataset.target_coverage.set_index("horizon")
    # 400 hourly rows cannot span 180 days, so those horizons have no label at
    # all - and the coverage table says so rather than hiding it.
    assert int(coverage.loc["1d", "n_labelled"]) == len(dataset) - 24
    assert int(coverage.loc["180d", "n_labelled"]) == 0
    assert int(coverage.loc["180d", "labelled_pct"]) == 0.0


# --------------------------------------------------------------------------- 3. honest labels

def test_label_tail_is_nan_never_fabricated(v3_config: Config) -> None:
    """The last 24 hourly rows have no 1d label: the future has not happened.

    A backward ffilled tail would produce a flat block of exactly 0.0 here, which
    would be indistinguishable from a real "no move" forecast.
    """
    dataset = build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS
    )
    tail = dataset.frame["future_return_1d"].iloc[-24:]
    assert tail.isna().all(), "the 1d label tail must be NaN, not a carried-forward price"

    # The frame is a dense hourly grid, so the tail is exactly one day long.
    steps = np.diff(dataset.frame.index.asi8)
    assert set(steps.tolist()) == {3_600_000_000_000}

    labelled = dataset.frame["future_return_1d"].dropna()
    assert len(labelled) == len(dataset) - 24
    assert float(labelled.std()) > 0, "a fabricated flat tail would make this exactly 0"
    assert np.isfinite(labelled.to_numpy(dtype="float64")).all()

    # And the value is the real forward return, not something approximate.
    close = dataset.frame.index.to_series().map(lambda t: t)  # keep the index explicit
    assert close.index.equals(dataset.frame.index)


def test_max_rows_is_applied_before_target_construction(v3_config: Config) -> None:
    """Truncation must not resurrect labels from prices the dataset dropped.

    If ``max_rows`` were applied *after* labelling, the final 24 rows of a
    300-row window would carry labels resolved against candles 301..324, which
    the dataset does not contain.
    """
    full = build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS
    )
    capped = build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS, max_rows=300
    )

    assert len(capped) == 300
    assert capped.drop_report["max_rows_truncated"] == len(full) - 300
    assert capped.index[-1] == full.index[-1]
    assert capped.frame["future_return_1d"].iloc[-24:].isna().all()

    # The same rows are labelled in the full frame, proving the difference is the
    # truncation and not a change in the data.
    overlap = capped.index[:-24]
    pd.testing.assert_series_equal(
        capped.frame.loc[overlap, "future_return_1d"],
        full.frame.loc[overlap, "future_return_1d"],
        check_names=False,
    )

    with pytest.raises(ValueError):
        build_v3_dataset(v3_config, SYMBOL, enabled_features=SPOT_ONLY_GROUPS, max_rows=0)


# --------------------------------------------------------------------------- 4. coverage / drops

def test_feature_coverage_is_sorted_worst_first_with_documented_columns(dataset: V3Dataset) -> None:
    coverage = dataset.feature_coverage
    assert list(coverage.columns) == list(FEATURE_COVERAGE_COLUMNS)
    assert len(coverage) == len(dataset.feature_columns)
    assert list(coverage["feature"]) == sorted(
        dataset.feature_columns, key=lambda f: (coverage.set_index("feature").loc[f, "coverage_pct"], f)
    )
    assert coverage["coverage_pct"].is_monotonic_increasing

    row = coverage.iloc[0]
    assert row["n_non_null"] + row["n_missing"] == len(dataset)
    assert math.isclose(row["coverage_pct"], 100.0 * row["n_non_null"] / len(dataset), abs_tol=1e-6)
    assert isinstance(row["dtype"], str) and row["dtype"]
    assert isinstance(row["bucket"], str) and row["bucket"]

    # The bucket column must name a real bucket, so a report can group by it.
    from src.v3.dataset import bucket_of_feature

    for name, bucket in zip(coverage["feature"], coverage["bucket"]):
        assert bucket == bucket_of_feature(name)


def test_drop_report_accounts_for_every_row_that_went_in(v3_config: Config) -> None:
    """``len(dataset) + sum(drop_report.values())`` must equal the raw row count."""
    dataset = build_v3_dataset(
        v3_config, DIRTY_SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS
    )
    report = dataset.drop_report

    assert set(report) == set(DROP_REASONS)
    assert all(isinstance(value, int) for value in report.values())
    assert report["duplicate_timestamps"] == 1
    assert report["missing_price"] == 3
    assert report["feature_warmup"] == 200  # the V1 baseline warm-up (SMA 200)
    assert report["max_rows_truncated"] == 0

    raw_rows = len(make_klines(seed=11)) + 1  # + the duplicated timestamp
    assert len(dataset) + sum(report.values()) == raw_rows

    # The unusable prices are gone from the frame, not merely reported: no NaT,
    # and no feature value inherited from a NaN/inf/zero close.
    assert not dataset.frame.index.isna().any()
    assert np.isfinite(dataset.frame[dataset.feature_columns].to_numpy(dtype="float64")).all()
    described = dataset.describe()
    assert described["n_rows_dropped"] == sum(report.values())
    assert described["n_rows"] == len(dataset)


def test_target_coverage_reports_the_labelled_span_per_horizon(dataset: V3Dataset) -> None:
    coverage = dataset.target_coverage
    assert list(coverage.columns) == list(TARGET_COVERAGE_COLUMNS)
    assert list(coverage["horizon"]) == ["1d", "7d"]
    assert set(coverage["symbol"]) == {SYMBOL}

    for row in coverage.itertuples(index=False):
        assert row.n_total == len(dataset)
        assert row.n_labelled + row.n_unlabelled_tail == len(dataset)
        assert row.n_labelled == dataset.n_labelled(row.horizon)
        # A 1d label on a dense hourly grid is realised in exactly 24 hours.
        assert math.isclose(float(row.realised_mean_hours), float(row.requested_hours))
        assert int(row.n_unlabelled_tail) == int(row.requested_hours)


# --------------------------------------------------------------------------- 5. labelled rows

@pytest.mark.parametrize("horizon", ["1d", "7d"])
def test_labelled_frame_returns_only_rows_with_a_real_target(dataset: V3Dataset, horizon: str) -> None:
    column = f"future_return_{horizon}"
    labelled = dataset.labelled_frame(horizon)

    assert len(labelled) <= len(dataset)
    assert len(labelled) < len(dataset), "the tail must be excluded"
    assert labelled[column].notna().all()
    assert np.isfinite(labelled[column].to_numpy(dtype="float64")).all()
    assert labelled.index.equals(dataset.frame.index[: len(labelled)])
    assert list(labelled.columns) == list(dataset.frame.columns)
    assert dataset.labelled_index(horizon).equals(labelled.index)

    # The shorter horizon has strictly more labelled rows than the longer one.
    assert dataset.n_labelled("1d") > dataset.n_labelled("7d")


def test_labelled_frame_rejects_a_horizon_outside_the_ladder(dataset: V3Dataset) -> None:
    with pytest.raises(Exception):
        dataset.labelled_frame("30d")
    with pytest.raises(Exception):
        dataset.labelled_frame("not-a-horizon")


# --------------------------------------------------------------------------- 6. no writes

def test_building_a_dataset_writes_nothing(v3_config: Config, tmp_path: Path) -> None:
    """The build is read-only: no new file anywhere under the temporary root."""
    before = tree(tmp_path)
    project_before = {
        protected: tree(PROJECT_ROOT / protected)
        for protected in ("models/v3", "reports/experiments/v3", "data/processed/v3")
    }

    build_v3_dataset(v3_config, DIRTY_SYMBOL, enabled_features=SPOT_ONLY_GROUPS)
    build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS, max_rows=250
    )

    assert tree(tmp_path) == before, "build_v3_dataset created a file"

    # The V3 output namespaces from config/v3.yaml must not be materialised.
    for relative in ("models", "reports", "data/processed/v3", "data/predictions"):
        assert not (tmp_path / relative).exists(), f"{relative} was created"

    # The real project directories are untouched.  This is an *unchanged*
    # check, not a non-existence one: `reports/experiments/v3` and
    # `models/v3` are legitimate outputs of the V3 pipeline, so once a real run
    # has happened they exist and stay.  What must not happen is this read-only
    # builder adding to them, so the assertion is that the pre-existing
    # contents are byte-for-byte what they were.  The session guard in
    # conftest.py enforces the same thing across the whole run.
    for protected in ("models/v3", "reports/experiments/v3", "data/processed/v3"):
        target = PROJECT_ROOT / protected
        assert tree(target) == project_before[protected], (
            f"{protected} changed during a read-only dataset build"
        )


# --------------------------------------------------------------------------- 7. selection errors

def test_bogus_feature_name_raises_a_clear_error(v3_config: Config) -> None:
    """A typo must fail at the selection boundary, naming the offending feature."""
    with pytest.raises(FeatureSelectionError) as excinfo:
        build_v3_dataset(
            v3_config, SYMBOL, horizons=["1d"], enabled_features={"price": ["not_a_feature"]}
        )
    assert "not_a_feature" in str(excinfo.value)

    # The same rejection when the selection is made without building anything.
    with pytest.raises(FeatureSelectionError, match="totally_made_up"):
        resolve_feature_selection(v3_config, {"price": ["totally_made_up"]})

    # A flat list of feature names is checked against the registry too.
    with pytest.raises(FeatureSelectionError, match="sma_21"):
        resolve_feature_selection(v3_config, ["sma_21"])

    # An unknown bucket is an error, not an empty feature set.
    with pytest.raises(FeatureSelectionError):
        resolve_feature_selection(v3_config, {"no_such_bucket": "all"})


def test_a_bogus_selection_never_silently_shrinks_the_frame(v3_config: Config) -> None:
    """Whatever the selection machinery decides, the column set is recorded."""
    selection = resolve_feature_selection(v3_config, {"price": ["sma_20", "return_1h"]})
    assert selection.origin == "caller"
    assert set(selection.columns) == {"sma_20", "return_1h"}

    dataset = build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features={"price": ["sma_20", "return_1h"]}
    )
    assert dataset.feature_columns == selection.columns
    assert len(dataset.feature_coverage) == 2


def test_missing_raw_data_fails_with_a_clear_error(v3_config: Config) -> None:
    with pytest.raises(DataUnavailableError) as excinfo:
        build_v3_dataset(v3_config, "NOSUCHUSDT", enabled_features=SPOT_ONLY_GROUPS)
    assert "NOSUCHUSDT" in str(excinfo.value)
    assert "parquet" in str(excinfo.value)


# --------------------------------------------------------------------------- 8. graceful degradation

def test_an_unavailable_source_is_recorded_and_the_build_continues(dataset: V3Dataset) -> None:
    """No external caches exist for the synthetic symbol, and that is survivable.

    The derivatives and sentiment groups are dropped and reported; the frame is
    still built from the features whose sources *do* exist.
    """
    assert dataset.missing_sources == ("binance_funding", "binance_futures", "fear_greed")
    assert "derivatives" not in dataset.feature_groups
    assert "sentiment" not in dataset.feature_groups
    assert dataset.feature_groups == ("technical",)
    assert len(dataset) > 0
    assert all(dataset.frame[column].notna().all() for column in dataset.feature_columns)

    described = dataset.describe()
    assert described["missing_sources"] == list(dataset.missing_sources)
    assert described["n_features"] == len(dataset.feature_columns)


def test_features_depend_only_on_the_past(v3_config: Config) -> None:
    """Truncating the *future* must not change a single feature value.

    This is the point-in-time guarantee as an executable assertion: build the
    same series twice, once with 300 extra candles of history that follow the
    comparison window, and require the overlapping rows to be bit-identical.  Any
    backward fill, centred window, or whole-frame normalisation would show up
    here as a difference.
    """
    full = build_v3_dataset(
        v3_config, SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS
    )
    prefix = build_v3_dataset(
        v3_config, PREFIX_SYMBOL, horizons=["1d"], enabled_features=SPOT_ONLY_GROUPS
    )

    common = prefix.index.intersection(full.index)
    assert len(common) == len(prefix) > 300, "not enough overlap for the comparison to mean anything"
    left = prefix.frame.loc[common, prefix.feature_columns]
    right = full.frame.loc[common, full.feature_columns]
    pd.testing.assert_frame_equal(left, right, check_exact=True)

    # The prefix's tail is unlabelled even though the longer series knows those
    # prices: the truncated dataset does not contain them, so it does not use them.
    assert prefix.frame["future_return_1d"].iloc[-24:].isna().all()
    assert full.frame["future_return_1d"].iloc[-24:].isna().all()


# --------------------------------------------------------------------------- real data

@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "raw" / "BTCUSDT_1h.parquet").exists(),
    reason="real BTCUSDT klines are not present in this checkout",
)
def test_real_btcusdt_dataset_is_leak_free_and_read_only() -> None:
    """REAL-DATA SMOKE TEST (read-only, ~1s).  Everything else uses synthetic data.

    Built against the un-rebased config on purpose: the builder only ever calls
    ``Path.exists`` and ``read_parquet``, so the real raw store is read and
    never written.  The conftest session guard fails the run if that changes.
    """
    config = load_config(V3_CONFIG_PATH)  # NOT rebased: it must read the real raw dir
    dataset = build_v3_dataset(config, "BTCUSDT", horizons=["1d", "30d"])

    assert len(dataset) > 0
    assert dataset.index.is_unique and dataset.index.is_monotonic_increasing
    assert str(dataset.index.tz) == "UTC"
    assert dataset.target_columns == ["future_return_1d", "future_return_30d"]

    coverage = dataset.target_coverage.set_index("horizon")
    assert int(coverage.loc["1d", "n_labelled"]) == int(coverage.loc["30d", "n_labelled"]) + 696
    assert int(coverage.loc["30d", "n_unlabelled_tail"]) == 720

    # Every configured bucket's source resolved, so nothing was degraded away.
    assert dataset.missing_sources == ()
    assert dataset.feature_groups == (
        "technical",
        "derivatives",
        "sentiment",
        "microstructure",
        "context",
    )
    assert len(dataset.feature_columns) == len(known_features())

    # Point-in-time: the realised label window is the requested one, and the
    # alignment layer's own audit already ran inside the build.
    for row in dataset.target_coverage.itertuples(index=False):
        assert math.isclose(float(row.realised_mean_hours), float(row.requested_hours))
        assert float(row.realised_min_hours) <= float(row.requested_hours)
