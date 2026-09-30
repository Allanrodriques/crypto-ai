"""Dataset-layer tests: label correctness, split purging, and leakage guards.

The purge is the single most important property in this project.  A training
label is resolved ``horizon`` candles into the future, so the tail of every
non-final split is dropped: otherwise the price a training row is scored against
would live inside the validation or test period.  These tests check the purge
*empirically*, by recomputing where each label resolves, rather than trusting the
slicing arithmetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.dataset.dataset_builder import (
    DataSplit,
    Dataset,
    DatasetBuilder,
    LeakageError,
    build_dataset,
)
from src.features.feature_engineering import FeatureEngineer
from tests.conftest import TEST_FEATURES, make_klines

HORIZON = 6
THRESHOLD = 0.005


@pytest.fixture
def builder(config) -> DatasetBuilder:
    return DatasetBuilder(config)


@pytest.fixture
def dataset(config, klines) -> Dataset:
    return DatasetBuilder(config).build(klines)


# --------------------------------------------------------------------------- label

def test_future_return_matches_a_hand_computed_ratio(builder, klines):
    labels = builder.build_target(klines["close"])
    position = 100
    expected = klines["close"].iloc[position + HORIZON] / klines["close"].iloc[position] - 1.0
    assert labels["future_return"].iloc[position] == pytest.approx(expected, rel=1e-12)


def test_target_is_the_threshold_rule(builder, klines):
    labels = builder.build_target(klines["close"])
    resolved = labels["target"].notna()
    assert np.array_equal(
        labels.loc[resolved, "target"].to_numpy(),
        (labels.loc[resolved, "future_return"] >= THRESHOLD).to_numpy().astype("float64"),
    )


def test_the_last_horizon_rows_have_no_resolvable_label(builder, klines):
    labels = builder.build_target(klines["close"])
    assert labels["target"].isna().sum() == HORIZON
    assert labels["future_return"].isna().sum() == HORIZON
    assert labels["target"].iloc[:-HORIZON].notna().all()
    # The NaN must sit at the end, not scattered through the series.
    assert labels["target"].notna().iloc[:-HORIZON].all()


def test_target_uses_only_the_future_close(builder):
    """Perturbing close[t+h] must change the label at t, and nothing else."""
    close = pd.Series(
        [100.0] * 60, index=pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC"), name="close"
    )
    baseline = builder.build_target(close)

    position = 20
    poisoned = close.copy()
    poisoned.iloc[position + HORIZON] *= 1.2  # enough to cross +0.5%
    actual = builder.build_target(poisoned)

    assert actual["target"].iloc[position] != baseline["target"].iloc[position]
    untouched = [i for i in range(len(close)) if i not in (position, position + HORIZON)]
    pd.testing.assert_series_equal(actual["target"].iloc[untouched], baseline["target"].iloc[untouched])


def test_exactly_at_the_threshold_counts_as_up(builder):
    """The label rule is ``>= threshold``, so the boundary itself is 'up'.

    Uses a binary-exact threshold (0.5) so the comparison is tested rather than
    floating-point noise: at the shipped 0.005, ``100.5/100 - 1`` evaluates to
    0.004999999999999893, which is legitimately 'down'.
    """
    index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=index)
    position = 10

    builder.threshold = 0.5
    close.iloc[position + HORIZON] = 150.0
    labels = builder.build_target(close)
    assert labels["future_return"].iloc[position] == 0.5
    assert labels["target"].iloc[position] == 1.0

    close.iloc[position + HORIZON] = 149.0
    below = builder.build_target(close)
    assert below["future_return"].iloc[position] < 0.5
    assert below["target"].iloc[position] == 0.0


def test_shipped_threshold_labels_moves_in_the_expected_direction(builder):
    index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=index)
    position = 10

    close.iloc[position + HORIZON] = 102.0  # +2%
    assert builder.build_target(close)["target"].iloc[position] == 1.0

    close.iloc[position + HORIZON] = 98.0  # -2%
    assert builder.build_target(close)["target"].iloc[position] == 0.0


# --------------------------------------------------------------------------- frame

def test_dataset_frame_has_no_missing_values(dataset):
    assert not dataset.frame.isna().any().any()
    assert dataset.frame.index.is_monotonic_increasing
    assert dataset.frame.index.is_unique


def test_features_and_labels_line_up(dataset, klines):
    assert set(dataset.feature_columns) == set(
        FeatureEngineer(TEST_FEATURES).build(klines).columns
    )
    # The label is reproduced from the market close, at the same timestamps.
    expected = (klines["close"].shift(-HORIZON) / klines["close"] - 1.0).reindex(dataset.frame.index)
    np.testing.assert_allclose(dataset.frame["future_return"].to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_warmup_and_unresolvable_rows_are_dropped(dataset, klines, config):
    engineer = FeatureEngineer(TEST_FEATURES)
    expected_rows = len(klines) - engineer.required_history - HORIZON
    assert len(dataset.frame) == expected_rows


def test_build_rejects_an_empty_frame(builder):
    empty = make_klines(0)
    with pytest.raises(ValueError, match="empty kline frame"):
        builder.build(empty)


def test_build_rejects_history_shorter_than_the_warmup(config):
    builder = DatasetBuilder(config)
    with pytest.raises(ValueError, match="warm-up|Feature matrix is empty"):
        builder.build(make_klines(15))


# --------------------------------------------------------------------------- splits

def test_splits_are_chronological_and_disjoint(dataset):
    order = ["train", "validation", "test"]
    previous: DataSplit | None = None
    for name in order:
        split = dataset.split(name)
        assert len(split) > 0, f"{name} split is empty"
        assert split.index.is_monotonic_increasing
        if previous is not None:
            assert split.index.min() > previous.index.max()
        previous = split


def test_split_ratios_are_close_to_configured(dataset, config):
    total = sum(len(s) for s in dataset.splits.values())
    for name in ("train", "validation", "test"):
        expected = float(config.split[f"{name}_ratio"]) * total
        assert len(dataset.split(name)) == pytest.approx(expected, rel=0.12), name


def test_splits_never_shuffle_time(dataset):
    """A shuffled split would make timestamps interleave; they must not."""
    combined = pd.DatetimeIndex(np.concatenate([s.index.values for s in dataset.splits.values()]))
    assert combined.is_monotonic_increasing
    assert combined.is_unique


# --------------------------------------------------------------------------- purge

def _label_resolution_time(split: DataSplit, horizon_ms: int) -> pd.DatetimeIndex:
    return split.index + pd.Timedelta(milliseconds=horizon_ms)


def test_no_training_label_resolves_inside_the_validation_period(dataset):
    train, validation = dataset.split("train"), dataset.split("validation")
    resolutions = _label_resolution_time(train, dataset.horizon_ms)
    assert resolutions.max() <= validation.index.min(), (
        f"last train label resolves at {resolutions.max()}, "
        f"validation starts at {validation.index.min()}"
    )


def test_no_validation_label_resolves_inside_the_test_period(dataset):
    validation, test = dataset.split("validation"), dataset.split("test")
    resolutions = _label_resolution_time(validation, dataset.horizon_ms)
    assert resolutions.max() <= test.index.min()


def test_purge_removes_exactly_the_horizon_from_non_final_splits(dataset):
    """The tail must be ``horizon`` rows, not more."""
    total = len(dataset.frame)
    n_train = int(total * dataset.metadata["split"]["train_ratio"])
    n_valid = int(total * dataset.metadata["split"]["validation_ratio"])
    assert len(dataset.split("train")) == n_train - HORIZON
    assert len(dataset.split("validation")) == n_valid - HORIZON


def test_test_split_is_not_purged(dataset):
    total = len(dataset.frame)
    n_train = int(total * dataset.metadata["split"]["train_ratio"])
    n_valid = int(total * dataset.metadata["split"]["validation_ratio"])
    assert len(dataset.split("test")) == total - n_train - n_valid


def _resplit(dataset: Dataset, splits: dict[str, DataSplit]) -> Dataset:
    """Rebuild a Dataset around hand-made splits, keeping the real label horizon."""
    return Dataset(
        symbol=dataset.symbol,
        interval=dataset.interval,
        horizon_candles=dataset.horizon_candles,
        threshold=dataset.threshold,
        feature_columns=dataset.feature_columns,
        frame=dataset.frame,
        splits=splits,
        metadata=dataset.metadata,
        horizon_ms=dataset.horizon_ms,
    )


def test_assert_no_leakage_passes_on_a_real_dataset(dataset):
    assert dataset.assert_no_leakage() is dataset


def test_assert_no_leakage_catches_a_too_small_purge(dataset):
    """Break the purge deliberately; the guard must notice.

    The training block is extended to the very last row it had before purging,
    so its final label resolves strictly inside the validation period.
    """
    train, validation = dataset.split("train"), dataset.split("validation")
    extended = dataset.frame.iloc[
        (dataset.frame.index < validation.index.min())
        & (dataset.frame.index <= train.index.max() + pd.Timedelta(hours=HORIZON))
    ]
    broken = dict(dataset.splits)
    broken["train"] = DataSplit(
        name="train",
        X=extended[dataset.feature_columns],
        y=extended["target"],
        frame=extended,
    )
    assert len(extended) > len(train), "fixture must actually restore purged rows"
    with pytest.raises(LeakageError, match="Purge is too small"):
        _resplit(dataset, broken).assert_no_leakage()


def test_assert_no_leakage_catches_overlapping_splits(dataset):
    train, validation = dataset.split("train"), dataset.split("validation")
    overlapping_frame = pd.concat([train.frame.tail(5), validation.frame])
    broken = dict(dataset.splits)
    broken["validation"] = DataSplit(
        name="validation",
        X=overlapping_frame[dataset.feature_columns],
        y=overlapping_frame["target"],
        frame=overlapping_frame,
    )
    with pytest.raises(LeakageError, match="not strictly after"):
        _resplit(dataset, broken).assert_no_leakage()


def test_assert_no_leakage_catches_an_unsorted_split(dataset):
    broken = dict(dataset.splits)
    test_split = dataset.split("test")
    broken["test"] = DataSplit(
        name="test",
        X=test_split.X.iloc[::-1],
        y=test_split.y.iloc[::-1],
        frame=test_split.frame.iloc[::-1],
    )
    with pytest.raises(LeakageError, match="sorted, unique"):
        _resplit(dataset, broken).assert_no_leakage()


def test_features_in_the_test_split_do_not_encode_the_label(dataset):
    """The design matrix must never contain the target or the forward return."""
    forbidden = {"target", "future_return"}
    for split in dataset:
        assert not (forbidden & set(split.X.columns))
        assert set(split.X.columns) == set(dataset.feature_columns)
        assert split.y.name == "target"


# --------------------------------------------------------------------------- persistence

def test_dataset_roundtrips_through_disk(dataset, tmp_path, config):
    paths = dataset.save(tmp_path)
    assert paths["frame"].exists() and paths["metadata"].exists()

    restored = Dataset.load(paths["frame"], config)
    assert restored.feature_columns == dataset.feature_columns
    assert list(restored.split("train").index) == list(dataset.split("train").index)
    np.testing.assert_allclose(
        restored.split("train").X.to_numpy(), dataset.split("train").X.to_numpy()
    )
    np.testing.assert_array_equal(
        restored.split("train").y.to_numpy(), dataset.split("train").y.to_numpy()
    )
    restored.assert_no_leakage()


def test_metadata_sidecar_records_the_experiment(dataset, tmp_path):
    paths = dataset.save(tmp_path)
    payload = json.loads(paths["metadata"].read_text())
    assert payload["symbol"] == dataset.symbol
    assert payload["interval"] == dataset.interval
    assert payload["n_features"] == len(dataset.feature_columns)
    assert payload["target"]["horizon_candles"] == HORIZON
    assert payload["target"]["threshold"] == THRESHOLD
    assert payload["target"]["definition"], "the target formula must be recorded in words"
    assert payload["feature_columns"] == dataset.feature_columns
    assert payload["split"]["method"]
    assert payload["split"]["purge_candles"] == HORIZON
    for name in ("train", "validation", "test"):
        block = payload["split"]["blocks"][name]
        assert block["rows"] > 0
        assert "class_distribution" in block
        assert block["start"] < block["end"]


def test_class_distribution_is_reported_and_correct(dataset):
    for split in dataset:
        distribution = split.class_distribution()
        assert distribution["n"] == len(split)
        assert distribution["n_up"] + distribution["n_down"] == len(split)
        assert distribution["share_up"] == pytest.approx(distribution["n_up"] / len(split))
        assert distribution["imbalance_ratio"] == pytest.approx(
            max(distribution["n_up"], distribution["n_down"]) / min(distribution["n_up"], distribution["n_down"])
        )


def test_functional_wrapper_matches_the_class(config, klines):
    built = build_dataset(config, klines)
    direct = DatasetBuilder(config).build(klines)
    assert list(built.split("train").index) == list(direct.split("train").index)
    np.testing.assert_allclose(built.split("train").X.to_numpy(), direct.split("train").X.to_numpy())


def test_build_is_deterministic(config, klines):
    a = DatasetBuilder(config).build(klines)
    b = DatasetBuilder(config).build(klines)
    np.testing.assert_allclose(a.split("train").X.to_numpy(), b.split("train").X.to_numpy())
    np.testing.assert_array_equal(a.split("test").y.to_numpy(), b.split("test").y.to_numpy())
