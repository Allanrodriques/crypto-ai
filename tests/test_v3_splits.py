"""Leakage tests for the V3 purged walk-forward splitter.

These are the tests that protect the headline claim: that V3 scores are
out-of-sample.  Every assertion here exists because the naive alternative -
random K-fold, or a plain ``train_test_split`` - silently fails it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v3.horizons import build_horizon_ladder
from src.v3.splits import PurgedWalkForwardSplitter, SplitError, assert_no_overlap
from src.v3.targets import build_future_returns

HORIZONS = ["1d", "7d", "30d", "90d", "180d"]


@pytest.fixture(scope="module")
def ladder():
    return build_horizon_ladder({"horizons": HORIZONS})


@pytest.fixture(scope="module")
def labelled_index(ladder):
    """Real BTC close series, restricted to rows with a real 180d target."""
    frame = pd.read_parquet("data/raw/BTCUSDT_1h.parquet", columns=["close"])
    targets = build_future_returns(frame["close"], ladder)
    mask = targets.labelled_mask("180d")
    return frame.index[mask.to_numpy()]


def test_folds_are_chronologically_separated(ladder, labelled_index) -> None:
    for horizon in ladder:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=4,
            test_fraction=0.08,
            validation_fraction=0.08,
        )
        for fold in splitter.split(labelled_index):
            assert_no_overlap(fold)
            assert fold.train_end < fold.validation_start < fold.validation_end
            assert fold.validation_end < fold.test_start < fold.test_end


def test_purge_removes_exactly_one_horizon_of_rows(ladder, labelled_index) -> None:
    """Purged training rows must equal the horizon's own bar count."""
    for horizon in ladder:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=3,
            test_fraction=0.08,
            validation_fraction=0.08,
        )
        fold = next(iter(splitter.split(labelled_index)))
        assert fold.n_purged_train == horizon.nominal_bars, (
            f"{horizon.label}: purged {fold.n_purged_train} rows, "
            f"expected {horizon.nominal_bars}"
        )


def test_no_training_label_window_reaches_validation_or_test(ladder, labelled_index) -> None:
    """The core guarantee: max(t + H) over training rows precedes validation."""
    for horizon in ladder:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=3,
            test_fraction=0.08,
            validation_fraction=0.08,
        )
        for fold in splitter.split(labelled_index):
            train_ts = labelled_index[fold.train]
            latest_label_end = (train_ts + horizon.delta).max()
            assert latest_label_end < fold.validation_start
            assert latest_label_end < fold.test_start


def test_validation_labels_do_not_reach_test(ladder, labelled_index) -> None:
    """The embargo is what stops validation labels reading test-period prices."""
    for horizon in ladder:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=3,
            test_fraction=0.08,
            validation_fraction=0.08,
        )
        for fold in splitter.split(labelled_index):
            val_ts = labelled_index[fold.validation]
            assert (val_ts + horizon.delta).max() <= fold.test_start
            gap = fold.test_start - fold.validation_end
            assert gap >= horizon.delta - pd.Timedelta(hours=1)


def test_embargo_is_load_bearing(ladder, labelled_index) -> None:
    """With embargo='none' the gap must actually shrink - proving it is applied."""
    horizon = next(h for h in ladder if h.label == "30d")
    common = dict(
        n_splits=3, test_fraction=0.08, validation_fraction=0.08
    )
    with_embargo = next(
        iter(PurgedWalkForwardSplitter(horizon=horizon, embargo="horizon", **common)
             .split(labelled_index))
    )
    without = next(
        iter(PurgedWalkForwardSplitter(horizon=horizon, embargo="none", **common)
             .split(labelled_index))
    )

    assert with_embargo.embargo == horizon.delta
    assert without.embargo == pd.Timedelta(0)
    gap_on = with_embargo.test_start - with_embargo.validation_end
    gap_off = without.test_start - without.validation_end
    assert gap_off < gap_on
    assert with_embargo.n_embargoed == horizon.nominal_bars
    assert without.n_embargoed == 0


def test_test_blocks_are_disjoint_across_folds(ladder, labelled_index) -> None:
    for horizon in ladder:
        splitter = PurgedWalkForwardSplitter(
            horizon=horizon,
            n_splits=4,
            test_fraction=0.08,
            validation_fraction=0.08,
        )
        seen: set[int] = set()
        previous_end: pd.Timestamp | None = None
        for fold in splitter.split(labelled_index):
            block = set(fold.test.tolist())
            assert not (block & seen), f"{horizon.label}: test rows reused across folds"
            seen |= block
            if previous_end is not None:
                assert fold.test_start > previous_end
            previous_end = fold.test_end


def test_training_window_expands_across_folds(ladder, labelled_index) -> None:
    horizon = next(h for h in ladder if h.label == "7d")
    folds = list(
        PurgedWalkForwardSplitter(
            horizon=horizon, n_splits=4, test_fraction=0.08, validation_fraction=0.08
        ).split(labelled_index)
    )
    sizes = [f.train.size for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert len(folds) == 4


def test_longest_horizon_still_yields_usable_folds(ladder, labelled_index) -> None:
    """180d with a 180d purge + embargo is the tightest case; it must still work."""
    horizon = next(h for h in ladder if h.label == "180d")
    folds = list(
        PurgedWalkForwardSplitter(
            horizon=horizon, n_splits=3, test_fraction=0.08, validation_fraction=0.08
        ).split(labelled_index)
    )
    assert len(folds) >= 3
    for fold in folds:
        assert fold.train.size > 1000


def test_shuffled_index_is_rejected(ladder, labelled_index) -> None:
    horizon = next(h for h in ladder if h.label == "1d")
    rng = np.random.default_rng(0)
    shuffled = labelled_index[rng.permutation(len(labelled_index))]
    with pytest.raises(SplitError, match="sorted"):
        list(PurgedWalkForwardSplitter(horizon=horizon, n_splits=2).split(shuffled))


def test_duplicate_index_is_rejected(ladder, labelled_index) -> None:
    horizon = next(h for h in ladder if h.label == "1d")
    with pytest.raises(SplitError, match="unique"):
        list(
            PurgedWalkForwardSplitter(horizon=horizon, n_splits=2).split(
                labelled_index.insert(len(labelled_index), labelled_index[-1])
            )
        )


def test_invalid_fractions_rejected(ladder) -> None:
    horizon = next(h for h in ladder if h.label == "1d")
    with pytest.raises(SplitError):
        PurgedWalkForwardSplitter(horizon=horizon, test_fraction=0.0)
    with pytest.raises(SplitError):
        PurgedWalkForwardSplitter(horizon=horizon, validation_fraction=1.5)
    with pytest.raises(SplitError, match="must be < 1"):
        PurgedWalkForwardSplitter(
            horizon=horizon,
            test_fraction=0.5,
            validation_fraction=0.4,
            min_train_fraction=0.3,
        )
    with pytest.raises(SplitError, match="embargo"):
        PurgedWalkForwardSplitter(horizon=horizon, embargo="sometimes").embargo_delta


def test_fold_positions_are_usable_for_iloc(ladder, labelled_index) -> None:
    """Folds carry positional indices that must index straight into a frame."""
    horizon = next(h for h in ladder if h.label == "30d")
    frame = pd.read_parquet("data/raw/BTCUSDT_1h.parquet", columns=["close"])
    targets = build_future_returns(frame["close"], build_horizon_ladder({"horizons": HORIZONS}))
    mask = targets.labelled_mask("30d").to_numpy()
    subset = frame.loc[mask]

    fold = next(
        iter(
            PurgedWalkForwardSplitter(
                horizon=horizon, n_splits=3, test_fraction=0.08, validation_fraction=0.08
            ).split(subset.index)
        )
    )
    train = subset.iloc[fold.train]
    assert len(train) == fold.train.size
    assert train.index.max() < fold.validation_start
    y = targets.frame.loc[train.index, "future_return_30d"]
    assert y.notna().all(), "training rows must have real labels"
