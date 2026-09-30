"""Purged, embargoed walk-forward splits for overlapping forward-return labels.

Why ordinary K-fold is wrong here
---------------------------------
Every V3 label is a *forward* return.  The label for a row at ``t`` is a
function of prices in ``[t, t + H]``.  Two consequences break standard
validation:

1. **Overlapping labels.**  Rows one hour apart share almost all of their future
   window.  At ``H = 30d`` two adjacent rows overlap by 99.86%.  Treating them
   as independent observations understates uncertainty enormously.

2. **Temporal leakage across the split boundary.**  A training row at
   ``t_train_end`` has a label that depends on prices *after* the split.  With
   a random split, validation rows routinely share their future window with
   training rows and the model is scored on information it has already seen.

The fix used here is the standard López de Prado construction: a **purge** on
each side of the boundary that removes every training row whose label window
touches the evaluation period, plus an **embargo** after the evaluation period.

Geometry of a single fold
-------------------------
For a horizon of length ``H``, in timestamp space::

    |<-- train -->|<- purge ->|<- val ->|<- purge ->|<- embargo ->|<-- test -->|
                                                            ^test starts here

* ``purge`` = ``H``.  Training rows with ``t + H >= val_start`` are dropped,
  because their label reads a price inside the validation window.
* ``embargo`` = ``H`` by default.  The rows immediately after validation are
  skipped before the *next* fold's training window begins, so a fold's
  validation labels cannot bleed into a later fold's training set.  Set
  ``embargo=HORIZON`` to disable it, which is what the leakage tests do in
  order to assert the embargo is load-bearing.

The test block is placed *after* the embargo, and its own rows are additionally
restricted to those whose label window is fully observed - otherwise a test row
near the end of the data would be scored against a NaN target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger
from src.v3.horizons import Horizon

logger = get_logger("v3.splits")


class SplitError(ValueError):
    """Raised when the data cannot support the requested fold geometry."""


@dataclass(frozen=True)
class HorizonFold:
    """One (train, validation, test) split for a single horizon.

    All three index sets are **positional** offsets into the horizon's labelled
    row list, so they can be used directly with ``.iloc``.  The timestamp bounds
    are carried alongside for reporting and for the leakage assertions.
    """

    fold: int
    horizon: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge: pd.Timedelta
    embargo: pd.Timedelta
    n_purged_train: int = 0
    n_embargoed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "horizon": self.horizon,
            "n_train": int(self.train.size),
            "n_validation": int(self.validation.size),
            "n_test": int(self.test.size),
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "validation_start": str(self.validation_start),
            "validation_end": str(self.validation_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "purge": str(self.purge),
            "embargo": str(self.embargo),
            "n_purged_train": int(self.n_purged_train),
            "n_embargoed": int(self.n_embargoed),
        }

    def geometry(self) -> dict[str, object]:
        """Human-readable assertion payload for the leakage tests."""
        return {
            "horizon": self.horizon,
            "purge": self.purge,
            "embargo": self.embargo,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "test_start": self.test_start,
        }


def embargo_mode(mode: str | None) -> pd.Timedelta | None:
    """Interpret the configured embargo, where ``None`` means "disable"."""
    if mode is None:
        return None
    key = str(mode).strip().lower()
    if key in {"none", "off", "disabled", "false"}:
        return None
    if key in {"horizon", "auto", "same_as_purge"}:
        return pd.Timedelta("0s")  # replaced by the caller with the horizon
    raise SplitError(
        f"Unknown v3.splits.embargo {mode!r}; use 'horizon' (default) or 'none'"
    )


@dataclass
class PurgedWalkForwardSplitter:
    """Chronological walk-forward splitter with purging and embargo.

    Parameters
    ----------
    horizon:
        The horizon this split is sized for.  Purge and embargo both default to
        this horizon's duration, because that is the width over which a label
        can see into a neighbouring block.
    n_splits:
        Number of test blocks to walk forward over.
    embargo:
        ``"horizon"`` (default), ``"none"``, or an explicit duration string.
        Controls the gap between the validation block and the test block.
    test_fraction:
        Fraction of the *usable* span each test block occupies.
    validation_fraction:
        Fraction of the usable span each validation block occupies.  The
        validation block is the bounded window that sits immediately before the
        embargo, so it must be sized explicitly - leaving it unbounded would
        swallow the whole history and leave no training data at all.
    min_train_fraction:
        Smallest permitted training fraction, so early folds are not fitted on a
        handful of rows.
    test_step:
        Step between consecutive test block starts.  ``None`` means
        non-overlapping test blocks packed to the end of the data.
    """

    horizon: Horizon
    n_splits: int = 5
    embargo: str | None = "horizon"
    test_fraction: float = 0.1
    validation_fraction: float = 0.1
    min_train_fraction: float = 0.2
    test_step: pd.Timedelta | None = None
    geometry: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.n_splits) < 1:
            raise SplitError(f"n_splits must be >= 1, got {self.n_splits}")
        for name, value in (
            ("test_fraction", self.test_fraction),
            ("validation_fraction", self.validation_fraction),
            ("min_train_fraction", self.min_train_fraction),
        ):
            if not 0 < float(value) < 1:
                raise SplitError(f"{name} must be in (0, 1), got {value}")
        if float(self.min_train_fraction) + float(self.validation_fraction) + float(
            self.test_fraction
        ) >= 1.0:
            raise SplitError(
                "min_train_fraction + validation_fraction + test_fraction must be < 1, "
                f"got {self.min_train_fraction} + {self.validation_fraction} + "
                f"{self.test_fraction}"
            )

    # ------------------------------------------------------------------ utils
    @property
    def purge(self) -> pd.Timedelta:
        return self.horizon.delta

    @property
    def embargo_delta(self) -> pd.Timedelta:
        mode = embargo_mode(self.embargo)
        if mode is None:
            return pd.Timedelta(0)
        if self.embargo is not None and str(self.embargo).strip().lower() in {
            "horizon",
            "auto",
            "same_as_purge",
        }:
            return self.horizon.delta
        return mode

    # ------------------------------------------------------------------ split
    def split(self, index: pd.DatetimeIndex) -> Iterator[HorizonFold]:
        """Yield folds over ``index``, the labelled rows for one horizon.

        ``index`` must already be restricted to rows with a real (non-NaN)
        target for this horizon - see :meth:`src.v3.targets
        .FutureReturnTargets.labelled_mask`.
        """
        index = pd.DatetimeIndex(index)
        if len(index) < 3:
            raise SplitError(
                f"Need at least 3 labelled rows for horizon {self.horizon.label}, got {len(index)}"
            )
        if not index.is_monotonic_increasing:
            raise SplitError("Split index must be sorted ascending")
        if not index.is_unique:
            raise SplitError("Split index must have unique timestamps")

        n = len(index)
        values = index.asi8
        start_ts, end_ts = index[0], index[-1]
        span = end_ts - start_ts

        test_span = span * float(self.test_fraction)
        val_span = span * float(self.validation_fraction)
        for name, width in (("test", test_span), ("validation", val_span)):
            if width <= pd.Timedelta(0):
                raise SplitError(
                    f"{name} block for horizon {self.horizon.label} rounds to zero width; "
                    "the labelled span is too short"
                )
        step = self.test_step if self.test_step is not None else test_span
        if step <= pd.Timedelta(0):
            raise SplitError("test_step must be positive")

        # Test blocks are packed to the *end* of the data so the most recent
        # period - the most relevant one - is always covered.
        first_test_start = end_ts - test_span - step * (int(self.n_splits) - 1)
        first_test_start = max(first_test_start, start_ts)

        embargo_delta = self.embargo_delta
        produced = 0
        for fold in range(int(self.n_splits)):
            test_start = first_test_start + step * fold
            test_end = test_start + test_span

            # Test rows must have their full label window inside the data, which
            # the caller already guaranteed by passing labelled rows only.
            test = np.flatnonzero((values >= test_start.value) & (values <= test_end.value))
            if test.size == 0:
                continue

            # The validation block is *bounded* and sits immediately before the
            # embargo.  This bound is load-bearing: an unbounded validation set
            # would reach all the way back to the first observation, making
            # `validation_start` the beginning of the data and purging every
            # training row.
            val_start = test_start - embargo_delta - val_span
            if val_start < start_ts:
                val_start = start_ts

            validation = np.flatnonzero(
                (values >= val_start.value) & (values < test_start.value - embargo_delta.value)
            )
            if validation.size == 0:
                continue

            # Everything before the validation block is a training candidate.
            train_candidates = np.flatnonzero(values < val_start.value)
            if train_candidates.size == 0:
                continue

            # Purge: a training row is unsafe if its label window reaches the
            # validation period.  This is the same definition shared with
            # src.v3.targets.overlaps_boundary.
            candidate_ts = index[train_candidates]
            safe = np.asarray((candidate_ts + self.purge) < val_start)
            train = train_candidates[safe]
            n_purged = int(train_candidates.size - train.size)

            if train.size == 0:
                continue
            if train.size < max(10, int(n * float(self.min_train_fraction))):
                logger.debug(
                    f"fold {fold} of {self.horizon.label}: only {train.size} training rows "
                    f"after purging, below min_train_fraction"
                )
                continue

            embargoed = 0
            if embargo_delta > pd.Timedelta(0):
                embargoed = int(
                    np.count_nonzero(
                        (values >= (test_start - embargo_delta).value) & (values < test_start.value)
                    )
                )

            produced += 1
            yield HorizonFold(
                fold=produced,
                horizon=self.horizon.label,
                train=train,
                validation=validation,
                test=test,
                train_start=index[train[0]],
                train_end=index[train[-1]],
                validation_start=index[validation[0]],
                validation_end=index[validation[-1]],
                test_start=index[test[0]],
                test_end=index[test[-1]],
                purge=self.purge,
                embargo=embargo_delta,
                n_purged_train=n_purged,
                n_embargoed=embargoed,
            )

        if produced == 0:
            raise SplitError(
                f"Horizon {self.horizon.label} produced no usable folds. The purge of "
                f"{self.purge} plus the embargo of {embargo_delta} consumes more of the "
                f"{span} labelled span than is available. Shorten the horizon, reduce "
                f"n_splits, or disable the embargo."
            )

    def describe(self, index: pd.DatetimeIndex) -> dict[str, object]:
        return {
            "horizon": self.horizon.label,
            "purge": str(self.purge),
            "embargo": str(self.embargo_delta),
            "n_splits_requested": int(self.n_splits),
            "test_fraction": float(self.test_fraction),
            "validation_fraction": float(self.validation_fraction),
            "min_train_fraction": float(self.min_train_fraction),
            "labelled_rows": int(len(index)),
        }


def assert_no_overlap(fold: HorizonFold) -> None:
    """Raise if a fold's blocks are not cleanly separated.

    Used by the leakage tests.  Checks three things that must hold for any
    horizon:

    1. train end < validation start (chronological)
    2. validation end < test start (chronological)
    3. every training row's label window ends strictly before the validation
       block begins - i.e. the purge was actually applied
    """
    if not fold.train_end < fold.validation_start:
        raise AssertionError(
            f"fold {fold.fold}/{fold.horizon}: training is not strictly before validation"
        )
    if not fold.validation_end < fold.test_start:
        raise AssertionError(
            f"fold {fold.fold}/{fold.horizon}: validation is not strictly before test"
        )
    if fold.purge > pd.Timedelta(0) and fold.train_end + fold.purge >= fold.validation_start:
        raise AssertionError(
            f"fold {fold.fold}/{fold.horizon}: purge too small. Last training row ends "
            f"{fold.train_end} and its {fold.purge} label window reaches "
            f"{fold.train_end + fold.purge}, which is at or after the validation start "
            f"{fold.validation_start}."
        )
