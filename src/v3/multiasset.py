"""Cross-symbol profiling, alignment and comparison for V3.

Why this module exists
----------------------
A single-symbol V3 run answers "does this model work on this market".  A
multi-asset run has to answer a harder question *first* - "is the comparison
between these markets even fair?" - and that is a data question, not a
modelling one.  On this repository's own cache the answer is no, by default,
and for three separate reasons that are all invisible in a single-symbol run:

* **Feature availability is per symbol.**  The Crypto Fear & Greed cache exists
  for one symbol only, so the ``sentiment`` bucket is available for that symbol
  and genuinely absent for the rest.  A symbol can also have a source whose
  file *exists* but stops years before the price history, which does not look
  like a missing file at all - it looks like a source.  Nothing here hard-codes
  which symbol has which problem: a bucket counts as available only when the
  builder actually produced features for it, and a short history shows up in the
  drop report and in the per-horizon labelled counts, whichever cause it has.
* **History length is per symbol, and a horizon is a wall-clock window.**  Two
  symbols with the same 5,000 hourly rows can support different ladders, and a
  180d label on a three-year history is not a small sample, it is *no* sample.
* **Nothing can be padded into existence.**  A horizon a symbol cannot support
  is skipped and recorded.  Forward-filling a label or a missing sentiment
  series would manufacture the very comparability the reader thinks they are
  being shown.

Like-for-like, or the numbers mean nothing
------------------------------------------
Comparing a 107-feature BTC model against a 72-feature ETH model does not tell
you which market is easier to predict; it tells you which feature set was
larger.  So every comparison this module produces is restricted to the
**intersection** of the participating symbols' feature columns - not their
bucket names, their columns - and that shared list is returned on the result as
:attr:`CrossSymbolResult.shared_features` so a report can print it.  Where the
buckets differ, the extra features are dropped for *every* symbol, including the
one that had them.  :func:`align_panel` applies the same rule and additionally
restricts to the intersection of the *available buckets*, so a caller building
one modelling frame across symbols cannot accidentally reintroduce the
asymmetry.

The remaining asymmetry is the target itself, and it is not fixable by
construction: a quiet market has a smaller return variance, so its RMSE is
smaller for reasons that have nothing to do with the model.  That is why
:func:`compare_across_symbols` reports ``target_std`` and
``rmse_skill_vs_zero`` (skill against the zero-return benchmark) next to the raw
error, and why ``rank_within_horizon`` is a ranking of raw RMSE that has to be
read beside them - a symbol at rank 1 may simply be the quiet one.

Reporting, never crashing
-------------------------
A poor symbol is a result, not an exception.  :func:`profile_symbol` returns a
profile with ``n_rows == 0`` and the reason in its notes;
:func:`build_cross_symbol_panel` and :func:`run_multiasset` record the reason in
``skipped`` and continue with the other symbols, and every ``(symbol, horizon)``
unit in a run is wrapped independently so one failing market cannot take the
run down with it.  A malformed *feature selection* is still a caller error and
still raises: a typo must not be filed as "the data was bad".

Cost
----
:func:`run_multiasset` defaults to the two baselines and ridge because a
cross-asset random-forest + xgboost run over eight horizons is hours of CPU.
It only runs horizons at least ``DEFAULT_MIN_SYMBOLS`` symbols support, and only
for symbols with enough rows, which is the difference between minutes and a day.
Nothing here writes to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import get_logger
from src.v3.dataset import (
    DataUnavailableError,
    FeatureSelectionError,
    V3Dataset,
    V3DatasetError,
    bucket_of_feature,
    build_v3_dataset,
    load_horizons,
    resolve_feature_selection,
)
from src.v3.horizons import Horizon, parse_horizon
from src.v3.walkforward import HorizonResult, V3RunResult, run_walk_forward

logger = get_logger("v3.multiasset")

#: The sibling bucket registry declares the canonical bucket order.  It is
#: imported the same guarded way ``src.v3.dataset`` does it, so this module keeps
#: working (on V2 group names as the taxonomy) if the sibling is absent.
try:  # pragma: no cover - depends on a sibling module
    from src.v3.buckets import BUCKET_NAMES as _BUCKET_NAMES
except ImportError:  # pragma: no cover - depends on a sibling module
    _BUCKET_NAMES: tuple[str, ...] = ()

#: Rows a horizon needs on one symbol before it is worth modelling.  Below this
#: the purged walk-forward splitter cannot produce a fold, so running it would
#: produce an error rather than a weak result.
DEFAULT_MIN_ROWS = 200

#: How many symbols must support a horizon before a cross-symbol comparison of
#: it means anything.  One symbol is not a comparison.
DEFAULT_MIN_SYMBOLS = 2

#: Deliberately small model set.  The two baselines are the controls every other
#: model has to beat and ridge is the cheapest model that can beat them; the tree
#: ensembles belong to the single-symbol harness.
DEFAULT_MODEL_NAMES: tuple[str, ...] = ("zero", "mean", "ridge")

#: Columns of :attr:`CrossSymbolResult.comparisons`, in report order.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "symbol",
    "horizon",
    "horizon_days",
    "rmse",
    "mae",
    "spearman_ic",
    "direction_accuracy",
    "n_predictions",
    "n_models",
    "n_features",
    "best_model",
    "target_std",
    "zero_rmse",
    "rmse_skill_vs_zero",
    "rank_within_horizon",
)

#: Columns of :attr:`CrossSymbolResult.horizon_support`, in report order.  One
#: row per (symbol, horizon), including the symbols that could not be profiled
#: at all - a symbol missing from this table would be a symbol nobody noticed.
HORIZON_SUPPORT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "horizon",
    "horizon_days",
    "n_labelled",
    "supported",
    "evaluated",
    "n_features",
    "n_buckets",
    "available_buckets",
    "skipped_reason",
)

#: Long-format panel columns, before the shared feature block.
_PANEL_LEADING_COLUMNS: tuple[str, ...] = ("symbol", "timestamp")
_PANEL_TRAILING_COLUMNS: tuple[str, ...] = ("target", "horizon")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _normalise_symbols(symbols: Sequence[str]) -> list[str]:
    """Upper-case, de-duplicate and reject an empty universe, order preserved."""
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a sequence of names, not a bare string")
    seen: dict[str, None] = {}
    for symbol in symbols:
        name = str(symbol).strip().upper()
        if not name:
            raise ValueError("A symbol must be a non-empty name such as 'BTCUSDT'")
        seen.setdefault(name, None)
    if not seen:
        raise ValueError("symbols=[] selects nothing; at least one symbol is required")
    return list(seen)


def _order_buckets(buckets: Iterable[str]) -> tuple[str, ...]:
    """Canonical bucket order, with anything the registry does not know last."""
    rank = {name: position for position, name in enumerate(_BUCKET_NAMES)}
    return tuple(sorted(set(buckets), key=lambda name: (rank.get(name, len(rank)), name)))


def _label_sort_key(label: str) -> tuple[pd.Timedelta, str]:
    """Order horizon labels by duration, not alphabetically (``1d`` < ``7d``)."""
    try:
        return parse_horizon(str(label)), str(label)
    except Exception:  # pragma: no cover - a label parse_horizon rejects
        return pd.Timedelta.max, str(label)


def _float(value: Any) -> float:
    """Coerce a metrics-table cell to a float, mapping anything unusable to NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _unsupported_reason(label: str, n_labelled: int, min_rows: int) -> str:
    """Why a symbol cannot be modelled on a horizon."""
    if n_labelled <= 0:
        return (
            f"no labels for {label}: the history is shorter than the {label} target "
            "window, so there is nothing to fit or score"
        )
    return f"{n_labelled} labelled rows for {label} is below min_rows={min_rows}"


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolProfile:
    """What one symbol can actually be asked to do, and why.

    Attributes
    ----------
    symbol:
        Upper-case symbol this profile describes.
    available_buckets:
        V3 buckets the builder really produced features for, in canonical order.
        A bucket whose source is absent, or whose features no builder emitted,
        is simply not here - which is how a per-symbol feature asymmetry becomes
        visible instead of being papered over.
    missing_buckets:
        Requested buckets that did *not* survive, same order.  A non-empty tuple
        is the reason a symbol's feature count is lower than another's.
    n_rows / n_features:
        Size of the frame the walk-forward harness would see.
    date_start / date_end:
        Span of that frame, ISO strings, or ``None`` when nothing was built.
    labelled_counts:
        ``horizon label -> rows with a real target``.  The tail of every horizon
        is NaN by design, so this is the number a support decision must use.
    notes:
        Human-readable findings: uncached sources, how much history the complete
        -rows-only filter cost, features that were requested but not built,
        horizons with no labels.  These are the sentences a report should print.
    """

    symbol: str
    available_buckets: tuple[str, ...]
    missing_buckets: tuple[str, ...]
    n_rows: int
    n_features: int
    date_start: str | None
    date_end: str | None
    labelled_counts: dict[str, int]
    notes: tuple[str, ...]

    def supports(self, horizon_label: str, min_rows: int = DEFAULT_MIN_ROWS) -> bool:
        """Whether this symbol has at least ``min_rows`` real labels for a horizon.

        This is the only question a caller should ask before running a horizon,
        and it is deliberately cheap: it is the *count* of labels, not the
        length of the file.  A symbol can have 40,000 rows and still not support
        a 180d horizon after the target window and the split geometry are taken
        off the top.
        """
        return int(self.labelled_counts.get(str(horizon_label), 0)) >= int(min_rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "available_buckets": list(self.available_buckets),
            "missing_buckets": list(self.missing_buckets),
            "n_rows": int(self.n_rows),
            "n_features": int(self.n_features),
            "date_start": self.date_start,
            "date_end": self.date_end,
            "labelled_counts": {str(k): int(v) for k, v in self.labelled_counts.items()},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CrossSymbolResult:
    """The outcome of profiling or running several symbols.

    Attributes
    ----------
    profiles:
        One profile per symbol that could be inspected at all.
    skipped:
        ``symbol -> reason`` for every symbol that produced no comparison rows,
        whether because it could not be built, because it supports no
        comparable horizon, or because every horizon it was run on failed.  A
        symbol that is silently absent is indistinguishable from a symbol that
        was never asked about, so the reason is mandatory.
    comparisons:
        One row per (symbol, horizon) that was actually evaluated - see
        :data:`COMPARISON_COLUMNS`.  Empty when nothing was run; availability
        lives in ``horizon_support`` instead.
    horizon_support:
        One row per (symbol, horizon) including the skipped ones - see
        :data:`HORIZON_SUPPORT_COLUMNS`.  ``skipped_reason`` is ``None`` for a
        pair that was evaluated.
    shared_features:
        The feature columns every compared symbol was restricted to.  Additive
        to the four core fields, and load-bearing: without it a reader cannot
        check that the symbols really were fitted on the same inputs.
    """

    profiles: tuple[SymbolProfile, ...]
    skipped: dict[str, str]
    comparisons: pd.DataFrame
    horizon_support: pd.DataFrame
    shared_features: tuple[str, ...] = ()

    def supported_horizons(self, min_symbols: int = DEFAULT_MIN_SYMBOLS) -> list[str]:
        """Horizons at least ``min_symbols`` profiled symbols can support.

        Read from ``horizon_support`` when that table exists, so the answer
        reflects the ``min_rows`` this result was built with rather than a
        default that may disagree with it.
        """
        table = self.horizon_support
        if table is not None and len(table) and "supported" in table.columns:
            counts = table.loc[table["supported"].astype(bool)].groupby("horizon").size()
            return [
                label
                for label in sorted(counts.index, key=_label_sort_key)
                if int(counts[label]) >= int(min_symbols)
            ]
        return common_horizons(self.profiles, min_symbols=min_symbols)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_symbols": len(self.profiles),
            "symbols": [profile.symbol for profile in self.profiles],
            "skipped": dict(self.skipped),
            "supported_horizons": self.supported_horizons(),
            "shared_features": list(self.shared_features),
            "profiles": [profile.to_dict() for profile in self.profiles],
            "horizon_support": self.horizon_support.to_dict("records"),
            "comparisons": self.comparisons.to_dict("records"),
        }


# ---------------------------------------------------------------------------
# empty tables
# ---------------------------------------------------------------------------

def _empty_comparison_frame() -> pd.DataFrame:
    """The documented comparison columns with no rows, so a schema is always visible."""
    return pd.DataFrame({column: pd.Series(dtype=_column_dtype(column)) for column in COMPARISON_COLUMNS})


def _empty_support_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=_column_dtype(column)) for column in HORIZON_SUPPORT_COLUMNS})


def _column_dtype(column: str) -> str:
    """Dtype for an empty table's column, so ``to_dict`` stays JSON-friendly."""
    if column in {"symbol", "horizon", "available_buckets", "skipped_reason", "best_model"}:
        return "object"
    if column in {"n_predictions", "n_models", "n_features", "n_buckets", "n_labelled",
                  "rank_within_horizon"}:
        return "int64"
    if column in {"supported", "evaluated"}:
        return "bool"
    return "float64"


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------

def _requested_buckets(config: Config, enabled_features: Any) -> tuple[str, ...]:
    """The buckets this configuration asks for, before any data is consulted.

    Falls back to the V2 feature groups when the V3 bucket registry is not
    importable, because ``bucket_of_feature`` falls back the same way and the two
    taxonomies have to agree or ``missing_buckets`` would be nonsense.
    """
    selection = resolve_feature_selection(config, enabled_features)
    return selection.buckets or selection.groups


def _retention_note(dataset: V3Dataset) -> str | None:
    """How much of the raw history survived, and what removed the rest.

    This is the sentence that makes a *truncated* source visible.  A futures
    cache that stops in 2022 does not raise and does not look missing: under
    complete-rows-only it simply deletes every row after 2022, and the resulting
    2,577-row frame is otherwise indistinguishable from a small, well-behaved
    dataset.  Nothing here knows which symbol that is; the arithmetic does.
    """
    dropped = {reason: int(count) for reason, count in dataset.drop_report.items() if int(count) > 0}
    n_input = int(len(dataset)) + int(sum(dropped.values()))
    if n_input <= 0 or len(dataset) >= n_input:
        return None
    worst = max(dropped, key=lambda reason: dropped[reason])
    note = (
        f"retained {len(dataset)} of {n_input} raw rows ({100.0 * len(dataset) / n_input:.1f}%); "
        f"largest drop was {worst}={dropped[worst]}"
    )
    if worst in {"missing_feature", "non_finite_feature"}:
        note += (
            " - a complete-rows-only filter removed rows because at least one selected feature was "
            "absent, which is how a cached external source that stops before the end of the price "
            "history presents itself: check missing_sources and the per-bucket notes"
        )
    return note


def _profile_from_dataset(dataset: V3Dataset, requested: Sequence[str]) -> SymbolProfile:
    """Turn a built dataset into a profile, reading only what the build recorded."""
    present = _order_buckets(bucket_of_feature(name) for name in dataset.feature_columns)
    missing = tuple(bucket for bucket in _order_buckets(requested) if bucket not in present)

    labelled = {horizon.label: int(dataset.n_labelled(horizon)) for horizon in dataset.horizons}
    notes: list[str] = []

    if dataset.missing_sources:
        dropped = (
            f", which is why {list(missing)} are unavailable" if missing else ""
        )
        notes.append(
            f"no cached source for {list(dataset.missing_sources)}; every feature group that needs "
            f"one was dropped{dropped}"
        )
    retention = _retention_note(dataset)
    if retention is not None:
        notes.append(retention)
    if dataset.max_rows is not None:
        notes.append(
            f"max_rows={dataset.max_rows} truncated the history before labelling, so the longest "
            "horizons lost up to their own window of otherwise-available labels"
        )
    if missing:
        notes.append(f"bucket(s) {list(missing)} were requested and are not available for this symbol")
    starved = [label for label, count in labelled.items() if count <= 0]
    if starved:
        notes.append(
            f"no labels at all for horizon(s) {starved}: the surviving history is shorter than "
            "their target window"
        )

    index = dataset.index
    return SymbolProfile(
        symbol=dataset.symbol,
        available_buckets=present,
        missing_buckets=missing,
        n_rows=int(len(dataset)),
        n_features=int(len(dataset.feature_columns)),
        date_start=str(index.min()) if len(index) else None,
        date_end=str(index.max()) if len(index) else None,
        labelled_counts=labelled,
        notes=tuple(notes),
    )


def _failed_profile(symbol: str, exc: Exception, requested: Sequence[str], labels: Sequence[str]) -> SymbolProfile:
    """A profile for a symbol whose data could not be read at all.

    Zero rows and zero features are the honest answer, and the exception text is
    the note: "this symbol was asked about and it could not be used" must not look
    the same as "this symbol was never asked about".
    """
    return SymbolProfile(
        symbol=symbol,
        available_buckets=(),
        missing_buckets=_order_buckets(requested),
        n_rows=0,
        n_features=0,
        date_start=None,
        date_end=None,
        labelled_counts={label: 0 for label in labels},
        notes=(f"{type(exc).__name__}: {exc}",),
    )


def _strict_profile(
    config: Config,
    symbol: str,
    horizons: Sequence[str] | Sequence[Horizon] | None,
    enabled_features: Any,
    max_rows: int | None,
) -> tuple[SymbolProfile, V3Dataset]:
    """Build a symbol's dataset and profile it, letting data failures propagate.

    The panel and the run need the *dataset* as well as the profile - the run
    hands it to the walk-forward harness - and they need the failure to arrive as
    an exception so it can be recorded against that symbol and the run continues.
    """
    dataset = build_v3_dataset(
        config, symbol, horizons=horizons, enabled_features=enabled_features, max_rows=max_rows
    )
    return _profile_from_dataset(dataset, _requested_buckets(config, enabled_features)), dataset


def profile_symbol(
    config: Config,
    symbol: str,
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    enabled_features: Any = None,
    max_rows: int | None = None,
) -> SymbolProfile:
    """Report what one symbol can support, instead of failing when it cannot.

    Parameters
    ----------
    config:
        A config carrying a ``v3:`` block.
    symbol:
        Symbol to inspect, e.g. ``"BTCUSDT"``.
    horizons, enabled_features, max_rows:
        Passed straight through to :func:`src.v3.dataset.build_v3_dataset`.

    Returns
    -------
    SymbolProfile
        Always.  A symbol whose klines are absent comes back with
        ``n_rows == 0`` and the reason in ``notes``, because "this market cannot
        be modelled" is a finding about the universe and not a bug in the caller.

    Notes
    -----
    A missing *source* (funding, futures, sentiment) is not an error at all: the
    builder drops the feature groups that need it and the profile says which
    buckets that cost.  A malformed *feature selection* is a caller error and
    still raises - a typo must not be filed as "the data was bad".
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol must be a non-empty string such as 'BTCUSDT'")

    try:
        labels = [horizon.label for horizon in load_horizons(config, horizons)]
    except V3DatasetError:
        # An unparseable ladder is a caller error; the build will say so in a
        # moment with a better message than a bare label list.
        labels = []

    try:
        profile, _ = _strict_profile(config, symbol, horizons, enabled_features, max_rows)
    except FeatureSelectionError:
        raise
    except (DataUnavailableError, V3DatasetError) as exc:
        logger.warning("profile_symbol %s: %s", symbol, exc)
        try:
            requested = _requested_buckets(config, enabled_features)
        except FeatureSelectionError:
            requested = ()
        return _failed_profile(symbol, exc, requested, labels)

    logger.info(
        "profile %s: %d rows, %d features, buckets %s, labelled %s",
        profile.symbol,
        profile.n_rows,
        profile.n_features,
        list(profile.available_buckets),
        profile.labelled_counts,
    )
    return profile


# ---------------------------------------------------------------------------
# shared feature selection
# ---------------------------------------------------------------------------

def _shared_features(datasets: Mapping[str, V3Dataset], symbols: Sequence[str]) -> tuple[str, ...]:
    """Columns every listed symbol actually has, in the first symbol's order.

    The intersection is taken over *columns*, not over bucket names, because a
    bucket is only a label: two symbols can both claim ``technical`` and still
    disagree on which of its features their builders emitted.  The order follows
    the first symbol so the panel and every report are byte-identical run to run.
    """
    if not symbols:
        return ()
    first = datasets[symbols[0]].feature_columns
    others = [set(datasets[symbol].feature_columns) for symbol in symbols[1:]]
    return tuple(name for name in first if all(name in other for other in others))


def _shared_buckets(datasets: Mapping[str, V3Dataset], symbols: Sequence[str], features: Sequence[str]) -> tuple[str, ...]:
    """The buckets the shared feature block actually spans."""
    if not symbols:
        return ()
    per_symbol = [
        {bucket_of_feature(name) for name in datasets[symbol].feature_columns} for symbol in symbols
    ]
    common = set.intersection(*per_symbol) if per_symbol else set()
    return _order_buckets(bucket for bucket in common if bucket in {bucket_of_feature(f) for f in features})


# ---------------------------------------------------------------------------
# availability tables
# ---------------------------------------------------------------------------

def _support_table(
    profiles: Sequence[SymbolProfile],
    ladder: Sequence[Horizon],
    min_rows: int,
    evaluated: Mapping[str, set[str]] | None = None,
    failures: Mapping[tuple[str, str], str] | None = None,
    unprofiled: Mapping[str, str] | None = None,
    common: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The per-(symbol, horizon) availability table, skips included.

    ``evaluated`` is ``None`` for an availability-only report: nothing was run and
    no reason is invented for that.  Otherwise it is the set of horizon labels
    each symbol was actually run on, and every other pair carries the reason it
    was not - which is the only place a reader can find out that a symbol was
    dropped for a *particular* horizon rather than skipped outright.
    """
    failures = failures or {}
    common_set = None if common is None else set(common)
    rows: list[dict[str, Any]] = []

    for symbol, reason in (unprofiled or {}).items():
        for horizon in ladder:
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon.label,
                    "horizon_days": float(horizon.days),
                    "n_labelled": 0,
                    "supported": False,
                    "evaluated": False,
                    "n_features": 0,
                    "n_buckets": 0,
                    "available_buckets": "",
                    "skipped_reason": reason,
                }
            )

    for profile in profiles:
        done = set() if evaluated is None else set(evaluated.get(profile.symbol, ()))
        for horizon in ladder:
            label = horizon.label
            n_labelled = int(profile.labelled_counts.get(label, 0))
            supported = profile.supports(label, min_rows)
            failure = failures.get((profile.symbol, label))
            if label in done:
                reason: str | None = None
            elif failure is not None:
                reason = failure
            elif not supported:
                reason = _unsupported_reason(label, n_labelled, min_rows)
            elif common_set is not None and label not in common_set:
                reason = f"not compared: fewer than {DEFAULT_MIN_SYMBOLS} symbols support {label}"
            else:
                reason = None
            rows.append(
                {
                    "symbol": profile.symbol,
                    "horizon": label,
                    "horizon_days": float(horizon.days),
                    "n_labelled": n_labelled,
                    "supported": bool(supported),
                    "evaluated": bool(label in done),
                    "n_features": int(profile.n_features),
                    "n_buckets": len(profile.available_buckets),
                    "available_buckets": ", ".join(profile.available_buckets),
                    "skipped_reason": reason,
                }
            )

    table = pd.DataFrame(rows, columns=list(HORIZON_SUPPORT_COLUMNS))
    return table.sort_values(["horizon", "symbol"], kind="mergesort", ignore_index=True)


def common_horizons(
    profiles: Sequence[SymbolProfile],
    min_symbols: int = DEFAULT_MIN_SYMBOLS,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> list[str]:
    """Horizon labels at least ``min_symbols`` profiles support, shortest first.

    ``min_rows`` defaults to the same bar :meth:`SymbolProfile.supports` uses, so
    a caller that does not pass one gets the documented behaviour.  It is a
    parameter because a run configured with a different bar must ask the same
    question with *its* bar, not with a default that quietly disagrees with the
    decision being recorded.
    """
    profiles = list(profiles)
    labels: dict[str, str] = {}
    for profile in profiles:
        for label in profile.labelled_counts:
            labels.setdefault(str(label), label)

    supported = [
        label
        for label in labels.values()
        if sum(1 for profile in profiles if profile.supports(label, min_rows)) >= int(min_symbols)
    ]
    return sorted(supported, key=_label_sort_key)


# ---------------------------------------------------------------------------
# the cross-symbol panel
# ---------------------------------------------------------------------------

def _collect_profiles(
    config: Config,
    symbols: Sequence[str],
    horizons: Sequence[str] | Sequence[Horizon] | None,
    enabled_features: Any,
    max_rows: int | None,
) -> tuple[list[SymbolProfile], dict[str, V3Dataset], dict[str, str]]:
    """Profile every symbol, isolating per-symbol failures.

    A symbol that cannot be built is recorded and skipped, never allowed to
    abort the universe: one stale cache in the middle of a five-symbol panel must
    not cost the other four.
    """
    profiles: list[SymbolProfile] = []
    datasets: dict[str, V3Dataset] = {}
    skipped: dict[str, str] = {}
    for symbol in _normalise_symbols(symbols):
        try:
            profile, dataset = _strict_profile(config, symbol, horizons, enabled_features, max_rows)
        except (DataUnavailableError, FeatureSelectionError, V3DatasetError) as exc:
            reason = f"could not be profiled: {type(exc).__name__}: {exc}"
            logger.warning("multiasset: %s %s", symbol, reason)
            skipped[symbol] = reason
            continue
        profiles.append(profile)
        datasets[symbol] = dataset
    return profiles, datasets, skipped


def _not_compared_reason(
    profile: SymbolProfile, ladder: Sequence[Horizon], common: Sequence[str], min_rows: int
) -> str | None:
    """Why a profiled symbol is still not in the comparison, or ``None`` if it is.

    Two different reasons are kept apart on purpose.  "Too few rows" says the
    data cannot answer the question; "no partner" says the data is fine but
    nothing else in the universe supports the same horizon, so a single-symbol
    result would be reported as a cross-symbol one.  Collapsing them would hide
    which one it is, and the fix for each is different.
    """
    counts = {horizon.label: int(profile.labelled_counts.get(horizon.label, 0)) for horizon in ladder}
    if any(count >= min_rows for label, count in counts.items() if label in set(common)):
        return None
    if not any(count >= min_rows for count in counts.values()):
        return f"not eligible: no horizon reaches min_rows={min_rows}; labelled rows are {counts}"
    supported = [label for label, count in counts.items() if count >= min_rows]
    return (
        f"not compared: supports {supported} but no horizon is supported by "
        f"{DEFAULT_MIN_SYMBOLS} symbols, so there is no cross-symbol comparison to make"
    )


def build_cross_symbol_panel(
    config: Config,
    symbols: Sequence[str],
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    enabled_features: Any = None,
    max_rows: int | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> CrossSymbolResult:
    """Profile a universe of symbols and report what is comparable.

    This is the cheap half of a cross-asset study: it reads every symbol once,
    records which buckets survived, which horizons each symbol can support, and
    why the others cannot - without fitting a single model.  Run it first; its
    ``skipped`` map is the honest list of "symbols I asked about and could not
    use", and ``horizon_support`` is the table that says which comparisons the
    remaining data can actually support.

    A symbol that cannot be profiled is recorded in ``skipped`` and the rest are
    still profiled.  A profiled symbol that ends up with no comparable horizon is
    also recorded - with its per-horizon counts if it is short of rows, with the
    horizons it does support if it simply has no partner - so it is visibly
    rejected rather than quietly absent.  ``comparisons`` is empty here by
    construction: it holds prediction metrics, and nothing was predicted.
    """
    ladder = load_horizons(config, horizons)
    profiles, _datasets, build_failures = _collect_profiles(
        config, symbols, horizons, enabled_features, max_rows
    )

    common = common_horizons(profiles, min_symbols=DEFAULT_MIN_SYMBOLS, min_rows=min_rows)
    skipped = dict(build_failures)
    for profile in profiles:
        reason = _not_compared_reason(profile, ladder, common, min_rows)
        if reason is not None:
            logger.warning("multiasset: %s %s", profile.symbol, reason)
            skipped.setdefault(profile.symbol, reason)

    support = _support_table(
        profiles, ladder, min_rows=min_rows, unprofiled=build_failures, common=common
    )
    result = CrossSymbolResult(
        profiles=tuple(profiles),
        skipped=skipped,
        comparisons=_empty_comparison_frame(),
        horizon_support=support,
    )
    logger.info(
        "cross-symbol panel: %d profiled, %d skipped, horizons supported by >= %d symbols: %s",
        len(profiles),
        len(skipped),
        DEFAULT_MIN_SYMBOLS,
        result.supported_horizons(),
    )
    return result


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _as_horizon_results(value: object) -> list[Any]:
    """Accept a run, a single result, or a sequence of results for one symbol."""
    if isinstance(value, V3RunResult):
        return list(value.results)
    if isinstance(value, HorizonResult):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, pd.DataFrame)):
        return [item for item in value if hasattr(item, "pooled_metrics")]
    return []


def _target_std(result: Any) -> float:
    """Dispersion of the realised target, the scale a cross-symbol RMSE divides by."""
    predictions = getattr(result, "predictions", None)
    if not isinstance(predictions, pd.DataFrame) or "target" not in predictions.columns:
        return float("nan")
    values = predictions["target"].to_numpy(dtype="float64", copy=False)
    if values.size == 0:
        return float("nan")
    return _float(np.nanstd(values))


def _comparison_row(result: Any, fallback_symbol: str) -> dict[str, Any] | None:
    """One ``(symbol, horizon)`` row, or ``None`` when the result has no metrics."""
    pooled = getattr(result, "pooled_metrics", None)
    if not isinstance(pooled, pd.DataFrame) or pooled.empty or "rmse" not in pooled.columns:
        logger.warning(
            "compare_across_symbols: %s/%s has no usable pooled metrics; row omitted",
            fallback_symbol,
            getattr(result, "horizon", "?"),
        )
        return None

    best_model = str(pooled.sort_values("rmse").index[0])
    best = pooled.loc[best_model]
    zero_rmse = _float(pooled.loc["zero", "rmse"]) if "zero" in pooled.index else float("nan")
    rmse = _float(best.get("rmse"))
    skill = 1.0 - rmse / zero_rmse if np.isfinite(zero_rmse) and zero_rmse > 0 else float("nan")
    predictions = getattr(result, "predictions", None)
    features = getattr(result, "feature_columns", []) or []

    return {
        "symbol": str(getattr(result, "symbol", "") or fallback_symbol),
        "horizon": str(getattr(result, "horizon", "")),
        "horizon_days": _float(getattr(result, "horizon_days", float("nan"))),
        "rmse": rmse,
        "mae": _float(best.get("mae")),
        "spearman_ic": _float(best.get("spearman_ic")),
        "direction_accuracy": _float(best.get("direction_accuracy")),
        "n_predictions": int(len(predictions)) if isinstance(predictions, pd.DataFrame) else 0,
        "n_models": int(len(pooled.index)),
        "n_features": int(len(features)),
        "best_model": best_model,
        "target_std": _target_std(result),
        "zero_rmse": zero_rmse,
        "rmse_skill_vs_zero": _float(skill),
        "rank_within_horizon": np.nan,
    }


def compare_across_symbols(results: Mapping[str, object]) -> pd.DataFrame:
    """One row per (symbol, horizon), ranked within its horizon.

    Parameters
    ----------
    results:
        ``symbol -> V3RunResult``, ``symbol -> HorizonResult``, or
        ``symbol -> sequence of HorizonResult``.  The symbol key is only a
        fallback: a result's own ``.symbol`` wins when it has one.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`COMPARISON_COLUMNS`, sorted by horizon then rank.  The
        metrics are the *best model's* pooled out-of-sample numbers, selected on
        RMSE exactly as :meth:`src.v3.walkforward.HorizonResult.best_model`
        does, so this table and the single-symbol report cannot disagree.

    Notes
    -----
    ``rank_within_horizon`` is a dense 1..k ranking of raw RMSE, and it is
    deliberately *not* the whole story.  A quiet symbol has a small return
    variance, so a lower RMSE can mean "easier target", not "better model".  Read
    it beside ``target_std`` (how much there was to predict) and
    ``rmse_skill_vs_zero`` (the error relative to predicting flat, which is the
    only scale-free version of the comparison).  Those two columns are what turn
    the ranking into an answer.
    """
    rows: list[dict[str, Any]] = []
    for symbol, value in results.items():
        for result in _as_horizon_results(value):
            row = _comparison_row(result, str(symbol))
            if row is not None:
                rows.append(row)

    if not rows:
        return _empty_comparison_frame()

    frame = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    # `dense` on purpose: two symbols whose errors are indistinguishable share a
    # rank and the next one takes the following integer, instead of the gap method
    # skipping a number that no symbol earned.
    frame["rank_within_horizon"] = frame.groupby("horizon")["rmse"].rank(
        method="dense", ascending=True, na_option="top"
    )
    return frame.sort_values(
        ["horizon", "rank_within_horizon", "symbol"],
        kind="mergesort",
        na_position="last",
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_multiasset(
    config: Config,
    symbols: Sequence[str],
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    enabled_features: Any = None,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    n_splits: int = 3,
    test_fraction: float = 0.08,
    validation_fraction: float = 0.08,
    min_rows: int = DEFAULT_MIN_ROWS,
    seed: int = 42,
) -> CrossSymbolResult:
    """Walk forward several symbols on the horizons they can actually support.

    The run is deliberately narrow in three ways, and each one is a correctness
    choice rather than a performance one:

    1. **Only comparable horizons run.**  A horizon fewer than
       ``DEFAULT_MIN_SYMBOLS`` symbols support is not a cross-symbol result, so
       it is not computed.  A single symbol supporting 180d does not buy a 180d
       *comparison*.
    2. **Only comparable features run.**  Every participating symbol is
       restricted to the intersection of their feature columns, so all of them
       were fitted on the same inputs; the shared list comes back on the result
       as ``shared_features``.  Otherwise a symbol with sentiment and one
       without would be compared on different models and the difference would be
       read as skill.
    3. **Every (symbol, horizon) is isolated.**  Each pair is attempted inside its
       own ``try``; a failure is recorded against that pair in
       ``horizon_support.skipped_reason`` and the run continues.  One broken
       symbol therefore costs one row of the comparison table, never the run.

    Parameters
    ----------
    model_names:
        Defaults to the two baselines plus ridge.  A full cross-asset
        random-forest + xgboost sweep over eight horizons is hours of CPU; the
        tree ensembles belong to the single-symbol harness.
    min_rows:
        Labelled rows a symbol needs on a horizon before it is run at all.
    seed, n_splits, test_fraction, validation_fraction:
        Forwarded to :func:`src.v3.walkforward.run_walk_forward`; the purge and
        embargo stay at their ``"horizon"`` defaults, which is the only setting
        that is correct for a forward-return label.

    Notes
    -----
    The full history of every symbol is used - there is no ``max_rows`` here,
    because a run that silently trains on a truncated window is exactly the
    comparison this module exists to prevent.  Use
    :func:`build_cross_symbol_panel` (which does take ``max_rows``) to look at
    availability cheaply first.
    """
    ladder = load_horizons(config, horizons)
    profiles, datasets, build_failures = _collect_profiles(
        config, symbols, horizons, enabled_features, max_rows=None
    )

    common = common_horizons(profiles, min_symbols=DEFAULT_MIN_SYMBOLS, min_rows=min_rows)
    if not common:
        logger.warning(
            "multiasset run: no horizon is supported by >= %d symbols (min_rows=%d); nothing was "
            "run. Availability is still reported in horizon_support.",
            DEFAULT_MIN_SYMBOLS,
            min_rows,
        )

    participants = [
        profile.symbol
        for profile in profiles
        if any(profile.supports(label, min_rows) for label in common)
    ]
    skipped = dict(build_failures)
    for profile in profiles:
        reason = _not_compared_reason(profile, ladder, common, min_rows)
        if reason is not None:
            logger.warning("multiasset run: %s %s", profile.symbol, reason)
            skipped.setdefault(profile.symbol, reason)

    shared = _shared_features(datasets, participants) if participants else ()
    if participants:
        if not shared:
            reason = "no feature is common to every participating symbol, so the symbols cannot be compared"
            for symbol in participants:
                skipped.setdefault(symbol, reason)
            participants = []
        else:
            logger.info(
                "multiasset run: restricting %d symbol(s) to the %d features they all have, "
                "spanning bucket(s) %s",
                len(participants),
                len(shared),
                list(_shared_buckets(datasets, participants, shared)),
            )

    runs: dict[str, list[Any]] = {}
    failures: dict[tuple[str, str], str] = {}
    supported = {profile.symbol: profile for profile in profiles}
    for symbol in participants:
        # The frame already carries every built column, so narrowing the declared
        # feature list is enough to make this symbol like-for-like; rebuilding
        # would cost minutes and change nothing.
        restricted = replace(datasets[symbol], feature_columns=list(shared))
        for horizon in ladder:
            label = horizon.label
            # A horizon can be comparable and still be out of reach for this one
            # symbol - 30d needs 30 days of history *after* the warm-up. Running it
            # anyway would hand the harness a frame with no labels for that horizon,
            # so the symbol is left out of the horizon and told why instead.
            if label not in common or not supported[symbol].supports(label, min_rows):
                continue
            try:
                run = run_walk_forward(
                    restricted,
                    [horizon],
                    model_names=model_names,
                    seed=seed,
                    n_splits=n_splits,
                    test_fraction=test_fraction,
                    validation_fraction=validation_fraction,
                )
            except Exception as exc:  # one market must not end the run
                reason = f"walk-forward failed: {type(exc).__name__}: {exc}"
                failures[(symbol, label)] = reason
                logger.warning("multiasset run: %s/%s %s", symbol, label, reason)
                continue
            runs.setdefault(symbol, []).extend(run.results)

    comparisons = compare_across_symbols(runs)
    evaluated = {symbol: {row.horizon for row in results} for symbol, results in runs.items()}

    for symbol in participants:
        if symbol in runs:
            continue
        attempted = "; ".join(
            f"{label}: {failures[(symbol, label)]}"
            for label in common
            if (symbol, label) in failures
        )
        skipped.setdefault(
            symbol,
            "walk-forward produced no comparison rows"
            + (f" ({attempted})" if attempted else " (no horizon was attempted)"),
        )

    support = _support_table(
        profiles,
        ladder,
        min_rows=min_rows,
        evaluated=evaluated,
        failures=failures,
        unprofiled=build_failures,
        common=common,
    )
    logger.info(
        "multiasset run: %d symbol(s) evaluated, %d skipped, %d comparison row(s)",
        len(runs),
        len(skipped),
        len(comparisons),
    )
    return CrossSymbolResult(
        profiles=tuple(profiles),
        skipped=skipped,
        comparisons=comparisons,
        horizon_support=support,
        shared_features=shared,
    )


# ---------------------------------------------------------------------------
# like-for-like panel
# ---------------------------------------------------------------------------

def align_panel(
    config: Config,
    symbols: Sequence[str],
    horizons: Sequence[str] | Sequence[Horizon],
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Stack several symbols into one long panel on a common feature set.

    Like :func:`run_multiasset`, the panel is restricted to the features every
    symbol has, so a single model fitted on the stack sees the same columns for
    every row.  It is additionally restricted to the intersection of the
    *available buckets*: a bucket one symbol does not have is not narrowed to
    the surviving features, it is dropped outright, because a model that saw a
    sentiment column for one symbol and nothing for another has learned the
    symbol identity rather than the market.

    Parameters
    ----------
    horizons:
        Labels or :class:`~src.v3.horizons.Horizon` objects.  Required, not
        defaulted: a panel over every configured horizon is a much bigger frame
        than anyone wants by accident.

    Returns
    -------
    pandas.DataFrame
        Long format, one row per (symbol, horizon, timestamp) with **no**
        unlabelled rows - the NaN tail is dropped per horizon rather than padded
        or back-filled, which is what keeps a symbol from appearing to have data
        it does not have.  Columns ``symbol, timestamp, <shared features...>,
        target, horizon``, where ``target`` is this horizon's forward return and
        ``horizon`` its label.  Sorted by horizon, symbol, timestamp.

    Notes
    -----
    A symbol that cannot be built, or that has no labels on any requested
    horizon, is logged and left out of the frame; it is never filled in.  If
    *nothing* survives, a :class:`~src.v3.dataset.V3DatasetError` is raised
    rather than returning an empty frame, because an empty panel is
    indistinguishable from a panel that was never built.
    """
    ladder = load_horizons(config, horizons)
    profiles, datasets, skipped = _collect_profiles(
        config, symbols, ladder, enabled_features=None, max_rows=max_rows
    )
    for symbol, reason in skipped.items():
        logger.warning("align_panel: leaving %s out of the panel - %s", symbol, reason)
    if not datasets:
        raise V3DatasetError(
            f"align_panel: no symbol could be built, so there is no panel to align; {skipped}"
        )

    order = [profile.symbol for profile in profiles]
    shared = _shared_features(datasets, order)
    if not shared:
        raise V3DatasetError(
            "align_panel: no feature is common to every symbol, so a stacked panel would not be "
            f"like-for-like; available feature counts are "
            f"{ {s: len(d.feature_columns) for s, d in datasets.items()} }"
        )
    logger.info(
        "align_panel: %d symbol(s) on the %d features they all have, spanning bucket(s) %s",
        len(datasets),
        len(shared),
        list(_shared_buckets(datasets, order, shared)),
    )

    blocks: list[pd.DataFrame] = []
    for profile in profiles:
        dataset = datasets[profile.symbol]
        for horizon in ladder:
            labelled = dataset.labelled_frame(horizon)
            if labelled.empty:
                logger.warning(
                    "align_panel: %s has no labels for %s and is left out of that horizon",
                    profile.symbol,
                    horizon.label,
                )
                continue
            block = labelled.loc[:, list(shared)].copy()
            block["target"] = labelled[horizon.target_column()].to_numpy()
            block["horizon"] = horizon.label
            block["symbol"] = profile.symbol
            blocks.append(block.rename_axis("timestamp").reset_index())

    if not blocks:  # pragma: no cover - only reachable if every frame is label-less
        raise V3DatasetError(
            f"align_panel: none of the {len(datasets)} symbol(s) has a single label on "
            f"{[horizon.label for horizon in ladder]}; the panel would be empty"
        )

    panel = pd.concat(blocks, ignore_index=True)
    columns = [*_PANEL_LEADING_COLUMNS, *shared, *_PANEL_TRAILING_COLUMNS]
    return panel.loc[:, columns].sort_values(
        ["horizon", "symbol", "timestamp"], kind="mergesort", ignore_index=True
    )
