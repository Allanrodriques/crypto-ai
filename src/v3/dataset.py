"""Point-in-time modelling frame for V3, one per symbol.

What this module produces
-------------------------
A :class:`V3Dataset`: features plus one ``future_return_<label>`` column per
horizon, on a single sorted, unique, UTC-indexed grid.  Everything the
downstream training and reporting code needs to judge whether that frame is
usable - per-feature coverage, per-horizon label coverage, and a row-by-row
account of every dropped row - travels with the frame rather than being
rediscovered later.

Point-in-time guarantee
-----------------------
``feature[t]`` uses information available at or before ``t``.  This module
enforces that structurally rather than by convention:

* **No forward fill and no back fill happen here at all.**  There is no
  ``ffill``, no ``bfill`` and no ``interpolate`` in this file, so the only way
  a sparse source could reach backwards is if a reused V2 helper did it.  The
  one such helper is :func:`src.alignment.availability.align_asof`, which joins
  *backward only* (``merge_asof(direction="backward")`` keyed on the
  **availability** timestamp, never the event timestamp) and refuses to carry a
  value forward past the source's ``max_age``.  It records the availability
  timestamp of every value it supplies, and
  :func:`src.alignment.availability.audit_no_lookahead` re-checks that
  provenance here, so a regression in the alignment layer fails this build
  instead of silently leaking.
* **The V2 feature builders are used unmodified.**  The technical block is the
  immutable V1 baseline; the microstructure, context, derivatives and sentiment
  builders are each causal by their own construction and their own tests.  V3
  reuses them rather than reimplementing an indicator, precisely so a
  point-in-time property cannot be lost in a rewrite.
* **Truncation happens before labelling.**  ``max_rows`` slices the price
  history *first* and the targets are built on the surviving window, so a
  truncated dataset's label tail is honestly NaN instead of being resolved
  against prices the dataset does not contain.
* **Labels are never imputed.**  The tail of every horizon stays NaN and
  :meth:`V3Dataset.labelled_frame` is the only supported way to get trainable
  rows, so "drop the unlabelled tail" is something a caller has to do
  deliberately.

Nothing here writes to disk.  The only filesystem calls are ``Path.exists`` and
``pandas.read_parquet``, so building a dataset is safe inside the test suite's
protected-directory guard.  Note in particular that the V2
:func:`src.dataset.multisource.load_or_fetch` helper is *not* used: it writes a
cache when it downloads, and this module never downloads.  It reads the
already-cached external sources itself and hands them to
:func:`src.dataset.multisource.align_sources`, which is the auditable part of
the V2 pipeline.

Degrading instead of crashing
-----------------------------
A genuinely unavailable source (no funding cache, no fear & greed cache) is
logged loudly, recorded in :attr:`V3Dataset.missing_sources`, and the feature
groups that depend on it are dropped from the selection.  Continuing with the
features that do exist is the honest response: the alternative is either a
crash on a missing cache or an all-NaN feature block that empties the dataset.
A missing *spot* kline file is fatal, because nothing can be built without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.alignment.availability import AlignmentError, audit_no_lookahead
from src.config import Config
from src.data.quality import SOURCE_CONTRACTS
from src.dataset.multisource import (
    AlignedSources,
    SourceBundle,
    SourceUnavailableError,
    align_sources,
    build_feature_matrix,
)
from src.features.groups import GROUP_ORDER, REGISTRY, known_features
from src.utils import get_logger
from src.v3.horizons import Horizon, build_horizon_ladder
from src.v3.targets import build_future_returns

logger = get_logger("v3.dataset")

# ---------------------------------------------------------------------------
# optional sibling registry
# ---------------------------------------------------------------------------
# `src.v3.buckets` is owned by a different part of the V3 work and may not exist
# yet.  The import is therefore guarded rather than assumed: when it is absent
# the full canonical V2 registry (107 features, `src.features.groups`) is used
# and bucket-level ablation is simply unavailable.  It is never faked with a
# stub, because a stub that silently accepted any name would make a typo in a
# config look like a working feature selection.
try:  # pragma: no cover - depends on a sibling module
    from src.v3.buckets import BUCKET_NAMES as _BUCKET_NAMES
    from src.v3.buckets import bucket_for_feature as _bucket_for_feature
    from src.v3.buckets import features_in_bucket as _features_in_bucket
    from src.v3.buckets import resolve_features as _resolve_bucket_features
    from src.v3.buckets import validate_selection as _validate_bucket_selection

    BUCKETS_AVAILABLE = True
    BUCKETS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - depends on a sibling module
    _BUCKET_NAMES: tuple[str, ...] | None = ()
    _bucket_for_feature = None
    _features_in_bucket = None
    _resolve_bucket_features = None
    _validate_bucket_selection = None
    BUCKETS_AVAILABLE = False
    BUCKETS_IMPORT_ERROR = exc
    logger.warning(
        "src.v3.buckets is not importable (%s); using the full %d-feature V2 registry from "
        "src.features.groups. Bucket-based feature ablation is unavailable for this run.",
        exc,
        len(known_features()),
    )

#: Bars per day on the base grid.  Mirrors ``v3.bars_per_day``; 24 for 1h.
DEFAULT_BARS_PER_DAY = 24

#: External sources a feature group cannot be built without.
GROUP_SOURCES: dict[str, tuple[str, ...]] = {
    "derivatives": ("binance_funding", "binance_futures"),
    "sentiment": ("fear_greed",),
}

#: Every row this builder can drop, in the order the drops happen.  The keys are
#: always present in :attr:`V3Dataset.drop_report` (0 when nothing was dropped)
#: and they are disjoint and exhaustive, so
#: ``len(dataset) + sum(dataset.drop_report.values()) == n_input_rows``.
DROP_REASONS: tuple[str, ...] = (
    "duplicate_timestamps",
    "missing_price",
    "feature_warmup",
    "missing_feature",
    "non_finite_feature",
    "max_rows_truncated",
)

#: Documented columns of :attr:`V3Dataset.feature_coverage`.
FEATURE_COVERAGE_COLUMNS: tuple[str, ...] = (
    "feature",
    "dtype",
    "n_non_null",
    "n_missing",
    "coverage_pct",
    "bucket",
)

#: Documented columns of :attr:`V3Dataset.target_coverage` -
#: :meth:`src.v3.targets.FutureReturnTargets.coverage_table` plus ``symbol``.
TARGET_COVERAGE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "horizon",
    "requested_hours",
    "realised_mean_hours",
    "realised_min_hours",
    "n_total",
    "n_labelled",
    "n_unlabelled_tail",
    "labelled_pct",
    "last_labelled_timestamp",
)

_TRUTHY = frozenset({"all", "true", "on", "yes", "full", "1"})
_FALSY = frozenset({"none", "false", "off", "no", "0", ""})


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class V3DatasetError(RuntimeError):
    """Raised when a V3 dataset cannot be built at all."""


class DataUnavailableError(V3DatasetError):
    """Raised when the inputs needed to build any frame are absent."""


class FeatureSelectionError(V3DatasetError, ValueError):
    """Raised for an unknown feature/bucket name or an empty selection.

    Subclasses :class:`ValueError` so a caller that only knows the V1/V2
    convention (``except ValueError``) still catches it.
    """


# ---------------------------------------------------------------------------
# config access
# ---------------------------------------------------------------------------

def v3_block(config: Config | Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``v3:`` section of a config as a plain dict.

    Accepts a :class:`~src.config.Config`, a raw config mapping, or the ``v3:``
    mapping itself, so callers that already extracted the block do not have to
    unwrap it twice.
    """
    if isinstance(config, Config):
        block = config.raw.get("v3")
    elif isinstance(config, Mapping):
        block = config.get("v3", config)
    else:
        raise TypeError(f"config must be a Config or a mapping, got {type(config).__name__}")
    if not isinstance(block, Mapping):
        raise V3DatasetError(
            "config has no `v3:` section; point Config at config/v3.yaml, which is the "
            "V3 contract (config/config.yaml is the frozen V1/V2 baseline)"
        )
    return dict(block)


def load_horizons(
    config: Config | Mapping[str, Any], horizons: Sequence[str] | Sequence[Horizon] | None = None
) -> tuple[Horizon, ...]:
    """The configured horizon ladder, or an explicit override.

    ``horizons`` may be a list of labels (``["1d", "7d"]``) or of already-built
    :class:`~src.v3.horizons.Horizon` objects.  Either way the result is sorted
    by duration, so the target column order is canonical.
    """
    block = v3_block(config)
    if horizons is None:
        return build_horizon_ladder(block)
    if isinstance(horizons, Horizon):
        return (horizons,)
    values = list(horizons)
    if not values:
        raise V3DatasetError("horizons=[] selects nothing; at least one horizon is required")
    if all(isinstance(value, Horizon) for value in values):
        return tuple(sorted(values, key=lambda h: (h.delta, h.label)))
    bars_per_day = int(block.get("bars_per_day", DEFAULT_BARS_PER_DAY))
    return Horizon.from_config(values, bars_per_day)


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSelection:
    """A resolved feature selection, grouped by the builder that produces it.

    Attributes
    ----------
    groups:
        V2 feature-group names in canonical registry order - the units the
        feature *builders* work in.
    by_group:
        ``{group: [feature, ...]}``, each list in that group's canonical order.
        This is what is handed to
        :func:`src.dataset.multisource.build_feature_matrix`.
    columns:
        The final column order of the frame: V3 bucket order when
        ``src.v3.buckets`` is available (it declares that order part of the
        contract), otherwise V2 group order.
    buckets:
        The V3 buckets the selection resolves to, in canonical order.
    origin:
        ``"config"`` or ``"caller"`` - whether the selection came from
        ``v3.features`` or from the ``enabled_features`` argument.
    warnings:
        Non-fatal notes, e.g. a configured bucket name the bucket registry does
        not know, or a requested feature the builders did not produce.
    """

    groups: tuple[str, ...]
    by_group: dict[str, list[str]]
    columns: list[str]
    origin: str
    buckets: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.columns)


def _flag(value: Any, *, label: str = "value") -> bool:
    """Interpret a YAML switch that may be a bool, ``"all"`` or ``"none"``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise FeatureSelectionError(
        f"{label}={value!r} is not a switch; use true/false (or the strings 'all'/'none')"
    )


def _owning_group(feature: str) -> str | None:
    """The canonical feature group that registers ``feature``, if any."""
    for group in GROUP_ORDER:
        if feature in REGISTRY[group].features:
            return group
    return None


def bucket_of_feature(feature: str) -> str:
    """The bucket a feature belongs to.

    Uses the V3 bucket registry when it is importable, otherwise the canonical
    V2 group.  An unknown feature maps to ``"unknown"`` rather than raising, so
    building a coverage table over a partial matrix cannot fail.
    """
    if _bucket_for_feature is not None:
        try:
            return str(_bucket_for_feature(feature))
        except Exception:  # pragma: no cover - defensive: unknown name
            pass
    return _owning_group(feature) or "unknown"


def _bucket_members(name: str) -> list[str]:
    """Every feature in a bucket / feature group."""
    if _features_in_bucket is not None:
        try:
            return list(_features_in_bucket(name))
        except Exception as exc:
            raise FeatureSelectionError(f"Unknown feature bucket {name!r}: {exc}") from exc
    if name in REGISTRY:
        return list(REGISTRY[name].features)
    raise FeatureSelectionError(
        f"Unknown feature bucket/group {name!r}. src.v3.buckets is not importable, so only the "
        f"canonical V2 groups are understood: {sorted(REGISTRY)}"
    )


def _bucket_known(name: str) -> bool:
    if name in REGISTRY:
        return True
    if _features_in_bucket is None:
        return False
    return name in set(_BUCKET_NAMES)


def _expand(requested: Any) -> tuple[dict[str, list[str]], str]:
    """Normalise a requested selection into ``(mapping, terms)``.

    ``mapping`` is ``{bucket_or_group: [feature, ...]}``; ``terms`` records how
    the request was expressed, which decides whether the bucket registry is the
    right authority for it:

    ``"bucket"``
        A mapping, i.e. ``{price: all}`` or ``{price: [sma_20, ...]}``.  Bucket
        terms - the V3 taxonomy, validated by ``src.v3.buckets``.
    ``"group"``
        A name (or list of names) that are V2 feature *groups*.  V2 groups and
        V3 buckets overlap in name (``technical``) but not in membership, so
        this is deliberately routed to the V2 registry instead of the bucket
        registry rather than being guessed at.
    ``"feature"``
        A list of individual feature names.
    ``"all"``
        Nothing was requested: the full registry.

    The switch form a config naturally wants (``{price: all}``) is expanded
    through :func:`features_in_bucket` so the strict ``{bucket: [feature, ...]}``
    form the bucket registry expects is what reaches it.
    """
    if requested is None:
        return {group: list(REGISTRY[group].features) for group in GROUP_ORDER}, "all"

    if isinstance(requested, str):
        if requested in REGISTRY:
            return {requested: list(REGISTRY[requested].features)}, "group"
        return {requested: _bucket_members(requested)}, "bucket"

    if isinstance(requested, Mapping):
        expanded: dict[str, list[str]] = {}
        for key, value in requested.items():
            name = str(key)
            if isinstance(value, (list, tuple, set, frozenset)):
                expanded[name] = [str(item) for item in value]
            elif _flag(value, label=f"features.{name}"):
                expanded[name] = _bucket_members(name)
            # an explicit false / "none" simply switches the bucket off
        return expanded, "bucket"

    if isinstance(requested, Sequence):
        names = [str(name) for name in requested]
        if not names:
            return {}, "feature"
        if all(name in REGISTRY for name in names):
            return {name: list(REGISTRY[name].features) for name in names}, "group"
        return {name: [name] for name in names}, "feature"

    raise FeatureSelectionError(
        f"Cannot interpret a feature selection of type {type(requested).__name__}; pass None, a "
        f"bucket->switch mapping, or a list of bucket/group/feature names"
    )


def _bucket_view(expanded: Mapping[str, list[str]], terms: str) -> dict[str, list[str]] | None:
    """The selection as a strict ``{bucket: [feature, ...]}`` mapping, if it is one.

    ``None`` means the request was expressed in V2 group terms or as a flat list
    of feature names, which the bucket registry is not the authority for.
    """
    if terms != "bucket" or not _BUCKET_NAMES or not expanded:
        return None
    known = set(_BUCKET_NAMES)
    if not all(name in known for name in expanded):
        return None
    return {name: list(members) for name, members in expanded.items() if members}


def _canonical_column_order(columns: Sequence[str]) -> list[str]:
    """Sort selected features into the contract column order.

    V3 bucket order when ``src.v3.buckets`` is importable - that module declares
    the order part of its contract - otherwise the order they were resolved in
    (V2 group order).
    """
    columns = list(columns)
    if not _BUCKET_NAMES:
        return columns
    rank = {name: position for position, name in enumerate(_BUCKET_NAMES)}
    return sorted(columns, key=lambda name: (rank.get(bucket_of_feature(name), len(rank)), name))


#: Keys of ``v3.features`` that configure the *build* rather than select
#: features, and so are not feature names.
_FEATURE_CONTROL_KEYS = frozenset({"mode", "buckets", "complete_rows_only", "groups"})

#: Accepted values of ``v3.features.mode``.
_FEATURE_MODES = frozenset({"all", "buckets", "explicit"})


def _configured_buckets(features_cfg: Mapping[str, Any], *, mode: str) -> Any:
    """The bucket selection declared in ``v3.features``.

    ``buckets`` is the declared taxonomy.  ``groups`` is an escape hatch for
    callers that want to name V2 feature groups directly.  ``None`` (no
    declaration) means the full registry.
    """
    if mode not in _FEATURE_MODES:
        raise FeatureSelectionError(
            f"v3.features.mode={mode!r} is not supported; use one of {sorted(_FEATURE_MODES)}"
        )
    if mode == "explicit":
        raise FeatureSelectionError(
            "v3.features.mode is 'explicit', so the selection must be passed as "
            "`enabled_features=` to build_v3_dataset()"
        )
    buckets = features_cfg.get("buckets")
    if isinstance(buckets, Mapping):
        return buckets
    groups = features_cfg.get("groups")
    if isinstance(groups, (list, tuple)) and groups:
        return [str(group) for group in groups]
    stray = {
        key: value
        for key, value in features_cfg.items()
        if key not in _FEATURE_CONTROL_KEYS
    }
    return stray or None


def resolve_feature_selection(
    config: Config | Mapping[str, Any], enabled_features: Any = None
) -> FeatureSelection:
    """Resolve ``v3.features`` (or an override) into a concrete column list.

    Parameters
    ----------
    config:
        A :class:`~src.config.Config`, a raw config mapping, or the ``v3:``
        block itself.
    enabled_features:
        Overrides the config's selection.  ``None`` uses ``v3.features``.
        Otherwise: a ``{bucket: true|false|"all"|"none"}`` mapping, a
        ``{bucket: [feature, ...]}`` mapping, a single bucket/group name, or a
        list of group names or feature names.

    Raises
    ------
    FeatureSelectionError
        On an unknown bucket, an unknown feature name, or a selection that
        resolves to nothing.  A typo must fail loudly: a silently dropped
        feature looks like a working ablation.
    """
    block = v3_block(config)
    features_cfg = dict(block.get("features") or {})
    mode = str(features_cfg.get("mode", "all")).strip().lower()
    origin = "config" if enabled_features is None else "caller"
    notes: list[str] = []

    if enabled_features is None:
        requested: Any = _configured_buckets(features_cfg, mode=mode)
    else:
        requested = enabled_features

    # A bucket name this build does not recognise must not silently empty the
    # feature set.  A config written against a different bucket registry
    # degrades to "no ablation" (the full registry) with a loud warning.
    if enabled_features is None and isinstance(requested, Mapping):
        unknown = [
            str(key)
            for key, value in requested.items()
            if _flag(value, label=f"v3.features.{key}") and not _bucket_known(str(key))
        ]
        if unknown and len(unknown) == len(requested):
            notes.append(
                f"none of the configured bucket names {unknown} are known to "
                f"{'src.v3.buckets.BUCKET_NAMES' if BUCKETS_AVAILABLE else 'the V2 group registry'}; "
                f"falling back to the full {len(known_features())}-feature registry"
            )
            logger.warning("V3 dataset: %s", notes[-1])
            requested = None

    expanded, terms = _expand(requested)

    # When the request is expressed in bucket terms, the bucket registry is the
    # authority: it validates the selection and returns the canonical ordering.
    # Its answer is then re-checked against the V2 feature registry below, so the
    # two taxonomies cannot silently disagree about what exists.
    view = _bucket_view(expanded, terms)
    if view is not None and _validate_bucket_selection is not None:
        try:
            _validate_bucket_selection(view)
        except Exception as exc:
            raise FeatureSelectionError(
                f"Invalid V3 feature selection {view!r}: {exc}"
            ) from exc
    if view is not None and _resolve_bucket_features is not None:
        try:
            authoritative = list(_resolve_bucket_features(view))
        except Exception as exc:
            raise FeatureSelectionError(
                f"Invalid V3 feature selection {view!r}: {exc}"
            ) from exc
        expanded = {"__resolved__": authoritative}

    by_group: dict[str, list[str]] = {}
    unknown_features: list[str] = []
    for members in expanded.values():
        for name in members:
            group = _owning_group(name)
            if group is None:
                unknown_features.append(name)
                continue
            bucket = by_group.setdefault(group, [])
            if name not in bucket:
                bucket.append(name)

    if unknown_features:
        bad = sorted(set(unknown_features))
        raise FeatureSelectionError(
            f"Unknown feature name(s) {bad}. Every V3 feature must be registered in "
            f"src.features.groups ({len(known_features())} features in "
            f"{sorted(REGISTRY)}); check the spelling, or list a group name to take it in full."
        )
    if not by_group:
        raise FeatureSelectionError(
            "Feature selection resolved to zero features; check v3.features / enabled_features."
        )

    ordered: dict[str, list[str]] = {}
    for group in GROUP_ORDER:
        wanted = set(by_group.get(group, ()))
        if wanted:
            ordered[group] = [name for name in REGISTRY[group].features if name in wanted]

    columns = _canonical_column_order([name for group in ordered for name in ordered[group]])
    buckets = tuple(
        name
        for name in (_BUCKET_NAMES or ())
        if any(bucket_of_feature(feature) == name for feature in columns)
    )

    return FeatureSelection(
        groups=tuple(ordered),
        by_group=ordered,
        columns=columns,
        origin=origin,
        buckets=buckets,
        warnings=tuple(notes),
    )


# ---------------------------------------------------------------------------
# read-only input loading
# ---------------------------------------------------------------------------

def raw_kline_path(config: Config, symbol: str, interval: str) -> Path:
    """Path of a symbol's raw hourly klines, or raise if it is not there.

    Follows the project's own naming convention ``<SYMBOL>_<interval>.parquet``
    under ``paths.raw_dir`` (and the ``external/`` sub-directory some sources
    use).  Never creates anything.
    """
    raw_dir = Path(config.paths.raw_dir)
    stem = f"{symbol}_{interval}"
    for directory in (raw_dir, raw_dir / "external"):
        candidate = directory / f"{stem}.parquet"
        if candidate.exists():
            return candidate
    raise DataUnavailableError(
        f"No raw klines for {symbol} at {interval} under {raw_dir}. Expected "
        f"{stem}.parquet. Run the V1 downloader first; this module never fetches data."
    )


def _read_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name="timestamp")
    return frame.sort_index()


def load_spot_klines(config: Config, symbol: str, interval: str | None = None) -> pd.DataFrame:
    """Load one symbol's raw klines read-only, on a sorted UTC index."""
    symbol = str(symbol).strip().upper()
    interval = str(interval or config.data.get("interval") or "1h")
    return _read_utc_index(pd.read_parquet(raw_kline_path(config, symbol, interval)))


def _source_path(config: Config, source: str, symbol: str) -> Path | None:
    raw_dir = Path(config.paths.raw_dir)
    for directory in (raw_dir, raw_dir / "external"):
        candidate = directory / f"{source}_{symbol}.parquet"
        if candidate.exists():
            return candidate
    return None


def _load_bundle(
    config: Config, symbol: str, spot: pd.DataFrame
) -> tuple[SourceBundle, tuple[str, ...]]:
    """Assemble the raw (unaligned) source bundle from whatever is cached.

    Returns the bundle plus the names of the sources that are genuinely absent,
    so the caller can record them instead of failing.
    """
    bundle = SourceBundle(spot=spot)
    missing: list[str] = []
    for source in ("binance_funding", "binance_futures", "fear_greed"):
        path = _source_path(config, source, symbol)
        if path is None:
            missing.append(source)
            logger.warning(
                "V3 dataset %s: cached source %r not found under %s; the feature groups that "
                "need it are dropped and the rest of the dataset is still built",
                symbol,
                source,
                config.paths.raw_dir,
            )
            continue
        frame = _read_utc_index(pd.read_parquet(path))
        if frame.empty:
            missing.append(source)
            logger.warning("V3 dataset %s: cached source %r (%s) is empty", symbol, source, path.name)
            continue
        logger.info("V3 dataset %s: source %s (%d rows) from %s", symbol, source, len(frame), path.name)
        if source == "binance_funding":
            bundle.funding = frame
        elif source == "binance_futures":
            bundle.futures = frame
        else:
            bundle.sentiment = frame
    return bundle, tuple(missing)


def _audit_alignment(aligned: AlignedSources, symbol: str) -> None:
    """Re-check every aligned external source against its recorded provenance.

    :func:`src.alignment.availability.align_asof` records the availability
    timestamp of each value it supplies.  This turns that provenance into a hard
    check at build time, so a look-ahead introduced in the alignment layer fails
    here rather than reaching a model.
    """
    for source, frame in (
        ("binance_funding", aligned.funding),
        ("binance_futures", aligned.futures),
        ("fear_greed", aligned.sentiment),
    ):
        if frame is None or frame.empty:
            continue
        spec = SOURCE_CONTRACTS.get(source)
        if spec is None:  # pragma: no cover - every source above is registered
            continue
        populated = [c for c in frame.columns if frame[c].notna().any()]
        if not populated:
            continue
        try:
            audit_no_lookahead(frame, spec.contract(), aligned.index, populated[0])
        except AlignmentError as exc:
            raise V3DatasetError(
                f"point-in-time audit failed for {source} ({symbol}): {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# coverage tables
# ---------------------------------------------------------------------------

def _feature_coverage(
    features: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Per-feature coverage, worst first.

    Sorted ascending by ``coverage_pct`` so a gap is the first row a reader
    sees, and ties broken by feature name so the order is deterministic.
    """
    total = int(len(features))
    rows = []
    for name in columns:
        series = features[name]
        n_non_null = int(series.notna().sum())
        rows.append(
            {
                "feature": name,
                "dtype": str(series.dtype),
                "n_non_null": n_non_null,
                "n_missing": total - n_non_null,
                "coverage_pct": round(100.0 * n_non_null / total, 6) if total else 0.0,
                "bucket": bucket_of_feature(name),
            }
        )
    table = pd.DataFrame(rows, columns=list(FEATURE_COVERAGE_COLUMNS))
    return table.sort_values(
        ["coverage_pct", "feature"], kind="mergesort", ignore_index=True
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V3Dataset:
    """One symbol's point-in-time modelling frame plus its data-quality record.

    Attributes
    ----------
    symbol:
        The symbol this frame belongs to (upper case).
    frame:
        ``feature_columns`` followed by ``target_columns``, indexed by unique
        sorted UTC timestamps.  The label tail of every horizon is ``NaN``.
    horizons:
        The horizon ladder, sorted by duration.
    feature_columns / target_columns:
        Column order of ``frame``'s two blocks.
    feature_coverage:
        Columns :data:`FEATURE_COVERAGE_COLUMNS`, sorted worst-coverage-first.
    target_coverage:
        Columns :data:`TARGET_COVERAGE_COLUMNS`; ``targets.coverage_table()``
        with the symbol attached.
    drop_report:
        Rows dropped per reason over :data:`DROP_REASONS`.  Keys are always
        present (0 when nothing dropped) and are disjoint and exhaustive, so
        ``len(dataset) + sum(drop_report.values())`` is the number of raw rows
        that went in.
    missing_sources:
        External sources that were requested and are not cached.  The feature
        groups that needed them were dropped; see the build log.
    feature_groups:
        The feature groups actually built, in canonical order.
    max_rows:
        The ``max_rows`` cap that was applied, or ``None``.
    """

    symbol: str
    frame: pd.DataFrame
    horizons: tuple[Horizon, ...]
    feature_columns: list[str]
    target_columns: list[str]
    feature_coverage: pd.DataFrame
    target_coverage: pd.DataFrame
    drop_report: dict[str, int]
    missing_sources: tuple[str, ...] = ()
    feature_groups: tuple[str, ...] = ()
    max_rows: int | None = None

    # ------------------------------------------------------------- basics
    def __len__(self) -> int:
        return int(len(self.frame))

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def resolve_horizon(self, value: Horizon | str) -> Horizon:
        """Look a horizon up by object or by label; raise if it is not in the ladder."""
        label = value.label if isinstance(value, Horizon) else str(value).strip()
        for horizon in self.horizons:
            if horizon.label == label:
                return horizon
        raise V3DatasetError(
            f"Horizon {label!r} is not in this dataset; available: {[h.label for h in self.horizons]}"
        )

    # ------------------------------------------------------------- labels
    def labelled_frame(self, horizon: Horizon | str) -> pd.DataFrame:
        """Rows with a *real* target for ``horizon``.

        The unlabelable tail is dropped here, by the caller, on purpose: it is
        the only place a V3 label may be filtered, and nothing in this module
        imputes a target.
        """
        column = self.resolve_horizon(horizon).target_column()
        if column not in self.target_columns:
            raise V3DatasetError(f"Target column {column!r} is not present in this dataset")
        mask = self.frame[column].notna()
        return self.frame.loc[mask]

    def labelled_index(self, horizon: Horizon | str) -> pd.DatetimeIndex:
        """Index of the rows :meth:`labelled_frame` would return."""
        return self.labelled_frame(horizon).index

    def n_labelled(self, horizon: Horizon | str) -> int:
        return int(len(self.labelled_index(horizon)))

    # ------------------------------------------------------------- report
    def describe(self) -> dict[str, object]:
        """A JSON-friendly summary of the frame and its data quality."""
        coverage = self.feature_coverage
        span = {
            "name": self.frame.index.name,
            "start": str(self.frame.index.min()) if len(self.frame) else None,
            "end": str(self.frame.index.max()) if len(self.frame) else None,
            "tz": str(self.frame.index.tz),
            "is_unique": bool(self.frame.index.is_unique),
            "is_monotonic_increasing": bool(self.frame.index.is_monotonic_increasing),
        }
        per_horizon: dict[str, dict[str, object]] = {}
        for row in self.target_coverage.itertuples(index=False):
            per_horizon[str(row.horizon)] = {
                "n_labelled": int(row.n_labelled),
                "n_unlabelled_tail": int(row.n_unlabelled_tail),
                "labelled_pct": float(row.labelled_pct),
                "requested_hours": float(row.requested_hours),
            }
        return {
            "symbol": self.symbol,
            "n_rows": len(self.frame),
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
            "feature_groups": list(self.feature_groups),
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "horizons": [horizon.to_dict() for horizon in self.horizons],
            "index": span,
            "max_rows": self.max_rows,
            "missing_sources": list(self.missing_sources),
            "drop_report": dict(self.drop_report),
            "n_rows_dropped": int(sum(self.drop_report.values())),
            "feature_coverage": {
                "min_pct": float(coverage["coverage_pct"].min()) if len(coverage) else None,
                "median_pct": float(coverage["coverage_pct"].median()) if len(coverage) else None,
                "max_pct": float(coverage["coverage_pct"].max()) if len(coverage) else None,
                "n_below_50pct": int((coverage["coverage_pct"] < 50.0).sum()),
            },
            "target_coverage": per_horizon,
            "bucket_registry_available": BUCKETS_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def build_v3_dataset(
    config: Config,
    symbol: str,
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    enabled_features: Any = None,
    max_rows: int | None = None,
) -> V3Dataset:
    """Assemble one symbol's leak-free V3 modelling frame.

    Parameters
    ----------
    config:
        A config carrying a ``v3:`` block (``config/v3.yaml``).
    symbol:
        Symbol to build, e.g. ``"BTCUSDT"``.
    horizons:
        Optional override of the configured ladder (labels or
        :class:`~src.v3.horizons.Horizon` objects).  ``None`` uses
        ``v3.horizons``.
    enabled_features:
        Optional override of the configured feature selection - see
        :func:`resolve_feature_selection`.
    max_rows:
        Keep only the most recent ``N`` rows.  It is applied to the price and
        feature history **before** the targets are built, so the truncated
        frame's label tail is honestly ``NaN``: no label ever resolves against
        a price the dataset does not contain.  A consequence worth stating: on a
        truncated frame the short horizons are fully labelled while the long
        ones lose up to 180 days of otherwise-available labels.

    Returns
    -------
    V3Dataset
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise V3DatasetError("symbol must be a non-empty string such as 'BTCUSDT'")

    block = v3_block(config)
    targets_cfg = dict(block.get("targets") or {})
    kind = str(targets_cfg.get("kind", "forward_return"))
    if kind != "forward_return":
        raise V3DatasetError(
            f"v3.targets.kind={kind!r} is not supported; this module builds forward-return targets"
        )
    price_column = str(targets_cfg.get("price_column", "close"))
    interval = str(block.get("interval") or config.data.get("interval") or "1h")
    complete_only = _flag(
        dict(block.get("features") or {}).get("complete_rows_only", True),
        label="v3.features.complete_rows_only",
    )

    ladder = load_horizons(block, horizons)
    selection = resolve_feature_selection(block, enabled_features)
    for note in selection.warnings:
        logger.warning("V3 dataset %s: %s", symbol, note)

    # ---------------------------------------------------------- 1. klines
    klines = load_spot_klines(config, symbol, interval=interval)
    n_input = int(len(klines))
    if n_input == 0:
        raise DataUnavailableError(f"Raw klines for {symbol} ({interval}) are empty")
    drop_report: dict[str, int] = {reason: 0 for reason in DROP_REASONS}

    if klines.index.has_duplicates:
        drop_report["duplicate_timestamps"] = int(klines.index.duplicated().sum())
        logger.warning(
            "V3 dataset %s: %d duplicate timestamp(s) in the raw klines; keeping the last of each",
            symbol,
            drop_report["duplicate_timestamps"],
        )
        klines = klines.loc[~klines.index.duplicated(keep="last")]

    if price_column not in klines.columns:
        raise DataUnavailableError(
            f"Raw klines for {symbol} have no {price_column!r} column "
            f"(v3.targets.price_column); available: {sorted(klines.columns)}"
        )

    # A close that is missing, non-numeric, non-finite or non-positive cannot
    # produce an honest return: 0/0 and x/0 are not prices.  Drop the row and
    # record it rather than carrying it as a feature row with no label.
    price = pd.to_numeric(klines[price_column], errors="coerce")
    values = price.to_numpy(dtype="float64")
    usable_price = np.isfinite(values) & (values > 0.0)
    drop_report["missing_price"] = int((~usable_price).sum())
    if drop_report["missing_price"]:
        logger.info(
            "V3 dataset %s: dropped %d row(s) with a missing/non-finite/non-positive %s",
            symbol,
            drop_report["missing_price"],
            price_column,
        )
    klines = klines.loc[usable_price]
    price = price.loc[usable_price]
    if klines.empty:
        raise DataUnavailableError(
            f"Every candle for {symbol} was dropped as missing/non-finite {price_column!r}"
        )

    # ------------------------------------------------- 2. external sources
    bundle, missing_sources = _load_bundle(config, symbol, klines)
    aligned = align_sources(bundle, klines.index)
    _audit_alignment(aligned, symbol)

    # ------------------------------------------------ 3. usable groups
    usable_groups: list[str] = []
    for group in selection.groups:
        absent = [s for s in GROUP_SOURCES.get(group, ()) if s in missing_sources]
        if absent:
            logger.warning(
                "V3 dataset %s: dropping feature group %r - source(s) %s are not available",
                symbol,
                group,
                absent,
            )
            continue
        usable_groups.append(group)
    if not usable_groups:
        raise DataUnavailableError(
            f"V3 dataset {symbol}: no buildable feature group - every requested group "
            f"{list(selection.groups)} needs one of the missing sources {list(missing_sources)}"
        )

    # ------------------------------------------------------ 4. features
    by_group = {group: selection.by_group[group] for group in usable_groups}
    try:
        matrix = build_feature_matrix(aligned, usable_groups, enabled_features=by_group)
    except SourceUnavailableError as exc:
        raise DataUnavailableError(f"V3 dataset {symbol}: {exc}") from exc

    matrix = _read_utc_index(matrix)
    matrix = matrix.loc[~matrix.index.duplicated(keep="last")]
    grid = klines.index.intersection(matrix.index)
    # Everything priced but not yet in the matrix is still inside a warm-up
    # window (or belongs to a group that could not be built).
    drop_report["feature_warmup"] = int(len(klines.index.difference(matrix.index)))
    matrix = matrix.reindex(grid)

    columns = [name for name in selection.columns if name in matrix.columns]
    # A feature absent because its whole group was pruned (its source is not
    # cached) was already reported when the group was dropped; only a feature
    # that a *built* group failed to produce is a real discrepancy.
    unbuilt = [
        name
        for name in selection.columns
        if name not in matrix.columns and _owning_group(name) in usable_groups
    ]
    if unbuilt:
        logger.warning(
            "V3 dataset %s: %d selected feature(s) were not produced by the builders and are "
            "absent from the frame: %s",
            symbol,
            len(unbuilt),
            unbuilt[:10],
        )
    if not columns:
        raise FeatureSelectionError(
            f"V3 dataset {symbol}: none of the {len(selection.columns)} selected features were "
            f"built for group(s) {usable_groups}"
        )

    features = matrix.loc[:, columns]
    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype="float64", copy=False)
    has_nan = pd.Series(np.isnan(numeric_values).any(axis=1), index=features.index)
    has_inf = pd.Series(np.isinf(numeric_values).any(axis=1), index=features.index)

    if complete_only:
        # A row with a missing feature is not trainable, and imputing it would
        # invent information.  The two reasons are counted disjointly so the
        # drop report stays exhaustive.
        drop_report["missing_feature"] = int((has_nan & ~has_inf).sum())
        drop_report["non_finite_feature"] = int((has_inf & ~has_nan).sum())
        incomplete = has_nan | has_inf
        if incomplete.any():
            logger.info(
                "V3 dataset %s: dropped %d row(s) with an incomplete feature vector "
                "(%d missing, %d non-finite)",
                symbol,
                int(incomplete.sum()),
                drop_report["missing_feature"],
                drop_report["non_finite_feature"],
            )
        features = features.loc[~incomplete]
    else:
        logger.warning(
            "V3 dataset %s: v3.features.complete_rows_only is false, so %d row(s) with missing "
            "features are kept; see feature_coverage",
            symbol,
            int((has_nan | has_inf).sum()),
        )

    if features.empty:
        raise DataUnavailableError(
            f"V3 dataset {symbol}: no rows survived - the history is shorter than the feature "
            f"warm-up for group(s) {usable_groups}"
        )

    # ----------------------------------------------------- 5. max_rows
    # Applied BEFORE target construction: the labels must resolve inside the
    # window the dataset actually contains, so the tail is honestly NaN.
    if max_rows is not None:
        limit = int(max_rows)
        if limit < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")
        if limit < len(features):
            drop_report["max_rows_truncated"] = int(len(features) - limit)
            logger.info(
                "V3 dataset %s: max_rows=%d kept the most recent %d of %d rows (applied before "
                "target construction, so the label tail is NaN)",
                symbol,
                limit,
                limit,
                len(features),
            )
            features = features.iloc[-limit:]

    # ------------------------------------------------------ 6. targets
    close = price.reindex(features.index)
    targets = build_future_returns(close, ladder, price_column=price_column)

    frame = pd.concat([features, targets.frame], axis=1)
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC", name="timestamp")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:  # pragma: no cover
        raise V3DatasetError("assembled frame is not a sorted, unique time series")

    target_columns = [horizon.target_column() for horizon in ladder]
    target_coverage = targets.coverage_table()
    target_coverage.insert(0, "symbol", symbol)

    dataset = V3Dataset(
        symbol=symbol,
        frame=frame,
        horizons=ladder,
        feature_columns=columns,
        target_columns=target_columns,
        feature_coverage=_feature_coverage(features, columns),
        target_coverage=target_coverage,
        drop_report=drop_report,
        missing_sources=missing_sources,
        feature_groups=tuple(usable_groups),
        max_rows=None if max_rows is None else int(max_rows),
    )

    accounted = len(dataset) + int(sum(drop_report.values()))
    if accounted != n_input:  # pragma: no cover - guards the drop report's contract
        raise V3DatasetError(
            f"drop report does not account for every row: {len(dataset)} kept + "
            f"{sum(drop_report.values())} dropped != {n_input} input rows"
        )

    logger.info(
        "V3 dataset %s: %d rows, %d features (%s), %d horizon(s) | dropped %s | missing sources: %s",
        symbol,
        len(dataset),
        len(columns),
        ", ".join(usable_groups),
        len(ladder),
        {k: v for k, v in drop_report.items() if v} or "nothing",
        list(missing_sources) or "none",
    )
    return dataset
