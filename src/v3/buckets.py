"""V3 user-facing feature buckets.

V2 groups its 107 features by *information source* (``derivatives``,
``sentiment``, ``microstructure``, ``context``).  That is the right axis for a
provenance argument and the wrong axis for someone asking "what am I turning
on?".  A V2 group mixes a price level with a funding rate and a trade-count
intensity, so "turn on ``technical``" in V2 quietly means *all 39* of them -
including three volume columns and seventeen price columns that most people
think of as separate things.

V3 re-buckets the *same* 107 features into six buckets that line up with how a
trader describes a setup, and adds nothing and drops nothing:

``price``
    Returns, moving averages and moving-average relationships - the series that
    say where price is relative to its own history.
``volume``
    The three volume columns, split out because volume is the most common
    single thing to ablate.
``technical``
    Momentum, volatility, Bollinger and candle structure, plus the eleven V2
    ``context`` columns: completed 4h/1d aggregates *are* technical analysis,
    and V2 only separated them because they were built by a different module.
``derivatives`` / ``sentiment`` / ``microstructure``
    Passed through unchanged from V2.

Why the membership is written out rather than derived
-----------------------------------------------------
The three pass-through buckets are read straight from the V2 registry, but
``price``/``volume``/``technical`` are literal tuples.  Deriving them with
prefix rules ("everything in V2 ``technical`` that is not a return or a moving
average") would be less code and would also mean that adding one feature to V2
silently changes what a V3 bucket means - an ablation labelled "price" would
quietly be testing something else.  Literals make the V3 taxonomy a reviewable
artefact, and :func:`_audit_partition` then fails loudly at import if the
literals stop covering the V2 registry exactly.  A taxonomy that cannot silently
drift is worth the duplication.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import pandas as pd

from src.features.groups import REGISTRY
from src.utils import get_logger

logger = get_logger("v3.buckets")

#: The six V3 buckets, in canonical order.  Order is the column order of every
#: V3 dataset and of every report table, so it is part of the contract.
BUCKET_NAMES: tuple[str, ...] = (
    "price",
    "technical",
    "volume",
    "derivatives",
    "sentiment",
    "microstructure",
)

#: Enabling every bucket is the default so that omitting the block from a
#: config can never mean "no features" - that is a zero-column dataset and a
#: mystifying crash three layers down.
DEFAULT_BUCKETS: tuple[str, ...] = BUCKET_NAMES

BUCKET_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "price": "Returns, moving averages and moving-average relationships",
        "technical": "Momentum, volatility, Bollinger bands, candle structure and "
        "completed 4h/1d context",
        "volume": "Volume level, relative volume and volume change",
        "derivatives": "Perpetual futures funding, futures price/volume and futures-spot basis",
        "sentiment": "Crypto Fear & Greed level, changes, rolling statistics and regime",
        "microstructure": "Trade counts, trade size and aggressive buy/sell order flow",
    }
)

#: Sizes the literals above must have.  A feature moved from one bucket to
#: another still partitions the registry, so the union check alone would not
#: notice; the size check does.
EXPECTED_BUCKET_SIZES: Mapping[str, int] = MappingProxyType(
    {"price": 17, "technical": 30, "volume": 3, "derivatives": 23, "sentiment": 12, "microstructure": 22}
)

# Written in reading order (horizons, then averages, then relationships) so the
# taxonomy can be reviewed by eye; sorted once below so that every accessor
# agrees on one order.
_PRICE: tuple[str, ...] = (
    "return_1h",
    "return_3h",
    "return_6h",
    "return_12h",
    "return_24h",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_12",
    "ema_26",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma20_vs_sma50",
    "sma50_vs_sma200",
)

_VOLUME: tuple[str, ...] = (
    "volume_sma_20",
    "volume_ratio",
    "volume_change",
)

_TECHNICAL: tuple[str, ...] = (
    # momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "roc_12",
    # volatility
    "atr_14",
    "atr_percent",
    "rolling_volatility_20",
    "rolling_volatility_50",
    # bollinger
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    # candle structure
    "candle_body",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "body_to_range",
    # V2 `context`: completed 4h/1d aggregates, which are technical analysis
    # that V2 separated only because a different module built them.
    "ctx_4h_return_1",
    "ctx_4h_return_3",
    "ctx_4h_ema_ratio",
    "ctx_4h_volatility",
    "ctx_4h_volume_ratio",
    "ctx_1d_return_1",
    "ctx_1d_return_3",
    "ctx_1d_ema_ratio",
    "ctx_1d_volatility",
    "ctx_htf_bias",
    "ctx_vol_regime_ratio",
)

#: bucket -> its features, sorted.  A read-only mapping: it is the answer to
#: "what is in this bucket", so a caller mutating it would corrupt the
#: partition guarantee that :func:`_audit_partition` establishes.
BUCKETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "price": tuple(sorted(_PRICE)),
        "technical": tuple(sorted(_TECHNICAL)),
        "volume": tuple(sorted(_VOLUME)),
        "derivatives": tuple(sorted(REGISTRY["derivatives"].features)),
        "sentiment": tuple(sorted(REGISTRY["sentiment"].features)),
        "microstructure": tuple(sorted(REGISTRY["microstructure"].features)),
    }
)

#: feature -> bucket.  The reverse index is what makes ``bucket_for_feature``
#: O(1) and what lets a misfiled feature be reported with the bucket it
#: actually belongs to, instead of only "unknown".
BUCKET_INDEX: Mapping[str, str] = MappingProxyType(
    {feature: bucket for bucket in BUCKET_NAMES for feature in BUCKETS[bucket]}
)

#: feature -> the V2 group it came from, kept so a bucket can be traced back to
#: its provenance in a report without the reader loading the V2 registry.
V2_GROUP_INDEX: Mapping[str, str] = MappingProxyType(
    {feature: group for group, spec in REGISTRY.items() for feature in spec.features}
)

#: All 107 features in canonical order: bucket order, sorted within a bucket.
ALL_FEATURES: tuple[str, ...] = tuple(
    feature for bucket in BUCKET_NAMES for feature in BUCKETS[bucket]
)


def _audit_partition() -> None:
    """Fail at import if the bucket literals stop covering the V2 registry.

    Three ways this can be wrong, all of them silent without this check: a
    feature in the V2 registry that no bucket claims (it would vanish from every
    dataset), a feature claimed by two buckets (it would be fed to the model
    twice, once from each), and a bucket whose size drifted because a literal
    was edited.  All three are reported together, so one import tells the whole
    story instead of one failure per run.
    """
    registry = {feature for spec in REGISTRY.values() for feature in spec.features}
    assigned = list(ALL_FEATURES)

    duplicates = sorted(name for name, n in Counter(assigned).items() if n > 1)
    unassigned = sorted(registry - set(assigned))
    unknown = sorted(set(assigned) - registry)
    wrong_sizes = {
        bucket: (len(BUCKETS[bucket]), EXPECTED_BUCKET_SIZES[bucket])
        for bucket in BUCKET_NAMES
        if len(BUCKETS[bucket]) != EXPECTED_BUCKET_SIZES[bucket]
    }

    problems: list[str] = []
    if duplicates:
        problems.append(f"features in more than one bucket: {duplicates}")
    if unassigned:
        problems.append(f"registry features in no bucket: {unassigned}")
    if unknown:
        problems.append(f"bucketed features absent from the V2 registry: {unknown}")
    if wrong_sizes:
        problems.append(
            "bucket sizes drifted from EXPECTED_BUCKET_SIZES "
            + ", ".join(f"{b}: {got} != {want}" for b, (got, want) in sorted(wrong_sizes.items()))
        )
    if problems:
        raise ValueError("V3 buckets do not partition the V2 feature registry: " + "; ".join(problems))

    logger.debug("V3 buckets partition the V2 registry: %d features", len(assigned))


_audit_partition()


@dataclass(frozen=True)
class BucketSelection:
    """A validated bucket selection resolved to a flat, ordered feature list.

    Frozen because a selection is the answer to a question that must not change
    after it has been recorded in a report: the same experiment run twice has
    to train on the same columns in the same order.
    """

    buckets: tuple[str, ...]
    features: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Feature count per selected bucket, in canonical bucket order."""
        return {bucket: len(BUCKETS[bucket]) for bucket in self.buckets}


def _check_sequence(value: object, bucket: str) -> Sequence[str]:
    """Reject a bare string where a list of names belongs.

    ``{"price": "return_1h"}`` is a YAML-shaped config typo, and iterating it
    yields twelve single letters - each of which is then reported as an unknown
    feature, which points the reader at the registry instead of at the config.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"Bucket {bucket!r} must map to a sequence of feature names, got the string {value!r}"
        )
    if not isinstance(value, Sequence):
        raise ValueError(
            f"Bucket {bucket!r} must map to a sequence of feature names, got {type(value).__name__}"
        )
    return value


def _resolve(enabled: Mapping[str, Sequence[str]] | None) -> BucketSelection:
    """Validate a bucket selection and resolve it to ordered feature names.

    Every rejection this project makes about a config is raised here, so a typo
    surfaces at load time with the offending name in the message rather than as
    a missing column three layers downstream.
    """
    if not enabled:
        return BucketSelection(buckets=DEFAULT_BUCKETS, features=ALL_FEATURES)

    unknown_buckets = [bucket for bucket in enabled if bucket not in BUCKETS]
    if unknown_buckets:
        raise ValueError(
            f"Unknown bucket(s) {sorted(unknown_buckets)}; known buckets: {list(BUCKET_NAMES)}"
        )

    buckets = tuple(bucket for bucket in BUCKET_NAMES if bucket in enabled)
    seen: dict[str, str] = {}
    for bucket in buckets:
        for feature in _check_sequence(enabled[bucket], bucket):
            if feature not in BUCKET_INDEX:
                raise ValueError(
                    f"Unknown feature {feature!r} in bucket {bucket!r}; "
                    f"it is not one of the {len(BUCKET_INDEX)} registered features"
                )
            actual = BUCKET_INDEX[feature]
            if actual != bucket:
                raise ValueError(
                    f"Feature {feature!r} is listed under bucket {bucket!r} but belongs "
                    f"to {actual!r}"
                )
            if feature in seen:
                raise ValueError(
                    f"Duplicate feature {feature!r}: listed under both "
                    f"{seen[feature]!r} and {bucket!r}"
                )
            seen[feature] = bucket

    if not seen:
        raise ValueError(
            f"Empty selection: buckets {list(buckets)} name no features. "
            "A model needs at least one; omit the selection entirely to use all buckets."
        )

    features = tuple(
        feature
        for bucket in buckets
        for feature in sorted(name for name, home in seen.items() if home == bucket)
    )
    return BucketSelection(buckets=buckets, features=features)


def bucket_for_feature(name: str) -> str:
    """The V3 bucket a feature belongs to, e.g. ``"price"`` for ``"sma_20"``."""
    try:
        return BUCKET_INDEX[name]
    except KeyError:
        raise ValueError(
            f"Unknown feature {name!r}; known features are the {len(BUCKET_INDEX)} "
            f"entries of src.v3.buckets.BUCKETS"
        ) from None


def features_in_bucket(bucket: str) -> list[str]:
    """Feature names in one bucket, sorted.

    Raises ``ValueError`` for an unknown bucket: a silent empty list here would
    turn "I mistyped the bucket" into a dataset with 30 fewer columns.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"Unknown bucket {bucket!r}; known buckets: {list(BUCKET_NAMES)}")
    return list(BUCKETS[bucket])


def resolve_features(enabled: Mapping[str, Sequence[str]] | None) -> list[str]:
    """Expand a bucket selection into the ordered feature list a model trains on.

    Parameters
    ----------
    enabled:
        ``bucket -> feature names``.  ``None`` or an empty mapping means "no
        restriction" and yields every feature in :data:`ALL_FEATURES`.  Any
        other mapping restricts the result to exactly the named features, in
        :data:`BUCKET_NAMES` order and sorted within a bucket, so the column
        order does not depend on the order the config happened to list them in.

    Raises
    ------
    ValueError
        On an unknown bucket, a feature that is not in the registry, a feature
        listed under a bucket it does not belong to, a repeated feature, or a
        selection that resolves to no features at all.
    """
    return list(_resolve(enabled).features)


def validate_selection(enabled: Mapping[str, Sequence[str]] | None) -> None:
    """Check a bucket selection without using it.  No-op when it is valid.

    Config loading calls this so an ablation is rejected at the boundary, with
    the offending name in the message, instead of producing a dataset that is
    quietly short the columns the report claims to have used.
    """
    _resolve(enabled)
    return None


def describe_buckets() -> dict[str, int]:
    """Feature count per bucket, in canonical order.  Sums to 107."""
    return {bucket: len(BUCKETS[bucket]) for bucket in BUCKET_NAMES}


def bucket_frame() -> pd.DataFrame:
    """One row per feature, tracing each through V2 and into V3.

    Columns ``feature``, ``v2_group``, ``v3_bucket``, sorted by bucket order
    then feature.  This is the audit table for a report: it is how a reader
    checks that no V2 feature was dropped and that the context columns really
    did land in ``technical``.

    Built fresh on every call from immutable tuples rather than cached, so a
    caller that sorts or mutates the result cannot corrupt the next one.
    """
    rows = [
        (feature, V2_GROUP_INDEX[feature], bucket)
        for bucket in BUCKET_NAMES
        for feature in BUCKETS[bucket]
    ]
    return pd.DataFrame(rows, columns=["feature", "v2_group", "v3_bucket"])


if __name__ == "__main__":  # pragma: no cover - manual audit, not a test
    from src.features.groups import known_features

    frame = bucket_frame()
    v2_groups = frame.groupby("v3_bucket")["v2_group"].agg(lambda s: ", ".join(sorted(set(s))))

    print("V3 feature buckets")
    print(f"  {'bucket':<14} {'count':>5}  V2 source group(s)")
    for bucket, count in describe_buckets().items():
        print(f"  {bucket:<14} {count:>5}  {v2_groups[bucket]}")
    print(f"  {'total':<14} {len(frame):>5}  (V2 registry: {sum(len(s.features) for s in REGISTRY.values())})")

    assert not frame["feature"].duplicated().any(), "a feature is in more than one bucket"
    assert set(frame["feature"]) == known_features(), "the buckets do not cover the V2 registry"
    assert describe_buckets() == dict(EXPECTED_BUCKET_SIZES), "bucket sizes drifted"
    print("OK: 107 features, each in exactly one bucket.")
