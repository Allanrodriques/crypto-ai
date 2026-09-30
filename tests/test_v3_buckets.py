"""Tests for the V3 user-facing feature buckets.

The property that matters here is completeness, not tidiness: the 107 V2
features must land in exactly one of the six V3 buckets, because a feature in
no bucket disappears from every dataset and one in two buckets is handed to the
model twice.  The individual accessors are then tested against that partition
rather than against hand-copied lists, so a legitimate re-bucketing in the
module does not need a matching edit here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.groups import REGISTRY
from src.features.groups import known_features as v2_known_features
from src.v3.buckets import (
    ALL_FEATURES,
    BUCKET_INDEX,
    BUCKET_NAMES,
    BUCKETS,
    DEFAULT_BUCKETS,
    bucket_for_feature,
    bucket_frame,
    describe_buckets,
    features_in_bucket,
    resolve_features,
    validate_selection,
)

TOTAL_FEATURES = 107

# One known feature per bucket, with the V2 group it came from.  Probes both
# directions of the re-bucketing: `ctx_htf_bias` moved from V2 `context` into
# V3 `technical`, and `volume_ratio` moved out of V2 `technical` into `volume`.
PROBES: tuple[tuple[str, str], ...] = (
    ("return_24h", "price"),
    ("sma20_vs_sma50", "price"),
    ("rsi_14", "technical"),
    ("ctx_htf_bias", "technical"),
    ("bb_position", "technical"),
    ("volume_ratio", "volume"),
    ("futures_spot_basis", "derivatives"),
    ("fear_greed_regime", "sentiment"),
    ("flow_price_divergence", "microstructure"),
)


# ------------------------------------------------------------------ partition


def test_buckets_partition_the_v2_registry() -> None:
    assigned = [feature for bucket in BUCKET_NAMES for feature in BUCKETS[bucket]]
    assert len(assigned) == TOTAL_FEATURES
    assert len(set(assigned)) == TOTAL_FEATURES, "a feature is claimed by more than one bucket"
    assert set(assigned) == v2_known_features(), "the buckets do not cover the V2 registry"


def test_no_registry_feature_is_unassigned() -> None:
    per_bucket = {feature: bucket for bucket in BUCKET_NAMES for feature in BUCKETS[bucket]}
    assert sorted(v2_known_features() - set(per_bucket)) == []


def test_bucket_sizes_match_the_documented_taxonomy() -> None:
    assert describe_buckets() == {
        "price": 17,
        "technical": 30,
        "volume": 3,
        "derivatives": 23,
        "sentiment": 12,
        "microstructure": 22,
    }
    assert sum(describe_buckets().values()) == TOTAL_FEATURES


def test_pass_through_buckets_are_identical_to_v2() -> None:
    """Derivatives, sentiment and microstructure are not re-bucketed at all."""
    for bucket in ("derivatives", "sentiment", "microstructure"):
        assert set(BUCKETS[bucket]) == set(REGISTRY[bucket].features)
    assert set(BUCKETS["technical"]) >= set(REGISTRY["context"].features)


def test_bucket_names_and_default_buckets_are_in_canonical_order() -> None:
    assert BUCKET_NAMES == (
        "price",
        "technical",
        "volume",
        "derivatives",
        "sentiment",
        "microstructure",
    )
    assert DEFAULT_BUCKETS == BUCKET_NAMES
    assert list(BUCKETS) == list(BUCKET_NAMES)


# --------------------------------------------------------------- accessors


@pytest.mark.parametrize(("feature", "bucket"), PROBES)
def test_bucket_for_feature_known_names(feature: str, bucket: str) -> None:
    assert bucket_for_feature(feature) == bucket


def test_bucket_for_feature_rejects_garbage() -> None:
    for bad in ("not_a_feature", "", "SMA_20", "return_25h"):
        with pytest.raises(ValueError, match="Unknown feature"):
            bucket_for_feature(bad)


def test_bucket_index_is_the_inverse_of_buckets() -> None:
    assert set(BUCKET_INDEX) == set(ALL_FEATURES)
    for bucket, features in BUCKETS.items():
        for feature in features:
            assert BUCKET_INDEX[feature] == bucket


def test_features_in_bucket_is_sorted_and_raises_on_unknown() -> None:
    for bucket in BUCKET_NAMES:
        features = features_in_bucket(bucket)
        assert features == sorted(BUCKETS[bucket])
        assert features == sorted(set(features)), "no feature repeated inside a bucket"
    with pytest.raises(ValueError, match="Unknown bucket"):
        features_in_bucket("orderflow")
    with pytest.raises(ValueError, match="Unknown bucket"):
        features_in_bucket("")


# --------------------------------------------------------------- resolution


def test_resolve_none_returns_everything_in_canonical_order() -> None:
    assert resolve_features(None) == list(ALL_FEATURES)
    assert len(resolve_features(None)) == TOTAL_FEATURES
    assert resolve_features(None) == resolve_features({}), "an empty mapping is also 'no restriction'"


def test_resolve_none_orders_by_bucket_then_sorted_within_bucket() -> None:
    resolved = resolve_features(None)
    expected = [f for bucket in BUCKET_NAMES for f in sorted(BUCKETS[bucket])]
    assert resolved == expected
    for bucket in BUCKET_NAMES:
        first = resolved.index(features_in_bucket(bucket)[0])
        last = resolved.index(features_in_bucket(bucket)[-1])
        assert first < last


def test_resolve_restricts_to_the_requested_features() -> None:
    resolved = resolve_features({"price": ["sma_20", "return_1h"], "volume": ["volume_ratio"]})
    assert resolved == ["return_1h", "sma_20", "volume_ratio"]


def test_resolve_ignores_the_order_names_were_listed_in() -> None:
    forward = resolve_features({"price": ["return_1h", "sma_20"], "sentiment": ["fear_greed_value"]})
    backward = resolve_features({"sentiment": ["fear_greed_value"], "price": ["sma_20", "return_1h"]})
    assert forward == backward


def test_resolve_covers_every_bucket_when_each_names_one_feature() -> None:
    mapping = {bucket: [features_in_bucket(bucket)[0]] for bucket in BUCKET_NAMES}
    assert resolve_features(mapping) == [features_in_bucket(bucket)[0] for bucket in BUCKET_NAMES]


def test_resolve_rejects_an_unknown_bucket_naming_it() -> None:
    with pytest.raises(ValueError, match="orderflow"):
        resolve_features({"orderflow": ["return_1h"]})


def test_resolve_rejects_an_unknown_feature_naming_it() -> None:
    with pytest.raises(ValueError, match="sma_999"):
        resolve_features({"price": ["sma_999"]})


def test_resolve_rejects_a_feature_listed_under_the_wrong_bucket() -> None:
    with pytest.raises(ValueError, match="rsi_14"):
        resolve_features({"price": ["rsi_14"]})


def test_resolve_rejects_a_repeated_feature() -> None:
    with pytest.raises(ValueError, match="return_1h"):
        resolve_features({"price": ["return_1h", "return_1h"]})


# -------------------------------------------------------------- bucket frame


def test_bucket_frame_shape_and_labels() -> None:
    frame = bucket_frame()
    assert list(frame.columns) == ["feature", "v2_group", "v3_bucket"]
    assert len(frame) == TOTAL_FEATURES
    assert frame["feature"].nunique() == TOTAL_FEATURES
    assert set(frame["v3_bucket"]) == set(BUCKET_NAMES)
    assert set(frame["v2_group"]) == set(REGISTRY)


def test_bucket_frame_is_sorted_by_bucket_order_then_feature() -> None:
    frame = bucket_frame()
    rank = {bucket: i for i, bucket in enumerate(BUCKET_NAMES)}
    keys = list(zip(frame["v3_bucket"].map(rank), frame["feature"]))
    assert keys == sorted(keys)
    assert frame["feature"].tolist() == list(ALL_FEATURES)


def test_bucket_frame_rows_agree_with_buckets_and_the_v2_registry() -> None:
    frame = bucket_frame().set_index("feature")
    for bucket, features in BUCKETS.items():
        assert sorted(frame.index[frame["v3_bucket"] == bucket]) == sorted(features)
    for group, spec in REGISTRY.items():
        assert sorted(frame.index[frame["v2_group"] == group]) == sorted(spec.features)


def test_bucket_frame_records_the_refolding_of_v2_technical() -> None:
    """V2 `technical` is split three ways, and `context` joins `technical`."""
    frame = bucket_frame().set_index("feature")
    v2_technical = set(REGISTRY["technical"].features)
    folded = {bucket: set(features) for bucket, features in BUCKETS.items()}

    # Every V2 technical feature is claimed by exactly one V3 bucket, and the
    # three claims together are exactly the V2 group.
    claims = [folded[b] & v2_technical for b in ("price", "volume", "technical")]
    assert sum(len(c) for c in claims) == len(v2_technical)
    assert set().union(*claims) == v2_technical
    assert folded["technical"] & set(REGISTRY["context"].features) == set(REGISTRY["context"].features)
    assert frame.loc["rsi_14", "v2_group"] == "technical"
    assert frame.loc["ctx_htf_bias", "v2_group"] == "context"


def test_bucket_frame_is_a_fresh_object_each_call() -> None:
    """A caller that mutates the frame must not corrupt the next one."""
    frame = bucket_frame()
    frame.loc[0, "v3_bucket"] = "tampered"
    assert bucket_frame().loc[0, "v3_bucket"] != "tampered"


# ---------------------------------------------------------------- validation


def test_validate_selection_is_a_no_op_when_valid() -> None:
    assert validate_selection(None) is None
    assert validate_selection({}) is None
    assert validate_selection({"price": ["sma_20"]}) is None
    assert validate_selection({bucket: features_in_bucket(bucket) for bucket in BUCKET_NAMES}) is None


def test_validate_selection_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="Empty selection"):
        validate_selection({"price": []})
    with pytest.raises(ValueError, match="Empty selection"):
        validate_selection({bucket: [] for bucket in BUCKET_NAMES})


def test_validate_selection_rejects_a_duplicate_feature() -> None:
    with pytest.raises(ValueError, match="Duplicate feature 'sma_20'"):
        validate_selection({"price": ["sma_20", "sma_20"]})


def test_validate_selection_rejects_an_unknown_bucket() -> None:
    with pytest.raises(ValueError, match="Unknown bucket"):
        validate_selection({"sentiment ": ["fear_greed_value"]})


def test_validate_selection_rejects_an_unknown_feature() -> None:
    with pytest.raises(ValueError, match="ctx_4h_return_2"):
        validate_selection({"technical": ["ctx_4h_return_2"]})


def test_validate_selection_rejects_a_bare_string_instead_of_a_list() -> None:
    """A YAML-shaped typo must not be reported as twelve unknown features."""
    with pytest.raises(ValueError, match="sequence of feature names"):
        validate_selection({"price": "return_1h"})


def test_all_features_constant_is_consistent_with_resolution() -> None:
    assert ALL_FEATURES == tuple(resolve_features(None))
    assert list(BUCKETS) == list(BUCKET_NAMES)
    assert isinstance(bucket_frame(), pd.DataFrame)
