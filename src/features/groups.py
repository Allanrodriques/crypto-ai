"""V2 feature-group registry.

The registry is the single source of truth for *which features belong to which
information source*.  Experiments select groups; the dataset builder expands
groups into columns.  Nothing in an experiment config names an individual
feature, so adding a feature to a group automatically flows into every
experiment that includes the group, and no experiment can silently drift away
from the baseline definition.

Groups
------
``technical``
    The V1 baseline set - returns, moving averages, MA relationships, momentum,
    volatility, Bollinger, volume and candle structure from OHLCV.  This group
    is immutable: EXP-00 is defined as exactly this group and is never edited.
``derivatives``
    Funding rates, futures price/volume, futures-spot basis.
``sentiment``
    Fear & Greed level, changes, rolling statistics, regime bucket.
``microstructure``
    Trade counts, aggressive buy/sell volume, order-flow imbalance.
``context``
    Completed higher-timeframe (4h, 1d) aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.features.context import CONTEXT_DOCS, GROUP as CONTEXT_GROUP
from src.features.derivatives import DERIVATIVES_DOCS, GROUP as DERIVATIVES_GROUP
from src.features.microstructure import GROUP as MICRO_GROUP, MICROSTRUCTURE_DOCS
from src.features.sentiment import GROUP as SENTIMENT_GROUP, SENTIMENT_DOCS

#: Order matters: it determines column order in every dataset, which in turn
#: makes dataset hashes comparable between experiments.
GROUP_ORDER: tuple[str, ...] = (
    "technical",
    DERIVATIVES_GROUP,
    SENTIMENT_GROUP,
    MICRO_GROUP,
    CONTEXT_GROUP,
)

#: Human-readable description of each group, echoed into experiment reports.
GROUP_DESCRIPTIONS: dict[str, str] = {
    "technical": "V1 baseline: OHLCV-derived returns, moving averages, momentum, "
    "volatility, Bollinger bands, volume and candle structure",
    DERIVATIVES_GROUP: "Perpetual futures funding rates, futures price/volume and the "
    "futures-spot basis (open interest unavailable historically)",
    SENTIMENT_GROUP: "Daily Crypto Fear & Greed index: level, changes, rolling statistics "
    "and an ordinal regime bucket, with no directional prior",
    MICRO_GROUP: "Trade-flow microstructure from exchange trade counts and aggressive "
    "buy/sell volume (order-book state has no historical endpoint)",
    CONTEXT_GROUP: "Higher-timeframe (4h, 1d) context from completed candles only",
}

#: Groups whose availability is limited; surfaced in the data availability matrix.
GROUP_LIMITATIONS: dict[str, str] = {
    DERIVATIVES_GROUP: (
        "Open interest, long/short ratio and taker-ratio history are capped at ~30 days by "
        "Binance and are therefore excluded rather than reconstructed."
    ),
    MICRO_GROUP: (
        "Bid/ask spread and order-book depth have no historical endpoint; only trade-flow "
        "microstructure is reconstructable and only from kline fields."
    ),
}


@dataclass(frozen=True)
class FeatureGroup:
    """A named bundle of features plus its provenance."""

    name: str
    features: tuple[str, ...]
    docs: Mapping[str, str]
    description: str
    limitation: str | None = None

    @property
    def size(self) -> int:
        return len(self.features)


def _technical_group() -> FeatureGroup:
    """The immutable V1 baseline group, read from the V1 registry.

    Imported lazily so this module stays importable while V1 is being edited.
    """
    from src.features.feature_engineering import FEATURE_GROUPS, FEATURE_DOCS

    # V1 splits OHLCV features into eight internal groups; EXP-00 is the union
    # of all of them, which is precisely the V1 feature set.
    features = tuple(name for group in FEATURE_GROUPS for name in FEATURE_GROUPS[group])
    return FeatureGroup(
        name="technical",
        features=features,
        docs={name: FEATURE_DOCS[name] for name in features},
        description=GROUP_DESCRIPTIONS["technical"],
    )


_BUILDERS: dict[str, tuple[tuple[str, ...], Mapping[str, str]]] = {
    DERIVATIVES_GROUP: (tuple(DERIVATIVES_DOCS), DERIVATIVES_DOCS),
    SENTIMENT_GROUP: (tuple(SENTIMENT_DOCS), SENTIMENT_DOCS),
    MICRO_GROUP: (tuple(MICROSTRUCTURE_DOCS), MICROSTRUCTURE_DOCS),
    CONTEXT_GROUP: (tuple(CONTEXT_DOCS), CONTEXT_DOCS),
}

REGISTRY: dict[str, FeatureGroup] = {"technical": _technical_group()}
for _name, (_features, _docs) in _BUILDERS.items():
    REGISTRY[_name] = FeatureGroup(
        name=_name,
        features=_features,
        docs=_docs,
        description=GROUP_DESCRIPTIONS[_name],
        limitation=GROUP_LIMITATIONS.get(_name),
    )


def get_group(name: str) -> FeatureGroup:
    if name not in REGISTRY:
        raise KeyError(f"Unknown feature group {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def feature_columns(groups: list[str] | tuple[str, ...]) -> list[str]:
    """Expand group names into an ordered, de-duplicated column list."""
    seen: dict[str, None] = {}
    for group in groups:
        for name in get_group(group).features:
            seen.setdefault(name, None)
    return list(seen)


def feature_docs(groups: list[str] | tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in groups:
        out.update(get_group(group).docs)
    return out


def group_sizes() -> dict[str, int]:
    return {name: group.size for name, group in REGISTRY.items()}


def known_features() -> set[str]:
    return {name for group in REGISTRY.values() for name in group.features}


def validate_groups(groups: list[str] | tuple[str, ...]) -> None:
    """Fail loudly on an unknown group - a typo must not silently drop features."""
    unknown = [g for g in groups if g not in REGISTRY]
    if unknown:
        raise KeyError(f"Unknown feature group(s) {unknown}; known: {sorted(REGISTRY)}")
