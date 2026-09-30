"""Experiment specification and registry.

An experiment is a *complete, reproducible description* of one research
question: which information sources feed which feature groups, over which
target, into which model.  Nothing about an experiment is implicit, and the
spec is serialised into every result artefact so a run can be reproduced from
its own output.

The five canonical experiments are defined in :data:`EXPERIMENTS` and map
directly to the research questions Q1-Q5:

============================  ==========================================
experiment                    question
============================  ==========================================
``EXP-00-OHLCV-TECHNICAL``    Q1  do price/technical features have signal?
``EXP-01-DERIVATIVES``        Q2  does derivatives positioning add value?
``EXP-02-SENTIMENT``          Q3  does sentiment add value?
``EXP-03-MICROSTRUCTURE``     Q4  does trade-flow microstructure add value?
``EXP-04-COMBINED``           Q5  does combining sources help?
============================  ==========================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dataset.factory import SplitSpec, TargetSpec
from src.features.groups import validate_groups
from src.utils import get_logger

logger = get_logger("experiments.spec")


class ExperimentError(ValueError):
    """Raised for a malformed or inconsistent experiment specification."""


@dataclass(frozen=True)
class ModelSpec:
    """Which model(s) to train, and with what override."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | str) -> "ModelSpec":
        if isinstance(data, str):
            return cls(name=data)
        return cls(name=str(data["name"]), params=dict(data.get("params", {}) or {}))


@dataclass(frozen=True)
class ExperimentSpec:
    """A fully-specified, reproducible research experiment."""

    name: str
    data_sources: tuple[str, ...]
    feature_groups: tuple[str, ...]
    target: TargetSpec
    model: ModelSpec
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    split: SplitSpec = field(default_factory=SplitSpec)
    description: str = ""
    notes: str = ""
    expected_feature_count: int | None = None
    unavailable_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ExperimentError("experiment name must not be empty")
        validate_groups(self.feature_groups)
        if not self.feature_groups:
            raise ExperimentError("an experiment must include at least one feature group")

    # ---------------------------------------------------------------- io

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        """Parse the YAML-shaped mapping documented in the V2 specification."""
        if "experiment" in data:
            block = data["experiment"]
            name = str(block["name"])
            sources = tuple(str(s) for s in block.get("data_sources", data.get("data_sources", ["binance_spot"])))
            groups = tuple(str(g) for g in block.get("feature_groups", data.get("feature_groups", [])))
            target_block = data.get("target", {})
            model_block = data.get("model", {"name": "xgboost"})
            split_block = data.get("split", {})
            meta = block
        else:
            name = str(data["name"])
            sources = tuple(str(s) for s in data.get("data_sources", ["binance_spot"]))
            groups = tuple(str(g) for g in data.get("feature_groups", []))
            target_block = data.get("target", {})
            model_block = data.get("model", {"name": "xgboost"})
            split_block = data.get("split", {})
            meta = data

        target = TargetSpec(
            horizon_candles=int(target_block.get("horizon_candles", 6)),
            threshold=float(target_block.get("threshold", 0.005)),
            mode=str(target_block.get("mode", "binary")),
        )
        split = SplitSpec(
            train_ratio=float(split_block.get("train_ratio", 0.70)),
            validation_ratio=float(split_block.get("validation_ratio", 0.15)),
            test_ratio=float(split_block.get("test_ratio", 0.15)),
        )
        return cls(
            name=name,
            data_sources=sources,
            feature_groups=groups,
            target=target,
            model=ModelSpec.from_mapping(model_block),
            symbol=str(data.get("symbol", "BTCUSDT")).upper(),
            interval=str(data.get("interval", "1h")),
            split=split,
            description=str(meta.get("description", "")),
            notes=str(meta.get("notes", "")),
            unavailable_features=tuple(str(f) for f in meta.get("unavailable_features", [])),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "symbol": self.symbol,
            "interval": self.interval,
            "data_sources": list(self.data_sources),
            "feature_groups": list(self.feature_groups),
            "target": self.target.to_dict(),
            "model": {"name": self.model.name, "params": self.model.params},
            "split": {**self.split.to_dict(), "purge_candles": self.target.horizon_candles},
            "notes": self.notes,
            "unavailable_features": list(self.unavailable_features),
        }

    def with_(self, **changes: Any) -> "ExperimentSpec":
        """Derive a variant (e.g. a different target) while keeping provenance."""
        return replace(self, **changes)

    @property
    def experiment_id(self) -> str:
        """Stable identifier, e.g. ``EXP-00-OHLCV-TECHNICAL``."""
        return self.name

    @property
    def hypothesis(self) -> str:
        """The testable claim this experiment makes, or its description."""
        return self.notes or self.description

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def backtest_threshold(self) -> float:
        """Fixed probability threshold for the simulated strategy.

        Sourced from the config so every experiment uses one number, and never
        from test results.  A threshold is the single most effective way to
        manufacture a good-looking backtest, so its provenance is recorded
        alongside every result.
        """
        return float(getattr(self, "_backtest_threshold", 0.60))

    def with_backtest_threshold(self, threshold: float) -> "ExperimentSpec":
        clone = replace(self)
        object.__setattr__(clone, "_backtest_threshold", float(threshold))
        return clone

    @property
    def slug(self) -> str:
        """Filesystem-safe directory name, derived from the experiment id.

        Derived from the *id* rather than the display name so that renaming a
        human-readable title never silently changes where a run's artefacts
        live, which would split a history of comparable results across folders.
        """
        raw = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.experiment_id).lower()
        return raw.strip("-")


# --------------------------------------------------------------------------- registry

#: Unavailable-by-design features, recorded so the report can explain absences.
OI_UNAVAILABLE = (
    "open_interest",
    "open_interest_change",
    "open_interest_return",
    "open_interest_vs_price",
)
BOOK_UNAVAILABLE = (
    "bid_ask_spread",
    "bid_volume",
    "ask_volume",
    "order_book_imbalance",
    "order_imbalance_1m",
    "order_imbalance_5m",
    "order_imbalance_15m",
)

EXPERIMENTS: dict[str, ExperimentSpec] = {}


def _register(spec: ExperimentSpec) -> ExperimentSpec:
    EXPERIMENTS[spec.name] = spec
    return spec


def _default_target() -> TargetSpec:
    return TargetSpec(horizon_candles=6, threshold=0.005, mode="binary")


_register(
    ExperimentSpec(
        name="EXP-00-OHLCV-TECHNICAL",
        data_sources=("binance_spot",),
        feature_groups=("technical",),
        target=_default_target(),
        model=ModelSpec("xgboost"),
        description=(
            "Q1 baseline. The V1 feature set, unchanged: returns, moving averages, MA "
            "relationships, RSI/MACD, ATR/volatility, Bollinger bands, volume and candle "
            "structure from spot OHLCV. Every other experiment is compared against this and "
            "this experiment is never modified."
        ),
        notes="Immutable reference point. Feature set is pinned by tests/test_features.py.",
    )
)

_register(
    ExperimentSpec(
        name="EXP-01-DERIVATIVES",
        data_sources=("binance_spot", "binance_futures", "binance_funding"),
        feature_groups=("technical", "derivatives"),
        target=_default_target(),
        model=ModelSpec("xgboost"),
        description=(
            "Q2. Adds perpetual futures funding rates, futures price/volume and the "
            "futures-spot basis on top of the baseline."
        ),
        notes=(
            "Open interest, long/short ratio and taker-ratio history are capped at ~30 days "
            "by Binance and are excluded rather than reconstructed or forward-filled."
        ),
        unavailable_features=OI_UNAVAILABLE,
    )
)

_register(
    ExperimentSpec(
        name="EXP-02-SENTIMENT",
        data_sources=("binance_spot", "fear_greed"),
        feature_groups=("technical", "sentiment"),
        target=_default_target(),
        model=ModelSpec("xgboost"),
        description=(
            "Q3. Adds the daily Crypto Fear & Greed index. Sentiment is supplied as level, "
            "change, rolling statistics and an ordinal regime bucket with no directional prior "
            "baked in, so the model can only find a relationship that the data supports."
        ),
        notes="A reading for day D is treated as knowable from D+1 00:00 UTC.",
    )
)

_register(
    ExperimentSpec(
        name="EXP-03-MICROSTRUCTURE",
        data_sources=("binance_spot",),
        feature_groups=("technical", "microstructure"),
        target=_default_target(),
        model=ModelSpec("xgboost"),
        description=(
            "Q4. Adds trade-flow microstructure derived from exchange trade counts and "
            "aggressive buy/sell volume, both of which exist for the entire history."
        ),
        notes=(
            "Order-book spread and depth have no historical endpoint on Binance. Book-shape "
            "features are therefore absent by design rather than reconstructed from snapshots."
        ),
        unavailable_features=BOOK_UNAVAILABLE,
    )
)

_register(
    ExperimentSpec(
        name="EXP-04-COMBINED",
        data_sources=("binance_spot", "binance_futures", "binance_funding", "fear_greed"),
        feature_groups=("technical", "derivatives", "sentiment", "microstructure", "context"),
        target=_default_target(),
        model=ModelSpec("xgboost"),
        description=(
            "Q5. All sources that are historically valid, combined: technical + derivatives + "
            "sentiment + microstructure + completed higher-timeframe context."
        ),
        notes=(
            "Higher-timeframe context uses only completed 4h/1d candles, shifted by one full "
            "bucket so no in-progress candle is ever read."
        ),
        unavailable_features=OI_UNAVAILABLE + BOOK_UNAVAILABLE,
    )
)


def get_experiment(name: str) -> ExperimentSpec:
    if name not in EXPERIMENTS:
        raise ExperimentError(f"Unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def baseline() -> ExperimentSpec:
    return EXPERIMENTS["EXP-00-OHLCV-TECHNICAL"]


def all_experiments() -> list[ExperimentSpec]:
    return [EXPERIMENTS[name] for name in sorted(EXPERIMENTS)]


def experiments_using(group: str) -> list[ExperimentSpec]:
    return [spec for spec in all_experiments() if group in spec.feature_groups]
