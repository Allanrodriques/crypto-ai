"""Project-wide configuration and path resolution.

Every tunable value in this project lives in ``config/config.yaml`` and is
accessed through :class:`Config`.  Nothing in ``src/`` hard-codes a period, a
threshold, a ratio or a random seed: they are always read from the config
object that is threaded through the pipeline.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

#: Repository root (the directory that contains ``config/``, ``src/``, ``data/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or internally inconsistent."""


@dataclass(frozen=True)
class Paths:
    """Resolved output directories.

    All of these can be overridden by the ``paths:`` block of the YAML file.
    Relative paths are interpreted against :data:`PROJECT_ROOT`.
    """

    root: Path
    raw_dir: Path
    processed_dir: Path
    predictions_dir: Path
    models_dir: Path
    reports_dir: Path
    metrics_dir: Path
    plots_dir: Path
    backtests_dir: Path
    notebooks_dir: Path

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], root: Path = PROJECT_ROOT) -> "Paths":
        def _resolve(key: str, default: str) -> Path:
            value = mapping.get(key, default)
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else (root / path)

        reports_dir = _resolve("reports_dir", "reports")
        return cls(
            root=root,
            raw_dir=_resolve("raw_dir", "data/raw"),
            processed_dir=_resolve("processed_dir", "data/processed"),
            predictions_dir=_resolve("predictions_dir", "data/predictions"),
            models_dir=_resolve("models_dir", "models"),
            reports_dir=reports_dir,
            metrics_dir=_resolve("metrics_dir", str(reports_dir / "metrics")),
            plots_dir=_resolve("plots_dir", str(reports_dir / "plots")),
            backtests_dir=_resolve("backtests_dir", str(reports_dir / "backtests")),
            notebooks_dir=_resolve("notebooks_dir", "notebooks"),
        )

    def ensure(self) -> "Paths":
        """Create every directory that the pipeline writes into."""
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.predictions_dir,
            self.models_dir,
            self.metrics_dir,
            self.plots_dir,
            self.backtests_dir,
            self.notebooks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {f: str(getattr(self, f)) for f in self.__dataclass_fields__}


@dataclass
class Config:
    """In-memory view of ``config/config.yaml``.

    Sections are exposed as plain dicts (:attr:`raw`) plus a few typed
    convenience properties that the pipeline uses often.  The raw mapping is
    deep-copied on access so callers cannot accidentally mutate the config that
    is threaded into training runs.
    """

    raw: dict[str, Any]
    source_path: Path
    paths: Paths = field(init=False)

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None, root: Path = PROJECT_ROOT) -> None:
        object.__setattr__(self, "raw", copy.deepcopy(raw))
        object.__setattr__(self, "source_path", source_path or DEFAULT_CONFIG_PATH)
        object.__setattr__(self, "paths", Paths.from_mapping(raw.get("paths", {}) or {}, root=root))
        self.validate()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, root: Path = PROJECT_ROOT) -> "Config":
        """Load and validate the YAML configuration.

        Parameters
        ----------
        path:
            Path to the YAML file.  Defaults to ``config/config.yaml``.
        root:
            Project root used to resolve relative output paths.
        """
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not config_path.is_absolute():
            config_path = root / config_path
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)

        if not isinstance(loaded, dict):
            raise ConfigError(f"Configuration root must be a mapping, got {type(loaded).__name__}")

        return cls(raw=loaded, source_path=config_path, root=root)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new :class:`Config` with ``a.b.c=value`` style overrides applied."""
        merged = copy.deepcopy(self.raw)
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            node = merged
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ConfigError(f"Cannot descend into non-mapping key: {dotted_key!r}")
            node[parts[-1]] = value
        return Config(raw=merged, source_path=self.source_path, root=self.paths.root)

    def rebase(self, root: str | os.PathLike[str]) -> "Config":
        """Return a copy whose relative paths resolve against ``root``.

        ``paths.root`` cannot simply be reassigned to redirect output.  Every
        path in :class:`Paths` was already resolved to an absolute path when
        this config was constructed, so mutating ``root`` afterwards leaves them
        pointing at the original project.  That is precisely how the test suite
        came to overwrite ``data/processed/BTCUSDT_1h_dataset.parquet`` - the
        real frozen dataset - with a 374-row fixture, because the isolation
        fixture set ``paths.root`` to a temp directory and believed it had
        worked.

        Rebasing *rebuilds* the paths instead, so a caller that wants its
        artefacts somewhere else actually gets them there.
        """
        return Config(raw=self.raw, source_path=self.source_path, root=Path(root))

    # -------------------------------------------------------------- sections

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def target(self) -> dict[str, Any]:
        return self.raw["target"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw.get("validation", {})

    @property
    def cv(self) -> dict[str, Any]:
        return self.raw["cv"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def horizon_candles(self) -> int:
        return int(self.target["horizon_candles"])

    @property
    def threshold(self) -> float:
        return float(self.target["threshold"])

    @property
    def symbol(self) -> str:
        return str(self.data["symbol"]).upper()

    @property
    def interval(self) -> str:
        return str(self.data["interval"])

    @property
    def random_state(self) -> int:
        return int(self.model["random_state"])

    # ------------------------------------------------------------- validate

    _REQUIRED_SECTIONS = ("data", "target", "features", "model", "split", "cv", "backtest")

    def validate(self) -> "Config":
        missing = [s for s in self._REQUIRED_SECTIONS if s not in self.raw]
        if missing:
            raise ConfigError(f"Missing required config section(s): {', '.join(missing)}")

        data = self.raw["data"]
        for key in ("symbol", "interval", "start_date"):
            if key not in data:
                raise ConfigError(f"Missing required config key: data.{key}")

        if int(self.raw["target"]["horizon_candles"]) < 1:
            raise ConfigError("target.horizon_candles must be >= 1")
        if float(self.raw["target"]["threshold"]) < 0:
            raise ConfigError("target.threshold must be >= 0")

        ratios = [
            float(self.raw["split"]["train_ratio"]),
            float(self.raw["split"]["validation_ratio"]),
            float(self.raw["split"]["test_ratio"]),
        ]
        if any(r <= 0 for r in ratios):
            raise ConfigError("split ratios must all be > 0")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ConfigError(f"split ratios must sum to 1.0, got {sum(ratios):.6f}")

        if int(self.raw["cv"]["n_splits"]) < 2:
            raise ConfigError("cv.n_splits must be >= 2")

        if not 0 < float(self.raw["backtest"]["probability_threshold"]) < 1:
            raise ConfigError("backtest.probability_threshold must be strictly between 0 and 1")
        if float(self.raw["backtest"]["transaction_cost_bps"]) < 0:
            raise ConfigError("backtest.transaction_cost_bps must be >= 0")

        return self

    def fingerprint(self) -> dict[str, Any]:
        """Config values that materially change a result (used in model metadata)."""
        return {
            "config_file": str(self.source_path),
            "symbol": self.symbol,
            "interval": self.interval,
            "start_date": str(self.data["start_date"]),
            "end_date": self.data.get("end_date"),
            "target": dict(self.target),
            "features": dict(self.features),
            "split": dict(self.split),
            "cv": dict(self.cv),
            "random_state": self.random_state,
            "backtest": dict(self.backtest),
        }


_CONFIG_ENV_VAR = "CRYPTO_ML_CONFIG"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load the config, honouring the ``CRYPTO_ML_CONFIG`` env var override."""
    return Config.load(path or os.environ.get(_CONFIG_ENV_VAR) or None)
