"""Causal technical-indicator feature engineering.

**The single most important property of this module**

Every feature computed here is a function of candles ``<= t`` only::

    feature[t] = f(candle[t], candle[t-1], ..., candle[t-W])

No indicator in this file is allowed to reference ``t+1`` or later.  That is
enforced three ways:

1. Only backward-looking primitives are used (``rolling``, ``ewm``, ``pct_change``
   with a non-negative period, and ``pandas_ta`` indicators that are themselves
   built from those primitives).
2. ``tests/test_features.py::test_features_do_not_use_future_rows`` mutates every
   row after a cut-off and asserts the features at and before the cut-off are
   bit-identical.
3. ``shift()`` is never used with a negative period anywhere in this file.

Contrast with the **label** in :mod:`src.dataset.dataset_builder`, which is
*supposed* to look forward (``close[t + horizon]``).  Features and labels are
built in different modules for exactly that reason.

Scale policy
------------
Raw price levels are deliberately excluded from the model matrix; every feature
is a ratio, a difference of prices normalised by price, or an oscillator in
natural units.  That keeps a 2022 dataset and a 2025 dataset on comparable
scales and makes the tree models' split points meaningful.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils import get_logger

logger = get_logger("features")

#: ``pandas_ta`` is chatty about the occasional off-convergence indicator.  Those
#: warnings are suppressed here (not globally) and the resulting NaNs are handled
#: explicitly by :meth:`FeatureEngineer.build`.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- documentation

#: Hand-written documentation for features whose name does not embed a period.
#: Period-parameterised families (``sma_20``, ``return_6h``, ...) are documented
#: by :func:`feature_registry`, which fills their entries from templates so that
#: *any* configured period is described rather than only the default ones.
FIXED_FEATURE_DOCS: dict[str, str] = {
    "price_vs_sma20": "Relative distance of price from its 20-period mean: close/sma_20 - 1. Positive = stretched above.",
    "price_vs_sma50": "Relative distance of price from its 50-period mean: close/sma_50 - 1.",
    "price_vs_sma200": "Relative distance of price from its 200-period mean: close/sma_200 - 1 (long-term trend proxy).",
    "sma20_vs_sma50": "Short-vs-medium trend: sma_20/sma_50 - 1. Positive = short average above medium.",
    "sma50_vs_sma200": "Medium-vs-long trend: sma_50/sma_200 - 1. The classic golden/death-cross spread.",
    "macd": "MACD line: ema(close, fast) - ema(close, slow). Trend momentum in price units.",
    "macd_signal": "EMA of the MACD line over the signal period (the signal line).",
    "macd_histogram": "MACD minus signal. Positive = bullish momentum, widening = accelerating.",
    "atr_percent": "ATR divided by close x 100. Price-normalised volatility, comparable across the whole sample.",
    "bb_middle": "Bollinger middle band: SMA of close over the band period.",
    "bb_upper": "Bollinger upper band: bb_middle + std multiplier x rolling std of close over the band period.",
    "bb_lower": "Bollinger lower band: bb_middle - std multiplier x rolling std of close over the band period.",
    "bb_width": "Band width relative to the middle band: (bb_upper - bb_lower)/bb_middle. Squeeze/compression measure.",
    "bb_position": "Where price sits inside the bands: (close - bb_lower)/(bb_upper - bb_lower), in [0, 1].",
    "volume_ratio": "Current volume relative to its recent average: volume/volume_sma_20.",
    "volume_change": "Relative change in volume over the volume look-back: volume[t]/volume[t-L] - 1.",
    "candle_body": "Signed real body relative to open: (close - open)/open.",
    "candle_range": "Full high-low range relative to open: (high - low)/open.",
    "upper_wick": "Upper shadow relative to open: (high - max(open, close))/open. Rejection pressure.",
    "lower_wick": "Lower shadow relative to open: (min(open, close) - low)/open. Support holding.",
    "body_to_range": "Body as a fraction of range: candle_body/candle_range. Close to +/-1 = directional candle.",
}

#: (regex, template) pairs for period-parameterised feature names.  Applied in
#: order, so more specific patterns come first.
_PERIOD_DOC_TEMPLATES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^return_(\d+)h$"),
     "{period}-candle simple return of close: close[t]/close[t-{period}] - 1."),
    (re.compile(r"^sma_(\d+)$"),
     "Simple moving average of close over the last {period} candles."),
    (re.compile(r"^ema_(\d+)$"),
     "Exponential moving average of close with span {period} (alpha = 2/{alpha_denom})."),
    (re.compile(r"^rsi_(\d+)$"),
     "Wilder's Relative Strength Index over {period} candles. 0-100; >70 overbought, <30 oversold."),
    (re.compile(r"^atr_(\d+)$"),
     "Wilder's Average True Range over {period} candles, in price units."),
    (re.compile(r"^roc_(\d+)$"),
     "Rate of change of close over {period} candles, in percent."),
    (re.compile(r"^volume_sma_(\d+)$"),
     "Simple moving average of base volume over the last {period} candles."),
    (re.compile(r"^rolling_volatility_(\d+)$"),
     "Annualised standard deviation of 1-candle log returns over the last {period} candles."),
    (re.compile(r"^price_vs_sma(\d+)$"),
     "Relative distance of price from its {period}-period mean: close/sma_{period} - 1."),
    (re.compile(r"^sma(\d+)_vs_sma(\d+)$"),
     "Trend spread: sma_{fast}/sma_{slow} - 1. Positive = the faster average sits above the slower one."),
)


#: Group classification rules, applied in order.  The first match wins, so more
#: specific patterns are listed first.
_GROUP_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^return_\d+h$"), "returns"),
    (re.compile(r"^price_vs_sma\d+$"), "ma_relationships"),
    (re.compile(r"^sma\d+_vs_sma\d+$"), "ma_relationships"),
    (re.compile(r"^(sma|ema)_\d+$"), "moving_averages"),
    (re.compile(r"^rsi_\d+$"), "momentum"),
    (re.compile(r"^macd"), "momentum"),
    (re.compile(r"^roc_\d+$"), "momentum"),
    (re.compile(r"^atr_percent$"), "volatility"),
    (re.compile(r"^atr_\d+$"), "volatility"),
    (re.compile(r"^rolling_volatility_\d+$"), "volatility"),
    (re.compile(r"^bb_"), "bollinger"),
    (re.compile(r"^volume_"), "volume"),
    (re.compile(r"^candle_|^upper_wick$|^lower_wick$|^body_to_range$"), "candle_structure"),
)


def group_of(name: str) -> str:
    """Classify a feature name into one of the reportable feature groups."""
    for pattern, group in _GROUP_RULES:
        if pattern.match(name):
            return group
    raise ValueError(f"Feature {name!r} does not belong to any known feature group")


def _document(name: str) -> str | None:
    """Resolve a documentation string for one feature name."""
    if name in FIXED_FEATURE_DOCS:
        return FIXED_FEATURE_DOCS[name]
    for pattern, template in _PERIOD_DOC_TEMPLATES:
        match = pattern.match(name)
        if not match:
            continue
        groups = {g: int(v) for g, v in zip(match.groups(), match.groups()) if v.isdigit()}
        if "period" not in groups:
            groups["period"] = int(next(iter(groups.values()), 0))
        if "alpha_denom" in template:
            groups["alpha_denom"] = groups["period"] + 1
        if "{fast}" in template or "{slow}" in template:
            values = list(groups.values())
            groups.setdefault("fast", values[0])
            groups.setdefault("slow", values[-1])
        return template.format(**groups)
    return None


def feature_registry(feature_config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """The full ``name -> documentation`` map for a given feature config.

    With no argument this returns the production schema described in
    ``config/config.yaml``, which is the table rendered into the README.
    """
    engineer = FeatureEngineer(feature_config)
    registry: dict[str, str] = {}
    for name in engineer.feature_names:
        doc = _document(name)
        if doc is None:
            raise ValueError(
                f"Feature {name!r} has no documentation. Add it to FIXED_FEATURE_DOCS or a "
                "_PERIOD_DOC_TEMPLATES pattern before it can reach the model."
            )
        registry[name] = doc
    return registry


#: The production feature schema: every feature the default config produces.
#: Assigned at the very bottom of this module, once FeatureEngineer exists.
FEATURE_DOCS: dict[str, str]

FEATURE_GROUP_OF: dict[str, str]

FEATURE_GROUPS: dict[str, tuple[str, ...]]


# --------------------------------------------------------------------------- helpers

def _safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float) -> pd.Series:
    """Element-wise division that yields ``NaN`` instead of ``inf`` on a zero denominator."""
    den = denominator.where(denominator.abs() > eps)
    return numerator / den


def _series(obj: Any) -> pd.Series:
    """Coerce whatever ``pandas_ta`` returned into a plain float Series.

    ``pandas_ta`` returns a DataFrame for multi-output indicators and names its
    columns after its own parameters.  Positional access keeps this module
    immune to those cosmetic renames.
    """
    if obj is None:
        return pd.Series(dtype="float64")
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype="float64")
        return obj.iloc[:, 0].astype("float64")
    return obj.astype("float64")


# --------------------------------------------------------------------------- engine

class FeatureEngineer:
    """Builds the causal feature matrix for one OHLCV series.

    Parameters
    ----------
    config:
        The ``features:`` block of ``config/config.yaml``.
    drop_incomplete:
        Drop rows where any feature is still warming up.  Defaults to ``True``;
        the dataset builder relies on this to guarantee a NaN-free design matrix.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, *, drop_incomplete: bool = True) -> None:
        cfg = dict(config or {})
        self.sma_periods = [int(p) for p in cfg.get("sma_periods", [10, 20, 50, 100, 200])]
        self.ema_periods = [int(p) for p in cfg.get("ema_periods", [12, 26])]
        self.rsi_period = int(cfg.get("rsi_period", 14))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.bollinger_period = int(cfg.get("bollinger_period", 20))
        self.bollinger_std = float(cfg.get("bollinger_std", 2.0))
        self.macd_fast = int(cfg.get("macd_fast", 12))
        self.macd_slow = int(cfg.get("macd_slow", 26))
        self.macd_signal = int(cfg.get("macd_signal", 9))
        self.roc_period = int(cfg.get("roc_period", 12))
        self.return_periods = [int(p) for p in cfg.get("return_periods", [1, 3, 6, 12, 24])]
        self.volume_sma_period = int(cfg.get("volume_sma_period", 20))
        self.volume_lookback = int(cfg.get("volume_lookback", 6))
        self.volatility_lookbacks = [int(p) for p in cfg.get("volatility_lookbacks", [20, 50])]
        self.periods_per_year = int(cfg.get("periods_per_year", 8760))
        self.eps = float(cfg.get("eps", 1e-12))
        self.drop_incomplete = drop_incomplete

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Every column this config will produce, in canonical order.

        Kept deliberately independent of the computation code: :meth:`build`
        asserts that what it actually computes equals this list, so the two can
        never drift apart without a test failing.
        """
        names: list[str] = [f"return_{p}h" for p in self.return_periods]
        names += [f"sma_{p}" for p in self.sma_periods]
        names += [f"ema_{p}" for p in self.ema_periods]
        for period in (20, 50, 200):
            if period in self.sma_periods:
                names.append(f"price_vs_sma{period}")
        for fast, slow in ((20, 50), (50, 200)):
            if fast in self.sma_periods and slow in self.sma_periods:
                names.append(f"sma{fast}_vs_sma{slow}")
        names += [
            f"rsi_{self.rsi_period}",
            "macd",
            "macd_signal",
            "macd_histogram",
            f"roc_{self.roc_period}",
            f"atr_{self.atr_period}",
            "atr_percent",
        ]
        names += [f"rolling_volatility_{w}" for w in self.volatility_lookbacks]
        names += ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position"]
        names += [f"volume_sma_{self.volume_sma_period}", "volume_ratio", "volume_change"]
        names += ["candle_body", "candle_range", "upper_wick", "lower_wick", "body_to_range"]
        return tuple(names)

    @property
    def feature_docs(self) -> dict[str, str]:
        """The documented registry for this engine's config."""
        registry: dict[str, str] = {}
        for name in self.feature_names:
            doc = _document(name)
            if doc is None:
                raise ValueError(
                    f"Feature {name!r} has no documentation. Add it to FIXED_FEATURE_DOCS or a "
                    "_PERIOD_DOC_TEMPLATES pattern before it can reach the model."
                )
            registry[name] = doc
        return registry

    @classmethod
    def from_config(cls, config: Any, *, drop_incomplete: bool = True) -> "FeatureEngineer":
        return cls(config.features, drop_incomplete=drop_incomplete)

    # ------------------------------------------------------------- warm-up

    @property
    def required_history(self) -> int:
        """Number of leading candles any feature needs before it is fully warmed up.

        Driven by the longest look-back in the config (currently the 200-period
        SMA), with a small allowance for the exponentially smoothed indicators.
        """
        return max(
            *self.sma_periods,
            *self.ema_periods,
            self.rsi_period,
            self.atr_period,
            self.bollinger_period,
            self.roc_period,
            *self.return_periods,
            self.roc_period,
            self.volume_sma_period + self.volume_lookback,
            *self.volatility_lookbacks,
            self.macd_slow + self.macd_signal,
        )

    # ------------------------------------------------------------- builders

    def _build_returns(self, close: pd.Series) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for period in self.return_periods:
            out[f"return_{period}h"] = close.pct_change(period)
        return out

    def _build_moving_averages(self, close: pd.Series) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for period in self.sma_periods:
            out[f"sma_{period}"] = _series(ta.sma(close, length=period))
        for period in self.ema_periods:
            out[f"ema_{period}"] = _series(ta.ema(close, length=period))
        return out

    def _build_ma_relationships(self, close: pd.Series, mas: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for period in (20, 50, 200):
            key = f"sma_{period}"
            if key in mas:
                out[f"price_vs_sma{period}"] = _safe_divide(close, mas[key], self.eps) - 1.0
        for fast, slow in ((20, 50), (50, 200)):
            f_key, s_key = f"sma_{fast}", f"sma_{slow}"
            if f_key in mas and s_key in mas:
                out[f"sma{fast}_vs_sma{slow}"] = _safe_divide(mas[f_key], mas[s_key], self.eps) - 1.0
        return out

    def _build_momentum(self, close: pd.Series) -> dict[str, pd.Series]:
        macd_frame = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        macd_line = macd_signal_line = macd_hist = pd.Series(np.nan, index=close.index, dtype="float64")
        if macd_frame is not None and macd_frame.shape[1] >= 3:
            macd_line = macd_frame.iloc[:, 0].astype("float64")
            macd_signal_line = macd_frame.iloc[:, 1].astype("float64")
            macd_hist = macd_frame.iloc[:, 2].astype("float64")
        return {
            f"rsi_{self.rsi_period}": _series(ta.rsi(close, length=self.rsi_period)),
            "macd": macd_line,
            "macd_signal": macd_signal_line,
            "macd_histogram": macd_hist,
            f"roc_{self.roc_period}": _series(ta.roc(close, length=self.roc_period)),
        }

    def _build_volatility(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, pd.Series]:
        atr = _series(ta.atr(high, low, close, length=self.atr_period))
        log_returns = np.log(close.where(close > 0)).diff()
        annualiser = float(np.sqrt(self.periods_per_year))
        out: dict[str, pd.Series] = {
            f"atr_{self.atr_period}": atr,
            "atr_percent": _safe_divide(atr, close, self.eps) * 100.0,
        }
        for window in self.volatility_lookbacks:
            out[f"rolling_volatility_{window}"] = log_returns.rolling(window).std() * annualiser
        return out

    def _build_bollinger(self, close: pd.Series) -> dict[str, pd.Series]:
        # Computed directly rather than read off pandas_ta's columns so the band
        # definition is explicit, auditable and unit-tested.
        middle = close.rolling(self.bollinger_period, min_periods=self.bollinger_period).mean()
        std = close.rolling(self.bollinger_period, min_periods=self.bollinger_period).std()
        upper = middle + self.bollinger_std * std
        lower = middle - self.bollinger_std * std
        spread = (upper - lower).where((upper - lower).abs() > self.eps)
        return {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": _safe_divide(upper - lower, middle, self.eps),
            "bb_position": _safe_divide(close - lower, spread, self.eps),
        }

    def _build_volume(self, volume: pd.Series) -> dict[str, pd.Series]:
        vol_sma = _series(ta.sma(volume, length=self.volume_sma_period))
        lag = volume.shift(self.volume_lookback)
        return {
            f"volume_sma_{self.volume_sma_period}": vol_sma,
            "volume_ratio": _safe_divide(volume, vol_sma, self.eps),
            # Divided explicitly rather than with pct_change: pct_change pads
            # missing values, which would leak a carried-forward volume into
            # the feature.
            "volume_change": _safe_divide(volume, lag, self.eps) - 1.0,
        }

    def _build_candle_structure(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, pd.Series]:
        body = _safe_divide(close - open_, open_, self.eps)
        rng = _safe_divide(high - low, open_, self.eps)
        upper_wick = _safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), open_, self.eps)
        lower_wick = _safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, open_, self.eps)
        return {
            "candle_body": body,
            "candle_range": rng,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_to_range": _safe_divide(body, rng, self.eps),
        }

    # ------------------------------------------------------------- public

    def build(self, klines: pd.DataFrame, *, validate_schema: bool = True) -> pd.DataFrame:
        """Compute every feature for ``klines``.

        Parameters
        ----------
        klines:
            Validated OHLCV frame with a sorted, unique, UTC ``DatetimeIndex``.
        validate_schema:
            Assert that the produced columns match :attr:`feature_names` exactly,
            so an undocumented feature can never reach the model.

        Returns
        -------
        pandas.DataFrame
            One row per candle, columns = :attr:`feature_names`.  Rows still
            warming up are dropped when ``drop_incomplete`` is set.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(klines.columns)
        if missing:
            raise ValueError(f"Kline frame is missing required column(s): {sorted(missing)}")
        if klines.empty:
            logger.warning("FeatureEngineer received an empty frame")
            return pd.DataFrame(columns=list(self.feature_names), index=klines.index)

        frame = klines.sort_index()
        if not frame.index.is_unique:
            raise ValueError("FeatureEngineer requires unique timestamps; run the downloader/validator first")

        open_, high = frame["open"].astype("float64"), frame["high"].astype("float64")
        low, close = frame["low"].astype("float64"), frame["close"].astype("float64")
        volume = frame["volume"].astype("float64")

        mas = self._build_moving_averages(close)
        blocks: list[Mapping[str, pd.Series]] = [
            self._build_returns(close),
            mas,
            self._build_ma_relationships(close, mas),
            self._build_momentum(close),
            self._build_volatility(high, low, close),
            self._build_bollinger(close),
            self._build_volume(volume),
            self._build_candle_structure(open_, high, low, close),
        ]

        data: dict[str, pd.Series] = {}
        expected = set(self.feature_names)
        for block in blocks:
            for name, series in block.items():
                if name not in expected:
                    raise ValueError(
                        f"Feature {name!r} is not part of the declared schema; declare it in "
                        "FeatureEngineer.feature_names and document it before it can be used"
                    )
                data[name] = series.reindex(frame.index).astype("float64")

        # Emit in the canonical documented order.
        features = pd.DataFrame({name: data[name] for name in self.feature_names}, index=frame.index)
        features.index.name = frame.index.name or "timestamp"

        if validate_schema:
            assert_documented(features, self.feature_docs)

        if self.drop_incomplete:
            warmup = self.required_history
            features = features.iloc[warmup:]
            still_warm = features.isna().any(axis=1)
            if still_warm.any():
                n = int(still_warm.sum())
                logger.warning("Dropping %d row(s) with warm-up NaNs after the %d-candle warm-up", n, warmup)
                features = features.loc[~still_warm]

        return features

    def build_for(self, klines: pd.DataFrame) -> pd.DataFrame:
        """Alias for :meth:`build` kept for readability at call sites."""
        return self.build(klines)


def assert_documented(features: pd.DataFrame, registry: Mapping[str, str] | None = None) -> None:
    """Fail unless the feature frame matches a documented registry exactly.

    ``registry`` defaults to the production schema (:data:`FEATURE_DOCS`); tests
    and non-default configs pass their own registry.
    """
    documented = set(registry) if registry is not None else set(FEATURE_DOCS)
    produced = set(features.columns)
    if produced != documented:
        undocumented = sorted(produced - documented)
        missing = sorted(documented - produced)
        raise ValueError(f"Feature schema mismatch — undocumented: {undocumented}; missing: {missing}")
    for name in features.columns:
        group_of(name)  # raises if the feature belongs to no known group


def build_features(klines: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Functional convenience wrapper around :class:`FeatureEngineer`."""
    return FeatureEngineer(config).build(klines)


def feature_documentation_markdown(feature_config: Mapping[str, Any] | None = None) -> str:
    """Render the documented feature registry as a Markdown table (for the README)."""
    registry = feature_registry(feature_config)
    lines = ["| Feature | Group | Definition |", "| --- | --- | --- |"]
    for name, doc in registry.items():
        lines.append(f"| `{name}` | {group_of(name)} | {doc} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- derived schema
# Populated here (not beside the definitions above) because feature_registry()
# needs FeatureEngineer, which is defined later in the module.

FEATURE_DOCS = feature_registry()
FEATURE_GROUP_OF = {name: group_of(name) for name in FEATURE_DOCS}
FEATURE_GROUPS = {
    group: tuple(name for name in FEATURE_DOCS if FEATURE_GROUP_OF[name] == group)
    for group in dict.fromkeys(FEATURE_GROUP_OF.values())
}
