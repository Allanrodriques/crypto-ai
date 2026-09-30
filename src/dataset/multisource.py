"""Multi-source dataset assembly.

This is the module that turns "the experiment says use technical + sentiment"
into a feature matrix.  Its ordering is deliberate and is the whole point:

    raw source -> availability contract -> conservative as-of align
              -> feature group build -> drop incomplete rows

Nothing downstream of the alignment step is allowed to look at a source
directly, so there is exactly one place where a new data source could
introduce look-ahead, and it is one auditable function.

Caching
-------
Funding, futures and sentiment downloads are cached under
``data/raw/external/<source>_<symbol>.parquet`` with a metadata sidecar, so the
experiment suite re-runs in seconds and a network outage does not invalidate
completed work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.alignment.availability import align_asof, coverage_report
from src.data.quality import SOURCE_CONTRACTS, SourceQuality, measure_source
from src.data.sources.funding import FundingSettings, fetch_funding_rates
from src.data.sources.futures import FuturesSettings, fetch_futures_klines
from src.data.sources.sentiment import fetch_fear_greed
from src.data.sources import sentiment as sentiment_source
from src.features.context import build_context_features
from src.features.derivatives import build_derivatives_features
from src.features.groups import get_group
from src.features.microstructure import build_microstructure_features
from src.features.sentiment import build_sentiment_features
from src.utils import get_logger, save_json, timestamp_to_utc

logger = get_logger("dataset.multisource")


class SourceUnavailableError(RuntimeError):
    """Raised when an experiment requests a source that cannot be obtained."""


@dataclass
class SourceBundle:
    """Raw (unaligned) source frames, keyed by source name."""

    spot: pd.DataFrame
    funding: pd.DataFrame | None = None
    futures: pd.DataFrame | None = None
    sentiment: pd.DataFrame | None = None
    quality: dict[str, SourceQuality] = field(default_factory=dict)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AlignedSources:
    """Sources after conservative as-of alignment to the feature grid."""

    index: pd.DatetimeIndex
    spot: pd.DataFrame
    funding: pd.DataFrame | None = None
    futures: pd.DataFrame | None = None
    sentiment: pd.DataFrame | None = None

    def coverage(self) -> dict[str, dict[str, Any]]:
        out = {"binance_spot": coverage_report(self.spot, len(self.index))}
        for name, frame in (("binance_funding", self.funding),
                            ("binance_futures", self.futures),
                            ("fear_greed", self.sentiment)):
            if frame is not None:
                out[name] = coverage_report(frame, len(self.index))
        return out


# --------------------------------------------------------------------------- loading

def _cache_path(cache_dir: Path, source: str, symbol: str) -> Path:
    return cache_dir / f"{source}_{symbol}.parquet"


def _cache_shortfall(
    index: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> str | None:
    """Describe why a cached series does not cover the requested window.

    Returns ``None`` when the cache is adequate.  A source is allowed to be
    *shorter* than requested at the edges only if the caller asked for nothing in
    particular; an explicit ``start`` must be honoured, otherwise a truncated
    download is indistinguishable from a complete one.
    """
    if start is not None:
        wanted = timestamp_to_utc(pd.Timestamp(start))
        if index.min() > wanted:
            return f"start={wanted}"
    if end is not None:
        wanted = timestamp_to_utc(pd.Timestamp(end))
        if index.max() < wanted:
            return f"end={wanted}"
    return None


def _fetch_would_help(
    meta: dict[str, Any],
    index: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> bool:
    """Decide whether re-downloading a short cache could plausibly extend it.

    Some sources legitimately begin later than the requested window (the fear &
    greed series only starts in 2018).  Without this check, a caller that
    requests a window the source can never satisfy would re-download the same
    data on every single run, burning API quota to obtain the identical answer.
    The metadata sidecar records what was asked for last time, so a cache that
    already failed this same request is treated as the best available.
    """
    if not meta:
        return True
    try:
        realized_start = pd.Timestamp(meta.get("start"))
        realized_end = pd.Timestamp(meta.get("end"))
    except (TypeError, ValueError):
        return True
    if realized_start.tzinfo is None:
        realized_start = realized_start.tz_localize("UTC")
    if realized_end.tzinfo is None:
        realized_end = realized_end.tz_localize("UTC")

    if start is not None:
        wanted = timestamp_to_utc(pd.Timestamp(start))
        if index.min() > wanted:
            # The cache starts too late.  Only re-fetch if the previous fetch
            # was not already asked to reach this far back.
            asked = meta.get("requested_start")
            if asked is None:
                return True
            try:
                asked_ts = pd.Timestamp(asked)
                if asked_ts.tzinfo is None:
                    asked_ts = asked_ts.tz_localize("UTC")
            except (TypeError, ValueError):
                return True
            return asked_ts > wanted or realized_start < index.min()
    if end is not None:
        wanted = timestamp_to_utc(pd.Timestamp(end))
        if index.max() < wanted:
            asked = meta.get("requested_end")
            if asked is None:
                return True
            try:
                asked_ts = pd.Timestamp(asked)
                if asked_ts.tzinfo is None:
                    asked_ts = asked_ts.tz_localize("UTC")
            except (TypeError, ValueError):
                return True
            return asked_ts < wanted or realized_end > index.max()
    return False


def load_or_fetch(
    source: str,
    *,
    symbol: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    cache_dir: Path,
    session=None,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Return a source frame, fetching it only when not already cached."""
    spec = SOURCE_CONTRACTS.get(source)
    if spec is None:
        raise SourceUnavailableError(f"Unknown source {source!r}")
    if not spec.historically_available:
        raise SourceUnavailableError(
            f"{source} has no usable history: {spec.unavailable_reason}"
        )

    path = _cache_path(cache_dir, source, symbol)
    if path.exists():
        frame = pd.read_parquet(path)
        if len(frame):
            # A cache is only reusable if it actually *covers* the requested
            # window.  Reusing a short read as though it were the full history
            # is how a 5-month funding file silently became a 9.6%-coverage
            # "multi-year" experiment.
            gap = _cache_shortfall(frame.index, start, end)
            if gap is None:
                logger.info("using cached %s (%d rows) from %s", source, len(frame), path.name)
                return frame
            meta: dict[str, Any] = {}
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, ValueError):
                    meta = {}
            if not _fetch_would_help(meta, frame.index, start, end):
                logger.warning(
                    "cached %s covers %s -> %s; %s was requested but an earlier identical "
                    "request returned the same span, so this is the full available history",
                    source,
                    frame.index.min(),
                    frame.index.max(),
                    gap,
                )
                return frame
            if not allow_download:
                raise SourceUnavailableError(
                    f"{source} cache for {symbol} covers {frame.index.min()} -> "
                    f"{frame.index.max()} but {gap} was requested, and --no-download "
                    f"forbids re-fetching. Re-run without --no-download to extend the "
                    f"cache, or narrow the requested window."
                )
            logger.warning(
                "cached %s covers %s -> %s but %s was requested; re-fetching",
                source,
                frame.index.min(),
                frame.index.max(),
                gap,
            )

    if not allow_download:
        raise SourceUnavailableError(
            f"No usable {source} cache for {symbol} in {cache_dir}, and --no-download "
            f"forbids fetching it. Run without --no-download once to populate the cache."
        )

    if source == "binance_funding":
        frame = fetch_funding_rates(
            FundingSettings(symbol=symbol, start=start, end=end), session=session
        )
    elif source == "binance_futures":
        frame = fetch_futures_klines(
            FuturesSettings(symbol=symbol, interval="1h", start=start, end=end), session=session
        )
    elif source == "fear_greed":
        frame = fetch_fear_greed()
    else:  # pragma: no cover - spot comes from the V1 raw store
        raise SourceUnavailableError(f"Source {source!r} is not fetched here")

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    save_json(
        {
            "source": source,
            "symbol": symbol,
            "rows": int(len(frame)),
            "start": str(frame.index.min()),
            "end": str(frame.index.max()),
            # Recorded so a later run can tell "this source cannot reach back
            # that far" from "this cache was never asked to".
            "requested_start": str(pd.Timestamp(start)) if start is not None else None,
            "requested_end": str(pd.Timestamp(end)) if end is not None else None,
            "contract": spec.contract().describe(),
            "endpoint": spec.endpoint,
        },
        path.with_suffix(".meta.json"),
    )
    return frame


def load_sources(
    *,
    symbol: str,
    sources: Sequence[str],
    spot: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    cache_dir: Path,
    session=None,
    allow_download: bool = True,
) -> SourceBundle:
    """Load every source an experiment asks for and measure its quality.

    When the caller does not state a window, the spot feature grid *is* the
    window.  Leaving it as ``None`` made every cache check a no-op, so a
    truncated download passed validation and the resulting coverage collapse
    went unnoticed; the grid bounds are what the features will actually need.
    """
    quality: dict[str, SourceQuality] = {}
    coverage: dict[str, dict[str, Any]] = {}

    index = pd.DatetimeIndex(spot.index, tz="UTC")
    if len(index):
        if start is None:
            start = index.min()
        if end is None:
            end = index.max()
    quality["binance_spot"] = measure_source("binance_spot", spot, feature_index=index, aligned=spot)
    coverage["binance_spot"] = coverage_report(spot, len(index))

    bundle = SourceBundle(spot=spot, quality=quality, coverage=coverage)
    for source in sources:
        if source == "binance_spot":
            continue
        if source not in SOURCE_CONTRACTS:
            raise SourceUnavailableError(f"Unknown data source {source!r}")
        try:
            frame = load_or_fetch(
                source,
                symbol=symbol,
                start=start,
                end=end,
                cache_dir=cache_dir,
                session=session,
                allow_download=allow_download,
            )
        except SourceUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - network dependent
            raise SourceUnavailableError(f"Could not obtain {source}: {exc}") from exc

        aligned_preview = align_asof(index, frame, SOURCE_CONTRACTS[source].contract())
        quality[source] = measure_source(
            source, frame, feature_index=index, aligned=aligned_preview
        )
        coverage[source] = coverage_report(aligned_preview, len(index))

        if source == "binance_funding":
            bundle.funding = frame
        elif source == "binance_futures":
            bundle.futures = frame
        elif source == "fear_greed":
            bundle.sentiment = frame

    bundle.quality = quality
    bundle.coverage = coverage
    for key, cov in coverage.items():
        logger.info("source %-18s coverage %6.2f%%  %s -> %s",
                    key, cov["coverage_pct"], cov.get("first_covered"), cov.get("last_covered"))
    return bundle


# --------------------------------------------------------------------------- alignment

def align_sources(bundle: SourceBundle, index: pd.DatetimeIndex) -> AlignedSources:
    """As-of align every external source to the hourly feature grid."""
    index = pd.DatetimeIndex(index, tz="UTC")
    aligned = AlignedSources(index=index, spot=bundle.spot.reindex(index))

    if bundle.funding is not None:
        contract = SOURCE_CONTRACTS["binance_funding"].contract()
        aligned.funding = align_asof(index, bundle.funding, contract, value_columns=["funding_rate"])
    if bundle.futures is not None:
        contract = SOURCE_CONTRACTS["binance_futures"].contract()
        columns = [c for c in ("close", "futures_volume", "trades", "taker_buy_volume")
                   if c in bundle.futures.columns]
        aligned.futures = align_asof(index, bundle.futures, contract, value_columns=columns)
    if bundle.sentiment is not None:
        contract = SOURCE_CONTRACTS["fear_greed"].contract()
        aligned.sentiment = align_asof(
            index, bundle.sentiment, contract, value_columns=["fear_greed_value"]
        )
    return aligned


# --------------------------------------------------------------------------- context

@dataclass
class ExperimentContext:
    """Spot data plus the aligned external sources, shared across experiments.

    Alignment is the expensive part (three as-of merges over ~41k hourly bars),
    and every experiment needs the same aligned result.  Building it once and
    handing it to each experiment also guarantees that all five experiments are
    scored against a byte-identical view of the source data - if each experiment
    re-aligned independently, a mid-run data refresh could silently make the
    later experiments incomparable to the earlier ones.
    """

    symbol: str
    interval: str
    spot: pd.DataFrame
    aligned: AlignedSources
    quality: dict[str, Any]
    cache_dir: Path

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.aligned.index

    def features(self, feature_groups: Sequence[str], **kwargs: Any) -> pd.DataFrame:
        """Build one feature group's union from the shared aligned sources."""
        return build_feature_matrix(self.aligned, feature_groups, **kwargs)

    def coverage_table(self) -> pd.DataFrame:
        rows = []
        for name, record in (self.quality or {}).items():
            rows.append({**({"source": name} if isinstance(record, dict) else {}), **{
                k: v for k, v in (record if isinstance(record, dict) else record.to_dict()).items()
            }})
        return pd.DataFrame(rows)


def build_context(
    *,
    symbol: str,
    sources: Sequence[str],
    spot: pd.DataFrame,
    cache_dir: str | Path,
    interval: str = "1h",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    session=None,
    allow_download: bool = True,
) -> ExperimentContext:
    """Load, align and quality-check every source once."""
    cache = Path(cache_dir)
    bundle = load_sources(
        symbol=symbol,
        sources=list(sources),
        spot=spot,
        start=start,
        end=end,
        cache_dir=cache,
        session=session,
        allow_download=allow_download,
    )
    aligned = align_sources(bundle, spot.index)
    quality = {name: record for name, record in (getattr(bundle, "quality", {}) or {}).items()}
    return ExperimentContext(
        symbol=symbol,
        interval=interval,
        spot=spot,
        aligned=aligned,
        quality=quality,
        cache_dir=cache,
    )


# --------------------------------------------------------------------------- features

def build_feature_matrix(
    aligned: AlignedSources,
    feature_groups: Sequence[str],
    *,
    enabled_features: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build the union of the requested feature groups on the hourly grid.

    ``enabled_features`` optionally restricts a group to a subset of its
    features, which is what the ablation study uses.  Passing ``None`` (the
    default) takes the group in full, so an experiment that names a group gets
    everything currently registered under it.
    """
    index = aligned.index
    frames: list[pd.DataFrame] = []
    wanted = dict(enabled_features or {})

    if "technical" in feature_groups:
        from src.features.feature_engineering import FeatureEngineer

        # Production defaults, identical to the V1 baseline definition.  Passing
        # the whole Config would re-apply config overrides, which would make the
        # baseline depend on the caller's config rather than on the registry.
        engineer = FeatureEngineer()
        technical = engineer.build(aligned.spot)
        subset = wanted.get("technical")
        if subset is not None:
            technical = technical[subset]
        frames.append(technical)

    if "derivatives" in feature_groups:
        if aligned.funding is None or aligned.futures is None:
            raise SourceUnavailableError(
                "derivatives features need both binance_funding and binance_futures"
            )
        frames.append(
            build_derivatives_features(
                index, aligned.funding, aligned.futures, aligned.spot,
                enabled=wanted.get("derivatives"),
            )
        )

    if "sentiment" in feature_groups:
        if aligned.sentiment is None:
            raise SourceUnavailableError("sentiment features need the fear_greed source")
        frames.append(
            build_sentiment_features(index, aligned.sentiment, enabled=wanted.get("sentiment"))
        )

    if "microstructure" in feature_groups:
        frames.append(
            build_microstructure_features(index, aligned.spot, enabled=wanted.get("microstructure"))
        )

    if "context" in feature_groups:
        frames.append(
            build_context_features(index, aligned.spot, enabled=wanted.get("context"))
        )

    if not frames:
        raise ValueError("No feature groups selected")

    matrix = pd.concat(frames, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    matrix.index.name = "timestamp"
    return matrix
