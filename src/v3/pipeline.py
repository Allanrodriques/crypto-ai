"""V3 pipeline orchestrator: config -> plan -> dataset -> walk-forward -> report.

Why this module exists
----------------------
V3 has four leaf modules - :mod:`src.v3.dataset`, :mod:`src.v3.walkforward`,
:mod:`src.v3.report` and :mod:`src.v3.analysis` - and every one of them is a
library: none of them knows which symbol is being run, where its output belongs,
or what the previous stage produced.  This module owns that knowledge and nothing
else.  It is the only place in V3 that decides *what a run is*, so
``scripts/run_v3.py``, the CLI and a notebook all execute exactly the same code
instead of three subtly different re-implementations (the lesson ``src/pipeline/
run.py`` already learned the hard way in V1/V2).

The stage order, and what each stage is allowed to assume
--------------------------------------------------------
=====  ==============  =========================================================
1      plan            pure config resolution; touches no data and no files
2      dataset         point-in-time frame for one symbol
3      walkforward     purged, embargoed out-of-sample evaluation
4      report          human-readable artefacts (optional, degrades loudly)
5      persist         one fitted estimator per (model, horizon)
6      manifest        the machine-readable record of all of the above
=====  ==============  =========================================================

Stage 6 runs *even when an earlier stage failed*, because a failed run that leaves
no record is indistinguishable from a run that never happened.  A failing stage is
recorded as a warning and flips :attr:`PipelineResult.ok` to ``False``; the
exception never escapes as a bare traceback, because the caller's next question
is "what did you actually manage to do?", and the answer is the manifest.

Writing is confined to ``out_dir``
---------------------------------
Every path this module writes - report artefacts, model artifacts, the manifest -
is under one directory, and nothing else.  That is what makes it safe to point at
a temporary root in the test suite: the frozen V1/V2 trees (``data/raw``,
``data/processed``, ``data/predictions``, ``models``, ``reports``) are inputs and
must never be touched.  ``tests/test_v3_pipeline.py`` asserts the containment by
walking the tree, because a leak here would be invisible in every other test.

Excluding feature buckets, and why it is not optional
-----------------------------------------------------
BTC's external caches in ``data/raw`` are truncated: the futures series stops at
2022-03-04 (1,500 rows) and funding at 2022-06-16 (500 rows).  Because
``complete_rows_only`` drops any row with a missing feature, *including* the
``derivatives`` bucket silently cuts BTC's history to 2,577 usable rows, which
makes the 30d and 180d horizons untrainable while the run still reports success.
Excluding it yields 40,833 rows, 84 features and eight trainable horizons.  So
:func:`resolve_plan` takes ``exclude_buckets``, the dataset stage logs which
buckets survived and how many rows they left, and a suspiciously thin history is
a warning rather than a quiet headline.
"""

from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import format_timestamp, get_logger, load_json, save_json, set_global_seed, utc_now
from src.v3.dataset import FeatureSelectionError, build_v3_dataset, bucket_of_feature, v3_block
from src.v3.horizons import Horizon, build_horizon_ladder
from src.v3.models import MODEL_NAMES, artifact_path, build_model, fit_model, save_model
from src.v3.walkforward import run_walk_forward

logger = get_logger("v3.pipeline")

# `src.v3.report` and `src.v3.analysis` are being developed alongside this
# module.  They are imported defensively so that the dataset and walk-forward
# halves of V3 stay runnable - and testable - on their own; a missing reporter
# is a logged warning, never an ImportError at the top of the pipeline.
try:  # pragma: no cover - depends on a sibling module
    from src.v3.report import write_report
except ImportError:  # pragma: no cover - depends on a sibling module
    write_report = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on a sibling module
    from src.v3.analysis import analyse
except ImportError:  # pragma: no cover - depends on a sibling module
    analyse = None  # type: ignore[assignment]


#: The V3 bucket taxonomy, when the registry is importable.  Falls back to the
#: V2 feature groups so a plan can still be resolved on a partial checkout.
try:
    from src.v3.buckets import BUCKET_NAMES
except ImportError:  # pragma: no cover - depends on a sibling module
    from src.features.groups import GROUP_ORDER as BUCKET_NAMES  # type: ignore[assignment]


#: Name of the run manifest written into ``out_dir`` by the last stage.
MANIFEST_NAME = "manifest.json"

#: Artefacts ``v3.reporting.artifacts`` declares a complete V3 run must produce.
#: Mirrored from ``config/v3.yaml`` so :func:`validate_outputs` can check a
#: directory without being handed a config.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "dataset_summary.json",
    "data_availability.csv",
    "horizon_label_coverage.csv",
    "feature_coverage.csv",
    "walkforward_folds.csv",
    "per_horizon_metrics.csv",
    "model_comparison.csv",
    "bucket_ablation.csv",
    "error_analysis.md",
    "summary.md",
)

#: ``config/v3.yaml`` names models in research vocabulary; ``src.v3.models`` is
#: the authority on what can actually be built.  These two are the same estimator
#: under two names, so they are translated rather than dropped.  Anything else
#: the config lists but the registry cannot build is dropped with a warning that
#: names it - silently scoring four of six configured models would be worse.
MODEL_NAME_ALIASES: dict[str, str] = {
    "naive_zero": "zero",
    "linear_ridge": "ridge",
}

#: Rows-per-purge-width below which a history is treated as suspiciously thin.
#: A run needs several multiples of the longest horizon's purge to fill even one
#: walk-forward fold; at 5x the 180d purge (4,320 bars) anything less is far more
#: likely to be a truncated external cache than a genuine short history.  This is
#: a reporting heuristic, never a hard failure.
MIN_ROWS_PER_HORIZON_BAR = 5


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStage:
    """One timed stage of a run.

    ``detail`` is whatever that stage chose to record about itself - row counts,
    resolved paths, a sub-timings mapping - as JSON-friendly values, so the whole
    tuple can be written straight into the manifest.
    """

    name: str
    seconds: float
    detail: dict[str, object]


@dataclass(frozen=True)
class PipelineResult:
    """Everything one :func:`run_pipeline` call produced.

    ``ok`` is the single answer to "did this run do what it said it would do?".
    It is ``False`` if any stage raised; a stage that was legitimately skipped
    (an unavailable reporter, ``save_models=False``) records itself in ``stages``
    and, where it changes what exists on disk, in ``warnings`` - but does not
    make the run a failure.
    """

    symbol: str
    run: object | None
    stages: tuple[PipelineStage, ...]
    report: object | None
    model_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    ok: bool

    @property
    def out_dir(self) -> Path | None:
        """The directory this run wrote to, if it got far enough to have one."""
        for stage in self.stages:
            out = stage.detail.get("out_dir")
            if isinstance(out, str):
                return Path(out)
        return None

    def stage(self, name: str) -> PipelineStage | None:
        """Look a stage up by name; ``None`` if it never ran."""
        for entry in self.stages:
            if entry.name == name:
                return entry
        return None

    def total_seconds(self) -> float:
        return float(sum(stage.seconds for stage in self.stages))

    def to_dict(self) -> dict[str, object]:
        """A JSON-safe view of the run.

        The heavy objects - ``run`` and ``report`` - are deliberately *not*
        dumped.  ``run`` is a frame-carrying result whose interesting part is
        already summarised per horizon in the manifest, and ``report`` is a
        path bundle; both are reduced to pointers so that this dict can be
        logged, returned from a CLI or embedded in a larger result without
        dragging megabytes of predictions along.
        """
        out_dir = self.out_dir
        manifest = out_dir / MANIFEST_NAME if out_dir is not None else None
        horizons: dict[str, object] = {}
        if self.run is not None:
            for result in getattr(self.run, "results", []) or []:
                try:
                    horizons[result.horizon] = result.to_dict()
                except Exception:  # pragma: no cover - defensive
                    horizons[result.horizon] = {"error": "could not summarise horizon result"}
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "seconds": self.total_seconds(),
            "out_dir": str(out_dir) if out_dir is not None else None,
            "manifest": str(manifest) if manifest is not None and manifest.exists() else None,
            "horizons": horizons,
            "report": _artefact_paths(self.report),
            "model_paths": [str(path) for path in self.model_paths],
            "stages": [
                {"name": stage.name, "seconds": stage.seconds, "detail": stage.detail}
                for stage in self.stages
            ],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _block(config: Config | Mapping[str, Any]) -> dict[str, Any]:
    """The ``v3:`` section of a config, tolerating a block passed directly."""
    return v3_block(config)


def _known_buckets(block: Mapping[str, Any]) -> tuple[str, ...]:
    """Every bucket this run understands, in canonical order.

    The registry is the authority.  The config's own ``buckets`` block is added
    in case the config declares a bucket this build has not heard of, so that
    excluding it can be reported as a typo against the config rather than
    silently doing nothing.
    """
    declared = (block.get("features") or {}).get("buckets")
    names = list(BUCKET_NAMES)
    if isinstance(declared, Mapping):
        names += [str(name) for name in declared]
    elif isinstance(declared, (list, tuple, set)):
        names += [str(name) for name in declared]
    seen: dict[str, None] = {}
    for name in names:
        seen.setdefault(str(name), None)
    return tuple(seen)


def _validated_exclusions(block: Mapping[str, Any], exclude_buckets: Sequence[str]) -> tuple[str, ...]:
    """Normalise ``exclude_buckets`` to canonical bucket names, or raise.

    A typo here has to fail loudly.  ``--exclude-buckets derivative`` would
    otherwise be a no-op and the run would proceed with the truncated cache and
    2,577 rows, which is precisely the outcome this pipeline exists to prevent.
    """
    if isinstance(exclude_buckets, (str, bytes)):
        raise FeatureSelectionError(
            "exclude_buckets must be a sequence of bucket names such as "
            "('derivatives',), not a bare string"
        )
    known = _known_buckets(block)
    requested: list[str] = []
    for raw in exclude_buckets or ():
        for part in str(raw).split(","):
            name = part.strip()
            if not name:
                continue
            if name not in known:
                raise FeatureSelectionError(
                    f"Unknown feature bucket {name!r}; known buckets: {list(known)}"
                )
            if name not in requested:
                requested.append(name)
    return tuple(name for name in known if name in requested)


def _bucket_selection(
    block: Mapping[str, Any], excluded: Sequence[str]
) -> dict[str, Any]:
    """The bucket -> switch mapping that implements the exclusions.

    Excluded buckets map to ``"none"``, which
    :func:`src.v3.dataset.resolve_feature_selection` reads as "switched off" -
    so they resolve to *no* features, which is the point.  A bucket the config
    does not mention is also off: enabling a bucket nobody asked for is how the
    truncated BTC derivatives cache ends up silently inside a dataset.
    """
    features_cfg = dict(block.get("features") or {})
    declared = features_cfg.get("buckets")
    groups = features_cfg.get("groups")

    if isinstance(declared, Mapping):
        source: dict[str, Any] = {str(key): value for key, value in declared.items()}
    elif isinstance(declared, (list, tuple, set)):
        source = {str(name): "all" for name in declared}
    elif isinstance(groups, (list, tuple, set)) and groups:
        source = {str(name): "all" for name in groups}
    else:
        source = {name: "all" for name in _known_buckets(block)}

    selection: dict[str, Any] = {}
    for bucket in _known_buckets(block):
        if bucket in excluded:
            selection[bucket] = "none"
        else:
            selection[bucket] = source.get(bucket, "none")
    return selection


def resolve_plan(
    config: Config | Mapping[str, Any],
    symbol: str,
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    models: Sequence[str] | None = None,
    exclude_buckets: Sequence[str] = (),
) -> dict[str, object]:
    """Resolve a config into the concrete plan one run will execute.

    Pure: no data is read and nothing is written, so a plan can be printed,
    diffed or asserted in a test without building a dataset.

    Parameters
    ----------
    config:
        A config carrying a ``v3:`` block (``config/v3.yaml``), or the block
        itself.
    symbol:
        The symbol to plan for.  Upper-cased, because the caches and the raw
        klines are named that way.
    horizons:
        Override of ``v3.horizons``.  Sorted by duration, so the plan is
        independent of the order the flags arrived in.
    models:
        Override of ``v3.models``, reported verbatim.  Which of these can
        actually be *built* is decided in :func:`run_pipeline`, which knows the
        estimator registry; a plan stays a statement of intent.
    exclude_buckets:
        Buckets to switch off, e.g. ``("derivatives",)``.

    Returns
    -------
    dict
        ``symbol``, ``horizons``, ``models``, ``enabled_features``,
        ``excluded_buckets``, ``max_rows``.  The first five are JSON-safe, so
        the plan can be embedded in the manifest verbatim.

    Raises
    ------
    FeatureSelectionError
        On an unknown bucket name, a horizon that will not parse, or an empty
        model list.
    """
    block = _block(config)
    name = str(symbol).strip().upper()
    if not name:
        raise ValueError("resolve_plan needs a symbol such as 'BTCUSDT', got an empty value")

    ladder = build_horizon_ladder(
        {**block, "horizons": _horizon_labels(horizons)} if horizons else block
    )

    model_names = (
        [str(item).strip() for item in models if str(item).strip()]
        if models
        else [str(item).strip() for item in (block.get("models") or MODEL_NAMES)]
    )
    if not model_names:
        raise FeatureSelectionError("No models selected; v3.models is empty and no override was given")

    excluded = _validated_exclusions(block, exclude_buckets)
    # `v3.max_rows` is not part of the shipped config; it is honoured if a caller
    # sets it so a smoke run can be declared entirely in YAML.  `run_pipeline`
    # overrides it with its own argument when given.
    declared_max_rows = block.get("max_rows")

    plan: dict[str, object] = {
        "symbol": name,
        "horizons": [horizon.label for horizon in ladder],
        "models": model_names,
        "enabled_features": _bucket_selection(block, excluded),
        "excluded_buckets": list(excluded),
        "max_rows": int(declared_max_rows) if declared_max_rows else None,
    }
    logger.debug(
        "V3 plan %s: horizons %s, models %s, excluded buckets %s",
        name,
        plan["horizons"],
        model_names,
        plan["excluded_buckets"] or "none",
    )
    return plan


def _horizon_labels(horizons: Sequence[str] | Sequence[Horizon] | None) -> list[Any]:
    """Accept labels or built :class:`Horizon` objects as one list of labels.

    Blank entries are dropped rather than handed to the horizon parser, because an
    empty label is a typo and ``Horizon.from_label("")`` reports it as an
    unparseable duration instead of "you passed nothing".
    """
    labels: list[Any] = []
    for item in horizons or ():
        label = item.label if isinstance(item, Horizon) else str(item).strip()
        if label:
            labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# stage bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    """The handles the stages fill in and later stages read.

    A stage body is written as a closure over this object rather than as a
    top-level function taking six arguments, because the stages genuinely are a
    chain: each one consumes what the previous one produced.
    """

    plan: dict[str, Any] = field(default_factory=dict)
    dataset: Any | None = None
    run: Any | None = None
    report: Any | None = None
    model_paths: tuple[Path, ...] = ()


@dataclass
class _StageLog:
    """Ordered stage record for one run, plus the failure bookkeeping.

    The reason a failing stage returns ``(detail, failed)`` rather than
    raising is that the caller must still get to the manifest stage: a run that
    failed at the dataset build has real information to record (which buckets
    were planned, which symbols are affected, what the error was) and that
    information belongs on disk.
    """

    stages: list[PipelineStage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True

    def run(
        self,
        name: str,
        fn: Callable[[], Mapping[str, Any]],
        *,
        context: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Time and record one stage, converting any exception into a warning."""
        started = time.perf_counter()
        where = f" for {context}" if context else ""
        try:
            detail = dict(fn() or {})
            failed = False
        except Exception as exc:
            detail = {"error": f"{type(exc).__name__}: {exc}"}
            failed = True
            self.ok = False
            message = f"stage '{name}'{where} failed: {type(exc).__name__}: {exc}"
            self.warnings.append(message)
            logger.error("%s", message, exc_info=True)
        self.stages.append(
            PipelineStage(name=name, seconds=time.perf_counter() - started, detail=detail)
        )
        return detail, failed

    def warn(self, message: str) -> None:
        """Record a non-fatal caveat that changes what the run can claim."""
        self.warnings.append(message)
        logger.warning("%s", message)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _artefact_paths(value: Any) -> list[str]:
    """Flatten a report artefact bundle (or anything shaped like one) to paths.

    ``write_report`` is owned by another module and may return its bundle as a
    mapping, a list or a single path; this accepts all three so the pipeline does
    not have to be edited when its shape settles.
    """
    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if node is None or depth > 4:
            return
        if isinstance(node, (str, os.PathLike)):
            found.append(str(node))
        elif isinstance(node, Mapping):
            for item in node.values():
                walk(item, depth + 1)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for item in node:
                walk(item, depth + 1)

    walk(value, 0)
    return found


def _relative_names(paths: Sequence[str], out_dir: Path) -> list[str]:
    """Report artefact paths as names relative to ``out_dir`` where possible."""
    names: list[str] = []
    for raw in paths:
        path = Path(raw)
        try:
            names.append(str(path.relative_to(out_dir)))
        except ValueError:
            names.append(path.name)
    return names


def _summarise(value: Any, *, limit: int = 4000) -> Any:
    """Make an arbitrary analysis/report payload safe to serialise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    for accessor in ("to_dict", "as_dict"):
        method = getattr(value, accessor, None)
        if callable(method):
            try:
                return _summarise(method(), limit=limit)
            except Exception:  # pragma: no cover - defensive
                break
    if isinstance(value, Mapping):
        return {str(key): _summarise(item, limit=limit) for key, item in list(value.items())[:200]}
    if isinstance(value, (list, tuple)):
        return [_summarise(item, limit=limit) for item in list(value)[:200]]
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _best_value(table: Any, metric: str) -> float | None:
    """The best (lowest) value of ``metric`` in a pooled-metrics table."""
    try:
        if metric not in table or table.empty:
            return None
        value = float(table[metric].min())
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return value if np.isfinite(value) else None


def _reconcile_models(requested: Sequence[str]) -> tuple[list[str], list[str]]:
    """Map requested model names onto the estimator registry.

    Returns ``(buildable, notes)``.  ``buildable`` is de-duplicated and ordered as
    requested; ``notes`` describes every name that had to be translated or
    dropped, so the omission is reported rather than discovered later as a
    silently absent row in a results table.
    """
    known = {str(name).lower() for name in MODEL_NAMES}
    buildable: list[str] = []
    notes: list[str] = []
    for raw in requested:
        name = str(raw).strip()
        lowered = name.lower()
        if lowered in known:
            resolved = lowered
        elif lowered in MODEL_NAME_ALIASES:
            resolved = MODEL_NAME_ALIASES[lowered]
            notes.append(
                f"model {name!r} is configured as {resolved!r}, which is what "
                f"src.v3.models registers for it"
            )
        else:
            notes.append(
                f"model {name!r} is not registered in src.v3.models "
                f"(available: {list(MODEL_NAMES)}); it is not scored in this run"
            )
            continue
        if resolved not in buildable:
            buildable.append(resolved)
    return buildable, notes


def _analysis_summary(run: Any, log: "_StageLog", symbol: str) -> dict[str, Any]:
    """Per-horizon error analysis, degrading per horizon rather than as a whole.

    ``analyse`` consumes one horizon's out-of-sample predictions, so it is called
    once per horizon.  A horizon it cannot analyse is recorded as an error entry
    and warned about; the report, the models and the manifest are already built by
    this point and must not be thrown away because an optional diagnostic failed.
    """
    summary: dict[str, Any] = {}
    for result in run.results:
        try:
            summary[result.horizon] = _summarise(analyse(result))
        except Exception as exc:
            summary[result.horizon] = {"error": f"{type(exc).__name__}: {exc}"}
            log.warn(f"{symbol}: error analysis unavailable for {result.horizon} ({type(exc).__name__}: {exc})")
    return summary


# Trailing (strictly backward-looking) columns used to label market regimes.
# Each is a feature the model already had at prediction time, so splitting
# performance by them conditions on something known, not on the outcome.
_REGIME_VOL_COLUMNS = ("rolling_volatility_20", "ctx_1d_volatility", "ctx_4h_volatility")
_REGIME_TREND_COLUMNS = ("price_vs_sma50", "sma20_vs_sma50", "macd_histogram")


def _attach_regime_context(
    results: Sequence[Any], dataset: Any, log: "_StageLog", symbol: str
) -> None:
    """Give each result the trailing series the regime breakdown needs.

    ``src.v3.analysis`` can label regimes but needs the values to label *from*,
    and those live in the dataset feature frame that a ``HorizonResult`` does
    not carry.  The renderer refuses to invent them - deriving a regime from
    the realised target would condition the breakdown on its own answer - so
    the stage that still holds the frame attaches them here instead.

    Best-effort by design: a missing or renamed column degrades the regime
    section, which is a loss of detail, and must never cost the models.
    """
    if not results or dataset is None:
        return
    frame = getattr(dataset, "frame", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    context: dict[str, pd.Series] = {}
    for key, candidates in (
        ("regime_values", _REGIME_VOL_COLUMNS),
        ("trend_values", _REGIME_TREND_COLUMNS),
    ):
        for column in candidates:
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not series.empty:
                    context[key] = series
                break
    missing = {"regime_values", "trend_values"} - set(context)
    if missing:
        log.warn(
            f"{symbol}: regime breakdown degraded; no trailing column found for "
            f"{sorted(missing)} (looked for {list(_REGIME_VOL_COLUMNS + _REGIME_TREND_COLUMNS)})"
        )
    if not context:
        return
    for result in results:
        # `HorizonResult` is frozen, and rightly so: these objects are the
        # scored payload.  The field is a mutable dict precisely so the labels
        # can be filled in place without a rebind; nothing here touches a
        # metric, a prediction or the geometry.
        result.regime_context.update(context)


def _dataset_detail(dataset: Any, plan: Mapping[str, Any], log: "_StageLog") -> dict[str, Any]:
    """Summarise a built dataset and shout if the history is suspiciously thin.

    The row count is the single most load-bearing number in a V3 run: it decides
    whether the long horizons are trainable at all.  It is logged per symbol
    together with the buckets that survived, and a thin history is a warning,
    because a run that trains a 180d model on 2,577 rows reports success just as
    loudly as one that trains it on 40,833.
    """
    columns = list(getattr(dataset, "feature_columns", []))
    used = sorted({bucket_of_feature(name) for name in columns})
    excluded = list(plan.get("excluded_buckets") or [])
    kept = [name for name in used if name not in excluded]
    n_rows = int(len(dataset))

    longest = max(dataset.horizons, key=lambda horizon: (horizon.delta, horizon.label))
    floor = MIN_ROWS_PER_HORIZON_BAR * longest.nominal_bars
    labelled = {horizon.label: dataset.n_labelled(horizon) for horizon in dataset.horizons}

    logger.info(
        "V3 %s: buckets [%s]%s -> %s rows x %d features",
        dataset.symbol,
        ", ".join(kept) or "none",
        f" (excluded: {', '.join(excluded)})" if excluded else "",
        f"{n_rows:,}",
        len(columns),
    )
    logger.info(
        "V3 %s: labelled rows per horizon %s",
        dataset.symbol,
        {label: f"{count:,}" for label, count in labelled.items()},
    )

    if n_rows < floor:
        log.warn(
            f"{dataset.symbol}: only {n_rows:,} usable rows for a longest horizon of "
            f"{longest.label} ({longest.nominal_bars} bars). A history this short is usually a "
            f"truncated external cache rather than a short market history - on BTCUSDT the "
            f"futures cache stops at 2022-03-04 and funding at 2022-06-16, which leaves ~2.6k "
            f"rows. Re-run with exclude_buckets=['derivatives'] to get the full 8-horizon ladder."
        )
    missing_sources = list(getattr(dataset, "missing_sources", ()))
    unplanned = [name for name in used if name not in kept]
    if unplanned and missing_sources:
        # A bucket can vanish for a reason the plan never asked for: its external
        # cache is not there.  Same symptom as the truncation trap, so it is
        # named rather than left for a reader to infer from the column count.
        log.warn(
            f"{dataset.symbol}: bucket(s) {', '.join(unplanned)} are absent without having been "
            f"excluded - source(s) {', '.join(missing_sources)} are not cached, and every row "
            f"that depended on them was dropped."
        )
    for label, count in labelled.items():
        if count <= 0:
            log.warn(f"{dataset.symbol}: horizon {label} has no labelled rows and cannot be scored")

    return {
        "symbol": dataset.symbol,
        "rows": n_rows,
        "n_features": len(columns),
        "feature_columns": columns,
        "buckets": used,
        "buckets_kept": kept,
        "excluded_buckets": excluded,
        "feature_groups": list(getattr(dataset, "feature_groups", ())),
        "horizons": [horizon.label for horizon in dataset.horizons],
        "labelled_rows": labelled,
        "span": [str(dataset.frame.index.min()), str(dataset.frame.index.max())],
        "drop_report": dict(getattr(dataset, "drop_report", {})),
        "missing_sources": missing_sources,
        "max_rows": getattr(dataset, "max_rows", None),
    }


def _selected_params(fold_metrics: Any, model_name: str) -> dict[str, Any]:
    """Hyper-parameters the harness picked for ``model_name``, from the last fold.

    ``run_horizon`` records the selected parameters per fold and does not retain
    the fitted estimators, so the persist stage re-reads them off the fold table
    (``repr(dict)`` in the ``params`` column) instead of guessing defaults.  An
    unparseable cell yields no parameters, which is a documented fallback rather
    than a failure: a model saved with default settings is still a usable
    artifact, and the manifest says which it was.
    """
    try:
        rows = fold_metrics[fold_metrics["model"] == model_name]
    except (KeyError, TypeError, AttributeError):
        return {}
    for raw in reversed(list(rows["params"])):
        try:
            value = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    return {}


def _persist_models(
    run: Any, dataset: Any, models: Sequence[str], out_dir: Path, seed: int
) -> tuple[list[Path], dict[str, Any]]:
    """Refit and save one estimator per (model, horizon) under ``out_dir``.

    ``run_walk_forward`` deliberately keeps no fitted estimator - holding
    ``n_horizons x n_folds`` models would be the run's largest memory cost - so
    the deployment artifact is refitted here on the horizon's labelled rows with
    the parameters the walk-forward selected, *after* evaluation has finished.
    The reported metrics remain those of the out-of-sample folds; this artifact is
    the thing you would score tomorrow's candles with, and the manifest says so.
    """
    frame = dataset.frame
    paths: list[Path] = []
    inventory: dict[str, Any] = {}
    for result in run.results:
        column = result.target_column
        labelled = frame.loc[frame[column].notna()] if column in frame else frame
        for model_name in result.models or ():
            if model_name not in models:
                continue
            params = _selected_params(result.fold_metrics, model_name)
            features = [name for name in result.feature_columns if name in labelled.columns]
            estimator = fit_model(
                build_model(model_name, seed=seed, **params),
                labelled[features],
                labelled[column].to_numpy(dtype=float),
            )
            path = save_model(estimator, artifact_path(out_dir, model_name, result.horizon))
            paths.append(path)
            entry = inventory.setdefault(model_name, {"horizons": {}, "refit_on_all_labelled_rows": True})
            entry["horizons"][result.horizon] = {
                "artifact": str(path),
                "params": params,
                "n_rows": int(len(labelled)),
                "n_features": len(features),
            }
    return paths, inventory


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def default_out_dir(config: Config) -> Path:
    """The configured V3 report directory, resolved against the config root.

    ``v3.outputs.report_dir`` is declared relative to the project, so it is
    re-resolved against ``config.paths.root`` - which is what makes a rebased
    test config write into the temporary root instead of the real ``reports/``.
    """
    outputs = dict(_block(config).get("outputs") or {})
    declared = Path(str(outputs.get("report_dir") or "reports/experiments/v3")).expanduser()
    return declared if declared.is_absolute() else Path(config.paths.root) / declared


def default_symbols(config: Config) -> tuple[str, ...]:
    """The V3 universe: ``v3.symbols``, or the primary symbol alone."""
    block = _block(config)
    symbols = [str(item).strip().upper() for item in (block.get("symbols") or []) if str(item).strip()]
    primary = str(block.get("primary_symbol") or "").strip().upper()
    if not symbols:
        return (primary,) if primary else (config.symbol,)
    return tuple(symbols)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_pipeline(
    config: Config,
    symbol: str | None = None,
    horizons: Sequence[str] | Sequence[Horizon] | None = None,
    models: Sequence[str] | None = None,
    exclude_buckets: Sequence[str] = (),
    out_dir: Path | None = None,
    max_rows: int | None = None,
    n_splits: int = 3,
    test_fraction: float = 0.08,
    validation_fraction: float = 0.08,
    save_models: bool = True,
    seed: int = 42,
    *,
    alpha: float = 0.1,
    embargo: str | None = None,
    write_reports: bool = True,
) -> PipelineResult:
    """Run one symbol's full V3 experiment and write its artefacts.

    Parameters
    ----------
    config:
        A config carrying a ``v3:`` block.  Its paths are never written to;
        every output goes under ``out_dir``.
    symbol:
        Defaults to ``v3.primary_symbol``.
    horizons, models, exclude_buckets:
        Overrides of the configured ladder / model set / bucket selection.
    out_dir:
        Where the manifest, the report and the model artifacts go.  Defaults to
        ``v3.outputs.report_dir``.  Nothing is ever written outside it.
    max_rows:
        Cap the dataset at the most recent N rows.  Applied before labels are
        built, so the truncated tail is honestly NaN.
    n_splits, test_fraction, validation_fraction:
        Walk-forward geometry.  Defaults are deliberately smaller than
        ``v3.validation`` so that an ad-hoc run is quick; pass the configured
        values (or read them from the block) for a headline experiment.
    save_models:
        Refit and persist one estimator per (model, horizon).
    seed:
        Seeded before anything runs, so two runs of the same command agree.
    alpha:
        Miscoverage level of the split-conformal band.
    embargo:
        Embargo policy; defaults to ``v3.validation.embargo`` (``"horizon"``).
    write_reports:
        ``False`` skips the report stage.  The run still succeeds, but says so.

    Returns
    -------
    PipelineResult
        Never raises for a stage failure: the failure is in ``warnings`` and
        ``ok`` is ``False``, and the manifest on disk has the details.
    """
    block = _block(config)
    name = str(symbol or block.get("primary_symbol") or config.symbol).strip().upper()
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    target.mkdir(parents=True, exist_ok=True)

    # First, before any RNG is drawn: a seed set afterwards does nothing.
    set_global_seed(seed)

    started = time.perf_counter()
    log = _StageLog()
    state = _RunState()
    logger.info("=== V3 pipeline start: %s -> %s ===", name, target)

    # ---------------------------------------------------------------- 1. plan
    def _plan_stage() -> dict[str, Any]:
        resolved = resolve_plan(
            config, name, horizons=horizons, models=models, exclude_buckets=exclude_buckets
        )
        if max_rows is not None:
            if int(max_rows) < 1:
                raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")
            resolved["max_rows"] = int(max_rows)
        buildable, notes = _reconcile_models(resolved["models"])  # type: ignore[arg-type]
        if not buildable:
            raise FeatureSelectionError(
                f"None of the requested models {list(resolved['models'])} can be built; "
                f"src.v3.models registers {list(MODEL_NAMES)}"
            )
        for note in notes:
            log.warn(f"{name}: {note}")
        resolved["models_to_run"] = buildable
        resolved["out_dir"] = str(target)
        resolved["seed"] = int(seed)
        resolved["n_splits"] = int(n_splits)
        resolved["test_fraction"] = float(test_fraction)
        resolved["validation_fraction"] = float(validation_fraction)
        resolved["alpha"] = float(alpha)
        resolved["embargo"] = (
            embargo if embargo is not None else (block.get("validation") or {}).get("embargo", "horizon")
        )
        resolved["save_models"] = bool(save_models)
        resolved["write_reports"] = bool(write_reports)
        state.plan = resolved
        return dict(resolved)

    plan_detail, failed = log.run("plan", _plan_stage, context=name)
    state.plan = plan_detail
    models_to_run: list[str] = list(state.plan.get("models_to_run") or [])
    # -------------------------------------------------------------- 2. dataset
    if not failed:

        def _dataset_stage() -> dict[str, Any]:
            built = build_v3_dataset(
                config,
                name,
                horizons=state.plan["horizons"],
                enabled_features=state.plan["enabled_features"],
                max_rows=state.plan["max_rows"],
            )
            state.dataset = built
            return _dataset_detail(built, state.plan, log)

        _, failed = log.run("dataset", _dataset_stage, context=name)

    # ------------------------------------------------------------ 3. walkforward
    # `results_ok` tracks the walk-forward, not the report: a broken renderer
    # must not cost the model artifacts, which are the deliverable.
    results_ok = not failed
    if results_ok:

        def _walkforward_stage() -> dict[str, Any]:
            ladder = build_horizon_ladder({**block, "horizons": state.plan["horizons"]})
            outcome = run_walk_forward(
                state.dataset,
                ladder,
                models_to_run,
                seed=seed,
                n_splits=n_splits,
                test_fraction=test_fraction,
                validation_fraction=validation_fraction,
                embargo=str(state.plan.get("embargo") or "horizon"),
                alpha=float(alpha),
            )
            state.run = outcome
            _attach_regime_context(outcome.results, state.dataset, log, name)
            return {
                "horizons": list(outcome.horizons),
                "models": list(outcome.config.get("models", [])),
                "n_features": len(outcome.feature_columns),
                "seconds": float(outcome.seconds),
                "dataset_rows": int(outcome.dataset.get("n_rows", 0)),
                "n_predictions": {
                    result.horizon: int(len(result.predictions)) for result in outcome.results
                },
                "best": {
                    result.horizon: {
                        "model": result.best_model(),
                        "rmse": _best_value(result.pooled_metrics, "rmse"),
                    }
                    for result in outcome.results
                },
            }

        _, failed = log.run("walkforward", _walkforward_stage, context=name)
    results_ok = results_ok and not failed

    # ---------------------------------------------------------------- 4. report
    if results_ok:  # the report renders results that now exist

        def _report_stage() -> dict[str, Any]:
            if not write_reports:
                log.warn(
                    f"{name}: report stage skipped by request (--no-report); no report artefacts written"
                )
                return {"skipped": True, "reason": "disabled by caller"}
            if write_report is None:
                log.warn(
                    f"{name}: src.v3.report is not importable, so no report artefacts were "
                    f"written. The walk-forward results are complete and are in the manifest."
                )
                return {"skipped": True, "reason": "src.v3.report unavailable"}
            extra = {
                "symbol": name,
                "plan": {
                    key: state.plan[key]
                    for key in ("horizons", "models", "excluded_buckets", "enabled_features")
                },
            }
            try:
                artefacts = write_report(state.run, target, extra=extra)
            except TypeError:
                # A reporter whose signature is narrower than documented still
                # produces the artefacts; only the merged extras are lost.
                artefacts = write_report(state.run, target)
            state.report = artefacts
            detail: dict[str, Any] = {
                "skipped": False,
                "summary": _artefact_paths(getattr(artefacts, "summary", None)),
                "tables": _artefact_paths(getattr(artefacts, "tables", None)),
                "figures": _artefact_paths(getattr(artefacts, "figures", None)),
            }
            if analyse is not None:
                detail["analysis"] = _analysis_summary(state.run, log, name)
            return detail

        _, failed = log.run("report", _report_stage, context=name)

    # --------------------------------------------------------------- 5. persist
    if results_ok:  # the models are fitted from the walk-forward, not from the report

        def _persist_stage() -> dict[str, Any]:
            if not save_models:
                return {"skipped": True, "reason": "save_models=False", "artifacts": []}
            paths, inventory = _persist_models(
                state.run, state.dataset, models_to_run, target, seed
            )
            state.model_paths = tuple(paths)
            return {
                "skipped": False,
                "artifacts": [str(path) for path in paths],
                "inventory": inventory,
            }

        _, failed = log.run("persist", _persist_stage, context=name)

    # -------------------------------------------------------------- 6. manifest
    manifest_path = target / MANIFEST_NAME
    artefact_names: list[str] = [MANIFEST_NAME]
    report_stage = next((s for s in log.stages if s.name == "report"), None)
    if report_stage is not None and not report_stage.detail.get("skipped"):
        for group in ("summary", "tables", "figures"):
            artefact_names += _relative_names(report_stage.detail.get(group) or [], target)
    else:
        log.warn(
            f"{name}: no report artefacts were produced, so validate_outputs() requires the "
            f"manifest only"
        )

    manifest: dict[str, Any] = {
        "generated_at": format_timestamp(utc_now()),
        "symbol": name,
        "ok": log.ok,
        "seconds": time.perf_counter() - started,
        "config": {
            "source": str(config.source_path),
            "root": str(config.paths.root),
            "fingerprint": config.fingerprint(),
            "v3": block,
        },
        "plan": dict(state.plan),
        "outputs": {
            "out_dir": str(target),
            "manifest": str(manifest_path),
            "required_artifacts": list(dict.fromkeys(artefact_names)),
            "report": _artefact_paths(state.report),
            "models": [str(path) for path in state.model_paths],
        },
        "stages": [
            {"name": stage.name, "seconds": stage.seconds, "detail": stage.detail}
            for stage in log.stages
        ],
        "stage_seconds": {stage.name: stage.seconds for stage in log.stages},
        "horizons": {},
        "model_inventory": {},
        "warnings": list(log.warnings),
    }
    if state.run is not None:
        for result in state.run.results:
            manifest["horizons"][result.horizon] = result.to_dict()
        persist_stage = next((s for s in log.stages if s.name == "persist"), None)
        manifest["model_inventory"] = (persist_stage.detail.get("inventory") if persist_stage else {}) or {}

    manifest["ok"] = log.ok
    manifest["warnings"] = list(log.warnings)

    def _manifest_stage() -> dict[str, Any]:
        save_json(manifest, manifest_path)
        logger.info("V3 %s: manifest -> %s", name, manifest_path)
        return {"path": str(manifest_path), "bytes": manifest_path.stat().st_size}

    log.run("manifest", _manifest_stage, context=name)

    logger.info(
        "=== V3 %s %s in %.1fs (%s) -> %s ===",
        name,
        "complete" if log.ok else "FAILED",
        time.perf_counter() - started,
        ", ".join(f"{s.name}={s.seconds:.2f}s" for s in log.stages) or "no stages",
        target,
    )

    return PipelineResult(
        symbol=name,
        run=state.run,
        stages=tuple(log.stages),
        report=state.report,
        model_paths=state.model_paths,
        warnings=tuple(log.warnings),
        ok=log.ok,
    )


def run_all_symbols(
    config: Config,
    symbols: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, PipelineResult]:
    """Run :func:`run_pipeline` over the V3 universe.

    Each symbol gets its own sub-directory (``<out_dir>/<SYMBOL>``) even when the
    caller does not pass one, because two symbols writing the same manifest would
    leave only the last one on disk - and a multi-symbol run that silently loses
    four of five results is worse than one that failed.

    A symbol that fails does not stop the others: the mapping it returns carries
    ``ok=False`` for it, and the caller decides.  Failures are logged loudly.
    """
    universe = [str(item).strip().upper() for item in (symbols or default_symbols(config))]
    base = kwargs.pop("out_dir", None)
    root = Path(base) if base is not None else default_out_dir(config)
    logger.info("V3: running %d symbol(s) under %s", len(universe), root)

    results: dict[str, PipelineResult] = {}
    for symbol in universe:
        try:
            results[symbol] = run_pipeline(config, symbol=symbol, out_dir=root / symbol, **kwargs)
        except Exception as exc:  # a plan-level failure the stage log could not hold
            message = f"symbol {symbol} could not be run: {type(exc).__name__}: {exc}"
            logger.error("%s", message, exc_info=True)
            results[symbol] = PipelineResult(
                symbol=symbol,
                run=None,
                stages=(),
                report=None,
                model_paths=(),
                warnings=(message,),
                ok=False,
            )
    return results


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_outputs(out_dir: Path) -> dict[str, object]:
    """Check that an output directory holds a complete run.

    "Complete" is defined by the manifest, because the manifest is the run's own
    statement of what it owed: a run that skipped reporting records no report
    artefacts as required, and this reports that run as consistent rather than
    demanding files it never claimed to write.  When the manifest is missing or
    unreadable the module-level :data:`REQUIRED_ARTIFACTS` is the yardstick
    instead, which is what makes a truncated directory visible rather than
    vacuously valid.

    Returns
    -------
    dict
        ``ok`` (no missing file), ``missing`` (names relative to ``out_dir``),
        ``manifest`` (the parsed manifest, empty when unreadable).
    """
    root = Path(out_dir)
    missing: list[str] = []
    manifest: dict[str, Any] = {}

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        missing.append(MANIFEST_NAME)
    else:
        try:
            loaded = load_json(manifest_path)
            manifest = dict(loaded) if isinstance(loaded, Mapping) else {}
            if not manifest:
                missing.append(MANIFEST_NAME)
        except (ValueError, OSError) as exc:
            logger.error("V3 validate: %s is not readable JSON (%s)", manifest_path, exc)
            missing.append(MANIFEST_NAME)

    recorded = (manifest.get("outputs") or {}).get("required_artifacts") if manifest else None
    required = list(recorded) if isinstance(recorded, (list, tuple)) and recorded else [MANIFEST_NAME]
    if not manifest:
        required = [MANIFEST_NAME, *REQUIRED_ARTIFACTS]

    for name in required:
        if not (root / str(name)).exists():
            missing.append(str(name))

    status = "OK" if not missing else f"MISSING {len(missing)}"
    logger.info("V3 validate %s: %s (%d required)", root, status, len(required))
    return {"ok": not missing, "missing": missing, "manifest": manifest}
