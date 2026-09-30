"""V3 report renderer: five CSVs, six PNGs and one auditable ``summary.md``.

Why this module exists
----------------------
V3 produces numbers in three places that never meet.  ``src.v3.walkforward``
returns per-fold and pooled metrics, ``src.v3.analysis`` slices them by regime
and calibration bin, and until now nothing read them together.  An unrendered
results table is a claim nobody can check, and the failure mode that matters is
not a crash - it is a document that looks complete, quotes one best-RMSE number,
and silently omits that the winner was ``trailing_mean``.

So this renderer is opinionated about three things.

1. **Every artefact derives from the pooled out-of-sample frame.**  Nothing is
   re-scored here.  The renderer's job is to make the numbers auditable, which
   means it must never produce a second, friendlier version of them.  Metric
   values are read, formatted and plotted; they are recomputed from raw
   predictions only in the two explicitly labelled local fallbacks below.
2. **Absence is stated, never implied.**  ``src.v3.analysis`` is written in
   parallel with this module and may be missing entirely, so every
   analysis-derived table degrades to an explicit one-row "not available"
   placeholder and every analysis-derived section degrades to a sentence saying
   so.  A reader must be able to tell "no regime breakdown was computed" apart
   from "the regime breakdown came back empty".
3. **A missing number prints as ``n/a``,** never as a blank cell and never as
   the string ``nan``.  A blank cell in a results table reads as a measured
   zero; a literal ``nan`` reads as a rendering bug.  Every value passes
   through :func:`_fmt` so a display defect cannot diverge from the arithmetic.

Local fallbacks, and exactly where they are used
------------------------------------------------
Two figures can still be drawn honestly when ``src.v3.analysis`` is absent,
because the ingredients are already in ``HorizonResult.predictions``:

* ``prediction_calibration.png`` rebuilds an equal-width reliability table from
  the pooled ``prob_calibrated`` column, using the same bin edges and the same
  column names ``src.v3.metrics.calibration_table`` emits, and puts its source
  in the title.
* ``return_distributions.png`` plots the realised target beside the best
  model's prediction straight from the pooled frame.

Nothing else is rebuilt locally.  A regime breakdown in particular *cannot* be
recovered here, because ``HorizonResult`` carries predictions and split geometry
but not the feature-level regime labels a breakdown would need.  And although
``return_distributions.csv`` degrades to a placeholder while its figure does
not, that asymmetry is deliberate: the CSV is the artefact downstream code
reads, and a schema that depends on import timing is worse than an honest gap.

Self-containment
----------------
V3 imports nothing from ``src.evaluation``.  The V1/V2 baseline is frozen and
this renderer shares nothing with it but the matplotlib rc conventions of
``src.evaluation.plots``, which are duplicated here on purpose so V3 can be
read, tested and run without the V1 evaluation stack.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("v3.report")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

try:  # pragma: no cover - depends on the sibling module landing
    from src.v3 import analysis as _analysis
except ImportError:  # the common case while analysis is still in development
    _analysis = None
except Exception as exc:  # a half-written sibling must not take the report down
    logger.warning("src.v3.analysis failed to import (%s); report sections degrade", exc)
    _analysis = None

try:  # keeps the naive-baseline list in sync without making it a hard dependency
    from src.v3.walkforward import BASELINES as _NAIVE_BASELINES
except Exception:  # pragma: no cover - only when the harness is unavailable
    _NAIVE_BASELINES = ("zero", "mean", "trailing_mean")


# ---------------------------------------------------------------- public surface

#: The five CSV basenames, in the order :func:`write_tables` writes them.
REQUIRED_TABLES: tuple[str, ...] = (
    "horizon_comparison.csv",
    "model_comparison.csv",
    "regime_analysis.csv",
    "prediction_calibration.csv",
    "return_distributions.csv",
)

#: The six PNG basenames, in the order :func:`write_report` writes them.
REQUIRED_FIGURES: tuple[str, ...] = (
    "horizon_rmse.png",
    "horizon_ic.png",
    "interval_coverage.png",
    "prediction_calibration.png",
    "return_distributions.png",
    "model_comparison.png",
)

#: Name of the markdown document :func:`write_summary` produces.
SUMMARY_NAME = "summary.md"

#: Pooled-metric columns the renderer knows how to read.  Absent ones resolve to
#: ``NaN`` rather than raising, because ``brier_empirical`` legitimately
#: disappears when a fold held too few residuals to fit a conformal band.
_METRIC_KEYS: tuple[str, ...] = (
    "rmse",
    "mae",
    "mse",
    "r2",
    "mape",
    "mape_coverage",
    "direction_accuracy",
    "balanced_direction_accuracy",
    "spearman_ic",
    "long_short_spread",
    "n",
    "coverage_lo",
    "coverage_hi",
    "brier_empirical",
    "brier_calibrated",
)

#: Columns of ``HorizonResult.interval_summary``, which averages the per-fold
#: conformal summaries rather than recomputing coverage over the pooled rows.
_INTERVAL_KEYS: tuple[str, ...] = (
    "mean_fold_coverage",
    "nominal_coverage",
    "mean_interval_width",
    "mean_calibration_rows",
)

#: Analysis entry points this module may call, and reports the status of.
_ANALYSIS_FUNCTIONS: tuple[str, ...] = (
    "analyse",
    "calibration_report",
    "prediction_interval_report",
    "decile_report",
    "return_distribution_report",
    "evaluate_by_regime",
)

#: Below this many pooled out-of-sample rows a coverage or IC estimate is
#: dominated by sampling noise and is not a property of the model.  Flagged in
#: the caveats rather than hidden, because *which* horizons trip this is itself
#: a finding about the run.
_SMALL_PREDICTION_ROWS = 200

#: Split-conformal bands are finite-sample and marginal, so realised coverage
#: wobbles around the nominal level even when the construction is correct.  A
#: band within this distance of nominal counts as having met its target; beyond
#: it the report says so in words rather than rounding the gap away.
_COVERAGE_TOLERANCE = 0.02

#: Histogram resolution for the return-distribution panels.
_DISTRIBUTION_BINS = 40

#: Visual language, matching ``src.evaluation.plots`` so the V2 and V3 figures
#: read as one project without sharing an import.
_STYLE: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
}

_COLORS: dict[str, str] = {
    "model": "#4c78a8",
    "benchmark": "#9aa5b1",
    "target": "#54a24b",
    "down": "#d62728",
    "ink": "#1f2933",
}


@dataclass(frozen=True)
class ReportArtefacts:
    """Every path a single :func:`write_report` call produced.

    Returned rather than only logged because the caller is usually a pipeline
    that wants the artefact list recorded beside the run that produced it, and
    because "did the report actually write twelve files" is a question a test can
    ask without parsing log output.
    """

    directory: Path
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    summary: Path

    def all(self) -> tuple[Path, ...]:
        """All twelve artefacts: the tables, then the figures, then the summary."""
        return (*self.tables, *self.figures, self.summary)


# ------------------------------------------------------------------- formatting


def _is_missing(value: Any) -> bool:
    """True for ``None``, ``NaN``, ``NaT`` and ``pd.NA``.

    ``pd.isna`` is vectorised, so a list or array argument returns an array whose
    truthiness raises; that is caught and reported as "not missing", because a
    container is never a single missing cell.
    """
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> float | None:
    """``float(value)`` when it is a finite number, otherwise ``None``.

    The workhorse for "is this cell usable": it treats a non-numeric object, a
    NaN and an infinity identically, which is what every guard in this module
    actually means by "missing".
    """
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fmt(value: Any, digits: int = 4) -> str:
    """Render a value for markdown, mapping every non-finite float to ``n/a``.

    This is the only formatter in the module, so the ``n/a`` policy cannot drift
    between the tables and the prose.
    """
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    number = _finite(value)
    if number is None:
        return "n/a" if not isinstance(value, (str, bytes)) else str(value)
    return f"{number:,.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    """Render a fraction as a percentage; missing stays ``n/a``."""
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.{digits}f}%"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Assemble markdown table lines; an empty body still yields a valid table."""
    lines = [
        "| " + " | ".join(str(head) for head in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _md_frame(
    frame: pd.DataFrame, *, columns: Sequence[str] | None = None, digits: int = 4
) -> list[str]:
    """Render a DataFrame as a markdown table, or a note when it has no rows."""
    if frame is None or frame.empty:
        return ["_Not available: no rows were produced for this section._"]
    used = [str(c) for c in (columns if columns is not None else frame.columns)]
    rows = [
        [_fmt(row[c], digits) if c in row.index else "n/a" for c in used]
        for _, row in frame.iterrows()
    ]
    return _md_table([c.replace("_", " ") for c in used], rows)


# ------------------------------------------------------------ analysis plumbing


def _analysis_call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call an ``src.v3.analysis`` function, or return ``None`` if unusable.

    The sibling module is developed alongside this one, so both its import and
    any individual function may be missing or broken.  A report with a missing
    section is recoverable; a report that does not exist is not, so every call is
    guarded and the failure is logged rather than raised.
    """
    func = getattr(_analysis, name, None) if _analysis is not None else None
    if func is None:
        logger.info("src.v3.analysis.%s is unavailable; that section degrades", name)
        return None
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.warning("src.v3.analysis.%s failed (%s); that section degrades", name, exc)
        return None


def _analysis_status() -> dict[str, str]:
    """Import-level availability of each analysis entry point.

    Surfaced in the summary so a reader can see which sections were computed
    rather than discovering an empty section and guessing why.
    """
    return {
        name: "available" if getattr(_analysis, name, None) is not None else "not available"
        for name in _ANALYSIS_FUNCTIONS
    }


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Coerce an analysis return value to a DataFrame, or ``None``."""
    return value if isinstance(value, pd.DataFrame) else None


def _frames_from_mapping(value: Any, *tokens: str) -> list[pd.DataFrame]:
    """Every DataFrame in a mapping whose key contains one of ``tokens``.

    ``analyse`` returns a dict of frames and its exact keys are the other
    author's choice, so the renderer matches on meaning ("regime", "calib")
    rather than binding to a spelling a rename would break.  There is
    deliberately *no* "return some other frame" fallback: silently substituting
    a distribution table for a missing regime table is precisely the kind of
    quiet substitution this module exists to prevent.
    """
    if not isinstance(value, Mapping):
        return []
    return [
        frame
        for key, frame in value.items()
        if isinstance(frame, pd.DataFrame)
        and any(token in str(key).lower() for token in tokens)
    ]


def _is_placeholder(frame: pd.DataFrame | None) -> bool:
    """True for the one-row "not available" frame :func:`_unavailable` builds."""
    return (
        frame is not None
        and not frame.empty
        and "status" in frame.columns
        and str(frame["status"].iloc[0]) == "not available"
    )


def _unavailable(source: str, reason: str) -> pd.DataFrame:
    """The one-row placeholder standing in for a missing analysis table.

    A schema-stable ``status``/``source``/``reason`` row keeps downstream readers
    working: they receive a file with columns and a row, and the row states
    plainly that nothing was computed.
    """
    return pd.DataFrame([{"status": "not available", "source": source, "reason": reason}])


def _tag(frame: pd.DataFrame, horizon: str, source: str) -> pd.DataFrame:
    """Stamp a frame with its provenance, without clobbering its own columns.

    The analysis reports already carry a per-row ``horizon`` column that is more
    specific than anything the renderer knows, so that one is left alone.

    ``source`` needs care: it is the renderer's provenance stamp on the tables that
    do not already use the name, but ``calibration_report`` uses ``source`` for the
    probability *route* - ``empirical`` versus ``calibrated`` - which is the
    distinction the table exists to make.  Overwriting it there would replace the
    finding with the name of the function that produced it, so an existing
    ``source`` column is preserved and the renderer's stamp goes to
    ``analysis_source``.  Assignment rather than ``insert`` is also what keeps a
    renamed analysis column from turning every write into a ``ValueError``.
    """
    tagged = frame.copy()
    provenance = "analysis_source" if "source" in tagged.columns else "source"
    tagged[provenance] = source
    if "horizon" in tagged.columns:
        return tagged[["horizon", *tagged.columns.drop("horizon")]]
    tagged.insert(0, "horizon", horizon)
    return tagged


# ------------------------------------------------------------------ run readers


def _results(run: Any) -> list[Any]:
    """The ``HorizonResult`` list of a run, tolerating a bare sequence."""
    results = getattr(run, "results", None)
    if results is None and isinstance(run, Sequence) and not isinstance(run, (str, bytes)):
        results = list(run)
    return list(results or [])


def _run_symbol(run: Any) -> str:
    return str(getattr(run, "symbol", "") or "n/a")


def _horizon_of(result: Any) -> str:
    return str(getattr(result, "horizon", "") or "n/a")


def _prediction_rows(result: Any) -> int:
    """Number of pooled prediction rows, preferring the frame over the summary."""
    predictions = getattr(result, "predictions", None)
    if isinstance(predictions, pd.DataFrame):
        return int(len(predictions))
    coverage = getattr(result, "coverage", None)
    if isinstance(coverage, Mapping):
        return int(coverage.get("prediction_rows", 0) or 0)
    return 0


def _pred_column(result: Any, model: str) -> pd.Series | None:
    """The prediction column of one model, or ``None`` when it does not exist."""
    predictions = getattr(result, "predictions", None)
    column = f"{model}_pred"
    if isinstance(predictions, pd.DataFrame) and column in predictions:
        return predictions[column]
    return None


def _numeric(values: Any) -> np.ndarray:
    """Coerce a column to a finite float array, dropping what will not convert."""
    if values is None:
        return np.empty(0, dtype=float)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _drawable(values: np.ndarray) -> np.ndarray:
    """Only the finite entries of a float array, for series that are actually drawn.

    NaN must be removed *before* anything reads a length, an ``n=`` label or a
    quantile: ``numpy`` silently drops NaN inside ``histogram`` but keeps it in
    ``.size``, so a panel whose every prediction failed would be captioned
    ``n=4,320`` over four visible bars - a lie drawn in the legend.  Dropping here
    makes the count in the label the count on the screen.
    """
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)] if array.size else array


def _pooled_records(run: Any) -> list[dict[str, Any]]:
    """Flatten every ``(horizon, model)`` pooled-metric cell into a plain dict.

    The renderer never indexes ``pooled_metrics`` directly.  A direct
    ``.loc[model, "rmse"]`` is a crash the first time a column is absent, and an
    absent column is a normal outcome rather than a bug: ``brier_empirical`` is
    gone whenever a fold held too few residuals to fit a conformal band.
    Normalising once to a fixed key set means every reader below can use ``.get``
    and none of them can raise.
    """
    records: list[dict[str, Any]] = []
    for result in _results(run):
        pooled = getattr(result, "pooled_metrics", None)
        indexed = pooled.to_dict("index") if isinstance(pooled, pd.DataFrame) else {}
        models = [str(i) for i in indexed] or list(getattr(result, "models", None) or []) or ["n/a"]
        for model in models:
            row = indexed.get(model) or {}
            records.append(
                {
                    "symbol": str(getattr(result, "symbol", "") or _run_symbol(run)),
                    "horizon": _horizon_of(result),
                    "model": model,
                    "n_predictions": _prediction_rows(result),
                    **{key: row.get(key, np.nan) for key in _METRIC_KEYS},
                }
            )
    return records


def _interval_records(run: Any) -> list[dict[str, Any]]:
    """Flatten ``HorizonResult.interval_summary`` into plain dicts."""
    records: list[dict[str, Any]] = []
    for result in _results(run):
        summary = getattr(result, "interval_summary", None)
        indexed = summary.to_dict("index") if isinstance(summary, pd.DataFrame) else {}
        models = [str(i) for i in indexed] or list(getattr(result, "models", None) or []) or ["n/a"]
        for model in models:
            row = indexed.get(model) or {}
            records.append(
                {
                    "horizon": _horizon_of(result),
                    "model": model,
                    **{key: row.get(key, np.nan) for key in _INTERVAL_KEYS},
                }
            )
    return records


def _horizons(run: Any, records: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """Horizon labels in the run's own order, with any stragglers appended."""
    ordered: list[str] = []
    for label in list(getattr(run, "horizons", None) or []):
        ordered.append(str(label))
    ordered += [_horizon_of(result) for result in _results(run)]
    ordered += [str(record.get("horizon", "")) for record in records]
    return list(dict.fromkeys(label for label in ordered if label))


def _by_horizon(
    records: Sequence[Mapping[str, Any]], horizons: Sequence[str]
) -> dict[str, list[Mapping[str, Any]]]:
    """Group records by horizon, in the run's horizon order."""
    return {label: [r for r in records if r.get("horizon") == label] for label in horizons}


def _extreme(
    group: Sequence[Mapping[str, Any]], key: str, *, largest: bool
) -> tuple[str | None, float]:
    """The model with the best (or worst) finite ``key``; ``(None, NaN)`` if none.

    Non-finite cells are skipped rather than sorted to the end, because a model
    with no finite RMSE must never be crowned the winner of an empty field.
    """
    best_name: str | None = None
    best_value = float("nan")
    for record in group:
        number = _finite(record.get(key))
        if number is None:
            continue
        current = _finite(best_value)
        if current is None or (number > current if largest else number < current):
            best_name, best_value = str(record.get("model")), number
    return best_name, best_value


def _is_naive(name: Any) -> bool:
    """True when a model is one of the harness's naive baselines."""
    return name is not None and str(name) in set(_NAIVE_BASELINES)


# ----------------------------------------------------------------- table sources


def _horizon_table(run: Any) -> pd.DataFrame:
    """``run.horizon_table()``, rebuilt from the results if that call fails.

    The harness raises when a horizon has no pooled metrics at all, which is
    exactly the degenerate run the renderer has to survive.  The rebuild keeps
    the same columns so the CSV schema does not depend on run quality.
    """
    try:
        frame = run.horizon_table()
        if isinstance(frame, pd.DataFrame):
            return frame
    except Exception as exc:
        logger.warning("run.horizon_table() failed (%s); rebuilding from results", exc)

    rows = [
        {
            "symbol": str(getattr(result, "symbol", "") or _run_symbol(run)),
            "horizon": _horizon_of(result),
            "horizon_days": getattr(result, "horizon_days", np.nan),
            "best_model": "n/a",
            "n_predictions": _prediction_rows(result),
        }
        for result in _results(run)
    ]
    return pd.DataFrame(
        rows, columns=["symbol", "horizon", "horizon_days", "best_model", "n_predictions"]
    )


def _model_table(run: Any) -> pd.DataFrame:
    """``run.model_table()``, or an empty frame when the run holds no results."""
    try:
        frame = run.model_table()
        if isinstance(frame, pd.DataFrame):
            return frame
    except Exception as exc:
        logger.warning("run.model_table() failed (%s); writing an empty table", exc)
    return pd.DataFrame()


def _regime_kwargs(result: Any) -> dict[str, Any]:
    """Trailing regime values a caller attached to ``result``, if any.

    Deliberately narrow: only the two keyword arguments ``analyse`` accepts are
    forwarded, and only when the attached values are Series carrying an index
    the predictions can align to.  Anything else is dropped rather than coerced,
    because a mis-aligned regime label would silently produce a breakdown whose
    rows mean nothing.
    """
    context = getattr(result, "regime_context", None)
    if not isinstance(context, Mapping):
        return {}
    kwargs: dict[str, Any] = {}
    for key in ("regime_values", "trend_values"):
        values = context.get(key)
        if isinstance(values, pd.Series) and len(values) > 0:
            kwargs[key] = values
    return kwargs


def _regime_table(run: Any) -> pd.DataFrame:
    """Per-horizon regime metrics, tagged with their source and horizon.

    Both regime slices of ``analyse`` are concatenated, because a volatility
    regime and a trend regime are two views of one question and dropping either
    would be a silent choice.

    The labels themselves are never synthesised here.  They have to be derived
    from *trailing* values that were known at prediction time, and deriving them
    from the realised target would condition the breakdown on the answer, so
    that is refused even though it would always produce a fuller table.  The
    caller (``src.v3.pipeline``) has the dataset feature frame, so it supplies
    the two trailing series on each result as ``regime_context``; when a run was
    produced without one, the section degrades to a note that says so instead of
    inventing labels.
    """
    frames: list[pd.DataFrame] = []
    for result in _results(run):
        for frame in _frames_from_mapping(
            _analysis_call("analyse", result, **_regime_kwargs(result)), "regime"
        ):
            if frame.empty:
                continue
            frames.append(_tag(frame, _horizon_of(result), "analyse"))
    if not frames:
        return _unavailable(
            "src.v3.analysis",
            "no regime rows were produced; regime labels need trailing values from the "
            "dataset feature frame, which this run did not carry, and they are not "
            "synthesised from the realised target because that would condition the "
            "breakdown on the answer",
        )
    return pd.concat(frames, ignore_index=True)


def _calibration_table(run: Any) -> pd.DataFrame:
    """Calibration rows from ``calibration_report``, one block per horizon.

    Strictly the calibration report.  Interval statistics are a different object
    with a different schema, and folding them into a file named
    ``prediction_calibration.csv`` would make a reader trust a column this
    module never promised.
    """
    frames: list[pd.DataFrame] = []
    for result in _results(run):
        frame = _as_frame(_analysis_call("calibration_report", result))
        if frame is None or frame.empty:
            for candidate in _frames_from_mapping(_analysis_call("analyse", result), "calib"):
                if not candidate.empty:
                    frame = candidate
                    break
        if frame is None or frame.empty:
            continue
        frames.append(_tag(frame, _horizon_of(result), "calibration_report"))
    if not frames:
        return _unavailable("src.v3.analysis", "calibration_report is unavailable")
    return pd.concat(frames, ignore_index=True)


def _return_distribution_table(run: Any) -> pd.DataFrame:
    """Return-distribution rows from ``return_distribution_report``.

    Called once per horizon, with an explicit ``{column: label}`` map because
    that function insists on labelling every column it summarises - including
    the target - and raises rather than guessing.
    """
    frames: list[pd.DataFrame] = []
    for result in _results(run):
        predictions = getattr(result, "predictions", None)
        if not isinstance(predictions, pd.DataFrame) or "target" not in predictions:
            continue
        pred_cols = [
            f"{m}_pred" for m in (getattr(result, "models", None) or []) if f"{m}_pred" in predictions
        ]
        label = _horizon_of(result)
        horizons = {column: label for column in ("target", *pred_cols)}
        frame = _as_frame(
            _analysis_call(
                "return_distribution_report",
                predictions,
                target_col="target",
                pred_cols=pred_cols or None,
                horizons=horizons or None,
            )
        )
        if frame is None or frame.empty:
            continue
        frames.append(_tag(frame, label, "return_distribution_report"))
    if not frames:
        return _unavailable(
            "src.v3.analysis",
            "return_distribution_report is unavailable; return_distributions.png was still "
            "drawn, directly from the pooled predictions",
        )
    return pd.concat(frames, ignore_index=True)


def write_tables(run: Any, out_dir: Path | str) -> tuple[Path, ...]:
    """Write all five CSVs into ``out_dir`` and return their paths in order.

    ``horizon_comparison.csv`` and ``model_comparison.csv`` come straight from the
    harness, because re-deriving them here would risk a second, disagreeing
    version of the same table.  The other three come from ``src.v3.analysis``,
    each degrading to a one-row ``status``/``source``/``reason`` placeholder when
    the analysis module is unavailable, so the file set is identical either way
    and a reader can tell a missing section from an empty one.
    """
    directory = _ensure_dir(out_dir)
    builders: dict[str, Any] = {
        "horizon_comparison.csv": lambda: _horizon_table(run),
        "model_comparison.csv": lambda: _model_table(run),
        "regime_analysis.csv": lambda: _regime_table(run),
        "prediction_calibration.csv": lambda: _calibration_table(run),
        "return_distributions.csv": lambda: _return_distribution_table(run),
    }
    paths: list[Path] = []
    for name in REQUIRED_TABLES:
        path = directory / name
        frame = builders[name]()
        if not isinstance(frame, pd.DataFrame):  # defensive: a builder must return a frame
            frame = _unavailable("src.v3.report", f"{name} produced no frame")
        frame.to_csv(path, index=False)
        logger.info("Table -> %s (%d row(s))", path, len(frame))
        paths.append(path)
    return tuple(paths)


# --------------------------------------------------------------------- plotting


def _ensure_dir(out_dir: Path | str) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _figure(width: float = 11.0, height: float = 5.0) -> tuple[plt.Figure, plt.Axes]:
    """A single-axes figure, created inside the project rc context."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _grid(nrows: int, ncols: int, width: float, height: float) -> tuple[plt.Figure, np.ndarray]:
    """A 2-D axes grid with ``squeeze=False``, so index access never changes shape."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(max(1, nrows), max(1, ncols), figsize=(width, height), squeeze=False)
    return fig, axes


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure and close it.

    ``plt.close`` is not optional housekeeping: six figures per run, and an
    unclosed figure holds its canvas in matplotlib's global registry until the
    process exits, which is how repeated runs in one session exhaust memory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure -> %s", path)
    return path


def _annotate_missing(
    ax: plt.Axes,
    message: str,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Replace an unplottable axes with a readable explanation.

    A chart with no data is still an artefact the reader will open, so it has to
    say *why* it is empty rather than showing an empty frame.  The title and axis
    labels are kept for the same reason: an empty chart with no title tells the
    reader which measure they are looking at only by opening the other five
    files, and the axes it would have had are exactly the ones being erased.
    """
    ax.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=9,
        color=_COLORS["ink"], wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plottable(values: Sequence[Any]) -> np.ndarray:
    """Values as floats with non-finite entries replaced by 0.0.

    A bar chart cannot draw a NaN, and the alternative - omitting the bar -
    would silently shrink the axis so the surviving bars looked larger than they
    are.  A zero-height bar is visibly absent, and :func:`_missing_caption`
    states the count in words.
    """
    numbers = np.array([_finite(value) for value in values], dtype=float)
    return np.nan_to_num(numbers, nan=0.0, posinf=0.0, neginf=0.0)


def _missing_caption(ax: plt.Axes, missing: int) -> None:
    """State on the figure how many cells had no finite value to draw."""
    if missing:
        ax.annotate(
            f"{missing} cell(s) had no finite value and are drawn as zero",
            xy=(0.5, -0.24), xycoords="axes fraction", ha="center",
            fontsize=7, color=_COLORS["ink"],
        )


def _grouped_bars(
    ax: plt.Axes,
    horizons: Sequence[str],
    models: Sequence[str],
    values: Mapping[tuple[str, str], Any],
    *,
    ylabel: str,
) -> int:
    """One bar group per horizon, one bar per model; returns the missing-cell count.

    Built as a matrix rather than one call per model so the geometry is
    identical for a two-model run and a seven-model run, and so a missing cell is
    counted in exactly the place it is drawn.
    """
    width = 0.82 / max(1, len(models))
    positions = np.arange(len(horizons), dtype=float)
    for index, model in enumerate(models):
        heights = _plottable([values.get((label, model), np.nan) for label in horizons])
        ax.bar(
            positions + (-0.41 + width * (index + 0.5)),
            heights,
            width=width * 0.92,
            label=model,
            color=_COLORS["model"] if len(models) == 1 else None,
            edgecolor="none",
        )
    ax.set_xticks(positions, [str(label) for label in horizons])
    ax.set_xlabel("Horizon (forward-return label)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)
    return sum(
        1
        for label in horizons
        for model in models
        if _finite(values.get((label, model))) is None
    )


def plot_rmse_by_horizon(run: Any, out_dir: Path | str) -> Path:
    """``horizon_rmse.png``: pooled out-of-sample RMSE by horizon and model.

    RMSE is the metric V3 optimises, so it is plotted first and plotted with the
    naive baselines beside the learned models.  A reader who sees every bar at a
    similar height has learned the more important thing than the ranking.
    """
    directory = _ensure_dir(out_dir)
    records = _pooled_records(run)
    horizons = _horizons(run, records)
    models = list(dict.fromkeys(str(r["model"]) for r in records))

    fig, ax = _figure(11.0, 5.0)
    if not records or not horizons:
        _annotate_missing(
            ax,
            "No pooled metrics were available for this run",
            title=f"{_run_symbol(run)}: pooled RMSE by horizon and model",
            xlabel="Horizon (forward-return label)",
            ylabel="Pooled out-of-sample RMSE (return units; 0.01 = 1%)",
        )
    else:
        missing = _grouped_bars(
            ax, horizons, models,
            {(r["horizon"], r["model"]): r.get("rmse") for r in records},
            ylabel="Pooled out-of-sample RMSE (return units; 0.01 = 1%)",
        )
        ax.set_title(
            f"{_run_symbol(run)}: pooled RMSE by horizon and model"
            + (f" - {models[0]} only" if len(models) == 1 else "")
        )
        if len(models) > 1:
            ax.legend(frameon=False, ncol=min(3, len(models)), loc="upper left")
        _missing_caption(ax, missing)
    return _save(fig, directory / "horizon_rmse.png")


def plot_ic_by_horizon(run: Any, out_dir: Path | str) -> Path:
    """``horizon_ic.png``: pooled Spearman IC by horizon and model.

    IC is the metric that survives heavy tails and monotone distortion, and it
    is the one to read *next to* RMSE rather than instead of it: a model can have
    a useless RMSE and a real ranking edge, or the reverse.  The zero line is
    drawn because a bar below it is the most common outcome in this report.
    """
    directory = _ensure_dir(out_dir)
    records = _pooled_records(run)
    horizons = _horizons(run, records)
    models = list(dict.fromkeys(str(r["model"]) for r in records))

    fig, ax = _figure(11.0, 5.0)
    if not records or not horizons:
        _annotate_missing(
            ax,
            "No pooled rank correlations were available for this run",
            title=f"{_run_symbol(run)}: pooled Spearman IC by horizon and model",
            xlabel="Horizon (forward-return label)",
            ylabel="Spearman rank IC (dimensionless, -1 to +1)",
        )
    else:
        missing = _grouped_bars(
            ax, horizons, models,
            {(r["horizon"], r["model"]): r.get("spearman_ic") for r in records},
            ylabel="Spearman rank IC (dimensionless, -1 to +1)",
        )
        ax.axhline(
            0.0, color=_COLORS["benchmark"], linestyle="--", linewidth=1.0,
            label="no ranking power",
        )
        ax.set_title(f"{_run_symbol(run)}: pooled Spearman IC by horizon and model")
        ax.legend(frameon=False, ncol=min(3, len(models) + 1), loc="upper left")
        _missing_caption(ax, missing)
    return _save(fig, directory / "horizon_ic.png")


def plot_interval_coverage(run: Any, out_dir: Path | str) -> Path:
    """``interval_coverage.png``: realised coverage against the nominal target.

    Pooled coverage is preferred, because it is the number a reader would compute
    by hand from the predictions.  The mean of the per-fold summaries is used
    when the pooled value is absent, and the title says so, because averaging
    per-fold ratios and recomputing one pooled ratio are different measurements
    and must not be mixed silently.
    """
    directory = _ensure_dir(out_dir)
    records = _pooled_records(run)
    intervals = {(r["horizon"], r["model"]): r for r in _interval_records(run)}
    horizons = _horizons(run, records)
    models = list(dict.fromkeys(str(r["model"]) for r in records))

    fig, ax = _figure(11.0, 5.0)
    if not records or not horizons:
        _annotate_missing(
            ax,
            "No interval coverage was available for this run",
            title=f"{_run_symbol(run)}: conformal band coverage vs nominal",
            xlabel="Horizon (forward-return label)",
            ylabel="Empirical coverage of the conformal band (fraction of rows)",
        )
    else:
        coverage: dict[tuple[str, str], Any] = {}
        nominal: list[float] = []
        used_fold_means = False
        for record in records:
            key = (record["horizon"], record["model"])
            value = _finite(record.get("coverage_lo"))
            if value is None:
                value = _finite(intervals.get(key, {}).get("mean_fold_coverage"))
                used_fold_means = used_fold_means or value is not None
            coverage[key] = value
            target = _finite(intervals.get(key, {}).get("nominal_coverage"))
            if target is not None:
                nominal.append(target)

        missing = _grouped_bars(
            ax, horizons, models, coverage,
            ylabel="Empirical coverage of the conformal band (fraction of rows)",
        )
        if nominal:
            average = float(np.mean(nominal))
            ax.axhline(
                average, color=_COLORS["down"], linestyle="--", linewidth=1.2,
                label=f"nominal target ({average:.2f})",
            )
            ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"{_run_symbol(run)}: conformal band coverage vs nominal"
            + (" - some bars are per-fold means" if used_fold_means else "")
        )
        ax.legend(frameon=False, ncol=min(3, len(models) + 1), loc="lower left")
        _missing_caption(ax, missing)
    return _save(fig, directory / "interval_coverage.png")


def _bin_curve(
    prob: np.ndarray, outcome: np.ndarray, source: str, n_bins: int = 10
) -> pd.DataFrame:
    """One equal-width reliability curve over ``[0, 1]``.

    Mirrors ``src.v3.metrics.calibration_table`` - equal-width bins with the last
    one closed on the right, the same column names, and empty bins kept with
    ``n == 0`` rather than dropped - so this is the same measurement and not a
    lookalike.  Kept as a function of arrays because the per-horizon horizons are
    pooled: two or three populated bins per horizon is noise, and the pooled
    curve is what a reader can actually check against
    ``prediction_calibration.csv``.
    """
    keep = np.isfinite(prob) & np.isfinite(outcome)
    prob, outcome = prob[keep], outcome[keep]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)
        count = int(mask.sum())
        rows.append(
            {
                "source": source,
                "bin_lower": lower,
                "bin_upper": upper,
                "n": count,
                "mean_predicted": float(prob[mask].mean()) if count else np.nan,
                "observed_frequency": float(outcome[mask].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _up_indicator(target: np.ndarray) -> np.ndarray:
    """1.0 for a realised up return, 0.0 for down, NaN where the target is unknown.

    ``(target > 0).astype(float)`` looks equivalent and is not: ``NaN > 0`` is
    ``False``, so a row whose target failed to materialise is scored as a down day
    and the bin's observed frequency is quietly biased downwards.  Emitting NaN
    instead hands the drop to :func:`_bin_curve`, which already excludes
    non-finite pairs.
    """
    return np.where(np.isfinite(target), (target > 0.0).astype(float), np.nan)


def _reliability_curves(run: Any) -> pd.DataFrame | None:
    """Reliability curves pooled over every horizon, or ``None`` if there are none.

    Two series and no more: the best-RMSE model's own empirical probability and
    the shared calibrated direction probability.  Plotting one curve per model
    would be the more complete-looking choice and the less readable one - six
    overlapping curves answer nothing a reader can act on - and the per-model
    numbers live in ``prediction_calibration.csv`` regardless.
    """
    records = _pooled_records(run)
    blocks: list[pd.DataFrame] = []
    for label, group in _by_horizon(records, _horizons(run, records)).items():
        result = next((r for r in _results(run) if _horizon_of(r) == label), None)
        if result is None:
            continue
        predictions = getattr(result, "predictions", None)
        if not isinstance(predictions, pd.DataFrame) or "target" not in predictions:
            continue
        target = _numeric(predictions["target"])
        best, _ = _extreme(group, "rmse", largest=False)
        wanted = []
        if best is not None and f"{best}_prob_empirical" in predictions:
            wanted.append((f"{best} (empirical)", f"{best}_prob_empirical"))
        if "prob_calibrated" in predictions:
            wanted.append(("calibrated direction model", "prob_calibrated"))
        for source, column in wanted:
            block = _bin_curve(
                _numeric(predictions[column]), _up_indicator(target), source
            )
            if int(block["n"].sum()) > 0:
                blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else None


_CURVE_SOURCE = (
    "equal-width reliability bins over the pooled probability columns; per-model "
    "calibration statistics are in prediction_calibration.csv"
)


def plot_prediction_calibration(run: Any, out_dir: Path | str) -> Path:
    """``prediction_calibration.png``: reliability of the direction probability.

    A point forecast is not a probability, so the only curves worth drawing here
    are the ones built from genuine probabilities - the best model's own residual
    CDF and the separately fitted direction model - against the diagonal that
    defines "calibrated".  Bin populations are written into the figure only as
    legend labels would be, so the reader is never shown a sparse bin as though it
    were a measurement.
    """
    directory = _ensure_dir(out_dir)
    curves = _reliability_curves(run)

    fig, ax = _figure(6.6, 5.8)
    ax.plot(
        [0, 1], [0, 1], color=_COLORS["benchmark"], linestyle="--", linewidth=1.0,
        label="perfect calibration",
    )
    plotted = False
    if curves is not None and not curves.empty:
        for source, block in curves.groupby("source", sort=True):
            points = block[["mean_predicted", "observed_frequency"]].dropna()
            if points.empty:
                continue
            plotted = True
            ax.plot(
                points["mean_predicted"], points["observed_frequency"], "o-",
                linewidth=1.5, markersize=5, label=f"{source} ({int(block['n'].sum()):,} rows)",
            )
    if not plotted:
        _annotate_missing(
            ax,
            "No calibrated probabilities were available for this run",
            title=f"{_run_symbol(run)}: direction-probability calibration",
            xlabel="Mean predicted probability P(up)",
            ylabel="Observed frequency of up returns",
        )
    else:
        ax.set_xlabel("Mean predicted probability P(up)")
        ax.set_ylabel("Observed frequency of up returns")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{_run_symbol(run)}: direction-probability calibration")
        ax.legend(frameon=False, loc="upper left", fontsize=7)
    return _save(fig, directory / "prediction_calibration.png")


def _distribution_panels(run: Any) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """``(horizon, target, prediction, model)`` per horizon, best model by RMSE."""
    records = _pooled_records(run)
    panels: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for label, group in _by_horizon(records, _horizons(run, records)).items():
        result = next((r for r in _results(run) if _horizon_of(r) == label), None)
        if result is None:
            continue
        predictions = getattr(result, "predictions", None)
        if not isinstance(predictions, pd.DataFrame) or "target" not in predictions:
            continue
        best, _ = _extreme(group, "rmse", largest=False)
        model = best or next(
            (str(m) for m in (getattr(result, "models", None) or []) if f"{m}_pred" in predictions),
            "",
        )
        column = _pred_column(result, model)
        panels.append(
            (
                label,
                _drawable(_numeric(predictions["target"])),
                _drawable(_numeric(column)) if column is not None else np.empty(0, dtype=float),
                str(model),
            )
        )
    return panels


def plot_return_distributions(run: Any, out_dir: Path | str) -> Path:
    """``return_distributions.png``: realised vs predicted return, per horizon.

    One panel per horizon on a *shared* density scale.  The shared scale is the
    point: on independent axes the narrower series always looks tighter, which
    would turn a measurement into a visual artefact.  The banded shape is also
    what puts every error metric in context, since a model is only as good as the
    spread of the thing it is trying to predict.
    """
    directory = _ensure_dir(out_dir)
    panels = _distribution_panels(run)
    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols)) if panels else 1
    fig, axes = _grid(nrows, ncols, 5.6 * ncols, 3.4 * nrows)

    pooled = np.concatenate(
        [values for _, target, predicted, _ in panels for values in (target, predicted) if values.size]
    ) if panels else np.empty(0, dtype=float)
    if pooled.size:
        low, high = (float(v) for v in np.quantile(pooled, [0.01, 0.99]))
    else:
        low, high = -0.01, 0.01
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        centre = low if np.isfinite(low) else 0.0
        low, high = centre - 0.005, centre + 0.005
    bins = np.linspace(low, high, _DISTRIBUTION_BINS + 1)

    for index, ax in enumerate(axes.ravel()):
        if index >= len(panels):
            ax.set_visible(False)
            continue
        label, target, predicted, model = panels[index]
        drawn = False
        for values, colour, name in (
            (target, _COLORS["target"], "realised target"),
            (predicted, _COLORS["model"], f"{model} prediction" if model else "prediction"),
        ):
            if values.size == 0:
                continue
            drawn = True
            ax.hist(
                np.clip(values, low, high), bins=bins, density=True, alpha=0.5,
                color=colour, label=f"{name} (n={values.size:,})",
            )
        if not drawn:
            _annotate_missing(
                ax,
                "No finite pooled predictions for this horizon",
                title=f"{label} - {target.size:,} pooled rows",
                xlabel="Forward return (fraction; 0.05 = +5%)",
                ylabel="Density (1 per unit return)",
            )
            continue
        ax.axvline(0.0, color=_COLORS["benchmark"], linestyle="--", linewidth=0.9)
        ax.set_title(f"{label} - {target.size:,} pooled rows")
        ax.set_xlabel("Forward return (fraction; 0.05 = +5%)")
        ax.set_ylabel("Density (1 per unit return)")
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        f"{_run_symbol(run)}: realised vs predicted return distribution by horizon "
        + ("(1st-99th pct clipped, shared scale)" if panels else ""),
        fontsize=10,
    )
    if not panels:
        # The loop above hid the only axes as surplus, so it has to come back before
        # it can carry the note; an invisible axes explains nothing.
        axes.ravel()[0].set_visible(True)
        _annotate_missing(
            axes.ravel()[0],
            "No pooled predictions were available for this run",
            xlabel="Forward return (fraction; 0.05 = +5%)",
            ylabel="Density (1 per unit return)",
        )
        fig.tight_layout()
        return _save(fig, directory / "return_distributions.png")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, directory / "return_distributions.png")


def plot_horizon_comparison(run: Any, out_dir: Path | str) -> Path:
    """``model_comparison.png``: per-horizon model ranking with direction accuracy.

    A small-multiple grid rather than a second grouped bar chart, so it answers a
    different question from ``horizon_rmse.png``: within one horizon, which
    models are competitive at all, and whether any of them beats 50% direction
    accuracy.  Naive baselines are greyed so a baseline win is visible at a
    glance rather than buried in a legend lookup.
    """
    directory = _ensure_dir(out_dir)
    records = _pooled_records(run)
    populated = {label: group for label, group in _by_horizon(records, _horizons(run, records)).items() if group}
    ncols = 2 if len(populated) > 1 else 1
    nrows = int(np.ceil(len(populated) / ncols)) if populated else 1
    tallest = max((len(group) for group in populated.values()), default=1)
    fig, axes = _grid(nrows, ncols, 5.8 * ncols, 0.5 * tallest + 1.7 * nrows)

    finite = [value for value in (_finite(r.get("rmse")) for r in records) if value is not None]
    limit = max(finite) * 1.22 if finite else 1.0

    for index, ax in enumerate(axes.ravel()):
        if index >= len(populated):
            ax.set_visible(False)
            continue
        label, group = list(populated.items())[index]
        ranked = sorted(
            group,
            key=lambda r: (_finite(r.get("rmse")) is None, _finite(r.get("rmse")) or 0.0),
        )
        names = [str(r["model"]) for r in ranked]
        values = _plottable([r.get("rmse") for r in ranked])
        positions = np.arange(len(names), dtype=float)
        ax.barh(
            positions, values,
            color=[_COLORS["benchmark"] if _is_naive(n) else _COLORS["model"] for n in names],
            edgecolor="none",
        )
        for position, record in zip(positions, ranked):
            accuracy = _finite(record.get("direction_accuracy"))
            if accuracy is not None:
                ax.annotate(
                    f" dir acc {accuracy * 100:.1f}%",
                    xy=(_finite(record.get("rmse")) or 0.0, position),
                    va="center", fontsize=7, color=_COLORS["ink"],
                )
        ax.set_yticks(positions, names)
        ax.set_xlim(0.0, limit)
        ax.set_xlabel("Pooled RMSE (return units)")
        ax.set_title(f"{label} - {int(ranked[0].get('n_predictions') or 0):,} pooled rows", fontsize=9)
        ax.grid(axis="y", alpha=0)

    if not populated:
        # As in the distribution figure, the loop hid the only axes as surplus.
        axes.ravel()[0].set_visible(True)
        _annotate_missing(
            axes.ravel()[0],
            "No pooled metrics were available for this run",
            xlabel="Pooled RMSE (return units)",
            ylabel="Model",
        )
    fig.suptitle(
        f"{_run_symbol(run)}: model comparison by horizon (grey = naive baseline)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97 if populated else 1.0))
    return _save(fig, directory / "model_comparison.png")


# -------------------------------------------------------------------- summary.md


def _geometry_records(run: Any) -> list[dict[str, Any]]:
    """Purge/embargo geometry per horizon, read from the splitter description."""
    records: list[dict[str, Any]] = []
    for result in _results(run):
        geometry = getattr(result, "geometry", None)
        geometry = geometry if isinstance(geometry, Mapping) else {}
        coverage = getattr(result, "coverage", None)
        coverage = coverage if isinstance(coverage, Mapping) else {}
        records.append(
            {
                "horizon": _horizon_of(result),
                "horizon_days": getattr(result, "horizon_days", np.nan),
                "purge": geometry.get("purge", np.nan),
                "embargo": geometry.get("embargo", np.nan),
                "n_splits": geometry.get("n_splits_requested", np.nan),
                "labelled_rows": geometry.get("labelled_rows", coverage.get("labelled_rows", np.nan)),
                "n_folds": coverage.get("n_folds", np.nan),
                "prediction_rows": _prediction_rows(result),
            }
        )
    return records


def _headline_records(run: Any, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per horizon: the RMSE winner, the IC winner, and both of their numbers.

    The two winners share a row because the interesting fact is usually their
    *disagreement*: the lowest-error model and the best-ranking model are
    frequently different models, and reporting only one of them hides that.
    """
    headline: list[dict[str, Any]] = []
    for label, group in _by_horizon(records, _horizons(run, records)).items():
        rmse_model, rmse_value = _extreme(group, "rmse", largest=False)
        ic_model, ic_value = _extreme(group, "spearman_ic", largest=True)
        rmse_row = next((r for r in group if r["model"] == rmse_model), None)
        ic_row = next((r for r in group if r["model"] == ic_model), None)
        headline.append(
            {
                "horizon": label,
                "n": group[0].get("n_predictions") if group else np.nan,
                "n_models": len(group),
                "best_rmse_model": rmse_model or "n/a",
                "rmse": rmse_value,
                "mae": (rmse_row or {}).get("mae", np.nan),
                "spearman_ic": (rmse_row or {}).get("spearman_ic", np.nan),
                "direction_accuracy": (rmse_row or {}).get("direction_accuracy", np.nan),
                "best_ic_model": ic_model or "n/a",
                "ic": ic_value,
                "rmse_of_ic_model": (ic_row or {}).get("rmse", np.nan),
            }
        )
    return headline


def _coverage_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Conformal coverage per ``(horizon, model)`` with an explicit verdict."""
    rows: list[dict[str, Any]] = []
    for record in records:
        empirical = _finite(record.get("coverage_lo"))
        source = "pooled rows"
        if empirical is None:
            empirical = record.get("interval_mean_fold_coverage")
            source = "mean of per-fold summaries"
        nominal = _finite(record.get("interval_nominal_coverage"))
        met: Any = np.nan
        if empirical is not None and nominal is not None:
            met = abs(empirical - nominal) <= _COVERAGE_TOLERANCE
        rows.append(
            {
                "horizon": record["horizon"],
                "model": record["model"],
                "empirical_coverage": empirical,
                "nominal_coverage": nominal,
                "met_nominal": met,
                "mean_interval_width": record.get("interval_mean_interval_width", np.nan),
                "calibration_rows": record.get("interval_mean_calibration_rows", np.nan),
                "source": source,
            }
        )
    return rows


def _brier_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Brier scores for the empirical and the calibrated direction probability."""
    return [
        {
            "horizon": record["horizon"],
            "model": record["model"],
            "brier_empirical": record.get("brier_empirical", np.nan),
            "brier_calibrated": record.get("brier_calibrated", np.nan),
            "n": record.get("n", np.nan),
        }
        for record in records
    ]


def _merged_records(run: Any) -> list[dict[str, Any]]:
    """Pooled metric cells with their interval-summary cells attached.

    Joining once, up front, keeps the coverage section from having to re-derive
    the ``(horizon, model)`` join a second time with a different key.
    """
    intervals = {(r["horizon"], r["model"]): r for r in _interval_records(run)}
    merged: list[dict[str, Any]] = []
    for record in _pooled_records(run):
        summary = intervals.get((record["horizon"], record["model"]), {})
        merged.append({**record, **{f"interval_{k}": v for k, v in summary.items()}})
    return merged


def _md_cell(value: Any, digits: int = 4) -> str:
    """Markdown cell for a value that may be a plain string such as a model name."""
    return value if isinstance(value, str) else _fmt(value, digits)


def _render_extra(extra: Mapping[str, object] | None) -> list[str]:
    """Render caller-supplied provenance as a table.

    Deliberately additive: an extra can annotate a run but never overwrite a
    measured number, so nothing a caller passes can make a metric look better
    than the harness computed it.
    """
    if not extra:
        return []
    rows = []
    for key in sorted(extra):
        value = extra[key]
        if isinstance(value, Mapping):
            rendered = "; ".join(f"{k}={_fmt(v)}" for k, v in value.items())
        else:
            rendered = _fmt(value)
        rows.append([str(key), rendered])
    return ["", "### Caller-supplied notes", ""] + _md_table(["field", "value"], rows) + [""]


def _dataset_lines(dataset: Mapping[str, Any]) -> list[str]:
    """Dataset description, drop report and missing sources, or a note."""
    if not isinstance(dataset, Mapping) or not dataset:
        return ["_No dataset description was attached to this run._"]
    index = dataset.get("index") if isinstance(dataset.get("index"), Mapping) else {}
    coverage = (
        dataset.get("feature_coverage")
        if isinstance(dataset.get("feature_coverage"), Mapping)
        else {}
    )
    lines = _md_table(
        ["field", "value"],
        [
            [name, _fmt(value)]
            for name, value in (
                ("symbol", dataset.get("symbol")),
                ("rows", dataset.get("n_rows")),
                ("features", dataset.get("n_features")),
                ("targets", dataset.get("n_targets")),
                ("feature groups", ", ".join(str(g) for g in (dataset.get("feature_groups") or [])) or None),
                ("index name", index.get("name")),
                ("start", index.get("start")),
                ("end", index.get("end")),
                ("timezone", index.get("tz")),
                ("index unique", index.get("is_unique")),
                ("index monotonic", index.get("is_monotonic_increasing")),
                ("rows dropped", dataset.get("n_rows_dropped")),
                ("bucket registry available", dataset.get("bucket_registry_available")),
            )
        ],
    )

    drop_report = dataset.get("drop_report")
    if isinstance(drop_report, Mapping) and drop_report:
        lines += ["", "**Drop report**", ""]
        lines += _md_table(
            ["dropped", "rows"],
            [[str(k), _fmt(v)] for k, v in sorted(drop_report.items(), key=lambda kv: str(kv[0]))],
        )

    missing = dataset.get("missing_sources")
    lines += ["", "**Missing sources**", ""]
    lines += (
        _md_table(["source"], [[str(m)] for m in missing])
        if missing
        else ["None recorded."]
    )

    if coverage:
        def share(key: str) -> str:
            value = _finite(coverage.get(key))
            return "n/a" if value is None else _pct(value / 100.0)

        lines += [
            "",
            "**Feature coverage**",
            "",
            f"Minimum {share('min_pct')}, median {share('median_pct')}, "
            f"maximum {share('max_pct')}; "
            f"{_fmt(coverage.get('n_below_50pct'))} feature(s) below 50% coverage.",
        ]

    target_coverage = dataset.get("target_coverage")
    if isinstance(target_coverage, Mapping) and target_coverage:
        lines += ["", "**Labelled rows per horizon**", ""]
        lines += _md_table(
            ["horizon", "labelled", "unlabelled tail", "labelled share"],
            [
                [
                    str(label),
                    _fmt(block.get("n_labelled") if isinstance(block, Mapping) else None),
                    _fmt(block.get("n_unlabelled_tail") if isinstance(block, Mapping) else None),
                    _pct(
                        float(block["labelled_pct"]) / 100.0
                        if isinstance(block, Mapping) and _finite(block.get("labelled_pct")) is not None
                        else np.nan
                    ),
                ]
                for label, block in target_coverage.items()
            ],
        )
    return lines


def _coverage_prose(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """State the coverage verdict in words rather than leaving it to the table."""
    if not rows:
        return ["_No coverage verdict could be formed: no interval summary was present._"]
    met = [r for r in rows if r.get("met_nominal") is True]
    under = [
        r for r in rows
        if r.get("met_nominal") is False
        and _finite(r.get("empirical_coverage")) is not None
        and _finite(r.get("nominal_coverage")) is not None
        and float(r["empirical_coverage"]) < float(r["nominal_coverage"])
    ]
    over = [
        r for r in rows
        if r.get("met_nominal") is False
        and _finite(r.get("empirical_coverage")) is not None
        and _finite(r.get("nominal_coverage")) is not None
        and float(r["empirical_coverage"]) >= float(r["nominal_coverage"])
    ]
    lines = [
        f"Empirical coverage met the nominal target for {len(met)} of {len(rows)} "
        f"model/horizon pairs, against a tolerance of {_pct(_COVERAGE_TOLERANCE)}."
    ]
    if under:
        worst = min(
            float(r["empirical_coverage"]) - float(r["nominal_coverage"]) for r in under
        )
        lines.append(
            f"{len(under)} pair(s) under-covered, the worst by {_pct(worst)}: the realised band "
            "contained fewer outcomes than it promised."
        )
    if over:
        best = max(
            float(r["empirical_coverage"]) - float(r["nominal_coverage"]) for r in over
        )
        lines.append(
            f"{len(over)} pair(s) over-covered, the widest by {_pct(best)}: the band was broader "
            "than the target and therefore carried less information than it appeared to."
        )
    return lines


def _brier_prose(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Compare empirical against calibrated Brier without over-reading either."""
    comparable = [
        r for r in rows
        if _finite(r.get("brier_empirical")) is not None
        and _finite(r.get("brier_calibrated")) is not None
    ]
    if not comparable:
        return [
            "No comparable Brier scores were available: both probability sources were missing "
            "or unscorable for every model, which usually means the direction model could not be "
            "fitted on the validation blocks."
        ]
    better = sum(
        1 for r in comparable if float(r["brier_calibrated"]) < float(r["brier_empirical"])
    )
    best = min(float(r["brier_calibrated"]) for r in comparable)
    return [
        f"The calibrated probability scored better than the empirical one for {better} of "
        f"{len(comparable)} comparable pairs. A Brier score on a single pooled sample is not a "
        "significance test, and the gap between two of them is well inside the noise of a few "
        "thousand overlapping rows. The best calibrated score here is "
        f"{_fmt(best)}; a constant 50/50 forecast scores 0.25, so anything at or above that has "
        "not beaten the coin flip."
    ]


def _reliability_prose(run: Any) -> list[str]:
    """A short reading of the reliability curves, or a note that there are none."""
    curves = _reliability_curves(run)
    if curves is None or curves.empty:
        return [
            "_No reliability curve is available: this run produced no finite probability, so "
            "nothing can be said about whether its direction forecast is calibrated._"
        ]
    lines = []
    for source, block in curves.groupby("source", sort=True):
        points = block[["mean_predicted", "observed_frequency"]].dropna()
        if points.empty:
            continue
        gap = (points["mean_predicted"] - points["observed_frequency"]).abs()
        lines.append(
            f"- `{source}`: {len(points)} populated bin(s), largest gap between mean predicted "
            f"probability and observed up-frequency {_fmt(float(gap.max()))}, mean absolute gap "
            f"{_fmt(float(gap.mean()))}."
        )
    if lines:
        lines.insert(0, "")
        lines.append("")
        lines.append(
            f"Source: {_CURVE_SOURCE}. A large gap sitting in a sparsely populated bin is a "
            "sample-size artefact rather than miscalibration; the bin counts in "
            "`prediction_calibration.csv` are there so that distinction can be made."
        )
    return lines


def _regime_lines(run: Any) -> list[str]:
    """Render the regime tables, or say plainly that they are unavailable."""
    frame = _as_frame(_regime_table(run))
    if _is_placeholder(frame) or frame is None or frame.empty:
        absent = _analysis is None
        reason = (
            "`src.v3.analysis` is not importable in this run"
            if absent
            else "`src.v3.analysis` ran but had no regime labels to score: they live in the "
            "dataset feature frame, which a run result does not carry"
        )
        return [
            "_Not available._ The regime breakdown needs `analyse`, and " + reason + ". It is not "
            "reconstructible here either: a `HorizonResult` carries predictions and split "
            "geometry, not the trailing values `analyse` labels regimes from. No regime "
            "conclusion should be drawn from this run."
        ]
    columns = [c for c in frame.columns if c not in ("source", "horizon")]
    return _md_frame(frame, columns=columns, digits=5) + [
        "",
        "Regimes are defined by an explicit trailing rule rather than by hindsight, and each "
        "block is scored on one model rather than on all of them averaged together. One pass "
        "over one symbol is still not evidence that any difference between regimes is stable.",
    ]


def _interval_report_lines(run: Any) -> list[str]:
    """The analysis interval report when one exists, as a note otherwise."""
    frames: list[pd.DataFrame] = []
    for result in _results(run):
        frame = _as_frame(_analysis_call("prediction_interval_report", result))
        if frame is None or frame.empty:
            continue
        frames.append(_tag(frame, _horizon_of(result), "prediction_interval_report"))
    if not frames:
        return [
            "",
            "_The pooled interval report from `src.v3.analysis` is not available; the table "
            "above was built from `interval_summary` and the pooled predictions instead._",
        ]
    return ["", "**Pooled interval report**", ""] + _md_frame(pd.concat(frames, ignore_index=True), digits=5)


def _decile_lines(run: Any) -> list[str]:
    """The rank ladder when ``decile_report`` answers, else a one-line note."""
    frames: list[pd.DataFrame] = []
    for result in _results(run):
        frame = _as_frame(_analysis_call("decile_report", result))
        if frame is None or frame.empty:
            continue
        frames.append(_tag(frame, _horizon_of(result), "decile_report"))
    if not frames:
        return ["", "_Rank ladder not available: `src.v3.analysis.decile_report` is unavailable._"]
    combined = pd.concat(frames, ignore_index=True)
    columns = [c for c in combined.columns if c not in ("source",)]
    return [
        "",
        "**Rank ladder (predicted decile against realised return)**",
        "",
        "A prediction that carries information produces a monotonically rising ladder. A flat or "
        "falling one means the model is adding magnitude, not ordering.",
        "",
    ] + _md_frame(combined, columns=columns, digits=5)


def _caveat_lines(
    run: Any,
    records: Sequence[Mapping[str, Any]],
    status: Mapping[str, str],
) -> list[str]:
    """The honest limitations of this run, as sentences rather than boilerplate."""
    small = sorted(
        {
            str(r["horizon"])
            for r in records
            if int(r.get("n_predictions") or 0) < _SMALL_PREDICTION_ROWS
        }
    )
    missing = sorted(name for name, state in status.items() if state != "available")
    lines = [
        f"- Forward-return windows at these horizons overlap almost completely, so the number of "
        f"pooled rows overstates the independent evidence. Thousands of rows at a 30-day horizon "
        f"are not thousands of independent observations - they are thousands of heavily "
        f"correlated ones - and no significance statement in this report accounts for the "
        f"dependence.",
        f"- One symbol, one pass, no nested cross-validation around the hyperparameter search. "
        f"The search was selected on validation folds and the metrics are pooled from test folds, "
        f"which is honest, but the selection itself is not re-validated.",
        f"- No transaction costs, slippage, funding or market impact are modelled anywhere in "
        f"this report. The long-short spread is a pre-cost number and is not a PnL estimate.",
        f"- Geometry is shared by every horizon in the run, but a longer horizon consumes more of "
        f"the sample in purging and embargo, so the long-horizon models are fitted on strictly "
        f"less data. Fewer folds at the long end means the pooled metrics there rest on fewer "
        f"independent decisions.",
    ]
    if small:
        lines.append(
            f"- {len(small)} horizon(s) produced fewer than {_SMALL_PREDICTION_ROWS:,} pooled "
            f"predictions ({', '.join(small)}). Coverage, IC and direction accuracy at those "
            f"horizons are dominated by sampling noise and should not be quoted."
        )
    if missing:
        lines.append(
            f"- {len(missing)} analysis function(s) were unavailable at render time "
            f"({', '.join(missing)}). The corresponding sections above say so rather than "
            f"reporting a result, and `regime_analysis.csv` holds an explicit not-available row."
        )
    return lines


def write_summary(
    run: Any, out_dir: Path | str, extra: Mapping[str, object] | None = None
) -> Path:
    """Write ``summary.md``: the report a reader is expected to actually read.

    Seven sections in a fixed order, because the order is an argument: what was
    run, what it produced, how certain that is, what it says about direction,
    whether it holds across regimes, what it cannot tell you, and what to open
    next.  Every missing number renders as ``n/a`` and every missing analysis
    section renders as a sentence saying so, so the document never implies a
    result that was not computed.

    ``extra`` is caller-supplied provenance (a git sha, a run note, a data
    vintage) appended to the metadata section.  It annotates the run and never
    overrides a measured number.
    """
    directory = _ensure_dir(out_dir)
    path = directory / SUMMARY_NAME

    records = _merged_records(run)
    horizons = _horizons(run, records)
    headline = _headline_records(run, records)
    coverage_rows = _coverage_records(records)
    brier_rows = _brier_records(records)
    config = getattr(run, "config", None)
    config = config if isinstance(config, Mapping) else {}
    dataset = getattr(run, "dataset", None)
    status = _analysis_status()
    symbol = _run_symbol(run)
    models = list(dict.fromkeys(str(r["model"]) for r in records)) or [
        str(m) for m in (config.get("models") or [])
    ]
    alpha = _finite(config.get("alpha"))

    lines: list[str] = [
        f"# V3 walk-forward report: {symbol}",
        "",
        "Every number below is computed from pooled out-of-sample test predictions. No row that "
        "a model was fitted on contributes to any metric in this document.",
        "",
        "## 1. Run metadata",
        "",
    ]
    lines += _md_table(
        ["field", "value"],
        [
            ["symbol", symbol],
            ["horizons", ", ".join(horizons) or "n/a"],
            ["features", _fmt(len(getattr(run, "feature_columns", None) or []))],
            ["models", ", ".join(models) or "n/a"],
            ["seed", _fmt(config.get("seed"))],
            ["n_splits", _fmt(config.get("n_splits"))],
            ["test fraction", _fmt(config.get("test_fraction"))],
            ["validation fraction", _fmt(config.get("validation_fraction"))],
            ["nominal interval coverage", _fmt(1.0 - alpha) if alpha is not None else "n/a"],
            ["total runtime (s)", _fmt(getattr(run, "seconds", None), 1)],
            ["analysis module", ", ".join(f"{name}: {state}" for name, state in status.items())],
        ],
    )
    lines += ["", "**Purge and embargo geometry**", ""]
    geometry = _geometry_records(run)
    lines += (
        _md_table(
            ["horizon", "days", "purge", "embargo", "splits", "labelled rows", "folds",
             "prediction rows"],
            [
                [_md_cell(r["horizon"]), _fmt(r["horizon_days"], 0), _fmt(r["purge"]),
                 _fmt(r["embargo"]), _fmt(r["n_splits"], 0), _fmt(r["labelled_rows"], 0),
                 _fmt(r["n_folds"], 0), _fmt(r["prediction_rows"], 0)]
                for r in geometry
            ],
        )
        if geometry
        else ["_No horizon geometry was attached to this run._"]
    )
    lines += _render_extra(extra)

    # ------------------------------------------------------------------ 2
    lines += [
        "",
        "## 2. Headline results",
        "",
        "The lowest-RMSE model and the highest-IC model per horizon. They are frequently different "
        "models: RMSE rewards getting the magnitude right, IC rewards ordering the rows correctly, "
        "and a forecast can do one without the other.",
        "",
    ]
    lines += (
        _md_table(
            ["horizon", "n", "best by RMSE", "rmse", "mae", "spearman ic", "dir acc",
             "best by IC", "ic", "rmse of IC model", "models"],
            [
                [_md_cell(r["horizon"]), _fmt(r["n"], 0), r["best_rmse_model"], _fmt(r["rmse"]),
                 _fmt(r["mae"]), _fmt(r["spearman_ic"]), _pct(r["direction_accuracy"]),
                 r["best_ic_model"], _fmt(r["ic"]), _fmt(r["rmse_of_ic_model"]),
                 _fmt(r["n_models"], 0)]
                for r in headline
            ],
        )
        if headline
        else ["_No headline results were available for this run._"]
    )
    lines += [""]
    for row in headline:
        lines.append(
            f"- **{row['horizon']}**: lowest RMSE is `{row['best_rmse_model']}` "
            f"(RMSE {_fmt(row['rmse'])}, MAE {_fmt(row['mae'])}, Spearman IC "
            f"{_fmt(row['spearman_ic'])}, direction accuracy "
            f"{_pct(row['direction_accuracy'])}); highest IC is `{row['best_ic_model']}` "
            f"(IC {_fmt(row['ic'])})."
        )

    naive = [r["horizon"] for r in headline if _is_naive(r["best_rmse_model"])]
    lines += ["", "**Naive-baseline check**", ""]
    if naive and len(naive) == len(headline):
        lines.append(
            f"A naive baseline wins on RMSE at every one of the {len(headline)} horizons "
            f"({', '.join(naive)}). Stated plainly because it is the expected result at long "
            "horizons: the learned models did not beat a constant or trailing-mean forecast out of "
            "sample, and no metric below changes that. The rank and direction statistics are "
            "reported because they are the only place any edge appears, not as a substitute for "
            "a win."
        )
    elif naive:
        lines.append(
            f"A naive baseline wins on RMSE at {len(naive)} of {len(headline)} horizons "
            f"({', '.join(naive)}). At those horizons the learned models did not beat a constant or "
            "trailing-mean forecast out of sample, so the RMSE ranking there should be read as a "
            "tie rather than as a result."
        )
    elif headline:
        lines.append(
            "No naive baseline wins on RMSE at any horizon, so every horizon is a genuine "
            "out-of-sample win over a constant or trailing-mean forecast. That is unusual at these "
            "horizons and warrants checking for leakage before it is believed."
        )
    lines += _decile_lines(run)

    # ------------------------------------------------------------------ 3
    lines += [
        "",
        "## 3. Uncertainty and interval coverage",
        "",
        "Bands are split-conformal, calibrated on validation residuals only. That coverage claim "
        "is marginal and finite-sample: it holds only if the calibration residuals are exchangeable "
        "with the test residuals, and overlapping forward windows at these horizons break exactly "
        "that assumption. Coverage below nominal is therefore the expected direction of the error, "
        "not evidence that the construction is broken.",
        "",
    ]
    lines += (
        _md_table(
            ["horizon", "model", "empirical coverage", "nominal", "met nominal", "mean width",
             "calibration rows", "source"],
            [
                [_md_cell(r["horizon"]), _md_cell(r["model"]), _pct(r["empirical_coverage"]),
                 _pct(r["nominal_coverage"]),
                 "n/a" if _is_missing(r["met_nominal"]) else ("yes" if r["met_nominal"] else "no"),
                 _fmt(r["mean_interval_width"]), _fmt(r["calibration_rows"], 0), _md_cell(r["source"])]
                for r in coverage_rows
            ],
        )
        if coverage_rows
        else ["_No interval coverage was available for this run._"]
    )
    lines += [""] + _coverage_prose(coverage_rows)
    lines += _interval_report_lines(run)

    # ------------------------------------------------------------------ 4
    lines += [
        "",
        "## 4. Probability and calibration",
        "",
        "**A point forecast is not a probability.** The scores below are Brier scores of two "
        "genuinely different objects, and neither is a squashed regression output: the empirical "
        "score comes from each model's own out-of-sample residual CDF, and the calibrated score "
        "from a separately fitted direction classifier. Rescaling a return prediction into [0, 1] "
        "and scoring it here would measure nothing.",
        "",
    ]
    lines += (
        _md_table(
            ["horizon", "model", "brier (empirical)", "brier (calibrated)", "n"],
            [
                [_md_cell(r["horizon"]), _md_cell(r["model"]), _fmt(r["brier_empirical"]),
                 _fmt(r["brier_calibrated"]), _fmt(r["n"], 0)]
                for r in brier_rows
            ],
        )
        if brier_rows
        else ["_No Brier scores were available for this run._"]
    )
    lines += [
        "",
        "The direction model is fitted once per fold, independently of which regressor produced the "
        "point forecast, so its calibrated probability is identical across the models within a "
        "horizon. Repeated calibrated values down a column are expected, not a copy error.",
        "",
    ]
    lines += _brier_prose(brier_rows)
    lines += _reliability_prose(run)

    # ------------------------------------------------------------------ 5
    lines += ["", "## 5. Regime breakdown", ""] + _regime_lines(run)

    # ------------------------------------------------------------------ 6
    lines += ["", "## 6. Data and caveats", ""]
    lines += _dataset_lines(dataset if isinstance(dataset, Mapping) else {})
    lines += ["", "**Caveats**", ""] + _caveat_lines(run, records, status)

    # ------------------------------------------------------------------ 7
    lines += ["", "## 7. Files written", "", "**Tables**", ""]
    lines += _md_table(
        ["file", "contents"],
        [
            ["horizon_comparison.csv", "best model per horizon with its pooled metrics"],
            ["model_comparison.csv", "every horizon/model pair with its pooled metrics"],
            ["regime_analysis.csv", "metrics by market regime, or a not-available row"],
            ["prediction_calibration.csv", "per-model calibration statistics for both "
                                           "probability routes"],
            ["return_distributions.csv", "return-distribution statistics"],
        ],
    )
    lines += ["", "**Figures**", ""]
    lines += _md_table(
        ["file", "contents"],
        [
            ["horizon_rmse.png", "pooled RMSE by horizon and model"],
            ["horizon_ic.png", "pooled Spearman IC by horizon and model"],
            ["interval_coverage.png", "conformal coverage against the nominal target"],
            ["prediction_calibration.png", "reliability diagram for the direction probability"],
            ["return_distributions.png", "realised vs predicted return distribution per horizon"],
            ["model_comparison.png", "per-horizon model ranking with direction accuracy"],
        ],
    )
    lines += [
        "",
        f"Summary: `{SUMMARY_NAME}` (this document).",
        "",
        "All twelve artefacts were written from the same pooled predictions by "
        "`src.v3.report`, so the numbers above and the numbers in the CSVs cannot disagree.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary -> %s", path)
    return path


def write_report(
    run: Any, out_dir: Path | str, extra: Mapping[str, object] | None = None
) -> ReportArtefacts:
    """Write the whole report - five tables, six figures, one summary - and index it.

    Each stage is also a public function and is called here exactly as a caller
    would call it, so there is one code path rather than a private fast path that
    could drift from the public one.  The analysis frames are therefore resolved
    more than once per run; that is cheap next to the walk-forward itself, and the
    alternative - threading a cache through the public signatures - would make
    the API harder to use for the sake of a few milliseconds.
    """
    directory = _ensure_dir(out_dir)
    tables = write_tables(run, directory)
    figures = (
        plot_rmse_by_horizon(run, directory),
        plot_ic_by_horizon(run, directory),
        plot_interval_coverage(run, directory),
        plot_prediction_calibration(run, directory),
        plot_return_distributions(run, directory),
        plot_horizon_comparison(run, directory),
    )
    summary = write_summary(run, directory, extra)

    produced = (tuple(p.name for p in tables), tuple(p.name for p in figures))
    if produced != (REQUIRED_TABLES, REQUIRED_FIGURES):
        logger.warning(
            "report artefact names differ from the required set: tables=%s figures=%s",
            produced[0],
            produced[1],
        )
    logger.info(
        "Report complete: %d table(s), %d figure(s) and a summary in %s",
        len(tables), len(figures), directory,
    )
    return ReportArtefacts(directory=directory, tables=tables, figures=figures, summary=summary)
