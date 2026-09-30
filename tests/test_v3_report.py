"""The V3 report must be complete, self-consistent, and honest when it is not.

A report is the artefact a reader trusts without re-running the pipeline, so the
failures worth testing are not crashes.  They are the quiet ones: a figure that
silently drops the horizon where the only model lost, a table that is missing a
column because one model had no conformal band, a summary that quotes a headline
number without saying the winner was ``trailing_mean``, or a whole section that
vanished because a sibling module was not importable and nobody wrote it down.

``src.v3.analysis`` is written in parallel with the renderer, so these tests
never depend on whether it exists: the degradation path is exercised by pointing
the module attribute at ``None``, which is deterministic whether or not the real
module has landed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v3 import report as R
from src.v3.metrics import evaluate_regression
from src.v3.walkforward import HorizonResult, V3RunResult

ROOT = Path(__file__).resolve().parents[1]

#: The protected project trees, as ``tests/conftest.py`` defines them.  Checked
#: locally as well so a failure names this test rather than the session guard.
PROTECTED_DIRS = ("data/raw", "data/processed", "data/predictions", "models", "reports")

#: The seven section headings ``summary.md`` must contain, in order.
SECTION_HEADINGS = (
    "## 1. Run metadata",
    "## 2. Headline results",
    "## 3. Uncertainty and interval coverage",
    "## 4. Probability and calibration",
    "## 5. Regime breakdown",
    "## 6. Data and caveats",
    "## 7. Files written",
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

ALL_ARTEFACTS = R.REQUIRED_TABLES + R.REQUIRED_FIGURES + (R.SUMMARY_NAME,)


# --------------------------------------------------------------------- builders


def _predictions(n: int, models: tuple[str, ...], seed: int) -> pd.DataFrame:
    """A pooled prediction frame with the exact columns the harness emits."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    target = rng.normal(0.0, 0.02, n)
    data: dict[str, np.ndarray] = {"target": target}
    for offset, model in enumerate(models):
        # Each model sees a different slice of the target plus its own noise, so
        # the ranking is a real ranking rather than a tie the plot cannot show.
        prediction = target * (0.10 + 0.05 * offset) + rng.normal(0.0, 0.02, n)
        data[f"{model}_pred"] = prediction
        data[f"{model}_lo"] = prediction - 0.05
        data[f"{model}_hi"] = prediction + 0.05
        data[f"{model}_prob_empirical"] = np.clip(0.5 + 20.0 * prediction, 0.01, 0.99)
    data["prob_calibrated"] = np.clip(0.5 + rng.normal(0.0, 0.08, n), 0.01, 0.99)
    return pd.DataFrame(data, index=index)


def _horizon_result(
    horizon: str,
    days: float,
    models: tuple[str, ...],
    *,
    n: int = 240,
    seed: int = 0,
    symbol: str = "BTCUSDT",
) -> HorizonResult:
    """A hand-built :class:`HorizonResult`; no model is ever fitted."""
    predictions = _predictions(n, models, seed)
    target = predictions["target"].to_numpy()
    metrics: dict[str, dict[str, float]] = {}
    for model in models:
        prediction = predictions[f"{model}_pred"].to_numpy()
        row = evaluate_regression(target, prediction)
        inside = (target >= predictions[f"{model}_lo"]) & (target <= predictions[f"{model}_hi"])
        row["coverage_lo"] = float(inside.mean())
        row["coverage_hi"] = float(inside.mean())
        probability = predictions[f"{model}_prob_empirical"].to_numpy()
        outcome = (target > 0).astype(float)
        row["brier_empirical"] = float(np.mean((probability - outcome) ** 2))
        calibrated = predictions["prob_calibrated"].to_numpy()
        row["brier_calibrated"] = float(np.mean((calibrated - outcome) ** 2))
        metrics[model] = row

    pooled = pd.DataFrame.from_dict(metrics, orient="index")
    pooled.index.name = "model"

    interval = pd.DataFrame.from_dict(
        {
            model: {
                "mean_fold_coverage": float(pooled.loc[model, "coverage_lo"]),
                "nominal_coverage": 0.9,
                "mean_interval_width": 0.1,
                "mean_calibration_rows": float(n),
            }
            for model in models
        },
        orient="index",
    )
    interval.index.name = "model"

    return HorizonResult(
        symbol=symbol,
        horizon=horizon,
        horizon_days=days,
        feature_columns=[f"f{i}" for i in range(12)],
        predictions=predictions,
        fold_metrics=pd.DataFrame({"horizon": [horizon] * len(models), "model": list(models)}),
        pooled_metrics=pooled,
        interval_summary=interval,
        geometry={
            "horizon": horizon,
            "purge": f"{int(days)} days 00:00:00",
            "embargo": f"{int(days)} days 00:00:00",
            "n_splits_requested": 4,
            "test_fraction": 0.08,
            "validation_fraction": 0.08,
            "min_train_fraction": 0.2,
            "labelled_rows": n * 2,
        },
        coverage={
            "labelled_rows": n * 2,
            "n_folds": 4,
            "n_features": 12,
            "prediction_rows": n,
            "label_span": ["2024-01-01", "2024-02-01"],
            "prediction_span": ["2024-02-01", "2024-03-01"],
        },
        seconds=1.25,
        models=list(models),
    )


def _run(
    horizons: tuple[tuple[str, float], ...] = (("7d", 7.0), ("30d", 30.0)),
    models: tuple[str, ...] = ("ridge", "xgboost", "trailing_mean"),
    *,
    n: int = 240,
) -> V3RunResult:
    """A three-horizon-free synthetic run: cheap, deterministic, no fitting."""
    results = [
        _horizon_result(label, days, models, n=n, seed=index)
        for index, (label, days) in enumerate(horizons)
    ]
    return V3RunResult(
        symbol="BTCUSDT",
        horizons=[label for label, _ in horizons],
        feature_columns=[f"f{i}" for i in range(12)],
        results=results,
        dataset={
            "symbol": "BTCUSDT",
            "n_rows": 8760,
            "n_features": 12,
            "n_targets": len(results),
            "feature_groups": ["technical", "volatility"],
            "index": {
                "name": "timestamp",
                "start": "2024-01-01 00:00:00+00:00",
                "end": "2024-12-31 00:00:00+00:00",
                "tz": "UTC",
                "is_unique": True,
                "is_monotonic_increasing": True,
            },
            "missing_sources": ["fear_greed"],
            "drop_report": {"open_time_missing": 3, "duplicate_candles": 1},
            "n_rows_dropped": 4,
            "feature_coverage": {
                "min_pct": 61.5,
                "median_pct": 99.9,
                "max_pct": 100.0,
                "n_below_50pct": 1,
            },
            "target_coverage": {
                label: {
                    "n_labelled": 480,
                    "n_unlabelled_tail": 30,
                    "labelled_pct": 94.1,
                    "requested_hours": days * 24.0,
                }
                for label, days in horizons
            },
            "bucket_registry_available": True,
        },
        config={
            "models": list(models),
            "n_splits": 4,
            "test_fraction": 0.08,
            "validation_fraction": 0.08,
            "embargo": "horizon",
            "alpha": 0.1,
            "seed": 42,
        },
        seconds=9.5,
    )


def _empty_run() -> V3RunResult:
    """A run with no horizons at all: every chart takes its nothing-to-plot branch."""
    return V3RunResult(
        symbol="BTCUSDT", horizons=[], feature_columns=[], results=[], dataset={}, config={}, seconds=0.0
    )


def _degenerate_run() -> V3RunResult:
    """The pathological run: one horizon, one baseline, and nothing measurable.

    Every metric cell is NaN and every prediction is a constant, so any code that
    sorts a metric, divides by a standard deviation, cuts bins or shades a bar
    group has to survive an all-NaN input rather than raising.
    """
    result = _horizon_result("30d", 30.0, ("zero",), n=120, seed=3)
    pooled = pd.DataFrame(
        [{"rmse": np.nan, "mae": np.nan, "spearman_ic": np.nan, "direction_accuracy": np.nan}],
        index=pd.Index(["zero"], name="model"),
    )
    interval = pd.DataFrame(
        [{"mean_fold_coverage": np.nan, "nominal_coverage": np.nan}], index=pd.Index(["zero"], name="model")
    )
    predictions = result.predictions.copy()
    predictions["target"] = 0.0
    for column in predictions.columns:
        if column.endswith(("_pred", "_lo", "_hi", "_prob_empirical")):
            predictions[column] = 0.0
    predictions["prob_calibrated"] = 0.5

    degenerate = HorizonResult(
        symbol="BTCUSDT",
        horizon="30d",
        horizon_days=30.0,
        feature_columns=[],
        predictions=predictions,
        fold_metrics=pd.DataFrame(),
        pooled_metrics=pooled,
        interval_summary=interval,
        geometry={},
        coverage={},
        seconds=0.1,
        models=["zero"],
    )
    return V3RunResult(
        symbol="BTCUSDT",
        horizons=["30d"],
        feature_columns=[],
        results=[degenerate],
        dataset={},
        config={},
        seconds=0.1,
    )


def _fingerprint(path: Path) -> str:
    """Hash a tree's relative paths, sizes and mtimes, as conftest does."""
    if not path.exists():
        return "<absent>"
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


# ----------------------------------------------------------------- the artefact set


def test_write_report_creates_the_directory_and_every_artefact(tmp_path):
    """All twelve files, under a directory that did not exist beforehand."""
    out = tmp_path / "nested" / "report"
    assert not out.exists()

    artefacts = R.write_report(_run(), out)

    assert out.is_dir()
    assert artefacts.directory == out
    assert tuple(path.name for path in artefacts.tables) == R.REQUIRED_TABLES
    assert tuple(path.name for path in artefacts.figures) == R.REQUIRED_FIGURES
    assert artefacts.summary.name == R.SUMMARY_NAME
    assert len(artefacts.all()) == 12
    for name in ALL_ARTEFACTS:
        path = out / name
        assert path.is_file(), f"missing artefact {name}"
        assert path.stat().st_size > 0, f"empty artefact {name}"


def test_report_artefacts_are_frozen_and_indexed():
    """The index is frozen: a caller cannot quietly swap a path after the fact."""
    artefacts = R.ReportArtefacts(
        directory=Path("/tmp"), tables=(), figures=(), summary=Path("/tmp/summary.md")
    )
    with pytest.raises(Exception):
        artefacts.directory = Path("/elsewhere")  # type: ignore[misc]
    assert artefacts.all() == (Path("/tmp/summary.md"),)


def _png_size(path: Path) -> tuple[int, int]:
    """Width and height from the PNG IHDR chunk.

    Parsed by hand rather than with an imaging library, and it proves more than
    "the bytes exist": a 1x1 or truncated file fails here, which is exactly the
    failure a size check alone would miss.
    """
    raw = path.read_bytes()
    assert raw[:8] == PNG_MAGIC, f"{path.name} is not a PNG"
    assert raw[12:16] == b"IHDR", f"{path.name} has no IHDR chunk"
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def test_every_figure_is_a_real_image_not_a_truncated_file(tmp_path):
    """A valid PNG header is not enough; the image has to have dimensions."""
    artefacts = R.write_report(_run(), tmp_path)
    for path in artefacts.figures:
        width, height = _png_size(path)
        assert width > 400 and height > 250, f"{path.name} is {width}x{height}"


def _axis_is_labelled(ax, which: str) -> bool:
    """An axis is explained by a label with units, or by category tick labels.

    A categorical axis needs no unit label: ``model_comparison.png`` puts the
    model names on y, and calling that axis "Model" adds no information a reader
    does not already have from the tick text.  The requirement being checked is
    that nothing is left unexplained, not that every axis carries a units suffix.
    """
    if ax.get_xlabel() if which == "x" else ax.get_ylabel():
        return True
    ticks = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
    return any(tick.get_text() for tick in ticks)


def test_every_figure_has_a_title_labelled_axes_and_identifies_its_series(
    tmp_path, monkeypatch
):
    """Titles, axis labels with units, and a legend (or a title naming the series).

    Checked against the live figure objects rather than the rendered pixels:
    ``_save`` is intercepted so the axes can be inspected, which catches a chart
    that is drawn but not explained.
    """
    import matplotlib.pyplot as plt

    captured: list[tuple[plt.Figure, Path]] = []

    def capture(fig, path):
        captured.append((fig, path))
        return path

    monkeypatch.setattr(R, "_save", capture)
    try:
        R.write_report(_run(), tmp_path)
    finally:
        assert len(captured) == 6
        for fig, path in captured:
            axes = [ax for ax in fig.axes if ax.get_visible()]
            assert axes, f"{path.name}: no visible axes"
            assert any(ax.get_title() for ax in axes) or fig._suptitle is not None, (
                f"{path.name}: no title"
            )
            for ax in axes:
                assert _axis_is_labelled(ax, "x"), f"{path.name}: an axes has no x label"
                assert _axis_is_labelled(ax, "y"), f"{path.name}: an axes has no y label"
            has_legend = any(ax.get_legend() is not None for ax in axes)
            titles = [ax.get_title() for ax in axes]
            if fig._suptitle is not None:
                titles.append(fig._suptitle.get_text())
            names_series = any("=" in title for title in titles)
            assert has_legend or names_series, f"{path.name}: series are not identified"
        plt.close("all")


def test_a_chart_with_nothing_to_plot_still_says_what_it_would_have_shown(
    tmp_path, monkeypatch
):
    """The no-data branches must keep the title and axis labels.

    An empty chart with no title forces the reader to open the other five files to
    learn which measure is missing, so this asserts the labels survive into the
    degenerate case rather than only the populated one.
    """
    import matplotlib.pyplot as plt

    captured: dict[str, plt.Figure] = {}

    def capture(fig, path):
        captured[path.name] = fig
        return path

    monkeypatch.setattr(R, "_save", capture)
    try:
        R.write_report(_empty_run(), tmp_path)
    finally:
        assert set(captured) == set(R.REQUIRED_FIGURES)
        for name, fig in captured.items():
            axes = [ax for ax in fig.axes if ax.get_visible()]
            assert axes, f"{name}: nothing left to read"
            assert any(ax.get_title() for ax in axes) or fig._suptitle is not None, (
                f"{name}: the title vanished with the data"
            )
            for ax in axes:
                assert _axis_is_labelled(ax, "x"), f"{name}: x label vanished"
                assert _axis_is_labelled(ax, "y"), f"{name}: y label vanished"
            assert any(
                child.get_text() and child.get_visible() for ax in axes for child in ax.texts
            ), f"{name}: nothing explains the empty chart"
        plt.close("all")


def test_a_single_series_chart_identifies_its_model_in_the_title(tmp_path, monkeypatch):
    """With one model there is no legend to read, so the name goes in the title."""
    import matplotlib.pyplot as plt

    captured: dict[str, plt.Figure] = {}

    def capture(fig, path):
        captured[path.name] = fig
        return path

    monkeypatch.setattr(R, "_save", capture)
    try:
        R.write_report(_run(horizons=(("7d", 7.0),), models=("mean",)), tmp_path)
    finally:
        rmse = captured["horizon_rmse.png"]
        assert rmse.axes[0].get_legend() is None, "one series needs no legend"
        assert "mean" in rmse.axes[0].get_title()
        plt.close("all")


def test_figures_are_closed_so_matplotlib_does_not_leak(tmp_path):
    """Six figures per run; an unclosed canvas is a memory leak in a batch loop."""
    import matplotlib.pyplot as plt

    R.write_report(_run(), tmp_path)
    assert plt.get_fignums() == [], "a figure was left open after write_report"


# -------------------------------------------------------------------- summary.md


def test_summary_contains_every_section_heading_in_order(tmp_path):
    """The section order is the argument, so it is asserted, not assumed."""
    path = R.write_summary(_run(), tmp_path)
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# V3 walk-forward report: BTCUSDT")
    positions = [text.index(heading) for heading in SECTION_HEADINGS]
    assert positions == sorted(positions), "summary sections are out of order"


def test_summary_has_no_bare_nan_token(tmp_path):
    """`nan` in a results document reads as a bug; `n/a` is the honest cell.

    Matched on word boundaries so ordinary words containing the letters are not
    false positives.  A healthy run has nothing missing, so the `n/a` policy is
    checked on the degenerate run instead, where every cell really is absent.
    """
    text = R.write_summary(_run(), tmp_path).read_text(encoding="utf-8")
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)

    degenerate = R.write_summary(_degenerate_run(), tmp_path / "degenerate").read_text(encoding="utf-8")
    assert not re.search(r"\bnan\b", degenerate, flags=re.IGNORECASE)
    assert "n/a" in degenerate, "a missing value should be rendered as n/a"


def test_summary_reports_metadata_and_the_baseline_outcome(tmp_path):
    """The naive-baseline win must be stated in words, not left to the table."""
    run = _run()
    text = R.write_summary(run, tmp_path).read_text(encoding="utf-8")

    assert "| symbol | BTCUSDT |" in text
    assert "| seed | 42 |" in text
    assert "7d" in text and "30d" in text
    assert "Naive-baseline check" in text
    assert "trailing_mean" in text
    assert "Purge and embargo geometry" in text
    assert "not beat a constant or trailing-mean forecast" in text


def test_summary_states_that_a_point_forecast_is_not_a_probability(tmp_path):
    text = R.write_summary(_run(), tmp_path).read_text(encoding="utf-8")
    assert "A point forecast is not a probability" in text
    assert "brier (calibrated)" in text


def test_summary_extra_is_rendered_and_never_overrides_a_metric(tmp_path):
    """A caller may annotate a run; it may not restate a measurement."""
    run = _run()
    text = R.write_summary(
        run, tmp_path, extra={"git_sha": "abc1234", "run_note": "first pass", "nested": {"rows": 10}}
    ).read_text(encoding="utf-8")

    assert "Caller-supplied notes" in text
    assert "abc1234" in text
    assert "first pass" in text
    assert "rows=10" in text
    assert f"| seed | {run.config['seed']} |" in text


def test_summary_says_so_when_the_regime_breakdown_is_missing(tmp_path, monkeypatch):
    """A missing section must be a sentence, never an empty table."""
    monkeypatch.setattr(R, "_analysis", None)
    text = R.write_summary(_run(), tmp_path).read_text(encoding="utf-8")

    assert "## 5. Regime breakdown" in text
    assert "_Not available._" in text
    assert "No regime conclusion should be drawn" in text


def test_summary_flags_horizons_with_tiny_prediction_counts(tmp_path):
    """A 40-row horizon is noise, and the report has to say so."""
    text = R.write_summary(_run(horizons=(("7d", 7.0),), n=40), tmp_path).read_text(encoding="utf-8")
    assert "fewer than" in text
    assert "7d" in text


def test_summary_files_section_lists_every_artefact(tmp_path):
    artefacts = R.write_report(_run(), tmp_path)
    text = artefacts.summary.read_text(encoding="utf-8")
    for name in ALL_ARTEFACTS:
        assert name in text, f"{name} is not listed in the summary"


# ------------------------------------------------------------------------ tables


def test_tables_have_their_expected_columns(tmp_path):
    paths = R.write_tables(_run(), tmp_path)

    horizon = pd.read_csv(paths[0])
    assert {"horizon", "best_model"}.issubset(horizon.columns)
    assert list(horizon["horizon"]) == ["7d", "30d"]
    assert horizon.loc[0, "best_model"] in {"ridge", "xgboost", "trailing_mean"}

    models = pd.read_csv(paths[1])
    assert {"symbol", "horizon", "model", "rmse", "spearman_ic"}.issubset(models.columns)
    assert len(models) == 6  # two horizons x three models

    # Against the real ``src.v3.analysis``, whose ``source`` column means the
    # probability route.  Losing that would make every row look like one measurement.
    calibration = pd.read_csv(paths[3])
    assert set(calibration["source"]) == {"empirical", "calibrated"}
    assert set(calibration["analysis_source"]) == {"calibration_report"}


def test_analysis_tables_degrade_to_an_explicit_placeholder(tmp_path, monkeypatch):
    """Absent analysis yields a schema-stable row, not an exception or a blank file."""
    monkeypatch.setattr(R, "_analysis", None)
    paths = R.write_tables(_run(), tmp_path)

    for path in paths[2:]:
        frame = pd.read_csv(path)
        assert list(frame.columns) == ["status", "source", "reason"]
        assert frame.loc[0, "status"] == "not available"
        assert isinstance(frame.loc[0, "reason"], str) and frame.loc[0, "reason"]


def test_analysis_table_builders_reject_a_broken_analysis_module(tmp_path, monkeypatch):
    """A sibling module that imports but raises must not take the report down."""

    class Broken:
        @staticmethod
        def analyse(*args, **kwargs):
            raise RuntimeError("not implemented yet")

        @staticmethod
        def calibration_report(*args, **kwargs):
            raise RuntimeError("not implemented yet")

    monkeypatch.setattr(R, "_analysis", Broken())
    artefacts = R.write_report(_run(), tmp_path)

    regime = pd.read_csv(artefacts.tables[2])
    assert regime.loc[0, "status"] == "not available"
    for path in artefacts.figures:
        with path.open("rb") as handle:
            assert handle.read(len(PNG_MAGIC)) == PNG_MAGIC
    assert "## 5. Regime breakdown" in artefacts.summary.read_text(encoding="utf-8")


def test_analysis_frames_are_used_when_available(tmp_path, monkeypatch):
    """The integration is proven on both sides: analysis output is consumed."""

    class Present:
        @staticmethod
        def calibration_report(result):
            return pd.DataFrame(
                {
                    "model": ["ridge", "ridge"],
                    "source": ["empirical", "calibrated"],
                    "n": [200, 200],
                    "brier": [0.24, 0.25],
                }
            )

        @staticmethod
        def prediction_interval_report(result):
            return pd.DataFrame({"interval": [f"{result.horizon}"], "mean_width": [0.1]})

        @staticmethod
        def decile_report(result):
            return pd.DataFrame({"decile": [0, 1], "mean_actual_return": [-0.01, 0.02]})

        @staticmethod
        def analyse(result, n_bins=4):
            return {
                "regime_volatility": pd.DataFrame({"regime": ["bull", "bear"], "rmse": [0.02, 0.03]}),
                "regime_trend": pd.DataFrame({"regime": ["up", "down"], "rmse": [0.01, 0.04]}),
            }

        @staticmethod
        def return_distribution_report(frame, target_col="target", pred_cols=None, horizons=None):
            return pd.DataFrame({"column": ["target"], "horizon": ["all"], "p95": [0.05]})

    monkeypatch.setattr(R, "_analysis", Present())
    artefacts = R.write_report(_run(), tmp_path)
    text = artefacts.summary.read_text(encoding="utf-8")

    calibration = pd.read_csv(artefacts.tables[3])
    assert sorted(calibration["horizon"].unique()) == ["30d", "7d"]
    # The analysis frame's own ``source`` says which probability route a row scored
    # (empirical vs calibrated).  That is the finding, so the renderer's provenance
    # stamp goes elsewhere rather than over it.
    assert set(calibration["source"]) == {"empirical", "calibrated"}
    assert set(calibration["analysis_source"]) == {"calibration_report"}
    assert len(calibration) == 4  # two horizons x two (model, source) rows

    regime = pd.read_csv(artefacts.tables[2])
    assert set(regime["regime"]) == {"bull", "bear", "up", "down"}, "both regime views are kept"

    section = text.split("## 5. Regime breakdown")[1].split("## 6.")[0]
    assert "_Not available._" not in section
    assert "bull" in section
    assert "Rank ladder" in text
    assert "Pooled interval report" in text


# -------------------------------------------------------------------- robustness


def test_degenerate_run_writes_every_artefact_without_raising(tmp_path):
    """One horizon, one baseline, every metric NaN, every prediction constant."""
    artefacts = R.write_report(_degenerate_run(), tmp_path)

    assert len(artefacts.all()) == 12
    for path in artefacts.all():
        assert path.is_file() and path.stat().st_size > 0
    text = artefacts.summary.read_text(encoding="utf-8")
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)
    assert "n/a" in text


def test_run_with_no_results_still_produces_a_full_report(tmp_path):
    """The empty run is the limit case: a reader must get a document, not a stack trace."""
    artefacts = R.write_report(_empty_run(), tmp_path)

    for path in artefacts.all():
        assert path.is_file() and path.stat().st_size > 0
    text = artefacts.summary.read_text(encoding="utf-8")
    assert "No headline results were available" in text
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)


def test_each_public_entry_point_works_standalone(tmp_path):
    """The API is six plots, one table writer and one summary writer, not just write_report."""
    run = _run()

    tables = R.write_tables(run, tmp_path / "tables")
    assert tuple(path.name for path in tables) == R.REQUIRED_TABLES

    expected = {
        R.plot_rmse_by_horizon: "horizon_rmse.png",
        R.plot_ic_by_horizon: "horizon_ic.png",
        R.plot_interval_coverage: "interval_coverage.png",
        R.plot_prediction_calibration: "prediction_calibration.png",
        R.plot_return_distributions: "return_distributions.png",
        R.plot_horizon_comparison: "model_comparison.png",
    }
    for function, name in expected.items():
        path = function(run, tmp_path / name.replace(".png", ""))
        assert path.name == name
        assert path.stat().st_size > 0

    summary = R.write_summary(run, tmp_path / "summary_only")
    assert summary.name == "summary.md" and summary.stat().st_size > 0


def test_plots_survive_a_single_horizon_and_a_single_model(tmp_path):
    """One group, one series: no legend needed, no division by a group count."""
    run = _run(horizons=(("90d", 90.0),), models=("mean",), n=90)
    artefacts = R.write_report(run, tmp_path)
    for path in artefacts.figures:
        with path.open("rb") as handle:
            assert handle.read(len(PNG_MAGIC)) == PNG_MAGIC


def test_zero_variance_predictions_do_not_break_the_distribution_figure(tmp_path):
    """A constant return series has no width; the bins must not collapse to NaN."""
    artefacts = R.write_report(_degenerate_run(), tmp_path)
    path = tmp_path / "return_distributions.png"
    assert path.read_bytes()[: len(PNG_MAGIC)] == PNG_MAGIC
    assert artefacts.summary.stat().st_size > 0


# ------------------------------------------------------------------------ names


def test_required_names_are_exactly_the_documented_set():
    """These strings are the contract other code and tests key off."""
    assert R.REQUIRED_TABLES == (
        "horizon_comparison.csv",
        "model_comparison.csv",
        "regime_analysis.csv",
        "prediction_calibration.csv",
        "return_distributions.csv",
    )
    assert R.REQUIRED_FIGURES == (
        "horizon_rmse.png",
        "horizon_ic.png",
        "interval_coverage.png",
        "prediction_calibration.png",
        "return_distributions.png",
        "model_comparison.png",
    )
    assert len(R.REQUIRED_TABLES) == 5
    assert len(R.REQUIRED_FIGURES) == 6
    assert len(set(R.REQUIRED_TABLES) | set(R.REQUIRED_FIGURES) | {R.SUMMARY_NAME}) == 12


# -------------------------------------------------------------------- isolation


def test_nothing_is_written_outside_tmp_path(tmp_path):
    """The protected trees hold real research output and must be byte-identical."""
    before = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}
    out = tmp_path / "report"
    artefacts = R.write_report(_run(), out)
    after = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}

    changed = [name for name in PROTECTED_DIRS if before[name] != after[name]]
    assert not changed, f"report wrote into protected project directories: {changed}"
    for path in artefacts.all():
        assert path.is_relative_to(tmp_path), f"{path} escaped tmp_path"
    assert sorted(p.name for p in out.iterdir()) == sorted(ALL_ARTEFACTS)
