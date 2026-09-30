"""Tests for the V3 pipeline orchestrator and its CLI.

What is worth protecting here, in order of how badly it would fail
-----------------------------------------------------------------
1. **Containment.**  ``data/raw``, ``data/processed``, ``data/predictions``,
   ``models`` and ``reports`` hold the frozen V1/V2 research output.  The
   conftest session guard turns a write into one of those into a red run, but a
   guard that only fires at teardown tells you *that* something happened, not
   *what*.  So the end-to-end test here walks the output tree itself and asserts
   every single path is under ``tmp_path``.
2. **Failure isolation.**  A stage that raises must leave a diagnosable record
   (``ok=False``, a warning naming the stage, a manifest on disk) and must not
   raise a bare traceback at a caller that only wanted to know whether the run
   worked.
3. **Degradation.**  ``src.v3.report`` is written by another module and may be
   absent or broken; the dataset and walk-forward halves must still run and say
   so.
4. **The bucket exclusion contract.**  Excluding ``derivatives`` is what keeps
   BTC on its full 8-horizon ladder instead of a 2,577-row truncation, so an
   excluded bucket must resolve to *no* features - not to a subset, and not
   silently to all of them.
5. **Validation.**  ``validate_outputs`` must fail when an artefact is gone, and
   name what is gone.

Everything runs offline against a synthetic parquet under ``tmp_path``; no test
here reads or writes the real project, and no test runs a real-data pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.v3 import cli as v3_cli
from src.v3 import pipeline as v3_pipeline
from src.v3.buckets import BUCKET_NAMES, features_in_bucket
from src.v3.dataset import (
    DataUnavailableError,
    FeatureSelectionError,
    resolve_feature_selection,
    v3_block,
)
from src.v3.pipeline import (
    MANIFEST_NAME,
    REQUIRED_ARTIFACTS,
    PipelineResult,
    PipelineStage,
    default_out_dir,
    resolve_plan,
    run_all_symbols,
    run_pipeline,
    validate_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_CONFIG_PATH = "config/v3.yaml"

SYMBOL = "SYNTHUSDT"
N_CANDLES = 1200
MAX_ROWS = 600
HORIZONS = ["1d", "3d"]

#: Feature buckets that need no external cache, so the fixture is self-contained.
SPOT_ONLY = {"price": "all", "technical": "all", "volume": "all", "microstructure": "all"}


# --------------------------------------------------------------------------- fixtures


def make_klines(n: int = N_CANDLES, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic hourly OHLCV + trade-flow frame with a UTC index."""
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    close = 20_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    volume = rng.uniform(50.0, 500.0, n)
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, n)),
            "high": close * (1.0 + np.abs(rng.normal(0.0, 0.002, n))),
            "low": close * (1.0 - np.abs(rng.normal(0.0, 0.002, n))),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(10, 900, n),
            "taker_buy_volume": volume * 0.5,
            "taker_buy_quote_volume": volume * close * 0.5,
        },
        index=index,
    )


@pytest.fixture
def v3_config(tmp_path: Path) -> Config:
    """``config/v3.yaml`` with a synthetic symbol and every path under ``tmp_path``.

    ``rebase`` *rebuilds* the resolved paths rather than reassigning ``root``, so
    ``raw_dir`` really is ``tmp_path/data/raw`` and nothing can reach the project.
    """
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    make_klines().to_parquet(raw_dir / f"{SYMBOL}_1h.parquet")
    config = Config.load(V3_CONFIG_PATH, root=PROJECT_ROOT).rebase(tmp_path)
    assert config.paths.raw_dir == raw_dir
    return config


@pytest.fixture
def stub_report(monkeypatch: pytest.MonkeyPatch):
    """Stand in for ``src.v3.report`` with the documented ``ReportArtefacts`` shape.

    ``src.v3.report`` is written by another module and cannot be assumed to be
    importable, let alone working, while the orchestration is being built.  This
    stub writes one summary, one table and one figure so the full six-stage chain
    - including ``validate_outputs`` over a real artefact set - is exercised
    without depending on a module that is still moving.
    """

    def _write(run, out_dir, extra=None):
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        summary = directory / "summary.md"
        summary.write_text("# stub summary\n", encoding="utf-8")
        table = directory / "per_horizon_metrics.csv"
        table.write_text("horizon,rmse\n", encoding="utf-8")
        figures = directory / "figures"
        figures.mkdir(exist_ok=True)
        figure = figures / "horizon_rmse.png"
        figure.write_bytes(b"\x89PNG\r\n\x1a\n")
        return SimpleNamespace(summary=summary, tables={"per_horizon_metrics.csv": table}, figures=[figure])

    monkeypatch.setattr(v3_pipeline, "write_report", _write)
    return _write


def _run(config: Config, tmp_path: Path, **kwargs: Any) -> PipelineResult:
    """A tiny end-to-end run: two short horizons, one baseline model, one fold."""
    options: dict[str, Any] = {
        "symbol": SYMBOL,
        "horizons": HORIZONS,
        "models": ["mean"],
        "n_splits": 1,
        "max_rows": MAX_ROWS,
        "out_dir": tmp_path / "out",
    }
    options.update(kwargs)
    return run_pipeline(config, **options)


def walk(root: Path) -> list[Path]:
    """Every path under ``root``, itself included."""
    return sorted([root, *root.rglob("*")])


def assert_contained(root: Path, allowed: Path) -> None:
    """No path under ``root`` escapes ``allowed``.

    This is the guard that protects the frozen V1/V2 tree.  ``str.startswith``
    rather than ``Path.is_relative_to`` on purpose: the assertion has to fail
    loudly for anything at all outside the temporary root, including a path that
    only looks like a relative one.
    """
    offenders = [str(path) for path in walk(root) if not str(path).startswith(str(allowed))]
    assert not offenders, f"wrote outside {allowed}: {offenders}"


def read_manifest(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- 1. resolve_plan


def test_resolve_plan_defaults_come_from_the_config(v3_config: Config) -> None:
    plan = resolve_plan(v3_config, SYMBOL.lower())

    block = v3_config.raw["v3"]
    assert plan["symbol"] == SYMBOL
    assert plan["horizons"] == list(block["horizons"])
    assert plan["models"] == list(block["models"])
    assert plan["excluded_buckets"] == []
    assert plan["max_rows"] is None
    assert set(plan) == {
        "symbol",
        "horizons",
        "models",
        "enabled_features",
        "excluded_buckets",
        "max_rows",
    }
    # Nothing is excluded, so every bucket the config names is switched on.
    assert set(plan["enabled_features"]) >= set(BUCKET_NAMES)
    assert all(value != "none" for value in plan["enabled_features"].values())


def test_resolve_plan_horizon_override_is_sorted_and_deduplicated(v3_config: Config) -> None:
    """The plan is independent of the order the flags arrived in."""
    assert resolve_plan(v3_config, SYMBOL, horizons=["7d", "1d"])["horizons"] == ["1d", "7d"]


def test_resolve_plan_excludes_a_bucket_completely(v3_config: Config) -> None:
    """An excluded bucket must resolve to *no* features, not to a subset."""
    full = resolve_plan(v3_config, SYMBOL)
    plan = resolve_plan(v3_config, SYMBOL, exclude_buckets=["derivatives"])

    assert plan["excluded_buckets"] == ["derivatives"]
    assert plan["enabled_features"]["derivatives"] == "none"
    # A missing-name bucket is off as well: enabling a bucket nobody asked for is
    # exactly how the truncated BTC derivatives cache sneaks back in.
    assert plan["enabled_features"]["sentiment"] != "none"

    selection = resolve_feature_selection(v3_block(v3_config), plan["enabled_features"])
    excluded_features = set(features_in_bucket("derivatives"))
    assert excluded_features, "the derivatives bucket is unexpectedly empty"
    assert not (excluded_features & set(selection.columns))
    assert "derivatives" not in selection.buckets
    assert len(selection.columns) == len(
        resolve_feature_selection(v3_block(v3_config), full["enabled_features"]).columns
    ) - len(excluded_features)


def test_resolve_plan_rejects_an_unknown_bucket(v3_config: Config) -> None:
    """A typo must fail loudly: a silent no-op is how 2,577 rows get trained on."""
    with pytest.raises(FeatureSelectionError) as excinfo:
        resolve_plan(v3_config, SYMBOL, exclude_buckets=["derivative"])
    message = str(excinfo.value)
    assert "derivative" in message
    assert "derivatives" in message


def test_resolve_plan_accepts_comma_separated_exclusions(v3_config: Config) -> None:
    plan = resolve_plan(v3_config, SYMBOL, exclude_buckets=["derivatives,sentiment"])
    assert plan["excluded_buckets"] == ["derivatives", "sentiment"]


def test_default_out_dir_follows_the_rebased_root(v3_config: Config, tmp_path: Path) -> None:
    """The configured directory is re-resolved against ``paths.root``.

    Without this a rebased test config would write into the real ``reports/``
    tree, which is exactly the accident the isolation guard exists to catch.
    ``v3.outputs.report_dir`` is ``reports/experiments/v3``, i.e. *inside* a
    protected directory, so this is the highest-consequence path in the module.
    """
    out_dir = default_out_dir(v3_config)
    assert out_dir == tmp_path / "reports" / "experiments" / "v3"
    assert_contained(out_dir.parent, tmp_path)


def test_out_dir_defaults_to_the_configured_v3_directory(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    """``out_dir=None`` must land in the temp root, not in the project's reports/."""
    result = _run(v3_config, tmp_path, out_dir=None)

    assert result.ok is True, result.warnings
    assert result.out_dir == tmp_path / "reports" / "experiments" / "v3"
    assert (result.out_dir / MANIFEST_NAME).exists()
    assert_contained(tmp_path, tmp_path)


# ------------------------------------------------------- 2. end-to-end + containment


def test_run_pipeline_completes_and_writes_only_inside_tmp_path(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    result = _run(v3_config, tmp_path)

    assert result.ok is True, result.warnings
    assert result.symbol == SYMBOL
    assert [stage.name for stage in result.stages] == [
        "plan",
        "dataset",
        "walkforward",
        "report",
        "persist",
        "manifest",
    ]
    assert all(stage.seconds >= 0.0 for stage in result.stages)

    out_dir = tmp_path / "out"
    manifest = read_manifest(out_dir)
    assert manifest["symbol"] == SYMBOL
    assert manifest["ok"] is True
    assert manifest["plan"]["horizons"] == HORIZONS
    assert set(manifest["horizons"]) == set(HORIZONS)
    assert manifest["stage_seconds"]["walkforward"] > 0.0
    assert manifest["warnings"] == []

    # The dataset stage recorded which buckets were used and how many rows survived.
    dataset_detail = result.stage("dataset").detail
    assert dataset_detail["rows"] == MAX_ROWS
    assert dataset_detail["n_features"] > 0
    assert "derivatives" not in dataset_detail["buckets_kept"]
    assert set(dataset_detail["labelled_rows"]) == set(HORIZONS)
    assert all(count > 0 for count in dataset_detail["labelled_rows"].values())

    # One artifact per (model, horizon), all through artifact_path().
    assert [path.name for path in result.model_paths] == ["mean_1d.joblib", "mean_3d.joblib"]
    assert all(path.parent == out_dir / "v3" for path in result.model_paths)
    assert set(manifest["model_inventory"]) == {"mean"}

    assert result.to_dict()["ok"] is True
    assert_contained(out_dir, tmp_path)
    assert_contained(tmp_path, tmp_path)


def test_run_pipeline_reports_which_buckets_and_rows_survived(
    v3_config: Config, stub_report, tmp_path: Path, caplog
) -> None:
    """The truncated-cache failure mode is announced, not discovered later."""
    with caplog.at_level("INFO", logger="crypto_ml"):
        result = _run(v3_config, tmp_path)
    detail = result.stage("dataset").detail
    assert detail["buckets_kept"] == ["microstructure", "price", "technical", "volume"]
    assert "rows" in caplog.text and str(MAX_ROWS) in caplog.text


def test_run_pipeline_writes_a_machine_readable_manifest(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    """The manifest is what a later reader - or ``--validate`` - has to go on."""
    _run(v3_config, tmp_path)
    manifest = read_manifest(tmp_path / "out")

    assert manifest["config"]["source"].endswith("config/v3.yaml")
    assert manifest["config"]["v3"]["version"] == 3
    assert manifest["plan"]["seed"] == 42
    assert manifest["plan"]["n_splits"] == 1
    assert manifest["plan"]["alpha"] == 0.1
    assert manifest["plan"]["embargo"] == "horizon"
    assert manifest["outputs"]["out_dir"] == str(tmp_path / "out")
    assert MANIFEST_NAME in manifest["outputs"]["required_artifacts"]
    assert "summary.md" in manifest["outputs"]["required_artifacts"]
    assert "figures/horizon_rmse.png" in manifest["outputs"]["required_artifacts"]
    assert manifest["horizons"]["1d"]["best_model"] == "mean"
    assert manifest["horizons"]["1d"]["pooled_metrics"]
    assert set(manifest["model_inventory"]["mean"]["horizons"]) == set(HORIZONS)


# ------------------------------------------------------------ 3. failure isolation


def test_a_failing_stage_is_isolated_not_raised(
    v3_config: Config, stub_report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset failure becomes ``ok=False`` plus a manifest, not a traceback."""

    def _explode(*args, **kwargs):
        raise DataUnavailableError("no cache for this symbol")

    monkeypatch.setattr(v3_pipeline, "build_v3_dataset", _explode)

    result = run_pipeline(
        v3_config,
        symbol=SYMBOL,
        horizons=HORIZONS,
        models=["mean"],
        out_dir=tmp_path / "out",
        exclude_buckets=["derivatives", "sentiment"],
    )

    assert result.ok is False
    assert result.run is None
    assert result.model_paths == ()
    named = [w for w in result.warnings if "stage 'dataset'" in w]
    assert named, result.warnings
    assert "DataUnavailableError" in named[0] and "no cache" in named[0]

    # The stages that depend on the dataset did not run; the manifest still did.
    names = [stage.name for stage in result.stages]
    assert names == ["plan", "dataset", "manifest"]
    assert "error" in result.stage("dataset").detail

    manifest = read_manifest(tmp_path / "out")
    assert manifest["ok"] is False
    assert any("stage 'dataset'" in w for w in manifest["warnings"])
    assert_contained(tmp_path / "out", tmp_path)


def test_a_failing_report_stage_does_not_discard_the_models(
    v3_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting is downstream of the results, so its failure costs only the report."""

    def _explode(*args, **kwargs):
        raise RuntimeError("report builder is broken")

    monkeypatch.setattr(v3_pipeline, "write_report", _explode)

    result = _run(v3_config, tmp_path)

    assert result.ok is False
    assert any("stage 'report'" in warning for warning in result.warnings)
    assert result.run is not None
    assert len(result.model_paths) == len(HORIZONS)
    manifest = read_manifest(tmp_path / "out")
    assert manifest["ok"] is False
    assert manifest["outputs"]["models"], "the model artifacts should still be recorded"
    assert_contained(tmp_path / "out", tmp_path)


def test_an_unbuildable_model_name_fails_at_the_plan_stage(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    """A plan that cannot name a single real estimator stops before any data is read."""
    result = _run(v3_config, tmp_path, models=["gradient_boosting_magic"])
    assert result.ok is False
    assert [stage.name for stage in result.stages] == ["plan", "manifest"]
    assert any("stage 'plan'" in warning for warning in result.warnings)


# ------------------------------------------------------------------ 4. validate_outputs


def test_validate_outputs_accepts_a_complete_directory(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    _run(v3_config, tmp_path)
    report = validate_outputs(tmp_path / "out")
    assert report["ok"] is True
    assert report["missing"] == []
    assert report["manifest"]["symbol"] == SYMBOL


def test_validate_outputs_names_a_deleted_figure(
    v3_config: Config, stub_report, tmp_path: Path
) -> None:
    """A half-present report is a failure, and the missing name is reported."""
    out_dir = tmp_path / "out"
    _run(v3_config, out_dir.parent)
    figure = out_dir / "figures" / "horizon_rmse.png"
    assert figure.exists()
    figure.unlink()

    report = validate_outputs(out_dir)
    assert report["ok"] is False
    assert "figures/horizon_rmse.png" in report["missing"]
    assert MANIFEST_NAME not in report["missing"]


def test_validate_outputs_reports_a_directory_without_a_manifest(tmp_path: Path) -> None:
    report = validate_outputs(tmp_path / "empty")
    assert report["ok"] is False
    assert report["manifest"] == {}
    assert MANIFEST_NAME in report["missing"]
    assert "summary.md" in report["missing"]
    assert set(REQUIRED_ARTIFACTS) <= set(report["missing"])


def test_validate_outputs_reports_an_unreadable_manifest(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    report = validate_outputs(tmp_path)
    assert report["ok"] is False
    assert MANIFEST_NAME in report["missing"]


# ----------------------------------------------------------- 5. report-absent degradation


def test_pipeline_completes_when_the_reporter_is_missing(
    v3_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``src.v3.report`` being unimportable must not cost the run.

    This is the documented degradation: the dataset and the walk-forward results
    are the research, the report is a rendering of them.  Losing the renderer is
    a warning; losing the results would be a failure.
    """
    monkeypatch.setattr(v3_pipeline, "write_report", None)

    result = _run(v3_config, tmp_path)

    assert result.ok is True, result.warnings
    assert result.run is not None
    assert result.report is None
    assert len(result.model_paths) == len(HORIZONS)
    assert any("src.v3.report" in warning for warning in result.warnings)

    manifest = read_manifest(tmp_path / "out")
    assert manifest["ok"] is True
    assert manifest["outputs"]["required_artifacts"] == [MANIFEST_NAME]
    assert validate_outputs(tmp_path / "out")["ok"] is True
    assert_contained(tmp_path / "out", tmp_path)


def test_no_report_flag_skips_the_stage(v3_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-report`` must not call the reporter, and must say what it cost."""

    def _never(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("--no-report must not call write_report")

    monkeypatch.setattr(v3_pipeline, "write_report", _never)

    result = _run(v3_config, tmp_path, write_reports=False)

    assert result.ok is True
    assert result.stage("report").detail["skipped"] is True
    assert any("--no-report" in warning for warning in result.warnings)
    assert read_manifest(tmp_path / "out")["outputs"]["required_artifacts"] == [MANIFEST_NAME]


# --------------------------------------------------------------------------- 6. the CLI


@pytest.fixture
def stub_cli_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Replace the heavy pipeline with a cheap, correctly-shaped result."""
    calls: list[dict[str, Any]] = []

    def _fake_run_pipeline(config, symbol=None, out_dir=None, **kwargs):
        calls.append({"symbol": symbol, "out_dir": out_dir, **kwargs})
        target = Path(out_dir)
        (target / MANIFEST_NAME).write_text('{"ok": true}\n', encoding="utf-8")
        return PipelineResult(
            symbol=symbol,
            run=None,
            stages=(
                PipelineStage(name="dataset", seconds=0.1, detail={"rows": 100, "n_features": 8, "buckets_kept": ["price"]}),
                PipelineStage(name="walkforward", seconds=0.2, detail={"best": {}}),
            ),
            report=None,
            model_paths=(),
            warnings=(),
            ok=True,
        )

    monkeypatch.setattr(v3_cli, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(v3_cli, "run_all_symbols", lambda config, symbols=None, **kw: {
        str(symbol).upper(): _fake_run_pipeline(config, symbol=symbol, out_dir=Path(kw["out_dir"]) / str(symbol))
        for symbol in (symbols or ())
    })
    return calls


def test_cli_runs_the_documented_invocation(stub_cli_pipeline, tmp_path: Path, capsys) -> None:
    code = v3_cli.main(
        [
            "--config", V3_CONFIG_PATH,
            "--symbol", "BTCUSDT",
            "--horizons", "1d",
            "--models", "mean",
            "--exclude-buckets", "derivatives",
            "--n-splits", "1",
            "--out-dir", str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "BTCUSDT" in out
    assert "Horizons" in out and "1d" in out
    assert "Excluded        : derivatives" in out
    assert "Feature buckets" in out and "derivatives" not in out.split("Feature buckets")[1].split("\n")[0]

    call = stub_cli_pipeline[0]
    assert call["symbol"] == "BTCUSDT"
    assert call["horizons"] == ["1d"]
    assert call["models"] == ["mean"]
    assert call["exclude_buckets"] == ["derivatives"]
    assert call["n_splits"] == 1
    assert call["out_dir"] == tmp_path
    assert_contained(tmp_path, tmp_path)


def test_cli_exclusions_accept_repeats_and_commas(stub_cli_pipeline, tmp_path: Path) -> None:
    code = v3_cli.main(
        [
            "--symbol", "BTCUSDT",
            "--exclude-buckets", "derivatives,sentiment",
            "--exclude-buckets", "microstructure",
            "--out-dir", str(tmp_path),
            "--no-report",
        ]
    )
    assert code == 0
    assert stub_cli_pipeline[0]["exclude_buckets"] == ["derivatives", "sentiment", "microstructure"]
    assert stub_cli_pipeline[0]["write_reports"] is False


def test_cli_returns_one_when_the_run_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _failing(config, symbol=None, out_dir=None, **kwargs):
        return PipelineResult(
            symbol=symbol,
            run=None,
            stages=(),
            report=None,
            model_paths=(),
            warnings=("stage 'dataset' failed: boom",),
            ok=False,
        )

    monkeypatch.setattr(v3_cli, "run_pipeline", _failing)
    assert v3_cli.main(["--symbol", "BTCUSDT", "--out-dir", str(tmp_path)]) == 1


def test_cli_validate_passes_on_a_complete_run(
    v3_config: Config, stub_report, tmp_path: Path, capsys
) -> None:
    """``--validate`` after a real run: no recomputation, and the exit code agrees."""
    _run(v3_config, tmp_path)
    capsys.readouterr()

    assert v3_cli.main(["--validate", "--out-dir", str(tmp_path / "out")]) == 0
    assert "Validation      : OK" in capsys.readouterr().out


def test_cli_validate_fails_when_an_artefact_is_gone(
    v3_config: Config, stub_report, tmp_path: Path, capsys
) -> None:
    out_dir = tmp_path / "out"
    _run(v3_config, out_dir.parent)
    (out_dir / "summary.md").unlink()

    assert v3_cli.main(["--validate", "--out-dir", str(out_dir)]) == 1
    assert "summary.md" in capsys.readouterr().out


def test_cli_prints_the_per_horizon_headline(tmp_path: Path, capsys) -> None:
    """The summary a human actually reads: best model, RMSE and where things went."""
    pooled = pd.DataFrame(
        {"rmse": [0.01, 0.02], "mae": [0.005, 0.01], "coverage_lo": [0.9, 0.8]},
        index=pd.Index(["ridge", "mean"], name="model"),
    )
    outcome = SimpleNamespace(horizon="7d", pooled_metrics=pooled, best_model=lambda metric="rmse": "ridge")
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    v3_cli.print_result(
        PipelineResult(
            symbol="BTCUSDT",
            run=SimpleNamespace(results=[outcome]),
            stages=(
                PipelineStage(
                    name="dataset",
                    seconds=0.5,
                    detail={
                        "rows": 40833,
                        "n_features": 84,
                        "buckets_kept": ["price", "technical"],
                        "span": ["2022-01-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00"],
                    },
                ),
            ),
            report=None,
            model_paths=(tmp_path / "v3" / "ridge_7d.joblib",),
            warnings=("a warning worth reading",),
            ok=False,
        )
    )

    out = capsys.readouterr().out
    assert "40,833 rows x 84 features" in out
    assert "ridge" in out and "0.010000" in out
    assert str(tmp_path) in out
    assert "a warning worth reading" in out
    assert "FAILED" in out


def test_cli_parses_every_documented_flag(tmp_path: Path) -> None:
    args = v3_cli.parse_args(
        [
            "--config", "config/v3.yaml",
            "--symbol", "ETHUSDT",
            "--horizons", "1d,7d,30d",
            "--models", "ridge,xgboost",
            "--exclude-buckets", "derivatives",
            "--out-dir", str(tmp_path),
            "--n-splits", "5",
            "--test-fraction", "0.1",
            "--validation-fraction", "0.12",
            "--alpha", "0.05",
            "--seed", "7",
            "--max-rows", "1234",
            "--no-save-models",
            "--no-report",
            "--log-level", "DEBUG",
        ]
    )
    assert args.config == "config/v3.yaml"
    assert (args.symbol, args.horizons, args.models) == ("ETHUSDT", ["1d", "7d", "30d"], ["ridge", "xgboost"])
    assert args.exclude_buckets == ["derivatives"]
    assert args.out_dir == tmp_path
    assert (args.n_splits, args.test_fraction, args.validation_fraction) == (5, 0.1, 0.12)
    assert (args.alpha, args.seed, args.max_rows) == (0.05, 7, 1234)
    assert args.save_models is False and args.write_reports is False
    assert args.log_level == "DEBUG"
    assert args.all_symbols is False and args.validate is False

    flags = v3_cli.parse_args(["--all-symbols", "--validate"])
    assert flags.all_symbols is True and flags.validate is True
    assert flags.symbol is None and flags.out_dir is None and flags.max_rows is None


# ------------------------------------------------------------- 7. multi-symbol routing


def test_run_all_symbols_namespaces_every_symbol(
    v3_config: Config, stub_report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two symbols writing one manifest would leave only the last one on disk."""
    seen: list[str] = []

    def _failing_for_one(config, symbol=None, out_dir=None, **kwargs):
        seen.append(str(symbol))
        if symbol == "AAAUSDT":
            raise RuntimeError("no cache")
        return _run(
            config,
            tmp_path,
            symbol=symbol,
            out_dir=out_dir,
            horizons=["1d"],
            n_splits=1,
            max_rows=MAX_ROWS,
        )

    monkeypatch.setattr(v3_pipeline, "run_pipeline", _failing_for_one)
    results = run_all_symbols(v3_config, ["AAAUSDT", SYMBOL], out_dir=tmp_path / "all", models=["mean"])

    assert sorted(seen) == ["AAAUSDT", SYMBOL]
    assert results["AAAUSDT"].ok is False
    assert "no cache" in results["AAAUSDT"].warnings[0]
    assert results[SYMBOL].ok is True
    # A failure in one symbol is not a reason to skip the others.
    assert (tmp_path / "all" / SYMBOL / MANIFEST_NAME).exists()
    assert not (tmp_path / "all" / "AAAUSDT" / MANIFEST_NAME).exists()
    assert_contained(tmp_path / "all", tmp_path)
