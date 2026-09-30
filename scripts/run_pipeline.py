#!/usr/bin/env python
"""Run the complete research pipeline in one command.

    download -> validate -> features + label -> splits -> train -> evaluate -> predict

Equivalent to running the four other scripts in order, using the same code.

The V2 multi-factor experiment suite (``scripts/run_experiments.py``) is a
separate, opt-in stage.  It is *not* part of the default run on purpose: V1 is
the frozen baseline whose numbers the V2 results are compared against, so
changing the default invocation would silently change the baseline.  Pass
``--experiments`` to run both.

Examples
--------
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-download          # reuse data/raw
    python scripts/run_pipeline.py --models xgboost
    python scripts/run_pipeline.py --no-plots --no-cv     # fast smoke run
    python scripts/run_pipeline.py --experiments          # V1 then the V2 suite
    python scripts/run_pipeline.py --experiments-only --experiments-smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.models import ALL_MODELS  # noqa: E402
from src.pipeline.run import run_full_pipeline  # noqa: E402
from src.utils import format_timestamp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full data -> features -> ML -> evaluation pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-download", action="store_true", help="Reuse the klines already in data/raw")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS),
                        help="Which registered models to train")
    parser.add_argument("--no-cv", action="store_true", help="Skip time-series cross-validation")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--no-predict", action="store_true", help="Skip the final inference stage")

    v2 = parser.add_argument_group(
        "V2 experiment suite",
        "Opt-in: the V1 run above is the frozen baseline and is left untouched.",
    )
    v2.add_argument("--experiments", action="store_true",
                    help="Also run the V2 multi-factor experiment suite after V1")
    v2.add_argument("--experiments-only", action="store_true",
                    help="Run only the V2 experiment suite, skipping the V1 pipeline")
    v2.add_argument("--experiments-smoke", action="store_true",
                    help="Fast V2 run: 2 walk-forward windows, no CV, no SHAP")
    v2.add_argument("--experiments-no-studies", action="store_true",
                    help="Skip the ablation, target and multi-asset studies in the V2 run")
    return parser.parse_args(argv)


def _run_experiments(args: argparse.Namespace) -> int:
    """Invoke the V2 suite through its own entry point.

    The suite is called rather than reimplemented so there is exactly one place
    that decides how an experiment is built, evaluated and reported.
    """
    from scripts.run_experiments import main as run_experiments_main

    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.no_download:
        argv.append("--no-download")
    if args.experiments_smoke:
        argv.append("--smoke")
    if not args.experiments_no_studies:
        argv.append("--all")
    print("\n" + "=" * 78)
    print("V2 EXPERIMENT SUITE")
    print("=" * 78)
    return run_experiments_main(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.experiments_only:
        return _run_experiments(args)

    outcome = run_full_pipeline(
        config,
        download=not args.no_download,
        model_names=args.models,
        run_cv=not args.no_cv,
        write_plots=not args.no_plots,
        skip_predict=args.no_predict,
    )

    dataset = outcome["dataset"]
    evaluation = outcome["evaluation"]
    training = outcome["training"]
    latest = outcome["latest_prediction"]

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"Symbol / interval : {config.symbol} {config.interval}")
    print(f"Data range        : {format_timestamp(dataset.frame.index.min())[:16]} -> {format_timestamp(dataset.frame.index.max())[:16]}")
    print(f"Dataset           : {len(dataset.frame):,} rows x {len(dataset.feature_columns)} features")
    print(f"Target            : {dataset.metadata['target']['definition']}")
    print(f"Validation        : {outcome['data'].validation.status()}")
    print(f"Selected model    : {training.selected_name}")
    print(f"Test period       : {format_timestamp(dataset.split('test').start)[:16]} -> "
          f"{format_timestamp(dataset.split('test').end)[:16]}")

    if latest is not None:
        print(f"\nLatest prediction : {latest.describe(config.symbol)}")
        print("                      (a model score, not a guaranteed probability of profit)")

    print("\nModel comparison on the untouched test period:")
    table = evaluation.summary_table()
    if not table.empty:
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nArtefacts:")
    print(f"  raw data     : {config.paths.raw_dir}")
    print(f"  dataset      : {config.paths.processed_dir}")
    print(f"  models       : {config.paths.models_dir}")
    print(f"  metrics      : {config.paths.metrics_dir}")
    print(f"  backtests    : {config.paths.backtests_dir}")
    print(f"  plots        : {config.paths.plots_dir} ({len(evaluation.plots)} figure(s))")
    print(f"  summary      : {config.paths.metrics_dir / 'summary.md'}")
    print(f"  predictions  : {outcome['prediction_path']}")
    print("\nModel quality and trading performance are reported separately: see reports/metrics/summary.md")

    if args.experiments:
        return _run_experiments(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
