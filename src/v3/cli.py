"""Command-line entry point for the V3 pipeline.

Why a CLI module rather than a ``scripts/`` argument parser
----------------------------------------------------------
V3 has flags the V1 scripts do not have (``--horizons``, ``--exclude-buckets``,
``--all-symbols``, ``--validate``) and a V3 run is a *plan* before it is a
computation: which horizons, which models, which feature buckets, written where.
``config/v3.yaml`` is the default answer to all of those, so the CLI's job is to
override it explicitly and print the resolved plan back before running it.  The
logic lives in :mod:`src.v3.pipeline`; this module only parses, prints and
decides the exit code, so ``scripts/run_v3.py`` stays a two-line shim and a
notebook can call ``main([...])`` and get the same behaviour.

Exit codes
----------
``0``  every requested run succeeded (and, with ``--validate``, the output
       directory is complete).
``1``  a run failed, or validation found a missing artefact, or the arguments
       named a config/symbol that cannot be run at all.

The reported numbers are out-of-sample walk-forward metrics: the best model per
horizon and its pooled RMSE, plus the conformal coverage.  Model quality is not
trading performance and neither is a guarantee of profit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from src.config import Config, load_config
from src.utils import format_timestamp, get_logger, set_global_seed
from src.v3.models import MODEL_NAMES
from src.v3.pipeline import (
    PipelineResult,
    default_out_dir,
    default_symbols,
    resolve_plan,
    run_all_symbols,
    run_pipeline,
    validate_outputs,
)

logger = get_logger("v3.cli")

DEFAULT_CONFIG = "config/v3.yaml"

#: Roots the CLI may configure, in one place so the help text and the defaults
#: cannot drift apart.
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _csv(value: str | None) -> list[str]:
    """Split one ``--flag`` occurrence, which may be comma separated.

    ``None`` means "the flag was not given", which is a different thing from an
    empty value: it must yield no items at all rather than one item spelled
    ``"None"``.
    """
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """The documented V3 command line."""
    parser = argparse.ArgumentParser(
        prog="run_v3",
        description="Run the V3 multi-horizon forward-return pipeline.",
        epilog=(
            "Every flag defaults to config/v3.yaml, so a bare run executes the configured "
            "plan. Excluding a bucket is how a truncated external cache is handled, e.g. "
            "--exclude-buckets derivatives for BTCUSDT."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--symbol", default=None, help="symbol to run (default: v3.primary_symbol)")
    parser.add_argument(
        "--horizons",
        default=None,
        help="comma-separated horizon labels, e.g. 1d,7d,30d (default: v3.horizons)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=f"comma-separated model names (default: v3.models); available: {list(MODEL_NAMES)}",
    )
    parser.add_argument(
        "--exclude-buckets",
        action="append",
        default=None,
        metavar="BUCKET[,BUCKET...]",
        help="feature bucket(s) to switch off; repeatable or comma separated",
    )
    parser.add_argument("--out-dir", default=None, help="output directory (default: v3.outputs.report_dir)")
    parser.add_argument("--max-rows", type=int, default=None, help="keep only the most recent N rows")
    parser.add_argument("--n-splits", type=int, default=3, help="walk-forward folds (default: 3)")
    parser.add_argument("--test-fraction", type=float, default=0.08, help="test block fraction (default: 0.08)")
    parser.add_argument(
        "--validation-fraction", type=float, default=0.08, help="validation block fraction (default: 0.08)"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage level (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="global seed (default: 42)")
    parser.add_argument("--no-save-models", dest="save_models", action="store_false", help="do not persist model artifacts")
    parser.add_argument("--no-report", dest="write_reports", action="store_false", help="do not write the report")
    parser.add_argument(
        "--all-symbols", dest="all_symbols", action="store_true", help="run every symbol in v3.symbols"
    )
    parser.add_argument(
        "--validate", action="store_true", help="check an existing output directory instead of running"
    )
    parser.add_argument("--log-level", default="INFO", choices=list(_LOG_LEVELS), help="logging verbosity")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` into the run's parameters.

    Comma-separated lists are expanded here rather than at each use site, so
    ``--exclude-buckets derivatives,sentiment`` and
    ``--exclude-buckets derivatives --exclude-buckets sentiment`` are the same
    request by the time anything downstream sees it.
    """
    args = build_parser().parse_args(argv)
    args.horizons = [item for value in (args.horizons,) for item in _csv(value)] or None
    args.models = [item for value in (args.models,) for item in _csv(value)] or None
    args.exclude_buckets = [item for value in (args.exclude_buckets or ()) for item in _csv(value)]
    args.out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    return args


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".6f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def print_plan(config: Config, symbol: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Print and return the resolved plan, before any data is read.

    Printing the plan is not decoration: the two ways this pipeline goes wrong -
    an unintended bucket, a horizon the data cannot support - are both visible
    here and invisible in the results.
    """
    plan = resolve_plan(
        config,
        symbol,
        horizons=args.horizons,
        models=args.models,
        exclude_buckets=args.exclude_buckets,
    )
    enabled = {
        name: value
        for name, value in dict(plan["enabled_features"]).items()
        if value not in (False, "none", None, [])
    }
    print("\n" + "=" * 78)
    print(f"V3 PIPELINE PLAN - {symbol}")
    print("=" * 78)
    print(f"Config          : {config.source_path}")
    print(f"Horizons        : {', '.join(plan['horizons'])}")  # type: ignore[arg-type]
    print(f"Models          : {', '.join(plan['models'])}")  # type: ignore[arg-type]
    print(f"Feature buckets : {', '.join(enabled) or 'none'}")
    if plan["excluded_buckets"]:
        print(f"Excluded        : {', '.join(plan['excluded_buckets'])}")  # type: ignore[arg-type]
    print(f"Folds           : {args.n_splits} (test {args.test_fraction}, validation {args.validation_fraction}, alpha {args.alpha})")
    print(f"Max rows        : {args.max_rows if args.max_rows is not None else 'all'}")
    print(f"Output          : {out_dir}")
    return plan


def print_result(result: PipelineResult) -> None:
    """Print one symbol's headline: best model and RMSE per horizon."""
    dataset_stage = result.stage("dataset")
    if dataset_stage is not None and not dataset_stage.detail.get("error"):
        detail = dataset_stage.detail
        print(
            f"Dataset         : {int(detail.get('rows', 0)):,} rows x "
            f"{int(detail.get('n_features', 0))} features "
            f"(buckets: {', '.join(detail.get('buckets_kept') or []) or 'none'})"
        )
        span = detail.get("span") or []
        if len(span) == 2:
            print(f"Span            : {format_timestamp(span[0])[:16]} -> {format_timestamp(span[1])[:16]}")

    print(f"\n{result.symbol}: out-of-sample walk-forward results")
    print(f"  {'horizon':<10} {'best model':<14} {'rmse':>10} {'mae':>10} {'coverage':>10}")
    for outcome in getattr(result.run, "results", []) or []:
        best = outcome.best_model()
        metrics = outcome.pooled_metrics.loc[best]
        coverage = metrics.get("coverage_lo", None)
        print(
            f"  {outcome.horizon:<10} {best:<14} {_fmt(metrics.get('rmse')):>10} "
            f"{_fmt(metrics.get('mae')):>10} "
            f"{(_fmt(coverage, '.3f') if coverage is not None else 'n/a'):>10}"
        )

    out_dir = result.out_dir
    if out_dir is None and result.model_paths:
        # No stage recorded the directory (a caller-constructed result, or a run
        # that failed before the plan finished): fall back to where the artifacts
        # actually landed, which is the answer the reader wants anyway.
        out_dir = result.model_paths[0].parent.parent
    print(f"\nOutput          : {out_dir}")
    if result.model_paths:
        print(f"Model artifacts : {len(result.model_paths)} file(s) under {out_dir}")
    if result.report is not None:
        print(f"Report          : {out_dir}")
    for warning in result.warnings:
        print(f"  warning       : {warning}")
    print(f"Status          : {'OK' if result.ok else 'FAILED'}")


def print_validation(out_dir: Path) -> bool:
    """Print the outcome of :func:`validate_outputs`; return its ``ok``."""
    report = validate_outputs(out_dir)
    if report["ok"]:
        print(f"Validation      : OK - {out_dir} holds a complete run")
        return True
    print(f"Validation      : FAILED - {out_dir} is missing {len(report['missing'])} artefact(s)")
    for name in report["missing"]:  # type: ignore[union-attr]
        print(f"  missing       : {name}")
    return False


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run (or validate) a V3 pipeline.  Returns a process exit code."""
    args = parse_args(argv)
    logging.getLogger("crypto_ml").setLevel(args.log_level)
    set_global_seed(args.seed)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Cannot load config {args.config}: {type(exc).__name__}: {exc}")
        return 1

    # ---- validate an existing directory: no computation at all.
    if args.validate and args.out_dir is not None:
        return 0 if print_validation(args.out_dir) else 1

    if args.all_symbols and args.symbol:
        logger.warning("--symbol is ignored with --all-symbols; running the configured universe")

    run_kwargs: dict[str, Any] = {
        "horizons": args.horizons,
        "models": args.models,
        "exclude_buckets": args.exclude_buckets,
        "max_rows": args.max_rows,
        "n_splits": args.n_splits,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "save_models": args.save_models,
        "write_reports": args.write_reports,
        "alpha": args.alpha,
        "seed": args.seed,
    }

    try:
        if args.all_symbols:
            universe = default_symbols(config)
            base = args.out_dir if args.out_dir is not None else default_out_dir(config)
            print("\n" + "=" * 78)
            print(f"V3 PIPELINE - {len(universe)} symbol(s): {', '.join(universe)}")
            print("=" * 78)
            results = run_all_symbols(config, universe, out_dir=base, **run_kwargs)
        else:
            symbol = str(args.symbol or default_symbols(config)[0]).strip().upper()
            out_dir = args.out_dir if args.out_dir is not None else default_out_dir(config)
            print_plan(config, symbol, args, out_dir)
            results = {symbol: run_pipeline(config, symbol=symbol, out_dir=out_dir, **run_kwargs)}
    except Exception as exc:
        logger.error("V3 pipeline could not start: %s: %s", type(exc).__name__, exc, exc_info=True)
        print(f"\nPipeline could not start: {type(exc).__name__}: {exc}")
        return 1

    print("\n" + "=" * 78)
    for symbol, result in results.items():
        print_result(result)
    failed = [symbol for symbol, result in results.items() if not result.ok]
    if failed:
        print(f"\n{len(failed)} of {len(results)} symbol(s) failed: {', '.join(failed)}")

    if args.validate:
        for result in results.values():
            out_dir = result.out_dir
            if out_dir is not None and not print_validation(out_dir):
                return 1
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
