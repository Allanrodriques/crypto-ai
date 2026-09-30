#!/usr/bin/env python
"""Run the V2 multi-factor experiment suite.

    for each spec in EXP-00..04:
        build the shared source context once, then
        train -> walk-forward -> regime/importance -> threshold -> backtest

and write, per experiment, a metrics bundle, predictions, the frozen threshold
curve, and a feature manifest hash.  Finally aggregate everything into
``experiment_report.md`` and ``data_availability.csv``.

Examples
--------
    python scripts/run_experiments.py                     # full suite
    python scripts/run_experiments.py --only EXP-00-OHLCV-TECHNICAL
    python scripts/run_experiments.py --smoke             # short walk-forward, no CV
    python scripts/run_experiments.py --output-dir reports/experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.downloader import KlineStore  # noqa: E402
from src.dataset.multisource import SourceUnavailableError, build_context  # noqa: E402
from src.evaluation.walkforward import WalkForwardPlan  # noqa: E402
from src.experiments.runner import run_experiment  # noqa: E402
from src.experiments.spec import EXPERIMENTS  # noqa: E402
from src.experiments.report import write_reports  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("scripts.experiments")

#: Every source any experiment may reference.  The context is built once and
#: shared, so no source is downloaded or aligned more than once per run.
ALL_SOURCES = ("binance_funding", "binance_futures", "fear_greed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 multi-factor experiment suite.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--symbol", default="BTCUSDT", help="Spot symbol to study")
    parser.add_argument("--interval", default="1h", help="Base candle interval")
    parser.add_argument("--output-dir", default="reports/experiments")
    parser.add_argument("--availability-out", default="reports/data_availability.csv")
    parser.add_argument("--only", nargs="+", default=None, choices=list(EXPERIMENTS),
                        help="Run only these experiment ids")
    parser.add_argument("--smoke", action="store_true",
                        help="Fewer walk-forward windows, no CV, no SHAP (fast sanity run)")
    parser.add_argument("--no-download", action="store_true",
                        help="Require the caches to already exist; never hit the network")
    parser.add_argument("--ablation", action="store_true",
                        help="Also run the add/drop feature-group ablation with paired bootstrap")
    parser.add_argument("--targets", action="store_true",
                        help="Also compare alternative label definitions (horizon/threshold/3-class)")
    parser.add_argument("--multiasset", action="store_true",
                        help="Also compare per-asset against global models across the symbol set")
    parser.add_argument("--all", action="store_true",
                        help="Run the experiments plus ablation, targets and multi-asset studies")
    return parser.parse_args(argv)


def _context_for_symbol(root: Path, config, interval: str, allow_download: bool = True):
    """Return a ``context_for(symbol)`` callable for the multi-asset study.

    Sentiment is BTC-specific, so a non-BTC context is built without it rather
    than with a BTC series silently attached to another asset - that would make
    a per-asset model look like it had sentiment information it never had.
    """

    def build(symbol: str):
        store = KlineStore(root / "data" / "raw", symbol, interval)
        if not store.exists():
            logger.warning("no %s %s cache; skipping that symbol", symbol, interval)
            return None
        sources = list(ALL_SOURCES) if symbol.upper().startswith("BTC") else ["binance_funding", "binance_futures"]
        return build_context(
            symbol=symbol,
            sources=sources,
            spot=store.load(),
            cache_dir=root / "data" / "raw",
            allow_download=allow_download,
        )

    return build


def _context(args: argparse.Namespace, config) -> "object":
    """Build the shared aligned source context, or fail with a clear reason."""
    root = Path(__file__).resolve().parent.parent
    store = KlineStore(root / "data" / "raw", args.symbol, args.interval)
    if not store.exists():
        raise SystemExit(
            f"No {args.symbol} {args.interval} cache in data/raw. "
            f"Run: python scripts/download_data.py --symbol {args.symbol} --interval {args.interval}"
        )
    spot = store.load()
    return build_context(
        symbol=args.symbol,
        sources=list(ALL_SOURCES),
        spot=spot,
        cache_dir=root / "data" / "raw",
        allow_download=not args.no_download,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    config = load_config(args.config)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    logger.info("loading and aligning shared source context")
    try:
        context = _context(args, config)
    except SourceUnavailableError as exc:
        # An offline run against an incomplete cache is a configuration
        # mistake, not a crash: report what is missing and how to fix it.
        raise SystemExit(f"cannot build source context: {exc}") from exc

    # Windows are anchored to the end of the pool and step back in time, so the
    # most recent windows land on the most recent data.  `gap` is the horizon, so
    # no training label reaches into its own test block.
    horizon = int(getattr(config, "horizon_candles", 6) or 6)
    plan = WalkForwardPlan(
        n_windows=2 if args.smoke else 5,
        train_size=5_000,
        test_size=1_500,
        gap=horizon,
    )

    chosen = [EXPERIMENTS[k] for k in args.only] if args.only else list(EXPERIMENTS.values())
    summaries = []
    for spec in chosen:
        logger.info("=== %s (%s)", spec.experiment_id, ", ".join(spec.feature_groups))
        result = run_experiment(
            spec,
            config,
            context=context,
            output_dir=out / spec.slug,
            walkforward_plan=plan,
            with_shap=not args.smoke,
            with_importance=not args.smoke,
        )
        summaries.append(result)
        # The headline lives under `test_metrics`, not `test`; reading the wrong
        # key logged NaN for every experiment and hid the fact that the suite
        # had run correctly.
        headline = result.metrics.get("test_metrics", result.metrics.get("test", {})) or {}
        logger.info(
            "%s: rows=%s features=%s test_roc_auc=%s manifest=%s",
            spec.experiment_id,
            result.dataset_rows,
            result.feature_count,
            round(float(headline.get("roc_auc", float("nan"))), 6),
            result.feature_manifest_hash,
        )

    # ---- optional secondary studies -------------------------------------
    # Each writes its own table under reports/experiments/ and is summarised in
    # its own section of the report, so the headline table stays a clean
    # one-row-per-experiment view of the same protocol.
    studies: dict[str, dict] = {}
    baseline = EXPERIMENTS["EXP-00-OHLCV-TECHNICAL"]
    resamples = 100 if args.smoke else 500

    if args.ablation or args.all:
        from src.experiments.ablation import run_ablation

        logger.info("running feature-group ablation")
        studies["ablation"] = run_ablation(
            baseline,
            config,
            context=context,
            output_dir=out / "ablation",
            walkforward=plan,
            n_resamples=resamples,
        )

    if args.targets or args.all:
        from src.experiments.targets import run_target_comparison

        logger.info("running target-definition comparison")
        studies["targets"] = run_target_comparison(
            baseline,
            config,
            context=context,
            output_dir=out / "targets",
            walkforward=None if args.smoke else plan,
            n_resamples=resamples,
        )

    if args.multiasset or args.all:
        from src.experiments.multiasset import DEFAULT_SYMBOLS, run_multiasset

        logger.info("running per-asset vs global comparison over %s", ", ".join(DEFAULT_SYMBOLS))
        studies["multiasset"] = run_multiasset(
            baseline,
            config,
            context_for=_context_for_symbol(
                root, config, args.interval, allow_download=not args.no_download
            ),
            output_dir=out / "multiasset",
            n_resamples=resamples,
        )

    manifest = [
        {
            "experiment_id": s.spec.experiment_id,
            "feature_groups": list(s.spec.feature_groups),
            "feature_count": s.feature_count,
            "dataset_rows": s.dataset_rows,
            "feature_manifest_hash": s.feature_manifest_hash,
        }
        for s in summaries
    ]
    (out / "experiment_index.json").write_text(json.dumps(manifest, indent=2))

    coverage = context.coverage_table()
    availability = root / args.availability_out
    availability.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(availability, index=False)
    logger.info("wrote %s", availability)

    write_reports(
        out=out,
        availability_path=availability,
        summaries=summaries,
        report_path=out / "experiment_report.md",
        studies=studies,
    )
    logger.info("wrote %s", out / "experiment_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
