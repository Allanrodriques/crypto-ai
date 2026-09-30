#!/usr/bin/env python
"""Stage 4 — evaluate the trained models on the held-out test period.

This is the only stage that reads the test split.  The model choice was already
frozen by ``train_model.py`` on validation, and the backtest threshold comes from
config, so nothing here can be tuned.

Examples
--------
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --no-plots
    python scripts/evaluate_model.py --backtest-baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline.run import load_dataset  # noqa: E402
from src.pipeline.train import train_models  # noqa: E402
from src.pipeline.evaluate import evaluate_models  # noqa: E402
from src.utils import format_timestamp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on the test period (used once).")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--backtest-baselines", action="store_true", help="Also backtest the naive baselines")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation during the refit stage")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config.paths.ensure()

    dataset = load_dataset(config)
    # Selection is repeated here so evaluate_model.py can run standalone; it uses
    # validation only and therefore does not read the test split.
    training = train_models(config, dataset, run_cv=not args.no_cv)
    evaluation = evaluate_models(
        config, training, write_plots=not args.no_plots, backtest_baselines=args.backtest_baselines
    )

    test = dataset.split("test")
    print()
    print(f"Test period: {format_timestamp(test.start)[:16]} -> {format_timestamp(test.end)[:16]} ({len(test):,} rows)")
    print(f"Selected   : {evaluation.selected_name}\n")

    table = evaluation.summary_table()
    columns = [c for c in table.columns]
    print("  " + f"{'model':<22}" + "".join(f"{c:>18}" for c in columns))
    print("  " + "-" * (22 + 18 * len(columns)))
    for name, row in table.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if v is None or (isinstance(v, float) and v != v):
                cells.append(f"{'n/a':>18}")
            elif "return" in c:
                cells.append(f"{float(v):>17.2%}" if "buy_hold" not in c else f"{float(v):>17.2%}")
            elif c == "sharpe":
                cells.append(f"{float(v):>18.3f}")
            else:
                cells.append(f"{float(v):>18.4f}")
        print("  " + f"{name:<22}" + "".join(cells))

    print(f"\nMetrics   : {config.paths.metrics_dir}")
    print(f"Backtests : {config.paths.backtests_dir}")
    if not args.no_plots:
        print(f"Plots     : {config.paths.plots_dir} ({len(evaluation.plots)} figure(s))")
    print(f"Summary   : {config.paths.metrics_dir / 'summary.md'}")
    print("\nModel quality and trading performance are different questions; both are in reports/metrics/summary.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
