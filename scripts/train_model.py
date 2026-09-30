#!/usr/bin/env python
"""Stage 3 — train the baseline and ML models.

Fits every model on the training block, scores on validation, runs purged
expanding-window cross-validation, selects a model on **validation** metrics, and
refits the winner on train+validation.  The test split is not read here.

Examples
--------
    python scripts/train_model.py
    python scripts/train_model.py --models xgboost random_forest
    python scripts/train_model.py --no-cv
    python scripts/train_model.py --no-refit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.models import ALL_MODELS  # noqa: E402
from src.pipeline.run import load_dataset  # noqa: E402
from src.pipeline.train import train_models  # noqa: E402
from src.utils import format_timestamp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models on train, select on validation.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS),
        help="Which registered models to train",
    )
    parser.add_argument("--no-cv", action="store_true", help="Skip time-series cross-validation")
    parser.add_argument("--no-refit", action="store_true", help="Do not refit the winner on train+validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config.paths.ensure()

    dataset = load_dataset(config)
    result = train_models(
        config,
        dataset,
        model_names=args.models,
        run_cv=not args.no_cv,
        refit_on_train_plus_validation=not args.no_refit,
    )

    print()
    print(f"Dataset  : {len(dataset.frame):,} rows, {format_timestamp(dataset.frame.index.min())[:10]} -> {format_timestamp(dataset.frame.index.max())[:10]}")
    print(f"Target   : {dataset.metadata['target']['definition']}")
    print("\nValidation metrics (model selection happens here, on validation only):")
    header = f"  {'model':<22} {'accuracy':>9} {'f1':>7} {'roc_auc':>8} {'pr_auc':>7} {'brier':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, trained in result.models.items():
        m, cal = trained.validation_metrics, trained.validation_calibration
        fmt = lambda v, spec=".4f": "n/a" if v is None else format(float(v), spec)  # noqa: E731
        print(
            f"  {name:<22} {fmt(m.get('accuracy')):>9} {fmt(m.get('f1')):>7} "
            f"{fmt(m.get('roc_auc')):>8} {fmt(m.get('pr_auc')):>7} "
            f"{fmt(cal.get('brier_score')):>7}"
        )
    print(f"\nSelected model : {result.selected_name}  (refit on {result.selected.fitted_on})")
    print(f"Artifacts      : {config.paths.models_dir}")
    print("\nThe test period has NOT been read. Run scripts/evaluate_model.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
