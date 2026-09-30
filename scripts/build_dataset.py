#!/usr/bin/env python
"""Stage 2 — build the leakage-free ML dataset from validated klines.

Produces ``data/processed/<SYMBOL>_<interval>_dataset.parquet`` plus a metadata
sidecar describing the label, the feature list, and the chronological split
boundaries.

Examples
--------
    python scripts/build_dataset.py
    python scripts/build_dataset.py --horizon 12 --threshold 0.01
    python scripts/build_dataset.py --train-ratio 0.6 --validation-ratio 0.2 --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline.run import build_dataset_stage, load_raw_data  # noqa: E402
from src.utils import format_timestamp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features, label and chronological splits.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--horizon", type=int, default=None, help="Override target.horizon_candles")
    parser.add_argument("--threshold", type=float, default=None, help="Override target.threshold")
    parser.add_argument("--train-ratio", type=float, default=None, help="Override split.train_ratio")
    parser.add_argument("--validation-ratio", type=float, default=None, help="Override split.validation_ratio")
    parser.add_argument("--test-ratio", type=float, default=None, help="Override split.test_ratio")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config).with_overrides(
        {
            "target.horizon_candles": args.horizon,
            "target.threshold": args.threshold,
            "split.train_ratio": args.train_ratio,
            "split.validation_ratio": args.validation_ratio,
            "split.test_ratio": args.test_ratio,
        }
    )
    config.paths.ensure()

    klines = load_raw_data(config)
    dataset = build_dataset_stage(config, klines)

    print()
    print(f"Dataset  : {len(dataset.frame):,} rows x {len(dataset.feature_columns)} features")
    print(f"Target   : {dataset.metadata['target']['definition']}")
    print(f"Splits   : purge={config.split.get('purge_candles') or dataset.horizon_candles} candles (chronological, never shuffled)")
    for name, split in dataset.splits.items():
        dist = split.class_distribution()
        print(
            f"  {name:<11} {len(split):>7,} rows  {format_timestamp(split.start)[:16]} -> "
            f"{format_timestamp(split.end)[:16]}  share_up={dist['share_up']:.3f}"
        )
    print("Leakage checks: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
