#!/usr/bin/env python
"""Stage 1 — download historical BTCUSDT klines from Binance.

Examples
--------
    python scripts/download_data.py
    python scripts/download_data.py --start 2024-01-01 --end 2024-06-30
    python scripts/download_data.py --symbol ETHUSDT --interval 4h
    python scripts/download_data.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline.run import prepare_data  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance klines and validate them.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--symbol", default=None, help="Override data.symbol, e.g. ETHUSDT")
    parser.add_argument("--interval", default=None, help="Override data.interval, e.g. 4h")
    parser.add_argument("--start", default=None, help="Override data.start_date, e.g. 2024-01-01")
    parser.add_argument("--end", default=None, help="Override data.end_date (default: up to the last closed candle)")
    parser.add_argument("--no-download", action="store_true", help="Validate whatever is already on disk")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation checks")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if validation reports an error")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config).with_overrides(
        {
            "data.symbol": args.symbol,
            "data.interval": args.interval,
            "data.start_date": args.start,
            "data.end_date": args.end,
        }
    )
    config.paths.ensure()

    result = prepare_data(
        config,
        download=not args.no_download,
        validate=not args.no_validate,
        fail_on_error=args.fail_on_error or None,
    )

    print()
    print(f"Raw data : {result.download.path if result.download else 'data/raw'}")  # type: ignore[union-attr]
    print(f"Candles  : {len(result.klines):,} rows")
    print(f"Range    : {result.klines.index.min()} -> {result.klines.index.max()}")
    print(f"Validation: {result.validation.status()}")
    for path in result.paths.values():
        print(f"  wrote {path}")
    return 0 if result.validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
