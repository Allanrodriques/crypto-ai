#!/usr/bin/env python
"""Run the V3 multi-horizon forward-return pipeline.

    plan -> dataset -> purged walk-forward -> report -> models -> manifest

A thin shim over :func:`src.v3.cli.main`, which is the only implementation: the
same code runs whether it is invoked from here, from ``python -m src.v3.cli``
or from a notebook, so a documented invocation can never drift from the
behaviour under test.

Every setting comes from ``config/v3.yaml`` unless a flag overrides it, and
every output goes under one directory (``--out-dir``).  Nothing is ever written
into the frozen V1/V2 trees.

Examples
--------
    python scripts/run_v3.py                                  # the configured plan
    python scripts/run_v3.py --exclude-buckets derivatives    # full BTC history
    python scripts/run_v3.py --symbol ETHUSDT --horizons 1d,7d --n-splits 5
    python scripts/run_v3.py --all-symbols --models mean --n-splits 1
    python scripts/run_v3.py --validate --out-dir reports/experiments/v3/BTCUSDT
    python scripts/run_v3.py --max-rows 4000 --no-save-models --no-report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.v3.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
