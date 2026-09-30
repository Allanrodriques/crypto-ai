"""Bootstrap significance must distinguish three outcomes, not two.

The failure this guards against is subtle and already happened once: a result
whose 95% interval sat entirely *below* zero was reported with
``significant=False``, rendering it identically to an interval that straddled
zero.  A reader scanning a table of flags cannot tell "no evidence" from
"evidence of harm", and the study then quietly overstates a group's value.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.bootstrap import PairedBlockBootstrap, paired_block_bootstrap_auc


def _result(point_a: float, point_b: float, ci_low: float, ci_high: float) -> PairedBlockBootstrap:
    return PairedBlockBootstrap(
        point_a=point_a,
        point_b=point_b,
        point_delta=point_a - point_b,
        ci_low=ci_low,
        ci_high=ci_high,
        probability_a_better=0.5,
        n_resamples=100,
        n_effective=1000,
        block_size=24,
        n_blocks=42,
    )


def test_interval_entirely_above_zero_is_significantly_better():
    r = _result(0.61, 0.60, 0.004, 0.010)
    assert r.significant is True
    assert r.significantly_worse is False
    assert r.verdict == "better"


def test_interval_entirely_below_zero_is_significantly_worse():
    """The 24h-target case: CI [-0.104, -0.015] must not read as 'no result'."""
    r = _result(0.47, 0.53, -0.104, -0.015)
    assert r.significant is False
    assert r.significantly_worse is True
    assert r.verdict == "worse"


def test_interval_straddling_zero_is_indistinguishable():
    r = _result(0.60, 0.60, -0.004, 0.010)
    assert r.significant is False
    assert r.significantly_worse is False
    assert r.verdict == "indistinguishable"


def test_missing_interval_is_indistinguishable_not_significant():
    for r in (
        _result(0.6, 0.5, None, None),
        _result(0.6, 0.5, 0.0, None),
    ):
        assert r.significant is False
        assert r.significantly_worse is False
        assert r.verdict == "indistinguishable"


def test_to_dict_exposes_both_directions_and_the_verdict():
    d = _result(0.47, 0.53, -0.104, -0.015).to_dict()
    assert d["significant_at_95"] is False
    assert d["significantly_worse_at_95"] is True
    assert d["verdict"] == "worse"


def test_identical_predictors_produce_an_interval_around_zero():
    """Two identical scores must not be reported as a significant difference."""
    rng = np.random.default_rng(0)
    n = 2_000
    y = rng.integers(0, 2, n)
    p = np.where(y == 1, 0.7, 0.3)
    r = paired_block_bootstrap_auc(y, p, p.copy(), n_resamples=200, block_size=24)
    assert r.point_delta == pytest.approx(0.0, abs=1e-12)
    assert r.ci_low <= 0.0 <= r.ci_high
    assert r.verdict == "indistinguishable"


def test_a_genuinely_better_predictor_is_detected_as_better():
    rng = np.random.default_rng(7)
    n = 4_000
    y = rng.integers(0, 2, n)
    # Noisy scores with a real AUC gap.  A deterministic split (0.85 vs 0.15)
    # would separate the classes perfectly and give both arms AUC 1.0, which
    # tests nothing about the bootstrap.
    weak = 0.5 + 0.06 * (2 * y - 1) + rng.normal(0, 0.20, n)
    strong = 0.5 + 0.30 * (2 * y - 1) + rng.normal(0, 0.20, n)
    r = paired_block_bootstrap_auc(y, strong, weak, n_resamples=200, block_size=24)
    assert r.point_delta > 0.0
    assert r.significant is True
    assert r.verdict == "better"


def test_a_genuinely_worse_predictor_is_detected_as_worse():
    rng = np.random.default_rng(7)
    n = 4_000
    y = rng.integers(0, 2, n)
    weak = 0.5 + 0.06 * (2 * y - 1) + rng.normal(0, 0.20, n)
    strong = 0.5 + 0.30 * (2 * y - 1) + rng.normal(0, 0.20, n)
    r = paired_block_bootstrap_auc(y, weak, strong, n_resamples=200, block_size=24)
    assert r.point_delta < 0.0
    assert r.significant is False
    assert r.significantly_worse is True
    assert r.verdict == "worse"


def test_report_tables_use_the_three_way_verdict_column():
    """A boolean column in a results table is what produced the ambiguity."""
    from src.experiments.report import _find_table, _markdown_table
    import pandas as pd
    from pathlib import Path
    import tempfile

    table = pd.DataFrame(
        {
            "variant": ["add_derivatives"],
            "delta_roc_auc": [0.0032],
            "ci_low": [-0.0044],
            "ci_high": [0.0102],
            "verdict": ["indistinguishable"],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "ablation").mkdir()
        table.to_csv(root / "ablation" / "ablation_table.csv", index=False)
        found = _find_table(root, "ablation")
        assert not found.empty
        text = _markdown_table(found, ["variant", "delta_roc_auc", "verdict"])
        assert "indistinguishable" in text


def test_a_stale_sibling_directory_cannot_shadow_the_current_table():
    """The report once described a 105-feature ablation from a deleted run.

    ``rglob`` sorted leftover ``_smoke_*`` directories ahead of the real output,
    so the report rendered artifacts that were no longer on disk.  The canonical
    study directory has to win.
    """
    import pandas as pd
    from pathlib import Path
    import tempfile

    from src.experiments.report import _find_table

    stale = pd.DataFrame({"variant": ["full_union"], "n_features": [105]})
    current = pd.DataFrame({"variant": ["full_union"], "n_features": [107]})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "_smoke_ablation").mkdir()
        (root / "ablation").mkdir()
        stale.to_csv(root / "_smoke_ablation" / "ablation_table.csv", index=False)
        current.to_csv(root / "ablation" / "ablation_table.csv", index=False)
        found = _find_table(root, "ablation")
        assert int(found["n_features"].iloc[0]) == 107


def test_a_flat_layout_is_still_found():
    import pandas as pd
    from pathlib import Path
    import tempfile

    from src.experiments.report import _find_table

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pd.DataFrame({"name": ["binary_h6_t50bps"]}).to_csv(
            root / "target_comparison.csv", index=False
        )
        assert not _find_table(root, "targets").empty


def test_a_missing_study_returns_an_empty_frame():
    import pandas as pd
    from pathlib import Path
    import tempfile

    from src.experiments.report import _find_table

    with tempfile.TemporaryDirectory() as tmp:
        assert _find_table(Path(tmp), "targets").empty
        assert _find_table(None, "targets").empty
        assert _find_table(Path(tmp) / "does-not-exist", "targets").empty
