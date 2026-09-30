"""The report must be auditable, not merely generated.

A results document is the one artifact in this project that a reader will trust
without re-running the code, so the failure modes that matter are the ones where
it looks right and says something unsupported.  These tests target exactly those:
a column that silently renders blank, an experiment compared against a different
one, a threshold credited to the wrong split, or causal language smuggled into a
prose section.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.experiments import report as R

ROOT = Path(__file__).resolve().parents[1]


def _backtest(**over) -> dict:
    base = {
        "threshold": 0.44,
        "n_trades": 14,
        "total_return": -0.0874,
        "buy_hold_return": -0.0821,
        "excess_vs_buy_hold": -0.0035,
        "win_rate": 0.2857,
        "profit_factor": 0.5372,
        "max_drawdown": 0.1601,
        "sharpe_ratio": -1.0185,
        "sortino_ratio": -0.1341,
        "calmar_ratio": -0.7569,
        "threshold_selection": {
            "threshold": 0.44,
            "objective": "excess_vs_buy_hold",
            "objective_value": -0.0328,
            "n_trades": 30,
            "stability": {
                "checked": True,
                "picks": {"first_half": 0.38, "second_half": 0.46},
                "agrees": False,
                "tolerance": 0.04,
            },
        },
    }
    base.update(over)
    return base


def _summary(exp_id: str, *, manifest: str, rows: int = 41342, backtest=None) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(
            experiment_id=exp_id,
            name=exp_id,
            hypothesis=f"hypothesis for {exp_id}",
            feature_groups=("technical",),
            slug=exp_id.lower(),
        ),
        metrics={
            "test_metrics": {"roc_auc": 0.6012, "pr_auc": 0.319, "brier_score": 0.21},
        },
        walkforward={"n_windows": 2, "roc_auc": {"mean": 0.5658, "min": 0.5288}},
        regimes={
            "test_metrics_by_regime": {
                "trend_regime": {
                    "bear": {"n": 4492, "metrics": {"roc_auc": 0.5991, "pr_auc": 0.3294}},
                    "bull": {"n": 797, "metrics": {"roc_auc": 0.5574, "pr_auc": 0.2789}},
                }
            }
        },
        backtest=_backtest() if backtest is None else backtest,
        dataset_metadata={
            "rows": rows,
            "n_rows": rows,
            "range_start": "2022-01-09 08:00:00 UTC",
            "range_end": "2026-09-28 00:00:00 UTC",
            "feature_manifest_hash": manifest,
            "split": {},
        },
        artifacts={},
        warnings=["a warning"],
        output_dir=f"reports/experiments/{exp_id}",
        feature_count=39,
        dataset_rows=rows,
        feature_manifest_hash=manifest,
    )


# --------------------------------------------------------------------- headline

def test_headline_reads_test_metrics_not_a_nonexistent_test_block():
    """Regression: the runner stores `test_metrics`; `test` yielded NaN columns."""
    table = R.headline_table([_summary("EXP-00", manifest="a" * 16)])
    row = table.iloc[0]
    assert row["test_roc_auc"] == 0.6012
    assert row["test_pr_auc"] == 0.319
    assert row["rows"] == 41342


def test_headline_reads_walkforward_metric_blocks():
    """Walk-forward metrics arrive as {mean, min, ...} blocks, not flat fields."""
    table = R.headline_table([_summary("EXP-00", manifest="a" * 16)])
    assert table.iloc[0]["wf_roc_auc_mean"] == 0.5658
    assert table.iloc[0]["wf_roc_auc_min"] == 0.5288


def test_headline_surfaces_excess_under_the_backtest_field_name():
    table = R.headline_table([_summary("EXP-00", manifest="a" * 16)])
    assert table.iloc[0]["excess_vs_buy_hold"] == -0.0035


def test_report_has_no_blank_cells_for_present_metrics():
    """A missing value is shown as '-', never as an empty string."""
    table = R.headline_table([_summary("EXP-00", manifest="a" * 16)])
    for column in table.columns:
        for value in table[column]:
            assert value is None or not (isinstance(value, str) and not value.strip())


# ----------------------------------------------------------------- comparability

def test_experiments_with_different_manifests_are_not_directly_comparable():
    summaries = [
        _summary("EXP-00", manifest="a" * 16),
        _summary("EXP-01", manifest="b" * 16),
    ]
    notes = " ".join(R.comparability_notes(summaries))
    assert "paired bootstrap" in notes
    assert "not by subtracting" in notes


def test_identical_manifests_are_reported_as_comparable():
    summaries = [
        _summary("EXP-00", manifest="a" * 16),
        _summary("EXP-01", manifest="a" * 16),
    ]
    notes = " ".join(R.comparability_notes(summaries))
    assert "directly comparable" in notes


def test_all_experiments_sharing_one_manifest_raises_a_suspicion_flag():
    """Different groups must produce different feature lists; if not, something is wrong."""
    summaries = [
        _summary(f"EXP-0{i}", manifest="a" * 16) for i in range(3)
    ]
    notes = " ".join(R.comparability_notes(summaries))
    assert "suspicious" in notes


def test_differing_date_ranges_are_flagged_as_non_comparable():
    a = _summary("EXP-00", manifest="a" * 16)
    b = _summary("EXP-01", manifest="b" * 16, rows=40102)
    b.dataset_metadata["range_start"] = "2022-04-01 08:00:00 UTC"
    notes = " ".join(R.comparability_notes([a, b]))
    assert "not comparable" in notes


# --------------------------------------------------------------------- threshold

def test_threshold_section_names_the_split_the_cut_came_from():
    text = R._threshold_section([_summary("EXP-00", manifest="a" * 16)])
    assert "validation" in text.lower()
    assert "frozen" in text.lower()
    assert "0.44" in text


def test_disagreeing_validation_halves_are_flagged_in_the_threshold_section():
    text = R._threshold_section([_summary("EXP-00", manifest="a" * 16)])
    assert "0.38" in text and "0.46" in text
    assert "unstable" in text.lower()


def test_agreeing_validation_halves_are_not_flagged_unstable():
    s = _summary("EXP-00", manifest="a" * 16)
    s.backtest["threshold_selection"]["stability"] = {
        "picks": {"first_half": 0.44, "second_half": 0.45},
        "agrees": True,
    }
    text = R._threshold_section([s])
    assert "unstable" not in text.lower()


def test_validation_trade_count_is_distinct_from_test_trade_count():
    """Reporting only one of the two hides how much the cut changed the sample."""
    s = _summary("EXP-00", manifest="a" * 16)
    text = R._threshold_section([s])
    assert "validation trades" in text
    assert "test trades" in text


# ------------------------------------------------------------------------ regime

def test_regime_share_is_derived_when_absent_from_the_bundle():
    lines = R._regime_section([_summary("EXP-00", manifest="a" * 16)])
    text = "\n".join(lines)  # the section returns a list of lines
    assert "84.9%" in text  # 4492 / (4492 + 797)
    assert "15.1%" in text  # 797 / 5289
    assert "100.0%" not in text, "shares must not default to 100%"


def test_regime_section_omits_definition_blocks_that_carry_no_measurements():
    """The bundle carries `definitions` and `test_distribution` too; they are
    not metric tables and must not be rendered as if they were."""
    text = R._regime_section([_summary("EXP-00", manifest="a" * 16)])
    assert "definitions (- of test rows)" not in text
    assert "test_distribution (- of test rows)" not in text


# ----------------------------------------------------------------------- report

def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["binance_spot", "fear_greed"],
            "rows": [41550, 41550],
            "covered": [41550, 41550],
            "coverage_pct": [100.0, 100.0],
            "first_covered": ["2022-01-01", "2022-01-01"],
            "last_covered": ["2026-09-28", "2026-09-28"],
        }
    )


def test_report_opens_by_separating_measurement_from_claim():
    text = R.build_report([_summary("EXP-00", manifest="a" * 16)], _availability())
    assert "measurement" in text.lower()
    assert "does not establish causation" in text.lower() or "none of it establishes causation" in text.lower()
    assert "paired" in text.lower() and "bootstrap" in text.lower()


def test_report_lists_availability_before_results():
    text = R.build_report([_summary("EXP-00", manifest="a" * 16)], _availability())
    assert text.index("## Data availability") < text.index("## Headline results")


def test_report_states_the_derived_group_hypotheses():
    summaries = [_summary("EXP-00", manifest="a" * 16), _summary("EXP-01", manifest="b" * 16)]
    text = R.build_report(summaries, _availability())
    assert "Hypotheses under test" in text
    assert "hypothesis for EXP-01" in text


def test_report_includes_warnings_verbatim():
    text = R.build_report([_summary("EXP-00", manifest="a" * 16)], _availability())
    assert "a warning" in text


def test_report_carries_interpretation_limits():
    text = R.build_report([_summary("EXP-00", manifest="a" * 16)], _availability())
    assert "Interpretation limits" in text
    assert "one symbol" in text.lower()


def test_causal_language_audit_catches_overreach():
    assert R.audit_language("This feature causes higher returns.") == ["causes"]
    assert R.audit_language("The result is associated with a stronger trend.") == []


def test_generated_report_passes_its_own_language_audit():
    text = R.build_report(
        [_summary("EXP-00", manifest="a" * 16), _summary("EXP-01", manifest="b" * 16)],
        _availability(),
    )
    assert R.audit_language(text) == []


def test_write_reports_raises_if_generated_text_overreaches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        R, "build_report", lambda *a, **k: "The model causes the returns to rise."
    )
    with pytest.raises(AssertionError, match="causal language"):
        R.write_reports(
            out=tmp_path,
            availability_path=tmp_path / "missing.csv",
            summaries=[_summary("EXP-00", manifest="a" * 16)],
            report_path=tmp_path / "r.md",
        )


def test_write_reports_emits_index_and_report(tmp_path):
    availability = tmp_path / "avail.csv"
    _availability().to_csv(availability, index=False)
    R.write_reports(
        out=tmp_path,
        availability_path=availability,
        summaries=[_summary("EXP-00", manifest="a" * 16)],
        report_path=tmp_path / "experiment_report.md",
    )
    assert (tmp_path / "experiment_report.md").exists()
    index = json.loads((tmp_path / "experiment_index.json").read_text())
    assert index[0]["experiment_id"] == "EXP-00"
    assert index[0]["feature_manifest_hash"] == "a" * 16


# ----------------------------------------------------------------- feature hash

def test_feature_manifest_hash_ignores_values_but_tracks_names_and_order():
    from src.dataset.factory import feature_manifest_hash

    a = feature_manifest_hash(["x", "y"])
    b = feature_manifest_hash(["y", "x"])
    assert a == feature_manifest_hash(["x", "y"])
    assert a != b, "order must change the hash: column order changes the model"


def test_manifest_hash_is_exposed_on_the_experiment_result():
    from src.experiments.runner import ExperimentResult

    for name in ("feature_count", "dataset_rows", "feature_manifest_hash"):
        assert isinstance(getattr(ExperimentResult, name), property)
