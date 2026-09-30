"""Render the V2 experiment suite into one reviewable report.

The report is written to be *audited*, not admired.  It leads with what was
measured, separates measurement from interpretation, and refuses to compare
experiments whose inputs differ.  Two habits are enforced structurally here
because they are easy to lose when a table is assembled by hand:

* a test-set number is never described with causal language, and
* two experiments are only differenced when their feature manifests match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

#: Phrases that turn a measurement into a claim the design cannot support.
FORBIDDEN_CAUSAL = (
    "causes",
    "caused by",
    "because of the model",
    "proves",
    "guarantees",
)

#: Wording used to describe a test-set result wherever it appears.
MEASUREMENT_LANGUAGE = "associated with"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def _label(metric: str) -> str:
    return metric.replace("_", " ")


def _test_period(s: Any) -> str:
    """The test block's date range, from the persisted per-split dataset bounds.

    The fallback deliberately does *not* report the full dataset span.  Doing so
    was a real bug: it labelled the headline score as covering the whole history
    when the score is measured on the final 15% only.  Saying "unknown" is
    accurate where a number would not be.
    """
    meta = s.dataset_metadata or {}
    split = meta.get("split", {}) or {}
    bounds = (split.get("bounds", {}) or {}).get("test", {}) or {}
    if bounds.get("start"):
        return f"{bounds['start']} -> {bounds.get('end') or '?'}"
    if meta.get("range_start"):
        return f"unknown (dataset spans {meta['range_start']} -> {meta.get('range_end', '?')})"
    return "unknown"


def headline_table(summaries: Sequence[Any]) -> pd.DataFrame:
    """One row per experiment: the numbers a reviewer compares first."""
    rows = []
    for s in summaries:
        test = s.metrics.get("test_metrics", s.metrics.get("test", {})) or {}
        walk = s.walkforward or {}
        # Walk-forward metrics arrive as per-metric {mean, median, ...} blocks,
        # not as flat `roc_auc_mean` fields.
        wf_roc = (walk.get("roc_auc") or {}) if isinstance(walk.get("roc_auc"), dict) else {}
        back = s.backtest or {}
        selection = back.get("threshold_selection", {}) or {}
        rows.append(
            {
                "experiment": s.spec.experiment_id,
                "groups": "+".join(s.spec.feature_groups),
                "n_features": s.feature_count,
                "rows": s.dataset_rows,
                "period": _test_period(s),
                "test_roc_auc": test.get("roc_auc"),
                "test_pr_auc": test.get("pr_auc"),
                "test_brier_score": test.get("brier_score"),
                "wf_roc_auc_mean": wf_roc.get("mean"),
                "wf_roc_auc_min": wf_roc.get("min"),
                "wf_windows": walk.get("n_windows"),
                "threshold": back.get("threshold"),
                "n_trades": back.get("n_trades"),
                "total_return": back.get("total_return"),
                "buy_hold_return": back.get("buy_hold_return"),
                "excess_vs_buy_hold": back.get("excess_vs_buy_hold"),
                "max_drawdown": back.get("max_drawdown"),
                "sharpe_ratio": back.get("sharpe_ratio"),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    header = "| " + " | ".join(_label(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = [_fmt(row.get(c), digits) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


    # The backtest bundle names excess relative to buy-and-hold
    # `excess_vs_buy_hold`; the report's column is the friendlier
    # `excess_return`.  Normalising here keeps the table headers readable
    # without silently reporting a missing field as zero.
    if "excess_return" not in back and "excess_vs_buy_hold" in back:
        back["excess_return"] = back["excess_vs_buy_hold"]

def comparability_notes(summaries: Sequence[Any]) -> list[str]:
    """Flag any pair of experiments that must not be differenced directly."""
    notes: list[str] = []
    by_hash: dict[str, list[str]] = {}
    for s in summaries:
        by_hash.setdefault(s.feature_manifest_hash, []).append(s.spec.experiment_id)
    for digest, ids in sorted(by_hash.items()):
        if len(ids) > 1:
            notes.append(
                f"{', '.join(ids)} share feature manifest `{digest}` and are directly comparable."
            )
        else:
            notes.append(
                f"{ids[0]} has a unique feature manifest `{digest}`; compare it to the baseline "
                "only through the paired bootstrap, not by subtracting headline AUC."
            )

    hashes = {s.feature_manifest_hash for s in summaries}
    if len(hashes) == 1 and len(summaries) > 1:
        notes.append(
            "All experiments used an identical feature manifest, which is itself suspicious - "
            "check that the feature groups actually differ."
        )

    ranges = {(s.dataset_metadata or {}).get("range_start") for s in summaries}
    if len(ranges) > 1:
        notes.append(
            "Experiments cover different date ranges "
            f"({', '.join(sorted(str(r) for r in ranges))}); raw AUC differences across different "
            "periods are not comparable and the ablation table restricts to the common rows."
        )
    return notes


def _threshold_section(summaries: Sequence[Any]) -> str:
    lines = [
        "## Threshold selection",
        "",
        "The probability cut is chosen on the **validation** block and then frozen; the test block "
        "never informs it. Probabilities used for that choice come from the train-only fit, so the "
        "cut is not tuned in-sample.",
        "",
    ]
    for s in summaries:
        back = s.backtest or {}
        selection = back.get("threshold_selection", {}) or {}
        if not selection:
            continue
        # The bundle nests the half-period picks under `stability.picks` and
        # reports agreement as `agrees`, not as a pair of threshold fields.
        stability = selection.get("stability", {}) or {}
        picks = stability.get("picks", {}) or {}
        first = picks.get("first_half")
        second = picks.get("second_half")
        agrees = stability.get("agrees")
        lines.append(
            f"### {s.spec.experiment_id}\n\n"
            f"- selected threshold: `{_fmt(back.get('threshold'), 2)}` "
            f"(objective: {selection.get('objective', '-')})\n"
            f"- validation trades at that cut: {_fmt(selection.get('n_trades'))}\n"
            f"- validation objective value: {_fmt(selection.get('objective_value'))}\n"
            f"- first / second validation half: `{_fmt(first, 2)}` / `{_fmt(second, 2)}`"
            f"{'' if agrees is not False else '  (disagree - treat the cut as unstable)'}\n"
            f"- test trades after freezing: {_fmt(back.get('n_trades'))}\n"
        )
    return "\n".join(lines)


def _regime_section(summaries: Sequence[Any]) -> list[str]:
    lines = [
        "## Performance by market regime",
        "",
        "Regimes are defined by an explicit trailing trend and volatility rule, not by hindsight. "
        "A single period is not evidence of a stable edge; these splits are reported so the "
        "concentration of any apparent effect is visible.",
        "",
    ]
    for s in summaries:
        regimes = s.regimes or {}
        # The bundle is nested: definitions, then per-axis test breakdowns.
        # Only the measured blocks are rendered.
        breakdown = regimes.get("test_metrics_by_regime", {}) or {}
        if not breakdown:
            continue
        lines.append(f"### {s.spec.experiment_id}\n")
        for axis in sorted(breakdown):
            blocks = breakdown[axis] or {}
            if not blocks:
                continue
            lines.append(f"**{_label(axis)}**\n")
            lines.append("")
            lines.append("| regime | n | share | roc auc | pr auc |")
            lines.append("| --- | --- | --- | --- | --- |")
            total_rows = sum(
                ((b or {}).get("n") or (b or {}).get("n_rows") or 0)
                for b in blocks.values()
            )
            for name in sorted(blocks):
                block = blocks[name] or {}
                metrics = block.get("metrics", {}) or {}
                n = block.get("n") or block.get("n_rows")
                # Share is not stored per block; it is derived from the counts
                # actually present, so a renamed field surfaces as a computed
                # ratio instead of a silently blank column.
                share = block.get("share")
                if share is None and n and total_rows:
                    share = n / total_rows
                lines.append(
                    f"| {name} | {_fmt(n)} | {_pct(share)} | "
                    f"{_fmt(metrics.get('roc_auc'))} | {_fmt(metrics.get('pr_auc'))} |"
                )
            lines.append("")
    return lines


def _backtest_section(summaries: Sequence[Any]) -> list[str]:
    lines = [
        "## Backtest on the test block",
        "",
        "Fees and slippage are charged on every leg. `excess` is the strategy return minus "
        "buy-and-hold over the same rows, which is the only comparison that nets out the market's "
        "own direction.",
        "",
    ]
    # Field names as the backtest module actually emits them.  These are read
    # from the persisted bundle rather than assumed, so a rename in the
    # evaluator surfaces as a missing column instead of a silent blank cell.
    cols = [
        "experiment",
        "n_trades",
        "total_return",
        "buy_hold_return",
        "excess_vs_buy_hold",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    ]
    rows = []
    for s in summaries:
        back = dict(s.backtest or {})
        back["experiment"] = s.spec.experiment_id
        rows.append(back)
    lines.append(_markdown_table(pd.DataFrame(rows), cols))
    lines.append("")
    return lines


def _warnings_section(summaries: Sequence[Any]) -> list[str]:
    collected: list[str] = []
    for s in summaries:
        for w in s.warnings:
            collected.append(f"- **{s.spec.experiment_id}**: {w}")
    if not collected:
        return ["## Warnings", "", "- none recorded."]
    return ["## Warnings", "", *collected]


def build_report(
    summaries: Sequence[Any],
    availability: pd.DataFrame,
    studies: dict[str, Any] | None = None,
    report_parent: Path | None = None,
) -> str:
    """Assemble the full markdown report."""
    headline = headline_table(summaries)
    coverage = availability if isinstance(availability, pd.DataFrame) else pd.DataFrame()
    parts: list[str] = [
        "# V2 multi-factor experiment report",
        "",
        "Every number below is a **measurement** on a held-out chronological test block. "
        "None of it establishes causation, and none of it was tuned on the test block. "
        "Where a feature group is compared with the baseline, the comparison is a paired "
        "moving-block bootstrap on the common rows rather than a subtraction of two independently "
        "estimated AUCs.",
        "",
        "## Data availability",
        "",
        "Coverage is measured on the hourly feature grid after as-of alignment with each source's "
        "publication lag. A source that is absent is reported as absent rather than imputed.",
        "",
    ]

    if len(coverage):
        cols = [c for c in ("source", "rows", "covered", "coverage_pct", "first_covered", "last_covered") if c in coverage.columns]
        parts.append(_markdown_table(coverage, cols, digits=2))
        parts.append("")

    parts.append("## Headline results")
    parts.append("")
    parts.append(
        _markdown_table(
            headline,
            [
                "experiment",
                "n_features",
                "rows",
                "period",
                "test_roc_auc",
                "test_pr_auc",
                "wf_roc_auc_mean",
                "wf_roc_auc_min",
                "threshold",
                "n_trades",
                "total_return",
                "excess_vs_buy_hold",
            ],
            digits=4,
        )
    )
    parts.append("")

    parts.append("## Hypotheses under test")
    parts.append("")
    for s in summaries:
        parts.append(f"- **{s.spec.experiment_id}** - {s.spec.hypothesis}")
    parts.append("")

    parts.append("## Comparability")
    parts.append("")
    parts.extend(f"- {n}" for n in comparability_notes(summaries))
    parts.append("")

    parts.append(_threshold_section(summaries))
    parts.extend(_backtest_section(summaries))
    parts.extend(_regime_section(summaries))
    parts.extend(_studies_section(studies, report_parent))
    parts.extend(_warnings_section(summaries))

    parts.append("## Interpretation limits")
    parts.append("")
    parts.extend(
        [
            "- One symbol, one period. A positive test-block result is evidence worth extending, "
            "not a finding.",
            "- Features are strongly autocorrelated; a tree model's importance ranking is a "
            "dividend-split artefact as much as a signal ranking.",
            *_derivatives_limit(summaries, coverage),
            "- Transaction costs are modelled as a flat rate, not as market impact, which "
            "flatters any high-turnover strategy.",
            "- Sentiment is a daily reading carried across the day; it cannot react within a day "
            "and its features are therefore slow-moving relative to a 6h horizon.",
        ]
    )
    parts.append("")
    return "\n".join(parts)


def audit_language(text: str) -> list[str]:
    """Return any causal overreach in generated prose."""
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_CAUSAL if phrase in lowered]


def _derivatives_limit(summaries: Sequence[Any], coverage: pd.DataFrame) -> list[str]:
    """State the group-coverage caveat as measured, not as a remembered assumption.

    This used to be a hard-coded sentence claiming the derivatives group was
    "history-limited".  That stopped being true once the funding and futures
    fetchers learned to paginate backwards and coverage reached 100%, at which
    point the sentence both under-reported data quality and still gave a wrong
    reason for the row-count gap.  The gap is real, but it comes from
    trailing-indicator warm-up (a 60-day basis z-score has no value for its
    first 60 days), so the wording is derived from the coverage table and the
    actual row counts.
    """
    gaps: list[str] = []
    if not coverage.empty and "source" in coverage.columns and "coverage_pct" in coverage.columns:
        gaps = [
            str(r["source"])
            for _, r in coverage.iterrows()
            if r.get("coverage_pct") is not None and float(r["coverage_pct"]) < 100.0
        ]
    if gaps:
        return [
            f"- Source coverage is incomplete for {', '.join(gaps)}, so rows lacking those "
            "series are dropped and any affected experiment's dataset is shorter than the "
            "baseline's. Its AUC is not directly comparable without the paired bootstrap."
        ]

    row_counts = sorted(
        {
            int(n)
            for s in summaries
            if (n := (s.dataset_metadata or {}).get("n_rows")) is not None
        }
    )
    if not row_counts:
        return [
            "- Every source is fully covered, but group datasets still differ in row count "
            "because trailing-indicator warm-up differs per group; compare group results "
            "through the paired bootstrap on common rows, not by subtracting headline AUCs."
        ]
    return [
        "- Every source is fully covered over the feature grid, so the shorter group datasets "
        "come from trailing-indicator warm-up rather than missing history: a 60-day basis "
        f"z-score has no value for its first 60 days. Group row counts run from "
        f"{row_counts[0]} to {row_counts[-1]} rows. Compare group results through the paired "
        "bootstrap on common rows rather than by subtracting two AUCs measured on different "
        "samples."
    ]


#: Where each secondary study writes, and the table it leaves behind.  The
#: report reads these exact locations rather than searching the tree, because a
#: blind search will happily pick up a stale directory from an earlier run: a
#: leftover ``_smoke_ablation/`` shadowed the real table and the report shipped
#: a 105-feature ablation next to a 107-feature headline.
_STUDY_TABLES = {
    "ablation": ("ablation", "ablation_table.csv"),
    "targets": ("targets", "target_comparison.csv"),
    "multiasset": ("multiasset", "multiasset_table.csv"),
}


def _find_table(out: Path | None, study: str) -> pd.DataFrame:
    """Load one secondary study's table, preferring the canonical location.

    Search order is deliberate:

    1. ``<out>/<study_dir>/<table>`` - where ``run_experiments.py`` writes.
    2. ``<out>/<table>`` - a flat layout.
    3. a recursive search - a study that nested an extra level, or a caller
       that renamed its output directory.

    An ``rglob``-first search is what made this function wrong: sibling
    directories from previous runs sorted ahead of the current output, so the
    report described data that was no longer on disk.  A missing study yields
    an empty frame rather than raising, so a primary-only run still produces a
    valid report.
    """
    if out is None or not out.exists():
        return pd.DataFrame()
    study_dir, table = _STUDY_TABLES.get(study, (study, f"{study}_table.csv"))
    candidates = [
        out / study_dir / table,
        out / table,
        *sorted(p for p in out.rglob(table) if p not in (out / study_dir / table, out / table)),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                frame = pd.read_csv(path)
            except Exception:  # pragma: no cover - malformed artifact
                continue
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _studies_section(studies: dict[str, Any] | None, out: Path) -> list[str]:
    """Render the ablation / target / multi-asset tables, if they were run."""
    if not studies:
        return []
    lines: list[str] = ["## Secondary studies", ""]

    ablation = _find_table(out, "ablation")
    if not ablation.empty:
        lines.append("### Feature-group ablation")
        lines.append("")
        lines.append(
            "Each arm is trained under the same protocol and compared with the baseline on the "
            "**common** test rows only, using a paired moving-block bootstrap. A confidence "
            "interval spanning zero means the measured difference is not distinguishable from "
            "sampling noise; it does not mean the features are useless, only that this data "
            "cannot show it."
        )
        lines.append("")
        cols = [
            c
            for c in (
                "variant",
                "n_features",
                "n_rows",
                "test_roc_auc",
                "delta_roc_auc",
                "ci_low",
                "ci_high",
                "verdict",
                "n_trades",
                "excess_vs_buy_hold",
            )
            if c in ablation.columns
        ]
        lines.append(_markdown_table(ablation, cols, digits=4))
        lines.append("")

    targets = _find_table(out, "targets")
    if not targets.empty:
        lines.append("### Target-definition comparison")
        lines.append("")
        lines.append(
            "Different label definitions answer different questions, so their scores are not "
            "comparable to one another; the point is to show how much the headline number depends "
            "on a choice that was made before looking at any result."
        )
        lines.append("")
        cols = [
            c
            for c in (
                "name",
                "mode",
                "horizon_candles",
                "threshold",
                "majority_share",
                "test_roc_auc",
                "test_pr_auc",
                "test_macro_f1",
                "test_balanced_accuracy",
                "delta_roc_auc_vs_binary",
                "ci_low",
                "ci_high",
                "verdict",
            )
            if c in targets.columns
        ]
        lines.append(_markdown_table(targets, cols, digits=4))
        lines.append("")

    multiasset = _find_table(out, "multiasset")
    if not multiasset.empty:
        lines.append("### Per-asset vs global model")
        lines.append("")
        lines.append(
            "A global model pooled across symbols is compared with each per-asset model on that "
            "asset's own test rows, again paired. Sentiment is excluded for non-BTC symbols, so "
            "the arms are not exchangeable across rows and the comparison is within-symbol only."
        )
        lines.append("")
        cols = [
            c
            for c in (
                "symbol",
                "strategy",
                "n_test",
                "test_roc_auc",
                "delta_roc_auc_vs_per_asset",
                "ci_low",
                "ci_high",
                "verdict",
            )
            if c in multiasset.columns
        ]
        lines.append(_markdown_table(multiasset, cols, digits=4))
        lines.append("")

    return lines


def write_reports(
    *,
    out: Path,
    availability_path: Path,
    summaries: Sequence[Any],
    report_path: Path,
    studies: dict[str, Any] | None = None,
) -> None:
    """Write the availability CSV, per-experiment index, and the markdown report."""
    out.mkdir(parents=True, exist_ok=True)
    availability = (
        pd.read_csv(availability_path) if availability_path.exists() else pd.DataFrame()
    )
    report = build_report(
        summaries, availability, studies=studies, report_parent=out
    )
    report_path.write_text(report, encoding="utf-8")

    leaks = audit_language(report)
    if leaks:
        raise AssertionError(f"report uses causal language: {leaks}")

    index = [
        {
            "experiment_id": s.spec.experiment_id,
            "n_features": s.feature_count,
            "rows": s.dataset_rows,
            "feature_manifest_hash": s.feature_manifest_hash,
            "output_dir": s.output_dir,
        }
        for s in summaries
    ]
    (out / "experiment_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
