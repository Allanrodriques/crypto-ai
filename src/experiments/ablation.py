"""Feature-ablation study: does each group earn its place?

Three questions, three different tables
---------------------------------------
``add``
    Baseline + one group.  Answers "would this data have helped at all?"
``drop``
    Full union minus one group.  Answers "is the group still pulling weight
    once everything else is present?"  This is the question that matters, and it
    is not the same as ``add``: a group can look useless when added alone because
    another group already contains the same information.
``leave-one-out (feature)``
    Union minus a single feature.  Pinpoints redundancy, but costs one full
    model fit per feature, so it is opt-in and bounded.

Every variant is scored on the *same* test rows, and each variant's test
ROC-AUC is bootstrapped against the baseline with a paired block bootstrap.  A
delta whose 95% interval spans zero is reported as indistinguishable from noise
rather than as an improvement - which, on a single 2-year test period, is the
honest reading for most variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.dataset.factory import build_dataset
from src.evaluation.bootstrap import paired_block_bootstrap_auc
from src.evaluation.metrics import classification_metrics, positive_class_proba
from src.evaluation.walkforward import WalkForwardPlan, run_walk_forward, summarise_windows
from src.experiments.spec import ExperimentSpec
from src.experiments.threshold import select_threshold
from src.models import build_model
from src.pipeline.train import train_models
from src.utils import get_logger, save_json, set_global_seed

logger = get_logger("experiments.ablation")

BASELINE_GROUPS = ("technical",)
DEFAULT_FOLD = ("derivatives", "sentiment", "microstructure", "context")


@dataclass
class AblationVariant:
    """One row of the ablation table."""

    variant: str
    kind: str
    feature_groups: tuple[str, ...]
    removed_feature: str | None = None
    n_features: int = 0
    n_rows: int = 0
    validation_roc_auc: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    delta_roc_auc: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    probability_better: float | None = None
    significant: bool = False
    #: One-sided flags in both directions, so a *negative* interval is not
    #: lumped in with an interval that straddles zero.
    significantly_worse: bool = False
    verdict: str = "not_compared"
    walkforward_roc_auc_mean: float | None = None
    n_trades: int | None = None
    total_return: float | None = None
    excess_vs_buy_hold: float | None = None
    threshold: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "kind": self.kind,
            "feature_groups": list(self.feature_groups),
            "removed_feature": self.removed_feature,
            "n_features": self.n_features,
            "n_rows": self.n_rows,
            "validation_roc_auc": self.validation_roc_auc,
            "test_roc_auc": self.test_roc_auc,
            "test_pr_auc": self.test_pr_auc,
            "delta_roc_auc": self.delta_roc_auc,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "probability_better": self.probability_better,
            "significant_at_95": self.significant,
            "significantly_worse_at_95": self.significantly_worse,
            "verdict": self.verdict,
            "walkforward_roc_auc_mean": self.walkforward_roc_auc_mean,
            "n_trades": self.n_trades,
            "total_return": self.total_return,
            "excess_vs_buy_hold": self.excess_vs_buy_hold,
            "threshold": self.threshold,
            "notes": self.notes,
        }


def _fit_and_score(
    spec: ExperimentSpec,
    config,
    context,
    feature_groups: Sequence[str],
    enabled_features: dict[str, list[str]] | None,
    *,
    walkforward: WalkForwardPlan | None,
    with_backtest: bool,
    threshold_min_trades: int,
):
    """Fit one variant and return everything the table needs."""
    seed = int(getattr(config, "random_state", 42))
    set_global_seed(seed)

    matrix = context.features(feature_groups, enabled_features=enabled_features)
    dataset = build_dataset(
        matrix,
        context.spot,
        spec.target,
        spec.split,
        symbol=context.symbol,
        interval=context.interval,
        feature_groups=list(feature_groups),
        data_sources=list(spec.data_sources),
    )
    if not len(dataset.frame):
        raise ValueError(f"empty dataset for groups {feature_groups}")

    training = train_models(config, dataset, model_names=[spec.model_name], run_cv=False)
    selected = training.selected

    test = dataset.splits["test"]
    proba = _proba(selected.estimator, test.X)
    test_metrics = classification_metrics(test.y, positive_class_proba(proba))

    wf_mean = None
    if walkforward is not None:
        X_pool = pd.concat([dataset.splits["train"].X, dataset.splits["validation"].X])
        y_pool = pd.concat([dataset.splits["train"].y, dataset.splits["validation"].y])
        windows = run_walk_forward(
            X_pool,
            y_pool,
            lambda: build_model(spec.model_name, config, dataset.feature_columns),
            walkforward,
            metric_fn=lambda yt, yp: {
                k: classification_metrics(yt, positive_class_proba(yp)).get(k)
                for k in ("roc_auc", "pr_auc")
            },
        )
        summary = summarise_windows(windows)
        wf_mean = (summary.get("roc_auc") or {}).get("mean")

    backtest = None
    threshold = None
    if with_backtest:
        from src.experiments.runner import _backtest_rules

        rules = _backtest_rules(config, dataset.horizon_candles, float(spec.backtest_threshold))
        validation = dataset.splits["validation"]
        validation_frame = validation.frame.copy()
        # Pre-refit (train-only) validation probabilities.  Scoring the refit
        # estimator here would make this an in-sample threshold fit, and every
        # ablation arm would inherit the same optimism.
        if selected.validation_probabilities is None:
            raise RuntimeError(
                f"model {selected.name!r} has no pre-refit validation probabilities; "
                "refusing to select a threshold on in-sample validation scores"
            )
        validation_frame["probability_up"] = positive_class_proba(
            np.asarray(selected.validation_probabilities)
        )
        selection = select_threshold(
            validation_frame, rules, min_trades=threshold_min_trades
        )
        threshold = selection.threshold
        from src.evaluation.backtest import BacktestRules, run_backtest

        final = BacktestRules(**{**rules.__dict__, "probability_threshold": threshold}).validate()
        test_frame = test.frame.copy()
        test_frame["probability_up"] = positive_class_proba(proba)
        backtest = run_backtest(test_frame, final).metrics

    return {
        "dataset": dataset,
        "proba": positive_class_proba(proba),
        "y": test.y,
        "test_metrics": test_metrics,
        "validation_roc_auc": selected.validation_metrics.get("roc_auc"),
        "walkforward_roc_auc_mean": wf_mean,
        "backtest": backtest,
        "threshold": threshold,
    }


def _common_test_index(results: Sequence[dict[str, Any]]) -> pd.Index:
    """The test rows every variant scored.

    This matters more than it looks.  Adding a group with a long warm-up
    (sentiment needs 30 days of history) shifts the chronological split
    boundaries, so two variants can end up scored over *different periods*.  A
    raw AUC comparison between them then measures the market's mood as much as
    the features, and would happily report a "win" for whichever variant happened
    to be tested against a trending stretch.

    Scoring every variant on the shared rows removes the period confound.  The
    cost is that the comparison is restricted to the rows all variants can
    supply, which is stated alongside the results.
    """
    indexes = [pd.Index(r["y"].index) for r in results if r.get("y") is not None]
    if not indexes:
        return pd.Index([])
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    return common


def _proba(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))
    raw = np.asarray(estimator.decision_function(X))
    return np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw


def run_ablation(
    spec: ExperimentSpec,
    config,
    *,
    context,
    output_dir: str | Path,
    baseline_groups: Sequence[str] = BASELINE_GROUPS,
    groups_under_test: Sequence[str] = DEFAULT_FOLD,
    walkforward: WalkForwardPlan | None = None,
    with_backtest: bool = True,
    threshold_min_trades: int = 30,
    n_resamples: int = 500,
    block_size: int = 24,
) -> dict[str, Any]:
    """Run the add / drop / union ablation and write the comparison table."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    available = [g for g in groups_under_test]
    full_groups = tuple(dict.fromkeys(tuple(baseline_groups) + tuple(available)))

    # ---- phase 1: fit every variant ---------------------------------------
    # Variants are all fitted first, then compared on a common row set, because
    # the comparison cannot be made until every variant's predictions exist.
    plans: list[tuple[str, str, tuple[str, ...], dict[str, list[str]] | None, str | None]] = [
        ("baseline", "baseline", tuple(baseline_groups), None, None)
    ]
    for group in available:
        if group in full_groups:
            plans.append((f"add_{group}", "add", tuple(baseline_groups) + (group,), None, None))
    for group in available:
        remaining = tuple(g for g in full_groups if g != group)
        if remaining and remaining != tuple(baseline_groups):
            plans.append((f"drop_{group}", "drop", remaining, None, None))
    plans.append(("full_union", "union", full_groups, None, None))

    fits: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for name, kind, groups, enabled, removed in plans:
        logger.info("ablation: fitting %s (%s) groups=%s", name, kind, list(groups))
        try:
            fits[name] = _fit_and_score(
                spec, config, context, groups, enabled,
                walkforward=walkforward, with_backtest=with_backtest,
                threshold_min_trades=threshold_min_trades,
            )
            fits[name]["kind"] = kind
            fits[name]["groups"] = groups
            fits[name]["removed"] = removed
        except Exception as exc:
            logger.warning("ablation %s failed: %s", name, exc)
            failures[name] = str(exc)

    if "baseline" not in fits:
        raise RuntimeError(f"baseline variant failed to fit: {failures.get('baseline')}")

    # ---- phase 2: compare on the rows every variant shares ------------------
    common = _common_test_index(list(fits.values()))
    base_common = fits["baseline"]["y"].loc[common]
    base_proba_common = pd.Series(
        fits["baseline"]["proba"], index=fits["baseline"]["y"].index
    ).loc[common].to_numpy()

    rows: list[AblationVariant] = []
    for name, kind, groups, _enabled, removed in plans:
        if name not in fits:
            rows.append(AblationVariant(
                variant=name, kind=kind, feature_groups=groups,
                removed_feature=removed, notes=[f"failed: {failures[name]}"],
            ))
            continue

        got = fits[name]
        y_common = got["y"].loc[common]
        proba_common = pd.Series(got["proba"], index=got["y"].index).loc[common].to_numpy()
        metrics_common = classification_metrics(y_common, proba_common)
        boot = paired_block_bootstrap_auc(
            y_common, proba_common, base_proba_common,
            n_resamples=n_resamples, block_size=block_size, labels=(name, "baseline"),
        )
        bt = got["backtest"] or {}
        notes: list[str] = []
        if len(common) < len(got["y"]):
            notes.append(
                f"scored on {len(common)} of {len(got['y'])} own test rows "
                f"(the common set across all variants)"
            )
        # The baseline is compared against itself, which yields a degenerate
        # zero-width interval.  Labelling it "indistinguishable" would read as
        # though the baseline had been tested against something.
        verdict = "reference" if kind == "baseline" else boot.verdict
        rows.append(AblationVariant(
            variant=name, kind=kind, feature_groups=groups, removed_feature=removed,
            n_features=len(got["dataset"].feature_columns),
            n_rows=len(got["dataset"].frame),
            validation_roc_auc=got["validation_roc_auc"],
            test_roc_auc=metrics_common.get("roc_auc"),
            test_pr_auc=metrics_common.get("pr_auc"),
            delta_roc_auc=boot.point_delta,
            ci_low=boot.ci_low, ci_high=boot.ci_high,
            probability_better=boot.probability_a_better,
            significant=boot.significant,
            significantly_worse=boot.significantly_worse,
            verdict=verdict,
            walkforward_roc_auc_mean=got["walkforward_roc_auc_mean"],
            n_trades=bt.get("n_trades"), total_return=bt.get("total_return"),
            excess_vs_buy_hold=bt.get("excess_vs_buy_hold"), threshold=got["threshold"],
            notes=notes,
        ))

    table = pd.DataFrame([r.to_dict() for r in rows])
    table.to_csv(out / "ablation_table.csv", index=False)
    save_json(
        {
            "baseline_groups": list(baseline_groups),
            "groups_under_test": list(available),
            "full_groups": list(full_groups),
            "common_test_rows": int(len(common)),
            "common_test_period": [str(common[0]), str(common[-1])] if len(common) else None,
            "common_test_note": (
                "Every variant is scored on the same test rows, because a group with a "
                "long warm-up shifts the chronological split and would otherwise give each "
                "variant a different evaluation period."
            ),
            "bootstrap": {
                "method": "paired moving-block bootstrap on the common test rows",
                "n_resamples": n_resamples,
                "block_size_rows": block_size,
            },
            "failed_variants": failures,
            "interpretation": _interpretation(table),
            "variants": [r.to_dict() for r in rows],
        },
        out / "ablation_summary.json",
    )
    logger.info("ablation: %d variants written to %s", len(rows), out / "ablation_table.csv")
    return {"table": table, "variants": rows, "baseline": rows[0]}


def _interpretation(table: pd.DataFrame) -> dict[str, Any]:
    """Plain-language reading of the table, kept separate from the numbers."""
    non_base = table[table["kind"] != "baseline"]
    if non_base.empty:
        return {"summary": "no variants completed"}
    winners = non_base[non_base["significant_at_95"].fillna(False)]
    losers = non_base[non_base["significantly_worse_at_95"].fillna(False)]
    # Anything that is neither directionally significant is indistinguishable
    # from the baseline.  This must exclude the significantly-worse arms: they
    # are a *finding*, not an absence of one, and folding them into
    # "indistinguishable" would understate evidence against a group.
    flat = non_base[
        (~non_base["significant_at_95"].fillna(False))
        & (~non_base["significantly_worse_at_95"].fillna(False))
        & (non_base["delta_roc_auc"].notna())
    ]
    return {
        "variants_tested": int(len(non_base)),
        "variants_significantly_better": winners["variant"].tolist(),
        "variants_significantly_worse": losers["variant"].tolist(),
        "variants_indistinguishable": flat["variant"].tolist(),
        "summary": (
            f"{len(winners)} of {len(non_base)} variants beat the baseline by more than "
            f"the block-bootstrap noise on this single test period; {len(losers)} were "
            f"significantly worse and {len(flat)} could not be distinguished from the "
            f"baseline either way. A variant that is not significant is not evidence "
            f"that the group is useless; it is evidence that this test period cannot "
            f"distinguish it from the baseline."
        ),
    }
