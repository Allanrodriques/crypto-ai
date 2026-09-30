"""Guards against validation-set leakage into the frozen decision threshold.

The threshold is the one place where a subtle refit-order mistake quietly
inflates every headline number.  ``train_models`` deliberately captures
validation probabilities from the *train-only* fit and then refits the chosen
model on train+validation.  Any consumer that re-scores the refit estimator on
the validation block has produced an in-sample fit, and the threshold it picks
is tuned on data the model has already memorised.

These tests are structural rather than statistical: they assert *where the
probabilities came from*, because a statistical test would need a seed and a
dataset to detect an optimism that is real but small.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import ablation, runner
from src.pipeline.train import TrainedModel

ROOT = Path(__file__).resolve().parents[1]


def _reads(nm: str) -> str:
    return inspect.getsource(getattr(runner, nm)) + inspect.getsource(getattr(ablation, nm))


def test_threshold_selection_never_scores_the_refit_estimator_on_validation():
    """Both entry points must use pre-refit validation probabilities."""
    for label, source in (
        ("runner.run_experiment", inspect.getsource(runner.run_experiment)),
        ("ablation._fit_and_score", inspect.getsource(ablation._fit_and_score)),
    ):
        # A forbidden call: predict on the validation matrix with the refit
        # estimator, then feed that to select_threshold.
        assert "_predict_proba(estimator, validation_split.X)" not in source, label
        assert "_proba(selected.estimator, validation.X)" not in source, label
        # The required one.
        assert "validation_probabilities" in source, label


def test_threshold_call_is_preceded_by_a_pre_refit_provenance_check():
    """A missing pre-refit array must raise, never silently fall back."""
    for label, source in (
        ("runner.run_experiment", inspect.getsource(runner.run_experiment)),
        ("ablation._fit_and_score", inspect.getsource(ablation._fit_and_score)),
    ):
        guard = source.find("validation_probabilities is None")
        assert guard != -1, f"{label} has no provenance guard"
        assert "raise RuntimeError" in source[guard : guard + 400], label


def test_trained_model_exposes_pre_refit_validation_probabilities():
    field_names = set(TrainedModel.__dataclass_fields__)
    assert "validation_probabilities" in field_names
    assert TrainedModel.__dataclass_fields__["validation_probabilities"].default is None


def test_refit_does_not_overwrite_the_captured_validation_probabilities():
    """The refit must not clobber the only honest validation signal."""
    from src.pipeline import train as train_mod

    source = inspect.getsource(train_mod.train_models)
    refit_at = source.find("refit_on_train_plus_validation and")
    assert refit_at != -1, "expected the documented refit step"
    tail = source[refit_at:]
    assert "validation_probabilities = None" not in tail
    # The refit assigns a new estimator but must leave the captured array alone.
    assert "selected.estimator = estimator" in tail


def test_ablation_arms_share_one_test_intersection():
    """Ablation deltas are only comparable on rows every arm actually scored."""
    source = inspect.getsource(ablation.run_ablation) + inspect.getsource(
        ablation._common_test_index
    )
    assert "common" in source, "expected a common test-row intersection"
    assert "intersection" in source, "the arms must be intersected, not aligned positionally"


def test_positive_class_proba_is_used_for_both_threshold_and_backtest():
    """A raw 2-column predict_proba must be reduced to P(up) consistently."""
    from src.evaluation.metrics import positive_class_proba

    binary = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert positive_class_proba(binary).tolist() == [0.8, 0.4]
    assert positive_class_proba(np.array([0.3, 0.7])).tolist() == [0.3, 0.7]
    for label, source in (
        ("runner.run_experiment", inspect.getsource(runner.run_experiment)),
        ("ablation._fit_and_score", inspect.getsource(ablation._fit_and_score)),
    ):
        assert "positive_class_proba" in source, label


def test_select_threshold_rejects_a_frame_without_probabilities():
    from src.evaluation.backtest import BacktestRules
    from src.experiments.threshold import select_threshold

    frame = pd.DataFrame(
        {"future_return": [0.01, -0.01] * 20, "target": [1, 0] * 20},
        index=pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC"),
    )
    rules = BacktestRules(probability_threshold=0.5)
    with pytest.raises(Exception):
        select_threshold(frame, rules, min_trades=1)


def test_experiment_specs_are_ordered_from_baseline_to_full():
    from src.experiments.spec import EXPERIMENTS

    ids = list(EXPERIMENTS)
    assert ids[0].startswith("EXP-00"), ids
    assert ids[-1].startswith("EXP-04"), ids
    assert len(set(ids)) == len(ids), "experiment ids must be unique"
    for spec in EXPERIMENTS.values():
        assert spec.feature_groups, f"{spec.experiment_id} has no feature groups"
        assert spec.hypothesis, f"{spec.experiment_id} has no hypothesis"
        assert spec.data_sources, f"{spec.experiment_id} has no data sources"


def test_baseline_spec_uses_only_technical_features():
    from src.experiments.spec import EXPERIMENTS

    baseline = EXPERIMENTS["EXP-00-OHLCV-TECHNICAL"]
    assert baseline.feature_groups == ("technical",)
    assert baseline.experiment_id == "EXP-00-OHLCV-TECHNICAL"


def test_report_claims_honest_threshold_provenance():
    """If a backtest reports a threshold, it must say where it came from."""
    source = inspect.getsource(runner.run_experiment)
    assert "threshold_provenance" in source
    assert "validation" in source.split("threshold_provenance")[1][:200]
