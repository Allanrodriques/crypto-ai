"""Quantify whether a measured difference is bigger than the noise.

The problem
-----------
Comparing two models by their test ROC-AUC produces a number that is precise and
meaningless: "0.6012 versus 0.6061" reads like evidence, but a model fitted on
41k autocorrelated hourly bars has an AUC that swings by several points depending
on which months you happened to score.  Reporting the delta alone invites the
reader to treat noise as a result.

Why an ordinary bootstrap will not do
-------------------------------------
Hourly crypto returns are strongly autocorrelated: a 0.5% move tends to be
followed by more 0.5% moves.  Resampling individual rows destroys that structure
and manufactures a confidence interval that is far too narrow, which is worse
than no interval because it looks rigorous.

So this module resamples *blocks of consecutive rows* (a moving-block
bootstrap), which preserves short-range dependence and yields an honest, wider
interval.  The two models are always scored on the *same* resample, so the
comparison is paired: the shared market path cancels and the interval reflects
genuine model disagreement rather than market luck.

What it can and cannot establish
-------------------------------
The interval is a statement about this test period, not about future markets.  A
delta whose 95% interval includes zero means "not distinguishable from noise in
the data we have" - which is a different claim from "the features do not help",
and the report should keep the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils import get_logger

logger = get_logger("evaluation.bootstrap")

#: Default block length in rows.  24h blocks preserve the daily autocorrelation
#: that dominates hourly crypto series while still yielding many resamples.
DEFAULT_BLOCK_SIZE = 24
DEFAULT_RESAMPLES = 1_000


@dataclass
class PairedBlockBootstrap:
    """Distribution of a paired metric difference over block resamples."""

    point_a: float | None
    point_b: float | None
    point_delta: float | None
    ci_low: float | None
    ci_high: float | None
    probability_a_better: float | None
    n_resamples: int
    n_effective: int
    block_size: int
    n_blocks: int
    deltas: np.ndarray | None = field(default=None, repr=False)
    note: str = ""

    @property
    def significant(self) -> bool:
        """True only if the whole 95% interval sits above zero.

        This is deliberately one-sided: the question each study asks is "does
        arm A beat arm B".  A one-sided flag is the right thing for that
        question, and conflating it with a two-sided test is what previously
        reported a clearly *negative* interval (the 24h target, CI
        ``[-0.104, -0.015]``) as ``significant=False``, indistinguishable from
        an interval straddling zero.  Use :attr:`significantly_worse` for the
        other direction.
        """
        if self.ci_low is None or self.ci_high is None or self.point_delta is None:
            return False
        return bool(self.ci_low > 0.0)

    @property
    def significantly_worse(self) -> bool:
        """True if the whole 95% interval sits below zero, i.e. B beat A.

        Without this, a result that is significantly *worse* is rendered
        identically to one that merely failed to beat the baseline.
        """
        if self.ci_low is None or self.ci_high is None or self.point_delta is None:
            return False
        return bool(self.ci_high < 0.0)

    @property
    def verdict(self) -> str:
        """Three-way reading: better, worse, or indistinguishable from noise."""
        if self.significant:
            return "better"
        if self.significantly_worse:
            return "worse"
        return "indistinguishable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_a": self.point_a,
            "point_b": self.point_b,
            "point_delta": self.point_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "probability_a_better": self.probability_a_better,
            "significant_at_95": self.significant,
            "significantly_worse_at_95": self.significantly_worse,
            "verdict": self.verdict,
            "n_resamples": self.n_resamples,
            "n_effective": self.n_effective,
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "note": self.note,
        }


def _moving_block_starts(n: int, n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """Random block start offsets, wrapping around the end of the sample."""
    if n_blocks <= 0:
        return np.empty(0, dtype=int)
    return rng.integers(0, n, size=n_blocks)


def _resample_indices(n: int, block_size: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Build ``(n_resamples, n)`` index matrix of concatenated random blocks."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    offsets = np.arange(block_size)
    index = (starts[:, :, None] + offsets[None, None, :]) % n
    return index.reshape(n_resamples, -1)[:, :n]


def paired_block_bootstrap_auc(
    y_true,
    proba_a,
    proba_b,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_resamples: int = DEFAULT_RESAMPLES,
    random_state: int = 42,
    labels: tuple[str, str] = ("variant", "baseline"),
) -> PairedBlockBootstrap:
    """Block-bootstrap the paired ROC-AUC difference ``AUC(a) - AUC(b)``."""
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(proba_a, dtype="float64").ravel()
    b = np.asarray(proba_b, dtype="float64").ravel()
    if not (y.shape == a.shape == b.shape):
        raise ValueError(f"shape mismatch: y={y.shape} a={a.shape} b={b.shape}")

    n = y.size
    if np.unique(y).size < 2:
        return PairedBlockBootstrap(None, None, None, None, None, None, 0, n, block_size, 0,
                                    note="undefined: single class in y_true")

    point_a = float(roc_auc_score(y, a))
    point_b = float(roc_auc_score(y, b))
    point_delta = point_a - point_b

    block = max(1, min(int(block_size), n))
    rng = np.random.default_rng(random_state)
    index = _resample_indices(n, block, n_resamples, rng)

    deltas = np.empty(n_resamples, dtype="float64")
    for r in range(n_resamples):
        rows = index[r]
        y_r = y[rows]
        # A block bootstrap can pull a single-class sample; that draw's AUC is
        # undefined, so it is skipped rather than scored as 0.
        if np.unique(y_r).size < 2:
            deltas[r] = np.nan
            continue
        deltas[r] = roc_auc_score(y_r, a[rows]) - roc_auc_score(y_r, b[rows])

    valid = deltas[np.isfinite(deltas)]
    if valid.size == 0:
        return PairedBlockBootstrap(point_a, point_b, point_delta, None, None, None,
                                    n_resamples, n, block, 0,
                                    note="no resample contained both classes")

    low, high = np.percentile(valid, [2.5, 97.5])
    return PairedBlockBootstrap(
        point_a=point_a,
        point_b=point_b,
        point_delta=point_delta,
        ci_low=float(low),
        ci_high=float(high),
        probability_a_better=float(np.mean(valid > 0.0)),
        n_resamples=int(valid.size),
        n_effective=n,
        block_size=block,
        n_blocks=int(np.ceil(n / block)),
        deltas=valid,
        note=(
            f"paired {block}-row block bootstrap of {labels[0]} minus {labels[1]}; "
            f"preserves short-range autocorrelation, so the interval is wider than an "
            f"i.i.d. bootstrap would give"
        ),
    )


def compare_predictions_table(
    y_true,
    predictions: dict[str, np.ndarray],
    baseline_key: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Bootstrap every prediction set in ``predictions`` against ``baseline_key``."""
    if baseline_key not in predictions:
        raise KeyError(f"baseline {baseline_key!r} not among {sorted(predictions)}")
    base = predictions[baseline_key]
    rows = []
    for key, proba in predictions.items():
        boot = paired_block_bootstrap_auc(y_true, proba, base, labels=(key, baseline_key), **kwargs)
        rows.append({"variant": key, "baseline": baseline_key, **boot.to_dict()})
    return rows
