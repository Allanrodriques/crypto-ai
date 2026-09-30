"""Pick the trading threshold on validation, then freeze it.

Why the threshold is the most dangerous knob in this project
----------------------------------------------------------
A probability threshold is a free parameter with the same power as a model
hyper-parameter, but it sits directly on the P&L.  Sweeping it on the test set
and reporting the best row is test-set tuning with extra steps: it reliably
manufactures a strategy that looks profitable and evaporates in production.

Two things guard against that here:

1. The threshold is chosen on the **validation** block only, then applied to the
   test block exactly once.  The test block never influences it.
2. The selection is constrained and its stability is measured, not assumed:
   thresholds producing too few trades are rejected outright, and the block is
   split in half to check whether the two halves agree.  When they disagree the
   result is reported as fragile rather than quietly presented as a finding.

Fixed thresholds do not generalise across models
------------------------------------------------
V1's logistic regression emitted probabilities spanning roughly 0.3-0.8, so a
hand-picked 0.60 was meaningful.  Gradient boosting produces compressed,
better-calibrated probabilities that often never reach 0.60 - a fixed 0.60 then
yields *zero trades* and a silently "safe" backtest that says nothing.  Selecting
per experiment on validation removes that artefact while keeping the procedure
identical across experiments, so their comparison stays fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.backtest import BacktestRules, run_backtest
from src.utils import get_logger

logger = get_logger("experiments.threshold")

#: Objectives the threshold may be chosen on, pre-registered so the choice is
#: not itself an after-the-fact decision.
OBJECTIVES = ("excess_vs_buy_hold", "sharpe_ratio", "profit_factor", "total_return")


@dataclass
class ThresholdSelection:
    """The chosen threshold plus the evidence behind it."""

    threshold: float
    objective: str
    min_trades: int
    n_trades: int
    objective_value: float | None
    curve: list[dict[str, Any]] = field(default_factory=list)
    stability: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "min_trades": self.min_trades,
            "n_trades": self.n_trades,
            "objective_value": self.objective_value,
            "curve": self.curve,
            "stability": self.stability,
            "note": self.note,
            "provenance": (
                "selected on the validation block; the test block never informed it"
            ),
        }


def _objective_value(metrics: dict[str, Any], objective: str) -> float | None:
    value = metrics.get(objective)
    return None if value is None else float(value)


def evaluate_threshold_curve(
    frame: pd.DataFrame,
    rules: BacktestRules,
    thresholds: Sequence[float],
    *,
    probability_column: str = "probability_up",
) -> list[dict[str, Any]]:
    """Backtest every candidate threshold, returning one summary row each."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        local = BacktestRules(**{**rules.__dict__, "probability_threshold": float(threshold)})
        local.validate()
        result = run_backtest(frame, local, probability_column=probability_column)
        m = result.metrics
        rows.append(
            {
                "probability_threshold": float(threshold),
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "total_return": m["total_return"],
                "excess_vs_buy_hold": m["excess_vs_buy_hold"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sortino_ratio": m["sortino_ratio"],
                "profit_factor": m["profit_factor"],
                "max_drawdown": m["max_drawdown"],
                "time_in_market": m["time_in_market"],
            }
        )
    return rows


def select_threshold(
    validation_frame: pd.DataFrame,
    rules: BacktestRules,
    *,
    thresholds: Sequence[float] | None = None,
    objective: str = "excess_vs_buy_hold",
    min_trades: int = 30,
    probability_column: str = "probability_up",
) -> ThresholdSelection:
    """Choose a threshold on the validation block and report how stable it is.

    Parameters
    ----------
    objective:
        Which validation statistic to maximise.  ``excess_vs_buy_hold`` is the
        default because raw return is trivially maximised by never trading: a
        threshold that makes no trades scores 0% return and looks respectable
        next to a losing strategy, while its *excess* over holding is clearly
        negative.
    min_trades:
        Reject thresholds that would leave too little trading to judge.  This is
        a robustness constraint, not a performance filter.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    grid = list(thresholds) if thresholds is not None else list(np.round(np.arange(0.30, 0.86, 0.02), 3))

    curve = evaluate_threshold_curve(
        validation_frame, rules, grid, probability_column=probability_column
    )
    table = pd.DataFrame(curve)

    eligible = table[table["n_trades"] >= min_trades].copy()
    note = ""
    if eligible.empty:
        # No threshold clears the trade-count floor.  Fall back to the most
        # *active* threshold rather than the best-scoring one, and say so: with
        # no trade count, objective values are noise.
        best = table.loc[table["n_trades"].idxmax()]
        return ThresholdSelection(
            threshold=float(best["probability_threshold"]),
            objective=objective,
            min_trades=min_trades,
            n_trades=int(best["n_trades"]),
            objective_value=_objective_value(best.to_dict(), objective),
            curve=curve,
            stability={"checked": False, "reason": "no threshold met the trade floor"},
            note=(
                f"No threshold in {grid[0]:.2f}-{grid[-1]:.2f} produced at least {min_trades} "
                f"validation trades; the model is too flat to trade this target. The most "
                f"active threshold was used and its statistics are not meaningful."
            ),
        )

    scored = eligible.dropna(subset=[objective])
    if scored.empty:
        best = eligible.iloc[0]
        note = f"All eligible thresholds had an undefined {objective}; the first was kept."
    else:
        best = scored.loc[scored[objective].idxmax()]

    stability = _stability(
        validation_frame, rules, grid, objective, min_trades, probability_column
    )
    if stability.get("agrees") is False:
        note = (
            "The two validation halves chose different thresholds, so this choice is "
            "period-dependent and should be treated as fragile."
        )

    logger.info(
        "threshold %.2f selected on validation by %s (n=%d trades, %s=%.4f)",
        float(best["probability_threshold"]), objective, int(best["n_trades"]),
        objective, float(best.get(objective) or 0.0),
    )
    return ThresholdSelection(
        threshold=float(best["probability_threshold"]),
        objective=objective,
        min_trades=min_trades,
        n_trades=int(best["n_trades"]),
        objective_value=_objective_value(best.to_dict(), objective),
        curve=curve,
        stability=stability,
        note=note,
    )


def _stability(
    frame: pd.DataFrame,
    rules: BacktestRules,
    grid: Sequence[float],
    objective: str,
    min_trades: int,
    probability_column: str,
) -> dict[str, Any]:
    """Split validation in half and check whether both halves agree.

    A threshold that only wins in one half of the validation period is a
    description of that half, not a property of the model.  Reporting the
    disagreement is more useful than hiding it behind a single number.
    """
    if len(frame) < 400:
        return {"checked": False, "reason": "validation block too short to split"}

    midpoint = len(frame) // 2
    picks: dict[str, float | None] = {}
    for name, part in (("first_half", frame.iloc[:midpoint]), ("second_half", frame.iloc[midpoint:])):
        curve = pd.DataFrame(
            evaluate_threshold_curve(part, rules, grid, probability_column=probability_column)
        )
        eligible = curve[(curve["n_trades"] >= max(10, min_trades // 3))].dropna(subset=[objective])
        picks[name] = (
            float(eligible.loc[eligible[objective].idxmax(), "probability_threshold"])
            if not eligible.empty
            else None
        )

    values = [v for v in picks.values() if v is not None]
    agrees = len(values) == 2 and abs(values[0] - values[1]) <= 0.04
    return {
        "checked": True,
        "picks": picks,
        "agrees": agrees,
        "tolerance": 0.04,
    }
