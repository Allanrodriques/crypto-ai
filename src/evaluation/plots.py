"""Research plots.  Every figure is written to ``reports/plots/``.

The ten figures the project produces:

===  ==========================================================================
 1   ``price_history``            close price over the full sample
 2   ``split_overview``           train / validation / test shaded on the price
 3   ``feature_distributions``    train-vs-test distribution of the top features
 4   ``confusion_matrix``         counts for the evaluated split
 5   ``roc_curve``                ROC with the diagonal as the coin-flip line
 6   ``precision_recall_curve``   PR curve against the positive-class base rate
 7   ``probability_over_time``    ``probability_up`` and realised outcome
 8   ``equity_curve``             strategy equity vs buy & hold
 9   ``drawdown_curve``           underwater plot of the strategy
10   ``feature_importance``       XGBoost gain-based importance, top 20
===  ==========================================================================

These are diagnostic figures for a research notebook.  Nothing here is a trading
chart and none of it should be read as evidence of future performance.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.utils import get_logger

logger = get_logger("evaluation.plots")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Consistent house style so figures read as one set.
_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
}
_COLORS = {
    "price": "#1f2933",
    "train": "#4c78a8",
    "validation": "#f58518",
    "test": "#54a24b",
    "model": "#b279a2",
    "benchmark": "#9aa5b1",
    "up": "#2ca02c",
    "down": "#d62728",
}


def _fig(width: float = 12.0, height: float = 5.0) -> tuple[plt.Figure, plt.Axes]:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _save(fig: plt.Figure, directory: str | os.PathLike[str], name: str) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot -> %s", path)
    return path


# --------------------------------------------------------------------------- 1-2: price & splits

def plot_price_history(frame: pd.DataFrame, directory: str | os.PathLike[str], *, symbol: str = "BTCUSDT",
                       interval: str = "1h") -> Path:
    """1. Close price across the full sample."""
    fig, ax = _fig(12, 5)
    ax.plot(frame.index, frame["close"], color=_COLORS["price"], linewidth=1.1, label="close")
    ax.set_title(f"{symbol} {interval} close price — {frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (quote currency)")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, directory, "price_history")


def plot_split_overview(frame: pd.DataFrame, splits: Mapping[str, Any], directory: str | os.PathLike[str],
                        *, symbol: str = "BTCUSDT") -> Path:
    """2. Chronological train / validation / test blocks shaded over the price."""
    fig, ax = _fig(12, 5)
    ax.plot(frame.index, frame["close"], color=_COLORS["price"], linewidth=0.9)
    for name, colour in (("train", _COLORS["train"]), ("validation", _COLORS["validation"]), ("test", _COLORS["test"])):
        split = splits.get(name)
        if split is None or not len(split):
            continue
        ax.axvspan(split.start, split.end, color=colour, alpha=0.18, label=f"{name} ({len(split):,} rows)")
    ax.set_title(f"{symbol} — chronological split (never shuffled). Later blocks are strictly later in time.")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    return _save(fig, directory, "split_overview")


# --------------------------------------------------------------------------- 3: distributions

def plot_feature_distributions(
    dataset_frame: pd.DataFrame,
    splits: Mapping[str, Any],
    feature_columns: Sequence[str],
    directory: str | os.PathLike[str],
    *,
    top_n: int = 8,
    by_importance: Mapping[str, float] | None = None,
) -> Path:
    """3. Train vs test distribution of the most informative features.

    A feature whose train and test distributions differ sharply is one whose
    learned relationship may not survive out of sample — visible here rather
    than hidden inside a single accuracy number.
    """
    if by_importance:
        ranked = sorted(feature_columns, key=lambda c: -float(by_importance.get(c, 0.0)))
    else:
        ranked = list(feature_columns)
    chosen = ranked[:top_n]

    ncols = 2
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), chosen):
        for split_name, colour in (("train", _COLORS["train"]), ("test", _COLORS["test"])):
            split = splits.get(split_name)
            if split is None or name not in split.X.columns:
                continue
            values = split.X[name].dropna().to_numpy()
            if values.size < 10:
                continue
            # Clip to the 1st-99th percentile so a single outlier cannot flatten the density.
            lo, hi = np.quantile(values, [0.01, 0.99])
            ax.hist(np.clip(values, lo, hi), bins=40, alpha=0.45, color=colour, label=split_name, density=True)
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
    for ax in axes.ravel()[len(chosen):]:
        ax.set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Feature distribution: train (blue) vs test (green) — 1st-99th pct clipped", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "feature_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot -> %s", path)
    return path


# --------------------------------------------------------------------------- 4-6: classifier diagnostics

def plot_confusion_matrix(y_true: Any, y_pred: Any, directory: str | os.PathLike[str], *,
                          title: str = "Confusion matrix", name: str = "confusion_matrix") -> Path:
    """4. Confusion matrix as annotated counts."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = _fig(5.2, 4.6)
    image = ax.imshow(matrix, cmap="Blues")
    ax.grid(False)
    labels = ["actual\nDOWN", "actual\nUP"]
    ax.set_xticks([0, 1], labels=["predicted\nDOWN", "predicted\nUP"])
    ax.set_yticks([0, 1], labels=labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=15, color="white" if matrix[i, j] > matrix.max() / 2 else "#1f2933")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _save(fig, directory, name)


def plot_roc_curve(y_true: Any, y_pred_proba: Any, directory: str | os.PathLike[str], *,
                   title: str = "ROC curve", name: str = "roc_curve",
                   comparison: Mapping[str, np.ndarray] | None = None) -> Path:
    """5. ROC curve, with the coin-flip diagonal for reference."""
    y_true = np.asarray(y_true).astype(int)
    fig, ax = _fig(5.6, 5.2)
    if comparison:
        for label, proba in comparison.items():
            fpr, tpr, _ = roc_curve(y_true, np.asarray(proba))
            ax.plot(fpr, tpr, linewidth=1.0, alpha=0.75, label=f"{label} (AUC {roc_auc_score(y_true, proba):.3f})")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_pred_proba))
    auc = roc_auc_score(y_true, np.asarray(y_pred_proba))
    ax.plot(fpr, tpr, color=_COLORS["model"], linewidth=2.0, label=f"model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=_COLORS["benchmark"], linestyle="--", linewidth=1.0, label="random (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} — AUC {auc:.4f}")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return _save(fig, directory, name)


def plot_precision_recall_curve(y_true: Any, y_pred_proba: Any, directory: str | os.PathLike[str], *,
                                 title: str = "Precision-Recall curve", name: str = "precision_recall_curve",
                                 comparison: Mapping[str, np.ndarray] | None = None) -> Path:
    """6. PR curve against the positive-class base rate.

    The flat line at the base rate is what a random classifier achieves, so the
    gap between curve and line is the entire claim to an edge.
    """
    y_true = np.asarray(y_true).astype(int)
    base_rate = float(y_true.mean())
    fig, ax = _fig(5.6, 5.2)
    if comparison:
        for label, proba in comparison.items():
            precision, recall, _ = precision_recall_curve(y_true, np.asarray(proba))
            ax.plot(recall, precision, linewidth=1.0, alpha=0.75, label=label)
    precision, recall, _ = precision_recall_curve(y_true, np.asarray(y_pred_proba))
    from sklearn.metrics import average_precision_score

    ax.plot(recall, precision, color=_COLORS["model"], linewidth=2.0,
            label=f"model (AP {average_precision_score(y_true, np.asarray(y_pred_proba)):.3f})")
    ax.axhline(base_rate, color=_COLORS["benchmark"], linestyle="--", linewidth=1.0,
               label=f"base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title} — positive share {base_rate:.1%}")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    return _save(fig, directory, name)


def plot_probability_over_time(frame: pd.DataFrame, directory: str | os.PathLike[str], *,
                               probability_column: str = "probability_up",
                               threshold: float = 0.5, title: str = "Predicted probability over time",
                               name: str = "probability_over_time") -> Path:
    """7. ``probability_up`` through time against the realised label and price."""
    fig, (ax_p, ax_t) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1],
                                     gridspec_kw={"hspace": 0.12})
    ax_p.plot(frame.index, frame[probability_column], color=_COLORS["model"], linewidth=0.8, label="probability_up")
    ax_p.axhline(threshold, color=_COLORS["down"], linestyle="--", linewidth=1.0,
                 label=f"decision threshold {threshold:.2f}")
    ax_p.fill_between(frame.index, 0.5, 1.0, where=frame[probability_column] >= 0.5, color=_COLORS["up"], alpha=0.12)
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_ylabel("P(up)")
    ax_p.set_title(title)
    ax_p.legend(loc="upper left", frameon=False, fontsize=8)

    if "target" in frame.columns:
        realised = frame["target"].to_numpy(dtype=float)
        ax_t.fill_between(frame.index, 0, 1, where=realised == 1, color=_COLORS["up"], alpha=0.55,
                          step="mid", label="realised UP")
        ax_t.fill_between(frame.index, 0, 1, where=realised == 0, color=_COLORS["down"], alpha=0.45,
                          step="mid", label="realised DOWN")
        ax_t.set_ylim(0, 1)
        ax_t.set_yticks([0, 1], ["DOWN", "UP"])
        ax_t.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    ax_t.set_xlabel("Time (UTC)")
    ax_t.set_ylabel("realised")
    return _save(fig, directory, name)


def plot_calibration_curve(curve: pd.DataFrame, directory: str | os.PathLike[str], *, title: str = "Calibration",
                           name: str = "calibration_curve") -> Path:
    """Reliability diagram: predicted probability against observed frequency."""
    fig, ax = _fig(5.6, 5.2)
    usable = curve.dropna(subset=["mean_predicted", "observed_frequency"])
    ax.plot([0, 1], [0, 1], color=_COLORS["benchmark"], linestyle="--", linewidth=1.0, label="perfect calibration")
    ax.plot(usable["mean_predicted"], usable["observed_frequency"], "o-", color=_COLORS["model"],
            linewidth=1.6, markersize=5, label="model")
    if not usable.empty:
        ax2 = ax.twinx()
        ax2.bar(usable["mean_predicted"], usable["n"], width=0.06, color=_COLORS["benchmark"], alpha=0.25)
        ax2.set_ylabel("samples in bin", color=_COLORS["benchmark"], fontsize=8)
        ax2.grid(False)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return _save(fig, directory, name)


# --------------------------------------------------------------------------- 8-9: backtest

def plot_equity_curve(equity_curve: pd.DataFrame, metrics: Mapping[str, Any], directory: str | os.PathLike[str], *,
                      title: str = "Equity curve", name: str = "equity_curve") -> Path:
    """8. Strategy equity against a buy & hold line on the same window."""
    fig, ax = _fig(12, 5)
    initial = float(metrics.get("initial_capital", 1.0) or 1.0)
    ax.plot(equity_curve.index, equity_curve["equity"], color=_COLORS["model"], linewidth=1.4, label="strategy (net of costs)")
    bh_return = metrics.get("buy_hold_return")
    if bh_return is not None:
        ax.plot(equity_curve.index, initial * (1.0 + float(bh_return)) * np.ones(len(equity_curve)),
                color=_COLORS["benchmark"], linestyle="--", linewidth=1.3, label=f"buy & hold ({float(bh_return):+.1%})")
    ax.axhline(initial, color=_COLORS["price"], linewidth=0.8, alpha=0.5, label="starting capital")
    ax.set_title(
        f"{title} — total {float(metrics.get('total_return', 0.0)):+.2%} vs buy & hold "
        f"{float(bh_return):+.2%} | {int(metrics.get('n_trades', 0))} trades" if bh_return is not None else title
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (quote currency)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return _save(fig, directory, name)


def plot_drawdown_curve(equity_curve: pd.DataFrame, directory: str | os.PathLike[str], *,
                        title: str = "Drawdown", name: str = "drawdown_curve") -> Path:
    """9. Underwater plot of the strategy equity."""
    equity = equity_curve["equity"]
    drawdown = (equity.cummax() - equity) / equity.cummax()
    fig, ax = _fig(12, 4)
    ax.fill_between(equity_curve.index, drawdown, 0, color=_COLORS["down"], alpha=0.55)
    ax.plot(equity_curve.index, drawdown, color=_COLORS["down"], linewidth=0.9)
    ax.set_title(f"{title} — max {drawdown.max():.2%}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Drawdown")
    ax.invert_yaxis()
    return _save(fig, directory, name)


# --------------------------------------------------------------------------- 10: importance

def plot_feature_importance(importance: pd.DataFrame, directory: str | os.PathLike[str], *, top_n: int = 20,
                            title: str = "XGBoost feature importance",
                            name: str = "feature_importance") -> Path:
    """10. Gain-based XGBoost importance, top ``top_n`` features.

    Importance shows what the model *used*, not what *causes* price movement.
    Correlated features split credit between them unpredictably, so this chart
    is a debugging aid — not evidence of a trading signal.
    """
    column = importance.columns[1]
    top = importance.head(top_n).sort_values(column)
    fig, ax = _fig(9, max(4.0, 0.32 * len(top) + 1.6))
    ax.barh(top["feature"], top[column], color=_COLORS["model"])
    ax.set_title(f"{title} — top {len(top)} by {column}")
    ax.set_xlabel(column.replace("_", " "))
    ax.grid(axis="y", alpha=0)
    return _save(fig, directory, name)
