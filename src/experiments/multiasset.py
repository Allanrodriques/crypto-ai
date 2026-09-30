"""Per-asset models versus one global multi-asset model.

The question
------------
A BTC-only model sees one asset's idiosyncrasies as if they were market
structure.  Two alternatives are tested here rather than assumed:

*per-asset*  a separate model per symbol.  Maximum idiosyncrasy, minimum data
             per model - and with 41k rows split 70/15/15, ETH and especially the
             smaller names have far less history to learn from.
*global*     one model on stacked rows, with the symbol available as a feature.
             More data and shared structure across assets, at the cost of
             forcing one set of weights on assets with different dynamics.

Neither is assumed better.  The interesting quantity is not which headline AUC
wins but whether the ranking *transfers*: a global model that beats a per-asset
model on BTC but collapses on XRP has not solved the problem, and averaging
across symbols would hide exactly that.

What is deliberately not claimed
--------------------------------
Nothing here is evidence of a tradable cross-sectional edge.  Both models are
long-only, single-asset-at-a-time, and scored on the same chronological split;
the comparison is about whether feature transfer across assets is worth pursuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import paired_block_bootstrap_auc
from src.evaluation.metrics import classification_metrics, positive_class_proba
from src.experiments.spec import ExperimentSpec
from src.models import build_model
from src.utils import get_logger, save_json, set_global_seed

logger = get_logger("experiments.multiasset")

#: Symbols compared.  Ordered by expected data depth so a short-history failure
#: is visible rather than silent.
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")

#: Minimum rows a symbol must contribute to be included at all.
MIN_ROWS = 5_000


@dataclass
class AssetResult:
    """Per-symbol outcome for one modelling strategy."""

    symbol: str
    strategy: str
    n_rows: int = 0
    n_train: int = 0
    n_test: int = 0
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_accuracy: float | None = None
    delta_roc_auc_vs_per_asset: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool = False
    #: One-sided flag in the other direction, so a clearly negative interval is
    #: distinguishable from one that merely straddles zero.
    significantly_worse: bool = False
    verdict: str = "not_compared"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "n_rows": self.n_rows,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_roc_auc": self.test_roc_auc,
            "test_pr_auc": self.test_pr_auc,
            "test_accuracy": self.test_accuracy,
            "delta_roc_auc_vs_per_asset": self.delta_roc_auc_vs_per_asset,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant_at_95": self.significant,
            "significantly_worse_at_95": self.significantly_worse,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    raw = np.asarray(model.decision_function(X))
    return np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw


def _fit_predict(spec, config, X_tr, y_tr, X_te, feature_columns) -> np.ndarray:
    """Fit on train only, return positive-class probabilities for the test rows."""
    set_global_seed(int(getattr(config, "random_state", 42)))
    model = build_model(spec.model_name, config, feature_columns)
    try:
        model.fit(X_tr, y_tr)
    except TypeError:
        model.fit(X_tr, y_tr)
    return positive_class_proba(_proba(model, X_te))


def _symbol_features(context_for, symbol: str, groups: Sequence[str]) -> pd.DataFrame | None:
    """Feature matrix for one symbol, or ``None`` when its data is unusable."""
    try:
        context = context_for(symbol)
    except Exception as exc:
        logger.warning("multi-asset: %s context unavailable: %s", symbol, exc)
        return None
    matrix = context.features(list(groups))
    if len(matrix) < MIN_ROWS:
        return None
    return matrix


def run_multiasset(
    spec: ExperimentSpec,
    config,
    *,
    context_for,
    output_dir: str | Path,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    n_resamples: int = 300,
    block_size: int = 24,
) -> dict[str, Any]:
    """Train per-asset and global models, then compare them per symbol.

    ``context_for(symbol)`` must return an
    :class:`~src.dataset.multisource.ExperimentContext` for that symbol, so this
    function never reaches out to the network itself.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = list(spec.feature_groups)

    per_symbol: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for symbol in symbols:
        logger.info("multi-asset: preparing %s", symbol)
        matrix = _symbol_features(context_for, symbol, groups)
        if matrix is None:
            skipped[symbol] = f"fewer than {MIN_ROWS} usable rows"
            continue
        from src.dataset.factory import build_dataset

        dataset = build_dataset(
            matrix, context_for(symbol).spot, spec.target, spec.split,
            symbol=symbol, interval=spec.interval,
            feature_groups=groups, data_sources=list(spec.data_sources),
        )
        if not len(dataset.frame):
            skipped[symbol] = "empty dataset after warm-up and label resolution"
            continue
        per_symbol[symbol] = {
            "dataset": dataset,
            "X_train": dataset.splits["train"].X,
            "y_train": dataset.splits["train"].y,
            "X_test": dataset.splits["test"].X,
            "y_test": dataset.splits["test"].y,
        }

    if not per_symbol:
        raise RuntimeError(f"no symbol had usable data; skipped: {skipped}")

    results: list[AssetResult] = []

    # ---- per-asset models ---------------------------------------------------
    for symbol, data in per_symbol.items():
        dataset = data["dataset"]
        proba = _fit_predict(
            spec, config, data["X_train"], data["y_train"], data["X_test"],
            dataset.feature_columns,
        )
        # Kept for the paired bootstrap below, so the per-asset model is fitted
        # exactly once and both strategies are compared on identical rows.
        data["test_proba"] = proba
        m = classification_metrics(data["y_test"], proba)
        results.append(AssetResult(
            symbol=symbol, strategy="per_asset", n_rows=len(dataset.frame),
            n_train=len(data["X_train"]), n_test=len(data["X_test"]),
            test_roc_auc=m.get("roc_auc"), test_pr_auc=m.get("pr_auc"),
            test_accuracy=m.get("accuracy"),
            # The per-asset row is the reference arm; it is never itself
            # tested against anything.
            verdict="reference",
        ))

    # ---- one global model on stacked rows ---------------------------------
    # The common feature set is the intersection, so a symbol missing a column
    # cannot silently receive a different (or zero-filled) one.
    common_columns: list[str] | None = None
    for data in per_symbol.values():
        cols = set(data["dataset"].feature_columns)
        common_columns = cols if common_columns is None else (common_columns & cols)
    if not common_columns:
        raise RuntimeError("no feature is shared across every symbol")
    columns = sorted(common_columns)

    X_tr = pd.concat(
        [d["X_train"][columns].assign(_symbol=s) for s, d in per_symbol.items()]
    )
    y_tr = pd.concat([d["y_train"] for d in per_symbol.values()])
    X_te = pd.concat(
        [d["X_test"][columns].assign(_symbol=s) for s, d in per_symbol.items()]
    )
    y_te = pd.concat([d["y_test"] for d in per_symbol.values()])

    # Encode the symbol as an ordinal so the global model can condition on it.
    symbol_order = {s: i for i, s in enumerate(per_symbol)}
    X_tr["_symbol_id"] = X_tr.pop("_symbol").map(symbol_order).astype(float)
    X_te["_symbol_id"] = X_te.pop("_symbol").map(symbol_order).astype(float)

    global_proba = _fit_predict(spec, config, X_tr, y_tr, X_te, columns + ["_symbol_id"])
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for symbol, data in per_symbol.items():
        n = len(data["X_test"])
        offsets[symbol] = (cursor, cursor + n)
        cursor += n

    for symbol, (lo, hi) in offsets.items():
        block = slice(lo, hi)
        y_block = y_te.iloc[block]
        p_block = global_proba[block]
        m = classification_metrics(y_block, p_block)
        results.append(AssetResult(
            symbol=symbol, strategy="global", n_rows=len(per_symbol[symbol]["dataset"].frame),
            n_train=len(per_symbol[symbol]["X_train"]), n_test=hi - lo,
            test_roc_auc=m.get("roc_auc"), test_pr_auc=m.get("pr_auc"),
            test_accuracy=m.get("accuracy"),
            notes=["trained on all symbols stacked, with a symbol-id feature"],
            verdict="not_compared",
        ))

    # ---- paired comparison per symbol -------------------------------------
    by_key = {(r.symbol, r.strategy): r for r in results}
    for symbol in per_symbol:
        per_asset = by_key[(symbol, "per_asset")]
        glob = by_key[(symbol, "global")]
        base = per_symbol[symbol].get("test_proba")
        if base is None:
            continue
        lo, hi = offsets[symbol]
        boot = paired_block_bootstrap_auc(
            per_symbol[symbol]["y_test"], global_proba[lo:hi], base,
            n_resamples=n_resamples, block_size=block_size, labels=("global", "per_asset"),
        )
        glob.delta_roc_auc_vs_per_asset = boot.point_delta
        glob.ci_low, glob.ci_high = boot.ci_low, boot.ci_high
        glob.significant = boot.significant
        glob.significantly_worse = boot.significantly_worse
        glob.verdict = boot.verdict

    table = pd.DataFrame([r.to_dict() for r in results])
    table.to_csv(out / "multiasset_table.csv", index=False)

    summary = _summarise(table, skipped, columns)
    save_json(
        {
            "symbols_requested": list(symbols),
            "symbols_used": sorted(per_symbol),
            "symbols_skipped": skipped,
            "shared_features": columns,
            "n_shared_features": len(columns),
            "summary": summary,
            "results": [r.to_dict() for r in results],
            "caveat": (
                "A global model beating a per-asset model on average can still lose on "
                "individual symbols; the per-symbol deltas are the meaningful output. "
                "Neither strategy demonstrates a tradable cross-sectional edge."
            ),
        },
        out / "multiasset_summary.json",
    )
    return {"table": table, "results": results, "summary": summary}


def _summarise(table: pd.DataFrame, skipped: dict[str, str], columns: Sequence[str]) -> dict[str, Any]:
    per_asset = table[table["strategy"] == "per_asset"]
    glob = table[table["strategy"] == "global"]
    mean_per = float(per_asset["test_roc_auc"].mean()) if not per_asset.empty else None
    mean_glob = float(glob["test_roc_auc"].mean()) if not glob.empty else None

    # Use the precomputed one-sided verdicts rather than re-deriving them from
    # the CI columns here: this keeps the prose and the table in agreement even
    # if the bootstrap's significance rule changes.
    winners = [
        f"{row['symbol']}: global better by {row['delta_roc_auc_vs_per_asset']:+.4f} "
        f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        for _, row in glob.iterrows()
        if bool(row.get("significant_at_95"))
    ]
    losers = [
        f"{row['symbol']}: global worse by {row['delta_roc_auc_vs_per_asset']:+.4f} "
        f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        for _, row in glob.iterrows()
        if bool(row.get("significantly_worse_at_95"))
    ]
    return {
        "mean_test_roc_auc_per_asset": mean_per,
        "mean_test_roc_auc_global": mean_glob,
        "global_mean_delta": (mean_glob - mean_per)
        if (mean_glob is not None and mean_per is not None)
        else None,
        "symbols_where_global_significantly_better": winners,
        "symbols_where_global_significantly_worse": losers,
        "symbols_skipped": skipped,
        "n_shared_features": len(columns),
    }
