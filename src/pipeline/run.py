"""End-to-end orchestration: download -> validate -> dataset -> train -> evaluate -> predict.

Every script in ``scripts/`` is a thin wrapper over a stage function here, so
running the stages individually and running ``run_pipeline.py`` execute *exactly*
the same code.  There is no second, divergent "full run" path.

Stage order and the isolation it guarantees
-------------------------------------------
=====  ==========================  =========================================
1      data download              idempotent; resumes, dedups, repairs gaps
2      validation                 report written, problems surfaced not hidden
3      dataset build              features + label + purged chronological splits
4      train                      fits on train, selects on validation
5      evaluate                   **first and only** read of the test split
6      predict                    scores fresh candles with the saved artifact
=====  ==========================  =========================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import Config, load_config
from src.data.binance_client import BinanceClient, ClientSettings
from src.data.downloader import KlineDownloader, KlineStore, normalise_klines
from src.data.validator import ValidationReport, validate_klines
from src.dataset.dataset_builder import Dataset, DatasetBuilder
from src.models import ALL_MODELS
from src.pipeline.evaluate import EvaluationResult, evaluate_models
from src.pipeline.predict import predict_and_save
from src.pipeline.train import TrainingResult, train_models
from src.utils import format_timestamp, get_logger, save_json, set_global_seed, utc_now

logger = get_logger("pipeline.run")


@dataclass
class DataStageResult:
    """Outcome of the download + validation stage."""

    klines: pd.DataFrame
    validation: ValidationReport
    download: Any | None = None
    paths: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.paths is None:
            self.paths = {}


# --------------------------------------------------------------------------- stage 1-2: data

def download_data(config: Config, **overrides: Any) -> Any:
    """Download the configured kline range, resuming if a partial file exists."""
    store = KlineStore(config.paths.raw_dir, config.symbol, config.interval)
    client = BinanceClient(ClientSettings.from_config(config.data))
    try:
        downloader = KlineDownloader.from_config(config, client, **overrides)
        return downloader.download()
    finally:
        client.close()


def load_raw_data(config: Config) -> pd.DataFrame:
    """Read the stored klines for the configured symbol/interval."""
    store = KlineStore(config.paths.raw_dir, config.symbol, config.interval)
    if not store.exists():
        raise FileNotFoundError(
            f"No raw data at {store.path()}. Run `python scripts/download_data.py` first."
        )
    frame = normalise_klines(store.load())
    if frame.empty:
        raise ValueError(f"Stored klines at {store.path()} are empty")
    logger.info(
        "Loaded %d raw candles from %s (%s .. %s)",
        len(frame), store.path(), format_timestamp(frame.index.min()), format_timestamp(frame.index.max()),
    )
    return frame


def prepare_data(
    config: Config,
    *,
    download: bool = False,
    validate: bool = True,
    fail_on_error: bool | None = None,
) -> DataStageResult:
    """Optionally download, then load and validate the raw klines."""
    result = download_data(config) if download else None
    klines = load_raw_data(config)

    if not validate:
        return DataStageResult(klines=klines, validation=_empty_report(config, klines), download=result)

    strict = config.validation.get("fail_on_error", False) if fail_on_error is None else fail_on_error
    report = validate_klines(
        klines,
        config.interval,
        symbol=config.symbol,
        validation_config=config.validation,
        start=klines.index.min(),
        end=klines.index.max(),
        fail_on_error=strict,
        save_to=config.paths.raw_dir / f"{config.symbol}_{config.interval}.validation.json",
    )
    logger.info("%s", report.summary())
    if not report.ok:
        logger.error(
            "Validation reported %d error(s). The pipeline continues so the problems stay visible, "
            "but treat downstream results as suspect.",
            len(report.errors),
        )
    return DataStageResult(klines=klines, validation=report, download=result)


def _empty_report(config: Config, klines: pd.DataFrame) -> ValidationReport:
    return ValidationReport(
        symbol=config.symbol,
        interval=config.interval,
        rows=len(klines),
        range_start=klines.index.min() if not klines.empty else None,
        range_end=klines.index.max() if not klines.empty else None,
    )


# --------------------------------------------------------------------------- stage 3: dataset

def build_dataset_stage(config: Config, klines: pd.DataFrame) -> Dataset:
    """Build and persist the ML dataset."""
    dataset = DatasetBuilder(config).build(klines)
    dataset.save(config.paths.processed_dir)
    for name, split in dataset.splits.items():
        dist = split.class_distribution()
        logger.info(
            "  %-11s rows=%-7d %s .. %s  share_up=%s",
            name, len(split), format_timestamp(split.start)[:16], format_timestamp(split.end)[:16],
            f"{dist['share_up']:.3f}" if dist.get("share_up") is not None else "n/a",
        )
    return dataset


def load_dataset(config: Config) -> Dataset:
    """Load a previously built dataset, re-slicing splits from the current config."""
    path = config.paths.processed_dir / f"{config.symbol}_{config.interval}_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Run `python scripts/build_dataset.py` first."
        )
    dataset = Dataset.load(path, config)
    dataset.assert_no_leakage()
    logger.info("Loaded dataset: %d rows, %d features, splits %s",
                len(dataset.frame), len(dataset.feature_columns),
                {k: len(v) for k, v in dataset.splits.items()})
    return dataset


# --------------------------------------------------------------------------- full run

def run_full_pipeline(
    config: Config | None = None,
    *,
    download: bool = True,
    model_names: Sequence[str] = ALL_MODELS,
    run_cv: bool = True,
    write_plots: bool = True,
    skip_predict: bool = False,
) -> dict[str, Any]:
    """Execute the complete research pipeline and return every artefact handle.

    Parameters
    ----------
    download:
        Fetch (or resume) raw klines first.  Set ``False`` to reuse whatever is
        already on disk.
    """
    config = config or load_config()
    set_global_seed(config.random_state)
    config.paths.ensure()
    started = utc_now()
    logger.info("=== crypto-ml-predictor pipeline start (%s) ===", config.source_path)

    # 1-2. data + validation
    data_stage = prepare_data(config, download=download)
    logger.info("Validation status: %s", data_stage.validation.status())

    # 3. dataset
    dataset = build_dataset_stage(config, data_stage.klines)

    # 4. train (fits on train, selects on validation)
    training = train_models(config, dataset, model_names, run_cv=run_cv)
    logger.info("Selected model: %s", training.selected_name)

    # 5. evaluate (first and only read of the test split)
    evaluation = evaluate_models(config, training, write_plots=write_plots)
    logger.info("Test metrics for %s: %s", evaluation.selected_name, evaluation.per_model[evaluation.selected_name]["test"]["metrics"].get("roc_auc"))

    # 6. predict with the persisted artifact
    prediction_path = None
    latest = None
    if not skip_predict:
        prediction_path, latest = predict_and_save(config, data_stage.klines, name=training.selected_name)

    run_metadata = {
        "started_at": started.isoformat(),
        "finished_at": utc_now().isoformat(),
        "config_file": str(config.source_path),
        "symbol": config.symbol,
        "interval": config.interval,
        "data_range": [
            format_timestamp(dataset.frame.index.min()),
            format_timestamp(dataset.frame.index.max()),
        ],
        "dataset_rows": int(len(dataset.frame)),
        "validation_status": data_stage.validation.status(),
        "selected_model": training.selected_name,
        "models_trained": list(model_names),
        "prediction_file": str(prediction_path) if prediction_path else None,
        "latest_prediction": {
            "timestamp": format_timestamp(latest.timestamp),
            "prediction": latest.prediction,
            "probability_up": latest.probability_up,
        } if latest else None,
        "model_metadata": str(config.paths.models_dir / "model_metadata.json"),
        "summary": str(config.paths.metrics_dir / "summary.md"),
    }
    save_json(run_metadata, config.paths.reports_dir / "pipeline_run.json")
    logger.info("=== pipeline complete in %.1fs ===", (utc_now() - started).total_seconds())
    return {
        "config": config,
        "data": data_stage,
        "dataset": dataset,
        "training": training,
        "evaluation": evaluation,
        "prediction_path": prediction_path,
        "latest_prediction": latest,
        "run_metadata": run_metadata,
    }
