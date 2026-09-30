"""Pipeline layer: orchestration only.

``train`` -> ``evaluate`` -> ``predict``.  The stage boundaries matter: only
:mod:`src.pipeline.evaluate` reads the test split, and only after
:mod:`src.pipeline.train` has frozen the model choice.
"""

from src.pipeline.evaluate import EvaluationResult, evaluate_models
from src.pipeline.predict import (
    ModelBundle,
    Prediction,
    load_model,
    predict_and_save,
    predict_from_klines,
    predict_latest,
)
from src.pipeline.train import TrainedModel, TrainingResult, train_models

__all__ = [
    "EvaluationResult",
    "ModelBundle",
    "Prediction",
    "TrainedModel",
    "TrainingResult",
    "evaluate_models",
    "load_model",
    "predict_and_save",
    "predict_from_klines",
    "predict_latest",
    "train_models",
]
