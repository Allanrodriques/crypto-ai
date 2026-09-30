"""Feature layer: causal technical indicators computed from validated OHLCV."""

from src.features.feature_engineering import (
    FEATURE_DOCS,
    FEATURE_GROUPS,
    FeatureEngineer,
    assert_documented,
    build_features,
    feature_documentation_markdown,
)

__all__ = [
    "FEATURE_DOCS",
    "FEATURE_GROUPS",
    "FeatureEngineer",
    "assert_documented",
    "build_features",
    "feature_documentation_markdown",
]
