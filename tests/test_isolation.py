"""The test suite must never write into the real project.

Regression cover for a failure that destroyed real research output: the test
config fixture set ``config.paths.root`` to a temp directory, but ``Paths`` had
already resolved every entry to an absolute path under the project, so the
override was a no-op.  ``tests/test_dataset.py`` then saved a 374-row, 31-feature
fixture over the frozen ``data/processed/BTCUSDT_1h_dataset.parquet``.

The symptom is invisible at the level of any single test - each one passes -
and only shows up later, when the baseline artefact turns out to be a toy.  So
these tests assert the *paths* rather than the model output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import PROTECTED_DIRS, ROOT, _fingerprint
from src.config import Config, load_config
from src.dataset.dataset_builder import build_dataset


def test_rebase_redirects_every_output_path(config, test_root):
    """All six output paths must land under the temp root, not the project."""
    paths = config.paths
    for attr in (
        "raw_dir",
        "processed_dir",
        "predictions_dir",
        "models_dir",
        "metrics_dir",
        "backtests_dir",
    ):
        resolved = getattr(paths, attr)
        assert test_root in resolved.parents or resolved == test_root, (
            f"paths.{attr} = {resolved} is not inside the test root {test_root}"
        )
    assert paths.root == test_root


def test_production_config_is_unaffected_by_the_test_fixture(test_root):
    """Rebasing a copy must not mutate the real configuration."""
    production = load_config()
    assert production.paths.processed_dir == ROOT / "data" / "processed"
    assert production.paths.models_dir == ROOT / "models"
    assert production.paths.reports_dir == ROOT / "reports"

    rebased = production.rebase(test_root)
    assert rebased.paths.processed_dir == test_root / "data" / "processed"
    assert production.paths.processed_dir == ROOT / "data" / "processed"


def test_rebase_preserves_every_other_setting(test_root):
    """Redirection must move paths and nothing else."""
    production = load_config()
    rebased = production.rebase(test_root)
    assert rebased.symbol == production.symbol
    assert rebased.interval == production.interval
    assert rebased.random_state == production.random_state
    assert rebased.raw == production.raw
    assert rebased.features == production.features
    assert rebased.target == production.target
    assert rebased.split == production.split


def test_rebased_config_still_validates(test_root):
    """A rebased config must be a first-class config, not a crippled copy."""
    rebased = load_config().rebase(test_root)
    assert isinstance(rebased, Config)
    rebased.validate()


def test_absolute_paths_in_the_mapping_are_not_hijacked(test_root):
    """An explicitly absolute path is the caller's decision and is honoured."""
    cfg = load_config().rebase(test_root).with_overrides(
        {"paths": {"processed_dir": "/tmp/crypto-ml-explicit"}}
    )
    assert cfg.paths.processed_dir == Path("/tmp/crypto-ml-explicit")


def test_dataset_builder_writes_only_inside_the_temp_directory(config, klines, test_root):
    """The end-to-end proof: build a dataset and find every file it created."""
    before = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}

    dataset = build_dataset(config, klines)
    assert len(dataset.frame) > 0

    # The artefact really was written...
    written = list((test_root / "data" / "processed").glob("*.parquet"))
    assert written, "build_dataset produced no parquet under the test root"

    # ...and it landed only there.
    after = {name: _fingerprint(ROOT / name) for name in PROTECTED_DIRS}
    assert before == after, "build_dataset wrote into the real project"
    assert not (ROOT / "data" / "processed" / "BTCUSDT_1h_dataset.parquet").with_suffix(
        ".replay"
    ).exists()


def test_protected_project_dirs_are_not_rewritten_by_the_whole_suite(
    project_dirs_unchanged,
):
    """The session guard reports nothing changed.

    ``project_dirs_unchanged`` is session-scoped and autouse: it fingerprints
    the protected trees, and its teardown comparison is what fails the run.  By
    the time a test body runs, ``before`` is the fingerprint taken at session
    start, so asserting on it here documents that the guard is live.
    """
    assert set(project_dirs_unchanged) == set(PROTECTED_DIRS)
    for name, digest in project_dirs_unchanged.items():
        assert digest != "<absent>" or not (ROOT / name).exists()


def test_frozen_dataset_artefact_is_not_a_fixture(test_root, config, klines):
    """A cheap tripwire against the original symptom.

    A test fixture dataset is small, covers a couple of weeks, and has far
    fewer features than the real 39.  If anyone points the suite back at the
    project paths, the frozen artefact trips this check the next time it is
    inspected.
    """
    build_dataset(config, klines)
    built = next((test_root / "data" / "processed").glob("*.parquet"))
    import pandas as pd

    frame = pd.read_parquet(built)
    assert len(frame.columns) < 39, "fixture unexpectedly has production feature count"
    assert (frame.index.max() - frame.index.min()).days < 30
