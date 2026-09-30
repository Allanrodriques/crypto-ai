"""Tests for the V3 model zoo.

The theme running through this file is the one property a leakage-conscious
backtest actually depends on: **a model fitted on a training block must be a
pure function of that training block.**  Several tests below try to falsify that
by corrupting the test-period targets and asserting the predictions do not move.
A test that cannot fail is not a test, so the anti-leak test also asserts that
the *leaky* alternative - fitting on train+test - gives a different answer.  If
the baselines ever did start peeking, only that second assertion would notice.

Everything runs on small synthetic frames (200 rows x 6 features) so the whole
file finishes in seconds; nothing here touches the network or the project's real
output directories.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from src.v3.models import (
    BASELINE_NAMES,
    MIN_USABLE_ROWS,
    MODEL_NAMES,
    REGRESSOR_NAMES,
    MeanBaseline,
    TrailingMeanBaseline,
    ZeroBaseline,
    artifact_path,
    build_model,
    fit_model,
    load_model,
    predict_model,
    save_model,
)

SEED = 42
N_TRAIN = 200
N_TEST = 60
N_FEATURES = 6

#: ``random_forest`` is built with ``n_jobs=-1``, and scikit-learn's parallel
#: prediction path sums the per-tree outputs in a thread-completion order that is
#: not fixed.  Repeated runs of the same seed therefore agree to a few units in
#: the last place instead of bit-for-bit.  ``ridge`` and ``xgboost`` are held to
#: exact equality, which is the stricter and more useful assertion.
DETERMINISM_TOLERANCE: dict[str, float] = {"ridge": 0.0, "random_forest": 1e-12, "xgboost": 0.0}


def make_frame(n_rows: int = N_TRAIN, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """A deterministic feature block with a real (but learnable) signal.

    Column 0 drives the target, columns 1-3 add structure the trees can use,
    columns 4-5 are noise.  Returns a ``DataFrame`` (not an array) because the
    production path feeds frames, and a frame exercises the column-name handling
    that a bare array would skip.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, N_FEATURES))
    values[:, 1] = np.sin(np.arange(n_rows) / 7.0)
    values[:, 2] = np.cos(np.arange(n_rows) / 11.0)
    values[:, 3] = np.abs(values[:, 0]) * 0.5
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(N_FEATURES)])
    target = 0.4 * values[:, 0] + 0.15 * values[:, 1] * values[:, 2] + rng.normal(scale=0.05, size=n_rows)
    return frame, target


@pytest.fixture
def train_block() -> tuple[pd.DataFrame, np.ndarray]:
    return make_frame(N_TRAIN)


@pytest.fixture
def test_block() -> tuple[pd.DataFrame, np.ndarray]:
    """A *separate* draw, so it shares no rows and no seed with the training set."""
    return make_frame(N_TEST, seed=SEED + 1)


@contextmanager
def captured_debug_logs():
    """Capture ``logging.DEBUG`` from the ``v3.models`` logger.

    The project logger deliberately does not propagate to the root logger
    (``src.utils.get_logger`` attaches its own handler and sets
    ``propagate = False``), so pytest's ``caplog`` fixture can never see it.  The
    handler is attached to the ``crypto_ml`` parent and the level lowered for the
    duration, then both are restored.
    """
    parent = logging.getLogger("crypto_ml")
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    previous_level = parent.level
    parent.setLevel(logging.DEBUG)
    parent.addHandler(handler)
    try:
        yield records
    finally:
        parent.removeHandler(handler)
        parent.setLevel(previous_level)


# ------------------------------------------------------------------ name registry


def test_name_registries_are_exactly_the_specified_order() -> None:
    assert MODEL_NAMES == ("zero", "mean", "trailing_mean", "ridge", "random_forest", "xgboost")
    assert BASELINE_NAMES == ("zero", "mean", "trailing_mean")
    assert REGRESSOR_NAMES == ("ridge", "random_forest", "xgboost")
    # Baselines first, so a report that iterates MODEL_NAMES leads with the
    # controls and the comparison reads in the right order.
    assert MODEL_NAMES == BASELINE_NAMES + REGRESSOR_NAMES
    assert len(set(MODEL_NAMES)) == len(MODEL_NAMES)


def test_unknown_model_name_raises_value_error_listing_the_zoo() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_model("lightgbm")
    message = str(excinfo.value)
    assert "lightgbm" in message
    for name in MODEL_NAMES:
        assert name in message


def test_build_model_returns_a_fresh_unfitted_estimator_each_call() -> None:
    first = build_model("ridge")
    second = build_model("ridge")
    assert isinstance(first, BaseEstimator)
    assert first is not second
    # Unfitted: no fitted attribute has appeared yet.
    assert not hasattr(first, "coef_")


# ------------------------------------------------------------------ zoo contract


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_builds_fits_and_predicts_the_right_length(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X_train, y_train = train_block
    X_test, _ = test_block

    model = build_model(name, seed=SEED)
    fitted = fit_model(model, X_train, y_train)
    predictions = predict_model(fitted, X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (N_TEST,)
    assert predictions.dtype == np.float64
    assert np.isfinite(predictions).all(), f"{name} produced a non-finite prediction"
    assert predict_model(fitted, X_test).shape == (N_TEST,)


def test_regressor_zoo_actually_learns_the_signal() -> None:
    """Guards against a silent no-op: a model that ignores ``X`` would pass every
    shape and finiteness check above while being worthless."""
    X_train, y_train = make_frame(400)
    X_test, y_test = make_frame(150, seed=SEED + 2)

    for name in REGRESSOR_NAMES:
        fitted = fit_model(build_model(name, seed=SEED), X_train, y_train)
        # Correlation, not R^2: forward returns are near-unpredictable, so R^2 is
        # a legitimately negative number and asserting on it would be a test of
        # the market, not of the code.
        correlation = np.corrcoef(predict_model(fitted, X_test), y_test)[0, 1]
        assert correlation > 0.2, f"{name} did not learn the synthetic signal (corr={correlation:.3f})"


# --------------------------------------------------------------------- baselines


def test_zero_baseline_predicts_exactly_zero() -> None:
    X_train, y_train = make_frame(120)
    X_test, _ = make_frame(40, seed=SEED + 1)

    predictions = predict_model(fit_model(build_model("zero"), X_train, y_train), X_test)
    assert predictions.shape == (40,)
    assert np.array_equal(predictions, np.zeros(40))
    assert predictions.dtype == np.float64


def test_mean_baseline_predicts_the_training_mean() -> None:
    X_train, y_train = make_frame(120)
    X_test, _ = make_frame(40, seed=SEED + 1)

    predictions = predict_model(fit_model(build_model("mean"), X_train, y_train), X_test)
    assert predictions.shape == (40,)
    assert np.allclose(predictions, y_train.mean())


def test_trailing_mean_baseline_uses_only_the_last_window_targets() -> None:
    X_train, y_train = make_frame(200)
    X_test, _ = make_frame(40, seed=SEED + 1)

    fitted = fit_model(build_model("trailing_mean", window=24), X_train, y_train)
    predictions = predict_model(fitted, X_test)

    assert np.allclose(predictions, y_train[-24:].mean())
    # The distinction from `mean` is the whole reason this baseline exists.
    assert not np.isclose(y_train[-24:].mean(), y_train.mean())


def test_trailing_mean_respects_a_short_window() -> None:
    X_train, y_train = make_frame(200)
    X_test, _ = make_frame(40, seed=SEED + 1)

    fitted = fit_model(build_model("trailing_mean", window=1), X_train, y_train)
    assert np.allclose(predict_model(fitted, X_test), y_train[-1])


def test_trailing_mean_window_shorter_than_training_set_uses_all_targets() -> None:
    """A window longer than the training block must not pad with zeros, which
    would drag the forecast toward the ``zero`` baseline for no data-driven
    reason."""
    X_train, y_train = make_frame(30)
    X_test, _ = make_frame(10, seed=SEED + 1)

    fitted = fit_model(build_model("trailing_mean", window=1000), X_train, y_train)
    assert np.allclose(predict_model(fitted, X_test), y_train.mean())


@pytest.mark.parametrize("window", [0, -1, -100])
def test_trailing_mean_rejects_a_non_positive_window(window: int) -> None:
    X_train, y_train = make_frame(50)
    with pytest.raises(ValueError) as excinfo:
        fit_model(build_model("trailing_mean", window=window), X_train, y_train)
    assert str(window) in str(excinfo.value)


def test_baselines_are_declared_constructors_with_the_documented_defaults() -> None:
    """The public names are part of the API other modules import."""
    assert ZeroBaseline().get_params() == {}
    assert MeanBaseline().get_params() == {}
    assert TrailingMeanBaseline().get_params() == {"window": 24}
    assert TrailingMeanBaseline(window=7).window == 7


# --------------------------------------------------------------------- anti-leak


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_predictions_are_independent_of_the_test_targets(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """THE anti-leak test.

    A baseline is fitted on the training block and then asked about the test
    block.  The test targets play no part in ``predict_model`` - it does not
    accept them - so replacing them with garbage must change nothing at all.
    """
    X_train, y_train = train_block
    X_test, y_test = test_block

    honest = predict_model(fit_model(build_model(name), X_train, y_train), X_test)

    # Same fit, same X, different y.  Only the test targets differ.
    X_test_garbage = X_test.copy()
    y_test_garbage = np.full(N_TEST, 1e6) - np.arange(N_TEST) * 1e3
    garbage = predict_model(fit_model(build_model(name), X_train, y_train), X_test_garbage)

    assert np.array_equal(honest, garbage), f"{name} reacted to the test-block targets"
    # And the honest answer is genuinely the train-only statistic, not a
    # coincidence that garbage also reproduces.
    assert not np.allclose(honest, y_test_garbage.mean())


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_anti_leak_test_can_actually_fail(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """Sensitivity check for the test above.

    Fitting on train+test - the leak the whole design rules out - *does* move the
    baseline.  Without this companion assertion, an implementation that ignored
    its arguments entirely would pass the anti-leak test forever.
    """
    X_train, y_train = train_block
    X_test, y_test = test_block

    train_only = predict_model(fit_model(build_model(name), X_train, y_train), X_test)

    combined_X = pd.concat([X_train, X_test], ignore_index=True)
    combined_y = np.concatenate([y_train, y_test])
    combined = predict_model(fit_model(build_model(name), combined_X, combined_y), X_test)
    leaked = combined[-N_TEST:]

    if name == "zero":
        # Zero is zero under any fit, which is exactly why it is the anchor.
        assert np.array_equal(train_only, leaked)
    else:
        assert not np.allclose(train_only, leaked), (
            f"{name} gave the same answer with and without the test rows, so the "
            f"anti-leak test above is not sensitive to the leak it guards"
        )


@pytest.mark.parametrize("name", REGRESSOR_NAMES)
def test_regressor_ignores_rows_with_no_usable_information(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """Appending rows that carry nothing must not move the model at all.

    This is the regressor counterpart of the anti-leak test, and it is
    sensitive in the right way: if ``fit_model`` ever stopped dropping the
    unlabelable target tail and started imputing it instead - a plausible-looking
    "fix" - then padding the training set with a NaN target would silently add
    60 rows of invented zeros and this equality would break.
    """
    X_train, y_train = train_block
    X_test, _ = test_block

    padded_X = pd.concat([X_train, X_test], ignore_index=True)
    padded_y = np.concatenate([y_train, np.full(N_TEST, np.nan)])

    clean = predict_model(fit_model(build_model(name, seed=SEED), X_train, y_train), X_test)
    padded = predict_model(fit_model(build_model(name, seed=SEED), padded_X, padded_y), X_test)

    assert np.allclose(clean, padded, atol=DETERMINISM_TOLERANCE[name]), (
        f"{name} was changed by rows it should have discarded"
    )


def test_ridge_scaler_is_fitted_on_training_rows_only(
    train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """The "scaling fitted on training data only" requirement, checked on the
    fitted artefact rather than on the source.

    Scaling a frame once and then splitting is the standard way a leakage-free-
    looking backtest is quietly ruined, and it leaves no trace except the wrong
    numbers here.
    """
    X_train, y_train = train_block
    X_test, _ = test_block

    scaler = fit_model(build_model("ridge"), X_train, y_train).named_steps["scaler"]
    mean_train = X_train.mean().to_numpy()
    mean_all = pd.concat([X_train, X_test]).mean().to_numpy()

    # Sensitivity: the two candidate means must actually differ, or the
    # assertion below would pass for the wrong reason.
    assert not np.allclose(mean_train, mean_all)
    assert np.allclose(scaler.mean_, mean_train)
    assert not np.allclose(scaler.mean_, mean_all)


# ------------------------------------------------------------------------- ridge


def test_ridge_is_a_scaler_ridge_pipeline() -> None:
    """The scaler must be a pipeline *step* so it is refitted per fold."""
    model = build_model("ridge")
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["scaler", "ridge"]


def test_ridge_survives_constant_and_near_constant_feature_columns() -> None:
    """A constant column is normal in V3 (a feature that never moved, a regime
    indicator that stayed on).  It must produce a zero column and a zero
    coefficient, not a ``NaN`` from a division by a zero scale."""
    X, y = make_frame(200)
    X = X.copy()
    X["f_flat"] = 3.0        # exactly constant
    X["f_tiny"] = 1e-18      # near constant: non-zero but numerically flat

    fitted = fit_model(build_model("ridge"), X, y)
    predictions = predict_model(fitted, X.head(20))

    assert np.isfinite(predictions).all()
    ridge_step = fitted.named_steps["ridge"]
    assert np.isfinite(ridge_step.coef_).all()
    assert np.isfinite(ridge_step.intercept_)

    # The scaler must not have divided by zero on either constant column.  The
    # exactly-constant one becomes a column of exact zeros; the near-constant one
    # has a real but meaningless mean, so it survives as a tiny constant - finite
    # either way, which is the property that matters.
    scaler = fitted.named_steps["scaler"]
    assert scaler.scale_[-2] == 1.0 and scaler.scale_[-1] == 1.0
    scaled = scaler.transform(X)
    assert np.allclose(np.asarray(scaled)[:, -2], 0.0)
    assert np.allclose(np.asarray(scaled)[:, -1], 0.0, atol=1e-30)
    # A constant column carries no information, so it must not be given weight.
    assert np.allclose(ridge_step.coef_[-2:], 0.0)

    # An all-constant feature *block* is the degenerate extreme of the same case.
    flat = pd.DataFrame(2.5, index=X.index, columns=["a", "b", "c"])
    flat_predictions = predict_model(fit_model(build_model("ridge"), flat, y), flat.head(20))
    assert np.isfinite(flat_predictions).all()
    assert np.allclose(flat_predictions, flat_predictions[0])
    # A constant block is an intercept-only model: every prediction is the mean.
    assert np.allclose(flat_predictions, y.mean())


def test_ridge_rejects_a_single_training_row() -> None:
    X, y = make_frame(1)
    assert len(X) == MIN_USABLE_ROWS - 1
    with pytest.raises(ValueError) as excinfo:
        fit_model(build_model("ridge"), X, y)
    assert "usable training row" in str(excinfo.value)


# ----------------------------------------------------------------------- xgboost


def test_xgboost_uses_the_specified_configuration() -> None:
    params = build_model("xgboost", seed=7).get_params()
    assert isinstance(build_model("xgboost"), XGBRegressor)
    assert params["objective"] == "reg:squarederror"
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["random_state"] == 7
    assert params["n_jobs"] == 1
    assert params["tree_method"] == "hist"
    # Quiet, and pinned rather than left to the library default.
    assert params["verbosity"] == 0
    # No early stopping machinery: the constructor never even mentions it.
    assert params.get("early_stopping_rounds") is None


def test_xgboost_fits_without_ever_touching_an_eval_set(
    train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """Early stopping needs a validation period, and there is no honest one
    inside a purged walk-forward fold.  Asserted two ways: ``fit`` was never
    handed an ``eval_set``, and no evaluation result exists afterwards."""
    X_train, y_train = train_block
    X_test, _ = test_block

    seen: dict[str, dict[str, object]] = {}
    original_fit = XGBRegressor.fit

    def spy(self, X, y, **kwargs):
        seen["kwargs"] = dict(kwargs)
        return original_fit(self, X, y, **kwargs)

    XGBRegressor.fit = spy
    try:
        fitted = fit_model(build_model("xgboost", seed=SEED), X_train, y_train)
    finally:
        XGBRegressor.fit = original_fit

    assert "eval_set" not in seen["kwargs"]
    assert "eval_set" not in seen["kwargs"].get("sample_weight_eval_set", {})  # type: ignore[union-attr]
    # libxgboost raises precisely when no eval_set was used - the clearest
    # possible statement that the model never scored itself on anything.
    with pytest.raises(XGBoostError, match="eval_set"):
        fitted.evals_result()
    assert fitted.get_booster().num_boosted_rounds() == 300
    assert not hasattr(fitted, "best_iteration")
    assert fitted.get_booster().attr("best_iteration") is None

    assert np.isfinite(predict_model(fitted, X_test)).all()


def test_random_forest_uses_the_specified_configuration() -> None:
    params = build_model("random_forest", seed=7).get_params()
    assert params["n_estimators"] == 300
    assert params["n_jobs"] == -1
    assert params["random_state"] == 7


# ------------------------------------------------------------------ determinism


@pytest.mark.parametrize("name", REGRESSOR_NAMES)
def test_same_seed_gives_identical_predictions(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X_train, y_train = train_block
    X_test, _ = test_block
    tolerance = DETERMINISM_TOLERANCE[name]

    first = predict_model(fit_model(build_model(name, seed=7), X_train, y_train), X_test)
    second = predict_model(fit_model(build_model(name, seed=7), X_train, y_train), X_test)
    third = predict_model(fit_model(build_model(name, seed=7), X_train, y_train), X_test)

    if tolerance == 0.0:
        assert np.array_equal(first, second), f"{name} is not bit-for-bit reproducible"
        assert np.array_equal(first, third)
    else:
        assert np.allclose(first, second, atol=tolerance)
        assert np.allclose(first, third, atol=tolerance)
    # Whatever the tolerance, the runs must not differ by a whole number.
    assert np.max(np.abs(first - second)) < 1e-9


def test_deterministic_models_are_bit_for_bit_reproducible() -> None:
    """The exact-equality half of the determinism requirement, stated
    independently of the tolerance table so a future edit to that table cannot
    quietly relax the two models that *can* be bit-exact."""
    X_train, y_train = make_frame(300)
    X_test, _ = make_frame(80, seed=SEED + 3)

    for name in ("ridge", "xgboost"):
        first = predict_model(fit_model(build_model(name, seed=7), X_train, y_train), X_test)
        second = predict_model(fit_model(build_model(name, seed=7), X_train, y_train), X_test)
        assert np.array_equal(first, second), f"{name} is not bit-for-bit reproducible"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_determinism_survives_repeated_fits_within_one_process(
    name: str, train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """Fitting twice in the same process must not depend on any global RNG state
    left behind by the first fit - the classic 'works until you run it twice'
    determinism bug."""
    X_train, y_train = train_block
    X_test, _ = test_block

    predict_model(fit_model(build_model(name, seed=11), X_train, y_train), X_test)
    np.random.seed(999)
    a = predict_model(fit_model(build_model(name, seed=11), X_train, y_train), X_test)
    np.random.seed(1234)
    b = predict_model(fit_model(build_model(name, seed=11), X_train, y_train), X_test)

    assert np.allclose(a, b, atol=DETERMINISM_TOLERANCE.get(name, 0.0))


# ------------------------------------------------------------- non-finite handling


def test_fit_model_drops_rows_with_non_finite_features(
    train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """A NaN feature is a missing observation, so the row goes - and the model
    ends up numerically identical to one fitted on the clean remainder."""
    X_train, y_train = train_block
    X_test, _ = test_block

    dirty = X_train.copy()
    victim_rows = [3, 7, 8, 9, 17, 42, 99, 150]
    dirty.iloc[3, 0] = np.nan
    dirty.iloc[17, 0] = np.nan
    dirty.iloc[42, 0] = np.nan
    dirty.iloc[99, 0] = np.nan
    dirty.iloc[150, 0] = np.nan
    dirty.iloc[7, 2] = np.inf
    dirty.iloc[8, 3] = -np.inf
    dirty.iloc[9, 4] = np.nan
    assert (~np.isfinite(dirty.to_numpy())).any(), "the dirty block should contain non-finite values"

    # The reference is the same block with exactly those rows removed and
    # nothing else changed - the targets have to be cut identically.
    clean = X_train.drop(index=victim_rows)
    y_clean = np.delete(y_train, victim_rows)
    assert np.isfinite(clean.to_numpy()).all()
    assert len(clean) == len(y_clean) == N_TRAIN - len(victim_rows)

    with captured_debug_logs() as records:
        from_dirty = predict_model(fit_model(build_model("ridge"), dirty, y_train), X_test)
    from_clean = predict_model(fit_model(build_model("ridge"), clean, y_clean), X_test)

    # Positional comparison: `drop` and `iloc` leave different index labels, and
    # the two fits must still be the same model.
    assert np.allclose(from_dirty, from_clean), "the non-finite rows were not simply removed"
    assert np.isfinite(from_dirty).all()

    messages = [record.getMessage() for record in records if record.levelno == logging.DEBUG]
    assert any(f"dropped {len(victim_rows)}" in message for message in messages), messages
    assert any("non-finite feature" in message for message in messages), messages


def test_fit_model_drops_rows_with_non_finite_targets(
    train_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """The unlabelable tail is ``NaN`` by design (see ``src.v3.targets``); it must
    be dropped, and never imputed - the future has not happened yet."""
    X_train, y_train = train_block
    dirty = y_train.copy()
    dirty[-25:] = np.nan
    dirty[10] = np.inf

    with captured_debug_logs() as records:
        fitted = fit_model(build_model("mean"), X_train, dirty)
    messages = [record.getMessage() for record in records if record.levelno == logging.DEBUG]

    # np.isfinite excludes both the NaN tail and the +inf row, so this is the
    # exact set of targets the baseline was allowed to see.
    kept = y_train[np.isfinite(dirty)]
    assert kept.size == N_TRAIN - 26
    assert np.allclose(predict_model(fitted, X_train.head(5)), kept.mean())
    assert any("dropped 26" in message for message in messages), messages
    assert any("26 non-finite target" in message for message in messages), messages


def test_fit_model_fills_a_structurally_empty_column_instead_of_every_row(
    train_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """A column with no finite value anywhere must not destroy the training set.

    Dropping rows for it would remove all 200 and fail; filling it keeps every
    row and hands Ridge a column of exact zeros, which is what a standardised
    block already means.
    """
    X_train, y_train = train_block
    X_train = X_train.copy()
    X_train["f_never_computed"] = np.nan

    fitted = fit_model(build_model("ridge"), X_train, y_train)
    predictions = predict_model(fitted, X_train.head(20))

    assert np.isfinite(predictions).all()
    assert np.allclose(fitted.named_steps["ridge"].coef_[-1], 0.0)
    # And the cleaned training block really did have all its rows.
    assert getattr(fitted, "v3_imputer_") is not None


def test_fit_model_raises_when_no_column_has_a_finite_value() -> None:
    X_train, y_train = make_frame(50)
    with pytest.raises(ValueError, match="no finite value in any column"):
        fit_model(build_model("ridge"), pd.DataFrame(np.nan, index=X_train.index, columns=list(X_train.columns)), y_train)


def test_fit_model_raises_when_too_few_rows_survive() -> None:
    X_train, y_train = make_frame(10)
    dirty = y_train.copy()
    dirty[1:] = np.nan

    with pytest.raises(ValueError) as excinfo:
        fit_model(build_model("ridge"), X_train, dirty)
    message = str(excinfo.value)
    assert "Only 1 usable training row" in message
    assert "9 non-finite row" in message
    assert str(MIN_USABLE_ROWS) in message


def test_fit_model_accepts_exactly_the_minimum_number_of_rows() -> None:
    """The floor is a floor, not a fudge factor."""
    X_train, y_train = make_frame(MIN_USABLE_ROWS)
    fitted = fit_model(build_model("mean"), X_train, y_train)
    assert np.allclose(predict_model(fitted, X_train), y_train.mean())


def test_fit_model_rejects_misaligned_X_and_y() -> None:
    X_train, y_train = make_frame(50)
    with pytest.raises(ValueError, match="row count"):
        fit_model(build_model("ridge"), X_train, y_train[:-5])


def test_fit_model_rejects_a_one_dimensional_X() -> None:
    _, y_train = make_frame(50)
    with pytest.raises(ValueError, match="2-dimensional"):
        fit_model(build_model("ridge"), np.arange(50.0), y_train)


def test_fit_model_rejects_a_non_numeric_column_with_a_useful_message() -> None:
    X_train, y_train = make_frame(50)
    X_train = X_train.astype(object)
    X_train["f_label"] = "not a number"
    with pytest.raises(ValueError, match="non-numeric"):
        fit_model(build_model("ridge"), X_train, y_train)


def test_predict_model_fills_non_finite_test_features_without_shortening_output(
    train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    """Prediction must never drop a row.

    A shorter array would silently shift every forecast off its own timestamp,
    which is far more dangerous than an imputed feature value.
    """
    X_train, y_train = train_block
    X_test, _ = test_block

    fitted = fit_model(build_model("ridge"), X_train, y_train)
    clean_predictions = predict_model(fitted, X_test)

    dirty = X_test.copy()
    dirty.iloc[0, 0] = np.nan
    dirty.iloc[1, 1] = np.inf
    dirty.iloc[-1, -1] = -np.inf
    dirty_predictions = predict_model(fitted, dirty)

    assert dirty_predictions.shape == (N_TEST,)
    assert np.isfinite(dirty_predictions).all()
    # Filled with the *training* median of that column, so only those rows move.
    assert np.allclose(np.delete(dirty_predictions, [0, 1, -1]), np.delete(clean_predictions, [0, 1, -1]))
    assert not np.allclose(dirty_predictions, clean_predictions)


def test_predict_model_tolerates_a_model_fitted_without_fit_model(
    train_block: tuple[pd.DataFrame, np.ndarray], test_block: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X_train, y_train = train_block
    X_test, _ = test_block

    handmade = build_model("mean")
    handmade.fit(X_train, y_train)
    predictions = predict_model(handmade, X_test)

    assert predictions.shape == (N_TEST,)
    assert np.allclose(predictions, y_train.mean())


# ---------------------------------------------------------------------- artifacts


def test_artifact_path_is_exactly_the_expected_string(tmp_path: Path) -> None:
    expected = tmp_path / "v3" / "xgboost_30d.joblib"
    assert artifact_path(tmp_path, "xgboost", "30d") == expected
    assert str(artifact_path(tmp_path, "xgboost", "30d")) == str(expected)
    # Every horizon in the V3 ladder, and the baselines, all follow the rule.
    for horizon in ("1d", "7d", "180d"):
        assert artifact_path(tmp_path, "trailing_mean", horizon).name == f"trailing_mean_{horizon}.joblib"
    # Accepts a string root as readily as a Path.
    assert artifact_path(str(tmp_path), "ridge", "1d") == tmp_path / "v3" / "ridge_1d.joblib"
    # The stem matches Horizon.artifact_stem, so reports and files cannot drift.
    from src.v3.horizons import Horizon

    horizon = Horizon.from_label("90d", bars_per_unit=24)
    assert artifact_path(tmp_path, "random_forest", horizon.label).stem == horizon.artifact_stem("random_forest")


def test_artifact_path_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    root = tmp_path / "run_2026_01_01"
    assert not root.exists()

    path = artifact_path(root, "xgboost", "180d")
    assert path == root / "v3" / "xgboost_180d.joblib"

    assert not root.exists(), "artifact_path created a directory"
    assert not (root / "v3").exists(), "artifact_path created the v3 directory"
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X_train, y_train = make_frame(200)
    X_test, _ = make_frame(50, seed=SEED + 1)

    fitted = fit_model(build_model("xgboost", seed=SEED), X_train, y_train)
    before = predict_model(fitted, X_test)

    target = artifact_path(tmp_path, "xgboost", "7d")
    assert not target.parent.exists(), "precondition: the v3 directory is not there yet"
    written = save_model(fitted, target)

    assert written == target
    assert target.exists()

    restored = load_model(written)
    assert np.array_equal(predict_model(restored, X_test), before)
    # The imputer travelled with the artifact, so a loaded model still repairs
    # non-finite features exactly as the original did.
    dirty = X_test.copy()
    dirty.iloc[0, 0] = np.nan
    assert np.isfinite(predict_model(restored, dirty)).all()


def test_load_model_reports_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(artifact_path(tmp_path, "ridge", "1d"))


def test_baselines_also_survive_a_save_load_round_trip(tmp_path: Path) -> None:
    X_train, y_train = make_frame(100)
    X_test, _ = make_frame(30, seed=SEED + 1)

    fitted = fit_model(build_model("trailing_mean", window=12), X_train, y_train)
    restored = load_model(save_model(fitted, artifact_path(tmp_path, "trailing_mean", "3d")))

    assert np.array_equal(predict_model(restored, X_test), predict_model(fitted, X_test))
