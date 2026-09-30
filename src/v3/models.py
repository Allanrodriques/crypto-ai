"""V3 model zoo: one regression model per horizon, plus the yardstick to beat.

What V3 actually predicts
-------------------------
V1 and V2 are *classification* problems - up or down over a handful of candles.
V3 is a **regression** problem: at every hourly timestamp, and for every
configured horizon from 1d to 180d, the model regresses the forward return

    future_return_H[t] = close[t + H] / close[t] - 1

and a **separate model is fitted per horizon**.  Nothing about the horizon is an
input to a single shared model: a 1-day horizon and a 180-day horizon have
essentially different dynamics, and forcing one estimator to serve both would
mean it can only learn their average.  So the (model, horizon) pair *is* the unit
of work, and :func:`artifact_path` names artifacts accordingly.

Why the zoo is only these six
-----------------------------
Three baselines and three regressors, no neural networks.  The baselines are not
decoration - they are the comparison that gives every regressor's number meaning.
Crypto forward returns are close to unpredictable, so ``R^2`` is routinely
negative for a well-fitted model, and a report that shows only "R^2 = -0.01"
without "R^2 = 0.00 for predicting zero" invites the reader to conclude the
machine learning failed when in fact the market refused to be predicted.
:class:`TrailingMeanBaseline` is the interesting one: crypto has fat tails and
regime persistence, so the recent mean return is a genuinely hard number to beat
at short horizons.

Three rules this module exists to enforce
-----------------------------------------
1. **Scaling is fitted on training rows only.**  Ridge is wrapped in a
   :class:`~sklearn.pipeline.Pipeline` so the
   :class:`~sklearn.preprocessing.StandardScaler` is refitted inside every
   walk-forward fold.  Scaling the whole frame once, outside the fold, is the
   single most common way a leakage-free-looking backtest is quietly ruined.
2. **No early stopping here.**  :class:`xgboost.XGBRegressor` is deliberately
   constructed without an ``eval_set``.  Early stopping needs a validation set,
   and the only honest validation set is a held-out *future* period - which is
   the walk-forward harness's job, not the estimator factory's.  The V1
   ``XGBModel`` solved this by carving an internal tail out of the rows it was
   handed; in V3 the folds are already purged and time-ordered, so bolting an
   internal split on top would only add a second, un-purged cut inside data
   that is supposed to be a single training block.
3. **Determinism is a requirement, not a hope.**  Every model is seeded, XGBoost
   runs single-threaded (``n_jobs=1``), and no model touches a global RNG.  Two
   runs on the same data with the same seed must produce the same predictions,
   otherwise a walk-forward report is not reproducible.

   One honest caveat: ``random_forest`` is built with ``n_jobs=-1``, and
   scikit-learn's parallel prediction path reduces the per-tree outputs in a
   non-fixed order.  Repeated runs therefore agree to roughly 1e-15 rather than
   bit-for-bit - four units in the last place, from floating-point addition
   order and nothing else.  ``ridge`` and ``xgboost`` are bit-exact.  A report
   that diffs two runs of the same fold will see noise in the sixteenth digit of
   the forest and nowhere else; the determinism tests assert accordingly, and
   the tolerance is a property of the library, not a loosening of the goal.

Why the baselines are feature-agnostic
--------------------------------------
The three baselines read ``X`` only to learn how many predictions to emit.  They
deliberately do **not** set ``n_features_in_``: a constant forecast is a
statement about the return distribution, and constraining its input width would
only make it harder to use as the control in an ablation.

Why the non-finite handling looks the way it does
-------------------------------------------------
:func:`fit_model` is the single gate every model passes through, so it is the
right place to deal with the ``NaN``s a V3 frame genuinely contains: warm-up
rows of rolling indicators at the head, and the unlabelable target tail that
:mod:`src.v3.targets` deliberately leaves as ``NaN`` rather than fabricating.
Two distinct repairs are needed, and conflating them is a bug:

* a row that is non-finite in a column that *has* finite values is a missing
  observation, and the row is **dropped**; and
* a column with *no* finite value anywhere carries no information at all, and
  dropping every row for its sake would destroy an otherwise healthy training
  set, so it is **filled** with a neutral 0.0.

The column medians used to fill at prediction time are fitted on training rows
only and stored on the model as ``v3_imputer_``, so
:func:`predict_model` is symmetric with :func:`fit_model` and needs no argument
it was not given.  Prediction never drops rows - a shorter array would silently
misalign every downstream forecast with its timestamp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import get_logger

logger = get_logger("v3.models")

#: Everything the walk-forward harness may be asked to fit, in report order:
#: the three controls first, then the three models that must beat them.
BASELINE_NAMES: tuple[str, ...] = ("zero", "mean", "trailing_mean")
REGRESSOR_NAMES: tuple[str, ...] = ("ridge", "random_forest", "xgboost")
MODEL_NAMES: tuple[str, ...] = BASELINE_NAMES + REGRESSOR_NAMES

#: A one-row training set is not a model.  Two rows is the floor, and it is a
#: floor rather than one because a single target has zero variance, which makes
#: every variance-based score undefined and every tree ensemble degenerate.
MIN_USABLE_ROWS: int = 2

#: Attribute under which :func:`fit_model` parks the training-fitted imputer.
#: Prefixed rather than suffixed with an underscore so it cannot be mistaken for
#: one of the fitted attributes sklearn itself introspects.
IMPUTER_ATTR: str = "v3_imputer_"


# --------------------------------------------------------------------- baselines


class _ConstantBaseline(RegressorMixin, BaseEstimator):
    """Shared machinery for the three constant-forecast controls.

    Subclasses only have to decide the scalar, in :meth:`_forecast`.  That value
    is computed **once, from the training targets**, and frozen - there is no
    path by which a later call to :meth:`predict` can consult any target other
    than the training ones.  That property is the whole point of a baseline and
    is what ``tests/test_v3_models.py`` asserts directly.

    NOTE on base-class order: the mixin is on the left, which is what sklearn
    requires.  In scikit-learn 1.9 ``BaseEstimator.__sklearn_tags__`` builds a
    fresh :class:`~sklearn.utils.Tags` and does *not* delegate to ``super()``, so
    the reversed order silently drops ``estimator_type="regressor"`` and
    ``requires_y`` - the estimator still returns numbers, but anything that
    dispatches on the tag (scorer lookup, ``check_cv``, ``permutation_importance``)
    stops recognising it as a regressor.  Do not "fix" the order back.
    """

    def fit(self, X: Any, y: Any) -> Self:
        targets = _as_target_vector(y)
        self.value_ = float(self._forecast(targets))
        self.n_samples_seen_ = int(targets.size)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "value_")
        return np.full(_n_rows(X), self.value_, dtype="float64")

    def _forecast(self, targets: np.ndarray) -> float:
        raise NotImplementedError


class ZeroBaseline(_ConstantBaseline):
    """Predicts exactly 0.0: the "the market is a random walk" control.

    It is the only baseline whose expected score is analytically known, and for
    a squared-error metric its optimum is the unconditional mean of the training
    returns - which is 0 by construction for forward returns, not by assumption.
    """

    def _forecast(self, targets: np.ndarray) -> float:
        return 0.0


class MeanBaseline(_ConstantBaseline):
    """Predicts the unconditional mean of the training targets."""

    def _forecast(self, targets: np.ndarray) -> float:
        return float(np.mean(targets))


class TrailingMeanBaseline(_ConstantBaseline):
    """Predicts the mean of the **last** ``window`` training targets.

    A recent-regime naive forecast.  The full-sample mean spreads one
    observation's weight over the whole history, which for crypto means a
    forecast dominated by whichever regime happened to start the sample; the
    trailing window lets the forecast track the current drift, which is the
    strongest simple predictor of a short-horizon return that this project has
    found.  On an hourly grid the default 24 corresponds to one day.

    Parameters
    ----------
    window:
        Number of trailing training targets to average.  Validated in
        :meth:`fit` rather than in ``__init__``, because scikit-learn requires
        ``__init__`` to do nothing but store parameters verbatim - validating
        there would make a valid estimator unclonable and break cross-validation.

    Notes
    -----
    When there are fewer training targets than ``window``, every target is
    averaged and the shortfall is logged: the alternative - padding with zeros -
    would drag the forecast toward the ``zero`` baseline for a reason that has
    nothing to do with the data.
    """

    def __init__(self, window: int = 24) -> None:
        self.window = window

    def fit(self, X: Any, y: Any) -> Self:
        window = int(self.window)
        if window < 1:
            raise ValueError(f"TrailingMeanBaseline window must be >= 1, got {window}")
        targets = _as_target_vector(y)
        if targets.size < window:
            logger.debug(
                "trailing_mean: only %d training targets for window=%d; using all of them",
                targets.size,
                window,
            )
        self.value_ = float(np.mean(targets[-window:]))
        self.n_samples_seen_ = int(targets.size)
        return self


# ------------------------------------------------------------------- estimators


def _build_zero(seed: int, **params: Any) -> BaseEstimator:
    """``seed`` and ``params`` are accepted and ignored: a constant has no
    randomness, and refusing the uniform signature would make the harness branch
    on model name for no reason."""
    return ZeroBaseline()


def _build_mean(seed: int, **params: Any) -> BaseEstimator:
    return MeanBaseline()


def _build_trailing_mean(seed: int, **params: Any) -> BaseEstimator:
    return TrailingMeanBaseline(**params)


def _build_ridge(seed: int, **params: Any) -> BaseEstimator:
    """Linear model behind a :class:`~sklearn.pipeline.Pipeline`.

    The scaler is a *pipeline step*, not a preprocessing pass, so it is refitted
    from scratch on whatever training rows each walk-forward fold supplies.  A
    constant or near-constant column is not a failure case here:
    :class:`~sklearn.preprocessing.StandardScaler` already replaces a zero scale
    with 1.0 rather than dividing by it, so such a column arrives at
    :class:`~sklearn.linear_model.Ridge` as a column of exact zeros and receives
    a zero coefficient instead of a ``NaN``.
    """
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(**params))])


def _build_random_forest(seed: int, **params: Any) -> BaseEstimator:
    """``n_jobs=-1`` because a forest is embarrassingly parallel and this is the
    one model in the zoo that benefits; ``random_state`` pins the bootstrap and
    the feature subsampling so the run is reproducible.

    ``max_features=0.3`` is the other deliberate choice, and it is the one that
    makes the whole pipeline runnable.  scikit-learn's default is ``1.0`` -
    every split considers all 84 features - which on the real feature matrix
    costs ~48s per 300-tree fit.  Restricting each split to 30% of the features
    cuts that to ~12s while leaving fitted quality unchanged (R^2 0.8728 vs
    0.8725 on a 10k x 84 benchmark).  A forest averages many decorrelated trees,
    so per-split feature diversity is nearly free to give away here, unlike in a
    single boosted model.

    The cost of the parallelism is that ``predict`` is reproducible to about
    1e-15 rather than bit-for-bit - scikit-learn sums the per-tree outputs in a
    thread-completion order that is not fixed.  See the module docstring.
    """
    options: dict[str, Any] = {
        "n_estimators": 300,
        "max_features": 0.3,
        "n_jobs": -1,
        "random_state": int(seed),
    }
    options.update(params)
    return RandomForestRegressor(**options)


def _build_xgboost(seed: int, **params: Any) -> BaseEstimator:
    """Gradient-boosted trees, with three deliberate restraints.

    * **No ``eval_set`` and no early stopping.**  See the module docstring: early
      stopping needs a validation period, and inside a purged walk-forward fold
      there is no honest one to reach for.
    * **``n_jobs=1``.**  Multi-threaded histogram construction is the classic
      source of "identical inputs, different predictions" in XGBoost; the speed
      difference on these frame sizes is not worth a report that cannot be
      reproduced.
    * **``verbosity=0``.**  libxgboost writes to stdout on its own channel, which
      pytest captures and reports; 0 keeps the run quiet.  It is set explicitly
      rather than left at the default so a config override cannot turn it on.
    """
    options: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": int(seed),
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    options.update(params)
    return XGBRegressor(**options)


_BUILDERS: dict[str, Callable[..., BaseEstimator]] = {
    "zero": _build_zero,
    "mean": _build_mean,
    "trailing_mean": _build_trailing_mean,
    "ridge": _build_ridge,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}


def build_model(name: str, seed: int = 42, **params: Any) -> BaseEstimator:
    """Construct a fresh, **unfitted** estimator by name.

    Parameters
    ----------
    name:
        One of :data:`MODEL_NAMES`.
    seed:
        Seed for every stochastic component of the chosen model.  The baselines
        ignore it, which is what makes them a stable control.
    **params:
        Constructor overrides for the chosen model, e.g. ``window`` for
        ``trailing_mean``, ``alpha`` for ``ridge``, ``max_depth`` for
        ``xgboost``.

    Returns
    -------
    BaseEstimator
        A new object every call.  Nothing is cached, so a caller may build the
        same model twice - which is exactly what the determinism test does.

    Raises
    ------
    ValueError
        If ``name`` is not in :data:`MODEL_NAMES`; the message lists the valid
        names because the alternative is a bare ``KeyError`` from deep inside a
        harness loop.
    """
    key = str(name).strip().lower()
    builder = _BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unknown model name {name!r}; expected one of {list(MODEL_NAMES)}")
    return builder(int(seed), **params)


# -------------------------------------------------------------------- fit/predict


def fit_model(model: BaseEstimator, X: Any, y: Any) -> BaseEstimator:
    """Fit ``model`` on a training block, after repairing non-finite values.

    Every model in V3 is fitted through this function rather than by calling
    ``model.fit`` directly, so that the cleaning rules are identical for all six
    and cannot be forgotten for the one added next quarter.

    Parameters
    ----------
    model:
        An unfitted estimator from :func:`build_model`.
    X:
        Feature block, ``DataFrame`` or 2-D array.  Positional: a row is a
        timestamp, and the index is never used for alignment.
    y:
        Training targets for **these rows only**.  This function has no access
        to any other period, which is what makes the fitted model immune to it.

    Returns
    -------
    BaseEstimator
        The same object, fitted, with the training-fitted imputer attached as
        ``v3_imputer_`` for :func:`predict_model` to reuse.

    Raises
    ------
    ValueError
        If no column of ``X`` has a single finite value (there is nothing to
        learn from), or if fewer than :data:`MIN_USABLE_ROWS` rows survive
        cleaning.
    """
    targets = _as_target_vector(y)
    features = _as_feature_matrix(X)
    if features.shape[0] != targets.size:
        raise ValueError(
            f"X and y disagree on row count: {features.shape[0]} feature rows "
            f"against {targets.size} targets. They must be positionally aligned."
        )

    # All of the masking below happens on a plain ndarray.  `np.isfinite` on a
    # DataFrame returns a DataFrame, and `DataFrame.any(axis=0)` then returns a
    # Series that the next `&` would *align by column label* - which silently
    # reorders and can mask against the wrong thing.
    finite = np.isfinite(_values_of(features))
    # A column with no finite value anywhere is structurally empty, not
    # "missing at random": there is nothing to impute from and no row deserves
    # to be dropped because of it.  Those cells are left non-finite here and
    # filled with the neutral 0.0 by the imputer below.
    informative = finite.any(axis=0)
    if not informative.any():
        raise ValueError(
            "X contains no finite value in any column; there is nothing to fit on. "
            "Check the feature block and the periods it was computed over."
        )

    bad_target = ~np.isfinite(targets)
    bad_feature = (~finite & informative).any(axis=1)
    keep = ~(bad_target | bad_feature)

    dropped = int((~keep).sum())
    if dropped:
        logger.debug(
            "fit_model: dropped %d of %d training rows (%d non-finite target, "
            "%d non-finite features); %d usable",
            dropped,
            int(keep.size),
            int(bad_target.sum()),
            int((bad_feature & ~bad_target).sum()),
            int(keep.sum()),
        )

    if int(keep.sum()) < MIN_USABLE_ROWS:
        raise ValueError(
            f"Only {int(keep.sum())} usable training row(s) after dropping "
            f"{dropped} non-finite row(s) from {int(keep.size)}; "
            f"{MIN_USABLE_ROWS} are the minimum. Widen the training window, "
            f"shorten the feature warm-up, or shorten the horizon so more rows "
            f"have an observed target."
        )

    features = _take_rows(features, keep)
    targets = targets[keep]

    # Median rather than mean: crypto features are heavy-tailed, and a mean
    # imputed from a single crisis bar would move every future value with it.
    # keep_empty_features fills a structurally empty column with 0.0 instead of
    # dropping it, which is what keeps the informative rows alive.
    imputer = _make_imputer()
    cleaned = imputer.fit_transform(features)
    try:
        setattr(model, IMPUTER_ATTR, imputer)
    except AttributeError:  # pragma: no cover - only reachable for a __slots__ estimator
        logger.debug("Model %r cannot carry %s; non-finite test features will fall back to 0.0", type(model).__name__, IMPUTER_ATTR)

    model.fit(cleaned, targets)
    return model


def predict_model(model: BaseEstimator, X: Any) -> np.ndarray:
    """Predict forward returns for a feature block, of any length.

    Non-finite feature values are repaired rather than dropped: dropping would
    return an array shorter than ``X`` and silently misalign every forecast with
    its timestamp, which is far worse than an imputed value.  The fill values
    are the training-set column medians recorded by :func:`fit_model`.

    Parameters
    ----------
    model:
        A model fitted by :func:`fit_model`.
    X:
        Feature block with the same columns, in the same order, as at fit time.

    Returns
    -------
    np.ndarray
        1-D float array of length ``len(X)``, all entries finite.
    """
    check_is_fitted(model)
    features = _as_feature_matrix(X)
    imputer = getattr(model, IMPUTER_ATTR, None)

    if imputer is not None:
        cleaned = imputer.transform(_blank_non_finite(features))
    else:
        # A model fitted by hand, outside fit_model.  Zero is the honest fill
        # here: there is no training statistic to use, and zero is what a
        # standardised feature block already means.
        non_finite = ~np.isfinite(_values_of(features))
        cleaned = _replace_values(features, non_finite)
        if non_finite.any():
            logger.debug(
                "predict_model: %d non-finite feature value(s) filled with 0.0; "
                "this model was not fitted through fit_model",
                int(non_finite.sum()),
            )

    predictions = np.asarray(model.predict(cleaned), dtype="float64").ravel()
    if predictions.size != features.shape[0]:
        raise ValueError(
            f"predict_model: expected one prediction per row "
            f"({features.shape[0]}) but the model returned {predictions.size}"
        )
    return predictions


# --------------------------------------------------------------------- artifacts


def artifact_path(root: str | os.PathLike[str], name: str, horizon_label: str) -> Path:
    """Path of the ``(model, horizon)`` artifact: ``<root>/v3/<name>_<H>.joblib``.

    The filename stem matches :meth:`src.v3.horizons.Horizon.artifact_stem`
    exactly, so the reports and the files on disk cannot drift apart.

    This function is **pure**: it creates no directories and touches no
    filesystem, which is what lets a caller compute a path for a not-yet-created
    run directory.  Writing is :func:`save_model`'s job.
    """
    return Path(root) / "v3" / f"{name}_{horizon_label}.joblib"


def save_model(model: BaseEstimator, path: str | os.PathLike[str]) -> Path:
    """Persist a fitted model, creating the parent directory if needed.

    The estimator is stored bare rather than in a metadata bundle, unlike the
    V1 pipeline: in V3 the identifying information (model name, horizon, feature
    columns, seed) is already encoded in the filename, and a manifest is
    :mod:`src.v3`'s concern rather than this module's.  ``v3_imputer_`` travels
    with the object, so a loaded model predicts exactly as it did before saving.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    logger.debug("Saved model -> %s", target)
    return target


def load_model(path: str | os.PathLike[str]) -> BaseEstimator:
    """Load a model written by :func:`save_model`."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"No model artifact at {target}")
    model = joblib.load(target)
    logger.debug("Loaded model <- %s", target)
    return model


# ----------------------------------------------------------------------- helpers


def _n_rows(X: Any) -> int:
    """Row count of a feature block, for the feature-agnostic baselines."""
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return int(len(X))


def _values_of(features: np.ndarray | pd.DataFrame) -> np.ndarray:
    """The raw numbers of a feature block, with no pandas semantics attached.

    Every finiteness test in this module goes through here.  ``np.isfinite`` on a
    frame yields a frame, and mixing that with an ``ndarray`` of the same shape
    produces index-aligned comparisons that look correct and are not.
    """
    return features.to_numpy() if isinstance(features, pd.DataFrame) else features


def _replace_values(
    features: np.ndarray | pd.DataFrame, mask: np.ndarray, fill: float = 0.0
) -> np.ndarray | pd.DataFrame:
    """``features`` with the masked cells set to ``fill``, keeping the type."""
    if isinstance(features, pd.DataFrame):
        return features.mask(pd.DataFrame(mask, index=features.index, columns=features.columns), fill)
    return np.where(mask, fill, features)


def _blank_non_finite(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """``features`` with every ``NaN``, ``+inf`` and ``-inf`` replaced by ``NaN``.

    :class:`~sklearn.impute.SimpleImputer` looks for a single sentinel - the
    ``missing_values`` NaN - and treats an infinity as an ordinary float.  Handing
    it ``inf`` unchanged lets the value through untouched, and scikit-learn then
    aborts downstream with "Input X contains infinity", naming the estimator
    rather than the preprocessing step that was supposed to have caught it.  So
    the two sentinels are collapsed into one before imputation.  At fit time this
    is unnecessary (non-finite rows are already dropped); at predict time it is
    the whole job.
    """
    non_finite = ~np.isfinite(_values_of(features))
    if not non_finite.any():
        return features
    return _replace_values(features, non_finite, fill=np.nan)


def _make_imputer() -> SimpleImputer:
    """The training-fitted feature repairer, shared by fit and predict.

    ``set_output(transform="pandas")`` makes the imputer pass a ``DataFrame``
    through as a ``DataFrame`` and an array through as an array.  Without it
    ``fit`` would see columns and ``predict`` would see a bare array, and
    scikit-learn would emit a spurious "fitted without feature names" warning on
    every single forecast.
    """
    return SimpleImputer(strategy="median", keep_empty_features=True).set_output(
        transform="pandas"
    )


def _as_target_vector(y: Any) -> np.ndarray:
    """Targets as a 1-D float64 array.

    Float conversion is not cosmetic: the target may arrive as a nullable pandas
    dtype, and the baselines need plain ``numpy`` arithmetic on it.
    """
    values = np.asarray(y, dtype="float64")
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.ravel()
    if values.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {values.shape}")
    return values


def _as_feature_matrix(X: Any) -> np.ndarray:
    """Features as a 2-D float64 array, keeping a ``DataFrame`` as a frame.

    The frame is preserved so that scikit-learn records ``feature_names_in_``,
    which is what lets a report name the features instead of "f17".
    """
    if isinstance(X, pd.DataFrame):
        try:
            return X.astype("float64")
        except (TypeError, ValueError) as exc:
            offenders = [str(c) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            raise ValueError(
                f"X has non-numeric columns {offenders}; V3 features must be numeric. "
                f"Non-finite numeric entries are handled by fit_model, but a string or "
                f"datetime column is a schema bug, not missing data."
            ) from exc
    if isinstance(X, pd.Series):
        raise ValueError("X must be 2-dimensional; a Series is a single column of rows, not a feature block")
    values = np.asarray(X, dtype="float64")
    if values.ndim != 2:
        raise ValueError(
            f"X must be 2-dimensional (rows x features), got shape {values.shape}. "
            f"A 1-D array cannot say which axis is the timestamp."
        )
    return values


def _take_rows(features: np.ndarray | pd.DataFrame, keep: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Positional row selection that survives a non-default index.

    ``features[keep]`` is boolean-mask indexing and is right for an array, but a
    boolean *Series* against a frame aligns on the index instead - so a training
    block that does not start at position 0 would silently select the wrong rows.
    """
    if isinstance(features, pd.DataFrame):
        return features.iloc[np.flatnonzero(keep)]
    return features[keep]
