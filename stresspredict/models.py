"""The model ladder. Every rung exists to make a specific claim falsifiable.

The ladder stops at HistGradientBoosting. No XGBoost, no LightGBM -- and the
claim is backed with a number rather than taste: on ~1.3k rows and ~20 features
the HGB-vs-XGBoost gap is smaller than the fold-to-fold standard error of the
grouped CV, which `evaluate` reports for exactly this purpose. Reporting the
std dev that licenses the decision is a better signal than a marginally better
score from a dependency nobody can justify.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CompositionFeaturizer
from .targets import Component

SEED = 20260921

# Searched inside the outer fold only. Ranges are deliberately small: at this
# dataset size an aggressive search overfits the inner split faster than it
# improves the outer score.
PARAM_SPACES: dict[str, dict] = {
    "dummy": {},
    "ridge": {},  # RidgeCV already selects alpha by internal CV
    "random_forest": {
        "regressor__regressor__model__max_depth": [None, 8, 16],
        "regressor__regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__model__max_features": [0.3, 0.5, 1.0],
    },
    "extra_trees": {
        "regressor__regressor__model__max_depth": [None, 8, 16],
        "regressor__regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__model__max_features": [0.3, 0.5, 1.0],
    },
    "hist_gbm": {
        "regressor__regressor__model__learning_rate": [0.03, 0.06, 0.1],
        "regressor__regressor__model__max_leaf_nodes": [7, 15, 31],
        "regressor__regressor__model__min_samples_leaf": [5, 10, 20],
        "regressor__regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def _bare(name: str):
    """Return (estimator, needs_scaling) for a ladder rung."""
    if name == "dummy":
        # The denominator for every claim in the report. A model that cannot
        # beat the median of the training targets has learned nothing.
        return DummyRegressor(strategy="median"), False
    if name == "ridge":
        # The linear control, and a free correctness test: the carbon
        # coefficient must be strongly positive for strength and negative for
        # elongation. If it is not, the pipeline is wired wrong -- and no amount
        # of gradient boosting stacked on top would reveal that.
        return RidgeCV(alphas=np.logspace(-3, 4, 40)), True
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), False
    if name == "extra_trees":
        # Zero marginal cost over the forest and frequently better on small,
        # noisy tabular data -- worth the one line it takes to find out.
        return ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), False
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), False
    raise KeyError(f"unknown model {name!r}; known: {sorted(LADDER)}")


def build(name: str, component: Component, feature_set: str = "comp") -> Pipeline:
    """Featuriser -> (scaler) -> estimator, predicting in the component's units.

    TransformedTargetRegressor applies the log/logit link, so cross-validated
    predictions arrive as MPa (or as a ratio) rather than as log-space numbers
    that somebody downstream has to remember to exponentiate.
    """
    estimator, needs_scaling = _bare(name)

    steps = [("features", CompositionFeaturizer(feature_set=feature_set))]
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    inner = Pipeline(steps)

    return Pipeline([
        ("regressor", TransformedTargetRegressor(
            regressor=inner,
            func=component.link,
            inverse_func=component.inverse_link,
            check_inverse=False,  # log/exp and logit/expit round-trip by construction
        )),
    ])


LADDER: tuple[str, ...] = ("dummy", "ridge", "random_forest", "extra_trees", "hist_gbm")

LADDER_ROLE: dict[str, str] = {
    "dummy": "baseline -- the denominator for every claim",
    "ridge": "linear control; coefficient signs are a free correctness test",
    "random_forest": "workhorse, near-zero tuning",
    "extra_trees": "zero marginal cost, often wins on small noisy data",
    "hist_gbm": "expected winner; the ladder stops here on purpose",
}


def ridge_coefficients(fitted: Pipeline) -> dict[str, float]:
    """Named coefficients from a fitted ridge pipeline, for the sign check."""
    inner = fitted.named_steps["regressor"].regressor_
    names = inner.named_steps["features"].get_feature_names_out()
    coefs = inner.named_steps["model"].coef_
    return {str(n): float(c) for n, c in zip(names, np.ravel(coefs), strict=True)}
