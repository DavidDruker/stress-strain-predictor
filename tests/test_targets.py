"""The invariant tests. If these fail, the central promise of the model is void."""

from __future__ import annotations

import numpy as np
import pytest

from stresspredict import features, models, targets
from tests.conftest import make_synthetic


@pytest.mark.parametrize("name", ["ratio", "gap"])
def test_invariant_holds_for_extreme_encoded_values(name):
    """UTS > YS must survive values no optimiser would ever produce.

    sigmoid saturates to exactly 1.0 and exp underflows to 0.0 in float64. Both
    would silently turn a strict inequality into an equality, so the guards are
    tested at the extremes rather than only in the comfortable middle.
    """
    param = targets.get(name)
    extreme = np.array([-1e6, -1e3, -50.0, -1e-9, 0.0, 1e-9, 50.0, 1e3, 1e6])

    if name == "ratio":
        uts = np.repeat(np.array([1.0, 250.0, 2000.0, 1e6]), len(extreme))
        z = np.tile(extreme, 4)
        landmarks = param.decode({
            "tensile_strength": uts,
            "yield_ratio": 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700))),
            "elongation": np.full_like(uts, 20.0),
        })
    else:
        ys = np.repeat(np.array([1.0, 250.0, 2000.0, 1e6]), len(extreme))
        z = np.tile(extreme, 4)
        landmarks = param.decode({
            "yield_strength": ys,
            "strength_gap": np.exp(np.clip(z, -700, 700)),
            "elongation": np.full_like(ys, 20.0),
        })

    holds, violations = targets.check_invariant(landmarks)
    assert holds, f"{violations} violations of UTS > YS under {name}"
    assert np.all(landmarks["tensile_strength"] > landmarks["yield_strength"])


@pytest.mark.parametrize("name", ["ratio", "gap"])
def test_invariant_over_10k_sampled_compositions(name):
    """Fit a real model, then assert UTS > YS on 10,000 sampled compositions.

    Sampled from the training hull, which is where the model is actually used --
    an invariant that only holds on the training rows themselves would be a
    property of the data, not of the parameterisation.
    """
    samples, comp_long = make_synthetic(n=400, seed=11)
    from stresspredict import grouping
    df = grouping.attach_groups(samples, comp_long)
    param = targets.get(name)

    X = features.design_frame(df, comp_long)
    fitted = {}
    for component in param.components:
        mask = component.trainable_mask(df).to_numpy()
        est = models.build("hist_gbm", component, feature_set="comp")
        fitted[component.name] = est.fit(
            X[mask], component.quantity(df[mask]).to_numpy(dtype=float)
        )

    rng = np.random.default_rng(3)
    lo, hi = X.min(numeric_only=True), X.max(numeric_only=True)
    grid = {c: rng.uniform(lo[c], hi[c], 10_000) for c in X.columns}
    import pandas as pd
    Xs = pd.DataFrame(grid)

    landmarks = param.decode({n: p.predict(Xs) for n, p in fitted.items()})
    holds, violations = targets.check_invariant(landmarks)
    assert holds, f"{violations}/10000 sampled compositions violated UTS > YS under {name}"
    assert np.all(np.isfinite(landmarks["tensile_strength"]))
    assert np.all(landmarks["yield_strength"] > 0)


@pytest.mark.parametrize("name", ["ratio", "gap"])
def test_encode_decode_round_trip(name):
    """Encoding true targets and decoding them back must return the originals."""
    import pandas as pd
    df = pd.DataFrame({
        "yield_strength": [250.0, 600.0, 1200.0],
        "tensile_strength": [400.0, 800.0, 1500.0],
        "elongation": [30.0, 18.0, 8.0],
    })
    param = targets.get(name)
    quantities = {c.name: c.quantity(df).to_numpy(dtype=float) for c in param.components}
    back = param.decode(quantities)
    for target in ("yield_strength", "tensile_strength", "elongation"):
        np.testing.assert_allclose(back[target], df[target].to_numpy(), rtol=1e-10)


def test_derived_outputs_are_consistent():
    landmarks = {
        "yield_strength": np.array([300.0, 900.0]),
        "tensile_strength": np.array([500.0, 1200.0]),
        "elongation": np.array([25.0, 10.0]),
    }
    d = targets.derived_outputs(landmarks)
    np.testing.assert_allclose(d["yield_ratio"], [0.6, 0.75])
    np.testing.assert_allclose(d["strength_gap"], [200.0, 300.0])
    np.testing.assert_allclose(d["strength_ductility_product"], [12500.0, 12000.0])


def test_check_invariant_detects_violations():
    bad = {
        "yield_strength": np.array([500.0, 100.0]),
        "tensile_strength": np.array([400.0, 900.0]),
        "elongation": np.array([10.0, 20.0]),
    }
    holds, violations = targets.check_invariant(bad)
    assert not holds and violations == 1


def test_trainable_mask_requires_every_input_column():
    """The gap component needs BOTH strengths; the ratio component too."""
    import pandas as pd
    df = pd.DataFrame({
        "yield_strength": [300.0, np.nan, 500.0],
        "tensile_strength": [500.0, 700.0, np.nan],
        "elongation": [20.0, 20.0, 20.0],
    })
    assert targets.RATIO["tensile_strength"].trainable_mask(df).tolist() == [True, True, False]
    assert targets.RATIO["yield_ratio"].trainable_mask(df).tolist() == [True, False, False]
    assert targets.GAP["strength_gap"].trainable_mask(df).tolist() == [True, False, False]
