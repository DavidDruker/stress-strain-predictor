"""Inference surface: output contract, the invariant, and the range guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stresspredict import features, grouping, models, predict, targets, train
from tests.conftest import make_synthetic


@pytest.fixture(scope="module")
def artifact() -> dict:
    """A cheap in-memory artifact with the same shape train.py writes to disk."""
    samples, comp_long = make_synthetic(n=300, seed=5)
    df = grouping.attach_groups(samples, comp_long)
    param = targets.get("ratio")
    X = features.design_frame(df, comp_long)

    pipelines = {}
    for component in param.components:
        mask = component.trainable_mask(df).to_numpy()
        pipelines[component.name] = models.build("ridge", component, "comp").fit(
            X[mask], component.quantity(df[mask]).to_numpy(dtype=float)
        )
    return {
        "pipelines": pipelines,
        "parameterisation": "ratio",
        "feature_set": "comp",
        "element_ranges": train.training_ranges(df, comp_long),
    }


def test_predict_returns_the_documented_contract(artifact):
    out = predict.predict({"C": 0.4, "Mn": 0.8, "Cr": 1.0}, artifact=artifact)
    for key in ("yield_strength", "tensile_strength", "elongation"):
        assert set(out[key]) == {"value", "unit"}
        assert np.isfinite(out[key]["value"])
        assert out[key]["value"] > 0
    assert out["yield_strength"]["unit"] == "MPa"
    assert out["elongation"]["unit"] == "%"
    assert set(out["derived"]) == {"yield_ratio", "strength_gap", "strength_ductility_product"}
    assert "caveat" in out


def test_predicted_uts_always_exceeds_ys(artifact):
    out = predict.predict({"C": 0.6, "Mn": 1.2, "Cr": 0.5}, artifact=artifact)
    assert out["tensile_strength"]["value"] > out["yield_strength"]["value"]
    assert 0 < out["derived"]["yield_ratio"] < 1
    assert out["derived"]["strength_gap"] > 0


def test_range_guard_fires_and_names_the_element(artifact):
    out = predict.predict({"C": 9.0, "Mn": 0.8}, artifact=artifact)
    assert out["in_training_range"] is False
    flagged = {f["element"] for f in out["range_guard"]}
    assert "C" in flagged
    entry = next(f for f in out["range_guard"] if f["element"] == "C")
    assert entry["issue"] == "outside training range"
    assert entry["value"] == 9.0
    assert entry["trained_max"] < 9.0


def test_range_guard_silent_inside_the_hull(artifact):
    out = predict.predict({"C": 0.4, "Mn": 1.0}, artifact=artifact)
    assert out["range_guard"] == []
    assert out["in_training_range"] is True


def test_unknown_element_is_rejected_loudly(artifact):
    with pytest.raises(ValueError, match="unknown element"):
        predict.predict({"C": 0.4, "Unobtainium": 1.0}, artifact=artifact)


def test_batch_prediction_matches_single(artifact):
    comps = [{"C": 0.2, "Mn": 0.5}, {"C": 0.5, "Mn": 1.5, "Cr": 2.0}]
    batch = predict.predict_batch(pd.DataFrame(comps), artifact=artifact)
    assert (batch["tensile_strength"] > batch["yield_strength"]).all()
    for i, c in enumerate(comps):
        single = predict.predict(c, artifact=artifact)
        assert batch["tensile_strength"].iloc[i] == pytest.approx(
            single["tensile_strength"]["value"], rel=1e-9
        )


def test_composition_string_parsing():
    assert predict._parse_composition("C=0.4, Mn=0.80,Cr=1") == {"C": 0.4, "Mn": 0.8, "Cr": 1.0}
    with pytest.raises(ValueError):
        predict._parse_composition("C 0.4")


def test_missing_artifact_gives_an_actionable_error():
    with pytest.raises(FileNotFoundError, match="stresspredict.train"):
        predict.load("artifacts/does_not_exist.joblib")
