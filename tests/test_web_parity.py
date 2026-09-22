"""The browser model must be the Python model, not a lookalike.

The same inference maths exists twice -- once in `stresspredict/` and once in
`web/stresspredict.js`. That is only defensible with a test that runs the actual
JavaScript and diffs it against the actual pipeline, on real compositions,
including the awkward ones with unreported elements. Without this, the demo
would drift from the evaluated model and every number on screen would quietly
stop meaning what the report says it means.

Skips cleanly when Node or the exported model is absent, so CI stays green.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stresspredict import features, predict, schema

MODEL_JSON = Path("web/model.json")
RUNNER = Path("web/parity_check.mjs")
ARTIFACT = Path("artifacts/hist_gbm_ratio_comp.joblib")
TOLERANCE = 1e-9


def _require(condition: bool, reason: str) -> None:
    if not condition:
        pytest.skip(reason)


@pytest.fixture(scope="module")
def node() -> str:
    exe = shutil.which("node")
    _require(exe is not None, "Node is not installed; the web parity check needs it")
    return exe


@pytest.fixture(scope="module")
def exported() -> dict:
    _require(MODEL_JSON.exists(),
             "web/model.json absent; run `python -m stresspredict.export_web`")
    return json.loads(MODEL_JSON.read_text(encoding="utf-8"))


def test_shipped_browser_model_is_the_shipped_python_model(exported):
    """The one drift guard that runs in CI.

    The row-level parity tests below need the dataset and the fitted joblib, both
    of which are gitignored -- so in CI they SKIP, and a green suite there proves
    nothing about agreement with Python. This test compares two committed files
    and therefore always runs.

    It caught a real incident: a `train --fast` invocation, run only to time how
    long training takes, silently overwrote the tuned artifact with an untuned
    one. The sidecar and the browser export then described different fits while
    every other test stayed green.

    A content fingerprint rather than a timestamp, because a legitimate retrain
    changes timestamps while the model is identical, and a tuned/untuned swap can
    leave the timestamps looking perfectly plausible.
    """
    sidecar_path = Path("artifacts/hist_gbm_ratio_comp.json")
    _require(sidecar_path.exists(), "committed sidecar missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert "model_fingerprint" in sidecar, "sidecar predates the fingerprint; re-run train"
    assert "model_fingerprint" in exported, "web export predates the fingerprint; re-run export_web"
    assert exported["model_fingerprint"] == sidecar["model_fingerprint"], (
        "the browser model and the Python model are different fits. "
        f"sidecar={sidecar['model_fingerprint'][:16]} (tuned={sidecar.get('tuned')}), "
        f"web={exported['model_fingerprint'][:16]} (tuned={exported.get('tuned')}). "
        "Re-run `python -m stresspredict.train` then `python -m stresspredict.export_web`."
    )
    assert exported.get("tuned") is True, "the shipped browser model was not tuned"
    assert exported["training_data"]["raw_sha256"] == sidecar["training_data"]["raw_sha256"], (
        "the two models were fitted on different source data"
    )


def test_export_matches_the_python_feature_layout(exported):
    """A silent feature-order change would corrupt every browser prediction."""
    assert exported["elements"] == list(schema.ELEMENTS)
    assert exported["parameterisation"] == "ratio"
    expected_derived = list(features.CompositionFeaturizer.DERIVED_NAMES)
    assert exported["derived_features"] == expected_derived
    assert exported["feature_names"][:len(schema.ELEMENTS)] == list(schema.ELEMENTS)
    assert exported["feature_names"][len(schema.ELEMENTS):][:len(expected_derived)] == (
        expected_derived
    )
    for name in exported["components"].values():
        assert name["n_trees"] == len(name["trees"])
        assert name["link"] in {"log", "logit"}


def test_javascript_reproduces_python_on_every_training_row(node, exported, tmp_path, real_data):
    """The check that makes two implementations acceptable."""
    _require(ARTIFACT.exists(), "no fitted artifact; run `python -m stresspredict.train`")
    df, comp_long = real_data

    dense = features.dense_composition(df, comp_long)
    compositions = [
        {el: float(v) for el, v in row.items() if pd.notna(v)}
        for row in dense.to_dict(orient="records")
    ]
    assert len(compositions) > 1000, "expected the full training set"

    inputs = tmp_path / "inputs.json"
    outputs = tmp_path / "outputs.json"
    inputs.write_text(json.dumps(compositions), encoding="utf-8")

    result = subprocess.run(
        [node, str(RUNNER), str(MODEL_JSON), str(inputs), str(outputs)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, f"node failed:\n{result.stderr}"

    js = pd.DataFrame(json.loads(outputs.read_text(encoding="utf-8")))
    py = predict.predict_batch(dense, artifact_path=ARTIFACT)

    for target in schema.TARGETS:
        a = js[target].to_numpy(dtype=float)
        b = py[target].to_numpy(dtype=float)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        worst = int(np.argmax(rel))
        assert rel.max() < TOLERANCE, (
            f"{target}: JS and Python disagree by {rel.max():.3e} "
            f"(row {worst}: js={a[worst]:.10f} py={b[worst]:.10f})"
        )


def test_javascript_upholds_the_invariant(node, exported, tmp_path):
    """UTS > YS must hold in the browser too, including at extreme inputs."""
    rng = np.random.default_rng(4)
    ranges = exported["element_ranges"]
    compositions = []
    for _ in range(2000):
        compositions.append({
            el: float(rng.uniform(r["min"], r["max"] * 1.5))  # deliberately past the hull
            for el, r in ranges.items()
        })

    inputs = tmp_path / "inputs.json"
    outputs = tmp_path / "outputs.json"
    inputs.write_text(json.dumps(compositions), encoding="utf-8")
    result = subprocess.run(
        [node, str(RUNNER), str(MODEL_JSON), str(inputs), str(outputs)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr

    js = pd.DataFrame(json.loads(outputs.read_text(encoding="utf-8")))
    assert (js["tensile_strength"] > js["yield_strength"]).all()
    assert (js[list(schema.TARGETS)] > 0).all().all()
    assert np.isfinite(js[list(schema.TARGETS)].to_numpy()).all()


def test_unreported_elements_are_handled_identically(node, exported, tmp_path):
    """'Not reported' must mean the same thing in both implementations.

    A user typing only C and Mn is the common case in the UI, and it is exactly
    where a mismatch between fill-with-zero and drop-the-column would hide.
    """
    _require(ARTIFACT.exists(), "no fitted artifact; run `python -m stresspredict.train`")
    compositions = [
        {"C": 0.40},
        {"C": 0.40, "Mn": 0.80},
        {"C": 0.08, "Cr": 18.0, "Ni": 8.0},
        {"C": 0.40, "Mn": 0.80, "Si": 0.25, "Cr": 1.0, "Mo": 0.2},
    ]
    inputs = tmp_path / "inputs.json"
    outputs = tmp_path / "outputs.json"
    inputs.write_text(json.dumps(compositions), encoding="utf-8")
    result = subprocess.run(
        [node, str(RUNNER), str(MODEL_JSON), str(inputs), str(outputs)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr

    js = json.loads(outputs.read_text(encoding="utf-8"))
    for composition, got in zip(compositions, js, strict=True):
        want = predict.predict(composition, artifact_path=ARTIFACT)
        for target in schema.TARGETS:
            assert abs(got[target] - want[target]["value"]) / want[target]["value"] < TOLERANCE, (
                f"{composition} -> {target}: js={got[target]} py={want[target]['value']}"
            )
