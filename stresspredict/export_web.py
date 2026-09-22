"""Export the fitted model to JSON so it can run in a browser.

The web demo must not be a mock-up. Exporting the actual fitted trees means the
page runs the same model as `predict.py`, offline and instantly, and a parity
test asserts the JavaScript reproduces Python to 1e-9 on every training row --
so a number shown in the UI is the number the evaluation measured, not a
re-implementation that drifted.

Thresholds are written at full precision on purpose. Rounding them could flip a
comparison for a feature value sitting close to a split boundary, which would be
a silent, input-dependent disagreement -- the worst kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from . import __version__, features, schema, targets

DEFAULT_ARTIFACT = Path("artifacts/hist_gbm_ratio_comp.joblib")
DEFAULT_OUT = Path("web/model.json")

# Link functions are named rather than serialised; the JS implements the same
# two and refuses anything it does not recognise.
LINK_NAMES = {"log": "log", "logit": "logit"}


def _link_name(component: targets.Component) -> str:
    if component.inverse_link is np.exp:
        return "log"
    if getattr(component.inverse_link, "__name__", "") == "expit":
        return "logit"
    raise ValueError(f"no web equivalent for the link on {component.name!r}")


def _export_trees(hgb) -> list[dict]:
    """Flatten the gradient-boosted trees into parallel arrays.

    Leaf values already carry the learning-rate shrinkage applied during fitting,
    so a prediction is simply baseline + the sum of one leaf value per tree.
    """
    out = []
    for stage in hgb._predictors:
        if len(stage) != 1:
            raise ValueError("multi-output boosting is not supported by the web export")
        nodes = stage[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported by the web export")
        out.append({
            "f": [int(v) for v in nodes["feature_idx"]],
            "th": [float(v) for v in nodes["num_threshold"]],
            "l": [int(v) for v in nodes["left"]],
            "r": [int(v) for v in nodes["right"]],
            "v": [round(float(v), 12) for v in nodes["value"]],
            "lf": [int(v) for v in nodes["is_leaf"]],
            "m": [int(v) for v in nodes["missing_go_to_left"]],
        })
    return out


def fingerprint(pipelines: dict) -> str:
    """A content hash of the fitted trees themselves.

    This exists so drift between the shipped browser model and the shipped
    Python model can be caught by a test that needs NEITHER the dataset nor the
    joblib -- both of which are gitignored, which is why the row-level parity
    test silently skips in CI and cannot be the only guard. Comparing timestamps
    would not do: a legitimate retrain changes those while the model is
    identical, and a tuned/untuned swap can leave them looking plausible.
    """
    h = hashlib.sha256()
    for name in sorted(pipelines):
        inner = pipelines[name].named_steps["regressor"].regressor_
        hgb = inner.named_steps["model"]
        h.update(name.encode())
        h.update(np.asarray(hgb._baseline_prediction, dtype=float).tobytes())
        for stage in hgb._predictors:
            nodes = stage[0].nodes
            for field in ("feature_idx", "num_threshold", "left", "right",
                          "value", "is_leaf", "missing_go_to_left"):
                h.update(np.ascontiguousarray(nodes[field]).tobytes())
    return h.hexdigest()


def export(artifact_path: Path = DEFAULT_ARTIFACT) -> dict:
    art = joblib.load(artifact_path)
    param = targets.get(art["parameterisation"])
    sidecar = art.get("sidecar", {})

    components: dict = {}
    feature_names: list[str] | None = None
    indicator_elements: list[str] | None = None

    for name, pipe in art["pipelines"].items():
        inner = pipe.named_steps["regressor"].regressor_
        featurizer = inner.named_steps["features"]
        hgb = inner.named_steps["model"]

        names = [str(n) for n in featurizer.get_feature_names_out()]
        if feature_names is None:
            feature_names = names
            indicator_elements = list(featurizer.indicator_cols_)
        elif names != feature_names:
            # Every component must see an identical feature matrix, or the JS
            # would need one featuriser per component and the ladder comparison
            # in the report would not have been like-for-like either.
            raise ValueError(f"component {name!r} has a different feature layout")

        baseline = np.ravel(np.asarray(hgb._baseline_prediction, dtype=float))
        if baseline.size != 1:
            raise ValueError("expected a scalar baseline prediction")

        components[name] = {
            "unit": param[name].unit,
            "link": _link_name(param[name]),
            "baseline": float(baseline[0]),
            "n_trees": len(hgb._predictors),
            "trees": _export_trees(hgb),
        }

    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stresspredict_version": __version__,
        "model": art.get("feature_set") and sidecar.get("model", "hist_gbm"),
        "parameterisation": art["parameterisation"],
        "feature_set": art["feature_set"],
        "elements": list(schema.ELEMENTS),
        "derived_features": list(features.CompositionFeaturizer.DERIVED_NAMES),
        "indicator_elements": indicator_elements,
        "feature_names": feature_names,
        "element_ranges": art["element_ranges"],
        "landmark_requires": param.landmark_requires,
        "training_data": sidecar.get("training_data", {}),
        "model_fingerprint": fingerprint(art["pipelines"]),
        "tuned": sidecar.get("tuned"),
        "n_train_rows": sidecar.get("n_train_rows"),
        "n_train_grades": sidecar.get("n_train_grades"),
        "limitations": sidecar.get("limitations", ""),
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export the fitted model to JSON for the web demo.")
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    bundle = export(args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, separators=(",", ":"))
    args.out.write_text(payload, encoding="utf-8")

    # The page loads the model through a <script> tag rather than fetch(): an
    # artifact's CSP is strict about runtime requests, and a plain script is the
    # one form that is reliably allowed. Same bytes, one assignment around them.
    script_path = args.out.with_suffix(".js")
    script_path.write_text(f"window.SP_MODEL={payload};\n", encoding="utf-8")

    n_nodes = sum(len(t["f"]) for c in bundle["components"].values() for t in c["trees"])
    size_kb = args.out.stat().st_size / 1024
    print(f"components : {', '.join(bundle['components'])}")
    print(f"features   : {len(bundle['feature_names'])}")
    print(f"trees      : {sum(c['n_trees'] for c in bundle['components'].values())}")
    print(f"nodes      : {n_nodes}")
    print(f"wrote {args.out}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
