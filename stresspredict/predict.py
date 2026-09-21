"""Inference: a composition dict in, landmark values out.

The range guard is four lines and is the most useful thing this module prints.
"Outside training range for: C (0.90 wt%, max seen 0.65)" tells an engineer
exactly which input to distrust and by how much. A scalar novelty score does
not, and engineers are right not to trust one. Full k-NN applicability-domain
scoring is Phase 3; naming the offending element is most of the value and costs
almost nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import schema, targets

ARTIFACT_DIR = Path("artifacts")
DEFAULT_ARTIFACT = ARTIFACT_DIR / "hist_gbm_ratio_comp.joblib"


def load(path: Path | str = DEFAULT_ARTIFACT) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train one first:  python -m stresspredict.train"
        )
    return joblib.load(path)


def _design_row(composition: dict[str, float]) -> pd.DataFrame:
    unknown = set(composition) - set(schema.ELEMENTS)
    if unknown:
        raise ValueError(
            f"unknown element(s) {sorted(unknown)}; this model covers {list(schema.ELEMENTS)}"
        )
    row = {el: float(composition.get(el, np.nan)) for el in schema.ELEMENTS}
    for col in schema.PROCESS_NUMERIC:
        row[col] = np.nan
    return pd.DataFrame([row])


def range_guard(composition: dict[str, float], element_ranges: dict) -> list[dict]:
    """Which supplied elements fall outside the range the model actually saw."""
    flags = []
    for el, value in composition.items():
        r = element_ranges.get(el)
        if r is None:
            flags.append({"element": el, "value": value, "issue": "never reported in training"})
        elif value < r["min"] or value > r["max"]:
            flags.append({
                "element": el, "value": value,
                "trained_min": r["min"], "trained_max": r["max"],
                "issue": "outside training range",
            })
    return flags


def predict(composition: dict[str, float], artifact: dict | None = None,
            artifact_path: Path | str = DEFAULT_ARTIFACT) -> dict:
    """Predict tensile landmarks for one composition in wt%.

    UTS > YS is guaranteed by the parameterisation, not checked after the fact.
    """
    art = artifact if artifact is not None else load(artifact_path)
    param = targets.get(art["parameterisation"])

    X = _design_row(composition)
    quantities = {name: np.asarray(pipe.predict(X), dtype=float)
                  for name, pipe in art["pipelines"].items()}
    landmarks = param.decode(quantities)
    derived = targets.derived_outputs(landmarks)

    out = {
        "yield_strength": {"value": float(landmarks["yield_strength"][0]), "unit": "MPa"},
        "tensile_strength": {"value": float(landmarks["tensile_strength"][0]), "unit": "MPa"},
        "elongation": {"value": float(landmarks["elongation"][0]), "unit": "%"},
        "derived": {
            "yield_ratio": float(derived["yield_ratio"][0]),
            "strength_gap": float(derived["strength_gap"][0]),
            "strength_ductility_product": float(derived["strength_ductility_product"][0]),
        },
        "range_guard": range_guard(composition, art["element_ranges"]),
        "caveat": (
            "Composition-only prediction: this is a grade-level prior and cannot "
            "distinguish heat treatments of the same chemistry. No calibrated "
            "uncertainty interval in v1."
        ),
    }
    out["in_training_range"] = not out["range_guard"]
    return out


def predict_batch(compositions: pd.DataFrame, artifact: dict | None = None,
                  artifact_path: Path | str = DEFAULT_ARTIFACT) -> pd.DataFrame:
    """Vectorised prediction for a frame of compositions (columns = elements)."""
    art = artifact if artifact is not None else load(artifact_path)
    param = targets.get(art["parameterisation"])

    X = compositions.reindex(columns=list(schema.ELEMENTS)).astype(float)
    for col in schema.PROCESS_NUMERIC:
        X[col] = np.nan

    quantities = {name: np.asarray(pipe.predict(X), dtype=float)
                  for name, pipe in art["pipelines"].items()}
    landmarks = param.decode(quantities)
    out = pd.DataFrame(landmarks, index=compositions.index)
    for k, v in targets.derived_outputs(landmarks).items():
        out[k] = v
    return out


def _parse_composition(text: str) -> dict[str, float]:
    """Parse 'C=0.4,Mn=0.8,Cr=1.0' into a dict of wt%."""
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected ELEMENT=VALUE, got {part!r}")
        el, val = part.split("=", 1)
        out[el.strip()] = float(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict tensile landmarks from a composition.")
    ap.add_argument("--composition", required=True,
                    help="wt%%, e.g. 'C=0.40,Mn=0.80,Cr=1.00,Mo=0.20'")
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args(argv)

    result = predict(_parse_composition(args.composition), artifact_path=args.artifact)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"  yield strength   {result['yield_strength']['value']:8.1f} MPa")
    print(f"  tensile strength {result['tensile_strength']['value']:8.1f} MPa")
    print(f"  elongation       {result['elongation']['value']:8.1f} %")
    print(f"  yield ratio      {result['derived']['yield_ratio']:8.3f}")
    print(f"  UTS x EL         {result['derived']['strength_ductility_product']:8.0f} MPa.%")
    if result["range_guard"]:
        print("\n  RANGE GUARD")
        for f in result["range_guard"]:
            if "trained_max" in f:
                print(f"    outside training range for: {f['element']} "
                      f"({f['value']:g} wt%, trained range "
                      f"{f['trained_min']:g}-{f['trained_max']:g})")
            else:
                print(f"    {f['element']}: {f['issue']}")
    else:
        print("\n  all elements within training range")
    print(f"\n  {result['caveat']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
