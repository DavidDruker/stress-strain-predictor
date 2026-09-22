"""Fit the shipped model and write it with enough provenance to be trusted.

The artifact carries a sidecar describing exactly what produced it: the SHA-256
of the training CSV, the seed, library versions, the tuning protocol, the group
count, the training ranges used by the range guard, and an explicit limitations
string. A model file with no provenance is not reproducible and should not be
believed -- least of all by the person who trained it six months later.

The shipped model is composition-only by design. The heat-treatment ablation is
a result in reports/, not a product surface.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import (
    __version__,
    evaluate,
    export_web,
    features,
    metrics,
    models,
    schema,
    splits,
    targets,
)

ARTIFACT_DIR = Path("artifacts")


def training_ranges(df: pd.DataFrame, comp_long: pd.DataFrame) -> dict:
    """Per-element min/max actually seen in training -- the range guard's basis."""
    dense = features.dense_composition(df, comp_long)
    out = {}
    for el in schema.ELEMENTS:
        col = dense[el].dropna()
        if len(col):
            out[el] = {"min": float(col.min()), "max": float(col.max()), "n_reported": int(len(col))}
    return out


def fit(df: pd.DataFrame, comp_long: pd.DataFrame, parameterisation: str,
        feature_set: str, model_name: str, tune: bool, seed: int) -> dict:
    """Fit one pipeline per component on all available rows for that component."""
    param = targets.get(parameterisation)
    fitted: dict = {}
    report: dict = {}

    X_all = features.design_frame(df, comp_long)
    for component in param.components:
        mask = component.trainable_mask(df).to_numpy()
        sub = df[mask].reset_index(drop=True)
        X = X_all[mask].reset_index(drop=True)
        y = component.quantity(sub).to_numpy(dtype=float)
        groups = sub["grade_group"].to_numpy(dtype=object)

        est, best = evaluate._fit_one(
            model_name, component, feature_set, X, y, groups, tune, seed
        )
        fitted[component.name] = est
        report[component.name] = {
            "n_train_rows": int(len(sub)),
            "n_train_grades": int(pd.Series(groups).nunique()),
            "unit": component.unit,
            "best_params": best,
        }
    return {"pipelines": fitted, "per_component": report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit and save the shipped model.")
    ap.add_argument("--model", default="hist_gbm", choices=list(models.LADDER))
    ap.add_argument("--parameterisation", default=targets.DEFAULT_PARAMETERISATION,
                    choices=sorted(targets.PARAMETERISATIONS))
    ap.add_argument("--features", default="comp", choices=list(features.FEATURE_SETS))
    ap.add_argument("--fast", action="store_true", help="skip hyper-parameter search")
    ap.add_argument("--seed", type=int, default=splits.DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args(argv)

    df, comp_long = evaluate.load_processed()
    manifest_path = evaluate.PROCESSED_DIR / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    tune = not args.fast
    print(f"fitting {args.model} / {args.parameterisation} / {args.features} "
          f"on {len(df)} rows, tuning: {'grouped nested search' if tune else 'OFF'}")
    result = fit(df, comp_long, args.parameterisation, args.features, args.model, tune, args.seed)

    # Reserved now so Phase 3 conformal calibration needs no restructuring.
    calib_mask = splits.reserve_calibration_holdout(df, frac=0.15, seed=args.seed)

    sidecar = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        # Content hash of the fitted trees. The web export records the same
        # value, so a test can prove the browser model and this model are the
        # same fit using only committed files.
        "model_fingerprint": export_web.fingerprint(result["pipelines"]),
        "stresspredict_version": __version__,
        "model": args.model,
        "parameterisation": args.parameterisation,
        "feature_set": args.features,
        "seed": args.seed,
        "tuned": tune,
        "tuning_protocol": (
            "HalvingRandomSearchCV, inner GroupKFold(3) grouped by grade, "
            "scoring=neg_mean_absolute_error" if tune else "none"
        ),
        "n_train_rows": int(len(df)),
        "n_train_grades": int(df["grade_group"].nunique()),
        "n_train_families": int(df["family_group"].nunique()),
        "per_component": result["per_component"],
        "training_data": {
            "source_id": manifest.get("source_id"),
            "raw_file": manifest.get("raw_file"),
            "raw_sha256": manifest.get("raw_sha256"),
            "rows_after_cleaning": manifest.get("rows_out"),
        },
        "element_ranges": training_ranges(df, comp_long),
        "calibration_holdout": {
            "reserved": True,
            "n_rows": int(calib_mask.sum()),
            "grouped_by": "grade_group",
            "purpose": "Phase 3 split-conformal calibration; unused in v1",
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": (
            "Composition-only. Heat treatment dominates steel mechanical properties, so "
            "predictions are effectively a grade prior and cannot distinguish two tempers of "
            "the same chemistry. Accuracy is bounded below by the measured within-composition "
            "spread reported in data/processed/data_manifest.json. Strain at UTS, Hollomon n "
            "and the yield plateau are NOT predicted -- no open source found reports them. "
            "Point predictions carry no calibrated uncertainty in v1. Not for safety-critical "
            "use without physical testing."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.parameterisation}_{args.features}"
    model_path = args.out_dir / f"{stem}.joblib"
    joblib.dump(
        {
            "pipelines": result["pipelines"],
            "parameterisation": args.parameterisation,
            "feature_set": args.features,
            "element_ranges": sidecar["element_ranges"],
            "sidecar": sidecar,
        },
        model_path,
        compress=3,
    )
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(metrics.round_floats(sidecar), indent=2), encoding="utf-8"
    )

    for name, info in result["per_component"].items():
        print(f"    {name:18s} {info['n_train_rows']:5d} rows / {info['n_train_grades']:4d} grades")
    print(f"\nwrote {model_path}")
    print(f"wrote {args.out_dir / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
