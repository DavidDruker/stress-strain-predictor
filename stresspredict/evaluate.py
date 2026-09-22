"""Cross-validated evaluation across protocols, models and targets.

Three things here are load-bearing and easy to get wrong:

1. Hyper-parameter search runs INSIDE each outer training fold, never once over
   the whole dataset. This is where portfolio projects leak.
2. The inner search is ALWAYS grouped by grade, even when the outer protocol is
   the deliberately leaky random K-fold. Tuning is never allowed to leak, even
   while we are demonstrating what an outer leak costs.
3. Components are cross-validated in their own units and only then assembled
   into landmarks, so the invariant that `targets` guarantees is enforced on
   out-of-fold predictions too -- not just on the final fitted model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from . import features, grouping, metrics, models, schema, splits, targets

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def load_processed(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples_path = processed_dir / "samples.csv"
    comp_path = processed_dir / "composition_long.csv"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"{samples_path} not found. Run:  python -m stresspredict.ingest --source steelbench"
        )
    samples = pd.read_csv(samples_path)
    comp = pd.read_csv(comp_path)
    return grouping.attach_groups(samples, comp), comp


def select_rows(df: pd.DataFrame, rows: str, restrict_ht: bool) -> pd.DataFrame:
    """Row filters for the measurement-kind and heat-treatment ablations."""
    out = df
    if rows == "measured":
        out = out[out["measurement_kind"] == "measured"]
    elif rows == "spec_minimum":
        out = out[out["measurement_kind"] == "spec_minimum"]
    elif rows != "all":
        raise ValueError(f"unknown --rows {rows!r}")
    if restrict_ht:
        # Ablation arm C: composition-only but on exactly arm B's rows, so the
        # comparison is not confounded by the row subset. Reporting C vs B rather
        # than A vs B is the whole point of the flag.
        out = out[out["austenitize_T"].notna() & out["temper_T"].notna()]
    return out.reset_index(drop=True)


def _fit_one(model_name: str, component, feature_set: str,
             X_tr: pd.DataFrame, y_tr: np.ndarray, groups_tr: np.ndarray,
             tune: bool, seed: int):
    """Fit one model on one training fold, tuning inside it if asked."""
    est = models.build(model_name, component, feature_set=feature_set)
    space = models.PARAM_SPACES.get(model_name, {})

    if not tune or not space:
        return est.fit(X_tr, y_tr), None

    n_groups = len(np.unique(groups_tr))
    if n_groups < 2:
        return est.fit(X_tr, y_tr), None

    search = HalvingRandomSearchCV(
        est,
        param_distributions=space,
        n_candidates=8,
        factor=3,
        # NOT the default. HalvingRandomSearchCV defaults to min_resources
        # "smallest", which on this dataset starts at 6 rows and selects
        # hyper-parameters from 6 then 18 samples -- tuning that is worse than
        # no tuning, while still costing the time and still being describable as
        # "nested search". "exhaust" sizes the schedule so the final rung trains
        # on the whole training fold (here 453 then 1,359 rows). It is also
        # measurably faster, because per-fit overhead stops dominating.
        min_resources="exhaust",
        cv=splits.inner_cv(n_splits=min(3, n_groups)),
        scoring="neg_mean_absolute_error",
        random_state=seed,
        # The estimators already parallelise over trees. Parallelising candidates
        # on top of that oversubscribes every core and is slower, not faster.
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_tr, y_tr, groups=groups_tr)
    return search.best_estimator_, search.best_params_


def _cv_component(df: pd.DataFrame, comp_long: pd.DataFrame, component,
                  protocol: splits.Protocol, model_name: str, feature_set: str,
                  tune: bool, seed: int) -> dict:
    """Out-of-fold predictions for one component under one protocol."""
    sub = df[component.trainable_mask(df)].reset_index(drop=True)
    result = {
        "component": component.name,
        "unit": component.unit,
        "n_rows": int(len(sub)),
        "per_fold": [],
        "oof": pd.Series(dtype=float),
        "skipped": None,
        "best_params": [],
    }
    if len(sub) < 50:
        result["skipped"] = f"only {len(sub)} rows available for this component"
        return result

    X = features.design_frame(sub, comp_long)
    y = component.quantity(sub).to_numpy(dtype=float)
    groups = protocol.groups(sub)
    # Tuning is grouped by grade regardless of the outer protocol -- see the
    # module docstring.
    inner_groups = sub["grade_group"].to_numpy(dtype=object)

    splitter = protocol.splitter()
    try:
        n_splits = splitter.get_n_splits(X, y, groups)
    except ValueError as exc:
        result["skipped"] = f"splitter unusable: {exc}"
        return result
    if n_splits < 2:
        result["skipped"] = (
            f"protocol yields only {n_splits} fold(s) on this row subset "
            f"({len(np.unique(groups)) if groups is not None else 0} groups present)"
        )
        return result

    oof = np.full(len(sub), np.nan)
    fold_names: list[str] = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        fitted, best = _fit_one(
            model_name, component, feature_set,
            X.iloc[tr], y[tr],
            inner_groups[tr], tune, seed,
        )
        oof[te] = fitted.predict(X.iloc[te])
        result["per_fold"].append(metrics.score(y[te], oof[te]))
        result["best_params"].append(best)
        fold_names.append(
            str(np.unique(groups[te])[0]) if groups is not None and len(np.unique(groups[te])) == 1
            else f"fold{k}"
        )

    result["fold_labels"] = fold_names
    result["n_folds"] = len(result["per_fold"])
    result["summary_mae"] = metrics.aggregate_folds(result["per_fold"], "mae")
    result["pooled"] = metrics.score(y, oof)
    result["oof"] = pd.Series(oof, index=sub["sample_id"].to_numpy())
    if protocol.name == "lofo_family":
        result["excluded_small_groups"] = splitter.excluded_groups(groups)
    return result


def evaluate(df: pd.DataFrame, comp_long: pd.DataFrame, protocol_name: str,
             model_names: tuple[str, ...], parameterisation: str, feature_set: str,
             tune: bool, seed: int) -> dict:
    """Run every model on every component of one parameterisation, one protocol."""
    protocol = splits.get(protocol_name)
    param = targets.get(parameterisation)

    out: dict = {
        "protocol": protocol.name,
        "protocol_question": protocol.question,
        "protocol_is_leaky": protocol.leaky,
        "group_column": protocol.group_column,
        "parameterisation": param.name,
        "feature_set": feature_set,
        "tuned_inside_outer_fold": tune,
        "seed": seed,
        "n_rows": int(len(df)),
        "n_grades": int(df["grade_group"].nunique()),
        "models": {},
    }

    # The fully-observed core: rows where all three landmarks are measured. Every
    # landmark is scored a second time on exactly this row set, so numbers can be
    # compared across targets and across parameterisations without the row count
    # silently changing underneath the comparison.
    core_mask = df[list(schema.TARGETS)].notna().all(axis=1)
    core_ids = set(df.loc[core_mask, "sample_id"])
    out["n_common_core_rows"] = len(core_ids)

    for model_name in model_names:
        t0 = time.perf_counter()
        comps = {
            c.name: _cv_component(df, comp_long, c, protocol, model_name,
                                  feature_set, tune, seed)
            for c in param.components
        }

        entry: dict = {
            "role": models.LADDER_ROLE[model_name],
            "components": {
                name: {k: v for k, v in c.items() if k != "oof"}
                for name, c in comps.items()
            },
            "landmarks": {},
        }

        # Assemble landmark predictions from component out-of-fold predictions.
        #
        # Each landmark is scored on every row it can actually be produced for,
        # using ONLY the components it needs. Requiring all three components to be
        # present would score UTS on the 659-row fully-observed core instead of all
        # 1,359 rows -- discarding exactly the extra coverage the ratio
        # parameterisation exists to provide, and biasing the subset toward the
        # easier specification-minimum rows. That silently flatters UTS by ~12%.
        oof_frames = {n: c["oof"] for n, c in comps.items() if len(c["oof"])}
        if oof_frames:
            wide = pd.DataFrame(oof_frames)          # outer join; NaN where absent
            truth_all = df.set_index("sample_id")
            oof_rows: dict = {}

            for lm in schema.TARGETS:
                required = param.landmark_requires[lm]
                if not set(required) <= set(wide.columns):
                    continue
                rows = wide.dropna(subset=list(required))
                if rows.empty:
                    continue
                # Components this landmark does not need are passed as NaN: the
                # assembly returns NaN for the other landmarks, which we ignore.
                quantities = {
                    c.name: (rows[c.name].to_numpy() if c.name in required
                             else np.full(len(rows), np.nan))
                    for c in param.components
                }
                yp = np.asarray(param.decode(quantities)[lm], dtype=float)
                truth = truth_all.loc[rows.index]
                yt = truth[lm].to_numpy(dtype=float)
                in_core = np.array([sid in core_ids for sid in rows.index])
                kind = truth["measurement_kind"].to_numpy()

                entry["landmarks"][lm] = {
                    "unit": schema.TARGET_UNITS[lm],
                    "full": metrics.score(yt, yp),
                    "common_core": metrics.score(yt[in_core], yp[in_core]),
                    # The only comparison that is like-for-like against the noise
                    # floor, which is itself measured per measurement kind.
                    "by_measurement_kind": {
                        k: metrics.score(yt[kind == k], yp[kind == k])
                        for k in np.unique(kind)
                    },
                }
                oof_rows[lm] = pd.DataFrame({
                    "sample_id": rows.index, "model": model_name,
                    "protocol": protocol.name,
                    "steel_family": truth["family_group"].to_numpy(),
                    "measurement_kind": kind,
                    "landmark": lm, "y_true": yt, "y_pred": yp,
                })

            # The invariant is checked wherever BOTH strengths are producible.
            both = param.landmark_requires["yield_strength"]
            rows = wide.dropna(subset=[c for c in both if c in wide.columns])
            if len(rows) and set(both) <= set(wide.columns):
                landmarks = param.decode({
                    c.name: (rows[c.name].to_numpy() if c.name in wide.columns
                             else np.full(len(rows), np.nan))
                    for c in param.components
                })
                holds, violations = targets.check_invariant(landmarks)
                entry["invariant_uts_gt_ys"] = {
                    "holds": bool(holds),
                    "violations": int(violations),
                    "n_checked": int(len(rows)),
                }

            # Kept out of the JSON; written alongside it as a CSV so the parity
            # plots never have to re-run the cross-validation.
            if oof_rows:
                entry["_oof"] = pd.concat(oof_rows.values(), ignore_index=True)

        entry["fit_seconds"] = round(time.perf_counter() - t0, 2)
        out["models"][model_name] = entry

    return out


def ridge_sign_check(df: pd.DataFrame, comp_long: pd.DataFrame,
                     parameterisation: str, feature_set: str) -> dict:
    """Metallurgical correctness test on ridge coefficients.

    Carbon must push strength up and elongation down. This is not a performance
    metric -- it is a wiring test. If these signs are wrong the featuriser,
    target transform or join is broken, and every other number in the report is
    meaningless regardless of how good it looks.

    The diagnostic components are fixed rather than taken from the active
    parameterisation, so the check covers yield strength, tensile strength and
    elongation directly whichever encoding is being evaluated -- a check that
    changed shape with the encoding would be a weaker test of the same wiring.
    """
    diagnostic = (
        targets.GAP["yield_strength"],
        targets.RATIO["tensile_strength"],
        targets.RATIO["elongation"],
    )
    checks: dict = {}
    for component in diagnostic:
        sub = df[component.trainable_mask(df)].reset_index(drop=True)
        if len(sub) < 50:
            continue
        X = features.design_frame(sub, comp_long)
        y = component.quantity(sub).to_numpy(dtype=float)
        fitted = models.build("ridge", component, feature_set=feature_set).fit(X, y)
        coefs = models.ridge_coefficients(fitted)
        checks[component.name] = {
            "carbon_coefficient": round(coefs.get("C", float("nan")), 5),
            "top_by_magnitude": sorted(
                ((k, round(v, 5)) for k, v in coefs.items()),
                key=lambda kv: -abs(kv[1]),
            )[:8],
        }

    expectations = []
    strength_components = [c for c in ("tensile_strength", "yield_strength") if c in checks]
    for name in strength_components:
        expectations.append({
            "component": name,
            "expected": "carbon coefficient > 0 (carbon raises strength)",
            "observed": checks[name]["carbon_coefficient"],
            "passed": bool(checks[name]["carbon_coefficient"] > 0),
        })
    if "elongation" in checks:
        expectations.append({
            "component": "elongation",
            "expected": "carbon coefficient < 0 (carbon reduces ductility)",
            "observed": checks["elongation"]["carbon_coefficient"],
            "passed": bool(checks["elongation"]["carbon_coefficient"] < 0),
        })
    return {
        "coefficients": checks,
        "expectations": expectations,
        "all_passed": bool(expectations) and all(e["passed"] for e in expectations),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-validated evaluation.")
    ap.add_argument("--protocol", default="all",
                    help="protocol name or 'all' (" + ", ".join(splits.PROTOCOLS) + ")")
    ap.add_argument("--models", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--parameterisation", default=targets.DEFAULT_PARAMETERISATION,
                    choices=sorted(targets.PARAMETERISATIONS))
    ap.add_argument("--features", default="comp", choices=list(features.FEATURE_SETS))
    ap.add_argument("--rows", default="all", choices=["all", "measured", "spec_minimum"])
    ap.add_argument("--restrict-ht-rows", action="store_true",
                    help="ablation arm C: keep only rows where both HT temperatures are reported")
    ap.add_argument("--fast", action="store_true",
                    help="skip nested hyper-parameter search (for iteration, not for reporting)")
    ap.add_argument("--seed", type=int, default=splits.DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args(argv)

    df_all, comp_long = load_processed()
    df = select_rows(df_all, args.rows, args.restrict_ht_rows)

    protocol_names = list(splits.PROTOCOLS) if args.protocol == "all" else [args.protocol]
    model_names = models.LADDER if args.models == "all" else tuple(
        m.strip() for m in args.models.split(",") if m.strip()
    )
    tune = not args.fast

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"eval_{args.parameterisation}_{args.features}_{args.rows}"
    if args.restrict_ht_rows:
        stem += "_htrows"

    bundle = {
        "stem": stem,
        "rows_filter": args.rows,
        "restrict_ht_rows": bool(args.restrict_ht_rows),
        "n_rows_evaluated": int(len(df)),
        "ridge_sign_check": ridge_sign_check(df, comp_long, args.parameterisation, args.features),
        "protocols": {},
    }

    sign = bundle["ridge_sign_check"]
    print(f"rows evaluated : {len(df)}  ({args.rows}"
          + (", HT-reported only" if args.restrict_ht_rows else "") + ")")
    print(f"parameterisation: {args.parameterisation}   features: {args.features}   "
          f"tuning: {'nested' if tune else 'OFF (--fast)'}")
    print(f"ridge sign check: {'PASS' if sign['all_passed'] else 'FAIL'}")
    for e in sign["expectations"]:
        print(f"    {'ok ' if e['passed'] else 'FAIL'} {e['component']:18s} C coef = {e['observed']:+.4f}")
    print()

    for name in protocol_names:
        t0 = time.perf_counter()
        print(f"[{name}] running...", flush=True)
        res = evaluate(df, comp_long, name, model_names, args.parameterisation,
                       args.features, tune, args.seed)
        bundle["protocols"][name] = res
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s")
        for model_name, entry in res["models"].items():
            lm = entry.get("landmarks", {})
            bits = []
            for target in schema.TARGETS:
                if target in lm and lm[target]["full"].get("n"):
                    bits.append(f"{target.split('_')[0]:>9s} MAE {lm[target]['full']['mae']:7.2f}")
            print(f"    {model_name:14s} " + "   ".join(bits))
        print()

    # Split the out-of-fold predictions out of the report before serialising:
    # the JSON stays readable and diff-able, the predictions stay available.
    oof_frames = []
    for protocol_result in bundle["protocols"].values():
        for entry in protocol_result["models"].values():
            frame = entry.pop("_oof", None)
            if frame is not None:
                oof_frames.append(frame)

    path = args.out_dir / f"{stem}{tag}.json"
    path.write_text(json.dumps(metrics.round_floats(bundle), indent=2), encoding="utf-8")
    print(f"wrote {path}")

    if oof_frames:
        oof_path = args.out_dir / f"oof_{stem}{tag}.csv"
        pd.concat(oof_frames, ignore_index=True).to_csv(oof_path, index=False)
        print(f"wrote {oof_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
