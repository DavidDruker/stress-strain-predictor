"""The measurement that decides what this project can honestly claim.

Rows sharing an exact composition differ only in processing history. A
composition-only model sees one feature vector for all of them and can emit only
one number, so the spread *within* a composition group is a hard lower bound on
its achievable MAE. No amount of model capacity crosses it.

Two estimates are reported because neither alone is honest:

* `mae_floor_insample` predicts each row with its own group's median. The median
  is fitted on the row it scores, so this is biased LOW.
* `mae_floor_loo` predicts each row with the median of the *other* rows in its
  group. That is an out-of-sample estimate, but at the small group sizes here it
  is fitted on n-1 points and so is biased HIGH.

The true floor is bracketed between them. Reporting one number would be a
cleaner story and a worse measurement.

Stratification by `measurement_kind` is not optional. 55% of SteelBench's open
release is specification MINIMA rather than measurements; those are near
deterministic given a grade, and pooling them in roughly halves the apparent
floor -- turning the project's headline number into an artefact of how the
benchmark was assembled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import grouping, schema


def _floor_for(values_by_group: list[np.ndarray]) -> dict:
    in_dev, loo_dev = [], []
    for v in values_by_group:
        med = float(np.median(v))
        in_dev.extend(np.abs(v - med))
        for i in range(len(v)):
            others = np.delete(v, i)
            loo_dev.append(abs(v[i] - float(np.median(others))))
    return {
        "n_groups": len(values_by_group),
        "n_rows": int(sum(len(v) for v in values_by_group)),
        "mae_floor_insample": round(float(np.mean(in_dev)), 2),
        "mae_floor_loo": round(float(np.mean(loo_dev)), 2),
        "median_group_range": round(
            float(np.median([v.max() - v.min() for v in values_by_group])), 2
        ),
        "max_group_range": round(float(np.max([v.max() - v.min() for v in values_by_group])), 2),
    }


def compute(
    samples: pd.DataFrame,
    composition_long: pd.DataFrame,
    min_group_size: int = 2,
) -> dict:
    """Within-composition spread per target, stratified by measurement kind."""
    df = grouping.attach_groups(samples, composition_long)

    strata = {
        "all_rows": df,
        "measured_only": df[df["measurement_kind"] == "measured"],
        "spec_minimum_only": df[df["measurement_kind"] == "spec_minimum"],
    }
    for tier, grp in df.groupby("provenance"):
        strata[f"provenance:{tier}"] = grp

    out: dict = {
        "method": (
            "Rows grouped by exact reported composition (rounded to "
            f"{schema.COMPOSITION_ROUND_DP} dp). Groups with >= {min_group_size} rows only. "
            "Units follow the target: MPa for strengths, percentage points for elongation."
        ),
        "headline_stratum": "measured_only",
        "targets": {},
    }

    for target in schema.TARGETS:
        per_stratum = {}
        for name, sub in strata.items():
            sub = sub[sub[target].notna()]
            groups = [
                g[target].to_numpy(dtype=float)
                for _, g in sub.groupby("comp_hash")
                if len(g) >= min_group_size
            ]
            per_stratum[name] = (
                _floor_for(groups) if groups
                else {"n_groups": 0, "n_rows": 0, "note": "no repeated compositions"}
            )
        out["targets"][target] = per_stratum

    return out


def summarise(floor: dict) -> str:
    """One readable block for the console and the data-quality report."""
    lines = ["Within-composition noise floor (hard lower bound on composition-only MAE):"]
    for target, strata in floor["targets"].items():
        unit = schema.TARGET_UNITS[target]
        lines.append(f"  {target} [{unit}]")
        for name in ("measured_only", "spec_minimum_only", "all_rows"):
            s = strata.get(name, {})
            if not s.get("n_groups"):
                lines.append(f"      {name:20s} -- no repeated compositions")
                continue
            lines.append(
                f"      {name:20s} floor {s['mae_floor_insample']:7.2f} .. "
                f"{s['mae_floor_loo']:7.2f}   "
                f"({s['n_groups']} groups, {s['n_rows']} rows)"
            )
    return "\n".join(lines)
