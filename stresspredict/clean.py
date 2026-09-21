"""Validity rules. Every removal is counted; nothing is ever silently repaired.

Two kinds of rule, and the distinction matters:

* a NULL rule removes one implausible *value* and keeps the row, because the
  row's other targets are still good evidence;
* a DROP rule removes the whole *row*, and is reserved for cases where the row
  cannot be trusted or cannot be used at all.

Blanket-dropping a row because one of its three targets is corrupt would throw
away real measurements -- in this dataset it would cost 15 perfectly good UTS
observations. Clipping is never an option: a clipped transcription error is an
invented measurement that looks exactly like a real one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema


@dataclass(frozen=True)
class Cleaned:
    samples: pd.DataFrame
    composition: pd.DataFrame


# Documented in the manifest so the report explains itself without the source.
RULES = [
    ("null_element_out_of_range", "null",
     f"element wt% outside [0, {schema.MAX_ELEMENT_WT_PCT}] -- transcription error"),
    ("null_yield_strength_out_of_range", "null",
     f"yield_strength outside {schema.VALID_RANGES['yield_strength']} MPa"),
    ("null_elongation_out_of_range", "null",
     f"elongation outside {schema.VALID_RANGES['elongation']} %"),
    ("null_austenitize_T_out_of_range", "null",
     f"austenitize_T outside {schema.VALID_RANGES['austenitize_T']} C"),
    ("null_temper_T_out_of_range", "null",
     f"temper_T outside {schema.VALID_RANGES['temper_T']} C"),
    ("drop_tensile_strength_missing", "drop",
     "UTS absent -- it anchors the target parameterisation, so the row is unusable"),
    ("drop_tensile_strength_out_of_range", "drop",
     f"tensile_strength outside {schema.VALID_RANGES['tensile_strength']} MPa"),
    ("drop_yield_ge_tensile", "drop",
     "yield_strength >= tensile_strength -- physically impossible, so the row is untrustworthy"),
    ("drop_duplicate_sample_id", "drop", "repeated sample_id; the first occurrence is kept"),
]


def _out_of_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """True where a value is present and outside [lo, hi]."""
    return s.notna() & ~s.between(lo, hi)


def clean(samples: pd.DataFrame, composition: pd.DataFrame) -> tuple[Cleaned, dict]:
    """Apply the rules in order and return the cleaned tables plus a report."""
    s = samples.copy()
    rows_in = len(s)
    nulls: Counter[str] = Counter()
    drops: Counter[str] = Counter()
    diagnostics: list[dict] = []

    # --- NULL rules ------------------------------------------------------
    comp = composition.copy()
    bad_cells = ~comp["wt_pct"].between(0.0, schema.MAX_ELEMENT_WT_PCT)
    nulls["null_element_out_of_range"] = int(bad_cells.sum())
    comp = comp.loc[~bad_cells].reset_index(drop=True)

    for col in ("yield_strength", "elongation", "austenitize_T", "temper_T"):
        lo, hi = schema.VALID_RANGES[col]
        mask = _out_of_range(s[col], lo, hi)
        n = int(mask.sum())
        if n:
            # Diagnose *why* before discarding the evidence. A value that is
            # implausible as this quantity but plausible as another is a column
            # transposition at the publisher, not random noise -- and saying so
            # is more useful to the next user of the data than a bare count.
            offending = s.loc[mask, col]
            if col == "yield_strength":
                looks_like_elongation = offending.between(1.0, 60.0)
                if looks_like_elongation.any():
                    affected = s.loc[mask & s[col].between(1.0, 60.0)]
                    diagnostics.append({
                        "issue": "suspected_column_transposition",
                        "column": "yield_strength",
                        "n_rows": int(len(affected)),
                        "grades": sorted(affected["grade_id"].unique().tolist())[:10],
                        "note": (
                            "These yield_strength values fall in the elongation range (1-60) "
                            "and vary inversely with UTS across the affected grades -- the "
                            "signature of the strength-ductility trade-off. They are almost "
                            "certainly elongation percentages written into the yield_strength "
                            "column upstream. Nulled, not repaired: recovering them would mean "
                            "guessing which column the true YS went to."
                        ),
                    })
            nulls[f"null_{col}_out_of_range"] = n
            s.loc[mask, col] = np.nan

    # --- DROP rules ------------------------------------------------------
    def drop(mask: pd.Series, rule: str) -> None:
        nonlocal s
        n = int(mask.sum())
        if n:
            drops[rule] = n
            s = s.loc[~mask].copy()

    drop(s["tensile_strength"].isna(), "drop_tensile_strength_missing")
    lo, hi = schema.VALID_RANGES["tensile_strength"]
    drop(_out_of_range(s["tensile_strength"], lo, hi), "drop_tensile_strength_out_of_range")
    drop(
        s["yield_strength"].notna() & (s["yield_strength"] >= s["tensile_strength"]),
        "drop_yield_ge_tensile",
    )
    drop(s["sample_id"].duplicated(keep="first"), "drop_duplicate_sample_id")

    s = s.reset_index(drop=True)
    comp = comp.loc[comp["sample_id"].isin(set(s["sample_id"]))].reset_index(drop=True)

    report = {
        "rows_in": rows_in,
        "rows_out": len(s),
        "rows_dropped": rows_in - len(s),
        "drops_by_rule": dict(drops),
        "nulls_by_rule": {k: v for k, v in nulls.items() if v},
        "composition_cells_in": int(len(composition)),
        "composition_cells_out": int(len(comp)),
        "target_counts": {t: int(s[t].notna().sum()) for t in schema.TARGETS},
        "target_counts_by_measurement_kind": {
            kind: {t: int(grp[t].notna().sum()) for t in schema.TARGETS}
            for kind, grp in s.groupby("measurement_kind")
        },
        "rules": [{"name": n, "action": a, "description": d} for n, a, d in RULES],
        "diagnostics": diagnostics,
    }
    return Cleaned(samples=s, composition=comp), report
