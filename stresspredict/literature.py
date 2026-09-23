"""Row selection for the literature-derived steel dataset (figshare 32755830 v2).

The dataset was produced by automated extraction from about 2,900 papers. Its own
quality flags pass every row, and its authors ask users to filter further before
quantitative use. This module is that filter. Unlike `clean`, which applies
validity rules to any source, these are *selection* rules specific to this
source: they decide which of its records answer the question this project asks
(room-temperature tensile landmarks of a steel of known wt% chemistry). Every
step is counted and lands in the manifest.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import schema

ELEMENT_COLUMNS = (
    "Co", "Ni", "Al", "W", "Ti", "Cr", "Ta", "B", "Mo", "Re", "Nb", "Si", "V", "Fe", "Cu", "Hf",
    "Ru", "Ir", "C", "Pt", "H", "N", "O", "F", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
    "Mn", "Zn", "Ge", "As", "Sr", "Y", "Zr", "Te", "Pd", "Ag", "Cd", "Sn", "Sb", "Ba", "La", "Ce",
    "Nd", "Er", "Os", "Pb", "Bi",
)
PROPERTIES = {"tensile strength": "tensile_strength", "yield strength": "yield_strength",
              "total elongation": "elongation"}

_WT_UNIT = re.compile(r"^(wt|mass|weight)", re.I)
_RT = re.compile(r"room[- ]temperature|ambient temperature|\bat ambient\b|\bRT\b"
                 r"|\b(?:2[0-7]|20)\s?°\s?C\b|\b(?:29[3-9]|300)\s?K\b", re.I)
_TENSILE = re.compile(r"tensil|uniaxial tension|tension test", re.I)
# Anything that makes the number something other than a plain RT tensile landmark.
_EXCLUDE = re.compile(
    r"hydrogen|charg|corros|nacl|seawater|immers|irradiat|neutron|fatigue|cyclic|creep|compress|"
    r"punch|impact|charpy|bend|torsion|shear|hot |elevated|cryogen|liquid (?:nitrogen|helium)|"
    r"\b77\s?K|pre-?strain|prestrain|notch|weld|high[- ]strain[- ]rate|dynamic|split hopkinson|"
    r"exposure|atmospher|oxid|aged at \d{3}", re.I)
_TEMPERATURE = re.compile(r"(-?\d{2,4})\s?°\s?C|(\d{2,4})\s?K\b")

# Held-out protection: a training row this close to a held-out steel is dropped.
HELDOUT_GUARD_WT_PCT = 0.5
_GUARD_ELEMENTS = ("C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Al", "Co", "Ti")


def parse_wt_pct(value) -> float:
    """'4.0-8.0' -> 6.0 (the nominal midpoint); '<0.01', 'Nil', '-', 'Bal.' -> NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)", text)
    return (float(m.group(1)) + float(m.group(2))) / 2 if m else np.nan


def _non_ambient_temperature(text: str) -> bool:
    for m in _TEMPERATURE.finditer(text):
        if m.group(1) is not None and not 15 <= float(m.group(1)) <= 30:
            return True
        if m.group(2) is not None and not 288 <= float(m.group(2)) <= 303:
            return True
    return False


def records(raw: pd.DataFrame) -> pd.DataFrame:
    """Property-level rows -> one row per sample, with parsed wt% columns."""
    p = raw[raw["property"].isin(PROPERTIES)]
    values = p.pivot_table(index="record_id", columns="property",
                           values="property_value_numeric", aggfunc="first")
    values = values.rename(columns=PROPERTIES)
    meta_cols = ["article_doi_normalized", "steel name", "composition unit",
                 "test route/condition", "synthesis and processing routes", *ELEMENT_COLUMNS]
    meta = p.groupby("record_id")[meta_cols].first()
    for el in ELEMENT_COLUMNS:
        meta[el] = meta[el].map(parse_wt_pct)
    out = meta.join(values)
    for t in schema.TARGETS:
        if t not in out:
            out[t] = np.nan
    return out


def select(df: pd.DataFrame, heldout_composition: pd.DataFrame | None = None
           ) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the selection rules in order, counting what each one removes."""
    steps: list[dict] = [{"rule": "records with at least one of UTS / YS / EL", "rows_after": len(df)}]

    def step(rule: str, mask: pd.Series) -> None:
        nonlocal df
        before = len(df)
        df = df[mask.reindex(df.index, fill_value=False)]
        steps.append({"rule": rule, "rows_removed": before - len(df), "rows_after": len(df)})

    step("composition unit is wt% / mass%",
         df["composition unit"].astype(str).str.strip().str.match(_WT_UNIT))
    step("carbon reported", df["C"].notna())
    comp = df[list(ELEMENT_COLUMNS)]
    non_fe = comp.drop(columns="Fe").fillna(0.0).sum(axis=1)
    step("plausible steel: non-Fe total <= 50 wt%, C <= 2.1, Fe (if given) >= 50",
         (non_fe <= 50) & (comp["C"] <= 2.1) & (comp["Fe"].isna() | (comp["Fe"] >= 50)))
    test = df["test route/condition"].fillna("").astype(str)
    step("test stated as room temperature", test.str.contains(_RT))
    test = df["test route/condition"].fillna("").astype(str)
    proc = df["synthesis and processing routes"].fillna("").astype(str)
    step("plain tensile test: no hydrogen, corrosion, irradiation, fatigue, compression, "
         "pre-strain, notch or weld context",
         test.str.contains(_TENSILE) & ~test.str.contains(_EXCLUDE)
         & ~proc.str.contains(r"hydrogen charg|irradiat", case=False))
    test = df["test route/condition"].fillna("").astype(str)
    step("no second, non-ambient temperature in the test text", ~test.map(_non_ambient_temperature))
    if heldout_composition is not None:
        a = np.nan_to_num(df[list(_GUARD_ELEMENTS)].to_numpy(float))
        b = np.nan_to_num(heldout_composition.reindex(columns=list(_GUARD_ELEMENTS)).to_numpy(float))
        dist = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2).min(axis=1) if len(a) else a[:, 0]
        step(f"not within {HELDOUT_GUARD_WT_PCT} wt% of a held-out steel (leakage guard)",
             pd.Series(dist > HELDOUT_GUARD_WT_PCT, index=df.index))
    return df, steps
