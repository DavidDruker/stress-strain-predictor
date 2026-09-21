"""Canonical schema shared by every module.

One place defines what a column is called, so adding a second alloy system in a
later phase means writing a new loader -- not renaming columns in nine files.
"""

from __future__ import annotations

# The nine elements SteelBench reports. Order is load-bearing: the feature
# matrix, the composition hash and the range guard all rely on it.
ELEMENTS: tuple[str, ...] = ("C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "Al")

# Heat-treatment columns. Not used by the shipped v1 model (composition-only);
# carried through ingest so the Phase 2 ablation needs no re-ingest.
PROCESS_NUMERIC: tuple[str, ...] = ("austenitize_T", "temper_T")

TARGETS: tuple[str, ...] = ("yield_strength", "tensile_strength", "elongation")

TARGET_UNITS = {
    "yield_strength": "MPa",
    "tensile_strength": "MPa",
    "elongation": "%",
}

# Identity / grouping columns carried on every sample.
ID_COLS: tuple[str, ...] = (
    "sample_id",      # globally unique, prefixed by source_id
    "source_id",      # which loader produced this row
    "grade_id",       # steel grade designation
    "source_label",   # publisher's own source name
    "provenance",     # publisher's provenance tier
    "steel_family",   # alloy family
    "measurement_kind",  # "measured" | "spec_minimum"  <- see data/README.md
)

# Rounding used when hashing a composition into a group key. SteelBench reports
# to at most 4 dp; 3 dp collapses float noise without merging real chemistries.
COMPOSITION_ROUND_DP = 3

# Physical validity bounds. Deliberately wide -- these catch transcription
# errors, not unusual alloys. Every rejection is counted in the manifest.
VALID_RANGES = {
    "tensile_strength": (150.0, 2500.0),
    "yield_strength": (100.0, 2200.0),
    "elongation": (0.5, 80.0),
    "austenitize_T": (500.0, 1300.0),
    "temper_T": (100.0, 800.0),
}

# Upper bound on any single element, wt%. Fe is the balance and is not stored.
MAX_ELEMENT_WT_PCT = 70.0
