"""Checks against the real SteelBench release.

Skipped when the data is absent (it is gitignored), so CI stays green on a clean
checkout while these still guard the numbers quoted in the README and docs.

The expected counts are those of the OPEN release actually deposited on Zenodo
(1,360 rows / 562 grades), NOT the 1,636 rows / 594 grades that the record
description and dataset card advertise for full SteelBench. The MMPDS and
internal-lab tiers are withheld for licensing. Asserting the advertised figure
would fail against the only file anyone can download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stresspredict import evaluate, features, schema, splits

MANIFEST = Path("data/processed/data_manifest.json")

RAW_ROWS_OPEN_RELEASE = 1360
GRADES_OPEN_RELEASE = 562
FAMILIES = 17


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("run `python -m stresspredict.ingest --source steelbench` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_ingest_loads_the_open_release(manifest):
    assert manifest["raw_rows"] == RAW_ROWS_OPEN_RELEASE
    assert manifest["source_id"] == "steelbench_v1_open"


def test_manifest_accounts_for_every_dropped_row(manifest):
    c = manifest["cleaning"]
    assert c["rows_in"] == c["rows_out"] + c["rows_dropped"]
    assert c["rows_dropped"] == sum(c["drops_by_rule"].values())
    for rule in c["drops_by_rule"]:
        assert any(r["name"] == rule for r in c["rules"]), f"{rule} is undocumented"
    for rule in c["nulls_by_rule"]:
        assert any(r["name"] == rule for r in c["rules"]), f"{rule} is undocumented"


def test_expected_columns_are_present(real_data):
    df, comp_long = real_data
    for col in schema.TARGETS + schema.PROCESS_NUMERIC + schema.ID_COLS:
        assert col in df.columns, f"missing {col}"
    assert set(comp_long["element"]) <= set(schema.ELEMENTS)


def test_grade_and_family_counts(real_data):
    df, _ = real_data
    assert df["grade_group"].nunique() == GRADES_OPEN_RELEASE
    assert df["family_group"].nunique() == FAMILIES


def test_targets_are_not_co_present(real_data):
    """Guards the design choice in targets.py.

    If UTS ever became as sparse as YS, the ratio parameterisation would lose the
    advantage it was chosen for and the default should be revisited.
    """
    df, _ = real_data
    n_uts = int(df["tensile_strength"].notna().sum())
    n_ys = int(df["yield_strength"].notna().sum())
    assert n_uts > n_ys, "ratio parameterisation assumes UTS is the best-populated target"
    assert n_uts >= 1300 and n_ys >= 900


def test_no_row_violates_the_physical_invariant_after_cleaning(real_data):
    df, _ = real_data
    both = df["yield_strength"].notna() & df["tensile_strength"].notna()
    assert (df.loc[both, "tensile_strength"] > df.loc[both, "yield_strength"]).all()


def test_target_values_are_inside_the_declared_ranges(real_data):
    df, _ = real_data
    for target in schema.TARGETS:
        lo, hi = schema.VALID_RANGES[target]
        present = df[target].dropna()
        assert present.between(lo, hi).all(), f"{target} escaped its declared range"


def test_noise_floor_is_measured_and_stratified(manifest):
    """The headline measurement must exist, and be the measured-only stratum."""
    floor = manifest["noise_floor"]
    assert floor["headline_stratum"] == "measured_only"
    ys = floor["targets"]["yield_strength"]["measured_only"]
    assert ys["n_groups"] > 0
    assert ys["mae_floor_insample"] > 0
    assert ys["mae_floor_loo"] >= ys["mae_floor_insample"]


def test_pooling_spec_minima_deflates_the_floor(manifest):
    """The trap this project exists to avoid, asserted on the real data.

    Specification minima are near-deterministic per grade. If pooling them did
    NOT deflate the measured spread, the stratification would be unnecessary --
    so this test documents why it is there.
    """
    ys = manifest["noise_floor"]["targets"]["yield_strength"]
    assert ys["spec_minimum_only"]["mae_floor_insample"] < ys["measured_only"]["mae_floor_insample"]
    assert ys["all_rows"]["mae_floor_insample"] < ys["measured_only"]["mae_floor_insample"]


def test_lofo_holds_out_twelve_families(real_data):
    """Reproduces the k=12 that SteelBench's own dataset card specifies."""
    df, _ = real_data
    groups = df["family_group"].to_numpy(dtype=object)
    splitter = splits.MinSizeLeaveOneGroupOut(min_test_rows=10)
    assert splitter.get_n_splits(groups=groups) == 12


def test_transposed_yield_column_is_caught(manifest):
    """The EMK0319/0320/0321 elongation-in-yield-strength rows."""
    diags = manifest["cleaning"]["diagnostics"]
    transposition = [d for d in diags if d["issue"] == "suspected_column_transposition"]
    assert transposition, "the known upstream transposition should be diagnosed"
    assert transposition[0]["n_rows"] >= 10


def test_grouped_protocol_never_leaks_a_grade_on_real_data(real_data):
    df, comp_long = real_data
    protocol = splits.get("gkf_grade")
    X = features.design_frame(df, comp_long)
    groups = protocol.groups(df)
    for train_idx, test_idx in protocol.splitter().split(X, None, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


@pytest.mark.slow
def test_ridge_signs_are_metallurgically_correct(real_data):
    """Carbon raises strength and lowers ductility. If not, the pipeline is broken."""
    df, comp_long = real_data
    check = evaluate.ridge_sign_check(df, comp_long, "ratio", "comp")
    assert check["all_passed"], check["expectations"]
