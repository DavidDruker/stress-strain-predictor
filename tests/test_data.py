"""Units, cleaning, grouping and the noise-floor measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stresspredict import clean, grouping, noise_floor, units


# --- units -------------------------------------------------------------
def test_ksi_mpa_round_trip():
    for v in (0.0, 1.0, 36.0, 150.0, 300.0):
        assert units.mpa_to_ksi(units.ksi_to_mpa(v)) == pytest.approx(v, rel=1e-12)
    assert units.ksi_to_mpa(100.0) == pytest.approx(689.4757, abs=1e-3)


def test_atomic_to_weight_percent_balances_with_iron():
    wt = units.atomic_to_weight_percent({"C": 1.0, "Mn": 1.0})
    # Carbon is light, so its wt% must come out below its at%.
    assert wt["C"] < 1.0
    assert wt["Mn"] == pytest.approx(0.986, abs=0.02)
    assert all(v > 0 for v in wt.values())
    assert "Fe" not in wt


def test_atomic_to_weight_percent_rejects_impossible_input():
    with pytest.raises(ValueError):
        units.atomic_to_weight_percent({"C": 60.0, "Mn": 60.0})


@pytest.mark.parametrize("raw,expected", [
    ("A5", "A5"), ("5d", "A5"), ("A 50 mm", "A50mm"), ("A80", "A80mm"),
    ("A4", "A4"), (None, "unknown"), ("whatever", "unknown"), ("", "unknown"),
])
def test_elongation_standard_tagging_never_guesses(raw, expected):
    assert units.tag_elongation_standard(raw) == expected


# --- cleaning ----------------------------------------------------------
def _frame(**overrides) -> pd.DataFrame:
    base = {
        "sample_id": ["a", "b", "c", "d"],
        "grade_id": ["G1", "G1", "G2", "G3"],
        "source_id": "t", "source_label": "t", "provenance": "t",
        "steel_family": "carbon", "measurement_kind": "measured",
        "austenitize_T": [900.0, 900.0, np.nan, 900.0],
        "temper_T": [600.0, np.nan, np.nan, 600.0],
        "yield_strength": [300.0, 400.0, 500.0, 600.0],
        "tensile_strength": [500.0, 600.0, 700.0, 800.0],
        "elongation": [20.0, 25.0, 15.0, 10.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _comp(ids=("a", "b", "c", "d")) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": list(ids) * 2,
        "element": ["C"] * len(ids) + ["Mn"] * len(ids),
        "wt_pct": [0.2] * len(ids) + [1.0] * len(ids),
    })


def test_drops_rows_where_yield_meets_or_exceeds_tensile():
    df = _frame(yield_strength=[300.0, 600.0, 500.0, 900.0])  # b: YS == UTS, d: YS > UTS
    cleaned, report = clean.clean(df, _comp())
    assert report["drops_by_rule"]["drop_yield_ge_tensile"] == 2
    assert set(cleaned.samples["sample_id"]) == {"a", "c"}
    assert report["rows_out"] == 2


def test_implausible_yield_is_nulled_not_dropped():
    """A bad YS must not cost the row's perfectly good UTS."""
    df = _frame(yield_strength=[10.0, 400.0, 500.0, 600.0])
    cleaned, report = clean.clean(df, _comp())
    assert report["nulls_by_rule"]["null_yield_strength_out_of_range"] == 1
    assert report["rows_out"] == 4, "the row must survive"
    row = cleaned.samples.set_index("sample_id").loc["a"]
    assert pd.isna(row["yield_strength"])
    assert row["tensile_strength"] == 500.0


def test_transposed_column_is_diagnosed_not_silently_dropped():
    df = _frame(yield_strength=[12.0, 15.0, 500.0, 600.0])
    _, report = clean.clean(df, _comp())
    diags = [d for d in report["diagnostics"] if d["issue"] == "suspected_column_transposition"]
    assert len(diags) == 1
    assert diags[0]["n_rows"] == 2
    assert diags[0]["column"] == "yield_strength"


def test_out_of_range_tensile_drops_the_row():
    df = _frame(tensile_strength=[50.0, 600.0, 700.0, 800.0])
    cleaned, report = clean.clean(df, _comp())
    assert report["drops_by_rule"]["drop_tensile_strength_out_of_range"] == 1
    assert "a" not in set(cleaned.samples["sample_id"])


def test_missing_tensile_drops_the_row():
    df = _frame(tensile_strength=[np.nan, 600.0, 700.0, 800.0])
    _, report = clean.clean(df, _comp())
    assert report["drops_by_rule"]["drop_tensile_strength_missing"] == 1


def test_duplicate_sample_ids_are_dropped_once():
    df = _frame(sample_id=["a", "a", "c", "d"])
    cleaned, report = clean.clean(df, _comp(ids=("a", "a", "c", "d")))
    assert report["drops_by_rule"]["drop_duplicate_sample_id"] == 1
    assert len(cleaned.samples) == 3


def test_cross_source_duplicate_keeps_the_earlier_sources_copy():
    """One datasheet row reached through two publishers must count once."""
    df = _frame(source_id=["sb", "sb", "me", "me"],
                tensile_strength=[500.0, 600.0, 500.0, 900.0],
                yield_strength=[300.0, 400.0, 300.0, 600.0],
                elongation=[20.0, 25.0, 20.0, 10.0])
    cleaned, report = clean.clean(df, _comp())
    assert report["drops_by_rule"]["drop_cross_source_duplicate"] == 1
    assert list(cleaned.samples["sample_id"]) == ["a", "b", "d"]


def test_repeats_within_one_source_are_not_cross_source_duplicates():
    """Two tempers reporting the same numbers inside one source are both real rows."""
    df = _frame(tensile_strength=[500.0, 500.0, 700.0, 800.0],
                yield_strength=[300.0, 300.0, 500.0, 600.0],
                elongation=[20.0, 20.0, 15.0, 10.0])
    _, report = clean.clean(df, _comp())
    assert "drop_cross_source_duplicate" not in report["drops_by_rule"]


def test_out_of_range_element_cell_is_removed():
    comp = _comp()
    comp.loc[0, "wt_pct"] = 500.0
    _, report = clean.clean(_frame(), comp)
    assert report["nulls_by_rule"]["null_element_out_of_range"] == 1
    assert report["composition_cells_out"] == report["composition_cells_in"] - 1


def test_composition_rows_follow_dropped_samples():
    df = _frame(yield_strength=[900.0, 400.0, 500.0, 600.0])  # row a is impossible
    cleaned, _ = clean.clean(df, _comp())
    assert "a" not in set(cleaned.composition["sample_id"])


def test_every_row_is_accounted_for():
    """rows_in == rows_out + rows_dropped, always. The manifest must balance."""
    df = _frame(yield_strength=[10.0, 700.0, 500.0, 600.0])
    _, report = clean.clean(df, _comp())
    assert report["rows_in"] == report["rows_out"] + report["rows_dropped"]
    assert report["rows_dropped"] == sum(report["drops_by_rule"].values())


def test_clean_does_not_mutate_its_inputs():
    df, comp = _frame(yield_strength=[10.0, 400.0, 500.0, 900.0]), _comp()
    before_df, before_comp = df.copy(), comp.copy()
    clean.clean(df, comp)
    pd.testing.assert_frame_equal(df, before_df)
    pd.testing.assert_frame_equal(comp, before_comp)


# --- grouping ----------------------------------------------------------
def test_identical_chemistry_hashes_together():
    comp = pd.DataFrame({
        "sample_id": ["a", "a", "b", "b"],
        "element": ["C", "Mn", "C", "Mn"],
        "wt_pct": [0.2, 1.0, 0.2000001, 1.0],  # below the rounding precision
    })
    h = grouping.composition_hash(comp)
    assert h["a"] == h["b"]


def test_different_chemistry_hashes_apart():
    comp = pd.DataFrame({
        "sample_id": ["a", "a", "b", "b"],
        "element": ["C", "Mn", "C", "Mn"],
        "wt_pct": [0.20, 1.0, 0.25, 1.0],
    })
    h = grouping.composition_hash(comp)
    assert h["a"] != h["b"]


def test_absent_element_is_not_the_same_as_zero():
    """'V not measured' and 'V = 0.00' are different states of knowledge."""
    comp = pd.DataFrame({
        "sample_id": ["a", "a", "b"],
        "element": ["C", "V", "C"],
        "wt_pct": [0.2, 0.0, 0.2],
    })
    h = grouping.composition_hash(comp)
    assert h["a"] != h["b"]


def test_attach_groups_falls_back_when_grade_is_missing():
    samples = _frame(grade_id=["", "nan", "G2", "G3"])
    out = grouping.attach_groups(samples, _comp())
    assert out.loc[0, "grade_group"].startswith("comp:")
    assert out.loc[1, "grade_group"].startswith("comp:")
    assert out.loc[2, "grade_group"] == "G2"
    # a and b share chemistry, so they must land in the SAME fallback group
    assert out.loc[0, "grade_group"] == out.loc[1, "grade_group"]


# --- noise floor -------------------------------------------------------
def test_noise_floor_recovers_a_known_spread():
    """Two rows per composition, 100 MPa apart: the floor must be 50 in-sample."""
    samples = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "grade_id": ["G", "G", "H", "H"],
        "source_id": "t", "source_label": "t", "provenance": "t",
        "steel_family": "carbon", "measurement_kind": "measured",
        "yield_strength": [400.0, 500.0, 600.0, 700.0],
        "tensile_strength": [700.0, 800.0, 900.0, 1000.0],
        "elongation": [20.0, 20.0, 10.0, 10.0],
    })
    comp = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "element": ["C"] * 4,
        "wt_pct": [0.2, 0.2, 0.5, 0.5],
    })
    floor = noise_floor.compute(samples, comp)
    ys = floor["targets"]["yield_strength"]["measured_only"]
    assert ys["n_groups"] == 2 and ys["n_rows"] == 4
    assert ys["mae_floor_insample"] == pytest.approx(50.0)
    assert ys["mae_floor_loo"] == pytest.approx(100.0)
    # elongation is constant within each composition -> a floor of exactly zero
    assert floor["targets"]["elongation"]["measured_only"]["mae_floor_insample"] == 0.0


def test_noise_floor_separates_measurement_kinds():
    """The spec-minimum stratum must be reported apart from the measured one."""
    samples = pd.DataFrame({
        "sample_id": list("abcd"),
        "grade_id": ["G", "G", "H", "H"],
        "source_id": "t", "source_label": "t", "provenance": ["t1", "t1", "t2", "t2"],
        "steel_family": "carbon",
        "measurement_kind": ["measured", "measured", "spec_minimum", "spec_minimum"],
        "yield_strength": [400.0, 600.0, 500.0, 500.0],
        "tensile_strength": [700.0, 900.0, 800.0, 800.0],
        "elongation": [20.0, 20.0, 10.0, 10.0],
    })
    comp = pd.DataFrame({"sample_id": list("abcd"), "element": ["C"] * 4,
                         "wt_pct": [0.2, 0.2, 0.2, 0.2]})
    floor = noise_floor.compute(samples, comp)
    strata = floor["targets"]["yield_strength"]
    assert strata["measured_only"]["mae_floor_insample"] == pytest.approx(100.0)
    assert strata["spec_minimum_only"]["mae_floor_insample"] == pytest.approx(0.0)
    # Pooling the two halves the apparent floor -- the exact trap being guarded.
    assert strata["all_rows"]["mae_floor_insample"] < strata["measured_only"]["mae_floor_insample"]
    assert floor["headline_stratum"] == "measured_only"


def test_noise_floor_ignores_singleton_compositions():
    samples = pd.DataFrame({
        "sample_id": ["a", "b"],
        "grade_id": ["G", "H"],
        "source_id": "t", "source_label": "t", "provenance": "t",
        "steel_family": "carbon", "measurement_kind": "measured",
        "yield_strength": [400.0, 900.0],
        "tensile_strength": [700.0, 1200.0],
        "elongation": [20.0, 10.0],
    })
    comp = pd.DataFrame({"sample_id": ["a", "b"], "element": ["C", "C"], "wt_pct": [0.2, 0.9]})
    floor = noise_floor.compute(samples, comp)
    assert floor["targets"]["yield_strength"]["measured_only"]["n_groups"] == 0
