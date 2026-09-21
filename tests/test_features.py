"""Featuriser behaviour, including the imputation semantics the ladder depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from stresspredict import features, schema


def test_dense_composition_keeps_missing_as_nan(synthetic):
    samples, comp_long = synthetic
    dense = features.dense_composition(samples, comp_long)
    assert list(dense.columns) == list(schema.ELEMENTS)
    assert len(dense) == len(samples)
    # Elements deliberately dropped in the fixture must arrive as NaN, not 0.
    assert dense[["Cu", "Al", "V"]].isna().any().all()
    assert not dense[["C", "Mn"]].isna().any().any()


def test_dense_composition_row_order_follows_samples(synthetic):
    """A silent row misalignment here would poison every downstream number."""
    samples, comp_long = synthetic
    shuffled = samples.sample(frac=1.0, random_state=0).reset_index(drop=True)
    dense = features.dense_composition(shuffled, comp_long)
    lookup = comp_long[comp_long["element"] == "C"].set_index("sample_id")["wt_pct"]
    np.testing.assert_allclose(dense["C"].to_numpy(), lookup.loc[shuffled["sample_id"]].to_numpy())


@pytest.mark.parametrize("feature_set", list(features.FEATURE_SETS))
def test_transform_emits_no_nan_and_matches_names(synthetic, feature_set):
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    f = features.CompositionFeaturizer(feature_set=feature_set).fit(X)
    out = f.transform(X)

    assert out.shape[0] == len(X)
    assert out.shape[1] == len(f.get_feature_names_out())
    assert np.isfinite(out).all(), "featuriser must not emit NaN or inf"


def test_missing_elements_become_zero_with_an_indicator(synthetic):
    """Blank means 'not added' (0.0) AND is recorded as blank -- both, not either."""
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    f = features.CompositionFeaturizer(feature_set="comp").fit(X)
    out = pd.DataFrame(f.transform(X), columns=f.get_feature_names_out())

    missing_cu = X["Cu"].isna().to_numpy()
    assert missing_cu.any()
    assert (out.loc[missing_cu, "Cu"] == 0.0).all()
    assert "missing__Cu" in out.columns
    np.testing.assert_array_equal(out["missing__Cu"].to_numpy(), missing_cu.astype(float))


def test_temperatures_impute_to_median_not_zero(synthetic):
    """0 C is not a plausible austenitising temperature and would wreck ridge."""
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    f = features.CompositionFeaturizer(feature_set="comp_ht").fit(X)
    out = pd.DataFrame(f.transform(X), columns=f.get_feature_names_out())

    filled = out.loc[X["austenitize_T"].isna(), "austenitize_T"]
    assert len(filled) > 0
    assert (filled > 500).all(), "temperatures must not be zero-filled"
    assert np.allclose(filled, f.process_medians_["austenitize_T"])


def test_imputation_statistics_come_only_from_fit(synthetic):
    """Medians must be learned on the training fold and reused, never recomputed."""
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    train, test = X.iloc[:200], X.iloc[200:]

    f = features.CompositionFeaturizer(feature_set="comp_ht").fit(train)
    median_from_train = f.process_medians_["austenitize_T"]
    out = pd.DataFrame(f.transform(test), columns=f.get_feature_names_out())
    filled = out.loc[test["austenitize_T"].isna().to_numpy(), "austenitize_T"]
    assert np.allclose(filled, median_from_train)


def test_comp_raw_omits_derived_terms(synthetic):
    """The ablation arm must actually differ, or the comparison measures nothing."""
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    raw = features.CompositionFeaturizer(feature_set="comp_raw").fit(X)
    full = features.CompositionFeaturizer(feature_set="comp").fit(X)

    assert "ce_iiw" not in raw.get_feature_names_out()
    assert "ce_iiw" in full.get_feature_names_out()
    assert len(full.get_feature_names_out()) == len(raw.get_feature_names_out()) + len(
        features.CompositionFeaturizer.DERIVED_NAMES
    )


def test_comp_ht_adds_process_columns(synthetic):
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    names = features.CompositionFeaturizer(feature_set="comp_ht").fit(X).get_feature_names_out()
    assert "austenitize_T" in names and "temper_T" in names
    names_comp = features.CompositionFeaturizer(feature_set="comp").fit(X).get_feature_names_out()
    assert "austenitize_T" not in names_comp


def test_metallurgical_formulas_are_correct():
    """Spot-check the derived terms against hand arithmetic."""
    el = pd.DataFrame([{ "C": 0.4, "Mn": 0.9, "Si": 0.25, "Cr": 1.0,
                         "Ni": 0.2, "Mo": 0.2, "V": 0.05, "Cu": 0.1, "Al": 0.03 }])
    d = features.CompositionFeaturizer._derive(el)
    assert d["ce_iiw"].iloc[0] == pytest.approx(0.4 + 0.9 / 6 + (1.0 + 0.2 + 0.05) / 5
                                                + (0.2 + 0.1) / 15)
    assert d["carbide_former"].iloc[0] == pytest.approx(1.25)
    assert d["total_alloy"].iloc[0] == pytest.approx(3.13)
    assert d["fe_balance"].iloc[0] == pytest.approx(96.87)
    assert d["cr_eq_schaeffler"].iloc[0] == pytest.approx(1.0 + 0.2 + 1.5 * 0.25)
    assert d["ni_eq_schaeffler"].iloc[0] == pytest.approx(0.2 + 30 * 0.4 + 0.5 * 0.9)


def test_featurizer_is_clonable_for_cross_validation(synthetic):
    """sklearn clones the transformer per fold; unfitted state must survive it."""
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    f = features.CompositionFeaturizer(feature_set="comp").fit(X)
    fresh = clone(f)
    assert fresh.feature_set == "comp"
    assert not hasattr(fresh, "process_medians_")


def test_rejects_unknown_feature_set(synthetic):
    samples, comp_long = synthetic
    X = features.design_frame(samples, comp_long)
    with pytest.raises(ValueError):
        features.CompositionFeaturizer(feature_set="nonsense").fit(X)
