"""Evaluation harness behaviour -- including the tuning schedule regression."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from stresspredict import evaluate, features, splits, targets


def test_tuning_actually_uses_the_training_fold(monkeypatch, synthetic_grouped):
    """The inner search must reach the full training fold on its last rung.

    HalvingRandomSearchCV defaults to min_resources="smallest", which on a
    dataset this size starts at 6 rows and picks hyper-parameters from 6 then 18
    samples. That is worse than not tuning at all, while still looking like a
    nested search from the outside and still costing the time. This test fails
    if that default is ever restored.
    """
    df, comp_long = synthetic_grouped
    component = targets.RATIO["tensile_strength"]
    X = features.design_frame(df, comp_long)
    y = component.quantity(df).to_numpy(dtype=float)
    groups = df["grade_group"].to_numpy(dtype=object)

    captured: list = []

    class Recording(HalvingRandomSearchCV):
        def fit(self, *args, **kwargs):
            out = super().fit(*args, **kwargs)
            captured.append(self)
            return out

    monkeypatch.setattr(evaluate, "HalvingRandomSearchCV", Recording)
    monkeypatch.setitem(
        evaluate.models.PARAM_SPACES, "hist_gbm",
        {"regressor__regressor__model__max_leaf_nodes": [7, 15]},
    )

    evaluate._fit_one("hist_gbm", component, "comp", X, y, groups, tune=True, seed=1)

    assert captured, "no search was run"
    search = captured[0]
    assert search.min_resources == "exhaust"
    assert search.n_resources_[-1] == len(X), (
        f"last rung trained on {search.n_resources_[-1]} of {len(X)} rows"
    )
    assert search.n_resources_[0] > 50, "the first rung is too small to select on"


def test_inner_search_is_grouped(monkeypatch, synthetic_grouped):
    """Tuning must not leak grades, even under the leaky outer protocol."""
    from sklearn.model_selection import GroupKFold

    df, comp_long = synthetic_grouped
    component = targets.RATIO["tensile_strength"]
    X = features.design_frame(df, comp_long)
    y = component.quantity(df).to_numpy(dtype=float)
    groups = df["grade_group"].to_numpy(dtype=object)

    captured: list = []

    class Recording(HalvingRandomSearchCV):
        def fit(self, X_, y_, **kwargs):
            captured.append((self.cv, kwargs.get("groups")))
            return super().fit(X_, y_, **kwargs)

    monkeypatch.setattr(evaluate, "HalvingRandomSearchCV", Recording)
    monkeypatch.setitem(
        evaluate.models.PARAM_SPACES, "hist_gbm",
        {"regressor__regressor__model__max_leaf_nodes": [7, 15]},
    )
    evaluate._fit_one("hist_gbm", component, "comp", X, y, groups, tune=True, seed=1)

    cv, passed_groups = captured[0]
    assert isinstance(cv, GroupKFold)
    assert passed_groups is not None, "groups were not handed to the inner splitter"
    np.testing.assert_array_equal(passed_groups, groups)


def test_untuned_path_skips_the_search(monkeypatch, synthetic_grouped):
    df, comp_long = synthetic_grouped
    component = targets.RATIO["tensile_strength"]
    X = features.design_frame(df, comp_long)
    y = component.quantity(df).to_numpy(dtype=float)
    groups = df["grade_group"].to_numpy(dtype=object)

    def explode(*a, **k):
        raise AssertionError("--fast must not run a search")

    monkeypatch.setattr(evaluate, "HalvingRandomSearchCV", explode)
    fitted, best = evaluate._fit_one("hist_gbm", component, "comp", X, y, groups,
                                     tune=False, seed=1)
    assert best is None
    assert np.isfinite(fitted.predict(X.iloc[:5])).all()


def test_row_selection_filters(synthetic_grouped):
    df, _ = synthetic_grouped
    assert len(evaluate.select_rows(df, "all", False)) == len(df)

    measured = evaluate.select_rows(df, "measured", False)
    assert (measured["measurement_kind"] == "measured").all()
    assert 0 < len(measured) < len(df)

    ht = evaluate.select_rows(df, "all", True)
    assert ht["austenitize_T"].notna().all() and ht["temper_T"].notna().all()
    assert len(ht) < len(df)

    with pytest.raises(ValueError):
        evaluate.select_rows(df, "nonsense", False)


def test_evaluate_produces_landmarks_and_upholds_the_invariant(synthetic_grouped):
    df, comp_long = synthetic_grouped
    result = evaluate.evaluate(df, comp_long, "gkf_grade", ("ridge",), "ratio",
                               "comp", tune=False, seed=1)
    entry = result["models"]["ridge"]

    assert entry["invariant_uts_gt_ys"]["holds"] is True
    assert entry["invariant_uts_gt_ys"]["violations"] == 0
    for target in ("yield_strength", "tensile_strength", "elongation"):
        assert entry["landmarks"][target]["full"]["n"] > 0
        assert np.isfinite(entry["landmarks"][target]["full"]["mae"])
    assert result["protocol"] == "gkf_grade"
    assert result["protocol_is_leaky"] is False


def test_common_core_is_the_same_rows_for_every_landmark(synthetic_grouped):
    """The like-for-like row set must not drift between targets."""
    df, comp_long = synthetic_grouped
    result = evaluate.evaluate(df, comp_long, "gkf_grade", ("ridge",), "ratio",
                               "comp", tune=False, seed=1)
    lm = result["models"]["ridge"]["landmarks"]
    counts = {t: lm[t]["common_core"]["n"] for t in lm}
    assert len(set(counts.values())) == 1, counts
    assert counts["elongation"] <= result["n_common_core_rows"]


def test_component_with_too_few_rows_is_skipped_with_a_reason(synthetic_grouped):
    df, comp_long = synthetic_grouped
    tiny = df.head(30).reset_index(drop=True)
    result = evaluate.evaluate(tiny, comp_long, "gkf_grade", ("ridge",), "ratio",
                               "comp", tune=False, seed=1)
    skipped = [c["skipped"] for c in result["models"]["ridge"]["components"].values()]
    assert any(s for s in skipped), "an unusable component must say why it was skipped"


def test_both_parameterisations_run_and_agree_on_the_invariant(synthetic_grouped):
    df, comp_long = synthetic_grouped
    for name in ("ratio", "gap"):
        result = evaluate.evaluate(df, comp_long, "gkf_grade", ("ridge",), name,
                                   "comp", tune=False, seed=1)
        assert result["models"]["ridge"]["invariant_uts_gt_ys"]["holds"] is True
        assert result["parameterisation"] == name


def test_ridge_sign_check_is_parameterisation_independent(synthetic_grouped):
    """The wiring test must cover all three landmarks whichever encoding is active."""
    df, comp_long = synthetic_grouped
    for name in ("ratio", "gap"):
        check = evaluate.ridge_sign_check(df, comp_long, name, "comp")
        covered = {e["component"] for e in check["expectations"]}
        assert covered == {"yield_strength", "tensile_strength", "elongation"}
        assert check["all_passed"], check["expectations"]


def test_headline_protocol_is_not_the_leaky_one():
    assert splits.get(splits.HEADLINE_PROTOCOL).leaky is False
