"""Report rendering and the per-protocol merge."""

from __future__ import annotations

import pytest

from stresspredict import report


def _bundle(protocol: str, feature_set: str = "comp", n_rows: int = 1359) -> dict:
    return {
        "stem": "eval_ratio_comp_all",
        "rows_filter": "all",
        "restrict_ht_rows": False,
        "n_rows_evaluated": n_rows,
        "ridge_sign_check": {
            "coefficients": {},
            "expectations": [{"component": "tensile_strength", "expected": "carbon > 0",
                              "observed": 0.09, "passed": True}],
            "all_passed": True,
        },
        "protocols": {
            protocol: {
                "protocol": protocol,
                "protocol_question": "q",
                "protocol_is_leaky": protocol == "random_kfold",
                "group_column": None if protocol == "random_kfold" else "grade_group",
                "parameterisation": "ratio",
                "feature_set": feature_set,
                "tuned_inside_outer_fold": True,
                "seed": 1,
                "n_rows": n_rows,
                "n_grades": 562,
                "n_common_core_rows": 661,
                "models": {
                    "hist_gbm": {
                        "role": "expected winner",
                        "components": {
                            "tensile_strength": {
                                "summary_mae": {"std": 5.0, "n_folds": 5},
                                "per_fold": [], "fold_labels": [],
                            },
                        },
                        "landmarks": {
                            "yield_strength": {"unit": "MPa", "full": {"n": 984, "mae": 130.0},
                                               "common_core": {"n": 661, "mae": 120.0}},
                            "tensile_strength": {"unit": "MPa", "full": {"n": 1359, "mae": 110.0},
                                                 "common_core": {"n": 661, "mae": 100.0}},
                            "elongation": {"unit": "%", "full": {"n": 661, "mae": 5.0},
                                           "common_core": {"n": 661, "mae": 5.0}},
                        },
                        "invariant_uts_gt_ys": {"holds": True, "violations": 0, "n_checked": 984},
                    }
                },
            }
        },
    }


def _manifest() -> dict:
    floor = {"n_groups": 92, "n_rows": 232, "mae_floor_insample": 107.2,
             "mae_floor_loo": 191.5, "median_group_range": 100.0, "max_group_range": 400.0}
    return {
        "source_id": "steelbench_v1_open",
        "raw_sha256": "a" * 64,
        "noise_floor": {
            "headline_stratum": "measured_only",
            "targets": {
                t: {"measured_only": dict(floor),
                    "spec_minimum_only": dict(floor, mae_floor_insample=40.0),
                    "all_rows": dict(floor, mae_floor_insample=61.2)}
                for t in ("yield_strength", "tensile_strength", "elongation")
            },
        },
    }


def test_merge_combines_protocols():
    merged = report.merge([_bundle("random_kfold"), _bundle("gkf_grade")])
    assert set(merged["protocols"]) == {"random_kfold", "gkf_grade"}
    assert merged["n_rows_evaluated"] == 1359


def test_merge_refuses_mismatched_runs():
    """Silently merging halves from different runs would be worse than failing."""
    with pytest.raises(ValueError, match="refusing to merge"):
        report.merge([_bundle("random_kfold"), _bundle("gkf_grade", n_rows=900)])
    with pytest.raises(ValueError, match="refusing to merge"):
        report.merge([_bundle("random_kfold"), _bundle("gkf_grade", feature_set="comp_raw")])


def test_merge_rejects_nothing():
    with pytest.raises(ValueError):
        report.merge([])


def test_render_produces_the_expected_sections():
    merged = report.merge([_bundle("random_kfold"), _bundle("gkf_grade")])
    text = report.render(merged, _manifest())

    assert "## The physical floor" in text
    assert "## MAE by protocol, model and target" in text
    assert "## What grade leakage is worth" in text
    assert "## Correctness checks" in text
    assert "LEAKY" in text, "the leaky protocol must be labelled in the output"
    assert "107.2" in text and "191.5" in text, "the measured floor must appear"
    assert "61.2" in text, "the deflated pooled floor must be shown for contrast"


def test_render_shows_the_leak_ratio():
    merged = report.merge([_bundle("random_kfold"), _bundle("gkf_grade")])
    text = report.render(merged, _manifest())
    assert "1.00x" in text  # identical fixtures -> ratio of exactly 1


def test_render_reports_invariant_violations():
    merged = report.merge([_bundle("gkf_grade")])
    text = report.render(merged, _manifest())
    assert "Physical invariant" in text
    assert "**0**" in text
