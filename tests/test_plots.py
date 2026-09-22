"""Figure generation, and the guards against figures that lie."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stresspredict import plots


def _oof(protocol: str = "gkf_grade", model: str = "hist_gbm", n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for landmark, lo, hi in (("yield_strength", 250, 1200),
                             ("tensile_strength", 400, 1600),
                             ("elongation", 3, 45)):
        truth = rng.uniform(lo, hi, n)
        frames.append(pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "model": model,
            "protocol": protocol,
            "steel_family": rng.choice(["carbon", "stainless"], n),
            "measurement_kind": rng.choice(["measured", "spec_minimum"], n),
            "landmark": landmark,
            "y_true": truth,
            "y_pred": truth * rng.uniform(0.85, 1.15, n),
        }))
    return pd.concat(frames, ignore_index=True)


def _bundle(protocols: tuple[str, ...]) -> dict:
    def one(p: str) -> dict:
        return {
            "protocol": p, "protocol_is_leaky": p == "random_kfold",
            "models": {"hist_gbm": {
                "landmarks": {t: {"unit": "MPa", "full": {"n": 60, "mae": 90.0}}
                              for t in ("yield_strength", "tensile_strength", "elongation")},
                "components": {"tensile_strength": {
                    "per_fold": [{"mae": 80.0, "n": 10}, {"mae": 120.0, "n": 12}],
                    "fold_labels": ["carbon", "stainless"],
                }},
            }},
        }
    return {"protocols": {p: one(p) for p in protocols}}


def _manifest() -> dict:
    fl = {"n_groups": 92, "n_rows": 232, "mae_floor_insample": 107.2, "mae_floor_loo": 191.5}
    return {"noise_floor": {"targets": {t: {"measured_only": dict(fl)}
                                        for t in ("yield_strength", "tensile_strength",
                                                  "elongation")}}}


def test_parity_writes_a_figure(tmp_path: Path):
    p = plots.parity(_oof(), "gkf_grade", "hist_gbm", tmp_path)
    assert p.exists() and p.stat().st_size > 0


def test_parity_refuses_to_write_an_empty_figure(tmp_path: Path):
    """A blank chart with real axes and a real filename looks like a result."""
    p = plots.parity(_oof(protocol="gkf_grade"), "loso_source", "hist_gbm", tmp_path)
    assert str(p) == "."
    assert list(tmp_path.iterdir()) == []


def test_leak_plot_needs_both_protocols(tmp_path: Path):
    """The plot's entire content is the gap; one bar is not a comparison."""
    assert str(plots.leak_plot(_bundle(("gkf_grade",)), "yield_strength", tmp_path)) == "."
    assert list(tmp_path.iterdir()) == []

    p = plots.leak_plot(_bundle(("gkf_grade", "random_kfold")), "yield_strength", tmp_path)
    assert p.exists()


def test_floor_plot_skips_without_the_headline_protocol(tmp_path: Path):
    assert str(plots.floor_plot(_bundle(("loso_source",)), _manifest(),
                                "yield_strength", tmp_path)) == "."
    p = plots.floor_plot(_bundle(("gkf_grade",)), _manifest(), "yield_strength", tmp_path)
    assert p.exists()


def test_lofo_plot_skips_when_absent(tmp_path: Path):
    assert str(plots.lofo_plot(_bundle(("gkf_grade",)), "hist_gbm",
                               "tensile_strength", tmp_path)) == "."
    p = plots.lofo_plot(_bundle(("lofo_family",)), "hist_gbm", "tensile_strength", tmp_path)
    assert p.exists()


def test_banana_plot_needs_both_uts_and_elongation(tmp_path: Path):
    p = plots.banana_plot(_oof(), "gkf_grade", "hist_gbm", tmp_path)
    assert p.exists()
    assert str(plots.banana_plot(_oof(), "nonexistent", "hist_gbm", tmp_path)) == "."
