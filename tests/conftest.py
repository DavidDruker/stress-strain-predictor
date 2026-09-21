"""Shared fixtures.

Unit tests run on synthetic data so the suite passes in CI, where the datasets
are deliberately not committed (see data/README.md). Tests that genuinely need
the real thing are marked `integration` and skip with a clear reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stresspredict import schema

PROCESSED = Path("data/processed")


def make_synthetic(n: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small dataset with the same shape and quirks as the real one.

    Deliberately includes the awkward parts: missing elements, missing targets,
    repeated compositions at different targets (the processing-noise signature),
    uneven family sizes and more than one source.
    """
    rng = np.random.default_rng(seed)

    C = rng.uniform(0.05, 0.9, n)
    Mn = rng.uniform(0.2, 2.0, n)
    Si = rng.uniform(0.1, 1.0, n)
    Cr = rng.uniform(0.0, 18.0, n)
    Ni = rng.uniform(0.0, 12.0, n)
    Mo = rng.uniform(0.0, 2.0, n)
    V = rng.uniform(0.0, 0.3, n)
    Cu = rng.uniform(0.0, 0.5, n)
    Al = rng.uniform(0.0, 0.1, n)

    # Monotone in carbon so the ridge sign check has something true to find.
    uts = 400 + 700 * C + 40 * Mn + 12 * Cr + 30 * Mo + rng.normal(0, 40, n)
    uts = np.clip(uts, 250, 2000)
    ratio = np.clip(0.55 + 0.15 * C + rng.normal(0, 0.05, n), 0.15, 0.95)
    ys = uts * ratio
    el = np.clip(38 - 25 * C - 0.2 * Cr + rng.normal(0, 3, n), 1.0, 60.0)

    grades = [f"G{i % 90:03d}" for i in range(n)]
    families = np.array(
        ["carbon"] * (n // 3) + ["low_alloy"] * (n // 3) + ["stainless"] * (n - 2 * (n // 3))
    )
    rng.shuffle(families)
    # A family too small to hold out, so MinSizeLeaveOneGroupOut has work to do.
    families[:3] = "exotic"
    sources = np.where(rng.random(n) < 0.5, "tierA", "tierB")

    samples = pd.DataFrame({
        "sample_id": [f"syn:{i}" for i in range(n)],
        "source_id": "synthetic",
        "grade_id": grades,
        "source_label": sources,
        "provenance": sources,
        "steel_family": families,
        "measurement_kind": np.where(sources == "tierA", "measured", "spec_minimum"),
        "austenitize_T": np.where(rng.random(n) < 0.5, rng.uniform(820, 1050, n), np.nan),
        "temper_T": np.where(rng.random(n) < 0.35, rng.uniform(450, 700, n), np.nan),
        "quench_medium": "oil",
        "yield_strength": np.where(rng.random(n) < 0.75, ys, np.nan),
        "tensile_strength": uts,
        "elongation": np.where(rng.random(n) < 0.5, el, np.nan),
        "elongation_standard": "unknown",
    })

    dense = pd.DataFrame({"C": C, "Mn": Mn, "Si": Si, "Cr": Cr, "Ni": Ni,
                          "Mo": Mo, "V": V, "Cu": Cu, "Al": Al})
    dense.insert(0, "sample_id", samples["sample_id"].values)
    # Some elements genuinely not reported -- "missing" must stay distinct from 0.
    for el_name in ("Cu", "Al", "V"):
        drop = rng.random(n) < 0.4
        dense.loc[drop, el_name] = np.nan

    comp_long = (dense.melt(id_vars=["sample_id"], var_name="element", value_name="wt_pct")
                 .dropna(subset=["wt_pct"]).reset_index(drop=True))
    return samples, comp_long


@pytest.fixture(scope="session")
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_synthetic()


@pytest.fixture(scope="session")
def synthetic_grouped(synthetic):
    from stresspredict import grouping
    samples, comp = synthetic
    return grouping.attach_groups(samples, comp), comp


@pytest.fixture(scope="session")
def real_data():
    """The actual processed dataset, or a skip with instructions."""
    if not (PROCESSED / "samples.csv").exists():
        pytest.skip(
            "processed data absent (it is gitignored); run "
            "`python -m stresspredict.ingest --source steelbench` to enable integration tests"
        )
    from stresspredict import grouping
    samples = pd.read_csv(PROCESSED / "samples.csv")
    comp = pd.read_csv(PROCESSED / "composition_long.csv")
    return grouping.attach_groups(samples, comp), comp


@pytest.fixture(scope="session")
def element_names() -> tuple[str, ...]:
    return schema.ELEMENTS
