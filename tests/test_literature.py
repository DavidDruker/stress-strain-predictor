"""Row selection for the literature-derived source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stresspredict import literature


@pytest.mark.parametrize("raw,expected", [
    ("0.40", 0.40), (0.4, 0.4), ("4.0-8.0", 6.0), ("0.10 – 0.20", 0.15),
])
def test_ranges_become_midpoints(raw, expected):
    assert literature.parse_wt_pct(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["<0.01", "<= 0.02", "Nil", "-", "Bal.", None])
def test_limits_and_balance_are_not_reported(raw):
    assert np.isnan(literature.parse_wt_pct(raw))


def _records(**overrides) -> pd.DataFrame:
    n = len(next(iter(overrides.values()))) if overrides else 1
    base = {el: [np.nan] * n for el in literature.ELEMENT_COLUMNS}
    base.update({
        "C": [0.2] * n, "Mn": [1.0] * n,
        "composition unit": ["wt.%"] * n,
        "test route/condition": ["tensile test at room temperature"] * n,
        "synthesis and processing routes": [""] * n,
    })
    base.update(overrides)
    return pd.DataFrame(base, index=[f"r{i}" for i in range(n)])


def test_only_plain_room_temperature_tensile_tests_survive():
    df = _records(**{"test route/condition": [
        "tensile test at room temperature",
        "tensile test at 600 °C",
        "room temperature tensile test on hydrogen-charged specimens",
        "tensile test at room temperature and at 77 K",
        "uniaxial tensile test",                     # temperature never stated
    ]})
    kept, _ = literature.select(df)
    assert list(kept.index) == ["r0"]


def test_atomic_percent_and_carbon_free_records_are_dropped():
    df = _records(**{"composition unit": ["wt.%", "at.%", "wt.%"], "C": [0.2, 0.2, np.nan]})
    kept, steps = literature.select(df)
    assert list(kept.index) == ["r0"]
    assert steps[0]["rows_after"] == 3
    assert all("rows_removed" in s for s in steps[1:])


def test_rows_near_a_heldout_steel_are_removed():
    df = _records(C=[0.02, 0.02], Ni=[18.0, 3.0], Co=[9.0, 0.0])
    heldout = pd.DataFrame({"C": [0.02], "Mn": [1.1], "Ni": [18.2], "Co": [8.9]})
    kept, steps = literature.select(df, heldout)
    assert list(kept.index) == ["r1"]
    assert steps[-1]["rows_removed"] == 1
