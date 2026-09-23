"""The Mendeley loader's two deliberate mappings: zeros and ASTM spec rows."""

from __future__ import annotations

import pandas as pd
import pytest

from stresspredict import ingest, schema


@pytest.fixture()
def tiny_mendeley(tmp_path):
    pytest.importorskip("openpyxl")
    rows = {
        "Entry": [1, 2],
        "Name": ["AISI 4130H Steel", "ASTM A36 Steel"],
        "Processing condition": ["Water Quenched 855C (1570F), 540C (1000F) Temper", "Grade B"],
        "Cluster Number (0 to 11)": [10, 11],
        **{el: [0.0] * 2 for el in schema.ELEMENTS},
        "(Ultimate) Tensile strength (MPa)": [1040, 400],
        "Yield strength (MPa)": [979, 250],
        "Ductility (%)": [18, 23],
    }
    rows["C"], rows["Mn"], rows["Cr"] = [0.30, 0.26], [0.50, 0.0], [0.95, 0.0]
    path = tmp_path / "m.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return ingest.load_mendeley(path)


def test_zero_composition_means_not_specified(tiny_mendeley):
    comp = tiny_mendeley.composition
    second = comp[comp["sample_id"] == "mendeley_jmwb9ddd43:2"]
    assert set(second["element"]) == {"C"}
    assert (comp["wt_pct"] > 0).all()


def test_astm_rows_are_spec_minima_and_grades_are_normalised(tiny_mendeley):
    s = tiny_mendeley.samples
    assert list(s["measurement_kind"]) == ["measured", "spec_minimum"]
    assert s["grade_id"].iloc[0] == "4130"
    assert s["austenitize_T"].iloc[0] == 855.0
    assert s["temper_T"].iloc[0] == 540.0
