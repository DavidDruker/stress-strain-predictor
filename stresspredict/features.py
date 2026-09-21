"""Feature construction: raw chemistry plus a small set of metallurgical terms.

Deliberately NOT matminer/Magpie. Magpie descriptors are composition-weighted
statistics over the periodic table, designed for chemically diverse search
spaces. Steels are 95-99% Fe with a handful of elements in the 0-5 wt% band, so
those statistics are dominated by iron, nearly constant across the dataset, and
actively blur the signal that matters -- the difference between 0.40 and 0.45
wt% C, which is simply the raw number. ~130 descriptors on ~1.3k rows would also
convert this into a feature-selection exercise instead of an evaluation-rigour
one, for ~100 MB of transitive dependencies.

Whether the hand-built terms earn their place is then TESTED per model
(`evaluate --features comp_raw` vs `comp`), not assumed. The expectation is a
clear gain for ridge and little or none for the gradient booster, which can
learn `C + Mn/6` by itself. If that is the result, it gets reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import schema

FEATURE_SETS = ("comp_raw", "comp", "comp_ht")


def dense_composition(samples: pd.DataFrame, composition_long: pd.DataFrame) -> pd.DataFrame:
    """Project long-format composition onto a fixed element matrix.

    Elements absent from the long table become NaN -- "not reported" -- which is
    a different thing from 0.0 and is kept distinct until the featuriser makes an
    explicit decision about it.
    """
    wide = composition_long.pivot_table(
        index="sample_id", columns="element", values="wt_pct", aggfunc="first"
    )
    wide = wide.reindex(columns=list(schema.ELEMENTS))
    return wide.reindex(samples["sample_id"].values).reset_index(drop=True)


def design_frame(samples: pd.DataFrame, composition_long: pd.DataFrame) -> pd.DataFrame:
    """The full input frame: elements plus heat-treatment columns."""
    X = dense_composition(samples, composition_long)
    for col in schema.PROCESS_NUMERIC:
        X[col] = samples[col].to_numpy(dtype=float)
    return X


class CompositionFeaturizer(BaseEstimator, TransformerMixin):
    """Impute, then derive. One class, so every model sees identical data.

    Imputation is split by what missingness MEANS, which a single global strategy
    would get wrong in one direction or the other:

    * elements  -> 0.0, because a blank means the element was not deliberately
      added, and zero is the physically correct reading of that;
    * temperatures -> the training median, because 0 C is not a plausible
      austenitising temperature and would wreck a linear model's scale.

    Both carry an explicit missing-indicator column, so a model can still learn
    "this was never reported" as its own signal. HistGradientBoosting could take
    NaN natively, but letting it see different data from ridge would invalidate
    the ladder comparison, which is the point of having a ladder.
    """

    def __init__(self, feature_set: str = "comp"):
        self.feature_set = feature_set

    # -- derived metallurgical terms ------------------------------------
    @staticmethod
    def _derive(el: pd.DataFrame) -> pd.DataFrame:
        C, Mn, Si = el["C"], el["Mn"], el["Si"]
        Cr, Ni, Mo = el["Cr"], el["Ni"], el["Mo"]
        V, Cu = el["V"], el["Cu"]  # Al enters only through total_alloy

        d = pd.DataFrame(index=el.index)
        # IIW carbon equivalent -- hardenability / weldability, the single most
        # widely used scalar summary of a steel's chemistry.
        d["ce_iiw"] = C + Mn / 6 + (Cr + Mo + V) / 5 + (Ni + Cu) / 15
        # Ito-Bessyo cold-cracking parameter; weights low-carbon steels differently.
        d["pcm"] = (C + Si / 30 + Mn / 20 + Cu / 20 + Cr / 20
                    + Ni / 60 + Mo / 15 + V / 10)
        d["total_alloy"] = el[list(schema.ELEMENTS)].sum(axis=1)
        d["fe_balance"] = 100.0 - d["total_alloy"]
        # Order-of-magnitude solid-solution strengthening proxy: approximate
        # ferrite strengthening coefficients in MPa per wt%. The absolute scale
        # is not meaningful; the ranking across alloys is.
        d["ss_proxy"] = 83 * Si + 32 * Mn + 33 * Ni + 11 * Cr + 11 * Mo + 39 * Cu
        d["carbide_former"] = Cr + Mo + V
        # Schaeffler equivalents. This dataset is 40% stainless across austenitic,
        # ferritic, duplex, martensitic and PH families, and these two terms are
        # what separate those structures -- first-order for elongation.
        d["cr_eq_schaeffler"] = Cr + Mo + 1.5 * Si
        d["ni_eq_schaeffler"] = Ni + 30 * C + 0.5 * Mn
        return d

    DERIVED_NAMES = ("ce_iiw", "pcm", "total_alloy", "fe_balance",
                     "ss_proxy", "carbide_former", "cr_eq_schaeffler", "ni_eq_schaeffler")

    # -- sklearn API ------------------------------------------------------
    def _columns(self) -> tuple[list[str], list[str]]:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        elements = list(schema.ELEMENTS)
        process = list(schema.PROCESS_NUMERIC) if self.feature_set == "comp_ht" else []
        return elements, process

    def fit(self, X: pd.DataFrame, y=None):
        elements, process = self._columns()
        self.element_cols_ = elements
        self.process_cols_ = process
        # Medians learned on training folds only; CV clones this object per fold,
        # so no test-fold statistic ever reaches a training transform.
        self.process_medians_ = {
            c: float(np.nanmedian(X[c].to_numpy(dtype=float))) if X[c].notna().any() else 0.0
            for c in process
        }
        self.indicator_cols_ = [c for c in elements + process if X[c].isna().any()]
        self.feature_names_ = self._build_names()
        self.n_features_in_ = X.shape[1]
        return self

    def _build_names(self) -> list[str]:
        names = list(self.element_cols_)
        if self.feature_set != "comp_raw":
            names += list(self.DERIVED_NAMES)
        names += list(self.process_cols_)
        names += [f"missing__{c}" for c in self.indicator_cols_]
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        el = X[self.element_cols_].astype(float)
        indicators = X[self.indicator_cols_].isna().astype(float) if self.indicator_cols_ else None

        el_filled = el.fillna(0.0)
        blocks = [el_filled]
        if self.feature_set != "comp_raw":
            blocks.append(self._derive(el_filled))
        if self.process_cols_:
            proc = X[self.process_cols_].astype(float)
            blocks.append(proc.fillna(value=self.process_medians_))
        if indicators is not None:
            indicators.columns = [f"missing__{c}" for c in indicators.columns]
            blocks.append(indicators)

        out = pd.concat(blocks, axis=1)
        return out[self.feature_names_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
