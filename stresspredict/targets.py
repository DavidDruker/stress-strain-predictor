"""Target parameterisation -- where the physical invariant is enforced.

UTS > YS is not a soft preference to be learned; it is a fact about tensile
tests. Three independent regressors on the raw targets will violate it on some
inputs, and `MultiOutputRegressor` gives no guarantee at all. Instead the
landmarks are predicted on a transformed parameterisation from which the
invariant follows algebraically, so no prediction can break it.

Two parameterisations are implemented, and which one ships is decided by
measurement rather than assertion (`evaluate --parameterisation`):

RATIO (default)
    UTS = exp(z_uts)                    z_uts   = log(UTS)
    YS  = UTS * sigmoid(z_ratio)        z_ratio = logit(YS / UTS)
    EL  = exp(z_el)                     z_el    = log(EL)
    sigmoid < 1 strictly, so YS < UTS always.

GAP
    YS  = exp(z_ys)                     z_ys  = log(YS)
    UTS = YS + exp(z_gap)               z_gap = log(UTS - YS)
    EL  = exp(z_el)                     z_el  = log(EL)
    exp > 0 strictly, so UTS > YS always.

RATIO is the default for a reason specific to this dataset: UTS is reported for
all 1,359 usable rows but YS for only 984, because all 360 NIMS rows carry UTS
alone. GAP anchors on YS and can therefore train its UTS pathway on 984 rows;
RATIO anchors on UTS and trains it on all 1,359 -- 38% more data for the
best-populated target -- while the yield ratio it predicts is itself a standard
metallurgical quantity rather than an artefact of the encoding.

A consequence worth stating plainly in the docs: inverse-transforming a
log-target prediction recovers the conditional MEDIAN, not the mean. MAE is
therefore the coherent primary metric here, not a stylistic preference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit


@dataclass(frozen=True)
class Component:
    """One regression sub-problem in a parameterisation."""

    name: str
    unit: str
    requires: tuple[str, ...]          # target columns needed to build it
    quantity: Callable[[pd.DataFrame], pd.Series]
    link: Callable[[np.ndarray], np.ndarray]      # quantity -> z
    inverse_link: Callable[[np.ndarray], np.ndarray]  # z -> quantity

    def trainable_mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        for col in self.requires:
            m &= df[col].notna()
        return m


def _strictly_below(value: np.ndarray, ceiling: np.ndarray) -> np.ndarray:
    """Force value < ceiling in float64, not merely in exact arithmetic.

    sigmoid(z) rounds to exactly 1.0 for z beyond ~37, which would make YS == UTS
    and break an invariant the docs promise holds always. This is a guard on the
    inverse transform, not a clip of a prediction: it moves a value by one ULP.
    """
    return np.minimum(value, np.nextafter(ceiling, -np.inf))


def _strictly_above(value: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Force value > floor in float64 (exp(z) can underflow to a no-op addition)."""
    return np.maximum(value, np.nextafter(floor, np.inf))


class Parameterisation:
    """A set of components plus the rule for assembling landmarks from them."""

    def __init__(self, name: str, components: tuple[Component, ...],
                 assemble: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
                 landmark_requires: dict[str, tuple[str, ...]]):
        self.name = name
        self.components = components
        self._assemble = assemble
        # Which components each landmark actually needs. Evaluation scores every
        # landmark on the rows it can genuinely be produced for, rather than on
        # the intersection of all three components -- under RATIO that is the
        # difference between scoring UTS on 1,359 rows and on 659.
        self.landmark_requires = landmark_requires

    def __getitem__(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)

    def decode(self, quantities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Assemble predicted quantities into landmarks, invariant guaranteed."""
        return self._assemble({k: np.asarray(v, dtype=float) for k, v in quantities.items()})


def _assemble_ratio(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    uts = q["tensile_strength"]
    ys = _strictly_below(uts * q["yield_ratio"], uts)
    return {"yield_strength": ys, "tensile_strength": uts, "elongation": q["elongation"]}


def _assemble_gap(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ys = q["yield_strength"]
    uts = _strictly_above(ys + q["strength_gap"], ys)
    return {"yield_strength": ys, "tensile_strength": uts, "elongation": q["elongation"]}


RATIO = Parameterisation(
    "ratio",
    (
        Component("tensile_strength", "MPa", ("tensile_strength",),
                  lambda df: df["tensile_strength"], np.log, np.exp),
        Component("yield_ratio", "-", ("yield_strength", "tensile_strength"),
                  lambda df: df["yield_strength"] / df["tensile_strength"], logit, expit),
        Component("elongation", "%", ("elongation",),
                  lambda df: df["elongation"], np.log, np.exp),
    ),
    _assemble_ratio,
    landmark_requires={
        "tensile_strength": ("tensile_strength",),
        "yield_strength": ("tensile_strength", "yield_ratio"),
        "elongation": ("elongation",),
    },
)

GAP = Parameterisation(
    "gap",
    (
        Component("yield_strength", "MPa", ("yield_strength",),
                  lambda df: df["yield_strength"], np.log, np.exp),
        Component("strength_gap", "MPa", ("yield_strength", "tensile_strength"),
                  lambda df: df["tensile_strength"] - df["yield_strength"], np.log, np.exp),
        Component("elongation", "%", ("elongation",),
                  lambda df: df["elongation"], np.log, np.exp),
    ),
    _assemble_gap,
    landmark_requires={
        "yield_strength": ("yield_strength",),
        "tensile_strength": ("yield_strength", "strength_gap"),
        "elongation": ("elongation",),
    },
)

PARAMETERISATIONS = {"ratio": RATIO, "gap": GAP}
DEFAULT_PARAMETERISATION = "ratio"


def get(name: str = DEFAULT_PARAMETERISATION) -> Parameterisation:
    if name not in PARAMETERISATIONS:
        raise KeyError(f"unknown parameterisation {name!r}; known: {sorted(PARAMETERISATIONS)}")
    return PARAMETERISATIONS[name]


def check_invariant(landmarks: dict[str, np.ndarray]) -> tuple[bool, int]:
    """Return (holds_everywhere, n_violations) for UTS > YS."""
    ys, uts = np.asarray(landmarks["yield_strength"]), np.asarray(landmarks["tensile_strength"])
    both = np.isfinite(ys) & np.isfinite(uts)
    violations = int(np.sum(uts[both] <= ys[both]))
    return violations == 0, violations


def derived_outputs(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Free engineering quantities -- no extra models, no extra assumptions."""
    ys = np.asarray(landmarks["yield_strength"], dtype=float)
    uts = np.asarray(landmarks["tensile_strength"], dtype=float)
    el = np.asarray(landmarks["elongation"], dtype=float)
    return {
        "yield_ratio": ys / uts,              # YS/UTS, the standard formability index
        "strength_gap": uts - ys,             # MPa of work hardening available
        "strength_ductility_product": uts * el,  # MPa.% -- the AHSS "banana curve" axis
    }
