"""Pure unit conversions and standard tagging.

SteelBench already reports MPa and %, so v1 does not call the converters on the
training path. They exist because the Phase 2 external test set and the Phase 6
non-ferrous sources will not, and a conversion bug found at that point would
silently poison a cross-source generalisation number -- the one claim in the
project that depends on two datasets agreeing about what a megapascal is.

Every function here is pure and total: no NaN handling, no DataFrames.
"""

from __future__ import annotations

KSI_TO_MPA = 6.894757293168361
MPA_TO_KSI = 1.0 / KSI_TO_MPA

# Standard atomic weights (IUPAC 2021), g/mol, for the elements we carry plus Fe.
ATOMIC_WEIGHT = {
    "Fe": 55.845, "C": 12.011, "Mn": 54.938, "Si": 28.085, "Cr": 51.996,
    "Ni": 58.693, "Mo": 95.95, "V": 50.942, "Cu": 63.546, "Al": 26.982,
}


def ksi_to_mpa(value: float) -> float:
    """Convert stress in ksi to MPa."""
    return value * KSI_TO_MPA


def mpa_to_ksi(value: float) -> float:
    """Convert stress in MPa to ksi."""
    return value * MPA_TO_KSI


def atomic_to_weight_percent(at_pct: dict[str, float]) -> dict[str, float]:
    """Convert an at% composition to wt%.

    The balance is taken to be Fe: whatever at% is unaccounted for is treated as
    iron before conversion, because alloy tables routinely omit the balance.
    Returns wt% for the *named* elements only (Fe is dropped again on the way
    out, matching how compositions are stored).
    """
    named = sum(at_pct.values())
    if named > 100.0 + 1e-9:
        raise ValueError(f"atomic percentages sum to {named:.4f} > 100")
    full = dict(at_pct)
    full["Fe"] = full.get("Fe", 0.0) + (100.0 - named)

    mass = {el: frac * ATOMIC_WEIGHT[el] for el, frac in full.items()}
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("composition has zero total mass")
    return {el: 100.0 * m / total for el, m in mass.items() if el != "Fe"}


# Elongation is only comparable within a gauge-length standard: A5 (gauge = 5x
# diameter) reads higher than A50mm on the same material, and the two are not
# convertible without the specimen geometry. We therefore TAG rather than
# convert, and the tag rides along so a later phase can stratify on it.
ELONGATION_STANDARDS = frozenset({"A5", "A4", "A50mm", "A80mm", "unknown"})


def tag_elongation_standard(raw: str | None) -> str:
    """Normalise a free-text elongation standard to a known tag.

    Anything unrecognised becomes "unknown" -- never guessed. An unknown tag is
    honest; a wrong conversion is not recoverable downstream.
    """
    if raw is None:
        return "unknown"
    key = str(raw).strip().replace(" ", "").replace("_", "").upper()
    aliases = {
        "A5": "A5", "A5D": "A5", "5D": "A5", "GL5D": "A5",
        "A4": "A4", "A4D": "A4", "4D": "A4",
        "A50": "A50mm", "A50MM": "A50mm", "50MM": "A50mm",
        "A80": "A80mm", "A80MM": "A80mm", "80MM": "A80mm",
    }
    return aliases.get(key, "unknown")
