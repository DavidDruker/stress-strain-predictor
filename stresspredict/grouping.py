"""Group keys for leakage-aware splitting.

Under a composition-only feature set, two rows of the same grade heat-treated
differently have *literally identical feature vectors and different targets*.
Any split that lets one of them train while the other tests is measuring grade
lookup, not alloy-design generalisation. These functions produce the keys that
stop that happening.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from . import schema


def composition_hash(composition_long: pd.DataFrame) -> pd.Series:
    """Hash each sample's reported chemistry into a stable group key.

    Rounded to the dataset's reporting precision before hashing, so float noise
    does not split a group. An element that was never reported is *absent* from
    the key rather than encoded as zero: "0.00 wt% V" and "V not measured" are
    different states of knowledge and must not collide.

    Returns a Series indexed by sample_id.
    """
    df = composition_long.copy()
    df["wt_pct"] = df["wt_pct"].round(schema.COMPOSITION_ROUND_DP)
    df = df.sort_values(["sample_id", "element"])

    def _digest(group: pd.DataFrame) -> str:
        payload = ";".join(f"{e}={v:.{schema.COMPOSITION_ROUND_DP}f}"
                           for e, v in zip(group["element"], group["wt_pct"], strict=True))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    return df.groupby("sample_id", sort=False)[["element", "wt_pct"]].apply(_digest).rename("comp_hash")


def attach_groups(samples: pd.DataFrame, composition_long: pd.DataFrame) -> pd.DataFrame:
    """Return `samples` with every grouping key attached."""
    out = samples.copy()
    ch = composition_hash(composition_long)
    out["comp_hash"] = out["sample_id"].map(ch)

    # A sample with no reported chemistry at all cannot be composition-grouped;
    # fall back to its own id so it forms a singleton rather than colliding with
    # every other chemistry-less sample.
    out["comp_hash"] = out["comp_hash"].fillna("nocomp:" + out["sample_id"])

    # Grade is the headline grouping. Where a source ships no grade label, the
    # composition hash is the honest stand-in: same chemistry, same group.
    grade = out["grade_id"].astype("string")
    missing_grade = grade.isna() | grade.str.strip().isin(["", "nan", "None"])
    out["grade_group"] = grade.where(~missing_grade, "comp:" + out["comp_hash"])

    out["family_group"] = out["steel_family"].astype("string").fillna("unknown")
    out["source_group"] = out["provenance"].astype("string").fillna("unknown")
    return out


GROUP_COLUMNS = ("grade_group", "family_group", "source_group", "comp_hash")
