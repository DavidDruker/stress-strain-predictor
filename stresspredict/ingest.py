"""Source loaders: raw publisher CSV -> canonical schema.

One loader per source. A loader's only job is to rename, convert and label; it
never filters, imputes or repairs. Everything that removes or nulls a value
lives in `clean`, so the manifest can account for it.

Composition is stored LONG (sample_id, element, wt_pct), not as nine fixed
columns, and projected to a dense matrix inside `features`. Nine hard-coded
columns would cost nothing today and force a rewrite the moment a titanium
source arrives with Fe, Al, V and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, clean, noise_floor, schema

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class Ingested:
    """A source loaded into canonical form, before cleaning."""

    samples: pd.DataFrame       # one row per sample, ID_COLS + process + targets
    composition: pd.DataFrame   # long: sample_id, element, wt_pct
    source_id: str
    raw_path: Path
    raw_sha256: str
    raw_rows: int
    parts: tuple = ()           # the per-source loads behind a merged source
    selection: tuple = ()       # source-specific row selection steps, counted


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_long(df: pd.DataFrame, elements: tuple[str, ...]) -> pd.DataFrame:
    """Melt dense element columns to long form, dropping absent measurements.

    Absence is represented by the row simply not existing. That is the whole
    reason for long format: it keeps "not reported" distinguishable from 0.0
    right up until `features` makes an explicit, documented decision about it.
    """
    present = [e for e in elements if e in df.columns]
    long = df.melt(
        id_vars=["sample_id"], value_vars=present,
        var_name="element", value_name="wt_pct",
    )
    return long.dropna(subset=["wt_pct"]).reset_index(drop=True)


def load_steelbench(path: Path | None = None) -> Ingested:
    """Load SteelBench v1.0 (open release).

    Deliberate mappings, all of which are checked by tests:

    * `data_tier == "emk_spec_verified"` -> measurement_kind "spec_minimum".
      55% of the open release is specification MINIMA, not measurements. They
      are near-deterministic given a grade, so pooling them into the
      within-composition spread halves it. Carrying the distinction as a
      first-class column is what lets `noise_floor` report an honest number.
    * `condition` is dropped: 98.9% missing, and the surviving 15 values are
      free text in mixed Russian/English. There is nothing to learn from it.
    * `split` is dropped: constant ("pretrain") in the open release, so the
      publisher's predefined split carries no information here.
    * `reduction_area`, `impact_J_avg`, `hardness` are carried through but are
      not v1 targets; they cost one column each and are the natural Phase 2+
      auxiliary targets.
    """
    path = Path(path) if path is not None else RAW_DIR / "steelbench_core_open.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n"
            f"  curl -L -o {path} "
            f"https://zenodo.org/api/records/18530558/files/steelbench_core_open.csv/content\n"
            f"See data/README.md for the DOI, licence and expected checksum."
        )

    raw = pd.read_csv(path)
    source_id = "steelbench_v1_open"

    out = pd.DataFrame()
    out["sample_id"] = source_id + ":" + raw["heat_id"].astype(str)
    out["source_id"] = source_id
    out["grade_id"] = raw["grade_id"].astype(str)
    out["source_label"] = raw["source"].astype(str)
    out["provenance"] = raw["data_tier"].astype(str)
    out["steel_family"] = raw["steel_family"].astype(str)
    out["measurement_kind"] = (
        raw["data_tier"].eq("emk_spec_verified").map({True: "spec_minimum", False: "measured"})
    )

    for col in schema.PROCESS_NUMERIC:
        out[col] = pd.to_numeric(raw[col], errors="coerce")
    out["quench_medium"] = raw["quench_medium"].astype("string")

    for col in schema.TARGETS:
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    # Elongation standard is not reported by SteelBench. Tag it honestly rather
    # than assuming A5 -- the tag is what a later cross-source comparison needs.
    out["elongation_standard"] = "unknown"

    for col in ("reduction_area", "impact_J_avg", "hardness"):
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    comp_dense = raw[list(schema.ELEMENTS)].apply(pd.to_numeric, errors="coerce")
    comp_dense.insert(0, "sample_id", out["sample_id"].values)

    return Ingested(
        samples=out,
        composition=_to_long(comp_dense, schema.ELEMENTS),
        source_id=source_id,
        raw_path=path,
        raw_sha256=sha256_of(path),
        raw_rows=len(raw),
    )


MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/jmwb9ddd43/files/"
                "cc944ab8-d3c8-448c-ac1d-172e8cc7e11d/file_downloaded")


def _grade_key(name: str) -> str:
    """'AISI 4130H Steel' -> '4130', matching SteelBench's Kaggle grade_id.

    The H (hardenability-band) and E (electric-furnace) variants share a
    chemistry with the base grade, so they are grouped with it: splitting them
    across folds would let the same steel sit on both sides of a split.
    """
    m = re.search(r"AISI\s+(?:Type\s+)?E?(\d{3,4}[A-Z]?)", name, re.I)
    if m:
        g = m.group(1).upper()
        return g[:-1] if g.endswith("H") and len(g) == 5 else g
    return re.sub(r"\s+steel$", "", name.strip(), flags=re.I)


def _first_celsius(text: pd.Series, pattern: str) -> pd.Series:
    return pd.to_numeric(text.str.extract(pattern, flags=re.I)[0], errors="coerce")


def load_mendeley(path: Path | None = None) -> Ingested:
    """Load 'A database of mechanical properties of steels' (Mendeley jmwb9ddd43).

    Deliberate mappings:

    * A composition of 0 means "not specified" in this file -- every row lists
      all 20 elements -- so zeros are dropped from the long table, the same state
      a blank cell has in SteelBench. Taking them literally would tell the model
      that thousands of carbon steels were certified free of Si and Cr.
    * `Name` starting with ASTM -> measurement_kind "spec_minimum". Those rows
      are specification grades/classes/thicknesses, i.e. minima, the same trap
      as SteelBench's EMK tier.
    * Temperatures are read from the free-text processing condition only where
      the text is unambiguous (a C value directly attached to temper / quench);
      everything else stays missing. Not used by the shipped composition model.
    * Test temperature is not reported; no row names a non-ambient test, so
      these are room-temperature datasheet values by convention.
    """
    path = Path(path) if path is not None else RAW_DIR / "mendeley_jmwb9ddd43.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n  curl -L -o {path} {MENDELEY_URL}\n"
            f"See data/README.md for the DOI, licence and expected checksum."
        )

    raw = pd.read_excel(path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    source_id = "mendeley_jmwb9ddd43"
    text = raw["Processing condition"].astype(str)

    out = pd.DataFrame()
    out["sample_id"] = source_id + ":" + raw["Entry"].astype(str)
    out["source_id"] = source_id
    out["grade_id"] = raw["Name"].astype(str).map(_grade_key)
    out["source_label"] = "Mendeley"
    out["provenance"] = "mendeley"
    out["steel_family"] = "unknown"
    out["measurement_kind"] = np.where(
        raw["Name"].astype(str).str.startswith("ASTM"), "spec_minimum", "measured")
    out["austenitize_T"] = _first_celsius(
        text, r"(?:quenched|austenitized|reheated to)(?: at| from)?\s+(\d{3,4})\s?C\b")
    temper_after = _first_celsius(text, r"(\d{3,4})\s?C\b[^,;]*?\btemper")
    out["temper_T"] = temper_after.fillna(
        _first_celsius(text, r"temper(?:ed)?,?(?: at)?\s+(\d{3,4})\s?C\b"))
    out["quench_medium"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    out["yield_strength"] = pd.to_numeric(raw["Yield strength (MPa)"], errors="coerce")
    out["tensile_strength"] = pd.to_numeric(raw["(Ultimate) Tensile strength (MPa)"],
                                            errors="coerce")
    out["elongation"] = pd.to_numeric(raw["Ductility (%)"], errors="coerce")
    out["elongation_standard"] = "unknown"
    for col in ("reduction_area", "impact_J_avg", "hardness"):
        out[col] = np.nan

    comp_dense = raw[list(schema.ELEMENTS)].apply(pd.to_numeric, errors="coerce")
    comp_dense = comp_dense.mask(comp_dense == 0.0)
    comp_dense.insert(0, "sample_id", out["sample_id"].values)

    return Ingested(
        samples=out,
        composition=_to_long(comp_dense, schema.ELEMENTS),
        source_id=source_id,
        raw_path=path,
        raw_sha256=sha256_of(path),
        raw_rows=len(raw),
    )


def load_merged() -> Ingested:
    """SteelBench + Mendeley, concatenated with their labels intact.

    Each row keeps its own source_id / provenance / measurement_kind, so any
    evaluation can still be split by source. Rows that appear in both (the same
    AISI datasheet row, reached through SteelBench's Kaggle tier) are removed by
    `clean`'s cross-source duplicate rule, keeping the SteelBench copy.
    """
    return _combine([load_steelbench(), load_mendeley()])


LITERATURE_URL = "https://ndownloader.figshare.com/files/68056564"


def load_literature(path: Path | None = None) -> Ingested:
    """Load the literature-derived steel dataset (figshare 10.6084/m9.figshare.32755830 v2).

    About 41k records extracted automatically from ~2,900 papers. The row
    selection in `literature.select` keeps only room-temperature tensile tests of
    a wt% chemistry with carbon reported, and drops any row near a held-out
    steel. Each paper becomes one grade group: its samples usually share one
    chemistry and differ only in processing, so splitting a paper across folds
    would leak.
    """
    from . import external, literature  # external imports this module

    path = Path(path) if path is not None else RAW_DIR / "steel_literature_figshare_32755830.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n  curl -L -o {path} {LITERATURE_URL}\n"
            f"See data/README.md for the DOI, licence and expected checksum."
        )
    raw = pd.read_csv(path, low_memory=False)
    recs = literature.records(raw)
    kept, steps = literature.select(recs, external.load_guard_composition())
    source_id = "literature_figshare_32755830"

    out = pd.DataFrame(index=kept.index)
    out["sample_id"] = source_id + ":" + kept.index.astype(str)
    out["source_id"] = source_id
    out["grade_id"] = "paper:" + kept["article_doi_normalized"].fillna(kept.index.to_series()).astype(str)
    out["source_label"] = "literature"
    out["provenance"] = "literature"
    out["steel_family"] = "unknown"
    out["measurement_kind"] = "measured"
    for col in schema.PROCESS_NUMERIC:
        out[col] = np.nan
    out["quench_medium"] = pd.Series(pd.NA, index=kept.index, dtype="string")
    for col in schema.TARGETS:
        out[col] = pd.to_numeric(kept[col], errors="coerce")
    out["elongation_standard"] = "unknown"
    for col in ("reduction_area", "impact_J_avg", "hardness"):
        out[col] = np.nan
    out = out.reset_index(drop=True)

    comp_dense = kept[list(schema.ELEMENTS)].reset_index(drop=True)
    comp_dense.insert(0, "sample_id", out["sample_id"].values)
    return Ingested(
        samples=out,
        composition=_to_long(comp_dense, schema.ELEMENTS),
        source_id=source_id,
        raw_path=path,
        raw_sha256=sha256_of(path),
        raw_rows=int(recs.shape[0]),
        selection=tuple(steps),
    )


def _combine(parts: list[Ingested]) -> Ingested:
    digest = hashlib.sha256("".join(p.raw_sha256 for p in parts).encode()).hexdigest()
    return Ingested(
        samples=pd.concat([p.samples for p in parts], ignore_index=True),
        composition=pd.concat([p.composition for p in parts], ignore_index=True),
        source_id="+".join(p.source_id for p in parts),
        raw_path=Path(" + ".join(str(p.raw_path).replace("\\", "/") for p in parts)),
        raw_sha256=digest,
        raw_rows=sum(p.raw_rows for p in parts),
        parts=tuple(parts),
    )


def load_merged_all() -> Ingested:
    """SteelBench + Mendeley + the literature set, labels intact."""
    return _combine([load_steelbench(), load_mendeley(), load_literature()])


LOADERS = {"steelbench": load_steelbench, "mendeley": load_mendeley, "literature": load_literature,
           "merged": load_merged, "merged_all": load_merged_all}


def run(source: str, out_dir: Path = PROCESSED_DIR) -> dict:
    """Ingest -> clean -> write processed tables, manifest and noise floor."""
    if source not in LOADERS:
        raise KeyError(f"unknown source {source!r}; known: {sorted(LOADERS)}")

    ing = LOADERS[source]()
    cleaned, report = clean.clean(ing.samples, ing.composition)

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.samples.to_csv(out_dir / "samples.csv", index=False)
    cleaned.composition.to_csv(out_dir / "composition_long.csv", index=False)

    floor = noise_floor.compute(cleaned.samples, cleaned.composition)

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stresspredict_version": __version__,
        "source_id": ing.source_id,
        "raw_file": str(ing.raw_path).replace("\\", "/"),
        "raw_sha256": ing.raw_sha256,
        "raw_rows": ing.raw_rows,
        "raw_files": [
            {"source_id": p.source_id, "file": str(p.raw_path).replace("\\", "/"),
             "sha256": p.raw_sha256, "rows": p.raw_rows,
             **({"selection": list(p.selection)} if p.selection else {})}
            for p in (ing.parts or (ing,))
        ],
        "rows_out": int(len(cleaned.samples)),
        "cleaning": report,
        "noise_floor": floor,
    }
    (out_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest a raw source into data/processed/.")
    ap.add_argument("--source", default="steelbench", choices=sorted(LOADERS))
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args(argv)

    m = run(args.source, args.out_dir)
    c = m["cleaning"]
    print(f"source        : {m['source_id']}")
    print(f"raw sha256    : {m['raw_sha256'][:16]}...")
    print(f"rows in       : {m['raw_rows']}")
    print(f"rows out      : {m['rows_out']}   (dropped {c['rows_dropped']})")
    if c["drops_by_rule"]:
        print("rows dropped by rule:")
        for rule, n in c["drops_by_rule"].items():
            print(f"    {n:5d}  {rule}")
    if c["nulls_by_rule"]:
        print("values nulled by rule:")
        for rule, n in c["nulls_by_rule"].items():
            print(f"    {n:5d}  {rule}")
    print("target availability after cleaning:")
    for t, n in c["target_counts"].items():
        print(f"    {n:5d}  {t}")
    print(f"\nwrote {args.out_dir}/samples.csv, composition_long.csv, data_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
