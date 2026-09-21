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
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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


LOADERS = {"steelbench": load_steelbench}


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
