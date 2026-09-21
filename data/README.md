# Data

No dataset is committed to this repository. Every file below is reproducible
from a DOI, and each is licensed by its publisher, not by this project's MIT
licence. `data/raw/` and `data/processed/` are gitignored; `data/external/` is
committed because it is small and vendored with provenance.

## Training and evaluation -- SteelBench v1.0 (open release)

| | |
|---|---|
| DOI | [10.5281/zenodo.18530558](https://doi.org/10.5281/zenodo.18530558) |
| Licence | CC BY 4.0 |
| File | `steelbench_core_open.csv` |
| MD5 | `ca0e58deadd2900ab7a3f96a25d5b0d2` |
| Rows | **1,360** (see the discrepancy note below) |
| Grades / families | 562 / 17 |

Fetch it with:

```bash
curl -L -o data/raw/steelbench_core_open.csv \
  https://zenodo.org/api/records/18530558/files/steelbench_core_open.csv/content
```

Attribution, as CC BY 4.0 requires: SteelBench Team, *SteelBench v1.0: A
Benchmark Dataset for Steel Mechanical Property Prediction with Grade-Shift
Evaluation* (2026), Zenodo, DOI 10.5281/zenodo.18530558, CC BY 4.0.

### Discrepancy between the advertised dataset and the deposited one

The Zenodo record description and `DATASET_CARD.md` both describe **1,636 rows
across 594 grades**, and mention a second file `steelbench_full.csv` with imputed
heat-treatment parameters.

Neither matches what is actually deposited. The record contains four files --
`steelbench_core_open.csv`, `DATASET_CARD.md`, `LICENSE`, `croissant.json` --
and the CSV has **1,360 rows across 562 grades**. The difference is the withheld
tiers: the open release excludes MMPDS and internal laboratory data for
licensing reasons, which the card documents in its provenance table
(753 + 360 + 247 = 1,360). `steelbench_full.csv` is not deposited at all.

This repository targets the deposited file, because it is the only one anyone
can download. `tests/test_integration.py` asserts 1,360 / 562 so the numbers in
the README cannot silently drift from the data.

### Two properties of this data that shape the whole project

**1. 55% of rows are specification minima, not measurements.** The
`emk_spec_verified` tier (753 rows) holds specification *minimum* values. Those
are near-deterministic given a grade, so including them in any within-composition
spread calculation roughly halves it. Since that spread is this project's
headline measurement -- the physical floor under a composition-only model -- the
distinction is carried as a first-class column, `measurement_kind`, and
`noise_floor` reports the strata separately. The measured-only number is the one
quoted.

**2. The three targets are not co-present.**

| Target | Rows (after cleaning) | Note |
|---|---|---|
| `tensile_strength` | 1,359 | complete |
| `yield_strength` | 984 | absent for *all* 360 NIMS rows |
| `elongation` | 661 | |

All 360 NIMS rows carry UTS alone. This is why the default target
parameterisation anchors on UTS and predicts the yield *ratio* rather than
anchoring on YS and predicting the strength gap -- see `stresspredict/targets.py`.

### Units

SteelBench reports strengths in MPa and elongation in %, so no conversion is
applied on ingest. The elongation gauge-length standard (A5 / A50mm / ...) is
**not** reported; it is tagged `unknown` rather than assumed, because elongation
is only comparable within a standard and the two are not inter-convertible
without the specimen geometry.

## Public yardstick (Phase 1b) -- matbench_steels

312 steels, composition-only, yield strength. Verified leaderboard MAE (MPa):
dummy 229.7 · RF-Regex 90.6 · MODNet 87.8 · AutoML-Mat 82.3 · TPOT-Mat 79.9.

To be vendored under `data/external/` with provenance, exported once in a
throwaway environment rather than taking a dependency on `matbench` (which pulls
matminer and pymatgen). Before any number is quoted, SteelBench-to-Citrine row
overlap is checked -- both draw on published literature, and shared rows would
contaminate the comparison.

## External zero-shot test set (Phase 2)

*A database of mechanical properties of steels*,
[10.17632/jmwb9ddd43.1](https://doi.org/10.17632/jmwb9ddd43.1), CC BY 4.0,
3,234 entries. Used strictly zero-shot and **never merged** into training.

## Deliberately excluded -- NIMS MatNavi

The richest steel property source available, and not used. Its terms prohibit
scraping and bulk acquisition, so it cannot be a pipeline dependency. It is
named here as a known source deliberately not scraped.

Note the distinction: 360 NIMS-derived rows arrive *inside* SteelBench under
SteelBench's own CC BY 4.0 licence. That is redistribution by the benchmark's
authors, not acquisition from MatNavi by this project.

## Why the sources are never merged

Each dataset has exactly one job. Merging would destroy SteelBench's shipped
grade / family / provenance labels, which are what make its evaluation protocols
citable, and "trained on SteelBench, evaluated zero-shot on 3,234 entries from an
independent source" is a strictly stronger claim than a merged pile of rows.
