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

## Second training source -- Mendeley steel database

| | |
|---|---|
| Title | *A database of mechanical properties of steels* (Ghorbani, Zhao, Birbilis) |
| DOI | [10.17632/jmwb9ddd43.1](https://doi.org/10.17632/jmwb9ddd43.1) |
| Licence | CC BY 4.0 |
| File | `Steel database with labelled clusters.xlsx`, saved as `data/raw/mendeley_jmwb9ddd43.xlsx` |
| SHA-256 | `eaa89971150cff2bd97d8a1fe20b21e908731fcf755b37155608b1200235b484` |
| Rows | 3,234; YS, UTS and elongation on every row |

```bash
curl -L -o data/raw/mendeley_jmwb9ddd43.xlsx \
  https://data.mendeley.com/public-files/datasets/jmwb9ddd43/files/cc944ab8-d3c8-448c-ac1d-172e8cc7e11d/file_downloaded
```

It began as the external zero-shot test set. The SteelBench-only model was
scored on it first: UTS MAE 249 MPa on 3,232 rows, a -210 MPa bias, and no
prediction above 1,052 MPa. After that it became training data.

Several properties of this file shape how it is loaded:

* **Room temperature by convention.** No row states a test temperature. Every
  "at X C" in the text is a heat-treatment step, and 8 rows say "at RT" outright.
* **A composition of 0 means "not specified"**, since every row lists all 20
  elements. The loader drops the zeros, which is the same state a blank has in
  SteelBench.
* **The ASTM rows are specification minima.** They list grade, class and
  thickness, so they are tagged `spec_minimum` like SteelBench's EMK tier.
* **Processing is free text**, grouped into 12 clusters. Temperatures are
  extracted where they are unambiguous. The shipped model does not use them.
* **It overlaps SteelBench.** SteelBench's Kaggle tier holds 112 AISI grades from
  the same handbook lineage. 308 rows repeat a SteelBench row's exact
  (UTS, YS, EL) and are dropped by `clean`'s `drop_cross_source_duplicate` rule,
  which keeps the SteelBench copy. Mendeley AISI names are also mapped to
  SteelBench grade ids, so grade-grouped splits cannot separate the two copies.
  Only 984 of the 3,232 rows share no grade, triple or composition with SteelBench.

## Held-out promotion set -- matminer `steel_strength`

| | |
|---|---|
| Source | Citrine dataset 153092, *Mechanical properties of some steels*, via matminer |
| URL | <https://ndownloader.figshare.com/files/13354691> (figshare 10.6084/m9.figshare.7250453, MIT) |
| SHA-256 | `e36501d7057cd833223bb8ed9948668b5ac90fd585d29a749f45af51c1d7f6ad` (matches matminer's metadata) |
| Rows | 312; YS and UTS on all, elongation on 303; wt% composition |

This set is never used for training or tuning. `python -m stresspredict.external`
scores a model on it. It has no composition within 0.3 wt% per element of any
SteelBench row, and no property overlap with either training source.
matbench_steels is the yield-only, deduplicated view of the same 312 steels.

It covers only ultra-high-strength steels. Every row has YS >= 1,000 MPa, mostly
maraging and secondary-hardening grades that are strengthened by Co, Ti and W,
which are not model inputs. Test temperature is not stated.

## Deliberately excluded -- NIMS MatNavi

The richest steel property source available, and not used. Its terms prohibit
scraping and bulk acquisition, so it cannot be a pipeline dependency. It is
named here as a known source deliberately not scraped.

Note the distinction: 360 NIMS-derived rows arrive *inside* SteelBench under
SteelBench's own CC BY 4.0 licence. That is redistribution by the benchmark's
authors, not acquisition from MatNavi by this project.

## How the sources are combined, and how they are not

The original rule was *never merge*, because merging would destroy SteelBench's
grade, family and provenance labels. Once the zero-shot test had shown what
SteelBench alone could not cover, the rule was changed, but its reason still
holds:

* **Labels survive the merge.** Every row keeps its own `source_id`,
  `provenance` and `measurement_kind`, so any result can still be split by
  source.
* **`data/processed/` stays SteelBench-only.** Every cross-validated number in
  `reports/results.md` is a SteelBench evaluation and must stay reproducible.
  The merged pool is written separately to `data/processed_merged/`.
* **One dataset is never merged.** `steel_strength` is the held-out set the
  shipped model was promoted on. The promotion rule was fixed before any
  held-out number was computed.
