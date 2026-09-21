# Data provenance

The chain from a DOI to a fitted model, in the order a reviewer would want to
walk it. Licences, checksums and the discrepancy between the advertised and
deposited SteelBench are in [`data/README.md`](../data/README.md); this document
covers what the code does to the bytes after download.

## The chain

```
Zenodo DOI 10.5281/zenodo.18530558
  -> data/raw/steelbench_core_open.csv          (SHA-256 recorded in the manifest)
  -> ingest.load_steelbench()                   rename / label only, no filtering
  -> clean.clean()                              every removal counted
  -> data/processed/samples.csv                 one row per sample
     data/processed/composition_long.csv        sample_id, element, wt_pct
     data/processed/data_manifest.json          the audit trail
  -> train.py  -> artifacts/*.joblib + *.json   sidecar carries the raw SHA-256
```

Each stage has exactly one job, and the boundaries are the point:

* **`ingest`** renames, converts and labels. It never filters, imputes or
  repairs. A loader that quietly dropped a bad row would make the manifest a
  lie.
* **`clean`** applies validity rules and counts every one.
* **`train`** records the SHA-256 of the raw file it descended from, so an
  artifact can always be traced to the exact bytes that produced it.

## Composition is stored long, not wide

`(sample_id, element, wt_pct)`, projected to a dense matrix only inside
`features`. Nine hard-coded element columns would cost nothing today and force a
rewrite the moment a titanium source arrives reporting Fe, Al and V. Absence is
represented by the row simply not existing, which keeps "not reported" distinct
from `0.0` right up to the point where `features` makes an explicit, documented
decision about it.

## Cleaning rules

Two kinds, and the distinction matters. A **null** rule removes one implausible
*value* and keeps the row; a **drop** rule removes the whole row. Blanket
dropping a row because one of its three targets is corrupt throws away real
measurements -- on this dataset it would cost 15 perfectly good UTS observations.

**Clipping is never used.** A clipped transcription error is an invented
measurement that looks exactly like a real one.

| Rule | Action | Rationale |
|---|---|---|
| element wt% outside [0, 70] | null the cell | transcription error |
| `yield_strength` outside [100, 2200] MPa | null the value | implausible for these alloys |
| `elongation` outside [0.5, 80] % | null the value | implausible |
| `austenitize_T` outside [500, 1300] C | null the value | implausible |
| `temper_T` outside [100, 800] C | null the value | implausible |
| `tensile_strength` missing | drop the row | UTS anchors the parameterisation |
| `tensile_strength` outside [150, 2500] MPa | drop the row | untrustworthy row |
| `yield_strength >= tensile_strength` | drop the row | physically impossible |
| duplicate `sample_id` | drop the row | keep first occurrence |

The manifest must balance: `rows_in == rows_out + rows_dropped`, and
`rows_dropped == sum(drops_by_rule)`. A test asserts both, so the report cannot
drift from the data.

## Diagnosis, not just rejection

When a value is implausible *as its own quantity* but plausible *as another one*,
that is an upstream column transposition, not random noise -- and saying so is
more useful to the next person than a bare count.

The manifest records one such finding in SteelBench: **13 rows across grades
EMK0319/0320/0321 carry values in the `yield_strength` column that fall in the
elongation range and vary inversely with UTS** -- 10 at 1035 MPa, 11 at 965,
12 at 895, 13 at 825, 15 at 690. A monotone inverse relationship between
"yield strength" and UTS is the signature of the strength-ductility trade-off,
which is what elongation does. They are almost certainly elongation percentages
written into the wrong column upstream.

They are **nulled and diagnosed, not repaired**: recovering them would mean
guessing which column the true yield strength went to.

## What `measurement_kind` is for

SteelBench's `emk_spec_verified` tier (753 of 1,360 rows) contains
**specification minima**, not measurements. Ingest maps that tier to
`measurement_kind = "spec_minimum"` and every other tier to `"measured"`.

This single column is load-bearing. Specification minima are near-deterministic
given a grade, so including them in the within-composition spread calculation
roughly halves it -- and that spread is this project's headline measurement. The
column is what lets `noise_floor` report the strata separately and quote the
measured-only number.

## Columns deliberately discarded

| Column | Why |
|---|---|
| `condition` | 98.9% missing; the 15 surviving values are free text in mixed Russian and English |
| `split` | constant (`pretrain`) in the open release, so it carries no information |

`reduction_area`, `impact_J_avg` and `hardness` are carried through unused: one
column each, and they are the natural auxiliary targets for a later phase.

## Reproducing the processed data

```bash
curl -L -o data/raw/steelbench_core_open.csv \
  https://zenodo.org/api/records/18530558/files/steelbench_core_open.csv/content
python -m stresspredict.ingest --source steelbench
```

Expected: MD5 `ca0e58deadd2900ab7a3f96a25d5b0d2`, 1,360 rows in, 1,359 out,
1 row dropped (`drop_yield_ge_tensile`), 15 values nulled
(`null_yield_strength_out_of_range`).
