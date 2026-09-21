# Limitations

Stated plainly, because a portfolio project that hides these is worth less than
one that measures them.

## 1. Composition-only, against a measured physical ceiling

The shipped model sees chemistry and nothing else. For steels this is the
binding constraint, not a detail: **heat treatment dominates mechanical
properties.** The same chemistry, quenched and tempered at 200 C versus 650 C,
can differ by 500+ MPa in yield strength. A composition-only model sees one
feature vector for both and can emit only one number.

So the model is honestly described as a **grade-level prior**, not a process
model. It answers "what do steels of roughly this chemistry usually do?" and
cannot answer "what will *this* heat, treated *this* way, do?".

This project does not treat that as a caveat to be buried. It measures it. The
within-composition spread of yield strength on measured rows is a hard lower
bound on any composition-only model's MAE -- see `docs/evaluation.md` and the
`noise_floor` block in `data/processed/data_manifest.json`. The residual between
the model and that floor is the only part a better model could ever recover; the
floor itself is processing, and no model with these inputs crosses it.

## 2. Landmarks that are NOT predicted

| Landmark | Status |
|---|---|
| Yield strength | predicted |
| Ultimate tensile strength | predicted |
| Total elongation at fracture | predicted |
| Young's modulus | **deliberately not modelled** -- it is ~200-210 GPa for essentially every steel regardless of alloying. A constant beats a model, and saying so is the correct answer. |
| **Strain at UTS (uniform elongation)** | **not predicted -- absent from every open dataset found** |
| Hollomon *n* and *K* | not predicted -- same gap |
| Yield-plateau end / Lüders strain | not predicted -- needs full curves, not summary tables |

The last three are the hard gate on reconstructing a continuous stress-strain
curve, and it is a **data** gate, not a modelling one. No open source located for
this project reports strain at UTS or a hardening exponent alongside composition.
Until one exists, any "predicted curve" would be a curve fitted through three
predicted points using an assumed hardening law -- which is a legitimate thing to
ship, but must be labelled a *model-based reconstruction constrained to pass
through predicted landmarks*, not a learned curve. It is gated behind Phase 4 for
exactly this reason.

## 3. No calibrated uncertainty in v1

`predict()` returns point values. The range guard reports which input elements
fall outside the training hull, which is actionable but is not a probability.
Split-conformal intervals are Phase 3; `splits.reserve_calibration_holdout`
already reserves a grade-grouped calibration set so adding them needs no
restructuring.

## 4. What the training data actually is

* **55% of rows are specification minima, not measurements.** SteelBench's
  `emk_spec_verified` tier holds specification *minimum* values. A model trained
  on them is partly learning "what is the guaranteed minimum for this grade",
  which is a different quantity from "what will this heat measure". Metrics are
  reported per measurement kind, and `--rows measured` restricts training and
  evaluation to genuine measurements.
* **1,359 usable rows after cleaning.** Small. Fold-to-fold standard errors are
  reported alongside every mean for this reason.
* **The targets are not co-present.** UTS 1,359 rows, YS 984, elongation 661. The
  elongation model is trained on less than half the data and its numbers should
  be read with that in mind.
* **Elongation gauge length is unknown.** A5 and A50mm are not inter-convertible
  without specimen geometry, and SteelBench does not report which was used. The
  elongation target therefore mixes standards, which inflates its irreducible
  error by an unknown amount. This is tagged, not corrected.
* **The open release is not the advertised dataset.** 1,360 rows, not the 1,636
  the record describes. See `data/README.md`.
* **One upstream transcription error was found and is documented**, not silently
  repaired: 13 rows carry what are almost certainly elongation percentages in the
  `yield_strength` column. They are nulled and diagnosed in the manifest.

## 5. Evaluation caveats

* **LOSO is weak for two of three targets.** The open release has three
  provenance tiers, but yield strength and elongation are reported by only two of
  them, so leave-one-source-out degenerates to a two-fold comparison for those
  targets. Cross-source generalisation for YS and elongation is genuinely
  under-tested here; Phase 2's zero-shot Mendeley evaluation exists to fix that.
* **LOFO holds out 12 of 17 families.** Five families have 1-3 rows, too few to
  score. They remain in training and are listed in the report rather than
  silently dropped.
* **Random K-fold numbers are leaky by construction** and are reported only as a
  labelled upper bound. Quoting them as the headline would be the single easiest
  way to make this project look better and mean less.

## 6. Scope

Steels only. The schema and the long-format composition store are built so a new
alloy system needs a new loader rather than a rewrite, but nothing here has been
validated on aluminium, titanium, copper or nickel alloys.

## 7. Not for safety-critical use

These are statistical predictions from a small, heterogeneous, partly
specification-derived dataset. They are not a substitute for physical testing and
must not be used for design allowables.

## 8. Data deliberately not used

**NIMS MatNavi** is the richest steel property source available and is not used.
Its terms prohibit scraping and bulk acquisition, so it cannot be a pipeline
dependency. (360 NIMS-derived rows do arrive inside SteelBench under SteelBench's
own CC BY 4.0 licence -- that is redistribution by the benchmark's authors, which
is a different thing from acquisition by this project.)
