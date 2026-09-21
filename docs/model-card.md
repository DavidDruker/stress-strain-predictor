# Model card -- stresspredict v0.1.0

## Model details

**What it is.** Three gradient-boosted regressors that predict the *landmarks* of
a steel's engineering stress-strain curve -- yield strength, ultimate tensile
strength and total elongation -- from bulk chemistry alone.

**Why landmarks and not a curve.** A tensile curve is thousands of points that
are almost entirely determined by a handful of them. Predicting the landmarks and
reconstructing between them is both a smaller learning problem and a more honest
one: the intermediate points carry no independent information. Reconstruction
itself is deliberately out of scope for v1 (see [Limitations](limitations.md)).

| | |
|---|---|
| Version | 0.1.0 |
| Type | `HistGradientBoostingRegressor` x3, on a transformed target parameterisation |
| Inputs | 9 element weight fractions (C, Mn, Si, Cr, Ni, Mo, V, Cu, Al) |
| Outputs | yield strength (MPa), tensile strength (MPa), elongation (%), plus derived yield ratio, strength gap and UTS x EL |
| Training data | SteelBench v1.0 open release, CC BY 4.0, DOI [10.5281/zenodo.18530558](https://doi.org/10.5281/zenodo.18530558) |
| Licence | MIT (code); the data is CC BY 4.0 and licensed separately |
| Framework | scikit-learn only -- no XGBoost, no matminer, no PyTorch |

### The physical invariant

UTS > YS is enforced **by construction**, not by a post-hoc check. Rather than
fitting three independent regressors on the raw targets, the model fits a
parameterisation from which the inequality follows algebraically:

```
UTS = exp(z_uts)                     z_uts   = log(UTS)
YS  = UTS * sigmoid(z_ratio)         z_ratio = logit(YS / UTS)
EL  = exp(z_el)                      z_el    = log(EL)
```

`sigmoid < 1` strictly, so no prediction the model can emit puts yield strength
at or above tensile strength. A test asserts this over 10,000 compositions
sampled from the training hull, including at encoded values no optimiser would
ever produce -- because `sigmoid` saturates to exactly 1.0 in float64 and would
otherwise turn a strict inequality into an equality.

An alternative parameterisation (predict YS, then the strength *gap*) is
implemented and gives the same guarantee. The ratio form is the default because
in this dataset UTS is reported for all 1,359 usable rows while YS is reported
for 984 -- all 360 NIMS rows carry UTS alone -- so anchoring on UTS trains the
best-populated target on 38% more data. The choice is settled by measurement
(`evaluate --parameterisation gap`), not assertion.

## Intended use

**Intended.** Early-stage alloy screening and triage: ranking candidate
chemistries, sanity-checking a specification, getting an order-of-magnitude
expectation before committing to a heat. Teaching and portfolio demonstration of
leakage-aware evaluation on small materials datasets.

**Not intended.** Design allowables, certification, safety-critical decisions, or
any use where the number substitutes for a tensile test. Also not intended for
distinguishing heat treatments -- see below, because this is the model's defining
constraint rather than a footnote.

## The defining constraint, measured rather than asserted

For steels, **heat treatment dominates mechanical properties**. The same
chemistry quenched and tempered at 200 C versus 650 C can differ by 500+ MPa in
yield strength. A composition-only model sees one feature vector for both and can
emit only one number. It is therefore honestly described as a **grade-level
prior**, not a process model.

Rather than treat that as a caveat, this project measures it. Rows sharing an
exact composition differ only in processing, so the spread *within* a composition
group is a hard lower bound on any composition-only model's MAE.

Measured on **measured rows only** (see the next section for why that
qualification matters):

| Target | Noise floor (in-sample .. LOO) | Groups / rows |
|---|---|---|
| Yield strength | **107.2 .. 191.5 MPa** | 92 / 232 |
| Tensile strength | **63.9 .. 103.3 MPa** | 212 / 592 |
| Elongation | **3.3 .. 6.1 pp** | 92 / 232 |

The two columns bracket the true floor from below and above. The residual between
the model's MAE and this band is the only part a better model could ever recover;
the band itself is processing, and no model with these inputs crosses it.

### The trap in that measurement

55% of SteelBench's open release (753 of 1,360 rows) is **specification minima**,
not measurements. Those are near-deterministic given a grade. Pooling them into
the same calculation gives a yield-strength floor of 61.2 .. 94.3 MPa -- roughly
**half** the measured-only value, and an artefact of how the benchmark was
assembled rather than a fact about steel.

Quoting the pooled number would have made the model look far closer to the
physical ceiling than it is. The distinction is carried as a first-class column
(`measurement_kind`), the strata are reported separately, and a test asserts that
pooling does deflate the floor so the reason stays documented in code.

## Evaluation

Full methodology in [evaluation.md](evaluation.md); generated results in
[`reports/results.md`](../reports/results.md).

The headline protocol is **grade-grouped 5-fold CV** (`gkf_grade`). Random K-fold
is also reported, explicitly labelled as a leaky upper bound, because under a
composition-only feature set two rows of the same grade at different tempers are
the same feature vector with different targets -- a random split puts a row's
twin in training and measures grade lookup rather than generalisation.

Hyper-parameters are searched **inside each outer training fold**, and the inner
search is grouped by grade even under the deliberately leaky outer protocol.

## Factors

Performance varies substantially across:

* **Steel family** -- reported per held-out family under `lofo_family`, with the
  worst family shown rather than a pooled mean.
* **Measurement kind** -- specification minima behave differently from
  measurements; `--rows measured` restricts to the latter.
* **Data source** -- `loso_source` holds out a whole provenance tier. This test is
  weak for yield strength and elongation, which only two of the three tiers
  report.
* **Position in the training hull** -- `predict()` reports which supplied elements
  fall outside the range seen in training and by how much.

## Metrics

MAE is primary. This is a consequence of the parameterisation, not a preference:
the targets are modelled through a log link, and inverse-transforming a log-space
prediction recovers the conditional **median**, which is the quantity MAE scores.
RMSE, R^2, median AE and MAPE are reported alongside; R^2 is deliberately not led
with, being the metric most inflated by grade leakage.

Per-fold standard deviation is reported for every number, and is what licenses
stopping the model ladder at HistGradientBoosting: a model difference smaller
than the fold-to-fold spread is not a difference.

## Correctness checks that run before any result is read

* **Ridge coefficient signs.** Carbon must be strongly positive for both
  strengths and negative for elongation. Not a performance metric -- a wiring
  test. If those signs are wrong, the featuriser, target transform or joins are
  broken and every other number is meaningless.
* **The invariant on out-of-fold predictions.** UTS > YS is verified on
  cross-validated predictions, not only on the final fitted model.
* **The manifest balances.** `rows_in == rows_out + rows_dropped`, and every
  rule that fired is documented.

## Training data

SteelBench v1.0 open release: **1,360 rows, 562 grades, 17 families**, of which
1,359 survive cleaning. Note this is *not* the 1,636 rows / 594 grades the Zenodo
record advertises -- the open deposit withholds the MMPDS and internal-lab tiers.
See [`data/README.md`](../data/README.md).

Targets are not co-present: UTS 1,359 rows, YS 984, elongation 661. Each
component trains on the rows where its own inputs exist.

One upstream data error was found and is documented rather than silently
repaired: 13 rows carry what are almost certainly elongation percentages in the
`yield_strength` column (values of 10-15 varying inversely with UTS). They are
nulled and diagnosed in the manifest.

## Ethical considerations

The dataset contains no personal data. The main risk is misuse: a plausible-looking
number applied to a safety-critical decision it cannot support. The model
therefore ships with a caveat string in every prediction, an explicit
`limitations` field inside the artifact sidecar, and a range guard that names
out-of-hull inputs.

NIMS MatNavi -- the richest available source -- was deliberately not scraped, as
its terms prohibit bulk acquisition.

## Caveats and recommendations

See [limitations.md](limitations.md) in full. The short version: composition-only
against a measured physical ceiling; strain at UTS and the hardening exponent are
not predicted because no open dataset reports them; no calibrated uncertainty in
v1; 55% of training rows are specification minima; and elongation is trained on
661 rows with an unknown mix of gauge-length standards.

Recommended next step for a user who needs more: supply heat-treatment
temperatures. The ablation flag exists (`--features comp_ht`) and quantifying
that gain is Phase 2.
