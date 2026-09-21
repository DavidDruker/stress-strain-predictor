# Evaluation methodology

The point of this document is that the headline number should be believable
without taking anything on trust.

## The problem this design exists to solve

Under a composition-only feature set, two rows of the same steel grade heat
treated differently have **literally identical feature vectors and different
targets**. A random K-fold split puts a row's exact twin in the training set.
The model then "predicts" the test row by recalling its twin, and the reported
score measures *grade lookup*, not alloy-design generalisation.

That is not a hypothetical. It is the default outcome of the obvious pipeline,
and it is the reason a composition-only steel model can report R^2 ~ 0.95 while
being useless for its stated purpose.

## The four protocols

| Protocol | Splitter | Group | Question it answers |
|---|---|---|---|
| `random_kfold` | `KFold(5, shuffle)` | -- | **Leaky upper bound only.** Reported and labelled as such. |
| `gkf_grade` | `GroupKFold(5)` | `grade_group` | Can it predict an unseen grade? **Headline protocol.** |
| `lofo_family` | leave-one-family-out | `family_group` | Does it transfer to an unseen alloy family? |
| `loso_source` | leave-one-source-out | `provenance` | Does it survive a change of laboratory? |

`random_kfold` is carried deliberately. The **gap** between it and `gkf_grade` is
the size of the leak, and publishing that gap is more informative than quietly
reporting the better of the two.

Where a source ships no grade label, rows are grouped by **composition hash**:
elements rounded to the dataset's reporting precision, then SHA-256. An element
that was never reported is absent from the key rather than encoded as zero --
"V not measured" and "V = 0.00 wt%" are different states of knowledge.

## Three things that are easy to get wrong, and how they are handled

**1. Tuning must be nested inside the outer fold.** `HalvingRandomSearchCV` runs
*within each outer training fold*, never once over the whole dataset. Tuning on
all the data and then reporting grouped CV is the single most common way a
project like this leaks, and it is invisible in the results table.

**2. The inner search is grouped too -- always.** Even under `random_kfold`,
whose outer split is deliberately leaky, the inner CV is `GroupKFold` on grade.
Otherwise the hyper-parameters are selected using exactly the twins the grouped
protocol exists to separate.

**3. Uneven folds must not be pooled.** `lofo_family` produces folds ranging from
~12 to ~347 rows. A single pooled MAE lets one catastrophic family hide behind
eleven good ones, so the per-fold array is kept and the **median and the worst
fold** are both reported. For an extrapolation claim the worst family is the
honest headline. Families with fewer than 10 rows are too small to score; they
stay in training and are listed explicitly rather than silently dropped. This
reproduces k=12, the value SteelBench's own dataset card specifies.

## Metrics

Reported per target on the original scale: **MAE (primary)**, RMSE, R^2, median
AE, MAPE for the strengths, and the per-fold standard deviation.

MAE leads for a mathematical reason, not a stylistic one. The targets are
modelled through a log link, and inverse-transforming a log-space prediction
recovers the conditional **median**, not the mean. MAE is the loss that median
minimises; leading with RMSE would be scoring a median predictor with a
mean-predictor's metric.

R^2 is reported for comparability with the literature and is deliberately not
led with -- it is the metric most inflated by grade leakage.

Every landmark is also scored a second time on the **common core**: the rows
where all three landmarks are measured. Because the three targets have different
availability (UTS 1,359 rows, YS 984, elongation 661), this fixed row set is what
makes numbers comparable across targets and across target parameterisations
without the row count changing underneath the comparison.

## The noise floor: what the model is measured against

Rows that share an exact composition differ only in processing history. A
composition-only model sees one feature vector for all of them and can emit only
one number, so **the spread within a composition group is a hard lower bound on
its achievable MAE**. No model capacity crosses it.

Two estimates are reported, because neither alone is honest:

* `mae_floor_insample` predicts each row with its own group's median -- the
  median is fitted on the row it scores, so this is biased **low**;
* `mae_floor_loo` predicts each row with the median of the *other* rows in its
  group -- out-of-sample, but fitted on n-1 points at small group sizes, so
  biased **high**.

The true floor is bracketed between them.

### Stratification is not optional here

55% of SteelBench's open release is **specification minima**, not measurements.
Those are near-deterministic given a grade, so pooling them into the spread
calculation roughly **halves** the apparent floor -- which would turn the
project's headline number into an artefact of how the benchmark was assembled.
The floor is therefore computed per `measurement_kind`, and the **measured-only**
stratum is the one quoted. `tests/test_integration.py` asserts that pooling does
deflate it, so the reason for the stratification stays documented in code.

## The wiring test

Before any result is read, ridge coefficients are checked for metallurgical
sanity: **carbon must be strongly positive for both strengths and negative for
elongation.** This is not a performance metric. It is a test that the featuriser,
target transform and joins are wired correctly. If those signs are wrong, every
other number in the report is meaningless no matter how good it looks, and a
gradient booster stacked on top would hide the fault rather than reveal it.

## Why the ladder stops at HistGradientBoosting

`dummy -> ridge -> random forest -> extra trees -> hist GBM`, and no further. The
claim that XGBoost would not change the conclusion is backed by the reported
fold-to-fold standard deviation: on ~1.3k rows a model difference smaller than
that spread is not a difference. Reporting the number that licenses the decision
is a stronger signal of judgment than a marginally better score.

## Reproducing

```bash
python -m stresspredict.ingest --source steelbench
python -m stresspredict.evaluate --protocol all
python -m stresspredict.plots
```

Every artifact carries a sidecar with the training CSV's SHA-256, the seed, the
library versions and the tuning protocol. Seeds are fixed and `n_jobs` explicit.
