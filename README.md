# stresspredict

Predicts the **landmarks** of a steel's engineering stress-strain curve -- yield
strength, ultimate tensile strength, total elongation -- from bulk chemistry
alone, and measures the result against the physical ceiling that composition-only
prediction actually has.

```bash
$ python -m stresspredict.predict --composition "C=0.40,Mn=0.80,Cr=1.00,Mo=0.20,Si=0.25"
  yield strength      806.5 MPa
  tensile strength    984.7 MPa
  elongation           18.2 %
  yield ratio         0.819
  UTS x EL            17940 MPa.%

  all elements within training range
```

There is also an **interactive test bench** in [`web/`](web), live at
**<https://daviddruker.github.io/stress-strain-predictor/>**: enter a chemistry, a
cross-section and a length, and it runs the same fitted model in the browser over
a 3D bar you can orbit. Asking for the geometry is what removes the percentages
from the answer -- stress in MPa is N/mm^2, so an area turns it into a force and a
length turns elongation into millimetres of travel:

| model output | with area and length |
|---|---|
| yield strength, MPa | **force at yield, kN** |
| tensile strength, MPa | **force at break, kN** |
| elongation, % | **stretch before break, mm** (and final length) |

It has two modes. **Pull to failure** runs the whole test and reads the numbers off
at the break. **Apply a load** ramps a dead load at a constant rate and answers the
question an engineer actually asks -- hang this much off it, what happens? Under
load control the bar is unstable the moment the applied force reaches the maximum
it can carry, so that is the rupture criterion rather than the fracture strain,
and the three outcomes are distinguished: springs back, keeps a permanent set, or
parts.

The fracture animation follows what a ductile steel does rather than a bar simply
splitting: uniform thinning until maximum load, then localisation into a neck
about 1.4 diameters wide while the rest of the gauge stops stretching, then
separation as a **cup and cone** -- voids coalesce at the neck centre where
triaxiality peaks, the flat fibrous crack runs outward, and the last ligament
fails in shear near the surface. How far the neck contracts is not invented: it
comes from a least-squares fit over the 414 heats in SteelBench that report
reduction of area, `RA% = 0.47 x EL% + 31.8`.

The page loads the 300 fitted trees as JSON and evaluates them in JavaScript, so
it is the evaluated model rather than a mock-up. `tests/test_web_parity.py` runs
that JavaScript under Node and asserts it matches Python to 1e-9 on every
training row -- the only thing that makes having the same maths in two languages
defensible. Regenerate the export with `python -m stresspredict.export_web`.

## The shipped model: two sources, promoted on a third

The model in `artifacts/` and on the test bench is fitted on **SteelBench + the
Mendeley steel database**: 4,283 rows, 665 grades. It was chosen by scoring it on
a **third dataset that nothing trains on**, under a rule written down before any
held-out number existed.

**Why a second source.** Scored zero-shot on 3,232 room-temperature Mendeley
steels, the SteelBench-only model missed by about 250 MPa, with a -210 MPa mean
bias and no prediction above 1,052 MPa. Nearly all of that came from quenched,
tempered, carburised and aged steels, which SteelBench barely covers.

**The overlap removed first.** Mendeley is not independent of SteelBench.
SteelBench's Kaggle tier holds 112 AISI grades from the same handbook lineage,
and 308 Mendeley rows repeat a SteelBench row's exact (UTS, YS, EL). A counted
cleaning rule (`drop_cross_source_duplicate`) drops those rows. Mendeley's AISI
names are also mapped to SteelBench's grade ids, so a grade-grouped split keeps
each grade on one side across *both* sources.

**The held-out set.** matminer `steel_strength` (Citrine), 312 steels. No
composition comes within 0.3 wt% per element of any SteelBench row, and no
properties match either source. Reproduce with
`python -m stresspredict.external --baseline <previous artifact>`.

| Held-out, 312 steels | UTS MAE | YS MAE | EL MAE | mean MAE ratio |
|---|---|---|---|---|
| SteelBench only (previous) | 757 MPa | 782 MPa | 6.75 pp | 1.00 |
| **SteelBench + Mendeley (shipped)** | **409** | **430** | **4.90** | **0.605** |
| same, + processing inputs | 616 | 620 | 4.01 | 0.734 |

This set only tests the strong end. Every held-out steel has YS >= 1,000 MPa:
maraging and secondary-hardening grades, strengthened by Co, Ti and W, none of
which are model inputs. It says nothing about typical-steel accuracy, and all
three models still miss it badly.

**What it cost.** In grade-grouped CV on the merged pool, the new model does
slightly *worse* on SteelBench's own rows than the SteelBench-only model: UTS 112
vs 98 MPa, YS 112 vs 102, elongation 5.19 vs 4.86 pp. Wider coverage of hardened
carbon and low-alloy steels cost a little accuracy on SteelBench's stainless-heavy
mix. The promotion rule did not gate on this; it is reported so the trade is
visible.

**Processing is the bigger lever, and it is not shipped.** Given processing class
and temperatures as inputs, the same CV cuts Mendeley-row error from 161 to 125
MPa for UTS and from 3.97 to 2.81 pp for elongation. It lost on the held-out set
only because that set has no processing information to give it. Shipping it would
also mean new inputs on the test bench.

**A third source was tried and rejected.** A CC BY literature dataset of about
41k records, extracted automatically from about 2,900 papers (figshare
32755830), was cut to 1,963 usable rows: room-temperature tensile tests, a wt%
chemistry with carbon, and no hydrogen, corrosion or irradiation context.
Separately, 30 rows within 0.5 wt% of a held-out steel were dropped as a
leakage guard. Adding it made grouped-CV strength error slightly better on
SteelBench (UTS 112 -> 102 MPa) but held-out error worse (UTS 409 -> 546 MPa,
mean ratio 1.137), so under the rule fixed beforehand it was not promoted.
`ingest --source merged_all` reproduces it, and the record is in
[`reports/promotion_rule_2.md`](reports/promotion_rule_2.md). This second
decision also means the held-out set has now been used twice, so a small
future win on it would be weaker evidence.

## Results: the evaluation methodology, on SteelBench

Everything from here to *Figures* evaluates the **SteelBench-only** model on the
SteelBench v1.0 open release: 1,359 rows after cleaning, 562 grades, 17 families.
It stays reproducible from `data/processed/`, which is SteelBench-only on purpose.
Composition-only. Hyper-parameters tuned inside every outer fold. Full tables in
[`reports/results.md`](reports/results.md).

### Headline: grade-grouped 5-fold CV

Each landmark is scored on every row it can be produced for, which is why the row
counts differ.

| Model | YS MAE (n=984) | UTS MAE (n=1,359) | Elongation MAE (n=661) |
|---|---|---|---|
| `dummy` (median) | 164.2 MPa | 190.6 MPa | 8.61 pp |
| `ridge` | 145.8 | 149.9 | 6.08 |
| `random_forest` | **99.2** | **93.5** | **4.77** |
| `extra_trees` | 100.3 | 93.7 | 4.82 |
| `hist_gbm` | 101.9 | 98.1 | 4.86 |

Every model beats the median baseline on every target under the grouped protocol.

### The point of the project: model error against the physical floor

The floor is the within-composition spread on **measured** rows — the error a
perfect composition-only model would still make, because the rows inside each
group differ by heat treatment the features cannot see. Model and floor are both
scored on measured rows, because comparing an all-rows MAE against a
measured-only floor would appear to beat a physical limit.

| Target | Measured noise floor | Shipped model, measured rows | |
|---|---|---|---|
| Yield strength | 107.2 – 191.5 MPa | 144.4 MPa | **inside the band** |
| Elongation | 3.3 – 6.1 pp | 5.94 pp | **inside the band** |
| Tensile strength | 63.9 – 103.3 MPa | 108.9 MPa | just above — real headroom |

Yield strength and elongation are **at the information-theoretic ceiling of
composition-only input**. The residual is not model error to be engineered away;
it is processing history, and no model with these inputs crosses it. Tensile
strength is the one target where a better model still has room.

Pooling specification minima into that floor would report 61.2 MPa for yield
strength instead of 107.2 — and the model would appear to beat physics. It does
not; spec-minimum rows are simply near-deterministic per grade (the same model
scores 87.7 MPa on them and 144.4 MPa on measurements).

### What grade leakage is worth

| Model | UTS: random K-fold → grade-grouped | Inflation |
|---|---|---|
| `dummy` | 184.1 → 190.6 | 1.04× |
| `ridge` | 141.1 → 149.9 | 1.06× |
| `random_forest` | 76.2 → 93.5 | 1.23× |
| `extra_trees` | 78.9 → 93.7 | 1.19× |
| `hist_gbm` | 74.7 → 98.1 | **1.31×** |

The dummy row is the control: ~0% inflation proves the two fold structures are
otherwise comparable, so the gap is the leak and not an artefact of splitting.
The leak is ~0 for models that cannot memorise and 15–31% for those that can,
largest for the highest-capacity model. **Reporting the random-K-fold number as
the headline would have overstated UTS accuracy by 31%.**

### Extrapolation, where it actually hurts

Pooled means hide this, so the worst fold is reported:

| Protocol | Question | UTS MAE | Worst fold |
|---|---|---|---|
| `lofo_family` | unseen alloy family | 143.1 | **232.6** (`stainless_steel`) |
| `loso_source` | unseen laboratory | 170.7 | **183.3** (`emk_spec_verified`) |

Best family is `carbon_low` at 48.9 MPa — a 4.8× spread across families. Any
claim about this model generalising to a new alloy class should quote 232.6, not
143.1. Five families with 1–3 rows are too small to hold out; they stay in
training and are listed in the report.

### Why HistGradientBoosting ships, and why the ladder stops there

On the headline protocol the three tree models are **statistically
indistinguishable**: random forest leads HistGBM on UTS by 4.6 MPa against a
fold-to-fold standard deviation of 25.7 MPa (SEM 11.5) — about 0.4 SEM. Calling
random forest the winner would be reading noise.

The tiebreak is robustness under source shift, where the separation is real:
HistGBM 170.7 MPa vs random forest 198.2 and extra trees 268.7, worst fold 183.3
vs 215.4 vs 295.2. HistGBM ships on that basis, not on the headline number.

The same standard deviation licenses stopping the ladder: a 4.6 MPa difference
between candidate models sits well inside a 25.7 MPa fold-to-fold spread, so
adding XGBoost could not produce a difference this evaluation could detect.

Ridge is worth one line as a cautionary result: it is competitive under
grade-grouped CV but collapses to **1,483 MPa** on the held-out `emk_spec_verified`
source fold, extrapolating linearly in log space outside the training hull.
Trees' inability to extrapolate — usually listed as a weakness — is what makes
them survive that shift.

### Correctness

* **UTS > YS held on 19,650 out-of-fold predictions across all four protocols —
  0 violations.** Guaranteed by the parameterisation, verified on cross-validated
  predictions rather than only on the final fit.
* **Ridge coefficient signs pass**: carbon +0.105 for yield strength, +0.094 for
  tensile strength, −0.118 for elongation. A wiring test, not a score.
* **The data manifest balances**: 1,360 rows in, 1,359 out, 1 dropped
  (`yield_strength >= tensile_strength`), 15 values nulled, every rule documented.

### Figures

In [`reports/figures/`](reports/figures), regenerate with `python -m stresspredict.plots`.

| Figure | What it shows |
|---|---|
| `floor_<target>.png` | model error against the measured noise floor — bars and band both on measured rows, so the comparison is like-for-like |
| `leak_<target>.png` | random K-fold vs grade-grouped, with the dummy as the control |
| `parity_gkf_grade_hist_gbm.png` | out-of-fold parity, coloured by measurement kind |
| `lofo_hist_gbm_*.png` | per-family error, worst family first |
| `banana_gkf_grade_hist_gbm.png` | predicted vs measured strength–ductility space |

The parity plot is worth reading for what it admits: predictions saturate near
1,000 MPa while measured values run to 1,900, and identical compositions produce
visible horizontal bands of identical predictions. That is the grade-prior
behaviour of a composition-only model, drawn rather than described.



### Where the model is entitled to an opinion

Every element has a range the training data actually covers. Outside it the model
is extrapolating, and a gradient-boosted tree extrapolates by repeating the
nearest leaf it knows -- it will return a confident number that means nothing.
The demo shows these ranges under each field and flags any value that leaves
them; `predict()` returns the same information in `range_guard`.

| Element | Min wt% | Max wt% | Heats reporting it |
|---|---|---|---|
| `C` | 0.008 | 2.05 | 4,280 |
| `Mn` | 0.02 | 20 | 4,265 |
| `Si` | 0.01 | 5 | 3,586 |
| `Cr` | 0.01 | 29 | 2,908 |
| `Ni` | 0.01 | 60.5 | 2,362 |
| `Mo` | 0 | 6.5 | 2,800 |
| `V` | 0 | 0.5 | 1,498 |
| `Cu` | 0.01 | 4 | 940 |
| `Al` | 0 | 3.75 | 631 |

Two things these ranges do **not** capture, and both matter more than the numbers
above. A chemistry can sit inside every single range and still belong to an alloy
family the model has never seen -- on four of twelve held-out families it did
worse than guessing the training median. And an element left blank is treated as
"not deliberately added", i.e. zero, which is right for a residual and wrong if
you simply did not measure it.

## What makes this different from the usual version of this project

**1. The ceiling is measured, not hand-waved.** For steels, heat treatment
dominates mechanical properties -- the same chemistry can differ by 500+ MPa in
yield strength depending on tempering. Most composition-only models mention this
in a limitations paragraph. This one quantifies it: rows sharing an exact
composition differ *only* by processing, so the spread within a composition group
is a hard lower bound on any composition-only model's MAE. The model is then
reported against that floor, which turns "composition-only is a limitation" into
"composition-only is at its measured ceiling, and here is the number."

**2. The obvious way to compute that floor is wrong, and the code says why.**
55% of the training data is *specification minima*, not measurements. Those are
near-deterministic per grade, and pooling them in roughly **halves** the apparent
floor -- flattering the model by an artefact of how the benchmark was assembled.
The distinction is a first-class column and a test asserts that pooling deflates
the number, so the reason cannot be quietly lost.

**3. UTS > YS holds by construction, not by luck.** Landmarks are predicted on a
transformed parameterisation (`UTS = exp(z)`, `YS = UTS * sigmoid(z')`) from which
the inequality follows algebraically. A test asserts it over 10,000 sampled
compositions *and* at encoded values where `sigmoid` saturates to exactly 1.0 in
float64 -- the case that would silently turn a strict inequality into an equality.

**4. The leak is published rather than avoided.** Under a composition-only feature
set, two rows of the same grade at different tempers are *literally the same
feature vector with different targets*. A random K-fold puts a row's twin in
training. Both protocols are reported, the random one labelled as a leaky upper
bound, and the gap between them is one of the results.

**5. The negative results are reported too** -- whether the hand-built
metallurgical features actually help, and per model.

## Install and run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows;  source .venv/bin/activate on Linux
python -m pip install -e ".[dev]"

# fetch the datasets (not committed -- DOIs, licences, checksums in data/README.md)
curl -L -o data/raw/steelbench_core_open.csv \
  https://zenodo.org/api/records/18530558/files/steelbench_core_open.csv/content
curl -L -o data/raw/mendeley_jmwb9ddd43.xlsx \
  https://data.mendeley.com/public-files/datasets/jmwb9ddd43/files/cc944ab8-d3c8-448c-ac1d-172e8cc7e11d/file_downloaded
curl -L -o data/raw/steel_strength.json.gz https://ndownloader.figshare.com/files/13354691

python -m stresspredict.ingest --source steelbench    # SteelBench-only: the evaluation below
python -m stresspredict.ingest --source merged --out-dir data/processed_merged
python -m pytest -q
python -m stresspredict.evaluate --protocol all       # 4 protocols x 5 models x 3 targets
python -m stresspredict.report                        # -> reports/results.md
python -m stresspredict.plots                         # -> reports/figures/
python -m stresspredict.train                         # merged pool -> artifacts/
python -m stresspredict.external                      # held-out steel_strength score
python -m stresspredict.predict --composition "C=0.40,Mn=0.80,Cr=1.00"
```

`--protocol all` runs a nested hyper-parameter search inside every outer fold and
takes a while. The four protocols are independent, so running them as separate
processes (`--protocol gkf_grade --tag gkf_grade`, etc.) gives identical numbers
in a quarter of the wall time. Add `--fast` to skip tuning while iterating -- for
iteration only, never for a reported number.

## How it works

```
data/raw/*.csv
  -> ingest.py     one loader per source -> canonical schema, labels only
  -> clean.py      validity rules; every drop and null counted in a manifest
  -> data/processed/{samples,composition_long}.csv + data_manifest.json
  -> features.py   9 elements + 8 metallurgical terms + missingness indicators
  -> targets.py    log/logit parameterisation; UTS > YS by construction
  -> splits.py     4 leakage-aware protocols; reserved calibration holdout
  -> models.py     dummy -> ridge -> RF -> extra trees -> HistGBM
  -> evaluate.py   nested tuning inside each outer fold -> reports/*.json
  -> train.py      final fit on the merged pool + sidecar with every source's SHA-256
  -> external.py   score on the held-out steel_strength set
  -> predict.py    composition dict -> landmarks + range guard
```

### Design decisions worth defending

**Composition is stored long** (`sample_id, element, wt_pct`) and projected to a
dense matrix inside `features`. Nine hard-coded columns would cost nothing today
and force a rewrite the moment a titanium source arrives reporting Fe, Al and V.

**No matminer / Magpie.** Magpie descriptors are composition-weighted statistics
built for chemically diverse search spaces. Steels are 95-99% Fe with a handful of
elements in the 0-5 wt% band, so those statistics are dominated by iron, nearly
constant across the dataset, and actively blur the signal that matters -- the
difference between 0.40 and 0.45 wt% C, which is just the raw number. ~130
descriptors on ~1.3k rows would also turn this into a feature-selection exercise
instead of an evaluation-rigour one, for ~100 MB of transitive dependencies.
Eight hand-built metallurgical terms (carbon equivalent, Pcm, Schaeffler Cr/Ni
equivalents, ...) are used instead -- and whether they help is then *measured*
per model rather than assumed.

**The ladder stops at HistGradientBoosting.** No XGBoost, and the claim is backed
with a number: on ~1.3k rows the HGB-vs-XGBoost gap is smaller than the
fold-to-fold standard error of the grouped CV, which is reported for every result.

**Missing is not zero.** A blank element means "not deliberately added", so it is
filled with 0.0; a blank temperature means "not reported", so it is filled with
the training median -- 0 C is not a plausible austenitising temperature. Both
carry an explicit missingness indicator. HistGradientBoosting could take NaN
natively, but letting it see different data from ridge would invalidate the ladder
comparison, which is the point of having a ladder.

**Nothing is clipped.** A clipped transcription error is an invented measurement
that looks exactly like a real one. Implausible values are nulled or dropped, and
counted.

## Reading the numbers

* **MAE is primary**, and that is a consequence of the model, not a preference:
  the targets are modelled through a log link, so inverting a prediction recovers
  the conditional *median* -- the quantity MAE scores.
* **`gkf_grade` is the headline.** `random_kfold` is reported only as a labelled
  leaky upper bound.
* **For LOFO, read the worst family**, not the mean. Pooling 12 uneven folds lets
  one catastrophic family hide behind eleven good ones.

## Repository layout

```
stresspredict/     the package (schema, units, ingest, clean, features, targets,
                   grouping, splits, models, metrics, noise_floor, evaluate,
                   train, predict, plots, report, export_web)
tests/             127 tests; integration tests skip cleanly without the dataset
web/               the test bench: app.html, the exported model, and
                   build_site.sh, which wraps it for GitHub Pages
tools/uitest/      drives the test bench in a real browser (puppeteer)
data/README.md     DOIs, licences, checksums, and what the data actually is
docs/              model card, evaluation methodology, provenance, limitations
reports/           committed results JSON, generated results.md, figures/
```

The test bench deploys to GitHub Pages on every push that touches `web/`
(`.github/workflows/pages.yml`). To preview the exact deployed page locally, run
`sh web/build_site.sh _site` and open `_site/index.html`.

## Known limits

Composition-only against a measured physical ceiling; **strain at UTS and the
hardening exponent are not predicted**, because no open dataset found reports
them alongside composition -- which is the real gate on reconstructing a
continuous curve; no calibrated uncertainty in v1; elongation is trained on 661
rows with an unknown mix of gauge-length standards. The full list, including what
the training data actually is, is in [docs/limitations.md](docs/limitations.md).

Not for design allowables or safety-critical use.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Data foundation: ingest, clean, manifest, noise floor | done |
| 1 | v1 model: features, targets, splits, ladder, evaluation, predict | done |
| 1b | matbench_steels / steel_strength yardstick + contamination check | done -- the held-out promotion set |
| 2 | Mendeley zero-shot test, then merged training; processing-input model measured | done; processing model not shipped |
| 3 | Split-conformal intervals + k-NN applicability domain | calibration holdout already reserved |
| 4 | Curve reconstruction | **gated on data** -- no open source reports strain at UTS |
| 5 | App / API | interactive test bench shipped ([live](https://daviddruker.github.io/stress-strain-predictor/)); API not before Phase 3 |
| 6 | Al / Ti / Cu / Ni | new loaders only, if the schema held |

## Licence and attribution

Code: MIT. Data: CC BY 4.0, licensed separately by its publishers -- see
[data/README.md](data/README.md). Training data is SteelBench v1.0
(DOI [10.5281/zenodo.18530558](https://doi.org/10.5281/zenodo.18530558), CC BY 4.0) and
*A database of mechanical properties of steels*, Ghorbani, Zhao and Birbilis
(DOI [10.17632/jmwb9ddd43.1](https://doi.org/10.17632/jmwb9ddd43.1), CC BY 4.0), with
cross-source duplicates removed. The held-out set is matminer `steel_strength`, from
Citrine dataset 153092 (figshare 10.6084/m9.figshare.7250453, MIT).
NIMS MatNavi was deliberately not scraped; its terms prohibit bulk acquisition.
