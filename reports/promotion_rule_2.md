# Promotion rule 2, fixed before any held-out number for the new candidate (2026-09-23)

Held-out set: matminer `steel_strength`, unchanged. This is its **second** use for a promotion
decision, so it is now slightly "spent": a small win on it is weaker evidence than the first.

Baseline: the currently shipped model (SteelBench + Mendeley, fingerprint 96dd5c9f24bbca58...).

Candidate D: SteelBench + Mendeley + the literature-derived set (figshare 10.6084/m9.figshare.32755830 v2,
CC BY 4.0), same pipeline, inputs, tuning and seed as the shipped model. The literature rows kept are:
- composition in wt% or mass%, with carbon reported;
- a plausible steel (non-Fe total <= 50 wt%, C <= 2.1);
- a test stated as room-temperature tensile, with no hydrogen, corrosion, irradiation, fatigue,
  compression, pre-strain, notch or weld context, and no second non-ambient temperature;
- UTS present.
Ranges become their midpoint; "<x", "Nil", "-" and "Bal." become not reported. Each paper is one grade
group. **Leakage guard:** any literature row within 0.5 wt% (max over C, Mn, Si, Cr, Ni, Mo, V, Al, Co,
Ti) of a held-out steel is removed from training. That is 30 rows, chosen before scoring.

Decision: D replaces the baseline if its mean MAE ratio over {UTS, YS, EL} on the held-out set is
< 1.0, and UTS > YS holds on every prediction.

Reported but not gating: grade-grouped CV of D on its pool, split by source.

## Outcome: not promoted

| Held-out, 312 steels | UTS MAE | YS MAE | EL MAE |
|---|---|---|---|
| Shipped (SteelBench + Mendeley) | 409 | 430 | 4.90 |
| D (+ literature) | 546 | 475 | 4.84 |

Mean MAE ratio 1.137, so the rule fails. The shipped model is unchanged. D predicts lower across the held-out
set (UTS bias -501 vs -290 MPa). A likely cause, untested: the literature adds many ordinary Cr-Ni steels whose
nine visible elements resemble the maraging chemistries, while the Co and Ti that separate them are not inputs.

Grade-grouped 5-fold CV (not gating), UTS / YS / EL MAE by source:

| Source | Shipped | D |
|---|---|---|
| SteelBench | 112 / 112 / 5.19 | 102 / 109 / 5.33 |
| Mendeley | 161 / 166 / 3.97 | 155 / 164 / 4.26 |
| Literature | - | 214 / 213 / 12.7 |

On typical steels, D is slightly better for strength and slightly worse for elongation. The literature rows are
themselves the noisiest source; an elongation error of 12.7 pp reflects mixed gauge lengths and extraction noise.
