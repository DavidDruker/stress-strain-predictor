# Promotion rule, fixed before any held-out number was computed (2026-09-23)

Held-out set: matminer `steel_strength` (Citrine 153092, figshare 13354691,
sha256 e36501d7...f6ad), 312 steels, UTS + YS (+ EL on 303). Zero composition or
property overlap with SteelBench or Mendeley (checked). Never used for training or tuning.

Arms:
- A  shipped model (SteelBench only, 9 elements, composition only)
- B  SteelBench + Mendeley, same inputs as A (drop-in for the web page)
- C  SteelBench + Mendeley, composition + processing (class + temperatures)

On the held-out set there is no processing information, so C is scored with processing = unknown.

Decision: a candidate replaces A if the mean over {UTS, YS, EL} of MAE_candidate / MAE_A on the held-out
set is < 1.0 (user's instruction: "even slightly better"), AND UTS > YS holds on every prediction.
If both B and C qualify, the lower ratio wins; on a tie within 0.01, B wins (no UI change needed).
Deploying C also needs processing inputs on the web page, which is checked with the user first.

Reported but not gating: grade-grouped 5-fold CV on the merged pool, split by source.

## Outcome

Arm B passed (mean MAE ratio 0.605; 0 UTS<=YS violations) and was promoted. Arm C: 0.734.
The repo retrain (`ingest --source merged`, `train`) reproduced arm B bit-for-bit (model fingerprint
`96dd5c9f24bbca58...`). Per-arm held-out scores and per-source grouped CV are in
`promotion_merge_experiment.json`.
