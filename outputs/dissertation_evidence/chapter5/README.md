# Chapter 5 Controlled Dissertation Evidence Package (local)

This local package was assembled from validated files on GitHub `Jaylin666/dissertation`, branch `main`, with HEAD checked as commit `10d04d16607557787ae626b2d2a338a4767c875d` (`Add Chapter 4 dissertation evidence package`).

No Elo/Glicko model has been rerun. No parameter search has been repeated. All CSVs in this folder are compact extractions/reformats of already-saved frozen values. The PNG files are local replots from those frozen values and are **not** claimed to be byte-identical copies of the original GitHub figures; the exact original source paths are recorded in `figure_manifest.csv` and `evidence_manifest.csv`.

## Evidence hierarchy

- Step 34: primary source for 2025 early-game performance.
- Steps 35-37: supporting mechanism evidence.
- Step 42: primary source for recorded-entry definitions, held-out 2025 entrants, and prematch scale alignment.
- Step 41: broader end-of-year/historical context only; appendix/background rather than the primary Chapter 5 scale comparison.
- Steps 38-40: outside the Chapter 5 core argument unless needed for explicit provenance/robustness.

## Section map

### 5.2 Defining recorded entrants
Use `entry_cohort_definitions_core.csv`.

Important wording:
- say **first recorded appearance**, not true career debut;
- 1985 model-start players are left-censored;
- 1990-2024 are post-burn-in recorded entrants under the primary 5-year definition;
- 2025 contains 76 held-out recorded entrants.

### 5.3 Early-game predictive performance
Use:
- `early_game_cumulative_core.csv`;
- `early_game_stage_core.csv`;
- `figures/stage_brier_replot.png`.

Headline pattern:
- first appearance: Elo Brier 0.210522 vs low-inflation Glicko 0.322316;
- first 20 appearances: 0.212656 vs 0.216538;
- non-overlapping stage results show the strongest weakness is concentrated at appearance 1.

### 5.4 First-appearance limitation
Use:
- `first_appearance_mechanism_core.csv`;
- `figures/predicted_vs_empirical_replot.png`.

Held-out 2025 first appearances:
- 76 appearances / 76 players / 74 unique games;
- Glicko mean focal probability 0.743448;
- empirical focal win rate 0.407895;
- Glicko prediction bias +0.335554;
- Elo mean focal probability 0.538536;
- Elo prediction bias +0.130642.

### 5.5.1 Initialisation at entry
Use Step 36 values recorded in `prematch_scale_alignment_2025_core.csv`:
- entrant rating = 1500;
- mean actual opponent rating = 1180.755;
- mean entrant-minus-opponent gap = +319.245.

Interpret this as a **relative rating-location problem**, not simply “1500 is absolutely too high”.

### 5.5.2 Why a common initial-rating shift is insufficient
Use `initial_rating_invariance_core.csv`.

Candidates 1000, 1100, 1200, 1300, 1400 and 1500 have identical saved validation metrics. A common additive shift changes the scale origin but not prediction-relevant differences.

### 5.5.3 Prematch contemporaneous scale alignment
Use:
- `prematch_scale_alignment_2025_core.csv`;
- `figures/prematch_scale_alignment_2025_replot.png`.

Primary 2025 values:
- fixed entry anchor = 1500;
- mean contemporaneous established rating = 1258.683;
- **median contemporaneous established rating = 1237.367**;
- initial minus contemporaneous median = **+262.659**;
- secondary 365-day active-established median = 1314.848;
- initial minus active median = +185.152.

The Step 42 prematch median 1237.367 is the Chapter 5 primary scale comparison. Do **not** substitute the Step 41 2025 end-of-year established-active median 1334.544.

Allowed interpretation: the evidence supports **relative initialisation mismatch / prematch scale misalignment**.
Prohibited interpretation: “rating-scale drift alone causally creates the prediction error”.

### 5.6 Robustness
Use:
- `early_game_event_cluster_ci_core.csv`;
- `burnin_sensitivity_core.csv`;
- `orientation_sensitivity_2025_core.csv`.

For dissertation-wide bootstrap consistency, prefer the game-level **event-cluster bootstrap** table when reporting confidence intervals. Do not attach its confidence interval to an appearance-level point estimate.

First-appearance game-level Brier difference:
- Elo minus Glicko = -0.114815;
- 95% event-cluster CI [-0.158209, -0.069123].

By first 10 / first 20, the event-cluster Brier intervals include zero.

Burn-in definitions of 1, 3, 5 and 10 years all retain all 76 held-out 2025 entrants.

The direct-focal orientation sensitivity gives a larger numerical overprediction, but the qualitative direction agrees. The current Meeting-7 convention remains the formal reporting source.

## Reporting rules

1. Use **games**, not matches, in dissertation prose/table labels.
2. Use **first recorded appearance / recorded entrant**, not true career debut.
3. Keep the 2025 held-out entrant evidence distinct from 1990-2024 in-sample descriptive mechanism evidence.
4. Brier score and log loss remain the primary predictive metrics; accuracy is supplementary.
5. Do not claim universal Glicko failure: the limitation is concentrated at entry/very early appearances.
6. Do not claim that a common initial rating of 1500 is itself the sole problem; common shifts are invariant.
7. Do not present Step 41 end-of-year scale statistics as Step 42 prematch entrant-scale statistics.
8. Do not mix player-cluster appearance-level bootstrap CIs with event-cluster game-level point estimates.
9. Do not use “statistically significant”, “proves”, “universally superior”, or “globally optimal”.
10. Structured new-player initialisation is future work, not an experiment required for the current thesis.

## Integrity note

`evidence_manifest.csv` records GitHub source paths, source Git blob identifiers where available, evidence role, section assignment, analysis unit, probability orientation/bootstrap convention, and local SHA-256 hashes.

This package is intended to be the local controlled source for drafting Chapter 5, not a replacement for the repository's original archived outputs.
