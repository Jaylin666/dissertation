# Meeting 6 Final Analysis Summary

## Purpose

This step finalises the Meeting 6 diagnostics after correcting the rating-distribution and Brier-decomposition methodology.

## Methodological Corrections

- The 22,682 rating rows are match-weighted player-match observations, not 22,682 unique players.
- A separate unique-player first-2025-appearance snapshot is now reported.
- The main Murphy decomposition now uses the common `outcome_a` and each model's `p_a_model`, so uncertainty is shared within each sample.
- The Glicko debut mechanism is described through initial rating, opponent rating, and RD orientation. In the direct debut-player expected score, the debut player's own RD=350 is not the opponent RD; however, the saved meeting5 probabilities are winner-perspective, so lost-debut cases are reconstructed as the complement of the experienced opponent's expected score.

## Validated Overall Comparison

- Overall Glicko low vs validation-best Elo: Brier difference 0.002349, CI [0.000902, 0.003789].
- Excluding debut: Brier difference 0.003080, CI [0.001611, 0.004528].

## Debut-Player Initialisation Mismatch

- Exactly-one-debut games: 72.
- Debut subgroup Brier difference: -0.112391, CI [-0.159484, -0.068208].
- Mean Glicko low debut rating difference: 336.980818.
- Mean Glicko low saved-orientation reconstructed debut probability: 0.779479.
- Mean Glicko low direct debut-perspective probability: 0.804099.
- Mean Elo reconstructed debut probability: 0.540677.

## Unique-Player Rating Snapshot and Debut-Opponent Ratings

- Glicko unique established players: 1046; initial rating percentile: 68.068834; initial minus median: 165.509099.
- Elo unique established players: 1046; initial rating percentile: 35.468451; initial minus median: -48.141777.
- Glicko debut-opponent initial percentile: 88.888889; initial minus opponent median: 373.951329.
- Elo debut-opponent initial percentile: 69.444444; initial minus opponent median: 42.055720.

## Overall Robustness After Excluding Debut

- The Glicko advantage strengthens from 0.002349 overall to 0.003080 after excluding debut matches.
- Both active and no debut: games=11036, delta Brier=0.003126, CI [0.001767, 0.004538].

## New Players Versus Returning Players

- The no-recent-activity result should be separated into debut/no-history cases, genuine returners, and missing-date records.
- Returning >=365 days, no debut: games=198, Glicko-vs-Elo delta Brier=0.001222.

## Contribution of Inactivity RD Inflation

- For cumulative returning >=365 days, Glicko C0 Brier - Glicko low Brier = 0.028615, CI [0.015852, 0.042308].
- The point estimates support RD inflation for long-inactivity samples, but these nested threshold samples are not mutually exclusive.

## No-Debut RD Results

- RD quartile 3: delta Brier=0.004012, CI [0.002076, 0.005864].
- RD quartile 4: delta Brier=0.003218, CI [-0.001681, 0.007933].
- The no-debut RD pattern is not monotonic, so it should not be presented as 'higher RD always means larger Glicko advantage'.

## Prediction Confidence

- Step 30 confidence diagnostics remain useful as mechanism diagnostics, but the final calibration/decomposition figures now use the common player-A outcome.

## Standard Brier Decomposition

- Overall common uncertainty: 0.248743.
- Glicko low overall reliability/resolution: 0.000342 / 0.060975.
- Elo overall reliability/resolution: 0.000439 / 0.058695.
- Bootstrap delta reliability (Elo - Glicko): 0.000097, CI [-0.000373, 0.000590].
- Bootstrap delta resolution (Glicko - Elo): 0.002279, CI [0.000890, 0.003719].

## Main Conclusions for Meeting 6

- Overall: games=11379, delta Brier=0.002349, CI [0.000902, 0.003789], message=Glicko low inflation is better overall than validation-best Elo on paired Brier and log loss.
- Overall excluding debut: games=11305, delta Brier=0.003080, CI [0.001611, 0.004528], message=After removing debut matches, the overall Glicko advantage becomes larger.
- Exactly one debut: games=72, delta Brier=-0.112391, CI [-0.159484, -0.068208], message=Validation-best Elo is clearly better in the debut subgroup; this is a stable initialisation mismatch diagnostic.
- New but not debut: 1-5 games: games=391, delta Brier=0.007181, CI [-0.015966, 0.028820], message=Exploratory subgroup result; use effect size and CI rather than a categorical claim.
- Total previous games 21-50: games=1258, delta Brier=0.005597, CI [0.001686, 0.009385], message=Exploratory subgroup result; use effect size and CI rather than a categorical claim.
- Recent activity 6-15: games=2649, delta Brier=0.004005, CI [0.001106, 0.007071], message=Exploratory subgroup result; use effect size and CI rather than a categorical claim.
- Returning >=365 days, no debut: games=198, delta Brier=0.001222, CI [-0.014325, 0.015761], message=RD inflation improves Glicko C0 for returners, but Glicko-vs-Elo evidence remains uncertain in this small subgroup.
- Returning >=730 days, no debut: games=83, delta Brier=0.005201, CI [-0.011717, 0.021606], message=RD inflation improves Glicko C0 for returners, but Glicko-vs-Elo evidence remains uncertain in this small subgroup.

## Findings That Remain Uncertain

- Glicko versus Elo within returning-player subgroups remains uncertain because confidence intervals are wide.
- New-but-not-debut low-experience groups have positive point estimates for Glicko, but the intervals can cross zero.
- RD quartile results are exploratory and should not be interpreted as a monotonic law.

## Recommended Meeting Figures

- M5/6-1: Overall model Brier score (`outputs/meeting6/figures/29_fig01_overall_brier_zoomed.png`).
- 30-1: Debut predicted probability versus actual win rate (`outputs/meeting6/figures/30_fig01_debut_probability_vs_actual.png`).
- 31-2: Initial rating relative to opponents faced by debut players (`outputs\meeting6\figures\31_fig02_debut_opponent_rating_distribution.png`).
- 31-3: Standard player-A calibration overall (`outputs\meeting6\figures\31_fig03_standard_calibration_overall.png`).
- 31-6: No recent activity, debut and missing-date decomposition (`outputs\meeting6\figures\31_fig06_zero_activity_decomposition_with_ci.png`).
- 31-7: Returner inflation contribution with CIs (`outputs\meeting6\figures\31_fig07_returner_inflation_with_ci.png`).
- 31-9: Overall exclusion robustness with CIs (`outputs\meeting6\figures\31_fig09_overall_exclusion_robustness_with_ci.png`).

## Appendix Figures

- 31-1: Initial rating vs unique established-player rating snapshot (`outputs\meeting6\figures\31_fig01_unique_player_rating_snapshot.png`).
- 31-4: Standard player-A calibration excluding debut (`outputs\meeting6\figures\31_fig04_standard_calibration_no_debut.png`).
- 31-5: Standard player-A calibration for debut matches (`outputs\meeting6\figures\31_fig05_standard_calibration_debut.png`).
- 31-8: Returner inflation contribution in exclusive bins (`outputs\meeting6\figures\31_fig08_returner_inflation_exclusive_bins.png`).
- 31-10: No-debut RD quartiles with CIs (`outputs\meeting6\figures\31_fig10_no_debut_rd_quartiles_with_ci.png`).

## Limitations

- These are paired predictive diagnostics, not causal proof.
- No model parameter is changed based on 2025 test results.
- Debut-opponent ratings describe the opponents actually faced by debut players, not the full player population.
- Favourite-perspective decomposition from Step 30 is retained only as an appendix diagnostic; the main decomposition is player-A based.

## Validation

- Final validation checks passed: 16 / 16.

## Files Written

- `outputs\meeting6\31_input_validation_checks.csv`
- `outputs\meeting6\31_rating_distribution_summary.csv`
- `outputs\meeting6\31_unique_player_rating_snapshot.csv`
- `outputs\meeting6\31_debut_opponent_rating_summary.csv`
- `outputs\meeting6\31_standard_brier_decomposition_summary.csv`
- `outputs\meeting6\31_standard_brier_decomposition_bins.csv`
- `outputs\meeting6\31_brier_decomposition_bootstrap.csv`
- `outputs\meeting6\31_debut_probability_mechanism.csv`
- `outputs\meeting6\31_debut_probability_mechanism_summary.csv`
- `outputs\meeting6\31_returner_cumulative_inflation_bootstrap.csv`
- `outputs\meeting6\31_returner_exclusive_bins.csv`
- `outputs\meeting6\31_meeting6_final_results.csv`
- `outputs\meeting6\31_meeting6_figure_manifest.csv`
- `outputs\meeting6\31_final_validation_checks.csv`
- `outputs\meeting6\31_meeting6_final_analysis_summary.md`
- `outputs\meeting6\figures\31_fig01_unique_player_rating_snapshot.png`
- `outputs\meeting6\figures\31_fig02_debut_opponent_rating_distribution.png`
- `outputs\meeting6\figures\31_fig03_standard_calibration_overall.png`
- `outputs\meeting6\figures\31_fig04_standard_calibration_no_debut.png`
- `outputs\meeting6\figures\31_fig05_standard_calibration_debut.png`
- `outputs\meeting6\figures\31_fig06_zero_activity_decomposition_with_ci.png`
- `outputs\meeting6\figures\31_fig07_returner_inflation_with_ci.png`
- `outputs\meeting6\figures\31_fig08_returner_inflation_exclusive_bins.png`
- `outputs\meeting6\figures\31_fig09_overall_exclusion_robustness_with_ci.png`
- `outputs\meeting6\figures\31_fig10_no_debut_rd_quartiles_with_ci.png`