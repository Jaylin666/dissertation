# Meeting 6 Step 33: Orientation-Corrected Final Results

## Purpose

This step recomputes all Meeting 6 probability-based results after the Step 32 Glicko probability-orientation audit.
It does not rerun Elo, Glicko, or adaptive-K ratings, and it does not reselect parameters.

## Probability Orientation Fix

The final Glicko probability is now the outcome-independent direct player-A probability:

`p_a_Glicko_low_fixed = expected_score(rating_A, rating_B, RD_B)`.

The Glicko formula itself is unchanged. The correction is only in evaluation orientation: the script no longer converts actual-winner probability to player-A probability using the match outcome.
The diagnostic symmetric probability remains a sensitivity check, not the standard Glicko probability.

## Overall Model Comparison

- Glicko low inflation: Brier=0.187604, log loss=0.551779, accuracy=0.711486.
- Validation-best Elo: Brier=0.190073, log loss=0.556534, accuracy=0.704456.
- Best adaptive-K Elo: Brier=0.190781, log loss=0.559185, accuracy=0.706213.
- Glicko C0: Brier=0.195708, log loss=0.571958, accuracy=0.693734.

Main paired Brier improvement, Elo minus Glicko low: 0.002469 [0.001045, 0.003911].
Main paired log-loss improvement, Elo minus Glicko low: 0.004756 [0.001305, 0.008307].

## What Changed From Step 29-31

- Old saved-winner orientation Glicko low Brier: 0.187724.
- Fixed direct player-A Glicko low Brier: 0.187604.
- The direction of the main conclusion is stable: Glicko low remains better overall than validation-best Elo.
- Step29-31 probability-based outputs are superseded by Step33 outputs. Rating-level outputs such as the unique-player snapshot and debut opponent rating distribution remain valid.

## Debut and Returning Players

- The debut anomaly remains after correction: validation-best Elo is much better for exactly-one-debut matches.
- The direct cause is not that the debut player's own RD=350 directly enters the current match expected-score formula. The direct expected score uses the opponent RD; the large debut probability mainly comes from the initial rating being high relative to many opponents.
- Returning/inactive-player estimates remain small-sample and uncertain for Glicko-vs-Elo, but low inactivity RD inflation still improves Glicko relative to C0.
- Overall inflation contribution, C0 Brier minus low-inflation Brier: 0.008104 [0.006260, 0.009990].

## Adaptive-K Comparison

- Overall adaptive-K Brier recovery ratio: 0.515002 (valid=True).
- Overall adaptive-K log-loss recovery ratio: 0.532855 (valid=True).

## Main Conclusions for Meeting 6

- Overall: games=11379, Brier diff=0.002469 [0.001045, 0.003911], log-loss diff=0.004756 [0.001305, 0.008307]; Glicko advantage with CI above zero.
- Overall excluding debut: games=11305, Brier diff=0.003236 [0.001839, 0.004707], log-loss diff=0.006752 [0.003441, 0.010193]; Glicko advantage with CI above zero.
- Exactly one debut: games=72, Brier diff=-0.118005 [-0.162170, -0.072543], log-loss diff=-0.308522 [-0.434018, -0.182291]; Elo better in debut diagnostic group.
- Total previous games <=5: games=465, Brier diff=-0.012687 [-0.031038, 0.006999], log-loss diff=-0.041129 [-0.090712, 0.011082]; uncertain; CI crosses zero.
- Recent games <=5: games=1635, Brier diff=-0.002162 [-0.008666, 0.004647], log-loss diff=-0.009704 [-0.027027, 0.008269]; uncertain; CI crosses zero.
- Inactive >=365 days, no debut: games=198, Brier diff=0.001258 [-0.013756, 0.015322], log-loss diff=-0.010658 [-0.057625, 0.029650]; uncertain; CI crosses zero.
- Inactive >=730 days, no debut: games=83, Brier diff=0.004710 [-0.012578, 0.022441], log-loss diff=0.005053 [-0.039205, 0.048548]; uncertain; CI crosses zero.
- Both players active last 365 days: games=11036, Brier diff=0.003289 [0.001974, 0.004623], log-loss diff=0.007103 [0.003818, 0.010469]; Glicko advantage with CI above zero.

## Validation

- Validation checks passed: 32 / 32.
- The fixed Glicko low Brier matches the Step 32 audit target of approximately 0.187604.
- Missing date information and debut groups were retained.
- All new outputs use the `33_` prefix.

## Files Written

- `outputs\meeting6\33_canonical_player_orientation_checks.csv`
- `outputs\meeting6\33_glicko_probability_reconstruction_checks.csv`
- `outputs\meeting6\33_orientation_corrected_per_match_scores_2025.csv`
- `outputs\meeting6\33_overall_model_metrics.csv`
- `outputs\meeting6\33_overall_pairwise_comparisons.csv`
- `outputs\meeting6\33_overall_bootstrap_confidence_intervals.csv`
- `outputs\meeting6\33_adaptive_k_improvement_recovered.csv`
- `outputs\meeting6\33_subgroup_model_performance_long.csv`
- `outputs\meeting6\33_subgroup_pairwise_comparisons.csv`
- `outputs\meeting6\33_subgroup_bootstrap_confidence_intervals.csv`
- `outputs\meeting6\33_debut_corrected_model_summary.csv`
- `outputs\meeting6\33_debut_corrected_player_perspective.csv`
- `outputs\meeting6\33_returning_player_corrected_results.csv`
- `outputs\meeting6\33_returning_exclusive_bins.csv`
- `outputs\meeting6\33_overall_exclusion_robustness.csv`
- `outputs\meeting6\33_standard_calibration_summary.csv`
- `outputs\meeting6\33_standard_calibration_bins.csv`
- `outputs\meeting6\33_standard_brier_decomposition_summary.csv`
- `outputs\meeting6\33_standard_brier_decomposition_bins.csv`
- `outputs\meeting6\33_brier_decomposition_bootstrap.csv`
- `outputs\meeting6\33_orientation_sensitivity_metrics.csv`
- `outputs\meeting6\33_orientation_sensitivity_bootstrap.csv`
- `outputs\meeting6\33_meeting6_final_results.csv`
- `outputs\meeting6\33_supersession_map.csv`
- `outputs\meeting6\33_final_validation_checks.csv`
- `outputs\meeting6\figures\33_fig01_overall_brier_zoomed.png`
- `outputs\meeting6\figures\33_fig02_exclusion_robustness_delta_brier.png`
- `outputs\meeting6\figures\33_fig03_debut_probability_vs_actual.png`
- `outputs\meeting6\figures\33_fig04_zero_activity_debut_decomposition.png`
- `outputs\meeting6\figures\33_fig05_returner_inflation_gain.png`
- `outputs\meeting6\figures\33_fig06_no_debut_rd_quartiles.png`
- `outputs\meeting6\figures\33_fig07_standard_player_a_calibration.png`
- `outputs\meeting6\figures\33_fig08_orientation_sensitivity.png`
- `outputs\meeting6\figures\33_fig09_prediction_confidence_mechanism.png`
- `outputs\meeting6\figures\33_fig10_debut_opponent_rating_distribution.png`
