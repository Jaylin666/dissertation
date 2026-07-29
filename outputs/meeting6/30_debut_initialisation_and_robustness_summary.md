# Meeting 6 Step 3: Debut Initialisation and Robustness Diagnostics

## Purpose

This diagnostic step investigates why validation-best Elo outperformed Glicko low-inflation in debut-player matches and whether the broader Glicko advantage is robust after excluding debut cases.

## Inputs and validation

- Step 29 per-match scores: `outputs\meeting6\29_per_match_model_scores_2025.csv`
- Step 28 match features: `outputs\meeting6\28_prematch_match_features_2025.csv`
- Validation checks passed: 16 / 16

## Model settings read from code/outputs

- Elo initial rating: 1500.000
- Glicko initial rating: 1500.000
- Glicko initial RD: 350.000
- Glicko MIN_RD / MAX_RD: 30.000 / 350.000
- Low-inflation C: 22.509257 over target periods 240
- Glicko probability formula: 1 / (1 + 10 ** (-g(RD_opponent) * (rating_player - rating_opponent) / 400))
- Elo probability formula: 1 / (1 + 10 ** ((rating_opponent - rating_player) / scale))

## The unexpected debut-player result

- No debut games: 11305
- Exactly one debut games: 72
- Both players debut games: 2
- Exactly-one-debut empirical debut win rate: 0.402778
- Glicko mean predicted debut win probability: 0.779479
- Validation-best Elo mean predicted debut win probability: 0.540677
- Mean Glicko minus Elo debut probability: 0.238802

## Initial rating and 2025 rating-scale compatibility

- Glicko initial rating minus 2025 established-player Glicko median: -12.738825
- Elo initial rating minus 2025 established-player Elo median: -172.622312
- This is a mechanism diagnostic only; it does not imply retuning the initial rating on the 2025 test set.

## Is the debut result driven by a few games or events?

- Full debut Brier difference: -0.112391
- Leave-one-event-out min/max: -0.120364 / -0.106566
- Sign changes after leaving one event out: 0

## Recent activity zero: debut versus genuine returners

- Debut / no previous history: games=74, Brier diff=-0.109354, inflation contribution=0.026395.
- Non-debut, inactive 365-729 days: games=113, Brier diff=-0.001821, inflation contribution=0.019820.
- Non-debut, inactive 730-1095 days: games=34, Brier diff=0.011877, inflation contribution=0.047405.
- Non-debut, inactive 1096+ days: games=49, Brier diff=0.000568, inflation contribution=0.041564.
- Missing date information: games=73, Brier diff=0.001320, inflation contribution=0.012154.

## RD inflation for genuine returning players

- Returning >=365 days, no debut: games=198, Glicko-vs-Elo Brier diff=0.001222, inflation delta Brier=0.028615.

## Overall robustness after excluding debut

- All games: games=11379, Brier diff=0.002349, CI [0.000870, 0.003853].
- Excluding all debut games: games=11305, Brier diff=0.003080, CI [0.001626, 0.004563].
- Both players have history: games=11305, Brier diff=0.003080, CI [0.001626, 0.004563].
- Both active in last 365 days and no debut: games=11036, Brier diff=0.003126, CI [0.001749, 0.004529].

## No-debut RD analysis

- quartile_1: games=2827, mean max RD=64.362399, Brier diff=0.002565.
- quartile_2: games=2826, mean max RD=79.613811, Brier diff=0.002525.
- quartile_3: games=2826, mean max RD=96.275744, Brier diff=0.004012.
- quartile_4: games=2826, mean max RD=154.398843, Brier diff=0.003218.

## Prediction confidence mechanism

- Glicko substantially less confident: games=1390, mean confidence diff=-0.086219, Brier diff=0.008344.
- Glicko slightly less confident: games=2849, mean confidence diff=-0.026730, Brier diff=0.002540.
- Similar confidence: games=2322, mean confidence diff=-0.000412, Brier diff=-0.000163.
- Glicko slightly more confident: games=2632, mean confidence diff=0.027189, Brier diff=0.001613.
- Glicko substantially more confident: games=2186, mean confidence diff=0.113609, Brier diff=0.001840.

## Brier reliability and resolution diagnostics

- This is an approximate favourite-perspective Brier decomposition based on fixed probability bins; it is sensitive to bin choice and does not replace the raw Brier score.
- Glicko low inflation: reliability=0.000298, resolution=0.017426, actual Brier=0.187724.
- Validation-best Elo: reliability=0.000305, resolution=0.017984, actual Brier=0.190073.

## Main robust findings

- Overall: games=11379, Brier diff=0.002349, CI [0.000870, 0.003853], flag=robust_glicko_advantage.
- Overall excluding debut: games=11305, Brier diff=0.003080, CI [0.001626, 0.004563], flag=robust_glicko_advantage.
- Exactly one debut: games=72, Brier diff=-0.112391, CI [-0.160264, -0.064860], flag=initialisation_mismatch.
- Both active and no debut: games=11036, Brier diff=0.003126, CI [0.001749, 0.004529], flag=robust_glicko_advantage.
- No-debut lowest RD quartile: games=2827, Brier diff=0.002565, CI [0.000872, 0.004437], flag=robust_glicko_advantage.

## Findings that remain uncertain

- Both players debut: games=2, Brier diff=0.000000, CI [0.000000, 0.000000], flag=small_sample.
- New but not debut: 1-5 games: games=391, Brier diff=0.007181, CI [-0.014131, 0.029752], flag=uncertain.
- Low experience but not debut: 1-20 games: games=1447, Brier diff=0.004531, CI [-0.004238, 0.013600], flag=uncertain.
- Non-debut zero recent games: games=196, Brier diff=0.001153, CI [-0.012945, 0.015465], flag=uncertain.
- Returning >=365, no debut: games=198, Brier diff=0.001222, CI [-0.014132, 0.015455], flag=uncertain.
- Returning >=730, no debut: games=83, Brier diff=0.005201, CI [-0.011886, 0.021890], flag=uncertain.

## Implications for Meeting 6

- The debut result should be reported directly: validation-best Elo is much better for exactly-one-debut matches in the current fixed 2025 test.
- The broader Glicko advantage remains after excluding debut matches.
- RD inflation improves Glicko C0 for long-inactivity cases, but this is not the same as proving Glicko low beats validation-best Elo in every returning-player subgroup.
- The calibration/Brier tension should be discussed as a reliability-resolution-confidence trade-off rather than a single accuracy claim.

## Limitations

- These are exploratory mechanism diagnostics, not causal proof.
- Some returning and debut subgroups are small.
- Historical rating-scale drift is not rerun here; the rating-scale analysis is a 2025 cross-section diagnostic.

## Files written

- `outputs\meeting6\30_input_validation_checks.csv`
- `outputs\meeting6\30_history_category_counts.csv`
- `outputs\meeting6\30_debut_player_perspective.csv`
- `outputs\meeting6\30_debut_model_summary.csv`
- `outputs\meeting6\30_initialisation_rating_scale_diagnostics.csv`
- `outputs\meeting6\30_2025_rating_distribution_summary.csv`
- `outputs\meeting6\30_top_influential_debut_matches.csv`
- `outputs\meeting6\30_debut_event_contributions.csv`
- `outputs\meeting6\30_debut_leave_one_event_out.csv`
- `outputs\meeting6\30_debut_leave_one_event_out_summary.csv`
- `outputs\meeting6\30_zero_recent_activity_decomposition.csv`
- `outputs\meeting6\30_returning_player_threshold_sensitivity.csv`
- `outputs\meeting6\30_overall_exclusion_robustness.csv`
- `outputs\meeting6\30_no_debut_subgroup_results.csv`
- `outputs\meeting6\30_no_debut_rd_quartile_results.csv`
- `outputs\meeting6\30_no_debut_rd_decile_results.csv`
- `outputs\meeting6\30_no_debut_rd_cutpoints.csv`
- `outputs\meeting6\30_prediction_confidence_diagnostics.csv`
- `outputs\meeting6\30_model_favourite_disagreement.csv`
- `outputs\meeting6\30_brier_decomposition_summary.csv`
- `outputs\meeting6\30_brier_decomposition_bins.csv`
- `outputs\meeting6\30_total_games_threshold_sensitivity.csv`
- `outputs\meeting6\30_recent_activity_threshold_sensitivity.csv`
- `outputs\meeting6\30_key_diagnostic_results.csv`
- `outputs\meeting6\30_diagnostic_validation_checks.csv`
- `outputs\meeting6\30_debut_initialisation_and_robustness_summary.md`
- `outputs\meeting6\figures\30_fig01_debut_probability_vs_actual.png`
- `outputs\meeting6\figures\30_fig02_debut_prediction_distribution.png`
- `outputs\meeting6\figures\30_fig03_initial_rating_population_position.png`
- `outputs\meeting6\figures\30_fig04_zero_recent_activity_decomposition.png`
- `outputs\meeting6\figures\30_fig05_returner_inflation_contribution.png`
- `outputs\meeting6\figures\30_fig06_overall_exclusion_robustness.png`
- `outputs\meeting6\figures\30_fig07_no_debut_rd_quartiles.png`
- `outputs\meeting6\figures\30_fig08_prediction_confidence_mechanism.png`
- `outputs\meeting6\figures\30_fig09_debut_leave_one_event_out.png`