# Meeting 6 Step 2: Where Glicko Helps

## Purpose

This script merges validated pre-match features with existing meeting 5 prediction files to analyse where Glicko low-inflation improves over validation-best Elo.

## Inputs and model alignment

- Prematch features: `outputs\meeting6\28_prematch_match_features_2025.csv`
- Fair Elo-vs-Glicko predictions: `outputs\meeting5_fair_elo_vs_glicko\meeting5_fair_elo_vs_glicko_predictions_2025.csv`
- Adaptive-K predictions: `outputs\meeting5_adaptive_k_elo\meeting5_adaptive_k_elo_predictions_2025.csv`
- Probability orientation: fair and adaptive files store actual-winner probability; this script converts to neutral player-A probability using player IDs.
- All model predictions are merged by `match_id == game_id == fcode`, not by row number.

## Overall model comparison

- Glicko low inflation: Brier=0.187724, log loss=0.552154, accuracy=0.711574.
- Validation-best Elo: Brier=0.190073, log loss=0.556534, accuracy=0.704543.
- Best adaptive-K Elo: Brier=0.190781, log loss=0.559185, accuracy=0.706301.
- Main paired Brier difference, Elo minus Glicko: 0.002349, 95% CI [0.000870, 0.003853].
- Main paired log-loss difference, Elo minus Glicko: 0.004381, 95% CI [0.000681, 0.007965].

## Where Glicko improves over validation-best Elo

- Maximum inactivity gap = 91-180 days: games=205, Brier diff=0.008752, 95% CI [-0.002461, 0.019590].
- Minimum total previous games = 1-5: games=391, Brier diff=0.007181, 95% CI [-0.014932, 0.030115].
- Minimum total previous games = 21-50: games=1258, Brier diff=0.005597, 95% CI [0.001517, 0.009317].
- Minimum games in last 365 days = 6-15: games=2649, Brier diff=0.004005, 95% CI [0.000892, 0.006991].
- Maximum inactivity gap = 181-365 days: games=504, Brier diff=0.003924, 95% CI [-0.003484, 0.011382].
- Glicko pre-match RD quartile = Q3: games=2844, Brier diff=0.003851, 95% CI [0.001949, 0.005823].

## Contribution of inactivity RD inflation

- 731-1095 days: games=34, Glicko C0 Brier - Glicko low Brier = 0.047405.
- 1096+ days: games=49, Glicko C0 Brier - Glicko low Brier = 0.041564.
- No previous history: games=74, Glicko C0 Brier - Glicko low Brier = 0.026395.

## Comparison with adaptive-K Elo

- Overall adaptive-K Brier recovery ratio: 0.524613 (valid=True).
- Overall adaptive-K log-loss recovery ratio: 0.545758 (valid=True).

## New and low-experience players

- Either player debut: games=74, Brier diff=-0.109354, log-loss diff=-0.275036.
- Minimum total games <= 5: games=465, Brier diff=-0.011365, log-loss diff=-0.035631.
- Minimum total games <= 20: games=1521, Brier diff=-0.001010, log-loss diff=-0.007461.

## Low recent activity and returning players

- Minimum games last 365 days <= 5: games=1635, Brier diff=-0.002149, log-loss diff=-0.009375.
- Either player inactive >= 365 days: games=198, Brier diff=0.001222, log-loss diff=-0.009085.
- Either player inactive >= 730 days: games=83, Brier diff=0.005201, log-loss diff=0.010529.

## Prediction confidence

- Elo favourite probability 0.50-0.60: games=3039, Brier diff=0.003120.
- Elo favourite probability 0.60-0.70: games=2785, Brier diff=0.003739.
- Elo favourite probability 0.70-0.80: games=2291, Brier diff=0.002920.
- Elo favourite probability 0.80-0.90: games=1825, Brier diff=0.000555.
- Elo favourite probability 0.90-1.00: games=1439, Brier diff=-0.000605.

## Pre-match RD analysis

- Reliable pre-match Glicko RD was available and RD quartile analysis was generated.

## Bootstrap uncertainty

- Bootstrap type is event-cluster when a subgroup has at least 10 events, with 2,000 replications.
- Small groups are retained and marked with `small_sample_warning`.

## Corrected calibration

- Glicko low inflation corrected calibration error: 0.012837.
- Validation-best Elo corrected calibration error: 0.006958.
- Best adaptive-K Elo corrected calibration error: 0.018188.

## Main conclusions for Meeting 6

- Overall: games=11379, Brier diff=0.002349, log-loss diff=0.004381, 95% Brier CI [0.000870, 0.003853]; favours Glicko, CI excludes 0.
- Either player debut: games=74, Brier diff=-0.109354, log-loss diff=-0.275036, 95% Brier CI [-0.155037, -0.062213]; favours Elo, CI excludes 0.
- Total previous games <= 5: games=465, Brier diff=-0.011365, log-loss diff=-0.035631, 95% Brier CI [-0.030220, 0.008107]; favours Elo, CI includes 0.
- Recent games <= 5: games=1635, Brier diff=-0.002149, log-loss diff=-0.009375, 95% Brier CI [-0.008796, 0.004568]; favours Elo, CI includes 0.
- Inactive >= 365 days: games=198, Brier diff=0.001222, log-loss diff=-0.009085, 95% Brier CI [-0.013241, 0.015922]; favours Glicko, CI includes 0.
- Inactive >= 730 days: games=83, Brier diff=0.005201, log-loss diff=0.010529, 95% Brier CI [-0.012059, 0.021244]; favours Glicko, CI includes 0.
- Both players active last 365 days: games=11036, Brier diff=0.003126, log-loss diff=0.006504, 95% Brier CI [0.001754, 0.004514]; favours Glicko, CI excludes 0.
- Highest RD quartile: games=2845, Brier diff=0.000497, log-loss diff=-0.002370, 95% Brier CI [-0.004360, 0.005441]; favours Glicko, CI includes 0.

## Limitations

- Subgroup results are exploratory and should not be interpreted as causal proof.
- Some debut and inactive groups are small, so confidence intervals can be wide.
- The script uses fixed meeting5 model outputs and does not retune models on 2025 results.

## Files written

- `outputs\meeting6\29_model_alignment_checks.csv`
- `outputs\meeting6\29_per_match_model_scores_2025.csv`
- `outputs\meeting6\29_overall_model_metrics.csv`
- `outputs\meeting6\29_subgroup_model_performance_long.csv`
- `outputs\meeting6\29_subgroup_pairwise_comparisons.csv`
- `outputs\meeting6\29_subgroup_bootstrap_confidence_intervals.csv`
- `outputs\meeting6\29_corrected_calibration_summary.csv`
- `outputs\meeting6\29_corrected_calibration_bins.csv`
- `outputs\meeting6\29_adaptive_k_improvement_recovered.csv`
- `outputs\meeting6\29_key_meeting6_results.csv`
- `outputs\meeting6\29_where_glicko_helps_validation_checks.csv`
- `outputs\meeting6\29_where_glicko_helps_summary.md`
- `outputs\meeting6\29_glicko_rd_quartile_cutpoints.csv`
- `outputs\meeting6\figures\29_fig01_overall_brier_zoomed.png`
- `outputs\meeting6\figures\29_fig02_delta_brier_by_total_games.png`
- `outputs\meeting6\figures\29_fig03_delta_brier_by_recent_activity.png`
- `outputs\meeting6\figures\29_fig04_delta_brier_by_inactivity.png`
- `outputs\meeting6\figures\29_fig05_inflation_gain_by_inactivity.png`
- `outputs\meeting6\figures\29_fig06_delta_brier_by_prediction_confidence.png`
- `outputs\meeting6\figures\29_fig07_delta_brier_by_glicko_rd.png`