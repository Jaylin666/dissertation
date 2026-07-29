# Glicko Probability Orientation Audit

## Conclusion

The audit confirms case C: this is an evaluation problem for Glicko probabilities, not only a field-naming issue.
Meeting 5 stores actual-winner probabilities. Because Glicko expected scores are not complementary when RDs differ, Step 29's conversion to player-A probability is outcome-dependent for Glicko.

## Code-Level Findings

- `glicko_core.expected_score(rating, opponent_rating, opponent_rd)` uses the opponent RD.
- `24_glicko_rd_inflation_sensitivity.py` computes `pred_winner_win = expected_score(winner_rating_before, loser_rating_before, loser_rd_before)`.
- `29_where_glicko_helps.py` converts winner probability to player-A probability as `p_a = p_winner` if player A won, else `1 - p_winner`.
- For Elo this conversion is harmless because direct probabilities are complementary. For Glicko it is not harmless.

## Complement Gap

- Glicko low mean absolute complement gap: 0.003930; max: 0.081797.
- Glicko C0 mean absolute complement gap: 0.002624; max: 0.087849.
- Current `p_a_Glicko_*` matches `p_A_direct` when player A won and `1 - p_B_direct` when player A lost.

## Metric Impact

- Existing Glicko low overall Brier: 0.187724; delta vs Elo: 0.002349.
- Fixed player-A direct Glicko low overall Brier: 0.187604; delta vs Elo: 0.002469, CI [0.001008, 0.003903].
- Symmetric diagnostic Glicko low overall Brier: 0.187649; delta vs Elo: 0.002424.
- Fixed player-A direct Glicko C0 overall Brier: 0.195708; delta vs Elo: -0.005635.

## Key Subgroups After Fixed Player-A Direct Correction

- Exactly one debut: delta Brier vs Elo = -0.118005, CI [-0.161309, -0.071619].
- Returning >=365 days, no debut: delta Brier vs Elo = 0.001258, CI [-0.013853, 0.015136].

## What Must Be Recomputed

- Glicko low and Glicko C0 Brier, log loss, accuracy, calibration, Murphy decomposition, subgroup comparisons and bootstrap intervals.
- Any Glicko-vs-Elo and Glicko-vs-adaptive-K comparison using Step 29/31 Glicko probabilities.
- Meeting figures based on those Glicko probabilities.

## What Can Be Retained

- Step 28 pre-match feature construction.
- Rating lists, RD distributions, runtime and rating-period/update-operation conclusions.
- Unique-player rating snapshot and debut-opponent rating distribution.
- Rating-level evidence for the debut initialisation mismatch.

## Validation

- Audit checks passed: 14 / 14.

## Files Written

- `outputs\meeting6\32_probability_orientation_audit_checks.csv`
- `outputs\meeting6\32_glicko_direct_probability_comparison.csv`
- `outputs\meeting6\32_glicko_complement_gap_summary.csv`
- `outputs\meeting6\32_probability_orientation_by_subgroup.csv`
- `outputs\meeting6\32_orientation_impact_on_metrics.csv`
- `outputs\meeting6\32_glicko_probability_orientation_audit_summary.md`