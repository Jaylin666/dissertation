# Meeting 7 Final Report Summary

## A. Executive Summary

Meeting 7 has produced a coherent answer to the supervisor's early-game question: ratings are least reliable for players with no previous recorded history, and the reason is not simply that the model has too little data, but that the Glicko initialisation and probability mechanism can strongly over-predict those players when they enter against lower-rated established opponents.

The full 2025 result still favours low-inflation Glicko overall, but the first recorded appearance is an exception where validation-best Elo is clearly better. The evidence weakens as the early-career window expands: the first_5 result still tends to favour Elo, while first_10 and first_20 are closer and should be described more cautiously.

A key diagnostic finding is that changing the common Glicko initial rating from 1000 to 1500 does not change validation or 2025 test performance. This is expected because a common additive shift to all ratings does not change rating differences. The problem is therefore not the absolute initial rating value, but the relative state created when a new player enters with the common initial rating against opponents whose Glicko ratings have already moved.

The orientation sensitivity audit concluded `ROBUST_TO_ORIENTATION`. Overall Brier conclusions and the first-appearance conclusion are robust across current, reversed and midpoint conventions. The caveat is: `EARLY_WINDOW_SIGNIFICANCE_VARIES_BY_CONVENTION`.

## B. Overall Model Comparison

Source: `outputs/meeting6/33_overall_model_metrics.csv` and independently checked against `outputs/meeting6/33_orientation_corrected_per_match_scores_2025.csv`.

| Model | Games | Brier | Log loss | Accuracy |
| --- | --- | --- | --- | --- |
| Validation-best Elo | 11379 | 0.190073 | 0.556534 | 0.704456 |
| Glicko low inflation | 11379 | 0.187604 | 0.551779 | 0.711486 |
| Glicko C0 | 11379 | 0.195708 | 0.571958 | 0.693734 |
| Best adaptive-K Elo | 11379 | 0.190781 | 0.559185 | 0.706213 |

Interpretation: low-inflation Glicko remains the best overall 2025 model by Brier score, log loss and accuracy. Glicko C0 is worse than both low-inflation Glicko and the validation-best Elo baseline, supporting the idea that uncertainty inflation is useful for the full test set.

## C. Early-Game Results

Source: `outputs/meeting7/34_cumulative_threshold_model_performance.csv`, `outputs/meeting7/34_pairwise_model_differences.csv`, and `outputs/meeting7/34_bootstrap_confidence_intervals.csv`. Delta Brier is Elo minus Glicko, so negative values favour Elo.

| Group | Appearances | Elo Brier | Glicko Brier | Glicko C0 Brier | Adaptive-K Brier | Elo-Glicko Delta | Bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first_1 | 76 | 0.210522 | 0.322316 | 0.348585 | 0.225332 | -0.111794 | [-0.156353, -0.065963] |
| first_5 | 406 | 0.233808 | 0.253809 | 0.257511 | 0.240644 | -0.020001 | [-0.036375, -0.002396] |
| first_10 | 855 | 0.221076 | 0.231489 | 0.235212 | 0.228035 | -0.010413 | [-0.022813, 0.002185] |
| first_20 | 1695 | 0.212656 | 0.216538 | 0.220814 | 0.217926 | -0.003882 | [-0.010851, 0.003080] |

Interpretation: the first recorded appearance is the clearest failure point for Glicko. By first_20, the Brier gap is much smaller, so the answer to 'how quickly do ratings become reliable?' is gradual rather than immediate: the most severe problem is at match 1, with partial convergence over the next 10 to 20 appearances.

## D. Non-Overlapping Stage Bins

Source: `outputs/meeting7/34_stage_bin_model_performance.csv` and `outputs/meeting7/35_glicko_rating_rd_summary.csv`.

| Stage | Appearances | Elo Brier | Glicko Brier | Elo-Glicko Delta | Mean Glicko RD |
| --- | --- | --- | --- | --- | --- |
| 1 | 76 | 0.210522 | 0.322316 | -0.111794 | 350.000000 |
| 2-5 | 330 | 0.239171 | 0.238032 | 0.001139 | 246.833949 |
| 6-10 | 449 | 0.209563 | 0.211306 | -0.001743 | 172.656305 |
| 11-20 | 840 | 0.204085 | 0.201320 | 0.002766 | 137.268638 |
| 21-50 | 1807 | 0.204480 | 0.202124 | 0.002356 | 107.953549 |
| 51+ | 19256 | 0.186733 | 0.183694 | 0.003038 | 77.086921 |

Interpretation: non-overlapping bins show the shape of the learning curve more cleanly than cumulative groups. Glicko is poor at stage 1, becomes competitive in stage 2-5, and then becomes close to or better than Elo in later stages. This supports the dissertation argument that Glicko's uncertainty machinery is not uniformly bad for new players; the main problem is the first recorded appearance and its initialisation context.

## E. Mechanism Analysis

Source: `outputs/meeting7/35_cumulative_probability_bias_summary.csv`, `outputs/meeting7/35_prediction_extremity_summary.csv`, and `outputs/meeting7/35_key_mechanism_results.csv`.

| Model | First_1 Mean p | Empirical Win Rate | Prediction Bias | Extreme p share |
| --- | --- | --- | --- | --- |
| Validation-best Elo | 0.538536 | 0.407895 | 0.130642 | 0.026316 |
| Glicko low inflation | 0.743448 | 0.407895 | 0.335554 | 0.105263 |
| Glicko C0 | 0.768341 | 0.407895 | 0.360446 | 0.092105 |
| Adaptive-K Elo | 0.570254 | 0.407895 | 0.162359 | 0.000000 |

Interpretation: first_1 focal players win about 40.8% of appearances, but Glicko low inflation predicts about 74.3% on average. The resulting bias of about 0.336 is much larger than Elo's bias of about 0.131. This is the central mechanism behind the first-appearance Brier gap.

## F. Initialisation Source Diagnostic

Source: `outputs/meeting7/36_debut_state_summary.csv`, `outputs/meeting7/36_debut_counterfactual_probability_diagnostics.csv`, and `outputs/meeting7/36_key_initialisation_diagnostic_results.csv`.

| Diagnostic | Mean p | Bias | Brier |
| --- | --- | --- | --- |
| Observed first_1 Glicko low | 0.743448 | 0.335554 | 0.322316 |
| Equalise focal rating to opponent | 0.500000 | 0.092105 | 0.250000 |

The first_1 focal Glicko rating is 1500.000 while the mean opponent Glicko rating is 1180.755015; the mean focal-minus-opponent difference is 319.244985. Equalising the focal rating to the opponent rating reduces the mean predicted probability to 0.500 and reduces Brier to 0.250. This shows that the first-appearance problem is mainly driven by relative rating state rather than RD alone.

## G. Initial Rating Sensitivity

Source: `outputs/meeting7/37_validation_initial_rating_metrics.csv`, `outputs/meeting7/37_test_initial_rating_metrics.csv`, and `outputs/meeting7/37_key_initial_rating_results.csv`.

| Period | Candidate Initial Ratings | Brier Range | Selected |
| --- | --- | --- | --- |
| Validation 2023-2024 | 1000, 1100, 1200, 1300, 1400, 1500 | 0.191256 to 0.191256 | 1500 |
| Test 2025 | 1000, 1100, 1200, 1300, 1400, 1500 | 0.187604 to 0.187604 | 1500 |

Interpretation: all candidate common initial ratings produced identical validation and 2025 test Brier scores. This is a useful methodological point for the dissertation: common initial rating is not an independently meaningful hyperparameter when all players are shifted together; rating differences and update dynamics matter.

## H. Asymmetric Adaptive-K Elo

Source: `outputs/meeting7/38_overall_model_metrics.csv`, `outputs/meeting7/38_asymmetric_k_summary.csv`, and `outputs/meeting7/38_glicko_gap_recovery.csv`.

| Model | Brier | Log loss | Accuracy |
| --- | --- | --- | --- |
| Validation-best Elo | 0.190073 | 0.556534 | 0.704456 |
| Adaptive-K reference | 0.190781 | 0.559185 | 0.706213 |
| Asymmetric adaptive-K | 0.190781 | 0.559185 | 0.706213 |
| Low-inflation Glicko | 0.187604 | 0.551779 | 0.711486 |

The proof-of-concept did assign different K values to the two players in 52.948414% of 2025 matches, with mean absolute K difference 6.215836. However, the asymmetric and symmetric adaptive-K predictions are numerically unchanged for 2025, and the maximum recovery fraction of the Glicko gap is 4.718e-12. This should be reported as a negative but informative proof-of-concept: simply allowing separate player K values is not enough to recover Glicko's advantage.

## I. Orientation Sensitivity

Source: `outputs/meeting7/39_orientation_sensitivity_comparison.csv`, `outputs/meeting7/39_complement_gap_summary.csv`, `outputs/meeting7/39_early_player_side_distribution.csv`, and Step 40 corrected reporting files.

| Group | Observations | Current Elo-Glicko Delta Brier | CI | Current Conclusion | Step 40 Note |
| --- | --- | --- | --- | --- | --- |
| overall | 11379 | 0.002469 | [0.001079, 0.003957] | GLICKO_BETTER | Overall Brier is robust; log-loss point estimates favour Glicko, with reversed CI crossing zero. |
| first_1 | 74 | -0.114815 | [-0.157896, -0.069459] | ELO_BETTER | First_1 Elo advantage and Glicko over-prediction are robust. |
| first_5 | 387 | -0.019756 | [-0.039335, 0.001015] | NO_CLEAR_DIFFERENCE | Point-estimate direction is broadly similar, but CI classification varies by convention, metric or analysis unit. |
| first_10 | 785 | -0.007721 | [-0.020956, 0.006764] | NO_CLEAR_DIFFERENCE | Point-estimate direction is broadly similar, but CI classification varies by convention, metric or analysis unit. |
| first_20 | 1463 | -0.001249 | [-0.010489, 0.007551] | NO_CLEAR_DIFFERENCE | Point-estimate direction is broadly similar, but CI classification varies by convention, metric or analysis unit. |

The all-match mean absolute complement gap for low-inflation Glicko is 0.003930, with maximum 0.081797. For first_1 focal appearances, 94.736842% are on the larger-ID side, which explains why the early-game convention matters more than the overall convention. Step 40 confirms that the main conclusions are robust, although some first_5 to first_20 confidence classifications vary by convention or metric.

## J. Recommended Figures and Tables

Recommended figures:

- `outputs/meeting7/figures/34_fig02_cumulative_brier_by_model.png`
- `outputs/meeting7/figures/34_fig03_cumulative_delta_brier_elo_minus_glicko_ci.png`
- `outputs/meeting7/figures/35_fig01_predicted_vs_empirical_by_stage.png`
- `outputs/meeting7/figures/35_fig05_glicko_rd_by_stage.png`
- `outputs/meeting7/figures/36_fig05_counterfactual_probability_comparison.png`
- `outputs/meeting7/figures/38_fig01_overall_brier_comparison.png`
- `outputs/meeting7/figures/39_fig03_elo_glicko_delta_brier_by_convention.png`

Figures to avoid or replace:

- `outputs/meeting7/figures/38_fig07_glicko_gap_recovery.png - exclude or replace with the recovery table because the near-zero recovery fraction makes the plotted trend visually uninformative.`

Core tables for the meeting report should be: overall model comparison, cumulative early-game Brier table, stage-bin Brier/RD table, first_1 mechanism table, initialisation counterfactual table, asymmetric-K overall table, and orientation sensitivity table.

## K. Suggested Meeting Report Structure

1. One-page summary of the research question and headline findings.
2. Overall model comparison after Step 33 orientation correction.
3. Early-game analysis: first_1, first_5, first_10 and first_20.
4. Mechanism: why Glicko fails on first recorded appearances.
5. Initialisation diagnostic and common initial-rating sensitivity.
6. Adaptive-K extension result.
7. Orientation sensitivity and final methodological caveats.
8. Questions for Chris and dissertation framing.

## L. Dissertation Framework to Discuss with Chris

Proposed dissertation structure:

- Introduction: prediction problem, why croquet ratings are interesting, and research questions.
- Data and preprocessing: raw match data, canonical player orientation, full-history construction, evaluation splits, and Step 33 probability correction.
- Methods: Elo baseline, validation-selected Elo, adaptive-K Elo, Glicko, uncertainty/RD and evaluation metrics.
- Main model comparison: full 2025 Elo-Glicko comparison, sensitivity checks, and why low-inflation Glicko beats Glicko C0 overall.
- Early-career reliability: appearance-level dataset, first_N and stage-bin analysis, mechanism and initialisation diagnostics.
- Extensions and robustness: asymmetric adaptive-K proof of concept, orientation sensitivity, bootstrap robustness, and limitations.
- Conclusion: what was learned about model flexibility, uncertainty, and the limits of rating systems for new players.

Questions to ask Chris:

- Should the dissertation frame the first_1 result as a limitation of Glicko initialisation, or more generally as a limitation of rating systems for players with no recorded history?
- Is it worth including the asymmetric adaptive-K proof-of-concept as a negative result, or should it be kept brief as robustness/extension material?
- How much detail should be given to the orientation sensitivity audit in the main text versus an appendix?
- Does Chris prefer the early-game chapter to focus on cumulative first_N groups, non-overlapping stage bins, or both?
