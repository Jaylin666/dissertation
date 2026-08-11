# Meeting 5 Glicko Implementation Validation

## Purpose

This validation step answers the supervisor's question: how am I gaining confidence in the implementation of Glicko? The aim is not to tune the model or perform the final Elo-vs-Glicko comparison. The aim is to check that the Glicko implementation behaves in ways that are consistent with the Glicko mechanism.

## Data and Existing Outputs Used

- glicko_core: `code\glicko_core.py`
- glicko_predictions: `outputs\glicko_implementation\glicko_mbm_predictions_1985_2025.csv`
- glicko_final_ratings: `outputs\glicko_implementation\glicko_mbm_final_ratings_1985_2025.csv`
- elo_final_ratings: `outputs\elo_optimization\elo_burnin_final_ratings_all_runs.csv`
- full_history_matches: `outputs\elo_optimization\matches_1985_2025_checked.csv`
- elo_baseline_decision: `outputs\elo_optimization\elo_baseline_decision_summary.md`
- glicko_rating_period_metrics: `outputs\glicko_implementation\glicko_rating_period_metrics_2025.csv`
- glicko_rd_summary: `outputs\glicko_implementation\glicko_mbm_rd_summary.csv`

Missing files:

- None

## Formula Sanity Checks

- A1: PASS; 0.500000000000
- A2: PASS; 0.595113968002
- A3: PASS; 0.404886031998
- A4: PASS; A rating 1500.0->1662.212; B rating 1500.0->1337.788; RD 350.0->290.231
- A5: PASS; weaker change=44.195; stronger change=-44.195

## Official-Style Example

- B1: PASS; new_rating=1464.106; new_rd=151.399

## RD Behaviour Checks

- Constants: DEFAULT_RD=350.0, MIN_RD=30.0, MAX_RD=350.0
- Final players checked: 5143
- Median final RD: 58.601
- Mean final RD: 77.030
- Min/max observed final RD: 30.000 / 302.938
- Players at MIN_RD: 1295
- Players near MAX_RD: 0

## Prediction-Before-Update Checks

- Prediction rows checked: 456382
- 2025 evaluation rows: 11379
- Prediction probability range: 0.000899 to 0.999354
- Max difference when recomputing pred_a_win from pre-rating/RD columns: 1.44328993201e-15

## Active-Player Elo-vs-Glicko Rating-List Similarity

For this implementation validation, rank correlation and top-list overlap are more important than raw rating differences because Elo and Glicko ratings are not necessarily on exactly the same scale.

- total_games_ge100: n=1831, Spearman=0.8469, Top50=0.640, Top100=0.630
- total_games_ge200: n=1206, Spearman=0.8644, Top50=0.700, Top100=0.630
- active_2025_games_ge5: n=950, Spearman=0.9141, Top50=0.800, Top100=0.800
- active_2025_games_ge5_total_games_ge100: n=565, Spearman=0.9218, Top50=0.800, Top100=0.830

## Validation Check Summary

- Checks passed: 16 / 16
- Overall status: PASS
- Scatter plot created: yes

## Interpretation For Supervisor

The validation checks give me more confidence that the Glicko implementation is behaving as intended. The formula sanity checks and official-style update example are consistent with expected Glicko behaviour. The saved predictions are valid probabilities and are evaluated on the same 2025 game set. For high-activity players, the Glicko and Elo rating lists are broadly similar, which suggests that the Glicko implementation is not producing implausible rankings. The next step is therefore to test inactivity RD inflation and then proceed to the fair Elo-vs-Glicko comparison.

## Remaining Limitations

- Passing sanity checks does not prove the implementation is mathematically perfect.
- Some checks depend on which columns were saved in previous output files.
- Elo and Glicko ratings are not guaranteed to be directly comparable on the raw rating scale, so rank-based checks are more important.
- Full confidence also requires sensitivity checks such as RD inflation and rating-period runtime comparison.

## Issues

- None
