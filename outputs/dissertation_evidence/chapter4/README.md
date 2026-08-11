# Chapter 4 Dissertation Evidence

This directory is the controlled evidence source for Chapter 4. It was assembled by copying existing compact results and figures only. No rating model was rerun, no source result was edited, and no per-game prediction history or other large row-level file was copied.

Use `evidence_manifest.csv` to trace each evidence file to its original path, generating script, analytical scope, probability orientation, selection status, superseded results and SHA-256 checksum.

## Section 4.2: Elo parameter selection

Use:

- `elo_validation_grid.csv` for the complete 28-setting grid;
- `elo_validation_selected_model.csv` for the frozen selected row.

The grid runs Elo from 2015 to 2025, uses 2015-2022 as burn-in, 2023-2024 as validation and 2025 as the held-out test year. It is parameter-selection evidence, not the final full-history model comparison. The final 2025 overall comparison in Section 4.3 uses ratings developed from the 1985-2025 full history.

The validation log loss, Brier score and accuracy for `K=30, scale=300` and `K=40, scale=400` are exactly equal in the saved grid. The ordering in Step 11 is validation log loss, validation Brier score, `K`, then `scale`, all ascending. The lower-`K` setting is therefore retained: `K=30, scale=300` has rank 1 and `K=40, scale=400` has rank 2. No 2025 test metric is used to break this tie.

## Section 4.3: Overall 2025 model comparison

Use:

- `overall_model_metrics.csv` for final overall Brier score, log loss and accuracy;
- `overall_pairwise_comparisons.csv` for point-estimate model differences;
- `overall_bootstrap_confidence_intervals.csv` for event-cluster confidence intervals based on 2,000 bootstrap replications;
- `overall_brier_zoomed.png` for the final overall Brier figure.

These are Step 33 orientation-corrected outputs based on 11,379 evaluation **games** in 629 events. Do not call the 11,379 rows matches in the dissertation; use games.

## Section 4.4: Calibration

Use:

- `calibration_summary.csv` for overall and subgroup weighted absolute calibration errors;
- `calibration_bins.csv` for statements about individual probability bands;
- `calibration_player_a.png` for the final calibration figure.

All three are Step 33 outputs using canonical Player A, `outcome_a` and direct Player-A probability. The bins cover 0.0-0.1 through 0.9-1.0. Do not use an actual-winner calibration interpretation.

## Section 4.5: Glicko sensitivity analyses

Use `glicko_inflation_sensitivity.csv` only as an exploratory 2025 sensitivity comparing the relative ordering of C0, low, medium and high inactivity-inflation settings. This historical table uses actual-winner probability. The formal headline comparison between Glicko C0 and low inflation must come from the Step 33 orientation-corrected files in Section 4.3, not from this sensitivity table.

Use `glicko_rating_period_metrics.csv` for the C=0 comparison of game-by-game, event-level, monthly and yearly rating periods. This table uses fixed canonical Player A, `actual_a_win` and direct Player-A probability. Its game-by-game result reconciles with Step 33 orientation-corrected Glicko C0 within floating-point tolerance. It must not be described as a low-inflation rating-period comparison. The Step 25 low-inflation winner-oriented table can describe historical period ordering but is not the formal numerical source used here.

## Section 4.6: Adaptive-K evidence

Use `adaptive_k_recovery.csv` for the short adaptive-K negative/proof-of-concept result. It is an exploratory extension, not a validation-selected replacement for the fixed Elo comparator.

## Selection and reporting status

- `validation_selected`: selected using the 2023-2024 Elo validation set before evaluating the frozen 2025 test set.
- `validation_selection_evidence`: the complete grid supporting that selection.
- `orientation_corrected_final`: final Step 33 evidence for Chapter 4 headline model comparisons.
- `exploratory_2025_sensitivity`: a 2025 sensitivity or proof-of-concept that must not be presented as independently validation-selected.

## Probability orientation and supersession rules

For Step 33 evidence, canonical Player A is defined independently of the game result as the lower numeric player identifier, `outcome_a` records whether that player won, and Glicko probabilities are calculated directly from Player A's perspective. This is not actual-winner orientation.

Do not use probability-based outputs from Steps 29-31 in place of the Step 33 overall, pairwise, bootstrap or calibration evidence. Step 33 supersedes those older probability-based results. Historical non-probability diagnostics may still be used when their scope is stated correctly.

## Integrity and scope

- SHA-256 hashes in `evidence_manifest.csv` identify the exact archived bytes.
- Copied CSV and PNG files are byte-identical to their listed sources.
- The only derived table is `elo_validation_selected_model.csv`, which is a seven-column, one-row extraction from the Elo validation grid.
- No per-game dataset, prediction history, full rating history or other large row-level file belongs in this directory.
