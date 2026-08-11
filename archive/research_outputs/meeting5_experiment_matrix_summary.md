# Meeting 5 Experiment Matrix

## Current Completed Work

- Elo baseline built from 2025 prototype to full-history 1985-2025 framework.
- Elo burn-in, rating-list stability, single-year rerun, and event-level volatility diagnostics completed.
- Elo baseline decision summary completed with conservative, default, and validation-best Elo roles.
- Glicko-1 core implemented and sanity checked.
- Full-history match-by-match Glicko baseline completed.
- Glicko rating-period sensitivity completed for match-by-match, event-level, monthly, and yearly periods under C=0.

## Existing File Check

- Required files found: 9 / 9
- Missing files: None

## Why This Experiment Matrix Is Needed

The previous work produced defensible Elo baselines and an initial Glicko-1 implementation. The next stage should avoid changing multiple things at once. This matrix separates implementation validation, Glicko inactivity RD inflation, rating-period runtime, fair Elo-vs-Glicko comparison, and adaptive-K Elo into distinct experiment groups.

## Fixed Evaluation Design

- Dataset: 1985-2025 full-history checked match-level dataset.
- Evaluation set: fixed 2025 games, expected n=11,379.
- Prediction rule: record pre-match prediction before updating ratings.
- Main prediction metrics: log loss and Brier score.
- Secondary metrics: accuracy, calibration, confidence bins, rating-list stability, RD behaviour, runtime, and active-player ranking overlap.
- Fair comparison principle: isolate one single difference whenever possible.

## Experiment Groups

- Adaptive-K Elo comparison: 2 planned rows
- Fair Elo-vs-Glicko comparison: 4 planned rows
- Final plots and summary tables: 1 planned rows
- Glicko implementation validation: 5 planned rows
- Glicko inactivity RD inflation sensitivity: 4 planned rows
- Glicko rating-period runtime comparison: 4 planned rows

## Priority Order Before Meeting 5

1. Glicko implementation validation
2. Glicko inactivity RD inflation sensitivity
3. Glicko rating-period runtime comparison
4. Fair Elo-vs-Glicko comparison
5. Adaptive-K Elo comparison
6. Final plots and summary tables

## Outputs To Prepare For Supervisor

- A concise implementation validation table for Glicko-1.
- RD inflation sensitivity table showing C=0, low C, medium C, and high C.
- Rating-period runtime table including runtime seconds, number of periods, update operations, and 2025 metrics.
- Fair Elo-vs-Glicko metric table using identical 2025 evaluation games.
- Calibration and confidence diagnostics for selected Elo and Glicko variants.
- Active-player rating-list similarity table, especially for high-volume players.

## Short English Summary For Meeting Notes

For Meeting 5, the next stage is to turn the completed Elo and initial Glicko work into a controlled comparison framework. The priority is to validate the Glicko implementation, isolate the effect of inactivity RD inflation, add runtime evidence for rating-period choices, and then compare Elo and Glicko under the same dataset, chronological ordering, prediction-before-update rule, and 2025 evaluation metrics.
