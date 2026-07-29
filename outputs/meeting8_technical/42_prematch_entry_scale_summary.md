# Meeting 8 Step 42: Prematch Entry-Scale and Cross-File Audit

## Purpose

Step 42 supplements Step 41 by using the exact model-processing order returned by Step 24 and by measuring rating-scale alignment immediately before each player's first recorded match. It reuses the frozen low-inflation Glicko configuration and does not tune or create a rating model.

## Definitions

- **First recorded appearance:** the row with the minimum frozen `match_sequence` for a player in the 1985-2025 processed history.
- **Model-start left censoring:** players first observed in 1985 may already have prior unrecorded experience.
- **Post-burn-in recorded entry:** first observed from 1990 onward after the primary five-calendar-year burn-in.
- **True career debut:** not observed and not claimed by this analysis.

## Exact processing-order audit

- Strict unique players: 5,143
- Step 41 classification available: True
- Step 41 mismatch rows: 0
- Strict classification agrees with Step 41: True

The strict definition uses the sequence already returned by `step24.load_matches()` and does not perform a second chronological sort.

## Cross-file reconciliation

- First-appearance rows: 76
- Unique players: 76
- Unique matches: 74
- Exactly-one-debut matches: 72
- Both-debut matches: 2
- Full history, Step 33, and Step 34 agree exactly: True
- Maximum absolute current-probability difference from Step 34: 5.285e-13

## Prematch contemporaneous scale

For each entrant, the primary contemporaneous scale includes all players with at least one prior processed match immediately before the entrant's match. The current focal player is excluded; when both players are new, both are excluded.

- 2025 mean contemporaneous established-player rating: 1258.683
- 2025 median contemporaneous established-player rating: 1237.367
- 2025 mean initial-minus-contemporaneous-median gap: 262.659
- 2025 median initial-minus-contemporaneous-median gap: 262.633
- 2025 mean active-established-365-day median: 1314.848
- 2025 mean initial-minus-active-365-day-median gap: 185.152
- 2025 mean actual opponent rating: 1180.755
- 2025 mean initial-minus-opponent gap: 319.245

These are prematch entry-time quantities. They are distinct from the Step 41 end-of-year 2025 established-active median of 1334.544; the end-of-year value must not be described as the contemporaneous entry-time scale.

## Probability-orientation sensitivity

- 1990-2024 current-convention bias: 0.268737
- 1990-2024 direct-focal bias: 0.288028
- 2025 current-convention mean probability: 0.743448
- 2025 direct-focal mean probability: 0.788093
- 2025 empirical win rate: 0.407895
- Qualitative post-burn-in over-prediction direction persists: True

The primary formal 2025 result remains the Meeting 7 current convention. Direct-focal probabilities are a historical orientation sensitivity only. Results from 1990-2024 are in-sample descriptive mechanism evidence, not an independent held-out test.

## Interpretation

The held-out 2025 first-appearance sample consists entirely of players who enter the recorded system after a long burn-in. The weakness is therefore not an artefact of the 1985 model start. Immediately before entry, the fixed 1500 anchor is high relative to both the contemporaneous established-player scale and the actual first opponent ratings. This supports a relative initialisation mismatch interpretation. However, first recorded appearance does not establish true career debut, and the 1990-2024 cohort results are descriptive historical mechanism evidence rather than an independent held-out test.

Rating-scale alignment is one component of the mechanism. This diagnostic does not make a causal claim that rating-scale drift alone creates the prediction error.

## Relationship to Step 41

- Step 41 provides broad burn-in classification and end-of-year scale trends.
- Step 42 provides strict processing-order classification, prematch scale alignment, and direct Step 33/34 reconciliation.
- The two steps are complementary.
- Step 42 invalidates no Step 41 result because explicit classification mismatches found: 0.

## Dissertation use

- Put operational entry definitions in Methodology.
- Put the held-out 2025 first-appearance result in the Early-game Results section.
- Put prematch scale alignment in the mechanism subsection.
- Label historical cohort and direct-probability results as sensitivity or supporting evidence.
- Treat the Step 41 end-of-year scale figure as descriptive rather than the primary initialisation diagnostic.

## Validation

- Total checks: 54
- Passed error checks: 48/48
- Failed error checks: 0
- Active warnings: 4
- Dataset: `outputs\elo_optimization\matches_1985_2025_checked.csv`
- Inactivity unit: month
- Reused variant: `low_inflation` with C=22.509257354846

### Active warnings

- event_dates_complete: Missing dates are retained; date-based active scales are unavailable for affected rows. Observed=46.
- contemporaneous_scale_available_for_all_entries: Unavailable values are expected at the beginning of 1985 before any player is established. Observed=2.
- active_365d_scale_available_for_all_entries: The secondary date-based scale can be unavailable for missing dates or an empty active pool. Observed=2.
- probability_convention_bias_difference_below_0_01: Material convention differences are reported as sensitivity evidence and do not replace the current convention. Observed=0.04464506725752726.
