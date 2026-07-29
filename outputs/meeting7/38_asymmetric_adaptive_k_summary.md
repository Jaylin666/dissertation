# Step 38: Asymmetric Adaptive-K Elo Proof-of-Concept

## Research question

This step tests whether assigning different adaptive-K update sizes to the two players in the same match can recover part of low-inflation Glicko's predictive advantage.

## Existing Step 27 rule

The reused rule is:

- previous_year_games <= 5: K = 30
- 6 <= previous_year_games <= 30: K = 20
- previous_year_games > 30: K = 10
- Elo scale = 300

The Step 27 audit found that the saved adaptive-K model already calculates player-specific `winner_K` and `loser_K`. Therefore the canonical Step 38 A/B asymmetric implementation is algebraically equivalent to the saved Step 27 reference.

## Reproduction check

Step 27 reproduction status: True.

## Validation and test design

Validation uses 2023-2024 only. The fixed 2025 test set contains 11,379 matches. No 2025 outcomes are used for parameter selection.

## Overall 2025 performance

| Model | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| Validation-best Elo | 0.190073 | 0.556534 | 0.704456 |
| Adaptive-K reference | 0.190781 | 0.559185 | 0.706213 |
| Asymmetric adaptive-K | 0.190781 | 0.559185 | 0.706213 |
| Low-inflation Glicko | 0.187604 | 0.551779 | 0.711486 |

## Early-game performance

The early-game comparison uses identical Step 34 focal-player appearance rows across all models. The asymmetric adaptive-K model is identical to the Step 27 adaptive-K reference, so it does not create a new early-game improvement.

## Activity and returning-player performance

Activity subgroup results are saved in `38_activity_subgroup_metrics.csv`. These subgroups use existing Step 33 flags and previous-year activity variables.

## Player-specific K and rating drift

K_A differs from K_B in 52.95% of 2025 matches. Mean absolute K difference is 6.2158. Total net rating drift during 2025 is -2097.905116.

## Recentered robustness

The maximum absolute probability difference after common-shift recentering is 3.664e-15. This confirms that common rating shifts affect displayed rating levels but not predicted probabilities when applied consistently.

## Recovery of the Glicko gap

Overall Brier recovery fraction is -0.000000. Because the asymmetric implementation reproduces the existing adaptive-K reference, it recovers none of the adaptive-K-to-Glicko gap.

## Conclusion and limitations

This proof-of-concept is useful primarily as a method audit. It shows that the supervisor's proposed player-specific update idea was already present in the Step 27 best adaptive-K implementation. Further improvement would therefore require a genuinely different, pre-specified adaptive-K rule or another source of uncertainty information, not merely rewriting the update in A/B asymmetric notation.

Validation checks: 23 PASS, 0 FAIL.
