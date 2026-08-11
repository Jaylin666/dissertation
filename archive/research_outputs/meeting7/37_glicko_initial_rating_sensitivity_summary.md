# Step 37 Glicko Initial-Rating Sensitivity

## 1. Research question
This experiment tests whether changing only the Glicko initial rating reduces new-player over-prediction without damaging validation-selected and fixed-test performance.

## 2. Why Step 36 justified this experiment
Step 36 identified the debut initial rating level as the dominant diagnostic source of the first-appearance over-prediction pattern.

## 3. Validation and test split
Candidate selection uses 2023-2024 validation matches only. The fixed test period is 2025.

## 4. Candidate initial ratings
Candidates tested: 1000, 1100, 1200, 1300, 1400, 1500. Initial RD remains 350.

## 5. Validation selection rule
The primary selection criterion is validation Brier score. Log loss, debut bias, and closeness to 1500 are used only within a tie tolerance of 1e-08.

## 6. Selected initial rating
The validation-selected initial rating is 1500. Validation Brier is 0.191256, compared with 0.191256 for 1500.
All candidate Brier scores are identical within tolerance: the maximum validation Brier difference is 0.000e+00, the maximum 2025 Brier difference is 0.000e+00, and the maximum first_1 bias difference is 0.000e+00. This is expected because changing a common initial rating shifts the whole Glicko rating scale but leaves rating differences and probabilities unchanged.

## 7. Overall 2025 performance
On 2025, selected Brier/log loss are 0.187604/0.551779; current 1500 Brier/log loss are 0.187604/0.551779.

## 8. Early-game prediction bias
First_1 bias changes from 0.335554 at 1500 to 0.335554 at the selected rating. First_5 bias changes from 0.143020 to 0.143020.

## 9. Early-game Brier and log loss
First_1 Brier changes from 0.322316 to 0.322316. First_5 Brier changes from 0.253809 to 0.253809.

## 10. Comparison with initial rating 1500
Validation improvement relative to 1500: False. Fixed 2025 Brier improvement relative to 1500: False.

## 11. Comparison with validation-best Elo
Overall 2025 Elo-minus-selected-Glicko Delta Brier is 0.002469; positive values mean selected Glicko has lower Brier than Elo.

## 12. Under-prediction check
Evidence of new-player under-prediction by the selected candidate: False.

## 13. Rating-distribution effects
Selected final rating mean/median/std are 1257.881/1236.456/269.544. Pure location shifts should not be interpreted as ranking-quality improvements.

## 14. Limitations
Only the initial rating is varied. Initial RD, inactivity inflation, expected score formula, and match-by-match rating period are fixed. First recorded appearances may not be true career debuts.

## 15. Main-specification status
Because this is a Meeting 7 sensitivity experiment selected on validation data, it should be treated as a sensitivity model unless the dissertation design explicitly updates the main Glicko specification after documenting the validation rule.
