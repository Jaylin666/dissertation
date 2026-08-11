# Step 39 Glicko Orientation Sensitivity Audit

## 1. Why this audit was necessary
Step 32 showed that the two direct Glicko expected scores need not be complementary when player RDs differ. Step 33 therefore used a canonical smaller-ID Player A convention. This audit checks whether Meeting 7 conclusions depend on that arbitrary direction.

## 2. Current Step 33 convention
The current convention is reproduced as `expected_score(rating_small, rating_large, RD_large)`, with the smaller player ID as canonical Player A.

## 3. Two direct Glicko expected scores
`E_small_direct` uses the larger player's RD. `E_large_direct` uses the smaller player's RD. Their sum can differ from one.

## 4. Complementarity gap
Overall mean absolute complement gap is 0.003930; maximum is 0.081797.

## 5. Player-ID side imbalance
For first_1 focal appearances, 5.26% are on the smaller-ID side and 94.74% are on the larger-ID side.

## 6. Overall performance under conventions
| model               |    brier |   log_loss |   accuracy |   prediction_bias |
|:--------------------|---------:|-----------:|-----------:|------------------:|
| Validation_best_Elo | 0.190073 |   0.556534 |   0.704456 |       0.000421004 |
| Glicko_low_current  | 0.187604 |   0.551779 |   0.711486 |       0.00198086  |
| Glicko_low_reversed | 0.188098 |   0.553908 |   0.711486 |       0.00220899  |
| Glicko_low_midpoint | 0.187831 |   0.552661 |   0.711486 |       0.00209492  |

## 7. Early-game performance under conventions
| model                   |    brier |   log_loss |   accuracy |   mean_predicted_probability |   prediction_bias |
|:------------------------|---------:|-----------:|-----------:|-----------------------------:|------------------:|
| Validation_best_Elo     | 0.210522 |   0.60329  |   0.644737 |                     0.538536 |          0.130642 |
| Glicko_low_current      | 0.322316 |   0.895574 |   0.513158 |                     0.743448 |          0.335554 |
| Glicko_low_reversed     | 0.360375 |   1.05364  |   0.513158 |                     0.786994 |          0.379099 |
| Glicko_low_midpoint     | 0.340437 |   0.962676 |   0.513158 |                     0.765221 |          0.357326 |
| Glicko_low_direct_focal | 0.359858 |   1.05218  |   0.513158 |                     0.788093 |          0.380199 |

## 8. First_1 over-prediction
Current first_1 mean probability is 0.7434483364113538; reversed is 0.7869939795121185; midpoint is 0.7652211579617361; empirical win rate is 0.40789473684210525.

## 9. Overall Glicko advantage
Overall low-inflation Glicko remains better than validation-best Elo under the current, reversed and midpoint conventions if the delta Brier remains positive in all three.

## 10. Direct focal diagnostic
The direct focal probability is diagnostic only because it is not complementary within a match and therefore is not the primary match-level proper-score forecast.

## 11. Final conclusion code
ROBUST_TO_ORIENTATION

## 12. Meeting 7 implications
The audit states whether Meeting 7 claims must be changed in `39_key_orientation_results.csv`.

## 13. Main convention status
This step does not replace the Step 33 main convention. It only provides sensitivity evidence around it.

Validation checks: 21 PASS / 0 FAIL.