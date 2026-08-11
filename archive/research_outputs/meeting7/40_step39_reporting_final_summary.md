# Step 40 Reporting Corrections and Finalisation

## 1. Purpose

Step 40 finalises the reporting around the Step 39 orientation sensitivity audit. It corrects interpretation metadata and creates wording that is precise enough for the Meeting 7 report.

## 2. Numerical results

No probability, Brier score, log-loss value, accuracy value or confidence interval was changed. Step 40 only creates corrected copies and final reporting files.

## 3. Candidate-bias metadata

`candidate_bias` rows now use the interpretation: positive value means over-prediction; negative value means under-prediction. Corrected candidate-bias metadata rows: 12.

## 4. Overall Brier conclusion

The overall Brier advantage of low-inflation Glicko remains statistically clear under the current, reversed and midpoint conventions.

## 5. Overall log-loss conclusion

The overall log-loss point estimate favours low-inflation Glicko under all three conventions, but the reversed-convention confidence interval crosses zero.

## 6. First_1 conclusion

The first_1 Validation-best Elo advantage and the Glicko upward prediction bias remain clear under all three conventions.

## 7. First_5 to first_20 evidence

The direction of point estimates is broadly consistent, but whether confidence intervals exclude zero varies across probability conventions, metrics and analysis units.

## 8. Scope of ROBUST_TO_ORIENTATION

`ROBUST_TO_ORIENTATION` means: The main overall and first-appearance conclusions are robust to the tested outcome-independent orientation conventions.

It does not mean every subgroup confidence interval receives an identical classification under every convention.

## 9. Secondary note

`EARLY_WINDOW_SIGNIFICANCE_VARIES_BY_CONVENTION` means: The statistical strength of the first_5 to first_20 results varies across conventions, metrics and analysis units, although the overall early-game pattern remains similar.

## 10. Meeting 7 wording

Use `40_meeting7_orientation_wording.md` for concise wording in the Meeting 7 report.

## 11. Dissertation robustness wording

In the dissertation, Step 39 should be described as an orientation sensitivity audit: the main convention remains Step 33, with reversed and midpoint conventions used to show that the key overall and first-appearance conclusions are not artifacts of canonical player-ID direction.

## 12. Further model work

No additional rating-model experiment is required before Meeting 7 on the basis of this orientation audit.
