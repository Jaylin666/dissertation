# Numbered-script migration map

The archive path preserves chronology. The active replacement identifies the
recommended current implementation; “archive only” means the script remains
scientifically auditable but is not part of the supported active workflow.

| Original script | Preserved location | Active replacement or status |
|---|---|---|
| `01_inspect_2025_data.py` | `archive/legacy_steps/01_inspect_2025_data.py` | Archive only; checked loading is in `code/io_utils.py` |
| `02_simple_elo_2025.py` | `archive/legacy_steps/02_simple_elo_2025.py` | `code/models/elo.py`, `code/pipelines/elo_pipeline.py` |
| `03_evaluate_elo_2025.py` | `archive/legacy_steps/03_evaluate_elo_2025.py` | `code/pipelines/elo_pipeline.py` |
| `04_elo_parameter_test_2025.py` | `archive/legacy_steps/04_elo_parameter_test_2025.py` | Archive-only parameter experiment |
| `05_summarise_elo_results_2025.py` | `archive/legacy_steps/05_summarise_elo_results_2025.py` | `code/pipelines/elo_pipeline.py` |
| `06_add_chinese_output_names.py` | `archive/legacy_steps/06_add_chinese_output_names.py` | Archive-only reporting utility |
| `06_convert_meeting_summary_to_word.py` | `archive/legacy_steps/06_convert_meeting_summary_to_word.py` | Archive-only reporting utility |
| `07_build_multiyear_match_dataset.py` | Git history | `code/data/build_matches.py` |
| `08_multiyear_elo.py` | `archive/legacy_steps/08_multiyear_elo.py` | `code/models/elo.py`, `code/pipelines/elo_pipeline.py` |
| `09_evaluate_multiyear_elo.py` | `archive/legacy_steps/09_evaluate_multiyear_elo.py` | `code/pipelines/elo_pipeline.py` |
| `10_burnin_experiment.py` | `archive/legacy_steps/10_burnin_experiment.py` | Archive-only experiment |
| `11_parameter_validation_multiyear.py` | `archive/legacy_steps/11_parameter_validation_multiyear.py` | Frozen settings in `code/config.py` |
| `12_rating_stability_analysis.py` | `archive/legacy_steps/12_rating_stability_analysis.py` | Archive-only diagnostic |
| `13_build_full_history_match_dataset.py` | `archive/legacy_steps/13_build_full_history_match_dataset.py` | `code/data/build_matches.py` |
| `14_elo_burnin_rating_list_stability.py` | `archive/legacy_steps/14_elo_burnin_rating_list_stability.py` | Archive-only diagnostic |
| `15_elo_single_year_rerun_convergence.py` | `archive/legacy_steps/15_elo_single_year_rerun_convergence.py` | Archive-only experiment |
| `16_elo_event_level_volatility.py` | `archive/legacy_steps/16_elo_event_level_volatility.py` | Archive-only diagnostic |
| `17_elo_baseline_decision_summary.py` | `archive/legacy_steps/17_elo_baseline_decision_summary.py` | Frozen settings in `code/config.py` |
| `18_glicko_core_sanity_check.py` | `archive/legacy_steps/18_glicko_core_sanity_check.py` | `code/models/glicko.py`, `tests/test_glicko_core.py` |
| `19_glicko_match_by_match_baseline.py` | `archive/legacy_steps/19_glicko_match_by_match_baseline.py` | `code/pipelines/glicko_pipeline.py` |
| `20_glicko_rating_period_sensitivity.py` | `archive/legacy_steps/20_glicko_rating_period_sensitivity.py` | Archive-only experiment |
| `21_create_elo_confirmation_answers_word.py` | `archive/legacy_steps/21_create_elo_confirmation_answers_word.py` | Archive-only reporting utility |
| `22_meeting5_experiment_matrix.py` | `archive/legacy_steps/22_meeting5_experiment_matrix.py` | Archive-only orchestration record |
| `23_glicko_implementation_validation.py` | `archive/legacy_steps/23_glicko_implementation_validation.py` | `tests/test_glicko_core.py`, `code/validation_utils.py` |
| `24_glicko_rd_inflation_sensitivity.py` | Git history | `code/pipelines/glicko_pipeline.py` |
| `25_glicko_rating_period_runtime_comparison.py` | `archive/legacy_steps/25_glicko_rating_period_runtime_comparison.py` | Archive-only experiment |
| `26_fair_elo_vs_glicko_comparison.py` | `archive/legacy_steps/26_fair_elo_vs_glicko_comparison.py` | `code/pipelines/comparison_pipeline.py` |
| `27_adaptive_k_elo_comparison.py` | `archive/legacy_steps/27_adaptive_k_elo_comparison.py` | Adaptive-K helpers in `code/models/elo.py`; negative result archived |
| `28_build_prematch_player_features.py` | `archive/legacy_steps/28_build_prematch_player_features.py` | Inputs consumed by `code/pipelines/comparison_pipeline.py` |
| `29_where_glicko_helps.py` | `archive/legacy_steps/29_where_glicko_helps.py` | `code/pipelines/comparison_pipeline.py` |
| `30_debut_initialisation_and_robustness_diagnostics.py` | `archive/legacy_steps/30_debut_initialisation_and_robustness_diagnostics.py` | `code/analysis/entry_diagnostics.py` |
| `31_methodological_corrections_and_meeting6_finalisation.py` | `archive/legacy_steps/31_methodological_corrections_and_meeting6_finalisation.py` | Superseded analysis retained for audit |
| `32_glicko_probability_orientation_audit.py` | `archive/legacy_steps/32_glicko_probability_orientation_audit.py` | `code/analysis/orientation.py` |
| `33_recompute_orientation_corrected_meeting6_results.py` | Git history | `code/pipelines/comparison_pipeline.py` |
| `34_early_game_analysis.py` | Git history | `code/analysis/early_game.py` |
| `35_early_game_mechanism_analysis.py` | `archive/legacy_steps/35_early_game_mechanism_analysis.py` | `code/analysis/early_game.py` |
| `36_glicko_initialisation_source_diagnostic.py` | `archive/legacy_steps/36_glicko_initialisation_source_diagnostic.py` | `code/analysis/entry_diagnostics.py` |
| `37_glicko_initial_rating_sensitivity.py` | `archive/legacy_steps/37_glicko_initial_rating_sensitivity.py` | Archive-only sensitivity experiment |
| `38_asymmetric_adaptive_k_elo.py` | `archive/legacy_steps/38_asymmetric_adaptive_k_elo.py` | Adaptive-K helpers in `code/models/elo.py`; negative result archived |
| `39_glicko_orientation_sensitivity_audit.py` | `archive/legacy_steps/39_glicko_orientation_sensitivity_audit.py` | `code/analysis/orientation.py`, `code/analysis/entry_diagnostics.py` |
| `40_finalize_orientation_reporting.py` | `archive/legacy_steps/40_finalize_orientation_reporting.py` | `code/pipelines/comparison_pipeline.py` |
| `41_burnin_entry_and_rating_drift_diagnostic.py` | Git history | `code/analysis/rating_drift.py` |
| `42_prematch_entry_scale_and_crossfile_audit.py` | Git history | `code/analysis/entry_diagnostics.py` |

Additional original files:

| Original file | Current location |
|---|---|
| `glicko_core.py` | Compatibility wrapper at `code/glicko_core.py`; implementation at `code/models/glicko.py` |
| `load_croquet_data.py` | `code/data/download.py` |
| `meeting7_final_code_audit.py` | `archive/legacy_steps/meeting7_final_code_audit.py` |

