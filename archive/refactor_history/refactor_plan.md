# Code Layout Refactor Plan

## Scope and protected baseline

- Original active-folder Python files: 46.
- Protected commit: `1538c7788e0d254d3d58f6a172ba364057063701`.
- Permanent annotated tag: `pre-code-cleanup-2026`.
- Working branch: `refactor/clean-code-layout`.
- Known untracked compact files under `outputs/` are outside the refactor and must not be staged or deleted.
- Scientific formulas, configurations, ordering, samples, probability conventions, and tracked golden results are frozen.

## Audit method

Every original Python file was parsed with the Python AST and inspected for:

- normal and dynamic imports;
- top-level functions and classes;
- local module dependencies;
- input and output filename constants;
- numbered-script references;
- `importlib.util.spec_from_file_location` calls;
- subprocess calls;
- references in tracked Markdown, CSV manifests, and the repository README;
- duplicate hashes and backup-like filenames;
- private absolute filesystem paths.

The audit found no byte-identical Python files, no backup-like Python filenames, and no private absolute Windows paths in Python source. The strongest structural similarity is between Steps 26 and 27, but Step 27 contains unique adaptive-K logic and is not deletable. Steps 02 and 08 share Elo pipeline structure but cover different historical stages. No permanent deletion is currently justified.

## Critical dependency findings

1. Step 13 dynamically loads Step 07.
2. Steps 41 and 42 dynamically load Step 24.
3. `glicko_core.py` is imported by nine historical Glicko and validation scripts.
4. Step 30 reads the Step 26 source text.
5. The final Meeting 7 audit references Steps 27 and 32-40 by filename.
6. Tracked result summaries cite numbered filenames and must remain unchanged as frozen evidence.

These findings require a compatibility wrapper for `glicko_core.py`, a complete `CODE_MAP.md`, and preservation of the chronological scripts in `archive/legacy_steps/`.

## Planned active structure

The active package will contain:

- immutable scientific configuration in `code/config.py`;
- shared path and CSV helpers in `code/io_utils.py`;
- structured error and warning checks in `code/validation_utils.py`;
- canonical Elo equations in `code/models/elo.py`;
- the single validated Glicko implementation in `code/models/glicko.py`;
- a temporary `code/glicko_core.py` compatibility wrapper;
- annual data acquisition and the canonical checked-match builder in `code/data/`;
- frozen Elo, Glicko, and orientation-corrected comparison pipelines in `code/pipelines/`;
- focal orientation, early-game, rating-drift, and strict entry diagnostics in `code/analysis/`;
- a small argparse command interface in `code/cli.py`.

Domain modules may remain substantial where splitting them would alter ordering or validation behaviour. Unrelated experiments will not be merged merely to reduce file count.

## Historical preservation

Superseded, meeting-specific, and one-off numbered scripts will move to `archive/legacy_steps/` with their original filenames. Scripts selected as the source for canonical active modules will follow Git rename history into their new locations; their exact pre-refactor versions remain recoverable through the permanent safety tag.

No file will be permanently deleted unless a later audit proves exact duplication or complete supersession with no unique logic. The deletion-candidate table is intentionally empty at the start of implementation.

## Validation strategy

1. Unit-test Elo and Glicko equations against frozen calculations.
2. Test canonical player-A and focal-player orientation independently of outcomes.
3. Test exact burn-in cohort and 2025 entry counts from tracked golden tables.
4. Test Step 33, Step 34, and full-history key reconciliation.
5. Test the frozen 2025 first-appearance headline metrics with strict tolerances.
6. Run active entry diagnostics into `outputs/refactor_validation/`.
7. Compare compact validation outputs with tracked references without overwriting them.
8. Stop if any error-level check or substantive numerical comparison fails.

## Commit sequence

1. Add modular core implementations and validation tests.
2. Consolidate data, model, analysis, and CLI entry points.
3. Archive legacy scripts and add complete mapping documentation.
4. Update repository documentation and final regression evidence.

Only the refactor branch will be pushed. The branch will not be merged automatically.
