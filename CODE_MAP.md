# Code and legacy mapping

The supported implementation is under `code/`. Chronological scripts under
`archive/legacy_steps/` are retained for provenance and are not active
dependencies. Archived scripts may rely on historical paths or excluded
intermediate data and are not guaranteed to run from the archive.

Earlier layouts remain available through Git history and the repository's
historical tags when exact development-era paths are required.

## Current supported implementation

| Responsibility | Current implementation | Supported command or check |
|---|---|---|
| Frozen parameters and expected values | `code/config.py` | imported by active modules and tests |
| Canonical Elo equations | `code/models/elo.py` | `python -m code.cli run-elo` |
| Canonical Glicko-1 equations | `code/models/glicko.py` | `python -m code.cli run-glicko` |
| Checked historical game construction | `code/data/build_matches.py` | `python -m code.cli build-data` |
| Source-data retrieval helper | `code/data/download.py` | run directly after confirming data access |
| Elo workflow | `code/pipelines/elo_pipeline.py` | `python -m code.cli run-elo` |
| Glicko workflow | `code/pipelines/glicko_pipeline.py` | `python -m code.cli run-glicko` |
| Orientation-corrected comparison | `code/pipelines/comparison_pipeline.py` | `python -m code.cli compare-models` |
| Probability-orientation diagnostics | `code/analysis/orientation.py` | covered by `tests/test_orientation.py` |
| Early-game analysis | `code/analysis/early_game.py` | `python -m code.cli early-game` |
| Recorded-entry and prematch-scale analysis | `code/analysis/entry_diagnostics.py` | `python -m code.cli entry-diagnostics` |
| Rating-drift support | `code/analysis/rating_drift.py` | used by entry diagnostics |
| Lightweight validation | `code/cli.py`, `tests/` | `python -m code.cli validate`; unit tests |

`code/glicko_core.py` remains only for import compatibility. There is one
active Glicko-1 formula implementation: `code/models/glicko.py`.

## Chronological-script mapping

| Historical step or file | Historical purpose | Current status / replacement |
|---|---|---|
| Steps 01-05 | Initial inspection, single-year Elo, evaluation, and scientific summaries | Audit-only scripts in `archive/legacy_steps/`; checked loading is consolidated in `code/io_utils.py`. |
| Step 07 | Multi-year checked data construction | Promoted into `code/data/build_matches.py`. |
| Steps 08-17 | Multi-year Elo, validation, burn-in, stability, and baseline decisions | Core workflow consolidated in `code/pipelines/elo_pipeline.py`; frozen settings are in `code/config.py`; distinct experiments remain audit-only. |
| Steps 18-21 and 23 | Glicko equation checks, baselines, period sensitivity, and validation | Equations and checks consolidated in `code/models/glicko.py` and `tests/test_glicko_core.py`; historical scientific experiments and checks remain audit-only. |
| Step 24 | Low-inflation Glicko workflow | Promoted into `code/pipelines/glicko_pipeline.py`. |
| Step 25 | Rating-period runtime comparison | Audit-only in `archive/legacy_steps/`. |
| Steps 26-31 | Fair comparison, adaptive-K, prematch features, and pre-correction diagnostics | Superseded historical sequence retained in the archive; applicable helpers are consolidated in the active pipelines and analyses. |
| Step 32 | Probability-orientation audit | Historical script archived; active definitions are in `code/analysis/orientation.py` and its tests. |
| Step 33 | Formal orientation-corrected comparison | Exact script archived; supported replacement is `code/pipelines/comparison_pipeline.py`. |
| Step 34 | Early-game analysis | Promoted into `code/analysis/early_game.py`. |
| Steps 35-37 | Early-game mechanism and initialisation diagnostics | Historical scripts archived; supported concepts are consolidated in `code/analysis/early_game.py` and `code/analysis/entry_diagnostics.py`. |
| Steps 38-40 | Adaptive-K and orientation sensitivity/reporting | Audit-only scripts; relevant helpers are in `code/models/elo.py`, `code/analysis/orientation.py`, and `code/pipelines/comparison_pipeline.py`. |
| Step 41 | Burn-in, recorded-entry, and rating-scale diagnostics | Promoted into `code/analysis/rating_drift.py` and the entry-diagnostics workflow. |
| Step 42 | Strict prematch recorded-entry and cross-file audit | Promoted into `code/analysis/entry_diagnostics.py`. |
| `glicko_core.py` | Reusable Glicko equations | Canonical code is `code/models/glicko.py`; compatibility wrapper retained. |
| `load_croquet_data.py` | Annual source-file download | Promoted into `code/data/download.py`; raw data remain untracked. |
| `meeting7_final_code_audit.py` | Historical code/output audit | Audit-only in `archive/legacy_steps/`; current automated checks are under `tests/`. |

Git history records the promotions and moves when file-level provenance is
required.

## Output provenance

Controlled dissertation evidence remains under
`outputs/dissertation_evidence/chapter4/` and
`outputs/dissertation_evidence/chapter5/`. `archive/research_outputs/` retains
only the compact archived inputs required by current validation/tests and the
existing source artefacts referenced by the evidence manifests.

Evidence manifests intentionally retain original generation paths such as
`outputs/meeting6/...`. Where the corresponding compact source is retained,
replace the leading `outputs/` with `archive/research_outputs/`. Git history
and the frozen submission tag preserve the complete published layout. No
controlled evidence values or hashes were changed by this cleanup.
