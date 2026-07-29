# Legacy research steps

This directory preserves the numbered Python scripts in their original
development order. They record how the analysis evolved across meetings and
remain useful for scientific audit, provenance, and interpretation of older
output filenames.

These scripts are not the recommended entry points for new runs. Some expect
large generated inputs, historical output directories, or another numbered
script to be present in its former location. Their exact pre-refactor state is
also protected by the Git tag `pre-code-cleanup-2026`.

Use the active modules under `code/` and the commands documented in
`code/README.md` for current work. `CODE_MAP.md` maps every numbered script to
its archive location and, where applicable, its active replacement.

No legacy script was deleted during the refactor. Thirty-eight root-level
historical scripts were moved here; Steps 7, 24, 33, 34, 41, and 42 were
promoted to active modules while retaining their history through Git renames.

