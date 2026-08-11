# Legacy research steps

This directory preserves the numbered Python scripts in their original
development order. They record how the analysis evolved across meetings and
remain useful for scientific audit, provenance, and interpretation of older
output filenames.

Archived scripts are preserved for auditability and source reference. They
retain their original filenames and historical implementation, but they are
not guaranteed to run directly from their archived locations because some
scripts use relative paths, dynamic imports, or historical output locations.
They must not be treated as active dependencies.

Use the active modules under `code/` and the commands documented in
`code/README.md` as the supported execution interface. `CODE_MAP.md` maps the
chronological script groups to their archive locations and, where applicable,
their active replacements.

No legacy script was deleted during the refactor. Thirty-eight root-level
historical scripts were moved here, and the exact original Step 33 script was
additionally restored from the safety tag for source reference. Steps 7, 24,
33, 34, 41, and 42 were promoted to active modules while retaining their Git
history.

The safety tag `pre-code-cleanup-2026` preserves the original runnable
directory layout. A user who needs that exact historical layout should check
out the tag rather than execute scripts from this archive. The archived
scripts have not all been individually rerun after being moved.
