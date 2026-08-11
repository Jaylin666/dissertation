# Legacy research steps

This directory preserves scientifically relevant numbered Python scripts in
their original development order. They record how the analysis evolved and
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

Steps 7, 24, 33, 34, 41, and 42 were promoted to active modules while
retaining their Git history. Scientific experiments, equation checks, model
validation, and evidence-generating scripts remain here. Helpers whose sole
purpose was Word export, translated filename copying, or experiment-plan
generation are omitted from the final technical deliverable.

The safety tag `pre-code-cleanup-2026` preserves the original runnable
directory layout. A user who needs that exact historical layout should check
out the tag rather than execute scripts from this archive. The archived
scripts have not all been individually rerun after being moved.
