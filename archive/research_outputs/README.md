# Retained scientific provenance outputs

This directory is a curated set of compact scientific source and validation
artefacts. It is not a general archive of meeting outputs and is not the
recommended dissertation evidence entry point.

Retained files satisfy at least one of two criteria:

- current lightweight validation or regression tests read the file; or
- a controlled Chapter 4/5 evidence or figure manifest identifies the file as
  an existing archived source artefact.

For a retained source, a historical path beginning `outputs/` maps by replacing
that prefix with `archive/research_outputs/`. Manifest paths without a retained
compact source remain historical provenance and can be resolved through Git
history or the frozen submission tag. The controlled evidence packages under
`outputs/dissertation_evidence/` remain the authoritative dissertation source.

Full analysis workflows do not use this directory as an output destination.
New generated results should remain under the ignored local `outputs/`
workspace unless intentionally reviewed for controlled-evidence inclusion.
