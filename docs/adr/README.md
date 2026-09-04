# Architecture Decision Records

Each architecture decision record (ADR) documents one technical decision: the context, the choice you made, the alternatives you considered, and the consequences. ADRs are append-only. If a later decision supersedes an ADR, add a new ADR that links back to the one it replaces instead of editing the original.

## Index

- [0001 — Record architecture decisions](0001-record-architecture-decisions.md)
- [0002 — Two-repo physical separation for public/private boundary](0002-two-repo-physical-separation.md)
- [0003 — `project_context` holds repo identity](0003-project-context-repo-identity-convention.md)

## Format

ADRs follow a lightweight Nygard-style template. See [the ADR template](template.md). Numbering is monotonic. File names use the pattern `NNNN-kebab-case-title.md`.

## When to write one

Write an ADR when the decision:

- Affects how code is structured across multiple modules.
- Constrains future work, for example by picking a dependency, a storage format, or an API contract.
- Was non-obvious, that is, you considered alternatives and would forget the reasoning in six months.

Don't write an ADR for routine bugfixes, dependency bumps, or local refactors.
