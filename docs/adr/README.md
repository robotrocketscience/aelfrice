# Architecture Decision Records

Each ADR documents one technical decision: context, the choice made, alternatives considered, and consequences. ADRs are append-only; if a decision is later superseded, add a new ADR that links back to the one it replaces rather than editing the original.

## Index

- [0001 — Record architecture decisions](0001-record-architecture-decisions.md)
- [0002 — Two-repo physical separation for public/private boundary](0002-two-repo-physical-separation.md)

## Format

ADRs follow a lightweight Nygard-style template (see [template.md](template.md)). Numbering is monotonic. File names are `NNNN-kebab-case-title.md`.

## When to write one

Write an ADR when the decision:

- Affects how code is structured across multiple modules.
- Constrains future work (e.g. picks a dependency, a storage format, an API contract).
- Was non-obvious — i.e. you considered alternatives and would forget the reasoning in six months.

Do **not** write an ADR for routine bugfixes, dependency bumps, or local refactors.
