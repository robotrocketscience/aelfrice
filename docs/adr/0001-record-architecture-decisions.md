# 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-04-30
- **Deciders:** @robotrocketscience

## Context

aelfrice is a solo-maintained library. Architectural decisions accumulate informally in commit messages, PR descriptions, and chat history. Six months later the *why* of a non-obvious choice is hard to reconstruct, and the only reviewer of past decisions is future-me.

## Decision

Adopt lightweight Architecture Decision Records (Nygard style) under `docs/adr/`. One markdown file per decision, monotonically numbered, append-only. No tooling, no published site — plain markdown read directly on GitHub.

## Alternatives considered

- **Keep decisions in commit messages and PR bodies.** Status quo. Rejected: rationale lives next to the diff, but is hard to discover later without knowing which PR to read.
- **log4brains with a published GitHub Pages site.** Rejected for now: adds a build step and Pages config for a single-author project. The CLI's templating value is small at this volume; revisit if/when the contributor count grows.
- **Notion / external doc site.** Rejected: ADRs should live with the code so they version together and survive tooling changes.

## Consequences

- **Positive:** Future-me has an audit trail for non-obvious choices. New contributors (if any) can read `docs/adr/` to understand why the codebase looks the way it does. Searchable in `git grep`.
- **Negative:** Discipline overhead — writing an ADR is friction at decision time. Mitigated by scoping ADRs to *non-obvious* decisions only, not every change.
- **Neutral:** Adds a `docs/adr/` directory to the repo.
