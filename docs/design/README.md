# Design specs

This directory holds internal design notes and per-feature specs. They're for contributors who work on a specific subsystem, not user documentation.

Filename conventions:

- `feature-*.md` — feature specs, usually paired with a tracking issue.
- `v2_*.md` / `v3_*.md` — version-scoped design proposals (some shipped, some superseded).
- everything else — implementation notes for a specific module or behavior (for example, `bayesian_ranking.md` and `bfs_multihop.md`).

A spec might be out of date relative to the shipped code. Treat the source as the truth and the spec as the historical intent.
