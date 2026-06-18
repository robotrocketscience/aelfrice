# 0003 — `project_context` holds repo identity

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** @robotrocketscience

## Context

`beliefs.project_context` was added in #858 (v3.2) as a within-repo retrieval-scope axis: two worktrees of the same repo share one DB (via `db_path()`'s git-common-dir routing), and the column was meant to let them see different *slices* of that shared store. The retrieval filter (`hook._filter_by_project_context`) reads it, and `db_paths.active_project_context()` resolves the active value from `AELFRICE_PROJECT_CONTEXT`.

In practice the column was dead weight (#970): no writer ever set a non-empty value, so 100% of rows were `''` and the filter was a no-op. Worse, `aelf migrate` dropped the column entirely, collapsing two repos' beliefs into one mutually-visible pool — the column could not provide defense-in-depth for the one case (a deliberate merge) where physical DB separation does not.

A convention had to be chosen, because the candidate meanings conflict at the data layer (single-string equality match — a row holds one or the other):

1. **Repo identity** — doubles as merge/federation provenance, but forecloses fine-grained within-repo slicing on this axis.
2. **Within-repo slice** — keeps #858's original intent, but needs a *separate* provenance column for the merge case.

## Decision

`project_context` holds **repo identity**: `<repo-root-basename>-<8 hex blake2b of the absolute git-common-dir>`, derived in `db_paths` from the same git-common-dir `db_path()` already keys on. Decisions 2–4 from #970:

- **Derivation (decision 2):** reuse the git-common-dir identity `db_path()` computes; no new identity source. Two worktrees of one repo share an identity.
- **Locked / global rows (decision 3):** L0 user-locked truths and federation-shared (`scope != 'project'`) beliefs stay `''` (always-visible). A repo-tag backfill must not scope a user's global preferences to a single repo, where an `aelf migrate` would then hide them.
- **Active-context resolution (decision 4):** the resolver default stays **env-driven** — `active_project_context()` returns `''` (no filter) unless `AELFRICE_PROJECT_CONTEXT` is set. Scoping is **opt-in**: export `AELFRICE_PROJECT_CONTEXT="$(repo identity)"` to activate it. This commit populates the column and makes it migrate-safe but does **not** silently reverse #858's documented "unset = no filter" contract.

Mechanics: `insert_belief` stamps the store's repo identity onto new eligible rows; a one-shot idempotent, reversible backfill stamps pre-existing eligible rows on first open; `aelf migrate` preserves an already-stamped value and stamps source-`''` eligible rows with the *source* repo's identity so provenance survives the merge.

## Alternatives considered

- **Within-repo slice (keep #858 intent).** Rejected: nothing populated it for nine months, and it provides no merge/federation provenance, which was the concrete gap. An explicit `project_context` value on a belief is still honoured verbatim, so per-slice tagging remains possible by opting in; it is just not the default.
- **Flip the resolver default to repo identity (auto-consult, decision 4 = yes).** Rejected for now: it reverses the documented "unset = no filter" contract and changes retrieval behaviour for every store (a no-op for a single repo, but a silent semantics change). Left as a follow-up that the operator can ratify; the column and migrate provenance land first.
- **A separate provenance column (e.g. reuse `local_scope_id`).** `local_scope_id` (#204) is an opaque per-DB UUID for version-vector provenance, not a stable, legible, cross-repo retrieval-scope key. Keeping provenance on `project_context` avoids a second axis for the chosen convention.

## Consequences

- **Positive:** the column is populated and consulted (when opted in); `aelf migrate` no longer collapses provenance; merged stores can attribute each belief to its origin repo.
- **Negative:** fine-grained within-repo slicing is no longer the default meaning of the column (only available via explicit per-belief values). Scoping is opt-in, so the defense-in-depth filter is inert until `AELFRICE_PROJECT_CONTEXT` is set.
- **Neutral:** a new `schema_meta` backfill marker (`project_context_backfill_complete`) and a `<repo-root-basename>-<hash>` token format now appear in stores. The home-dir fallback store carries no identity (`''`), so non-repo usage is unchanged.
