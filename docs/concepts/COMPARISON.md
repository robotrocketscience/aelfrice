# Comparison: memory system vs. retrieval index

[Leonard Lin's review of agentic-memory implementations](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md) states the bar directly:

> The biggest differentiator is not "vector DB vs SQLite" — it's **write correctness and governance**: provenance / audit trail, write gates / confirmation, conflict handling, reversibility (inspect / edit / delete).

By that bar, "a vector store with a similarity query" isn't a memory system.
It's a search index. A memory system has to answer *who wrote this, when,
through what ingress, what supersedes it, and how do I take it back*. This page
describes how aelfrice meets each pillar.

## The four pillars

| Lin's pillar | What it means | How aelfrice does it |
|---|---|---|
| **Provenance / audit trail** | Every row traces back to the action that wrote it: who, when, and through what ingress channel. | Every belief carries an `origin` column, and the `ORIGINS` set holds eight validated tier values. `/aelf:wonder` phantoms use `speculative`, which gets written at runtime but stays out of the validated `ORIGINS` set ([`src/aelfrice/models.py`](../../src/aelfrice/models.py)). The `scope` column (`project` / `global` / `shared:<name>`) tags federation visibility, and the append-only `ingest_log` records every raw input, so you can tear the DB down and rebuild the beliefs from that log alone. **Edges aren't covered:** every edge is written outside the log, so a rebuild restores the beliefs but not the typed graph described in the row below ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)). Open the file in any SQLite browser. Nothing is hidden. |
| **Write gates / confirmation** | Persistence isn't unconditional: some writes need explicit approval, and no path launders an external-origin claim into ground truth. | `aelf lock` is the only path to user-asserted ground truth. `aelf confirm` raises the `(α, β)` posterior but can't change `origin`. Phantom promotion has two explicit surfaces: `aelf promote <id>` is the explicit path, and `aelf lock <text>` is the implicit auto-promote path, which fires on a content-hash exact match or on a normalized-token Jaccard ≥ 0.9. Both surfaces write audit rows. Feedback accumulates rather than overwrites, so one harmful click moves the mean without erasing a belief. |
| **Conflict handling** | Competing claims about the same thing show up as a conflict instead of one quietly overwriting the other. | `CONTRADICTS`, `SUPERSEDES`, and `RESOLVES` are edge types in their own right, so a disagreement becomes a graph relation rather than a vanished row. `/aelf:reason` emits a typed `VERDICT` (`SUFFICIENT` / `PARTIAL` / `UNCERTAIN` / `INSUFFICIENT` / `CONTRADICTORY`) alongside typed `IMPASSES` (`TIE` / `GAP` / `CONSTRAINT_FAILURE` / `NO_CHANGE`), so a downstream agent can act on the disagreement. Per-scope version vectors preserve causal ordering across worktrees and federation peers. |
| **Reversibility (inspect / edit / delete)** | Mutations stay auditable and partly undoable, and you keep control of your own memories. | `aelf delete`, `aelf unlock`, `aelf promote --to-scope`, and `aelf feedback` all write audit rows, and the `ingest_log` is append-only and replay-capable. Read-only federation lets a project surface peer beliefs through `knowledge_deps.json` without taking ownership, and a foreign-id mutation raises `ForeignBeliefError` at the API surface. At the top level, `aelf uninstall --archive backup.aenc` encrypts and removes the data, `--purge` wipes it, and `--keep-db` leaves it untouched. No vendor lock-in. |

## Compared with CLAUDE.md and hand-maintained files

The standard workaround for "agent keeps forgetting" is more files: `STATE.md`,
`DECISIONS.md`, and a `CLAUDE.md` with cross-references to runbooks. Every
cross-reference rests on an assumption: that the agent reads the file, finds
the correct section, and follows that section. That assumption fails in
predictable ways:

- The agent reads the rule and runs `git push` anyway.
- Cross-references break without warning after compaction.
- State files go stale as soon as someone forgets to update them.

Each new failure mode adds another file.

aelfrice replaces the chain with a mechanism: the hook injects the matched
beliefs into the prompt before the model sees your message. The injection isn't
voluntary, and the agent can't skip it.

| Manual approach | What breaks | What aelfrice does |
|---|---|---|
| Rules in `CLAUDE.md` | The agent reads them, then doesn't follow them. | The hook injects the matched beliefs per prompt, not per session |
| Cross-references | The agent skips a section or reads the wrong one. | The hook injects the matched beliefs directly |
| Hand-maintained state files | One missed update breaks the chain. | The SQLite DB is the state, so there's no manual sync |

## Related reading

- [Design philosophy](PHILOSOPHY.md) — the principles that lock these choices in.
- [Architecture overview](ARCHITECTURE.md) — system shape, retrieval lanes, and the edge model.
- [Known limitations](../user/LIMITATIONS.md) — what the partial ranking does and doesn't cover.
