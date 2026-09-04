# Comparison: memory system vs. retrieval index

[Leonard Lin's review of agentic-memory implementations](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md) states the bar directly:

> The biggest differentiator is not "vector DB vs SQLite" — it's **write correctness and governance**: provenance / audit trail, write gates / confirmation, conflict handling, reversibility (inspect / edit / delete).

By that bar, "a vector store with a similarity query" isn't a memory system.
It's a search index. A memory system has to answer *who wrote this, when,
through what ingress, what supersedes it, and how do I take it back*. This page
describes how aelfrice meets each pillar.

## The four pillars

| Lin's pillar | What it means | aelfrice mechanism |
|---|---|---|
| **Provenance / audit trail** | Every row traces back to the action that wrote it: who, when, via what ingress channel. | Every belief has an `origin` column. The `ORIGINS` set holds eight validated tier values. `/aelf:wonder` phantoms use `speculative`, which is written at runtime but excluded from the validated `ORIGINS` set ([`src/aelfrice/models.py`](../../src/aelfrice/models.py)). The `scope` column (`project` / `global` / `shared:<name>`) tags federation visibility. The append-only `ingest_log` records every raw input. You can tear the DB down and rebuild the beliefs from this log alone. **Edges are not covered:** every edge is written outside the log. A rebuild therefore restores the beliefs but not the typed graph in the row below ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)). Open the file in any SQLite browser. Nothing is hidden. |
| **Write gates / confirmation** | Persistence is not unconditional. Some writes need explicit approval. No path launders an external-origin claim into ground truth. | `aelf lock` is the only path to user-asserted ground truth. `aelf confirm` raises the `(α, β)` posterior, but it cannot change `origin`. Phantom promotion has two explicit surfaces. `aelf promote <id>` is the explicit path. `aelf lock <text>` is the implicit auto-promote path, and it fires on a content-hash exact match or on a normalized-token Jaccard ≥ 0.9. Both surfaces write audit rows. Feedback accumulates rather than overwrites. One harmful click moves the mean. It does not erase a belief. |
| **Conflict handling** | Competing claims about the same thing are surfaced, not overwritten without warning. | `CONTRADICTS`, `SUPERSEDES`, and `RESOLVES` are edge types in their own right. A disagreement is a graph relation, not a vanished row. `/aelf:reason` emits a typed `VERDICT` (`SUFFICIENT` / `PARTIAL` / `UNCERTAIN` / `INSUFFICIENT` / `CONTRADICTORY`). It also emits typed `IMPASSES` (`TIE` / `GAP` / `CONSTRAINT_FAILURE` / `NO_CHANGE`). A downstream agent can therefore act on the disagreement. Per-scope version vectors preserve causal ordering across worktrees and federation peers. |
| **Reversibility (inspect / edit / delete)** | Mutations remain auditable and partially undoable. The user controls their own memories. | `aelf delete`, `aelf unlock`, `aelf promote --to-scope`, and `aelf feedback` all write audit rows. The `ingest_log` is append-only and replay-capable. Read-only federation lets a project surface peer beliefs through `knowledge_deps.json` without taking ownership. A foreign-id mutation raises `ForeignBeliefError` at the API surface. At the top level, `aelf uninstall --archive backup.aenc` encrypts and removes the data. `--purge` wipes the data. `--keep-db` leaves the data untouched. There is no vendor lock-in. |

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

| Manual approach | What breaks | aelfrice |
|---|---|---|
| Rules in `CLAUDE.md` | The agent reads them. The agent does not follow them. | The hook injects the matched beliefs per prompt, not per session |
| Cross-references | The agent skips a section, or reads the wrong section. | The hook injects the matched beliefs directly |
| Hand-maintained state files | One missed update breaks the chain | The state is the SQLite DB. There is no manual sync |

## Related reading

- [PHILOSOPHY.md](PHILOSOPHY.md) — the design principles that lock these choices in.
- [ARCHITECTURE.md](ARCHITECTURE.md) — system shape, retrieval lanes, and the edge model.
- [LIMITATIONS.md](../user/LIMITATIONS.md) — what the partial ranking does and doesn't cover.
