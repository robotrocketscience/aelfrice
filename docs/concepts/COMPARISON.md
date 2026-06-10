# Comparison: memory system vs. retrieval index

[Leonard Lin's review of agentic-memory implementations](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md) frames the bar bluntly:

> The biggest differentiator is not "vector DB vs SQLite" — it's **write correctness and governance**: provenance / audit trail, write gates / confirmation, conflict handling, reversibility (inspect / edit / delete).

By that bar, "a vector store with a similarity query" is not a memory system — it is a search index. A memory system has to answer *who wrote this, when, via what ingress, what supersedes it, and how do I take it back*. This page lays out how aelfrice meets each pillar.

## The four pillars

| Lin's pillar | What it means | aelfrice mechanism |
|---|---|---|
| **Provenance / audit trail** | Every row traces back to the action that wrote it: who, when, via what ingress channel. | `origin` column on every belief — nine tier values including `speculative` for `/aelf:wonder` phantoms ([`src/aelfrice/models.py`](../../src/aelfrice/models.py)). The `scope` column (`project` / `global` / `shared:<name>`) tags federation visibility. Append-only `ingest_log` records every raw input — tear the DB down and rebuild from this log alone. Open the file in any SQLite browser; nothing hidden. |
| **Write gates / confirmation** | Persistence is not unconditional. Some writes need explicit approval; external-origin claims cannot be laundered into ground truth. | `aelf lock` is the only path to user-asserted ground truth. `aelf confirm` bumps the `(α, β)` posterior but cannot flip `origin`. Phantom promotion has two explicit surfaces: `aelf promote <id>` for the explicit path, and `aelf lock <text>` with content-hash exact match or normalized-token Jaccard ≥ 0.9 for the implicit auto-promote — both write audit rows. Feedback accumulates rather than overwrites: one harmful click nudges the mean, it doesn't erase a belief. |
| **Conflict handling** | Competing claims about the same thing are surfaced, not silently overwritten. | First-class edge types `CONTRADICTS`, `SUPERSEDES`, `RESOLVES` — disagreement is a graph relation, not a vanished row. `/aelf:reason` emits a typed `VERDICT` (`SUFFICIENT` / `PARTIAL` / `UNCERTAIN` / `INSUFFICIENT` / `CONTRADICTORY`) plus typed `IMPASSES` (`TIE` / `GAP` / `CONSTRAINT_FAILURE` / `NO_CHANGE`) so a downstream agent can act on the disagreement. Per-scope version vectors preserve causal ordering across worktrees and federation peers. |
| **Reversibility (inspect / edit / delete)** | Mutations remain auditable and partially undoable. The user is the boss of their memories. | `aelf delete`, `aelf unlock`, `aelf promote --to-scope`, and `aelf feedback` all write audit rows; the `ingest_log` is append-only and replay-capable. Read-only federation lets a project surface peer beliefs via `knowledge_deps.json` without taking ownership — foreign-id mutations raise `ForeignBeliefError` at the API surface. Top level: `aelf uninstall --archive backup.aenc` encrypts and removes; `--purge` wipes; `--keep-db` leaves data untouched. No vendor lock-in. |

## And vs. CLAUDE.md / hand-maintained files

The standard workaround for "agent keeps forgetting" is more files: `STATE.md`, `DECISIONS.md`, a `CLAUDE.md` with cross-references to runbooks. Every cross-reference is a bet that the agent will read the file, find the right section, and follow what it says. The failure modes are predictable: the agent reads the rule and runs `git push` anyway; cross-references break silently after compaction; state files rot the moment someone forgets to update them. Each new failure mode begets another file.

aelfrice replaces the chain with a mechanism. Matched beliefs are in the prompt, prepended by the hook before the model sees your message. Not voluntary; nothing the agent can skip.

| Manual approach | What breaks | aelfrice |
|---|---|---|
| Rules in `CLAUDE.md` | Agent reads them; doesn't follow them | Injected per-prompt, not per-session |
| Cross-references | Agent skips or reads the wrong section | Matched beliefs injected directly |
| Hand-maintained state files | One missed update breaks the chain | State is the SQLite DB; no manual sync |

## Related reading

- [PHILOSOPHY.md](PHILOSOPHY.md) — design principles that lock these choices in.
- [ARCHITECTURE.md](ARCHITECTURE.md) — system shape, retrieval lanes, edge model.
- [LIMITATIONS.md](../user/LIMITATIONS.md) — what the partial ranking does and doesn't cover.
