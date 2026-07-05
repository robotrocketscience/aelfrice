# claude-memory write-through mirror

**Status:** shipped (#985); **default-on post-consent** since #1089. Hook
installed by `aelf setup` (`--no-claude-memory-mirror` to skip). The first
`aelf setup` for a project that has a claude-memory dir runs a one-shot
reconcile (the consent event) and turns the mirror on; opt out with
`AELFRICE_MIRROR_CLAUDE_MEMORY=0` or `[memory] mirror_claude_memory=false`.
**Dependencies:** stdlib only (`tomllib`, `re`) for the parser; the reconcile
sweep (`claude_memory_reconcile.py`) additionally uses the derivation/store
path. Reuses
[`claude_memory.py`](../../src/aelfrice/claude_memory.py) (already the
parser for `/aelf:audit-claude-memory`) and the `insert_or_corroborate`
idempotency path.
**Risk:** low. One-way mirror; never authoritative over the memory files;
PostToolUse hook returns silently on every failure path.

## Summary

The upstream auto-memory tool ships on by default, writing one-fact
markdown files into `~/.claude/projects/<encoded>/memory/` plus a
`MEMORY.md` index. Without a write-through the belief graph and the
claude-memory store **drift**: a fact written to memory is not in
the graph unless separately `aelf lock`'d by hand.
`/aelf:audit-claude-memory` can *detect* the divergence after the fact;
this hook keeps the two in sync at write time.

```
Write/Edit a file under …/.claude/projects/<x>/memory/<name>.md
        ↓
  PostToolUse hook fires (matcher = Write|Edit|MultiEdit)
        ↓  is_memory_fact_path(path)?      (cheap reject; skips MEMORY.md)
        ↓  is_mirror_enabled()?            (opt-in flag; default OFF)
        ↓  parse_memory_file(disk text)    (frontmatter + body)
        ↓  derive() + insert_or_corroborate
  belief lands in the per-project brain.db
```

It is a **one-way mirror** (claude-memory → aelfrice graph). aelfrice
never writes back to the memory files.

## Enablement (default-on **post-consent**, #1089)

Resolved by `claude_memory.is_mirror_enabled()`, precedence first-wins:

1. `AELFRICE_MIRROR_CLAUDE_MEMORY` env var (truthy/falsy normalised).
2. Explicit kwarg from the caller.
3. `[memory] mirror_claude_memory` in `.aelfrice.toml`.
4. **Consent sentinel present → True** (#1089).
5. Default **False**.

Originally the mirror was purely opt-in (#985). #1089 makes it **default-on
after a one-time consent event**, keeping the explicit env/TOML opt-out as the
override. The consent event is the first-project reconcile (below): once it
writes the per-project sentinel, `is_mirror_enabled()` returns `True` for that
project without a flag flip. An explicit env/TOML value still wins first, so a
user who sets `AELFRICE_MIRROR_CLAUDE_MEMORY=0` stays off regardless.

The hook *entry* installs by default (unchanged); for any non-memory
`Write`/`Edit` it returns after a single path-shape check and never consults
the flag or imports the store.

## Full-set reconcile + the consent event (#1089)

The write-event mirror only sees in-session writes, so pre-existing fact files
and any write the hook missed never reached the graph.
`claude_memory_reconcile.reconcile_claude_memory(store, memory_dir)` closes
that hole: it sweeps `<memory_dir>/*.md` (skipping `MEMORY.md`) through the
same `ingest_memory_text` the hook uses, so the sweep and the hook share one
frontmatter → origin/prior mapping. Idempotent — a re-run corroborates, not
duplicates.

`maybe_reconcile_claude_memory` wraps it in the G4 temporal-spine one-shot
pattern (#1064): sentinel-gated per project, safe to call from every
`aelf setup`, deferring (without writing the sentinel) when the mirror is
explicitly opted out. `aelf setup` calls it **only when the project already
has a claude-memory dir** — consent is deferred until there is memory to sync,
so a fresh host that hasn't adopted the auto-memory tool is untouched. The
sentinel lives *beside the belief store* (`db_path().parent`) so it is
per-project by construction and is removed with the store on
uninstall/rebuild (a fresh store then re-consents).

Surfaces: `aelf setup` announces the consent event on stderr; `aelf doctor`
shows a `claude-memory mirror: ON/off (reconciled?)` row; and
`aelf reconcile-claude-memory [--project] [--force]` re-runs the sweep
manually (e.g. after bulk out-of-session memory edits — `--force` runs past
the sentinel).

Ranking curated claude-memory above conversational capture (#1089 axis 2) is a
separate follow-up; this layer is ingestion + enablement only.

## Frontmatter → belief mapping (ratified 2026-06-23, #985)

A memory file's `metadata.type` selects the belief origin:

| `metadata.type`        | origin           | prior      | lock |
|------------------------|------------------|------------|------|
| `user`, `feedback`     | `user_validated` | undeflated | none |
| `project`, `reference` | `agent_inferred` | deflated   | none |
| absent / unrecognised  | `agent_inferred` | deflated   | none |

The belief **content** is the file body (frontmatter stripped). The mirror
**never locks**: `lock_level=user` (L0) stays reserved for an explicit
`aelf lock`. Auto-locking from a mirror would conflate the upstream tool's
taxonomy with aelfrice's ground-truth-freeze semantics and could flood L0.

### Why `user`/`feedback` ride a `route_overrides` decision

The per-type origin decision depends on the file's frontmatter, which the
hook reads from `raw_meta`. The replay-equality harness (`replay.py`) nulls
`raw_meta` before re-deriving, so any origin logic that reads `raw_meta`
inside the deterministic `derive()` path would drift under replay. Instead:

- `project`/`reference`/absent → no override → `derive()`'s deterministic
  classifier path → `agent_inferred`. Fully replay-stable; this is what the
  `claude_memory_v0_1.jsonl` replay-soak corpus exercises.
- `user`/`feedback` → the hook freezes a `RouteOverrides`
  (`origin=user_validated`, undeflated prior) and passes it on the
  `DerivationInput` field. This is the #265 "host computed a decision
  `derive()` can't reconstruct; freeze it" mechanism, which `replay.py`
  already round-trips.

So `derive()` carries **no** `claude_memory`-specific branch.

## Idempotency

The belief id is content-derived (`_belief_id(body, "claude_memory")`),
keyed on the source kind rather than the file path. A byte-identical
re-write therefore resolves to the same id and `content_hash`, and
`insert_or_corroborate` records a corroboration instead of inserting a
duplicate. An **edit** that changes the body mints a new belief; supersession
of the stale row is left to the consuming agent, consistent with #605
(dedup/contradiction/relatedness gates live in the agent, not aelfrice).
`update-in-place` / `SUPERSEDES` edges are deliberately out of scope for v1.

## Relationship to `/aelf:audit-claude-memory`

With the mirror enabled, the audit becomes a reconciliation/repair tool for
*pre-existing* drift (memory written before the mirror was turned on) rather
than the only bridge. The audit's "store-exclusive" class shrinks to exactly
that pre-mirror backlog.

## Privacy boundary

The hook reads files under `~/.claude/` at runtime and writes them into the
local per-project `brain.db`. Nothing crosses a network boundary. No memory
content is committed to the repository: the test fixtures and the
replay-soak corpus are **synthetic** (fabricated facts), per the locked
`~/.claude`-content-never-to-a-public-remote rule.
