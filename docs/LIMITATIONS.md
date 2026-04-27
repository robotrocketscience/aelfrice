# Limitations

Known limitations of `aelfrice` at v1.0.0. Each entry includes the
behavior, why it's a limitation, and what to expect in upcoming
releases. Status is tracked in the project's issue queue.

## Auto-memory write path under Claude Code

**Behavior.** When `aelfrice` is installed alongside Claude Code's
built-in file-based auto-memory system, Claude Code's harness
directive routes any "save a memory" intent to its own file-based
store (under `.claude/projects/.../memory/*.md` plus a `MEMORY.md`
index), not to the `aelfrice` MCP server. The MCP server stays
connected and remains queryable for retrieval, but it does not
receive new beliefs from normal session activity. New beliefs only
enter the `aelfrice` store via explicit tool calls (`aelf remember`,
`aelf onboard`, MCP `aelf:remember`) or bulk import.

**Why this is a limitation.** The README's central claim is that
`apply_feedback` is the endpoint that makes `aelfrice` distinct from
plain RAG: a memory which actually applies feedback should outperform
one that doesn't. If the MCP receives no new beliefs during a
conversation, then `apply_feedback` is firing against a snapshot
written at install time (or at the most recent explicit
`aelf onboard` / `aelf remember` call), not against beliefs the agent
forms during current work. The feedback loop is intact mathematically
but starved of fresh inputs.

**v1.0.1 partial mitigation.** v1.0.1 closes the retrieval-side of
the loop: the `UserPromptSubmit` hook records every retrieval as a
`feedback_history` row tagged `source='hook'`, and `apply_feedback`
moves posteriors based on actual hook-driven retrievals. This means
even without a new write path, beliefs already in the store are
exercised by feedback during normal use. The write path itself
remains gated by the harness directive at v1.0.1.

**Workaround today.** To make the `aelfrice` MCP the canonical write
path, edit your `~/.claude/CLAUDE.md` to remove or rephrase the
auto-memory harness directive, and rely on `aelf remember` (CLI) or
the MCP `aelf:remember` tool for new beliefs. This is a user-side
configuration change; `aelfrice` does not attempt to override the
harness in code.

**Tracked.** Lab issue `jso/aelfrice-lab#78`. v1.2 will publish
`docs/HARNESS_INTEGRATION.md` with a documented procedure for users
who want the MCP to be canonical without manually editing
`CLAUDE.md`.

## v1.0 retrieval is BM25-only

**Behavior.** `aelfrice.retrieval.retrieve()` uses two layers: L0
(locked beliefs auto-loaded) and L1 (FTS5 with BM25 ranking). It does
not use embeddings, HRR, BFS multi-hop, or entity-index resolution.

**Why this is a limitation.** Cross-session counting and
multi-hop chain queries underperform on the published benchmarks. The
v2-research-line codebase reached MAB MH 60% Opus (paper best <=7%)
with entity-index + BFS; v1.0's BM25-only path does not approach that
on multi-hop tasks. Single-hop and structured-fact queries are not
affected.

**Tracked.** v1.x evidence-gate per the README contract. Re-port of
the v2-research-line retrieval stack is queued for v1.3+ pending
benchmark deltas on the v1 baseline. Lab `docs/V2_REENTRY_QUEUE.md`
catalogs the full plan.

## v1.0 MCP surface is 8 tools

**Behavior.** The v1.0 MCP server exposes 8 tools. The
v2-research-line codebase exposed up to 25 tools.

**Why this is a limitation.** Tools that surface graph analytics
(`graph_metrics`), document linking (`link_docs`), Obsidian import
(`import_obsidian`), and explicit feedback acknowledgment
(`confirm`) are not available at v1.0. Workflows that depended on
those tools in the v2-research-line codebase will need to use direct
SQLite queries or wait for the relevant ports.

**Tracked.** Lab issue `jso/aelfrice-lab#59`. Cheaper tools
(`graph_metrics`, `confirm`) ship in v1.1 / v1.2 alongside their
natural integrations. Full surface recovery is v2.0.

## v1.0 project identity is keyed by `SHA256(cwd)`

**Behavior.** The project's SQLite database is stored at
`~/.aelfrice/projects/<hash>/memory.db` where `<hash>` is computed
from the current working directory. Moving the project directory or
running `aelfrice` from a `git worktree` produces a different hash and
a different (empty) database.

**Why this is a limitation.** A project's accumulated memory is lost
on directory rename, machine move, or worktree creation. The
local-only memory promise quietly breaks under these moves.

**Tracked.** Lab issues `jso/aelfrice-lab#106` (closed; verifies in
v1.0.1 regression test), `#107` (in-repo `.git/aelfrice/` storage),
`#110` (`.aelfrice.toml` for portable identity), `#111` (orphan-DB
migration tooling). All scheduled for v1.1.0.

## v1.0 has no `aelf --version` flag

**Behavior.** `aelf --version` returns an argparse error.

**Tracked.** Lab issue `jso/aelfrice-lab#141`. Fixed in v1.0.1.
