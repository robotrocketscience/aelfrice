# Harness integration: aelfrice + Claude Code

Operational guide for users running aelfrice alongside Claude Code's
built-in auto-memory system.

If you only want one rule of thumb: **after `aelf setup --transcript-ingest`,
both stores coexist productively and you do not need to do anything
else.** The rest of this document is for users who want a more
opinionated arrangement.

## The two stores

Claude Code (the CLI harness) and aelfrice each maintain a memory
store. They live in different places and are written by different
mechanisms.

| | Claude Code auto-memory | aelfrice |
|---|---|---|
| Storage | `~/.claude/projects/<slug>/memory/*.md` + `MEMORY.md` index | `<git-common-dir>/.git/aelfrice/memory.db` (SQLite) |
| Format | Markdown files with YAML frontmatter | FTS5-indexed beliefs with α/β posteriors and typed edges |
| Write path | Harness directive in `~/.claude/CLAUDE.md` ("If the user explicitly asks you to remember something, save it…") | Explicit `aelf remember` / `aelf:remember` / `aelf onboard`, plus the v1.2 hooks (transcript-ingest, commit-ingest) |
| Read path | Auto-loaded into every prompt by Claude Code itself | The `UserPromptSubmit` hook injects retrieval results above each prompt |
| Determinism | None (LLM decides what to write and when) | Bit-level reproducible — write log replay reconstructs every state |
| Apply feedback | No | Yes — every retrieval moves posteriors, contradictions resolve via tie-breaker |

The two stores capture different things by design and they do not
merge automatically.

## What v1.2 changes

Earlier versions of this guide (`LIMITATIONS.md § harness conflict`,
v1.0–v1.1) recommended hand-editing `~/.claude/CLAUDE.md` to disable
the auto-memory directive and route everything to the MCP. v1.2
ships three hook-based capture paths that close the original
limitation without requiring you to fight the harness:

- **`SessionStart`** ([context_rebuilder.md](context_rebuilder.md)).
  Injects L0 locked beliefs at session open under the
  `<aelfrice-baseline>` tag, before any user prompt fires.
- **`UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact`
  transcript-ingest** ([transcript_ingest.md](transcript_ingest.md)).
  Every conversation turn is appended to
  `<git-common-dir>/.git/aelfrice/transcripts/turns.jsonl`. On
  compaction the JSONL rotates and `aelf ingest-transcript` lowers
  the rotated file into beliefs and edges in the brain graph.
- **`PostToolUse:Bash` commit-ingest** ([commit_ingest_hook.md](commit_ingest_hook.md)).
  After every successful `git commit` Bash call, the hook runs the
  triple extractor on the commit message body and inserts the
  resulting beliefs and edges under a deterministic
  `sha256(branch + ":" + commit_hash)[:16]` session id.

After running `aelf setup --transcript-ingest --commit-ingest
--session-start`, aelfrice receives fresh beliefs from normal
session activity without the auto-memory directive being involved
either way. The two stores coexist; one is no longer starving the
other.

The harness directive itself is unchanged. Claude Code continues to
write `.md` files when the model decides to. aelfrice continues to
write SQLite rows from the hooks. **They are parallel pipelines, not
competing ones.**

## Three coexistence modes

Pick the one that matches your tolerance for the auto-memory
mechanism.

### Mode 1 — Coexist (recommended default)

Both stores active. `aelf setup --transcript-ingest --commit-ingest
--session-start` plus the default `aelf setup` (UserPromptSubmit hook
for retrieval). Auto-memory continues to write `.md` files; you read
them from `MEMORY.md` like before. aelfrice writes SQLite rows; you
query them with `aelf search` or get them injected via the retrieval
hook on every prompt.

When this is right:
- You like having the human-readable `.md` index for grep-able review.
- You don't mind that the two stores diverge over time — auto-memory
  keeps "user prefers vim" while aelfrice keeps "we always use uv,
  see PR #109."
- You want zero `CLAUDE.md` edits.

When this is wrong:
- You want a single canonical answer to "what does this agent know
  about me?" Mode 2 or 3 fits better.

### Mode 2 — aelfrice is canonical, auto-memory is read-only

Same hooks as Mode 1, plus an edit to `~/.claude/CLAUDE.md` that
removes or rephrases the auto-memory write directive. The harness
no longer creates new `.md` files; existing files keep loading on
session open.

The directive to remove is the block under `# auto memory` (or
similar — exact wording varies by Claude Code version). Replace it
with:

```markdown
# Memory

This project uses aelfrice as the canonical memory store. To save
something durable, call `aelf:remember` or `aelf:lock` (MCP tools).
Do NOT create new files under .claude/projects/.../memory/ — those
are read-only legacy.
```

When this is right:
- You want one source of truth for new memories.
- You like the deterministic-replay properties of aelfrice.
- You're willing to use the MCP tool surface explicitly when you
  want something durable saved.

When this is wrong:
- You depend on the auto-memory's "save proactively without being
  asked" behaviour. Aelfrice does not do that — every write to
  aelfrice is from an explicit hook event or an explicit user/agent
  call.

### Mode 3 — aelfrice only, auto-memory disabled

Mode 2 plus deleting (or archiving) the contents of
`~/.claude/projects/<slug>/memory/`. Auto-memory has nothing to
load and nothing to write. You also delete the `MEMORY.md` index in
that directory.

When this is right:
- You want a clean slate.
- You're consolidating onto aelfrice as part of a larger workflow
  cleanup.

When this is wrong:
- The existing `.md` content is valuable. Migrate it first (next
  section) before deleting.

## Migrating existing auto-memory content into aelfrice

If you've accumulated `.md` files under
`~/.claude/projects/<slug>/memory/` and want them in aelfrice as
beliefs:

```bash
cd ~/.claude/projects/<slug>/memory
aelf onboard .
```

`aelf onboard` is the standard scanner path — it will walk the
directory, parse Markdown headings and prose, classify candidate
sentences, and insert the survivors as beliefs with
`origin=agent_inferred`. To upgrade the ones you actually
acknowledge, use `aelf validate <id>` (v1.2+) on each — that flips
the origin to `user_validated` without locking, which is the right
tier for "I read this and confirmed it" content. Use `aelf lock` for
constraints you want locked.

The migration is one-shot. Re-running `aelf onboard` on the same
directory after edits is idempotent — content-hash skips
already-ingested rows.

## Decision matrix

| Want | Use |
|---|---|
| Just install and have it work | Mode 1 (default) |
| One canonical answer to "what is remembered" | Mode 2 |
| Pure aelfrice, no harness `.md` files | Mode 3 (after migration) |
| Save something durable right now | `aelf:remember` (MCP) or `aelf remember` (CLI) |
| Save something the agent must never forget | `aelf:lock` |
| Acknowledge an onboard belief without locking it | `aelf:validate` |
| See what aelfrice currently holds | `aelf search "<query>"` or `aelf:search` |
| See what auto-memory currently holds | `cat ~/.claude/projects/<slug>/memory/MEMORY.md` |

## What this does not address

- **Multi-machine sync.** aelfrice's DB lives under `.git/aelfrice/`
  and is not git-tracked. Auto-memory's `.md` files live under
  `~/.claude/projects/` and are also not synced unless you set up
  your own dotfiles repository. Neither store handles multi-machine
  sync; that is out of scope at v1.2.
- **Cross-project federation.** Each git project gets its own
  aelfrice store. Auto-memory has its own per-project directory.
  Cross-project knowledge sharing is on the v1.3 retrieval-wave
  roadmap.
- **Deleting auto-memory entries from inside aelfrice.** If you
  migrate `.md` content into aelfrice and later edit the originals
  externally, aelfrice will not notice. Re-run `aelf onboard` after
  significant changes.

## Troubleshooting

**"I called `aelf:remember` but the belief isn't appearing in `MEMORY.md`."**
Expected. `aelf:remember` writes to the aelfrice DB only. Use
`aelf search` or `aelf:locked` to confirm the write landed. The two
stores do not mirror each other.

**"I see the same fact in both stores."**
Expected if you ran the migration in § "Migrating existing
auto-memory content" or if the auto-memory directive captured the
same fact during a session where you also called `aelf:remember`.
Use `aelf:validate` to mark the aelfrice copy as
`user_validated`; the duplicate in `.md` form is harmless and you
can ignore or delete it.

**"Auto-memory is creating new `.md` files faster than I want."**
This is a harness behaviour, not an aelfrice one. Edit
`~/.claude/CLAUDE.md` per Mode 2 to slow it down or stop it.

**"Hooks aren't firing."**
Run `aelf doctor`. It validates that every hook command in
`settings.json` resolves to an executable on disk. Most hook silence
traces back to a venv mismatch or a missing `aelf-*` console script.

## See also

- [LIMITATIONS.md § harness conflict](LIMITATIONS.md) — the original
  v1.0/v1.1 limitation this doc closes.
- [transcript_ingest.md](transcript_ingest.md) — the per-turn
  capture pipeline.
- [commit_ingest_hook.md](commit_ingest_hook.md) — the git-commit
  capture pipeline.
- [promotion_path.md](promotion_path.md) — the
  `agent_inferred → user_validated` mechanism that lets users tier
  imported `.md` content explicitly.
