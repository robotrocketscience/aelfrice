# Harness integration: aelfrice + Claude Code

Operational guide for users running aelfrice alongside Claude Code's
built-in auto-memory system.

If you want one rule only: **after `aelf setup`, both stores
coexist productively. You do not need to do anything else.** The
default-on hook bundle of v2.1+ connects aelfrice to normal session
activity. That bundle holds the transcript-ingest hook, the
commit-ingest hook and the session-start hook. You do not have to edit
`CLAUDE.md`. The rest of this document is for the users who
want a more opinionated arrangement.

## The two stores

Claude Code (the CLI harness) and aelfrice each maintain a memory
store. The two stores live in different places. Different mechanisms
write them.

| | Claude Code auto-memory | aelfrice |
|---|---|---|
| Storage | The `~/.claude/projects/<slug>/memory/*.md` files and the `MEMORY.md` index | One `<git-common-dir>/aelfrice/memory.db` (SQLite) for each project. `~/.aelfrice/memory.db` only outside a git working tree |
| Format | Markdown files with YAML frontmatter | Beliefs indexed with full-text search version 5 (FTS5), with α/β posteriors and typed edges |
| Write path | A harness directive in `~/.claude/CLAUDE.md` ("If the user explicitly asks you to remember something, save it…") | An explicit `aelf lock`, `aelf:lock` or `aelf onboard` call. The default-on capture hooks of v2.1+ also write (transcript-ingest, commit-ingest) |
| Read path | Auto-loaded into every prompt by Claude Code itself | The `UserPromptSubmit` hook injects retrieval results above each prompt |
| Determinism | None. The large language model (LLM) decides what to write and when to write it | Reproducible at bit level. A replay of the write log reconstructs every *belief* state. **The edges are an exception.** The code as shipped writes them outside the log ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)) |
| Apply feedback | No | Yes. An explicit `used` or `harmful` signal moves the posteriors. The *exposure* from retrieval is audit-only by default (#1086), and it is recorded as recurrence rather than as evidence. A contradiction resolves through a tie-breaker |

The two stores capture different things by design. They do not
merge. One deliberate exception exists, and it is one-way. The
claude-memory mirror hook (v3.7+, #985) ingests the writes to the
harness memory files into the belief graph. Since v4.0 (#1089) that
hook runs by default, once the reconcile at first setup records the
consent for the project. The aelfrice store tracks the harness store.
Nothing ever moves in the other direction.

## What the hook bundle does

Earlier versions of this guide (`LIMITATIONS.md § harness conflict`,
v1.0–v1.1) recommended one manual step. The step was to edit
`~/.claude/CLAUDE.md` by hand. The purpose of the edit was to disable
the auto-memory directive. The other purpose was to route everything
to the Model Context Protocol (MCP). The capture paths that use hooks shipped at v1.2 and
went default-on at v2.1
([#529](https://github.com/robotrocketscience/aelfrice/issues/529)).
Those paths close the original limitation, and you do not have to work
against the harness:

- **`SessionStart`** ([hook_hardening.md](../design/hook_hardening.md)).
  Injects the L0 locked beliefs at session open, under the
  `<aelfrice-baseline>` tag. This injection happens before any user
  prompt fires.
- **`UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact`
  transcript-ingest** ([transcript_ingest.md](../design/transcript_ingest.md)).
  The hook appends every conversation turn to
  `<git-common-dir>/aelfrice/transcripts/turns.jsonl`. On
  compaction the JSONL file rotates. `aelf ingest-transcript` then
  lowers the rotated file into beliefs and edges in the brain graph.
- **`PostToolUse:Bash` commit-ingest** ([commit_ingest_hook.md](../design/commit_ingest_hook.md)).
  After every successful `git commit` call through Bash, the hook runs
  the triple extractor on the body of the commit message. The hook
  then inserts the resulting beliefs and edges under a deterministic
  session id, `sha256(branch + ":" + commit_hash)[:16]`.

Run `aelf setup`. aelfrice then receives fresh beliefs from normal
session activity, and the auto-memory directive takes no part either
way. These three hooks are default-on since
v2.1 ([#529](https://github.com/robotrocketscience/aelfrice/issues/529)).
To opt out of one hook, use `--no-transcript-ingest`,
`--no-commit-ingest` or `--no-session-start`. The two stores coexist.
One store no longer deprives the other of content.

The harness directive itself is unchanged. Claude Code continues to
write `.md` files when the model decides to write them. aelfrice
continues to write SQLite rows from the hooks. **The two are parallel
pipelines. They do not compete.**

## Three coexistence modes

Pick the one that matches your tolerance for the auto-memory
mechanism.

### Mode 1 — Coexist (recommended default)

Both stores are active. `aelf setup` installs the full default-on
bundle. The bundle holds the UserPromptSubmit retrieval, the
transcript-ingest hook, the commit-ingest hook and the session-start
hook (all v2.1+). The bundle also holds the Stop correction-lock hook,
the PreToolUse search-tool hooks, the pre-issue duplicate guard and the
session-start recap (the v3.x additions). Auto-memory continues to
write `.md` files, and you read them from `MEMORY.md` as before.
aelfrice writes SQLite rows. You query the rows with `aelf search`, or
the retrieval hook injects them on every prompt.

When this is right:
- You want the human-readable `.md` index for review with grep.
- You accept that the two stores diverge over time. Auto-memory
  keeps "user prefers vim" while aelfrice keeps "we always use uv,
  see PR #109."
- You want no edits to `CLAUDE.md`.

When this is wrong:
- You want a single canonical answer to the question "what does this
  agent know about me?". Mode 2 or Mode 3 fits better.

### Mode 2 — aelfrice is canonical, auto-memory is read-only

Mode 2 uses the same hooks as Mode 1. It adds an edit to
`~/.claude/CLAUDE.md` that removes or rephrases the auto-memory write
directive. The harness then creates no new `.md` files. The existing
files continue to load at session open.

The directive to remove is the block under `# auto memory` (or
similar — exact wording varies by Claude Code version). Replace it
with:

```markdown
# Memory

This project uses aelfrice as the canonical memory store. To save
something durable, call `/aelf:lock`. Do NOT create
new files under .claude/projects/.../memory/ — those are read-only
legacy.
```

When this is right:
- You want one source of truth for new memories.
- You want the deterministic-replay properties of aelfrice.
- You accept that you must run `/aelf:lock` explicitly to save
  something durable.

When this is wrong:
- You depend on the "save proactively without being asked" behaviour
  of auto-memory. aelfrice does not do that. Every write to
  aelfrice comes from an explicit hook event, or from an explicit call
  by the user or the agent.

### Mode 3 — aelfrice only, auto-memory disabled

Mode 3 is Mode 2 with one more step. Delete the contents of
`~/.claude/projects/<slug>/memory/`, or archive them. Auto-memory then
has nothing to load and nothing to write. Delete the `MEMORY.md` index
in that directory too.

When this is right:
- You want to start with no auto-memory content.
- You are consolidating onto aelfrice as part of a larger cleanup of
  your workflow.

When this is wrong:
- The existing `.md` content is valuable. Before you delete it,
  migrate it with the procedure in the next section.

## Migrating existing auto-memory content into aelfrice

You may have accumulated `.md` files under
`~/.claude/projects/<slug>/memory/`. To bring them into aelfrice as
beliefs, run the commands below.

Run the command from the project root. Do not run it from the memory directory. aelfrice resolves the DB from the working directory (`$AELFRICE_DB` → the git-common-dir of the cwd → `~/.aelfrice/memory.db`). An onboard run *from inside* `~/.claude/...` therefore writes the beliefs into the global fallback DB instead of into the DB of the project.

```bash
cd <your-project-root>
aelf onboard ~/.claude/projects/<slug>/memory
```

`aelf onboard` is the standard scanner path. It walks the
directory. It parses the Markdown headings and the prose. It
classifies the candidate sentences. It inserts the survivors as
beliefs with `origin=agent_inferred`. To upgrade a belief that you
acknowledge, run `aelf promote <id>` on it. That command flips
the origin to `user_validated` and does not lock the belief. That
tier is the correct one for "I read this and confirmed it" content.
Use `aelf lock` for the constraints that you want locked.

The migration is one-shot. A second run of `aelf onboard` on the same
directory after edits is idempotent. The content hash makes the run
skip the rows that are already ingested. Use `aelf promote <id>`
to flip the origin to `user_validated`, and to re-scope the belief if
you want. `aelf promote` shipped at v1.7.0 (#391), and the
`--to-scope` re-scoping was added at v3.0
([#689](https://github.com/robotrocketscience/aelfrice/issues/689)).
The older `aelf validate <id>` verb still works as an alias.

## Decision matrix

| Want | Use |
|---|---|
| Install it and have it work with no more steps | Mode 1 (default) |
| One canonical answer to "what is remembered" | Mode 2 |
| Pure aelfrice, no harness `.md` files | Mode 3 (after migration) |
| Save something the agent must never forget | `aelf:lock` |
| Acknowledge an onboard belief without locking it | `aelf:promote` |
| See what aelfrice currently holds | `aelf search "<query>"` or `aelf:search` |
| See what auto-memory currently holds | `cat ~/.claude/projects/<slug>/memory/MEMORY.md` |

## What this does not address

- **Sync across machines.** The DB of aelfrice lives under
  `.git/aelfrice/`, and git does not track it. The `.md` files of
  auto-memory live under `~/.claude/projects/`. Those files are also
  not synced, unless you set up your own dotfiles repository. Neither
  store handles sync across machines. That work stays out of scope as
  of v3.5 (see [LIMITATIONS § Sharing, sync, or distributed-write federation](../user/LIMITATIONS.md)).
- **Cross-project federation.** Each git project gets its own
  aelfrice store. Auto-memory has its own directory for each project.
  v3.0 shipped *read-only* cross-project federation through
  `knowledge_deps.json` ([#650](https://github.com/robotrocketscience/aelfrice/issues/650)
  / [#655](https://github.com/robotrocketscience/aelfrice/issues/655)
  / [#661](https://github.com/robotrocketscience/aelfrice/issues/661)).
  A peer DB is opened read-only. A mutation against a foreign id is
  rejected at the API surface. Multi-writer federation is out of scope,
  per the v3.0 ratification.
- **Deletion of auto-memory entries from inside aelfrice.** You can
  migrate `.md` content into aelfrice and later edit the original
  files with another tool. aelfrice does not see those edits. Run
  `aelf onboard` again after a significant change.

## Troubleshooting

**"I called `aelf:lock` but the belief isn't appearing in `MEMORY.md`."**
This behaviour is expected. `aelf:lock` writes to the aelfrice DB
only. Use `aelf search` or `aelf:locked` to confirm that the write
landed. The mirror is one-way. It copies the writes to the harness
memory files into aelfrice (#985), and never the reverse.

**"I see the same fact in both stores."**
This result is expected in two cases. You ran the migration in
§ "Migrating existing auto-memory content". Or the auto-memory
directive captured the same fact during a session in which you also
called `aelf:lock`.
Use `aelf:promote` to mark the aelfrice copy as `user_validated`.
The duplicate in `.md` form is harmless. You can ignore it or
delete it.

**"Auto-memory is creating new `.md` files faster than I want."**
This is a harness behaviour, not an aelfrice behaviour. Edit
`~/.claude/CLAUDE.md` as Mode 2 describes, to slow the harness down or
to stop it.

**"Hooks aren't firing."**
Run `aelf doctor`. It validates that every hook command in
`settings.json` resolves to an executable on disk. Most silent hooks
trace back to a mismatch of the virtual environment, or to a missing
`aelf-*` console script.

## See also

- [LIMITATIONS.md § harness conflict](../user/LIMITATIONS.md) — the original
  harness-conflict limitation of v1.0/v1.1 that this document closed. That entry was removed from that file at v1.2.
- [transcript_ingest.md](../design/transcript_ingest.md) — the capture
  pipeline for each turn.
- [commit_ingest_hook.md](../design/commit_ingest_hook.md) — the capture
  pipeline for a git commit.
- [promotion_path.md](../design/promotion_path.md) — the
  `agent_inferred → user_validated` mechanism. It lets a user tier
  imported `.md` content explicitly.
