# Install

## Prerequisites

- Python 3.12 or 3.13. (`uv` handles the Python version for you — no need to install Python separately.)
- [`uv`](https://docs.astral.sh/uv/). The supported install channel (#730). If you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or any agent that can spawn a hook on `UserPromptSubmit`.
- Linux, macOS, or Windows. Linux and macOS run the full test suite on every PR; Windows is covered by a smoke job — see [LIMITATIONS.md § Compatibility](LIMITATIONS.md#compatibility) for exactly what that does and does not assert.

## 1. Install the package

```bash
uv tool install aelfrice                # core (deps: numpy, scipy, snowballstemmer — local-only)
uv tool install "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

Or from source (developer install):

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice && uv sync
```

This installs eleven console scripts: `aelf` (the CLI) and ten hook entry-points the host spawns (`aelf-hook` for per-prompt retrieval, plus `aelf-transcript-logger`, `aelf-pre-compact-hook`, `aelf-commit-ingest`, `aelf-search-tool-hook`, `aelf-agent-context-hook`, `aelf-session-start-hook`, `aelf-stop-hook`, `aelf-pre-issue-hook`, `aelf-claude-memory-mirror`).

> **Migrating from pipx / pip?** As of v3.0.x aelfrice is uv-only (#730). If you previously installed via pipx, run `pipx uninstall aelfrice && uv tool install aelfrice` once. `aelf upgrade-cmd` will surface the same migration line on the next upgrade check. pip-installed users: `pip uninstall -y aelfrice && uv tool install aelfrice`.

Verify:

```bash
aelf --version       # aelf X.Y.Z
which aelf           # which env owns the binary
```

## 2. Wire it into Claude Code

```bash
aelf setup
```

This is idempotent. Run it again any time you change Python envs or move projects. It writes:

1. **A `UserPromptSubmit` hook** to `settings.json` so each prompt routes through `aelf-hook` for retrieval before the agent sees it.
2. **A `statusLine` notifier** that surfaces a one-line update banner only when a new release is available (empty otherwise).
3. **The full default-on auto-capture hook set and the bundled `/aelf:*` slash commands** — see § "Hooks installed by `aelf setup`" below.

Auto-detection picks the right scope and command path:

| Run from… | `--scope` | `--command` |
|---|---|---|
| inside a project venv | `project` (writes `<project>/.claude/settings.json`) | `<project>/.venv/bin/aelf-hook` |
| a `uv tool`-installed `aelf` outside any venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` |
| a venv unrelated to `cwd` | `user` | first `aelf-hook` on `$PATH`, falls back to the active venv |

Override with `--scope user|project` and `--command /abs/path/aelf-hook` when you need to.

### Codex host

Running Codex CLI instead? The same lifecycle verbs take `--host codex`:

```bash
aelf setup --host codex      # hooks into ~/.codex/hooks.json + $aelf-* agent skills
aelf doctor --host codex     # verify wiring; reports the installed skill count
aelf unsetup --host codex    # remove the aelfrice hooks and skills together
```

`aelf setup --host codex` writes the hook set to `~/.codex/hooks.json` (#1052) and installs the `/aelf:*` slash-command bundle as `$aelf-*` agent skills under `~/.agents/skills/` (v4.1.0+), generated from the same source bundle so the two surfaces never drift. Pass `--no-codex-skills` to install hooks only (`--codex-skills` is the default). Skill install is idempotent and prunes orphans, but only ever touches aelfrice-generated skills (an `AELFRICE-CODEX-SKILL` marker gates replacement/removal). Full detail — invocation, the generation transform, and the Codex-specific approval caveats — in [SLASH_COMMANDS § Codex host](SLASH_COMMANDS.md#codex-host-aelf--skills).

## 3. Verify wiring

```bash
aelf doctor
```

`aelf doctor` runs two checks back-to-back: hook resolution (every `command` in `settings.json` checked against `$PATH`; surfaces stale `bash <missing>.sh 2>/dev/null || true` wrappers from older installs) and the structural graph audit (orphan threads, FTS5 sync, locked-belief contradictions, corpus volume). Exits 1 on any structural failure so CI can gate on it. Empty store on a fresh project is normal — the corpus-volume warning only fires once the project is at least 7 days old.

Scope to one half:

```bash
aelf doctor hooks      # hook resolution only
aelf doctor graph      # structural auditor only
```

### Duplicate hook entries (v4.2.0+)

`settings.json` is a shared global config with several independent
writers — the harness itself, `aelf setup`, auto-install, and your own
edits. A host settings migration, a dotfiles merge, or a hand edit can
leave two entries for the same aelfrice hook. Both fire on every event,
so the only symptom is that prompts get slower for no visible reason:
nothing is broken, both paths resolve, and every check passes.

`aelf doctor` now reports these, and `--prune` collapses them:

```bash
aelf doctor hooks              # reports duplicates (read-only)
aelf doctor hooks --fix        # collapses them, keeping the first of each
```

The collapse only ever touches `aelf-*` entries; hooks you wrote
yourself are left exactly as they are, even if you have deliberately
listed one twice. Duplicates are counted separately from the stale-path
prune, because the two repairs answer different questions — a pruned
entry pointed at a venv that no longer exists, a collapsed one resolved
perfectly well and was simply installed twice.

### Hook timeouts

Every hook aelfrice installs carries an explicit `timeout` (v4.2.0+),
declared per hook in the bundled manifest: 15s for hooks that gate a
user-visible action, 30s for the bulk-ingest hooks. This is what bounds
the "a hook must never block your prompt" contract at the host level —
without it a hook waiting on another session's SQLite write lock would
stall for the host's default rather than aelfrice's budget.

The budgets have headroom over the real worst case rather than hugging
it: a cold-start retrieval on a ~46k-belief store measured under 3s when
it had to rebuild its BM25 sidecar from scratch. Override with
`aelf setup --timeout <seconds>` if your store is very large or your
disk is slow.

`aelf health` and `aelf stats` remain callable as back-compat aliases — hidden from default `--help` output but listed under `aelf --help --advanced`. The canonical replacements are `aelf doctor graph` (structural auditor, replaces `health`) and `aelf status` (counts, aliases `stats`).

## 4. Onboard a project

```bash
cd <project-root>
aelf onboard .
```

Walks the project (filesystem, git log, Python AST) and ingests structural facts as candidate beliefs. Typically under a second on a 50k-LOC project. Re-running is idempotent; it dedupes on `(source, sentence)`.

## 5. Lock the rules you care about

```bash
aelf lock "never push to main; use scripts/publish.sh"
aelf lock "all commits SSH-signed with ~/.ssh/id_rrs"
aelf locked                          # list what's locked
```

Locked beliefs short-circuit decay and are always returned at L0. They're the ones that survive.

Restart Claude Code. The next prompt that mentions "push" will already have your rules attached.

## Database

SQLite. Path resolution order:

1. `$AELFRICE_DB` — explicit override. `:memory:` is honoured (handy for tests).
2. `<git-common-dir>/aelfrice/memory.db` — when `cwd` is inside a git work-tree (v1.1.0+). Worktrees of one repo share a single DB through `--git-common-dir`. `.git/` is not git-tracked, so the brain graph never crosses the git boundary.
3. `~/.aelfrice/memory.db` — fallback for non-git directories.

Pin a project with `export AELFRICE_DB=/abs/path/.aelfrice.db` (works well with `direnv`).

### Migrating from v1.0.x

v1.0.x kept a single global DB at `~/.aelfrice/memory.db`. v1.1.0 resolves per-project. Port beliefs forward with:

```bash
cd <project-root>
aelf migrate                # dry-run; reports what would copy
aelf migrate --apply        # actually copy filtered beliefs
aelf migrate --apply --all  # copy every belief from the legacy DB
aelf migrate --from /alt/path/memory.db --apply
```

`aelf migrate` opens the source DB read-only (SQLite `mode=ro` URI). Project-mention filtering (default) restricts the copy to beliefs that name the active project; `--all` skips it. Idempotent on re-run.

### Batch ingest of historical sessions

If you have prior Claude Code sessions sitting at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, you can backfill them:

```bash
aelf ingest-transcript --batch ~/.claude/projects/
aelf ingest-transcript --batch ~/.claude/projects/ --since 2026-01-01
```

Auto-detects the JSONL format on a per-line basis (handles both aelfrice's transcript-logger output and Claude Code's internal session shape). Idempotent on re-run.

> **Privacy.** Session JSONLs may contain pasted secrets, customer data, or anything you typed in chat. Batch ingestion brings all of that into the local belief graph. There is no PII scrubber on the v1.2 ingest path. Review before backfilling.

## Hooks installed by `aelf setup`

Bare `aelf setup` wires the v1.2.0 auto-capture pipeline alongside the read-side `UserPromptSubmit` retrieval hook:

| Hook | Event(s) | Default | What it does |
|---|---|---|---|
| UserPromptSubmit retrieval | `UserPromptSubmit` | always | injects matched beliefs as `<aelfrice-memory>` block |
| transcript-ingest | `UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact` | **on** | logs every turn to a per-project JSONL; PreCompact rotates the file and ingests it into beliefs/edges |
| commit-ingest | `PostToolUse:Bash` | **on** | each successful `git commit` runs the triple extractor on the message |
| session-start | `SessionStart` | **on** | new sessions open with L0 locked beliefs already injected |
| stop-lock-prompt | `Stop` | **on** | prompt to lock correction-class beliefs from this session (#582) |
| search-tool | `PreToolUse:Grep` / `Glob` | **on** (v3.0.1+) | belief-store check before the agent's own Grep/Glob fires |
| search-tool-bash | `PreToolUse:Bash` | **on** (v3.0.1+) | belief-store check before shell grep/rg/find/fd/ack fires |
| pre-issue-guard | `PreToolUse:Bash` | **on** (v3.4.0+) | blocks `gh issue create` when the title overlaps an existing issue or shipped commit at or above 0.5 Jaccard (#941) |
| claude-memory-mirror | `PostToolUse:Write` / `Edit` / `MultiEdit` | **on** (v3.7.0+) | one-way mirror of host claude-memory fact-file writes into the belief graph (#985). Enabled by `AELFRICE_MIRROR_CLAUDE_MEMORY` / `[memory] mirror_claude_memory`, or — since v4.0 (#1089) — by the per-project consent sentinel the one-shot reconcile writes at first `aelf setup`, so on a set-up project it mirrors by default; an explicit env `0` / TOML `false` always wins over the sentinel (that is the opt-out) |
| agent-context | `PreToolUse:Agent` / `Task` | **on** | dispatched subagents inherit L0 locked + task-relevant beliefs via prompt injection; fail-open passthrough, kill switch `AELFRICE_AGENT_CONTEXT=0`, opt out `--no-agent-context` (#1068) |
| rebuilder | `PreCompact` (installed) — block ships on `SessionStart(source="compact")` | off | retrieval-curated context rebuilder (augment-mode, v1.4 alpha). `--rebuilder` installs a `PreCompact` entry, and that hook does trigger-mode bookkeeping only — it injects nothing. The block itself ships *after* compaction, on the already-present default-on `SessionStart` hook ([#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)). |

Opt out per-hook (persists across upgrades via `~/.aelfrice/opt-out-hooks.json`):

```bash
aelf setup --no-transcript-ingest      # skip the four transcript-logger hooks
aelf setup --no-commit-ingest          # skip the commit-message ingest hook
aelf setup --no-session-start          # skip the SessionStart locked-belief injection
aelf setup --no-stop-hook              # skip the Stop lock-prompt hook
aelf setup --no-search-tool            # skip the PreToolUse:Grep|Glob hook
aelf setup --no-search-tool-bash       # skip the PreToolUse:Bash hook
aelf setup --no-pre-issue-guard        # skip the issue-dup detection guard
aelf setup --no-claude-memory-mirror   # skip the claude-memory → belief-graph mirror hook
aelf setup --no-agent-context          # skip the PreToolUse:Agent worker-context injection
```

The SessionStart recap line ("N beliefs written since last session", v3.5+, #934) rides on the SessionStart hook rather than having its own manifest entry — `aelf setup --no-sessionstart-recap` suppresses it at install time but is not persisted in `opt-out-hooks.json`.

Opt in to the off-by-default hooks:

```bash
aelf setup --rebuilder                 # post-compaction context rebuilder (alpha)
```

`aelf unsetup` mirrors: bare invocation removes every default-on hook. `--no-*` flags suppress per-hook removal.

All hooks fail open: every failure path returns exit 0 — a hook problem must never break a prompt or a commit. The one intentional exception to non-blocking behavior is the pre-issue-guard, which exits 2 to block a duplicate `gh issue create` (bypass by setting `ALLOW_DUP_ISSUE=1` in the host's environment); its own failure paths still exit 0.

### Self-installing hook manifest (v3.0+)

The list of default-on hooks above is declared in `src/aelfrice/data/hook_manifest.json` and ships in the wheel. The first `aelf <cmd>` invocation after a fresh install or a bare `uv tool upgrade aelfrice` reconciles the installed manifest version against `~/.aelfrice/installed-manifest-version` and merges any new entries into `~/.claude/settings.json` automatically. This closes the loop on bare package-manager upgrades — you no longer have to remember to re-run `aelf setup` to pick up hooks added in newer releases.

What auto-install does:

* Happy path (stamp == installed version) is one stat + one short file read. No JSON parse, no settings.json read.
* On mismatch, takes an `fcntl` exclusive lock on `~/.aelfrice/.auto-install.lock` so concurrent `aelf` invocations cannot race on the merge.
* Reuses the same install primitives `aelf setup` calls; the on-disk shape of settings.json is byte-identical.
* Adds only entries the manifest claims by basename — anything the user added to settings.json by hand is preserved.
* Respects opt-outs: if you ever ran `aelf setup --no-transcript-ingest`, that choice is persisted at `~/.aelfrice/opt-out-hooks.json` and survives upgrades. Re-running `aelf setup` (without the `--no-*` flag) rescinds the opt-out.
* Prints a single stderr line when entries were actually added: `aelfrice: hooks updated to v3.5.0 (was v3.4.0) — added: pre_issue_guard`.

Opt-out controls:

```bash
export AELFRICE_NO_AUTO_INSTALL=1   # power user: I manage settings.json by hand
aelf setup --no-stop-hook           # disable one hook; persists across upgrades
```

`aelf doctor` flags broken or unresolvable hook commands and nags when any of the four v2.1 auto-capture hooks (transcript-ingest, commit-ingest, session-start, stop-hook) is missing. It does not reconcile settings.json against the manifest, so newer manifest hooks (search-tool, pre-issue-guard) are not flagged when absent — the auto-installer is the reconciliation path.

> **Privacy note.** Default-on transcript-ingest means every turn you type lands in the per-project SQLite DB on `PreCompact` rotation. The DB is local-only (no network, no telemetry — see § "What you get for free" in the README and [PRIVACY.md](PRIVACY.md)) but the JSONL has no PII scrubber. If you paste secrets, customer data, or anything you don't want indexed in chat, opt out with `--no-transcript-ingest` and use `aelf lock` / `aelf onboard` for explicit ingestion only.

### Legacy-schema detection + auto-migrate (`aelf doctor`, v3.0+)

`aelf doctor` scans all per-project DBs under `~/.aelfrice/projects/*/memory.db` and migrates any that use the pre-v1.x schema (no `origin` column on the `beliefs` table) in place. DBs on the old schema cannot participate in the v2.x / v3.0 lifecycle — `agent_remembered`, `user_validated`, calibrated weights, `aelf:promote`, federation `scope` — because the columns that track them are absent.

Auto-migrate behaviour (v3.0+, [#593](https://github.com/robotrocketscience/aelfrice/issues/593)): for each detected legacy DB, doctor renames `memory.db` → `memory.db.pre-v1x.bak` (atomic POSIX rename) and runs the existing `migrate()` core with `copy_all=True, apply=True` against the backup, writing a fresh modern-schema DB back at the original path. Beliefs land with `origin=ORIGIN_UNKNOWN`. The backup is preserved verbatim and never overwritten — if a stale `<path>.pre-v1x.bak` already exists from a prior failed run, the migration aborts for that DB and the failure is surfaced under `report.failed_migrate_dbs` instead of clobbering recoverable state. Success path emits one line per DB:

```
migrated ~/.aelfrice/projects/2e7ed55e017a/memory.db: 35,332 beliefs, 412ms (backup at ~/.aelfrice/projects/2e7ed55e017a/memory.db.pre-v1x.bak)
```

Empty DBs (zero rows) are silently skipped. Failures surface a residual nag pointing at the manual `aelf migrate --from <path> --apply` path for that specific DB.

For manual migrations (e.g. a DB outside the standard `~/.aelfrice/projects/` tree):

```bash
aelf migrate --from /alt/path/memory.db          # dry-run
aelf migrate --from /alt/path/memory.db --apply  # write
```

### Concurrent installs (v4.2.0+)

`settings.json` has more than one writer: `aelf setup`, `aelf unsetup`, `aelf doctor --fix`, the automatic hook merge that runs on CLI invocations — and the host itself, whenever you approve a permission or change a setting.

Since [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) every aelfrice mutation of that file takes an exclusive lock on a sibling `settings.json.lock` and writes once, so aelfrice's own writers can no longer overwrite each other. Two consequences you may see:

```
setup aborted: another aelfrice process is writing settings.json
(could not acquire ~/.claude/settings.json.lock within 10.0s). Nothing
was changed; re-run `aelf setup`.
```

Another aelfrice process held the lock longer than the wait. Nothing was written; re-run the command.

```
setup aborted: ~/.claude/settings.json was modified by another process
while aelfrice was writing it; no changes were made. Re-run the command.
```

Something that does not take aelfrice's lock — in practice the host — replaced the file mid-install. aelfrice will not commit over it, because doing so would discard whatever the host just wrote. Re-run the command; every change aelfrice makes to this file is convergent, so a second run is safe.

The host cannot be made to take aelfrice's lock, so this check is a detector, not a cure. It converts what used to be silent loss of a setting into a message telling you to try again.

### Incomplete store migrations (v4.2.0+)

Opening the store runs a set of one-shot schema migrations, each stamped as complete only after its work lands. Before [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) a migration that raised took the whole store with it: because the completion marker was never written, the next open re-ran the same pass and failed the same way, and every entry point — CLI, hooks, MCP — opens the store. There was no way back short of editing SQLite by hand.

A migration that cannot finish no longer stops the store from opening. The failure is recorded, the pass is skipped, and the store keeps its pre-migration shape: reads and writes work, and because the completion marker stays unset the migration is retried on every subsequent open, so upgrading to a build that fixes it repairs the store automatically.

`aelf doctor` is where that state surfaces:

```
store migration(s) INCOMPLETE — the store opens and is usable, but one or more one-shot migrations could not finish:
  _maybe_consolidate_content_hash_duplicates: IntegrityError('UNIQUE constraint failed: edges.src, edges.dst, edges.type')
fix: these retry automatically on every open, so upgrading (`aelf upgrade`) is the first thing to try. If the same
migration keeps failing, report the error above — the store will keep working in the meantime.
```

Try `aelf upgrade` first. If the same migration keeps failing, the error text above is the useful thing to include in a report; nothing needs to be done urgently, since the store stays usable in the meantime.

### Pruning dormant per-project DBs (`aelf doctor --prune-dormant`, v3.0+)

Some per-project DBs hold beliefs from projects you worked on briefly with an older aelfrice version, then abandoned. They never get migrated, never get touched, and just sit there. `aelf doctor --prune-dormant` lists DBs whose `memory.db` mtime is older than `--idle-days` (default 30) and lets you delete them one at a time.

```bash
aelf doctor --prune-dormant                # dry-run: list dormant DBs only
aelf doctor --prune-dormant --idle-days 90 # tighter idle threshold
aelf doctor --prune-dormant --apply        # prompts [y/N] per DB; deletes on 'y'
```

Sample output:

```
found 2 dormant per-project DB(s) (idle >= 30d, 41,615 beliefs, 18,432.0 KiB total):
  ~/.aelfrice/projects/2e7ed55e017a/memory.db (35,332 beliefs, 15,120.4 KiB, idle 35d)
  ~/.aelfrice/projects/18a856c7a96b/memory.db (6,283 beliefs, 3,311.6 KiB, idle 32d)

dry-run only. re-run with `--apply` to be prompted [y/N] per DB.
```

`--apply` prompts per DB; the default at the prompt is N, so anything other than `y`/`yes` (including a bare Enter or piping `/dev/null`) preserves the file. There is no `--yes` shortcut — destructive deletion is always per-DB and explicit. Unlike `aelf migrate`, this does not move beliefs anywhere; it only removes the DB file.

The dormant scan is schema-agnostic — both pre-v1.x and modern-schema DBs are flagged when idle. A DB you still want to migrate should go through `aelf migrate --from <path> --apply` (above) before pruning, not after.

## Update notifier

```bash
aelf upgrade-cmd          # prints the canonical upgrade command (`run: …`) when an
                          # update is available; otherwise `aelfrice is up to date`
aelf upgrade-cmd --check  # same output, no behaviour difference
/aelf:upgrade             # imperative slash: detect + run + re-setup
```

`aelf upgrade-cmd` emits `run: uv tool upgrade aelfrice` on a uv-managed install. If aelfrice was installed via another tool (pipx, pip, system), the `run:` line is the migration chain (`pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent) — uv is the single supported install channel (#730). The CLI does not execute the upgrade itself: replacing the running interpreter mid-process is unreliable on Windows.

The orange statusline banner appears automatically when an update is available. Disable with `export AELF_NO_UPDATE_CHECK=1`.

## Uninstall

You must pick a disposition for the DB:

```bash
aelf uninstall --keep-db              # leave the DB in place (safe default)
aelf uninstall --archive backup.aenc  # encrypt to file then delete
aelf uninstall --purge                # permanently delete (three confirmation gates)
uv tool uninstall aelfrice            # finally remove the wheel
```

`--archive` uses Fernet (AES-128-CBC + HMAC, scrypt-derived key). Recover later:

```python
from pathlib import Path
from aelfrice.lifecycle import decrypt_archive
open("out.db","wb").write(decrypt_archive(Path("backup.aenc"), "password"))
```

Requires the `[archive]` extra.

### What gets removed

The store is not a single file. Alongside `memory.db` the package writes
SQLite sidecars (`-wal`, `-shm`), the BM25F index, any backup DBs made by
past migrations, the per-turn hook injection log, the belief-write feed log,
and the `transcripts/`, `rebuild_logs/`, and `telemetry/` directories.
Several of those hold belief content verbatim.

`--purge` and `--archive` both operate on that whole set. Each one prints
the manifest — every path and its size — before it destroys anything, so
you can check the total against what you expect.

`--archive` encrypts **the belief database only**. Everything else in the
set is either derived from the database (BM25F index, injection log, feed
log, telemetry) or a rolling capture buffer (`transcripts/`), so it is
deleted rather than added to the archive. The command says so and asks for
confirmation before taking your password — an encrypted archive sitting
next to plaintext copies of the same content is not a guarantee.

Before v4.2.0 both flags deleted only `memory.db`, and `--archive` did not
checkpoint the write-ahead log first. On a store held open by a running
hook that meant the archive captured a stale database — in the worst case
an empty one — while the real content stayed on disk in plaintext in
`memory.db-wal` (#1173). If you archived a store under an earlier version,
that archive may be incomplete; check it with `decrypt_archive` before
relying on it.

If `$AELFRICE_DB` points somewhere other than a directory aelfrice created
(a `.git/aelfrice/` or `~/.aelfrice/`), the generically-named artifacts
cannot be safely attributed to aelfrice — `$HOME/transcripts/` may well be
yours. Those are listed for you to remove by hand instead of being deleted.

### `~/.aelfrice/`

That directory is a second location with different ownership: it belongs to
no single store. Uninstall therefore removes what aelfrice put there **by
name**, and never sweeps it.

Install state goes in **every** mode, `--keep-db` included, because each
file records that a step already happened and a survivor would make a
reinstall read a stale decision as current:

- the **LLM classification consent sentinel** — the load-bearing one. It
  used to outlive the uninstall, so a later reinstall read the grant as
  still valid and never re-prompted (#1186). It now goes, and a reinstall
  asks again.
- the manifest-version and uv-migration stamps, so reinstalling the same
  version re-merges the hooks instead of short-circuiting into an
  installed-but-inert state.
- the temporal-spine backfill sentinel, the auto-install lock, and the
  `claude-memory` reconcile sentinel.
- `logs/hook-failures.log` (and `logs/` itself, once empty).

Captured data goes only when you asked for data to go, i.e. under `--purge`
and `--archive` but not `--keep-db`: `telemetry.jsonl` and a legacy
`transcripts/` directory left by pre-repo-store versions.

Kept in every mode:

- `projects/` holds **other projects' stores**, one `memory.db` per
  project-id slug. `aelf uninstall` disposes of one store — the one
  `db_path()` resolves to — so removing this would destroy belief corpora
  you did not ask it to touch. Dispose of those separately.
- `shared/` is the conventional home for read-only **federation peers**
  (`knowledge_deps.json`), which is another store's corpus by another name.
- `config.json` is your configuration, kept for the same reason as
  `opt-out-hooks.json`: it records your decision that a hook should not be
  installed, and that should survive a reinstall.

Anything else found in `~/.aelfrice/` is **listed and left alone** — the
directory can hold files aelfrice never wrote, and the destructive modes
name every path they are about to remove before they remove it.

If you want the directory gone entirely, and you have checked
`ls ~/.aelfrice/projects/` and `ls ~/.aelfrice/shared/` first:
`rm -rf ~/.aelfrice/` after running `uninstall`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | Confirm `~/.local/bin` (uv tool shim dir) is on `$PATH`. `uv tool update-shell` adds it for you. |
| Hook fires but no `<aelfrice-memory>` block appears | `aelf doctor` — usually the hook command points at a deleted script. |
| `aelf doctor` says "skipped (shell metacharacters)" on a hook line | Stale install. `aelf setup` rewrites the hook in place. |
| Two worktrees of the same repo see the same beliefs | Working as designed — they share `--git-common-dir`. Pin one with `AELFRICE_DB`. |
| `aelf search` returns "store is empty" | Run `aelf onboard .` from the project root. |
| `SQLite database is locked` under heavy concurrent writes | v1.1.0+ uses WAL + `busy_timeout=5000`. If you still see it, file an issue with the repro. |
