# Install

## Prerequisites

- Python 3.12 or 3.13. `uv` manages the Python version for you, so you don't have to install Python separately.
- [`uv`](https://docs.astral.sh/uv/), the supported install channel (#730). If you don't have `uv`, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or any agent that can spawn a hook on `UserPromptSubmit`.
- Linux, macOS, or Windows. Linux runs the full test suite on every pull request, a smoke job covers Windows, and no workflow tests macOS automatically. For what the smoke job asserts and what it doesn't, see [the compatibility notes in the limitations list](LIMITATIONS.md#compatibility).

## 1. Install the package

```bash
uv tool install aelfrice                # core (deps: numpy, scipy, snowballstemmer — local-only)
uv tool install "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

Or install from the source repository, which is the developer install:

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice && uv sync
```

This command installs eleven console scripts. The first is `aelf`, the command-line interface (CLI). The other ten are hook entry points that the host spawns:

- `aelf-hook` does the per-prompt retrieval
- `aelf-transcript-logger`
- `aelf-pre-compact-hook`
- `aelf-commit-ingest`
- `aelf-search-tool-hook`
- `aelf-agent-context-hook`
- `aelf-session-start-hook`
- `aelf-stop-hook`
- `aelf-pre-issue-hook`
- `aelf-claude-memory-mirror`

> **Migration from pipx or pip.** As of v3.0.x, aelfrice supports only uv (#730). If you installed aelfrice with pipx, run `pipx uninstall aelfrice && uv tool install aelfrice` once. If you installed aelfrice with pip, run `pip uninstall -y aelfrice && uv tool install aelfrice`. At the next upgrade check, `aelf upgrade-cmd` gives you the same migration line.

To verify the install, run:

```bash
aelf --version       # aelf X.Y.Z
which aelf           # which env owns the binary
```

## 2. Wire it into Claude Code

```bash
aelf setup
```

`aelf setup` is idempotent. Run it again whenever you change Python environments, and again whenever you move projects. The command writes:

1. **A `UserPromptSubmit` hook** in `settings.json`. Every prompt then goes through `aelf-hook` for retrieval before the agent receives it.
2. **A `statusLine` notifier**. The notifier shows a one-line update banner only when a new release is available, and the banner is empty the rest of the time.
3. **The full default-on auto-capture hook set and the bundled `/aelf:*` slash commands**. See [Hooks installed by `aelf setup`](#hooks-installed-by-aelf-setup) later on this page.

Automatic detection selects the scope and the command path:

| Run from… | `--scope` | `--command` |
|---|---|---|
| inside a project venv | `project` (writes `<project>/.claude/settings.json`) | `<project>/.venv/bin/aelf-hook`, or `<project>\.venv\Scripts\aelf-hook.exe` on Windows |
| a `uv tool`-installed `aelf` outside any venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` |
| a venv unrelated to `cwd` | `user` | first `aelf-hook` on `$PATH`, falls back to the active venv |

To set these values yourself, override the detection with `--scope user|project` and `--command /abs/path/aelf-hook`.

### Codex host

If you run Codex CLI, add `--host codex` to the same lifecycle verbs:

```bash
aelf setup --host codex      # hooks into $CODEX_HOME/hooks.json (else ~/.codex) + $aelf-* skills
aelf doctor --host codex     # verify wiring; reports the installed skill count
aelf unsetup --host codex    # remove the aelfrice hooks and skills together
```

`aelf setup --host codex` writes the hook set to the Codex configuration home (#1052). The same command installs the `/aelf:*` slash-command bundle as `$aelf-*` agent skills under `~/.agents/skills/` (v4.1.0+). aelfrice generates both surfaces from one source bundle, so the two never differ. To install the hooks only, pass `--no-codex-skills`. The default is `--codex-skills`.

The skill install is idempotent and removes orphan skills. It touches only the skills aelfrice generated, because an `AELFRICE-CODEX-SKILL` marker controls replacement and removal. For the full detail — the invocation, the generation transform, and the approval warnings specific to Codex — see [the Codex host section of the slash-command reference](SLASH_COMMANDS.md#codex-host-aelf--skills).

#### `$CODEX_HOME` (unreleased)

Codex reads its configuration from `$CODEX_HOME` when that variable has a non-empty value, and from `~/.codex` otherwise. All three verbs resolve the home directory the same way (#1427), so each of these configurations is wired where that Codex reads:

- an isolated profile
- a continuous integration (CI) runner
- a portable install
- two side-by-side Codex configurations

`aelf doctor --host codex` prints the resolved home on its first line, so you can see which directory the command inspected.

A `$CODEX_HOME` that doesn't exist is an error, not a condition for a fallback, and a `$CODEX_HOME` that names something other than a directory is an error too. In both cases, setup, doctor, and unsetup refuse the operation, create nothing, and leave `~/.codex` unchanged.

Both refusals match how Codex behaves. `CODEX_HOME=/nowhere codex mcp list` reports `CODEX_HOME points to "/nowhere", but that path does not exist` and then stops.

aelfrice doesn't create the directory for you. Codex refuses to start against such a configuration home, so a report of "setup succeeded" would be false. Create the directory, correct the variable, or unset the variable.

`$CODEX_HOME` does **not** move the `$aelf-*` agent skills. `~/.agents/skills/` is a standard path across agents, and other agents keep their skills in the same directory.

#### Shared `hooks.json` (unreleased)

`hooks.json` belongs to Codex, not to aelfrice. One document holds your own entries, another installer's entries, and aelfrice's.

Setup and unsetup do three things to that document:

1. They take an advisory lock on a sibling `hooks.json.lock`.
2. They re-check the content hash of the file immediately before they commit.
3. They replace the file atomically (#1428).

An aelfrice run therefore merges a concurrent edit instead of overwriting it.

`unsetup` does none of this against a home that has no `hooks.json`; run `unsetup` there and it creates nothing, not even the lock file.

Where a `hooks.json` was present, the zero-byte `hooks.json.lock` stays on disk. That's intentional. If another process holds that lock, deleting the lock puts two writers on two inodes with no mutual exclusion at all.

If the file keeps changing across three attempts, the command refuses and writes nothing. Run the command again.

Two refusals are new and intentional. aelfrice leaves a non-object `hooks` value byte-for-byte unchanged and reports it, and it does the same with a non-list value on an event it installs into. A `null` value gets the same treatment, because `null` is a value at that position and not an absence.

aelfrice can merge only into the documented `{"hooks": {"<Event>": [...]}}` shape. Merging into any other would delete structure it didn't write. Correct the file, or pass `--force` to replace it.

One window stays open. A process that takes no lock can still replace the file in the instant between the final check of aelfrice and the rename. Closing that window needs a mutation protocol from Codex, and Codex doesn't offer one today.

## 3. Verify wiring

```bash
aelf doctor
```

`aelf doctor` runs two checks, one after the other:

1. Hook resolution. The check compares every `command` in `settings.json` against `$PATH`. It also reports the stale `bash <missing>.sh 2>/dev/null || true` wrappers left by older installs.
2. The structural graph audit. The audit covers orphan threads, full-text search version 5 (FTS5) sync, contradictions between locked beliefs, and corpus volume.

`aelf doctor` exits 1 on any structural failure, so CI can gate on the command. An empty store on a new project is normal. The corpus-volume warning fires only when the project is at least 7 days old.

To run one half only, use one of these commands:

```bash
aelf doctor hooks      # hook resolution only
aelf doctor graph      # structural auditor only
```

### Duplicate hook entries (v4.2.0+)

`settings.json` is a shared global configuration file with several
independent writers: the harness itself, `aelf setup`, auto-install,
and your own edits. A migration of the host settings, a merge of
dotfiles, or an edit by hand can leave two entries for the same
aelfrice hook. Both entries fire on every event, and the only symptom
is that prompts get slower for no visible reason. Nothing is broken:
both paths resolve, and every check passes.

`aelf doctor` now reports these entries, and `--fix` collapses them:

```bash
aelf doctor hooks              # reports duplicates (read-only)
aelf doctor hooks --fix        # collapses them, keeping the first of each
```

The collapse touches only the `aelf-*` entries. The hooks that you
wrote yourself stay exactly as they are, including a hook you listed
twice on purpose. aelfrice counts the duplicates separately from the
stale-path prune, because the two repairs answer different questions.
A pruned entry pointed at a venv that no longer exists; a collapsed
entry resolved correctly and was installed twice.

### Hook timeouts

Every hook that aelfrice installs has an explicit `timeout` (v4.2.0+).
The bundled manifest declares the timeout for each hook. A hook that
gates a user-visible action gets 15s. The bulk-ingest hooks get 30s.
The timeout is what bounds the "a hook must never block your prompt"
contract at the host level. Without the timeout, a hook can
wait for the SQLite write lock of another session, and then stall for
the default of the host rather than the budget of aelfrice.

The budgets have headroom over the real worst case, rather than
sitting close to it. A cold-start retrieval on a store of approximately 46k beliefs
measured less than 3s. That retrieval had to rebuild its Best Matching
25 (BM25) sidecar from the start. If your store is very large, or if
your disk is slow, override the budget with
`aelf setup --timeout <seconds>`.

`aelf health` and `aelf stats` are still callable as back-compatible aliases. The default `--help` output hides them, and `aelf --help --advanced` lists them. The canonical replacement for `health` is `aelf doctor graph`, the structural auditor. The canonical replacement for `stats` is `aelf status`. Run `aelf status` for the counts.

## 4. Onboard a project

```bash
cd <project-root>
aelf onboard .
```

`aelf onboard .` walks the project and ingests the structural facts as candidate beliefs. The command reads the filesystem, the git log, and the Python abstract syntax tree (AST). It typically takes less than one second on a project of 50k lines of code. A second run is idempotent, because the command dedupes on `(source, sentence)`.

## 5. Lock the rules you care about

```bash
aelf lock "never push to main; use scripts/publish.sh"
aelf lock "all commits SSH-signed with ~/.ssh/id_rrs"
aelf locked                          # list what's locked
```

Locked beliefs short-circuit decay: aelfrice always returns them at L0, so they're the beliefs that survive.

Restart Claude Code. The next prompt that mentions "push" arrives with your rules already injected.

## Database

The store is a SQLite database. The path resolution order is:

1. `$AELFRICE_DB` — the explicit override. aelfrice honors `:memory:`, which is useful for tests.
2. `<git-common-dir>/aelfrice/memory.db` — used when `cwd` is inside a git work-tree (v1.1.0+). The worktrees of one repo share a single DB through `--git-common-dir`. Git doesn't track `.git/`, so the brain graph never crosses the git boundary.
3. `~/.aelfrice/memory.db` — the fallback for a directory that isn't in a git work-tree.

To pin a project, run `export AELFRICE_DB=/abs/path/.aelfrice.db`. This method works well with `direnv`.

### Migrating from v1.0.x

v1.0.x kept a single global DB at `~/.aelfrice/memory.db`. v1.1.0 resolves the path for each project. To port the beliefs forward, use these commands:

```bash
cd <project-root>
aelf migrate                # dry-run; reports what would copy
aelf migrate --apply        # actually copy filtered beliefs
aelf migrate --apply --all  # copy every belief from the legacy DB
aelf migrate --from /alt/path/memory.db --apply
```

`aelf migrate` opens the source DB read-only with the SQLite `mode=ro` URI. The project-mention filter is the default, and it restricts the copy to the beliefs that name the active project. `--all` skips the filter. `aelf migrate` is idempotent on a second run.

### Batch ingest of historical sessions

If you have prior Claude Code sessions sitting at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, you can backfill them:

```bash
aelf ingest-transcript --batch ~/.claude/projects/
aelf ingest-transcript --batch ~/.claude/projects/ --since 2026-01-01
```

The command detects the JSONL format line by line, so it handles both the transcript-logger output of aelfrice and the internal session shape of Claude Code. It's idempotent on re-run.

> **Privacy.** A session JSONL file can hold pasted secrets, customer data, or any other text that you typed in the chat. Batch ingestion brings all of that content into the local belief graph. The v1.2 ingest path has no scrubber for personally identifiable information (PII). Review the sessions before you backfill them.

## Hooks installed by `aelf setup`

Bare `aelf setup` installs the v1.2.0 auto-capture pipeline, together with the read-side `UserPromptSubmit` retrieval hook:

| Hook | Event(s) | Default | What it does |
|---|---|---|---|
| UserPromptSubmit retrieval | `UserPromptSubmit` | always | injects the matched beliefs as an `<aelfrice-memory>` block |
| transcript-ingest | `UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact` | **on** | logs every turn to a per-project JSONL file. PreCompact rotates that file, then ingests it into beliefs and edges |
| commit-ingest | `PostToolUse:Bash` | **on** | each successful `git commit` runs the triple extractor on the message |
| session-start | `SessionStart` | **on** | new sessions open with the L0 locked beliefs already injected |
| stop-lock-prompt | `Stop` | **on** | prompts you to lock this session's correction-class (#582) and directive (#1315) beliefs |
| search-tool | `PreToolUse:Grep` / `Glob` | **on** (v3.0.1+) | checks the belief store before the agent's own Grep or Glob fires |
| search-tool-bash | `PreToolUse:Bash` | **on** (v3.0.1+) | checks the belief store before a shell grep, rg, find, fd, or ack fires |
| pre-issue-guard | `PreToolUse:Bash` | **on** (v3.4.0+) | blocks `gh issue create` when the title overlaps an existing issue or a shipped commit at 0.5 Jaccard or above (#941) |
| claude-memory-mirror | `PostToolUse:Write` / `Edit` / `MultiEdit` | **on** (v3.7.0+) | mirrors the host claude-memory fact-file writes one-way into the belief graph (#985). Either `AELFRICE_MIRROR_CLAUDE_MEMORY` or `[memory] mirror_claude_memory` enables the hook, and since v4.0 (#1089) so does the per-project consent sentinel, which the one-shot reconcile writes at the first `aelf setup`. A project that's set up therefore mirrors by default. An explicit env `0` or TOML `false` always wins over the sentinel; that's the opt-out |
| agent-context | `PreToolUse:Agent` / `Task` | **on** | dispatched subagents inherit the L0 locked beliefs and the task-relevant ones through prompt injection. The hook is a fail-open passthrough. Set `AELFRICE_AGENT_CONTEXT=0` to disable the injection, or pass `--no-agent-context` to opt out (#1068) |
| rebuilder | `PreCompact` (installed) — block ships on `SessionStart(source="compact")` | off | the retrieval-curated context rebuilder (augment-mode, v1.4 alpha). `--rebuilder` installs a `PreCompact` entry that does the trigger-mode bookkeeping only and injects nothing. The block itself ships *after* the compaction, on the default-on `SessionStart` hook that's already present ([#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)) |

To opt out of one hook, use the matching option. The opt-out persists across upgrades in `~/.aelfrice/opt-out-hooks.json`.

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

The SessionStart recap line ("N beliefs written since last session", v3.5+, #934) has no manifest entry of its own. The SessionStart hook prints the recap line. `aelf setup --no-sessionstart-recap` suppresses the recap line at install time, and `opt-out-hooks.json` doesn't persist that choice.

To opt in to the hooks that are off by default, use this command:

```bash
aelf setup --rebuilder                 # post-compaction context rebuilder (alpha)
```

`aelf unsetup` mirrors `aelf setup`. A bare invocation removes every default-on hook. The `--no-*` flags suppress the removal of the matching hook.

All hooks fail open. Every failure path returns exit 0, because a hook problem must never break a prompt or a commit. The pre-issue-guard is the one intentional exception to the non-blocking behavior: it exits 2 to block a duplicate `gh issue create`. To bypass the guard, set `ALLOW_DUP_ISSUE=1` in the host's environment. The failure paths of the guard still exit 0.

### Self-installing hook manifest (v3.0+)

`src/aelfrice/data/hook_manifest.json` declares the list of default-on hooks above, and that file ships in the wheel. The first `aelf <cmd>` invocation after a fresh install does two things, and so does the first `aelf <cmd>` invocation after a bare `uv tool upgrade aelfrice`:

1. It reconciles the installed manifest version against `~/.aelfrice/installed-manifest-version`.
2. It merges any new entries into `~/.claude/settings.json` automatically.

You no longer have to remember to run `aelf setup` again after a bare upgrade with the package manager. The hooks that a newer release adds arrive without that step.

What auto-install does:

* In the usual path the stamp equals the installed version. That path is one stat and one short file read. It runs no JSON parse. It does not read settings.json.
* On a mismatch it takes an `fcntl` exclusive lock on `~/.aelfrice/.auto-install.lock`, so concurrent `aelf` invocations can't race on the merge.
* It reuses the same install primitives that `aelf setup` calls. The shape of settings.json on disk is byte-identical.
* It adds only the entries that the manifest claims by basename. It preserves everything that you added to settings.json by hand.
* It respects the opt-outs. If you ever ran `aelf setup --no-transcript-ingest`, `~/.aelfrice/opt-out-hooks.json` persists that choice, and the choice survives upgrades. To rescind the opt-out, run `aelf setup` again without the `--no-*` flag.
* It prints a single stderr line when it added entries: `aelfrice: hooks updated to v3.5.0 (was v3.4.0) — added: pre_issue_guard`.

Opt-out controls:

```bash
export AELFRICE_NO_AUTO_INSTALL=1   # power user: I manage settings.json by hand
aelf setup --no-stop-hook           # disable one hook; persists across upgrades
```

`aelf doctor` flags a hook command that is broken or that doesn't resolve. `aelf doctor` also reports when one of the four v2.1 auto-capture hooks is missing: transcript-ingest, commit-ingest, session-start, and stop-hook.

`aelf doctor` doesn't reconcile settings.json against the manifest. That's why `aelf doctor` doesn't flag the newer manifest hooks, search-tool and pre-issue-guard, when they're absent. The auto-installer is the reconciliation path.

> **Privacy note.** transcript-ingest is on by default, so every turn that you type goes into the per-project SQLite DB at the `PreCompact` rotation. The DB is local-only: it uses no network and sends no telemetry. See § "What you get for free" in the README and [the privacy documentation](PRIVACY.md).
>
> The JSONL file has no PII scrubber. If you paste secrets, customer data, or anything else that you don't want indexed in the chat, opt out with `--no-transcript-ingest`. Then use `aelf lock` or `aelf onboard` for explicit ingestion only.

### Legacy-schema detection + auto-migrate (`aelf doctor`, v3.0+)

`aelf doctor` scans every per-project DB under `~/.aelfrice/projects/*/memory.db`. Where a DB uses the pre-v1.x schema, `aelf doctor` migrates it in place. A DB on that schema has no `origin` column on the `beliefs` table.

A DB on the old schema can't participate in the v2.x and v3.0 lifecycle, because the columns that track the lifecycle are absent. That lifecycle includes `agent_remembered`, `user_validated`, the calibrated weights, `aelf:promote`, and the federation `scope`.

Auto-migrate (v3.0+, [#593](https://github.com/robotrocketscience/aelfrice/issues/593)) works like this. For each legacy DB that it detects, doctor does three things:

1. doctor renames `memory.db` to `memory.db.pre-v1x.bak` with an atomic POSIX rename.
2. doctor runs the existing `migrate()` core with `copy_all=True, apply=True` against the backup.
3. doctor writes a fresh modern-schema DB back at the original path.

Each copied belief gets `origin=ORIGIN_UNKNOWN`. doctor preserves the backup verbatim and never overwrites it. If a stale `<path>.pre-v1x.bak` from an earlier failed run is already present, the migration stops for that DB, and `report.failed_migrate_dbs` surfaces the failure. doctor doesn't overwrite the recoverable state.

The success path emits one line per DB:

```
migrated ~/.aelfrice/projects/2e7ed55e017a/memory.db: 35,332 beliefs, 412ms (backup at ~/.aelfrice/projects/2e7ed55e017a/memory.db.pre-v1x.bak)
```

doctor skips an empty DB (zero rows) silently. A failure prints a residual warning that points at the manual `aelf migrate --from <path> --apply` path for that specific DB.

For a manual migration — a DB outside the standard `~/.aelfrice/projects/` tree, for example — use these commands:

```bash
aelf migrate --from /alt/path/memory.db          # dry-run
aelf migrate --from /alt/path/memory.db --apply  # write
```

### Concurrent installs (v4.2.0+)

`settings.json` has more than one writer:

- `aelf setup`
- `aelf unsetup`
- `aelf doctor --fix`
- the automatic hook merge that runs on CLI invocations
- the host itself, each time you approve a permission or change a setting

Since [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) every aelfrice mutation of that file takes an exclusive lock on a sibling `settings.json.lock`, and then writes once. aelfrice's own writers can no longer overwrite each other. You might see two consequences:

```
setup aborted: another aelfrice process is writing settings.json
(could not acquire ~/.claude/settings.json.lock within 10.0s). Nothing
was changed; re-run `aelf setup`.
```

Another aelfrice process held the lock longer than the wait. aelfrice wrote nothing. Run the command again.

```
setup aborted: ~/.claude/settings.json was modified by another process
while aelfrice was writing it; no changes were made. Re-run the command.
```

Something that doesn't take the aelfrice lock replaced the file during the install. In practice that something is the host. aelfrice doesn't commit over the new file, because a commit would discard what the host just wrote. Run the command again. Every change that aelfrice makes to this file is convergent, so a second run is safe.

You can't make the host take the aelfrice lock, so this check is a detector and not a cure. Before the check, a setting was lost silently. The check now gives you a message that tells you to try again.

### Incomplete store migrations (v4.2.0+)

Opening the store runs a set of one-shot schema migrations. Each migration is stamped as complete only after it finishes its work.

Before [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) a migration that raised an error took the whole store with it. The completion marker was never written, so the next open re-ran the same pass and failed the same way. Every entry point opens the store: the CLI, the hooks, and MCP. The only way back was to edit the SQLite database by hand.

A migration that can't finish no longer stops the store from opening. aelfrice records the failure and skips the pass. The store keeps its pre-migration shape, and reads and writes work. The completion marker stays unset, so aelfrice retries the migration on every later open. An upgrade to a build that fixes the migration thus repairs the store automatically.

`aelf doctor` is where that state surfaces:

```
store migration(s) INCOMPLETE — the store opens and is usable, but one or more one-shot migrations could not finish:
  _maybe_consolidate_content_hash_duplicates: IntegrityError('UNIQUE constraint failed: edges.src, edges.dst, edges.type')
fix: these retry automatically on every open, so upgrading (`aelf upgrade`) is the first thing to try. If the same
migration keeps failing, report the error above — the store will keep working in the meantime.
```

Try `aelf upgrade` first. If the same migration keeps failing, include the error text above in a report. Nothing urgent is required, because the store stays usable in the meantime.

### Pruning dormant per-project DBs (`aelf doctor --prune-dormant`, v3.0+)

Some per-project DBs hold beliefs from projects that you worked on briefly with an older aelfrice version and then abandoned. Nothing migrates those DBs. Nothing touches them. They stay on disk. `aelf doctor --prune-dormant` lists the DBs whose `memory.db` mtime is older than `--idle-days`, which defaults to 30 days. The command lets you delete those DBs one at a time.

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

`--apply` prompts you for each DB. The default at the prompt is N. Any answer other than `y` or `yes` preserves the file, and a bare Enter and a pipe from `/dev/null` are such answers.

There is no `--yes` shortcut, because a destructive deletion is always explicit and for one DB at a time. Unlike `aelf migrate`, this command doesn't move beliefs anywhere. The command only removes the DB file.

The dormant scan doesn't examine the schema. The scan flags a pre-v1.x DB and a modern-schema DB equally when the DB is idle. If you still want to migrate a DB, run `aelf migrate --from <path> --apply` (described earlier) before you prune the DB, not after.

## Update notifier

```bash
aelf upgrade-cmd          # prints the canonical upgrade command (`run: …`) when an
                          # update is available; otherwise `aelfrice is up to date`
aelf upgrade-cmd --check  # same output, no behaviour difference
/aelf:upgrade             # imperative slash: detect + run + re-setup
```

`aelf upgrade-cmd` emits `run: uv tool upgrade aelfrice` on a uv-managed install. If another tool (pipx, pip, system) installed aelfrice, the `run:` line is the migration chain instead. That chain is `pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent. uv is the single supported install channel (#730).

The CLI does not run the upgrade itself, because replacing the running interpreter mid-process is unreliable on Windows.

The orange statusline banner appears automatically when an update is available. To disable the banner, run `export AELF_NO_UPDATE_CHECK=1`.

## Uninstall

Pick a disposition for the DB:

```bash
aelf uninstall --keep-db              # leave the DB in place (safe default)
aelf uninstall --archive backup.aenc  # encrypt to file then delete
aelf uninstall --purge                # permanently delete (three confirmation gates)
uv tool uninstall aelfrice            # finally remove the wheel
```

`--archive` uses Fernet with AES-128-CBC, HMAC, and a scrypt-derived key. To recover the DB later, use this code:

```python
from pathlib import Path
from aelfrice.lifecycle import decrypt_archive
open("out.db","wb").write(decrypt_archive(Path("backup.aenc"), "password"))
```

The `--archive` option and this recovery code both require the `[archive]` extra.

### What gets removed

The store is not a single file. Alongside `memory.db` the package
writes these items:

- the SQLite sidecars (`-wal`, `-shm`)
- the fielded Best Matching 25 (BM25F) index
- any backup DBs made by past migrations
- the per-turn log of the hook injection
- the feed log of the belief writes
- the `transcripts/`, `rebuild_logs/`, and `telemetry/` directories

Several of those items hold belief content verbatim.

`--purge` and `--archive` both operate on that whole set. Each one
prints the manifest before it destroys anything. The manifest gives
every path and its size, so you can check the total against what you
expect.

`--archive` encrypts **the belief database only**. Every other item in
the set is either derived from the database or a rolling capture
buffer. The derived items are the BM25F index, the injection log, the
feed log, and the telemetry; the rolling capture buffer is
`transcripts/`. The command deletes those items rather than adding
them to the archive.

The command reports this behavior, and asks for your confirmation
before it takes your password. An encrypted archive next to plaintext
copies of the same content is not a guarantee.

Before v4.2.0 both flags deleted only `memory.db`. `--archive` also
didn't checkpoint the write-ahead log (WAL) first. On a store held
open by a running hook that meant the archive captured a stale
database. In the worst case the archive captured an empty database.
The real content stayed on disk in plaintext in `memory.db-wal`
(#1173).

If you archived a store under an earlier version, that archive might
be incomplete. Check the archive with `decrypt_archive` before you
rely on it.

aelfrice creates a `.git/aelfrice/` directory or a `~/.aelfrice/`
directory. If `$AELFRICE_DB` points somewhere other than such a
directory, aelfrice can't safely attribute the generically named
artifacts to itself. `$HOME/transcripts/` can be your own directory. aelfrice
therefore lists those artifacts for you to remove by hand instead of
deleting them.

### `~/.aelfrice/`

That directory is a second location with different ownership, and it
belongs to no single store. So uninstall removes what aelfrice put
there **by name**, and never sweeps the directory.

The install state goes in **every** mode, `--keep-db` included. Each
of those files records that a step already happened. A file that
survived would make a reinstall read a stale decision as current. The
install state is:

- the **LLM classification consent sentinel**. This one is
  load-bearing. The sentinel used to outlive the uninstall, so a later
  reinstall read the grant as still valid and never re-prompted
  (#1186). The sentinel now goes. A reinstall asks again.
- the manifest-version stamp and the uv-migration stamp. A reinstall
  of the same version therefore re-merges the hooks. A stamp that
  survives the uninstall makes the reinstall short-circuit into a
  state that is installed but inert.
- the temporal-spine backfill sentinel, the auto-install lock, and the
  `claude-memory` reconcile sentinel.
- `logs/hook-failures.log`, and `logs/` itself once it is empty.

The captured data goes only when you asked for data to go. That is
under `--purge` and `--archive`, but not under `--keep-db`. The
captured data is `telemetry.jsonl` and a legacy `transcripts/`
directory left by pre-repo-store versions.

aelfrice keeps these items in every mode:

- `projects/` holds **other projects' stores**. There is one
  `memory.db` for each project-id slug. `aelf uninstall` disposes of
  one store, the store that `db_path()` resolves to. Removing
  `projects/` would destroy belief corpora that you didn't ask
  `aelf uninstall` to touch. Dispose of those stores separately.
- `shared/` is the conventional home for read-only **federation
  peers** (`knowledge_deps.json`). A federation peer is the corpus of
  another store by another name.
- `config.json` is your configuration. aelfrice keeps it for the same
  reason as `opt-out-hooks.json`. `opt-out-hooks.json` records your
  decision that a hook should not be installed, and that decision
  should survive a reinstall.

aelfrice **lists and leaves alone** anything else found in
`~/.aelfrice/`. The directory can hold files that aelfrice never wrote. The
destructive modes name every path they are about to remove before they
remove it.

To remove the directory entirely, first check
`ls ~/.aelfrice/projects/` and `ls ~/.aelfrice/shared/`. Then run
`uninstall`. Then run `rm -rf ~/.aelfrice/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | Confirm that `~/.local/bin`, the uv tool shim directory, is on `$PATH`. `uv tool update-shell` adds it for you. |
| Hook fires but no `<aelfrice-memory>` block appears | Run `aelf doctor`. Usually the hook command points at a deleted script. |
| `aelf doctor` says "skipped (shell metacharacters)" on a hook line | The install is stale. `aelf setup` rewrites the hook in place. |
| Two worktrees of the same repo see the same beliefs | This is the designed behavior. The two worktrees share `--git-common-dir`. Pin one of them with `AELFRICE_DB`. |
| `aelf search` returns "store is empty" | Run `aelf onboard .` from the project root. |
| `SQLite database is locked` under heavy concurrent writes | v1.1.0+ uses WAL and `busy_timeout=5000`. If you still see this error, file an issue with the repro. |
