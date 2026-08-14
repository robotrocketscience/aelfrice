# Install

## Prerequisites

- Python 3.12 or 3.13. (`uv` manages the Python version for you. You do not have to install Python separately.)
- [`uv`](https://docs.astral.sh/uv/). This is the supported install channel (#730). If you do not have `uv`, run `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or any agent that can spawn a hook on `UserPromptSubmit`.
- Linux, macOS or Windows. Linux and macOS run the full test suite on every pull request. A smoke job covers Windows. See [LIMITATIONS.md § Compatibility](LIMITATIONS.md#compatibility) for what the smoke job asserts and what it does not assert.

## 1. Install the package

```bash
uv tool install aelfrice                # core (deps: numpy, scipy, snowballstemmer — local-only)
uv tool install "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

Or install from the source repository. This is the developer install:

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice && uv sync
```

This command installs eleven console scripts. The first script is `aelf`, the CLI. The other ten scripts are hook entry-points that the host spawns:

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

> **Migration from pipx or pip.** As of v3.0.x aelfrice supports only uv (#730). If you installed aelfrice with pipx, run `pipx uninstall aelfrice && uv tool install aelfrice` one time. If you installed aelfrice with pip, run `pip uninstall -y aelfrice && uv tool install aelfrice`. `aelf upgrade-cmd` gives the same migration line at the next upgrade check.

Verify the install:

```bash
aelf --version       # aelf X.Y.Z
which aelf           # which env owns the binary
```

## 2. Wire it into Claude Code

```bash
aelf setup
```

`aelf setup` is idempotent. Run it again each time you change Python environments. Run it again each time you move projects. The command writes these items:

1. **A `UserPromptSubmit` hook** in `settings.json`. Each prompt then goes through `aelf-hook` for retrieval before the agent receives the prompt.
2. **A `statusLine` notifier**. The notifier shows a one-line update banner only when a new release is available. The banner is empty at all other times.
3. **The full default-on auto-capture hook set and the bundled `/aelf:*` slash commands**. See the section "Hooks installed by `aelf setup`" below.

Automatic detection selects the scope and the command path:

| Run from… | `--scope` | `--command` |
|---|---|---|
| inside a project venv | `project` (writes `<project>/.claude/settings.json`) | `<project>/.venv/bin/aelf-hook`, or `<project>\.venv\Scripts\aelf-hook.exe` on Windows |
| a `uv tool`-installed `aelf` outside any venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` |
| a venv unrelated to `cwd` | `user` | first `aelf-hook` on `$PATH`, falls back to the active venv |

If you must set these values yourself, override the detection with `--scope user|project` and `--command /abs/path/aelf-hook`.

### Codex host

If you run Codex CLI, add `--host codex` to the same lifecycle verbs:

```bash
aelf setup --host codex      # hooks into $CODEX_HOME/hooks.json (else ~/.codex) + $aelf-* skills
aelf doctor --host codex     # verify wiring; reports the installed skill count
aelf unsetup --host codex    # remove the aelfrice hooks and skills together
```

`aelf setup --host codex` writes the hook set to the configuration home of Codex (#1052). The same command installs the `/aelf:*` slash-command bundle as `$aelf-*` agent skills under `~/.agents/skills/` (v4.1.0+). aelfrice generates the two surfaces from the same source bundle, therefore the two surfaces never become different. To install the hooks only, pass `--no-codex-skills`. The default is `--codex-skills`.

The skill install is idempotent and removes orphan skills. The skill install touches only the skills that aelfrice generated, because an `AELFRICE-CODEX-SKILL` marker controls replacement and removal. [SLASH_COMMANDS § Codex host](SLASH_COMMANDS.md#codex-host-aelf--skills) gives the full detail: the invocation, the generation transform and the approval warnings that are specific to Codex.

#### `$CODEX_HOME` (unreleased)

Codex reads its configuration from `$CODEX_HOME` when that variable has a non-empty value. In all other cases Codex reads its configuration from `~/.codex`. All three verbs resolve the home directory in the same way (#1427). Each of these configurations is therefore wired where that Codex reads:

- an isolated profile
- a CI runner
- a portable install
- two side-by-side Codex configurations

`aelf doctor --host codex` prints the resolved home on its first line. You can therefore see which directory the command inspected.

A `$CODEX_HOME` that does not exist is an error, not a condition for a fallback. A `$CODEX_HOME` that names something other than a directory is also an error. In both cases setup, doctor and unsetup refuse the operation. The three verbs create nothing and do not change `~/.codex`.

Both refusals agree with the behaviour of Codex. `CODEX_HOME=/nowhere codex mcp list` reports `CODEX_HOME points to "/nowhere", but that path does not exist` and then stops.

aelfrice does not create the directory for you. Codex refuses to start against such a configuration home, therefore a report of "setup succeeded" would be false. Create the directory, correct the variable, or unset the variable.

`$CODEX_HOME` does **not** move the `$aelf-*` agent skills. `~/.agents/skills/` is a standard path across agents. Other agents keep their skills in the same directory.

#### Shared `hooks.json` (unreleased)

`hooks.json` belongs to Codex, not to aelfrice. One document holds your own entries, the entries of another installer and the entries of aelfrice.

Setup and unsetup do three things to that document:

1. They take an advisory lock on a sibling `hooks.json.lock`.
2. They check the content hash of the file again immediately before they commit.
3. They replace the file atomically (#1428).

An aelfrice run therefore merges a concurrent edit. An aelfrice run never overwrites a concurrent edit.

`unsetup` does none of this against a home that has no `hooks.json`. In that case `unsetup` creates nothing, the lock file included.

Where a `hooks.json` was present, the zero-byte `hooks.json.lock` stays on disk. This is intentional. If another process holds that lock, deletion of the lock puts two writers on two inodes with no mutual exclusion at all.

If the file continues to change during three attempts, the command refuses and writes nothing. Run the command again.

Two refusals are new and intentional. aelfrice leaves a non-object `hooks` value byte-for-byte unchanged and reports it. aelfrice does the same with a non-list value on an event that aelfrice installs into. A `null` value gets the same treatment, because `null` is a value at that position and not an absence.

aelfrice can merge only into the documented `{"hooks": {"<Event>": [...]}}` shape. A change to any other shape would delete structure that aelfrice did not write. Correct the file, or pass `--force` to replace it.

One window is not closed. A process that takes no lock can still replace the file in the instant between our final check and the rename. To close that window, Codex must offer a mutation protocol. Codex does not offer such a protocol now.

## 3. Verify wiring

```bash
aelf doctor
```

`aelf doctor` runs two checks one after the other:

1. Hook resolution. The check compares every `command` in `settings.json` against `$PATH`. The check also reports the stale `bash <missing>.sh 2>/dev/null || true` wrappers from older installs.
2. The structural graph audit. The audit covers the orphan threads, the full-text search version 5 (FTS5) sync, the contradictions between locked beliefs, and the corpus volume.

`aelf doctor` exits 1 on any structural failure, therefore CI can gate on the command. An empty store on a new project is normal. The corpus-volume warning fires only when the project is at least 7 days old.

To run one half only, use one of these commands:

```bash
aelf doctor hooks      # hook resolution only
aelf doctor graph      # structural auditor only
```

### Duplicate hook entries (v4.2.0+)

`settings.json` is a shared global configuration file with several
independent writers: the harness itself, `aelf setup`, auto-install
and your own edits. A migration of the host settings, a merge of
dotfiles, or an edit by hand can leave two entries for the same
aelfrice hook. Both entries fire on every event. The only symptom is
that prompts become slower for no visible reason. Nothing is broken.
Both paths resolve. Every check passes.

`aelf doctor` now reports these entries. `--prune` collapses them:

```bash
aelf doctor hooks              # reports duplicates (read-only)
aelf doctor hooks --fix        # collapses them, keeping the first of each
```

The collapse touches only the `aelf-*` entries. The hooks that you
wrote yourself stay exactly as they are. This is also true when you
listed one of your own hooks two times on purpose. aelfrice counts the
duplicates separately from the stale-path prune, because the two
repairs answer different questions. A pruned entry pointed at a venv
that no longer exists. A collapsed entry resolved correctly and was
installed two times.

### Hook timeouts

Every hook that aelfrice installs has an explicit `timeout` (v4.2.0+).
The bundled manifest declares the timeout for each hook. A hook that
gates a user-visible action gets 15s. The bulk-ingest hooks get 30s.
The timeout is what bounds the "a hook must never block your prompt"
contract at the level of the host. Without the timeout, a hook can
wait for the SQLite write lock of another session. Such a hook then
stalls for the default of the host and not for the budget of aelfrice.

The budgets have headroom over the real worst case. The budgets are
not set close to that worst case. A cold-start retrieval on a store of approximately 46k beliefs
measured less than 3s. That retrieval had to rebuild its Best Matching
25 (BM25) sidecar from the start. If your store is very large, or if
your disk is slow, override the budget with
`aelf setup --timeout <seconds>`.

`aelf health` and `aelf stats` are still callable as back-compatible aliases. The default `--help` output hides them. `aelf --help --advanced` lists them. The canonical replacement for `health` is `aelf doctor graph`, the structural auditor. The canonical replacement for `stats` is `aelf status`. `aelf status` reports the counts.

## 4. Onboard a project

```bash
cd <project-root>
aelf onboard .
```

`aelf onboard .` walks the project and ingests the structural facts as candidate beliefs. The command reads the filesystem, the git log and the Python abstract syntax tree (AST). The command typically needs less than one second on a project of 50k lines of code. A second run is idempotent, because the command dedupes on `(source, sentence)`.

## 5. Lock the rules you care about

```bash
aelf lock "never push to main; use scripts/publish.sh"
aelf lock "all commits SSH-signed with ~/.ssh/id_rrs"
aelf locked                          # list what's locked
```

Locked beliefs short-circuit decay. aelfrice always returns the locked beliefs at L0. Locked beliefs are the beliefs that survive.

Restart Claude Code. The next prompt that mentions "push" will already have your rules attached.

## Database

The store is a SQLite database. The path resolution order is:

1. `$AELFRICE_DB` — the explicit override. aelfrice honours `:memory:`. That value is useful for tests.
2. `<git-common-dir>/aelfrice/memory.db` — used when `cwd` is inside a git work-tree (v1.1.0+). The worktrees of one repo share a single DB through `--git-common-dir`. Git does not track `.git/`, therefore the brain graph never crosses the git boundary.
3. `~/.aelfrice/memory.db` — the fallback for a directory that is not in a git work-tree.

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

`aelf migrate` opens the source DB read-only with the SQLite `mode=ro` URI. The project-mention filter is the default. That filter restricts the copy to the beliefs that name the active project. `--all` skips the filter. `aelf migrate` is idempotent on a second run.

### Batch ingest of historical sessions

If you have prior Claude Code sessions sitting at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, you can backfill them:

```bash
aelf ingest-transcript --batch ~/.claude/projects/
aelf ingest-transcript --batch ~/.claude/projects/ --since 2026-01-01
```

Auto-detects the JSONL format on a per-line basis (handles both aelfrice's transcript-logger output and Claude Code's internal session shape). Idempotent on re-run.

> **Privacy.** A session JSONL file can hold pasted secrets, customer data, or any other text that you typed in the chat. Batch ingestion brings all of that content into the local belief graph. The v1.2 ingest path has no scrubber for personally identifiable information (PII). Review the sessions before you backfill them.

## Hooks installed by `aelf setup`

Bare `aelf setup` installs the v1.2.0 auto-capture pipeline. The command installs that pipeline together with the read-side `UserPromptSubmit` retrieval hook:

| Hook | Event(s) | Default | What it does |
|---|---|---|---|
| UserPromptSubmit retrieval | `UserPromptSubmit` | always | injects the matched beliefs as an `<aelfrice-memory>` block |
| transcript-ingest | `UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact` | **on** | logs every turn to a JSONL file for each project. PreCompact rotates the file. PreCompact then ingests the file into beliefs and edges |
| commit-ingest | `PostToolUse:Bash` | **on** | each successful `git commit` runs the triple extractor on the message |
| session-start | `SessionStart` | **on** | new sessions open with the L0 locked beliefs already injected |
| stop-lock-prompt | `Stop` | **on** | prompts you to lock the correction-class beliefs (#582) and the directive beliefs (#1315) from this session |
| search-tool | `PreToolUse:Grep` / `Glob` | **on** (v3.0.1+) | checks the belief store before the agent's own Grep or Glob fires |
| search-tool-bash | `PreToolUse:Bash` | **on** (v3.0.1+) | checks the belief store before a shell grep, rg, find, fd or ack fires |
| pre-issue-guard | `PreToolUse:Bash` | **on** (v3.4.0+) | blocks `gh issue create` when the title overlaps an existing issue or a shipped commit at 0.5 Jaccard or above (#941) |
| claude-memory-mirror | `PostToolUse:Write` / `Edit` / `MultiEdit` | **on** (v3.7.0+) | mirrors the fact-file writes of the host claude-memory into the belief graph in one direction (#985). `AELFRICE_MIRROR_CLAUDE_MEMORY` or `[memory] mirror_claude_memory` enables the hook. Since v4.0 (#1089) the per-project consent sentinel also enables it. The one-shot reconcile writes that sentinel at the first `aelf setup`. A project that is set up therefore mirrors by default. An explicit env `0` or TOML `false` always wins over the sentinel. That is the opt-out |
| agent-context | `PreToolUse:Agent` / `Task` | **on** | dispatched subagents inherit the L0 locked beliefs and the task-relevant beliefs by prompt injection. The hook is a fail-open passthrough. To disable the injection, set `AELFRICE_AGENT_CONTEXT=0`. To opt out, pass `--no-agent-context` (#1068) |
| rebuilder | `PreCompact` (installed) — block ships on `SessionStart(source="compact")` | off | the retrieval-curated context rebuilder (augment-mode, v1.4 alpha). `--rebuilder` installs a `PreCompact` entry. That hook does the trigger-mode bookkeeping only. That hook injects nothing. The block itself ships *after* the compaction, on the default-on `SessionStart` hook that is already present ([#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)) |

To opt out of one hook, use the applicable option. The opt-out persists across upgrades in `~/.aelfrice/opt-out-hooks.json`.

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

The SessionStart recap line ("N beliefs written since last session", v3.5+, #934) has no manifest entry of its own. The SessionStart hook prints the recap line. `aelf setup --no-sessionstart-recap` suppresses the recap line at install time. `opt-out-hooks.json` does not persist that choice.

To opt in to the hooks that are off by default, use this command:

```bash
aelf setup --rebuilder                 # post-compaction context rebuilder (alpha)
```

`aelf unsetup` mirrors `aelf setup`. A bare invocation removes every default-on hook. The `--no-*` flags suppress the removal of the applicable hook.

All hooks fail open. Every failure path returns exit 0, because a hook problem must never break a prompt or a commit. The pre-issue-guard is the one intentional exception to the non-blocking behaviour. The pre-issue-guard exits 2 to block a duplicate `gh issue create`. To bypass the guard, set `ALLOW_DUP_ISSUE=1` in the host's environment. The failure paths of the guard still exit 0.

### Self-installing hook manifest (v3.0+)

`src/aelfrice/data/hook_manifest.json` declares the list of default-on hooks above. That file ships in the wheel. The first `aelf <cmd>` invocation after a fresh install does two things. The first `aelf <cmd>` invocation after a bare `uv tool upgrade aelfrice` does the same two things:

1. It reconciles the installed manifest version against `~/.aelfrice/installed-manifest-version`.
2. It merges any new entries into `~/.claude/settings.json` automatically.

You therefore no longer have to remember to run `aelf setup` again after a bare upgrade with the package manager. The hooks that a newer release adds arrive without that step.

What auto-install does:

* In the usual path the stamp is equal to the installed version. That path is one stat and one short file read. It runs no JSON parse. It does not read settings.json.
* On a mismatch it takes an `fcntl` exclusive lock on `~/.aelfrice/.auto-install.lock`. Concurrent `aelf` invocations therefore cannot race on the merge.
* It reuses the same install primitives that `aelf setup` calls. The shape of settings.json on disk is byte-identical.
* It adds only the entries that the manifest claims by basename. It preserves everything that the user added to settings.json by hand.
* It respects the opt-outs. If you ever ran `aelf setup --no-transcript-ingest`, `~/.aelfrice/opt-out-hooks.json` persists that choice, and the choice survives upgrades. To rescind the opt-out, run `aelf setup` again without the `--no-*` flag.
* It prints a single stderr line when it added entries: `aelfrice: hooks updated to v3.5.0 (was v3.4.0) — added: pre_issue_guard`.

Opt-out controls:

```bash
export AELFRICE_NO_AUTO_INSTALL=1   # power user: I manage settings.json by hand
aelf setup --no-stop-hook           # disable one hook; persists across upgrades
```

`aelf doctor` flags a broken hook command and a hook command that does not resolve. `aelf doctor` also reports when one of the four v2.1 auto-capture hooks is missing. Those four hooks are transcript-ingest, commit-ingest, session-start and stop-hook.

`aelf doctor` does not reconcile settings.json against the manifest. `aelf doctor` therefore does not flag the newer manifest hooks, search-tool and pre-issue-guard, when they are absent. The auto-installer is the reconciliation path.

> **Privacy note.** transcript-ingest is on by default. Every turn that you type therefore goes into the per-project SQLite DB at the `PreCompact` rotation. The DB is local-only: it uses no network and sends no telemetry. See § "What you get for free" in the README and [PRIVACY.md](PRIVACY.md).
>
> The JSONL file has no PII scrubber. If you paste secrets, customer data, or anything else that you do not want indexed in the chat, opt out with `--no-transcript-ingest`. Then use `aelf lock` or `aelf onboard` for explicit ingestion only.

### Legacy-schema detection + auto-migrate (`aelf doctor`, v3.0+)

`aelf doctor` scans all per-project DBs under `~/.aelfrice/projects/*/memory.db`. `aelf doctor` migrates in place every DB that uses the pre-v1.x schema. A DB on the pre-v1.x schema has no `origin` column on the `beliefs` table.

A DB on the old schema cannot participate in the v2.x and v3.0 lifecycle, because the columns that track the lifecycle are absent. That lifecycle includes `agent_remembered`, `user_validated`, the calibrated weights, `aelf:promote` and the federation `scope`.

The auto-migrate behaviour (v3.0+, [#593](https://github.com/robotrocketscience/aelfrice/issues/593)) is this. For each legacy DB that it detects, doctor does three things:

1. doctor renames `memory.db` to `memory.db.pre-v1x.bak` with an atomic POSIX rename.
2. doctor runs the existing `migrate()` core with `copy_all=True, apply=True` against the backup.
3. doctor writes a fresh modern-schema DB back at the original path.

Each copied belief gets `origin=ORIGIN_UNKNOWN`. doctor preserves the backup verbatim and never overwrites it. If a stale `<path>.pre-v1x.bak` from an earlier failed run is already present, the migration aborts for that DB. `report.failed_migrate_dbs` then surfaces the failure. doctor does not overwrite the recoverable state.

The success path emits one line per DB:

```
migrated ~/.aelfrice/projects/2e7ed55e017a/memory.db: 35,332 beliefs, 412ms (backup at ~/.aelfrice/projects/2e7ed55e017a/memory.db.pre-v1x.bak)
```

doctor skips an empty DB (zero rows) silently. A failure prints a residual warning. That warning points at the manual `aelf migrate --from <path> --apply` path for that specific DB.

Use these commands for a manual migration, for example a DB outside the standard `~/.aelfrice/projects/` tree:

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

Since [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) every aelfrice mutation of that file takes an exclusive lock on a sibling `settings.json.lock`. Each mutation then writes one time. The writers of aelfrice can therefore no longer overwrite each other. You may see two consequences:

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

Something that does not take the aelfrice lock replaced the file during the install. In practice that something is the host. aelfrice does not commit over the new file, because a commit would discard what the host just wrote. Run the command again. Every change that aelfrice makes to this file is convergent, therefore a second run is safe.

You cannot make the host take the aelfrice lock. This check is therefore a detector and not a cure. Before the check, a setting was lost silently. The check now gives you a message that tells you to try again.

### Incomplete store migrations (v4.2.0+)

An open of the store runs a set of one-shot schema migrations. Each migration is stamped as complete only after it finishes its work.

Before [#1161](https://github.com/robotrocketscience/aelfrice/issues/1161) a migration that raised an error took the whole store with it. The completion marker was never written. The next open therefore re-ran the same pass and failed the same way. Every entry point opens the store: the CLI, the hooks and MCP. The only way back was an edit of the SQLite database by hand.

A migration that cannot finish no longer stops the store from opening. aelfrice records the failure and skips the pass. The store keeps its pre-migration shape. Reads and writes work. The completion marker stays unset, therefore aelfrice retries the migration on every subsequent open. An upgrade to a build that fixes the migration thus repairs the store automatically.

`aelf doctor` is where that state surfaces:

```
store migration(s) INCOMPLETE — the store opens and is usable, but one or more one-shot migrations could not finish:
  _maybe_consolidate_content_hash_duplicates: IntegrityError('UNIQUE constraint failed: edges.src, edges.dst, edges.type')
fix: these retry automatically on every open, so upgrading (`aelf upgrade`) is the first thing to try. If the same
migration keeps failing, report the error above — the store will keep working in the meantime.
```

Try `aelf upgrade` first. If the same migration keeps failing, include the error text above in a report. No urgent action is necessary, because the store stays usable in the meantime.

### Pruning dormant per-project DBs (`aelf doctor --prune-dormant`, v3.0+)

Some per-project DBs hold beliefs from projects that you worked on briefly with an older aelfrice version and then abandoned. Nothing migrates those DBs. Nothing touches them. They stay on disk. `aelf doctor --prune-dormant` lists the DBs whose `memory.db` mtime is older than `--idle-days`. The default is 30 days. The command lets you delete those DBs one at a time.

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

`--apply` prompts you for each DB. The default at the prompt is N. Any answer other than `y` or `yes` preserves the file. A bare Enter and a pipe from `/dev/null` are such answers.

There is no `--yes` shortcut, because a destructive deletion is always explicit and for one DB at a time. Unlike `aelf migrate`, this command does not move beliefs anywhere. The command only removes the DB file.

The dormant scan does not examine the schema. The scan flags a pre-v1.x DB and a modern-schema DB equally when the DB is idle. You may still want to migrate a DB. In that case, run `aelf migrate --from <path> --apply` (above) before you prune the DB, not after.

## Update notifier

```bash
aelf upgrade-cmd          # prints the canonical upgrade command (`run: …`) when an
                          # update is available; otherwise `aelfrice is up to date`
aelf upgrade-cmd --check  # same output, no behaviour difference
/aelf:upgrade             # imperative slash: detect + run + re-setup
```

`aelf upgrade-cmd` emits `run: uv tool upgrade aelfrice` on a uv-managed install. If another tool (pipx, pip, system) installed aelfrice, the `run:` line is the migration chain instead. That chain is `pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent. uv is the single supported install channel (#730).

The CLI does not execute the upgrade itself. A replacement of the running interpreter mid-process is unreliable on Windows.

The orange statusline banner appears automatically when an update is available. To disable the banner, run `export AELF_NO_UPDATE_CHECK=1`.

## Uninstall

You must pick a disposition for the DB:

```bash
aelf uninstall --keep-db              # leave the DB in place (safe default)
aelf uninstall --archive backup.aenc  # encrypt to file then delete
aelf uninstall --purge                # permanently delete (three confirmation gates)
uv tool uninstall aelfrice            # finally remove the wheel
```

`--archive` uses Fernet with AES-128-CBC, HMAC and a scrypt-derived key. To recover the DB later, use this code:

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
- the log of the hook injection for each turn
- the feed log of the belief writes
- the `transcripts/`, `rebuild_logs/` and `telemetry/` directories

Several of those items hold belief content verbatim.

`--purge` and `--archive` both operate on that whole set. Each one
prints the manifest before it destroys anything. The manifest gives
every path and its size. You can therefore check the total against
what you expect.

`--archive` encrypts **the belief database only**. Every other item in
the set is either derived from the database or a rolling capture
buffer. The derived items are the BM25F index, the injection log, the
feed log and the telemetry. The rolling capture buffer is
`transcripts/`. The command deletes those items rather than adding
them to the archive.

The command reports this behaviour. The command asks for your
confirmation before it takes your password. An encrypted archive next
to plaintext copies of the same content is not a guarantee.

Before v4.2.0 both flags deleted only `memory.db`. `--archive` also
did not checkpoint the write-ahead log (WAL) first. On a store held
open by a running hook that meant the archive captured a stale
database. In the worst case the archive captured an empty database.
The real content stayed on disk in plaintext in `memory.db-wal`
(#1173).

If you archived a store under an earlier version, that archive may be
incomplete. Check the archive with `decrypt_archive` before you rely
on it.

aelfrice creates a `.git/aelfrice/` directory or a `~/.aelfrice/`
directory. If `$AELFRICE_DB` points somewhere other than such a
directory, aelfrice cannot safely attribute the generically-named
artifacts to itself. `$HOME/transcripts/` can be your own directory. aelfrice
therefore lists those artifacts for you to remove by hand instead of
deleting them.

### `~/.aelfrice/`

That directory is a second location with different ownership. It
belongs to no single store. Uninstall therefore removes what aelfrice
put there **by name**. Uninstall never sweeps the directory.

The install state goes in **every** mode, `--keep-db` included. Each
of those files records that a step already happened. A file that
survived would make a reinstall read a stale decision as current. The
install state is:

- the **LLM classification consent sentinel**. This one is
  load-bearing. The sentinel used to outlive the uninstall. A later
  reinstall therefore read the grant as still valid and never
  re-prompted (#1186). The sentinel now goes. A reinstall asks
  again.
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

These items are kept in every mode:

- `projects/` holds **other projects' stores**. There is one
  `memory.db` for each project-id slug. `aelf uninstall` disposes of
  one store, the store that `db_path()` resolves to. Removal of
  `projects/` would destroy belief corpora that you did not ask
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

If you want the directory gone entirely, first check
`ls ~/.aelfrice/projects/` and `ls ~/.aelfrice/shared/`. Then run
`uninstall`. Then run `rm -rf ~/.aelfrice/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | Confirm that `~/.local/bin`, the shim directory of the uv tool, is on `$PATH`. `uv tool update-shell` adds it for you. |
| Hook fires but no `<aelfrice-memory>` block appears | Run `aelf doctor`. Usually the hook command points at a deleted script. |
| `aelf doctor` says "skipped (shell metacharacters)" on a hook line | The install is stale. `aelf setup` rewrites the hook in place. |
| Two worktrees of the same repo see the same beliefs | This is the designed behaviour. The two worktrees share `--git-common-dir`. Pin one of them with `AELFRICE_DB`. |
| `aelf search` returns "store is empty" | Run `aelf onboard .` from the project root. |
| `SQLite database is locked` under heavy concurrent writes | v1.1.0+ uses WAL and `busy_timeout=5000`. If you still see this error, file an issue with the repro. |
