# Install

## Prerequisites

- Python 3.12 or 3.13.
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`. `uv` handles the Python version for you.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or any agent that can spawn a hook on `UserPromptSubmit`.

## 1. Install the package

```bash
pip install aelfrice                # core, zero runtime deps
pip install "aelfrice[mcp]"         # add the MCP server (fastmcp)
pip install "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

Or with `uv`:

```bash
uv tool install aelfrice
```

Or from source:

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice && uv sync
```

This installs two console scripts: `aelf` (the CLI) and `aelf-hook` (the hook entry-point Claude Code spawns on each prompt).

Verify:

```bash
aelf --version       # aelfrice X.Y.Z
which aelf           # which env owns the binary
```

## 2. Wire it into Claude Code

```bash
aelf setup
```

This is idempotent. Run it again any time you change Python envs or move projects. It writes:

1. **A `UserPromptSubmit` hook** to `settings.json` so each prompt routes through `aelf-hook` for retrieval before the agent sees it.
2. **A `statusLine` notifier** that surfaces a one-line update banner only when a new release is available (empty otherwise).

Auto-detection picks the right scope and command path:

| Run from… | `--scope` | `--command` |
|---|---|---|
| inside a project venv | `project` (writes `<project>/.claude/settings.json`) | `<project>/.venv/bin/aelf-hook` |
| a `pipx`-installed `aelf` outside any venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` |
| a venv unrelated to `cwd` | `user` | first `aelf-hook` on `$PATH`, falls back to the active venv |

Override with `--scope user|project` and `--command /abs/path/aelf-hook` when you need to.

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

`aelf health` and `aelf stats` remain callable as back-compat aliases — hidden from default `--help` output but listed under `aelf --help --advanced`. The canonical replacements are `aelf doctor graph` and `aelf status`.

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

---

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

---

## Hooks installed by `aelf setup`

Bare `aelf setup` wires the v1.2.0 auto-capture pipeline alongside the read-side `UserPromptSubmit` retrieval hook:

| Hook | Event(s) | Default | What it does |
|---|---|---|---|
| UserPromptSubmit retrieval | `UserPromptSubmit` | always | injects matched beliefs as `<aelfrice-memory>` block |
| transcript-ingest | `UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact` | **on** | logs every turn to a per-project JSONL; PreCompact rotates the file and ingests it into beliefs/edges |
| commit-ingest | `PostToolUse:Bash` | **on** | each successful `git commit` runs the triple extractor on the message |
| session-start | `SessionStart` | **on** | new sessions open with L0 locked beliefs already injected |
| rebuilder | `PreCompact` | off | retrieval-curated context rebuilder (augment-mode, v1.4 alpha) |
| search-tool | `PreToolUse:Grep` / `Glob` | off | belief-store check before the agent's own search tool fires |

Opt out per-hook:

```bash
aelf setup --no-transcript-ingest      # skip the four transcript-logger hooks
aelf setup --no-commit-ingest          # skip the commit-message ingest hook
aelf setup --no-session-start          # skip the SessionStart locked-belief injection
```

Opt in to the off-by-default hooks:

```bash
aelf setup --rebuilder                 # PreCompact context rebuilder (alpha)
aelf setup --search-tool               # PreToolUse:Grep|Glob memory-first search
```

`aelf unsetup` mirrors: bare invocation removes every default-on hook. `--no-*` flags suppress per-hook removal.

All hooks are non-blocking. Every failure path returns exit 0 — a hook problem must never break a prompt or a commit.

### Self-installing hook manifest (v2.2+)

The list of default-on hooks above is declared in `src/aelfrice/data/hook_manifest.json` and ships in the wheel. The first `aelf <cmd>` invocation after a fresh install or a bare `pipx upgrade aelfrice` (or `uv tool upgrade aelfrice` / `pip install -U aelfrice`) reconciles the installed manifest version against `~/.aelfrice/installed-manifest-version` and merges any new entries into `~/.claude/settings.json` automatically. This closes the loop on bare package-manager upgrades — you no longer have to remember to re-run `aelf setup` to pick up hooks added in newer releases.

What auto-install does:

* Happy path (stamp == installed version) is one stat + one short file read. No JSON parse, no settings.json read.
* On mismatch, takes an `fcntl` exclusive lock on `~/.aelfrice/.auto-install.lock` so concurrent `aelf` invocations cannot race on the merge.
* Reuses the same install primitives `aelf setup` calls; the on-disk shape of settings.json is byte-identical.
* Adds only entries the manifest claims by basename — anything the user added to settings.json by hand is preserved.
* Respects opt-outs: if you ever ran `aelf setup --no-transcript-ingest`, that choice is persisted at `~/.aelfrice/opt-out-hooks.json` and survives upgrades. Re-running `aelf setup` (without the `--no-*` flag) rescinds the opt-out.
* Prints a single stderr line when entries were actually added: `aelfrice: hooks updated to v2.2.0 (was v2.1.0) — added: stop_lock_prompt`.

Opt-out controls:

```bash
export AELFRICE_NO_AUTO_INSTALL=1   # power user: I manage settings.json by hand
aelf setup --no-stop-hook           # disable one hook; persists across upgrades
```

`aelf doctor` continues to flag drift — manual edits to settings.json that diverge from the manifest, or hooks the auto-installer would write that the file is missing.

> **Privacy note.** Default-on transcript-ingest means every turn you type lands in the per-project SQLite DB on `PreCompact` rotation. The DB is local-only (no network, no telemetry — see § "Your data stays yours" in the README) but the JSONL has no PII scrubber. If you paste secrets, customer data, or anything you don't want indexed in chat, opt out with `--no-transcript-ingest` and use `aelf lock` / `aelf onboard` for explicit ingestion only.

### Legacy-schema detection (`aelf doctor`, v2.1+)

`aelf doctor` scans all per-project DBs under `~/.aelfrice/projects/*/memory.db` and flags any that use the pre-v1.x schema (no `origin` column on the `beliefs` table) and have at least one row. DBs on the old schema cannot participate in the v2.x lifecycle — `agent_remembered`, `user_validated`, calibrated weights, `aelf:promote` — because the column that tracks origin is absent.

When legacy DBs are found the doctor report appends a block like:

```
legacy-schema per-project DBs detected (pre-v1.x, no `origin` column).
  ~/.aelfrice/projects/2e7ed55e017a/memory.db (35,332 beliefs, idle 16d)
  ~/.aelfrice/projects/18a856c7a96b/memory.db (6,283 beliefs, idle 13d)
fix: `aelf migrate --from <path> --apply` per DB to copy beliefs
     into the current project's modern-schema DB.
```

The block is quiet when every scanned DB already has the `origin` column. Empty DBs (zero rows) are silently skipped.

To migrate a legacy DB:

```bash
aelf migrate --from ~/.aelfrice/projects/<id>/memory.db          # dry-run
aelf migrate --from ~/.aelfrice/projects/<id>/memory.db --apply  # write
```

### Pruning dormant per-project DBs (`aelf doctor --prune-dormant`, v2.1+)

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

---

## Update notifier

```bash
aelf upgrade           # prints the right pip-upgrade line for your env
aelf upgrade --check   # yes/no, no command line printed
```

`aelf upgrade` detects venv vs pipx vs system and tells you the line. It does not run pip itself — replacing the running interpreter mid-process is unreliable on Windows.

The orange statusline banner appears automatically when an update is available. Disable with `export AELF_NO_UPDATE_CHECK=1`.

---

## Uninstall

You must pick a disposition for the DB:

```bash
aelf uninstall --keep-db              # leave the DB in place (safe default)
aelf uninstall --archive backup.aenc  # encrypt to file then delete
aelf uninstall --purge                # permanently delete (three confirmation gates)
pip uninstall aelfrice                # finally remove the wheel
```

`--archive` uses Fernet (AES-128-CBC + HMAC, scrypt-derived key). Recover later:

```python
from aelfrice.lifecycle import decrypt_archive
open("out.db","wb").write(decrypt_archive("backup.aenc","password"))
```

Requires the `[archive]` extra.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | Confirm `~/.local/bin` (pipx) or `<venv>/bin` is on `$PATH`. |
| Hook fires but no `<aelfrice-memory>` block appears | `aelf doctor` — usually the hook command points at a deleted script. |
| `aelf doctor` says "skipped (shell metacharacters)" on a hook line | Stale install. `aelf setup` rewrites the hook in place. |
| Two worktrees of the same repo see the same beliefs | Working as designed — they share `--git-common-dir`. Pin one with `AELFRICE_DB`. |
| `aelf search` returns "store is empty" | Run `aelf onboard .` from the project root. |
| `SQLite database is locked` under heavy concurrent writes | v1.1.0+ uses WAL + `busy_timeout=5000`. If you still see it, file an issue with the repro. |
