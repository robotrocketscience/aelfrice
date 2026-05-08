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

## Optional hooks (v1.2+)

`aelf setup` wires only the `UserPromptSubmit` hook by default. The rest are opt-in:

```bash
aelf setup --transcript-ingest      # turn-by-turn capture; PreCompact rotates and re-ingests
aelf setup --commit-ingest          # PostToolUse:Bash hook ingests commit messages
aelf setup --session-start          # SessionStart hook injects locked beliefs at session boot
aelf setup --rebuilder              # PreCompact context rebuilder (alpha)
```

Each is independently uninstallable: `aelf unsetup --transcript-ingest`, etc.

All hooks are non-blocking. Every failure path returns exit 0 — a hook problem must never break a prompt or a commit.

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
