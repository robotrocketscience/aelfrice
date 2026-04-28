# INSTALL

## Prerequisites

- Python 3.12 or 3.13
- Either `pip` (works) or [`uv`](https://docs.astral.sh/uv/) (recommended, used in CI)

## Install from PyPI

```bash
pip install aelfrice                # core (zero runtime deps)
pip install "aelfrice[mcp]"         # add the MCP server (fastmcp)
pip install "aelfrice[archive]"     # add the encrypted-archive uninstall path
```

Two console scripts: `aelf` (the CLI) and `aelf-hook` (the Claude Code hook entry-point).

Verify:

```bash
aelf --version       # prints "aelfrice X.Y.Z"
aelf health          # prints "brain mode: insufficient_data" on a fresh DB
which aelf           # confirms which Python env owns the binary
```

## Install from source

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice
uv sync                  # core
uv sync --extra mcp      # add MCP server
uv sync --extra dev      # add pytest, pyright
```

## Database

SQLite. Resolution order:

1. `$AELFRICE_DB` if set (override; use `:memory:` for tests).
2. `<git-common-dir>/aelfrice/memory.db` when `cwd` is inside a git work-tree (v1.1.0). Two worktrees of one repo share a `--git-common-dir` and therefore one DB. `.git/` is not git-tracked, so the brain graph never crosses the git boundary.
3. `~/.aelfrice/memory.db` as the fallback for non-git directories.

Pin per-project (overriding the chain) with `export AELFRICE_DB=/abs/path/.aelfrice.db` — handy via direnv.

### Migrating from v1.0.x

If you ran v1.0.x you have a single global DB at `~/.aelfrice/memory.db`. v1.1.0+ resolves per-project under `.git/aelfrice/memory.db`. `aelf migrate` ports beliefs forward (added in v1.1.0, [PR #104](https://github.com/robotrocketscience/aelfrice/pull/104)):

```bash
cd <project-root>
aelf migrate              # dry-run; reports what would copy
aelf migrate --apply      # actually copy filtered beliefs into the project DB
aelf migrate --apply --all  # copy every belief from the legacy global DB
aelf migrate --from /alt/path/memory.db --apply  # source override
```

`aelf migrate` opens the source DB read-only (SQLite `mode=ro` URI) so it can never accidentally write back. Idempotent on re-run. Project-mention filtering (default) restricts the copy to beliefs whose content references the active project's name; `--all` skips the filter.

### Batch ingest of historical sessions

If you have prior Claude Code sessions sitting at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, you can backfill them into the active project's belief store with a single batch invocation (added in v1.2.0+; issue [#115](https://github.com/robotrocketscience/aelfrice/issues/115)):

```bash
cd <project-root>
aelf ingest-transcript --batch ~/.claude/projects/                     # ingest everything
aelf ingest-transcript --batch ~/.claude/projects/ --since 2026-01-01   # only mtime ≥ cutoff
```

Auto-detects the JSONL format on a per-line basis: handles both aelfrice's own transcript-logger `turns.jsonl` archives and Claude Code's internal session logs. Idempotent — re-running on the same directory inserts zero new beliefs because `ingest_turn` dedupes per `(source, sentence)`. `aelf setup` prints a one-line hint with the file count and the exact batch command when historical JSONLs are detected during install.

> **Privacy trade-off.** Session JSONLs may contain pasted secrets, API keys, conversation transcripts, customer data, or anything else you typed at Claude Code in the past. Batch ingestion brings all of that into the local belief graph at `<repo>/.git/aelfrice/memory.db` — same machine, same disk, never leaves the host (no telemetry, no network call), but it does end up in a queryable index. There is no PII scrubber wired into the v1.2 ingest path; secret-stripping during ingestion is a v1.3+ task. Until then, **review the JSONLs before running `--batch` against any directory you wouldn't want re-emitted via retrieval.** Use `--since` to scope to recent sessions if older logs predate your secret-handling discipline.

## Wire into Claude Code

```bash
aelf setup                                        # auto-detect scope + path
aelf setup --scope user                            # force user-scope (~/.claude/settings.json)
aelf setup --scope project --project-root .        # force project-scope (.claude/settings.json)
aelf setup --no-statusline                         # hook only, no statusline
aelf setup --transcript-ingest                     # opt in to live conversation capture (v1.2+)
aelf setup --commit-ingest                         # opt in to commit-message ingest (v1.2+)
aelf unsetup                                       # remove the prompt hook + statusline
aelf unsetup --transcript-ingest                   # remove the transcript-logger hooks too
aelf unsetup --commit-ingest                       # also remove the commit-ingest hook
aelf doctor                                        # verify hook commands resolve
```

Idempotent. `aelf setup` wires two things by default:

1. **`UserPromptSubmit` hook** (`aelf-hook`). Reads each prompt payload, runs retrieval, emits an `<aelfrice-memory>...</aelfrice-memory>` block above the prompt. Every failure mode exits 0 with no output — the hook can't block a prompt.
2. **`statusLine` notifier** (`aelf statusline`). Prints an orange one-line update banner in the Claude Code statusbar **only when an update is available**, empty otherwise. Reads the cached PyPI check; never makes network calls. Banner disappears automatically once you've upgraded.

`--transcript-ingest` (v1.2+) additionally wires four hook events (`UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact`) to the `aelf-transcript-logger` entry point. Each conversation turn lands as one JSONL line in `<git-common-dir>/aelfrice/transcripts/turns.jsonl`; `PreCompact` rotates the log into a sibling `archive/` directory and spawns `aelf ingest-transcript` detached so the turns enter the brain graph as session-tagged beliefs with `DERIVED_FROM` edges between consecutive turns. Sub-10ms per-turn budget; non-blocking on every failure path. Storage stays under `.git/`, which git does not track — transcripts never cross the git boundary. Opt-in at v1.2; the default may flip on once disk-footprint and latency telemetry are in.

`--commit-ingest` (v1.2+) additionally wires a `PostToolUse:Bash` entry to the `aelf-commit-ingest` entry point. After every successful `git commit`, the hook parses the commit message, runs the triple extractor on the prose, and persists the resulting beliefs and edges under a session id derived as `sha256(branch + ":" + commit_hash)[:16]` (stable across re-fires; idempotent against the same commit). Median-latency budget is 30 ms; the hook bails non-blockingly on every failure mode so a `git commit` never feels broken. Opt-in at v1.2; default-flip candidate at v1.3 once production-corpus latency telemetry confirms the budget holds.

### How `aelf setup` picks scope and command path

Without explicit flags, `aelf setup` looks at the active interpreter and current directory and routes the install correctly so the same command works in three layouts:

| You ran `aelf setup` from… | `--scope` resolves to | `--command` resolves to |
|---|---|---|
| inside a project venv (e.g. `cd ~/projects/foo && .venv/bin/aelf setup`) | `project` (writes `<project>/.claude/settings.json`) | absolute path: `<project>/.venv/bin/aelf-hook` |
| a pipx-installed `aelf` outside any project venv | `user` (writes `~/.claude/settings.json`) | first `aelf-hook` on `$PATH` (typically `~/.local/bin/aelf-hook`) |
| a venv in some other repo | `user` | first `aelf-hook` on `$PATH`, falling back to the active venv |

Effect: when Claude Code is launched in `~/projects/foo`, the `UserPromptSubmit` hook runs that project's venv `aelf-hook`. When launched anywhere else, the global pipx hook runs. No `$PATH` collision, no dangling symlinks, no per-machine scripting needed.

**Recommended setup for working on multiple projects:**

```bash
pipx install aelfrice                                 # global default lives in ~/.local/bin/
aelf setup                                            # writes ~/.claude/settings.json -> ~/.local/bin/aelf-hook

cd ~/projects/foo && uv sync                          # project venv
.venv/bin/aelf setup                                  # writes <project>/.claude/settings.json -> <project>/.venv/bin/aelf-hook
```

Override either resolution at any time with explicit `--scope` / `--command`.

`aelf setup` also silently removes the legacy `/usr/local/bin/aelf{,-hook}` symlinks if they point at a deleted target — these were written by older install scripts and otherwise shadow a healthy `pipx`-managed `aelf` on `$PATH`. Real files and live symlinks are never touched.

### `aelf doctor`

Run after any setup (or at any time) to verify wiring. Walks `~/.claude/settings.json` and `<cwd>/.claude/settings.json` (when present), inspects every hook command and the top-level `statusLine`, and reports any program token whose absolute path is missing or whose bare name isn't on `$PATH`. Exits `1` when at least one broken command is found, so you can gate CI on it.

```bash
aelf doctor                                  # scan both scopes
aelf doctor --user-settings /tmp/test.json   # explicit user settings file
aelf doctor --project-root /path/to/repo     # explicit project root
```

Composition with an existing `statusLine`:

| Existing command shape | What `aelf setup` does |
|---|---|
| empty / not configured | install standalone `aelf statusline` |
| already `aelf statusline` (any form) | idempotent no-op |
| simple existing command (e.g. `echo my-bar`) | append ` ; aelf statusline 2>/dev/null` to it |
| complex (`\|`, `<<`, `&&`, `\|\|`, `` ` ``, `\`) | left untouched, hint printed to stderr |

`aelf unsetup` reverses surgically: drops the `statusLine` field if it's just ours, strips the `; aelf statusline ...` suffix to restore the original command otherwise.

## Update notifier

Every `aelf <cmd>` invocation and every `UserPromptSubmit` hook fire kicks off a TTL-gated **fire-and-forget** background check against `https://pypi.org/pypi/aelfrice/json`. The result lands at `~/.cache/aelfrice/update_check.json`. Cache is good for 24 hours; subsequent calls within the window are no-ops.

Opt out:

```bash
export AELF_NO_UPDATE_CHECK=1     # disables the check globally
```

When the cache says an update is available you'll see two things:

1. The orange `⬆ aelfrice X.Y.Z available, run: aelf upgrade │ ` snippet in your Claude Code statusline (if wired).
2. The same banner printed to stderr after most `aelf` CLI commands. Skipped for `aelf upgrade`, `aelf uninstall`, and `aelf statusline` so it doesn't double-print or stomp on machine-readable output.

To upgrade:

```bash
aelf upgrade           # prints the right pip-upgrade line for this env
aelf upgrade --check   # yes/no, no command line printed
```

`aelf upgrade` detects venv vs pipx vs system and adjusts:

| Context | Command emitted |
|---|---|
| pipx-managed install | `pipx upgrade aelfrice` |
| generic venv (incl. `uv venv`, conda) | `pip install --upgrade aelfrice` |
| system / user-site | `pip install --user --upgrade aelfrice` |

When an update is available, the output also shows the wheel's published SHA-256 plus the PyPI release URL so you can hash-pin the install:

```bash
pip install aelfrice==1.2.3 --hash=sha256:abc123…  # pip-side strong integrity
```

pip already verifies SHA-256 on every download against PyPI's JSON API; surfacing the hash here just lets the security-conscious user feed it into `--require-hashes` mode if they want.

## Wire into Codex / generic MCP hosts

Don't run `aelf setup` — that writes Claude-specific entries. Register the server directly:

```json
{
  "mcpServers": {
    "aelfrice": {
      "command": "aelf",
      "args": ["serve"]
    }
  }
}
```

Or, from a checkout: `["uv", "run", "--project", "/abs/path/to/aelfrice", "python", "-m", "aelfrice.mcp_server"]`. The `[mcp]` extra is required. Tools register under `aelf:*`.

## Run the benchmark

```bash
aelf bench
```

Prints a single JSON document with `hit_at_1` / `hit_at_3` / `hit_at_5` / `mrr` and `p50_latency_ms` / `p99_latency_ms` against a deterministic 16-belief × 16-query corpus. `--db PATH` to use a real DB; `--top-k N` to override hit-depth.

## Development

```bash
uv sync --extra dev
uv run pytest                # ~530 tests, ~7s
uv run pyright               # strict
uv run pytest -m regression  # cumulative integration scenarios
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aelf: command not found` | run via `uv run aelf …` or `pipx install aelfrice` |
| `ModuleNotFoundError: fastmcp` | `pip install "aelfrice[mcp]"` |
| `unable to open database file` | parent dir of `$AELFRICE_DB` doesn't exist |
| Hook registered, no memory block | DB is empty — run `aelf onboard <project>` |
| MCP tools not visible in host | restart the host — MCP servers load at launch |
| Hook silently fails outside a project venv | run `aelf doctor` — most likely a stale absolute path or bare `aelf-hook` not on `$PATH`. Re-run `aelf setup` from the matching install context |
| Wrong project's memory shows up in another project | each project should have its own `.claude/settings.json`. From inside the project venv: `.venv/bin/aelf setup` (auto-routes to project scope) |

## Uninstall

aelfrice ships an explicit teardown command. You **must** pick exactly one of `--keep-db`, `--purge`, or `--archive PATH` so the brain-graph SQLite DB doesn't get accidentally lost or accidentally retained. The command operates on the resolved DB path for the current `cwd` (see [§ Database](#database) for the resolution chain).

```bash
aelf uninstall --keep-db                           # safe default for review
aelf uninstall --archive ~/aelf-backup.aenc        # encrypt then delete
aelf uninstall --purge                              # permanently delete (gates fire)
pip uninstall aelfrice                              # finally, remove the wheel
```

Verification:

```bash
aelf --version 2>&1 | grep -q aelfrice || echo "wheel removed"
# Resolve DB path for current cwd, then test it:
DB=$(uv run python -c "from aelfrice.cli import db_path; print(db_path())")
test -f "$DB" && echo "db still there at $DB" || echo "db gone"
```

### `--keep-db`

DB preserved at the resolved path. The default also runs `unsetup` so the Claude Code hook + statusline are removed from `~/.claude/settings.json`. Pass `--keep-hook` if you want to keep those wired (e.g. you plan to reinstall and skip onboarding).

### `--archive PATH`

Encrypts the DB to PATH and deletes the original. Format:

```
8 bytes:   "AELFENC1" magic
16 bytes:  scrypt salt
remainder: Fernet-encrypted SQLite bytes (AES-128-CBC + HMAC-SHA256)
```

Key derivation: `scrypt(password, salt, N=2**14, r=8, p=1, len=32)`, base64-urlsafe-encoded for Fernet. The salt is embedded so only the password is needed for recovery — no metadata to remember.

Password input:

```bash
aelf uninstall --archive PATH                       # interactive prompt (twice, must match)
echo -n "secret" | aelf uninstall --archive PATH --password-stdin
```

Never pass the password on argv — it would leak via `ps` / `/proc/cmdline`.

Recovery later:

```python
from aelfrice.lifecycle import decrypt_archive
data = decrypt_archive(Path("/tmp/aelf-backup.aenc"), "secret")
Path("/tmp/restored.db").write_bytes(data)
# Now AELFRICE_DB=/tmp/restored.db aelf search "..." works.
```

Wrong password raises `cryptography.fernet.InvalidToken`. Requires `pip install 'aelfrice[archive]'`; without the extra, the CLI prints a clear install hint and exits non-zero.

### `--purge`

Permanently deletes the resolved DB. Three gates fire before deletion:

1. The `--purge` flag must be passed explicitly. Default behavior is an error pointing at `--help`.
2. Print the target path + size, then require the user to type `PURGE` verbatim (case-sensitive).
3. Final `[y/N]` confirmation.

`--yes` skips prompts (1)–(3). It does **not** auto-pass `--purge`. There is no env-var override for the `--purge` flag; if you want to script unattended teardown you must explicitly add both `--purge --yes`.
