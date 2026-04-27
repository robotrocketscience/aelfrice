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

SQLite at `~/.aelfrice/memory.db` by default. Override with `$AELFRICE_DB` (use `:memory:` for tests).

## Wire into Claude Code

```bash
aelf setup                                        # ~/.claude/settings.json
aelf setup --scope project --project-root .       # project-local
aelf setup --no-statusline                        # hook only, no statusline
aelf unsetup                                      # remove both
```

Idempotent. `aelf setup` wires two things:

1. **`UserPromptSubmit` hook** (`aelf-hook`). Reads each prompt payload, runs retrieval, emits an `<aelfrice-memory>...</aelfrice-memory>` block above the prompt. Every failure mode exits 0 with no output — the hook can't block a prompt.
2. **`statusLine` notifier** (`aelf statusline`). Prints an orange one-line update banner in the Claude Code statusbar **only when an update is available**, empty otherwise. Reads the cached PyPI check; never makes network calls. Banner disappears automatically once you've upgraded.

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

## Uninstall

aelfrice ships an explicit teardown command. You **must** pick exactly one of `--keep-db`, `--purge`, or `--archive PATH` so the brain-graph SQLite DB at `~/.aelfrice/memory.db` doesn't get accidentally lost or accidentally retained.

```bash
aelf uninstall --keep-db                           # safe default for review
aelf uninstall --archive ~/aelf-backup.aenc        # encrypt then delete
aelf uninstall --purge                              # permanently delete (gates fire)
pip uninstall aelfrice                              # finally, remove the wheel
```

Verification:

```bash
aelf --version 2>&1 | grep -q aelfrice || echo "wheel removed"
test -f ~/.aelfrice/memory.db && echo "db still there" || echo "db gone"
```

### `--keep-db`

DB preserved at `~/.aelfrice/memory.db`. The default also runs `unsetup` so the Claude Code hook + statusline are removed from `~/.claude/settings.json`. Pass `--keep-hook` if you want to keep those wired (e.g. you plan to reinstall and skip onboarding).

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

Permanently deletes `~/.aelfrice/memory.db`. Three gates fire before deletion:

1. The `--purge` flag must be passed explicitly. Default behavior is an error pointing at `--help`.
2. Print the target path + size, then require the user to type `PURGE` verbatim (case-sensitive).
3. Final `[y/N]` confirmation.

`--yes` skips prompts (1)–(3). It does **not** auto-pass `--purge`. There is no env-var override for the `--purge` flag; if you want to script unattended teardown you must explicitly add both `--purge --yes`.
