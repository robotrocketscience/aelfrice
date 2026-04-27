# INSTALL

## Prerequisites

- Python 3.12 or 3.13
- Either `pip` (works) or [`uv`](https://docs.astral.sh/uv/) (recommended, used in CI)

## Install from PyPI

```bash
pip install aelfrice              # core
pip install "aelfrice[mcp]"       # add the MCP server (fastmcp)
```

Two console scripts: `aelf` (the CLI) and `aelf-hook` (the Claude Code hook entry-point).

Verify: `aelf health` should print `brain mode: insufficient_data` on a fresh DB.

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
aelf unsetup                                      # remove
```

Idempotent. The hook reads each `UserPromptSubmit` payload, runs retrieval, and emits an `<aelfrice-memory>...</aelfrice-memory>` block above the prompt. Every failure mode exits 0 with no output — the hook can't block a prompt.

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

```bash
aelf unsetup
rm -rf ~/.aelfrice
pip uninstall aelfrice
```
