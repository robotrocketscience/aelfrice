# INSTALL

Installation, configuration, and verification for aelfrice.

> aelfrice is pre-`1.0.0`. The first installable PyPI release is `v1.0.0`.
> Until then install editable from a clone.

---

## Prerequisites

- **Python 3.12 or 3.13.** Declared in `pyproject.toml` as
  `requires-python = ">=3.12"`. Python 3.11 will not resolve.
- **[uv](https://docs.astral.sh/uv/)** is the supported workflow tool.
  Plain `pip` works too, but every command in this doc is written for
  `uv` because that's what CI runs.
- **No external services.** aelfrice is stdlib-only at the dependency
  level (`dependencies = []` in `pyproject.toml`). The optional `[mcp]`
  extra adds [`fastmcp`](https://pypi.org/project/fastmcp/) for the
  MCP server entry point.

---

## Editable install from source

```bash
git clone git@github.com:robotrocketscience/aelfrice.git
cd aelfrice
uv sync                # creates .venv and installs aelfrice in editable mode
```

`uv sync` reads `pyproject.toml` + `uv.lock` and produces a
deterministic environment. To include the optional MCP server
dependency:

```bash
uv sync --extra mcp
```

Two console scripts are exposed by `[project.scripts]`:

| Script | Entry point | Purpose |
|---|---|---|
| `aelf` | `aelfrice.cli:main` | the user-facing CLI (10 subcommands) |
| `aelf-hook` | `aelfrice.hook:main` | the Claude Code `UserPromptSubmit` hook entry-point |

Verify the install:

```bash
uv run aelf stats
uv run aelf health
```

Both should exit `0`. On a fresh DB, `aelf health` reports
`brain mode: insufficient_data` — that's correct.

---

## Database location

aelfrice persists to a single SQLite file with WAL journaling and an
FTS5 mirror of belief content. Resolution order:

1. `$AELFRICE_DB` if set (absolute or relative path).
2. `~/.aelfrice/memory.db` otherwise. Parent directory is created on
   first write.

For tests, set `AELFRICE_DB=:memory:` to use an in-memory store, or
point at a `tmp_path` fixture.

---

## Quickstart: onboard, search, lock, feedback

```bash
# Ingest an existing project as a corpus of beliefs
uv run aelf onboard ~/code/some-project

# Keyword retrieval (L0 locked auto-load + L1 FTS5 BM25, token-budgeted)
uv run aelf search "kitchen layout"

# Lock a belief as user-asserted ground truth (will not decay)
uv run aelf lock "the build target is wasm32-unknown-unknown"

# List all locked beliefs (and any with nonzero demotion_pressure)
uv run aelf locked --pressured

# Apply Bayesian feedback against a retrieval hit
uv run aelf feedback <belief-id> used     # bumps alpha
uv run aelf feedback <belief-id> harmful  # bumps beta + (if locked) demotion_pressure

# Inspect the brain
uv run aelf stats
uv run aelf health
```

Run `uv run aelf --help` for the full subcommand list and per-command
flags.

---

## Wiring aelfrice into Claude Code

aelfrice ships `aelf setup` to install a `UserPromptSubmit` hook in
Claude Code's `settings.json` so every prompt is augmented with the
most relevant locked beliefs and FTS5 hits.

```bash
# User scope (default): ~/.claude/settings.json
uv run aelf setup

# Project scope: <project-root>/.claude/settings.json
uv run aelf setup --scope project --project-root .

# Custom command override (e.g., wrap aelf-hook in a status-message bash script)
uv run aelf setup --command /usr/local/bin/my-hook.sh --status-message "thinking..."
```

The default `--command` is `aelf-hook` (the script entry-point added
by `pyproject.toml`). The install is idempotent: a second
`aelf setup` invocation reports `hook already installed` and writes
nothing.

To remove:

```bash
uv run aelf unsetup
```

`aelf unsetup` matches by command string, so passing the same
`--scope` / `--settings-path` / `--command` flags you used at install
time finds the right entry. A no-op invocation reports
`no matching hook`.

The hook itself reads the `UserPromptSubmit` JSON payload from stdin,
runs aelfrice retrieval, and writes an
`<aelfrice-memory>...</aelfrice-memory>` block to stdout that Claude
Code injects above the user's message. Every failure mode (empty
stdin, malformed JSON, retrieval errors) exits `0` with no output, so
the hook can never block a prompt from reaching the model.

---

## Development install

To run tests, type-check, and contribute:

```bash
uv sync --extra dev          # pytest, pytest-timeout, pyright
uv run pytest                # ~500 tests, ~6s on Apple Silicon
uv run pyright               # strict mode, src + tests
uv run pytest -m regression  # cumulative integration scenarios only
```

A 5-second wall-clock timeout is the default for all tests
(`pyproject.toml` `[tool.pytest.ini_options]`). Subprocess-driven
tests (e.g., the v0.7.0 setup→hook→unsetup regression) override
per-test via `@pytest.mark.timeout(N)`.

---

## Uninstall

```bash
uv run aelf unsetup    # remove the Claude Code hook
rm -rf ~/.aelfrice     # delete the local memory DB (irreversible)
uv pip uninstall aelfrice
```
