<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Bayesian memory for AI coding agents. Local-only. Auditable.

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
[![Staging Gate](https://github.com/robotrocketscience/aelfrice/actions/workflows/staging-gate.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/staging-gate.yml)
[![Downloads](https://img.shields.io/pypi/dm/aelfrice.svg)](https://pypi.org/project/aelfrice/)

> [!NOTE]
> v1.0 ships the surface — local SQLite store, retrieval, the `apply_feedback` endpoint, the onboarding scanner, an 11-command CLI, an MCP server, Claude Code wiring, and a reproducible benchmark harness. **Retrieval ranking is BM25-only at v1.0** — feedback updates the math but doesn't yet move ranking. The v1.x line wires posterior into ranking and closes [known issues](docs/LIMITATIONS.md#known-issues-at-v10).

You had a doc with the conversation. You re-explained your stack last session. You wrote a runbook the agent didn't read. The notes you keep adding don't actually keep your agent from forgetting — they just give you more to maintain.

aelfrice is a small SQLite-backed memory that the agent can't skip. Lock the rules you don't want forgotten. Onboard a project once. Every prompt thereafter gets the relevant slice injected before the agent answers.

```
No GPU. No network. No telemetry. No cloud.
SQLite at ~/.aelfrice/memory.db. That's the whole runtime.
```

Every retrieval result is traceable to the beliefs and rules that produced it. Every state of the system is reproducible from its write log. We are not aware of another agent-memory system that combines bit-level reproducibility, named-rule traceability, write-log historical reconstruction, and audit comprehensible to a non-technical reviewer as a single system property. See [PHILOSOPHY § Determinism is the property](docs/PHILOSOPHY.md#determinism-is-the-property).

## 60 seconds

```bash
$ pip install aelfrice
$ aelf onboard .
$ aelf lock "Never push directly to main; use scripts/publish.sh"
$ aelf setup        # wires the UserPromptSubmit hook into Claude Code
```

Same operations are available as MCP tools and Claude Code slash commands. Full demo: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Roadmap

| | Status | |
|---|---|---|
| v0.1 – v1.0 | **shipped** | core memory, CLI, MCP, hook wiring, synthetic benchmark, PyPI publish |
| v1.0.1 | **shipped** | launch fix-up — hook→retrieval wiring, onboard noise, `aelf --version` |
| v1.0.2 | **shipped** | per-project install routing, `aelf doctor`, release-docs CI gate |
| v1.0.3 | **shipped** | contradiction tie-breaker + `aelf resolve`, onboard perf regression, CONFIG.md |
| v1.1.0 | planned | project identity (`.git/aelfrice/`), edges→threads, status/health split |
| v1.2.0 | planned | commit-ingest hook, triple-extraction port, harness-integration doc |
| v1.3 | planned | retrieval wave — entity index + BFS multi-hop + LLM classification |
| v2.0 | planned | feature parity with the earlier research line + full benchmark reproducibility |

Per-version detail with deliverables, recovery inventory, and structural-fix rationale: [docs/ROADMAP.md](docs/ROADMAP.md). Per-issue tracking: [docs/LIMITATIONS.md](docs/LIMITATIONS.md#known-issues-at-v10).

## Install

```bash
pip install aelfrice                # core (zero runtime deps)
pip install "aelfrice[mcp]"         # add MCP server
pip install "aelfrice[archive]"     # add encrypted DB archive on uninstall
aelf --version                       # confirm install
aelf setup                           # wire hook + statusline into Claude Code
aelf doctor                          # verify hook commands resolve
aelf health                          # confirm wiring + store init
```

`aelf setup` wires two things into Claude Code's `settings.json`:

1. The **UserPromptSubmit hook** (`aelf-hook`) which injects relevant beliefs into every Claude Code prompt.
2. The **statusline notifier** (`aelf statusline`) which shows an orange `⬆ aelfrice X.Y.Z available, run: aelf upgrade` banner *only* when an update is pending. When you're up to date the banner is empty and your statusline looks unchanged.

`aelf setup` auto-routes the install per-project: when run from a project venv it writes `<project>/.claude/settings.json` pointing at `<project>/.venv/bin/aelf-hook`; when run outside any project it writes `~/.claude/settings.json` pointing at the first `aelf-hook` on `$PATH` (typically a `pipx`-installed global). Explicit `--scope user|project` overrides.

If you already have a custom `statusLine` configured, `aelf setup` composes its snippet onto the end of your command (preserving your bar). If your existing command uses pipes, here-docs, `&&`, backticks, or backslashes it's left untouched and you get a one-line hint about manual composition.

`--no-statusline` opts out of the auto-wire if you want hook-only.

[docs/INSTALL.md](docs/INSTALL.md) covers Codex wiring, generic MCP hosts, and troubleshooting.

## Upgrade

```bash
aelf upgrade           # prints the right pip-upgrade line for your env
aelf upgrade --check   # yes/no, no command line printed
```

`aelf upgrade` detects venv vs pipx vs system and tells you the exact line. It does **not** execute pip itself: replacing the running package mid-process is unreliable on Windows and can leave a broken interpreter. You run the line.

When an update is available the output also includes the published wheel SHA-256 plus the PyPI release URL so you can hash-pin the install if you want.

The orange statusline banner appears automatically when an update is pending and disappears once you're up to date — no manual refresh needed.

Opt out of the update notifier at any time with `export AELF_NO_UPDATE_CHECK=1`.

## Uninstall

aelfrice has an explicit teardown command. You **must** pick exactly one disposition for the brain-graph DB:

```bash
aelf uninstall --keep-db       # leave ~/.aelfrice/memory.db alone (safe)
aelf uninstall --archive ~/aelf-backup.aenc   # encrypt then delete
aelf uninstall --purge         # permanently delete (redundant gates fire)
pip uninstall aelfrice         # finally, remove the wheel
```

Verify removal:

```bash
aelf --version 2>&1 | grep -q "aelfrice" || echo "removed"
```

Details:

- **`--keep-db`** — DB preserved. Default also runs `unsetup` (removes hook + statusline). Pass `--keep-hook` to keep those too.
- **`--archive PATH`** — DB encrypted (AES-128-CBC + HMAC via Fernet, scrypt-derived key) to PATH, then original deleted. Password is read interactively (twice, must match) or via `--password-stdin`. Recover later with `python -c "from aelfrice.lifecycle import decrypt_archive; open('out.db','wb').write(decrypt_archive('PATH','pw'))"`. Requires `pip install 'aelfrice[archive]'`.
- **`--purge`** — Permanently deletes the DB. Three gates fire before deletion: (1) the flag must be passed explicitly, (2) you must type `PURGE` verbatim, (3) a final `[y/N]` confirmation. `--yes` skips the prompts but does **not** auto-pass `--purge`.

[docs/INSTALL.md](docs/INSTALL.md#uninstall) has the full uninstall reference including archive-recovery details.

## Docs

[QUICKSTART](docs/QUICKSTART.md) · [COMMANDS](docs/COMMANDS.md) · [MCP](docs/MCP.md) · [SLASH_COMMANDS](docs/SLASH_COMMANDS.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [CONFIG](docs/CONFIG.md) · [PHILOSOPHY](docs/PHILOSOPHY.md) · [PRIVACY](docs/PRIVACY.md) · [LIMITATIONS](docs/LIMITATIONS.md) · [CHANGELOG](CHANGELOG.md)

[CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md) · [CITATION](CITATION.cff) · [MIT](LICENSE)
