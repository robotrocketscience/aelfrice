<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Bayesian memory for AI coding agents. Local-only. Auditable.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)

> [!NOTE]
> v1.0 ships the surface — local SQLite store, retrieval, the `apply_feedback` endpoint, the onboarding scanner, an 11-command CLI, an MCP server, Claude Code wiring, and a reproducible benchmark harness. **Retrieval ranking is BM25-only at v1.0** — feedback updates the math but doesn't yet move ranking. The v1.x line wires posterior into ranking and closes [known issues](docs/LIMITATIONS.md#known-issues-at-v10).

You had a doc with the conversation. You re-explained your stack last session. You wrote a runbook the agent didn't read. The notes you keep adding don't actually keep your agent from forgetting — they just give you more to maintain.

aelfrice is a small SQLite-backed memory that the agent can't skip. Lock the rules you don't want forgotten. Onboard a project once. Every prompt thereafter gets the relevant slice injected before the agent answers.

```
No GPU. No network. No telemetry. No cloud.
SQLite at ~/.aelfrice/memory.db. That's the whole runtime.
```

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
| v0.1 – v1.0 | **shipped** | core memory, CLI, MCP, hook wiring, benchmark harness, PyPI publish |
| v1.x | next | feedback in retrieval ranking, [known issues](docs/LIMITATIONS.md#known-issues-at-v10) |

## Install

```bash
pip install aelfrice              # core
pip install "aelfrice[mcp]"       # add MCP server
aelf setup                         # wire UserPromptSubmit hook into Claude Code
```

[docs/INSTALL.md](docs/INSTALL.md) covers Codex wiring, generic MCP hosts, and troubleshooting.

## Docs

[QUICKSTART](docs/QUICKSTART.md) · [COMMANDS](docs/COMMANDS.md) · [MCP](docs/MCP.md) · [SLASH_COMMANDS](docs/SLASH_COMMANDS.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [PHILOSOPHY](docs/PHILOSOPHY.md) · [PRIVACY](docs/PRIVACY.md) · [LIMITATIONS](docs/LIMITATIONS.md) · [CHANGELOG](CHANGELOG.md)

[CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md) · [CITATION](CITATION.cff) · [MIT](LICENSE)
