# aelfrice

**Bayesian memory designed for feedback-driven learning.** Today: knowledge store, retrieval, feedback loop, scanner, CLI, MCP server, and Claude Code wiring all landed on `main`. Next on the path to `v1.0.0`: docs/license and benchmark harness.

> ⚠️ **Status: under active rebuild — do not depend on this for production.** The first installable release is `v1.0.0`; until then, modules land milestone-by-milestone. The previous codebase (`v2.0`) is preserved at [`aelfrice-v0`](https://github.com/robotrocketscience/aelfrice-v0) (archived, read-only).

## Why aelfrice

LLM agents repeatedly forget what they learned. Vector-RAG remembers documents but doesn't remember **outcomes**: which retrievals helped, which hurt, which were corrected. aelfrice is being built as a small, local, SQLite-backed memory that will:

- Keep each belief's confidence as a **Beta-Bernoulli posterior** (`α/(α+β)`), so confidence updates are mathematically grounded, not vibes.
- Treat every retrieval as a feedback opportunity: `apply_feedback(belief_id, used|harmful)` will update posteriors, propagate valence through the belief graph, and record the event for audit.
- Let you **lock** beliefs you trust as ground truth. Lock floor short-circuits decay. Contradicting feedback against a locked belief accumulates `demotion_pressure`; cross a threshold and the lock auto-demotes.
- Stay local. Pure stdlib SQLite. No vector DB, no embeddings, no cloud sync. Local-only is a feature.

The central claim — that a memory which actually applies feedback outperforms one that doesn't — will be testable end-to-end at `v1.0.0`.

## v1.0.0 milestone roadmap

The rebuild is structured as a sequence of small, atomic milestones.

| Milestone | Status | What lands |
|---|---|---|
| **v0.1.0** | shipped | `models.py` + `config.py` + `store.py` (SQLite WAL + FTS5 + CRUD + broker-attenuated `propagate_valence` + `demotion_pressure` read/write) |
| **v0.2.0** | shipped | `scoring.py` (Beta-Bernoulli posterior mean, type-specific decay with lock-floor short-circuit, basic relevance combiner) + tests for Bayesian inertia, decay-required, and lock-floor sharpness |
| **v0.3.0** | shipped | `retrieval.py` — locked beliefs auto-load + FTS5 keyword search with BM25. Token-budgeted output. No HRR, no BFS multi-hop, no entity-index in v1.0. |
| **v0.4.0** | shipped | `feedback.py` — the central new endpoint (`apply_feedback`, `feedback_history` table, demotion-pressure-acted-on) + `correction.py` |
| **v0.5.0** | shipped | `scanner.py` — onboarding with filesystem walk, git log, simple AST extractors |
| **v0.6.0** | shipped | `cli.py` (8 commands) + `mcp_server.py` (8 MCP tools) + `slash_commands/` |
| **v0.7.0** | shipped | `setup.py` — Claude Code wiring (`UserPromptSubmit` hook for retrieval injection) + `aelfrice.hook` entry-point + `aelf setup` / `aelf unsetup` CLI + `aelf-hook` script |
| **v0.8.0** | next | docs, `CHANGELOG.md`, `LICENSE` (MIT), final `pyproject.toml` |
| **v0.9.0-rc** | planned | minimal benchmark harness producing a publishable score |
| **v1.0.0** | planned | tag, push, PyPI publish |

After `v1.0.0`, the `v1.x` series recovers `v2.0` features incrementally with evidence justifying each addition.

## What works today (v0.7.0)

- SQLite-backed Belief and Edge storage with WAL journaling and FTS5 search
- Beta-Bernoulli confidence math; per-type half-life decay
- Lock floor (a `user`-locked belief does not decay)
- Broker-confidence-attenuated valence propagation through the belief graph
- Retrieval (locked-belief auto-load + BM25 FTS5, token-budgeted) and the `apply_feedback` endpoint (Beta-Bernoulli updates + demotion-pressure auto-demote)
- Onboarding scanner (filesystem walk + git log + AST extractors), `cli.py` (10 commands), `mcp_server.py` (8 tools, `[mcp]` extra), and 10 slash commands under `slash_commands/`
- Claude Code wiring: `aelf setup` / `aelf unsetup` install or remove a `UserPromptSubmit` hook in Claude Code's `settings.json`; `aelf-hook` is the script entry-point that reads the hook payload, runs retrieval, and emits an `<aelfrice-memory>` block as injected context
- ~500 test functions across the unit suite, including 3 pre-registered property tests (Bayesian inertia, decay necessity, lock-floor sharpness) and end-to-end regression scenarios for retrieval, feedback, onboarding, and the Claude Code setup→hook→unsetup round-trip

What does NOT work yet: install via `pip`, final docs/CHANGELOG/LICENSE (v0.8.0), and the v0.9.0-rc benchmark harness. Wait for `v1.0.0` if you want a stable release; until then, install editable from source (`uv pip install -e .`) and run `aelf setup` to wire the hook into Claude Code.

## docs

- [docs/INSTALL.md](docs/INSTALL.md) — install, configure, quickstart, Claude Code wiring.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, data model, Bayesian update path, retrieval flow, Claude Code integration diagram.

## design notes

- **Clean by construction, not clean by audit.** Filtering is a tripwire, not a gate.
- **`apply_feedback` is the central endpoint.** Not an admin command, not a hook, not an undocumented path. Read+write `demotion_pressure` is required.
- **No HRR, no LLM, no embeddings in v1.0.** Stdlib + SQLite only. Heavier machinery lands in `v1.x` with evidence justifying inclusion.

## Contributing

Closed to outside contribution until `v1.0.0` ships. Issues welcome at that point.

## License

[MIT](LICENSE). Copyright (c) 2026 robotrocketscience.
