# aelfrice

**Bayesian memory designed for feedback-driven learning.** Today: knowledge store with Bayesian priors. Coming in `v0.4.0`: the feedback loop that teaches the store.

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
| **v0.3.0** | next | `retrieval.py` — locked beliefs auto-load + FTS5 keyword search with BM25. Token-budgeted output. No HRR, no BFS multi-hop, no entity-index in v1.0. |
| **v0.4.0** | planned | `feedback.py` — the central new endpoint (`apply_feedback`, `feedback_history` table, demotion-pressure-acted-on) + `correction.py` |
| **v0.5.0** | planned | `scanner.py` — onboarding with filesystem walk, git log, simple AST extractors |
| **v0.6.0** | planned | `cli.py` (8 commands) + `mcp_server.py` (8 MCP tools) + `slash_commands/` |
| **v0.7.0** | planned | `setup.py` — Claude Code wiring (`UserPromptSubmit` hook for retrieval injection) |
| **v0.8.0** | planned | docs, `CHANGELOG.md`, `LICENSE` (MIT), final `pyproject.toml` |
| **v0.9.0-rc** | planned | minimal benchmark harness producing a publishable score |
| **v1.0.0** | planned | tag, push, PyPI publish |

After `v1.0.0`, the `v1.x` series recovers `v2.0` features incrementally with evidence justifying each addition.

## What works today (v0.2.0)

- SQLite-backed Belief and Edge storage with WAL journaling and FTS5 search
- Beta-Bernoulli confidence math; per-type half-life decay
- Lock floor (a `user`-locked belief does not decay)
- Broker-confidence-attenuated valence propagation through the belief graph
- 16 tests, including 3 pre-registered tests for Bayesian inertia, decay necessity, and lock-floor sharpness

What does NOT work yet: retrieval, the feedback endpoint, the CLI, the MCP server, onboarding, install via `pip`. Wait for `v1.0.0` if you want to use this.

## design notes

- **Clean by construction, not clean by audit.** Filtering is a tripwire, not a gate.
- **`apply_feedback` is the central endpoint.** Not an admin command, not a hook, not an undocumented path. Read+write `demotion_pressure` is required.
- **No HRR, no LLM, no embeddings in v1.0.** Stdlib + SQLite only. Heavier machinery lands in `v1.x` with evidence justifying inclusion.

## Contributing

Closed to outside contribution until `v1.0.0` ships. Issues welcome at that point.

## License

Will land at `v0.8.0` (MIT planned). Until then, all rights reserved by the author.
