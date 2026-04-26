# aelfrice

**Bayesian memory that learns from feedback.** Lock + correct + observe over many sessions and the agent stops re-discovering things.

> ⚠️ **Status: under active rebuild — do not depend on this for production.** The first installable release is `v1.0.0`; until then, modules land milestone-by-milestone. The previous codebase (`v2.0`) is preserved at [`aelfrice-v0`](https://github.com/robotrocketscience/aelfrice-v0) (archived, read-only).

## Why aelfrice

LLM agents repeatedly forget what they learned. Vector-RAG remembers documents but doesn't remember **outcomes**: which retrievals helped, which hurt, which were corrected. aelfrice is a small, local, SQLite-backed memory that:

- Keeps each belief's confidence as a **Beta-Bernoulli posterior** (`α/(α+β)`), so confidence updates are mathematically grounded, not vibes.
- Treats every retrieval as a feedback opportunity. `apply_feedback(belief_id, used|harmful)` updates posteriors, propagates valence through the belief graph, and records the event for audit.
- Lets you **lock** beliefs you trust as ground truth. Lock floor short-circuits decay. Contradicting feedback against a locked belief accumulates `demotion_pressure`; cross a threshold and the lock auto-demotes.
- Stays local. Pure stdlib SQLite. No vector DB, no embeddings, no cloud sync. Local-only is a feature.

The central claim — that a memory which actually applies feedback outperforms one that doesn't — is testable from day one. The MVP includes UAT and benchmarks that exercise the feedback loop end-to-end.

## v1.0.0 milestone roadmap

The rebuild is structured as a sequence of small, atomic milestones. Each is a `gate:` commit you can read top-to-bottom to decide go/nogo.

| Milestone | Status | What lands |
|---|---|---|
| **v0.1.0** | ✅ shipped (`a1a656b`) | `models.py` + `config.py` + `store.py` (SQLite WAL + FTS5 + CRUD + Setr broker-attenuated `propagate_valence` + `demotion_pressure` read/write) |
| **v0.2.0** | ✅ shipped (`b32491d`) | `scoring.py` (Beta-Bernoulli posterior mean, type-specific decay with lock-floor short-circuit, basic relevance combiner) + R&D-derived tests for Bayesian inertia (E1b), decay-required (E3), and lock-floor sharpness (E2) |
| **v0.3.0** | next | `retrieval.py` — L0 (locked beliefs auto-load) + L1 (FTS5 keyword search with BM25). Token-budgeted output. **No HRR, no BFS multi-hop, no entity-index in v1.0.** |
| **v0.4.0** | planned | `feedback.py` (the central new endpoint — `apply_feedback`, `feedback_history` table, demotion-pressure-acted-on) + `correction.py` (the 92% no-LLM correction detector ported from v2.0) |
| **v0.5.0** | planned | `scanner.py` — onboarding with 3 extractors (filesystem walk, git log basic, simple AST). The other 5 extractors land in v1.2 |
| **v0.6.0** | planned | `cli.py` (8 commands: `onboard`, `search`, `lock`, `locked`, `demote`, `feedback`, `stats`, `health`) + `mcp_server.py` (8 MCP tools matching the CLI) + `slash_commands/` (8 Claude Code slash commands) |
| **v0.7.0** | planned | `setup.py` — Claude Code wiring (`UserPromptSubmit` hook for L0+L1 retrieval injection). Other hooks deferred to v1.1+ |
| **v0.8.0** | planned | docs (`INSTALL.md`, `COMMANDS.md`), `CHANGELOG.md`, `LICENSE` (MIT), final `pyproject.toml` (deps: `anthropic` + `nltk` only) |
| **v0.9.0-rc** | planned | minimal benchmark harness (one of LoCoMo / StructMemEval / custom) producing a publishable score |
| **v1.0.0** | planned | tag, push, PyPI publish via Trusted Publishing |

After v1.0.0, the `v1.x` series recovers v2.0 features incrementally with R&D justification per `MVP_SCOPE.md` deferral schedule (HRR retrieval → v1.1; entity-index → v1.2; lock tiers `promoted`/`freeze` → v1.3; LLM classification → v1.4; anneal maintenance → v1.5; etc.).

## What works today (v0.2.0)

- SQLite-backed Belief and Edge storage with WAL journaling and FTS5 search
- Beta-Bernoulli confidence math; per-type half-life decay
- Lock floor (a `user`-locked belief does not decay)
- Setr broker-confidence-attenuated valence propagation through the belief graph
- 16 tests, including 3 R&D-pre-registered tests validating Bayesian inertia, decay necessity, and lock-floor sharpness

What does NOT work yet: retrieval, the feedback endpoint, the CLI, the MCP server, onboarding, install via `pip`. Wait for v1.0.0 if you want to use this.

## Design principles

These constrain every commit. See [`MIGRATION_PLAN.md`](https://github.com/robotrocketscience/aelfrice-dev — private) for the full list. Highlights:

- **Clean by construction, not clean by audit.** Filtering is a tripwire, not a gate.
- **No bespoke audit pipelines.** Stock tooling (`gitleaks`, GitHub Actions branch protection) replaces the v2.0 publish-quarantine machinery.
- **Pattern files and threat models live in the private workspace.** Public repo never sees the answer key.
- **`apply_feedback` is the central endpoint.** Not an admin command, not a hook, not an undocumented path. Read+write `demotion_pressure` is required.
- **No HRR, no LLM, no embeddings in v1.0.** Stdlib + SQLite only. Heavier machinery lands in v1.x with evidence justifying inclusion.

## Contributing

Closed to outside contribution until v1.0.0 ships. Issues welcome at that point.

## License

Will land at v0.8.0 (MIT planned). Until then, all rights reserved by the author.

## Provenance

Rebuild driven by R&D rounds 1-6 (24 experiments) and the [MVP scope contract](https://github.com/robotrocketscience/aelfrice-dev — private). The legacy v2.0 codebase remains accessible at [`aelfrice-v0`](https://github.com/robotrocketscience/aelfrice-v0) (private, archived).
