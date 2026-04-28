# Roadmap

How aelfrice gets from v1.0 to v2.0.

Per-issue tracking: [LIMITATIONS.md](LIMITATIONS.md). Release log: [CHANGELOG.md](../CHANGELOG.md).

## Origin

aelfrice is a rebuild of an earlier research line on Bayesian + graph-backed memory for AI coding agents. The research codebase explored FTS5 retrieval, vocabulary bridging, BFS multi-hop traversal, entity-indexed retrieval, type-aware compression, correction detection, and a multi-tool MCP surface, with results against MAB, LoCoMo, LongMemEval, StructMemEval, and AMA-Bench.

aelfrice **v1.0 is the foundation for that surface, not the surface itself.** v1.0 ships a stable core (SQLite store, Beta-Bernoulli scoring, BM25 retrieval, CLI, MCP, Claude Code wiring, a synthetic benchmark harness). The v1.x line recovers the remaining features incrementally with evidence required for each. v2.0 is the planned feature-parity release.

This is a rebuild, not a port. Structural issues that survived the research line are being fixed at the foundation layer. Every behavioural claim is backed by a test or a benchmark, with a transparent issue trail for items that are not.

## Versions at a glance

| Version | Status | Theme |
|---|---|---|
| v1.0.x | shipped | core memory, CLI, MCP, hook wiring, install routing, contradiction tie-breaker |
| **v1.1.0** | shipped | per-project DBs, `aelf migrate`, `edges`→`threads` rename, `aelf health` rewrite |
| **v1.2.0** | shipped | auto-capture pipeline (transcript-ingest, commit-ingest, SessionStart), `agent_inferred → user_validated` promotion, triple extractor, `--batch` JSONL ingest, CLI consolidation, `INEDIBLE` per-file opt-out |
| **v1.2.x** | planned | search-tool `PreToolUse` hook — memory-first context on Grep/Glob |
| **v1.3.0** | planned | retrieval wave — entity index + BFS multi-hop + LLM classification + posterior-weighted ranking |
| **v1.4.0** | planned | context rebuilder — PreCompact retrieval-curated continuation |
| **v2.0.0** | planned | feature parity with the research line + benchmark reproducibility |

## What shipped

### v1.0.x — surface

Stable core: SQLite + FTS5 store, Beta-Bernoulli scoring, L0+L1 retrieval at a 2,000-token budget, `apply_feedback` with audit log, onboarding scanner (filesystem + git log + Python AST), CLI, MCP server, Claude Code hook wiring, synthetic benchmark harness, contradiction tie-breaker (`aelf resolve`), per-project install routing (`aelf doctor`), release-docs CI gate.

### v1.1.0 — project identity

- Per-project DB resolution: `<git-common-dir>/aelfrice/memory.db` inside any git work-tree, falls back to `~/.aelfrice/memory.db` outside.
- `aelf migrate` ports beliefs from the legacy global store. Read-only on the source.
- Worktree concurrency tested under WAL + `busy_timeout=5000`.
- `aelf health` rewritten as the structural auditor (orphan threads, FTS5 sync, locked contradictions). v1.0 regime classifier preserved as `aelf regime`. `aelf status` aliases `aelf health`.
- `edges` → `threads` user-facing rename. Internal schema unchanged. Deprecation window covered both keys.
- Onboard git-recency weighting: `belief.created_at` is the source file's most recent commit, so decay penalises stale branches.
- `agent_inferred → user_validated` promotion path designed (implemented in v1.2).

### v1.2.0 — auto-capture and triple extraction

- **Commit-ingest hook.** `PostToolUse:Bash` ingests every successful `git commit` through the triple extractor under a deterministic git-derived session id. Densely populates `Edge.anchor_text`, `Belief.session_id`, and `DERIVED_FROM` edges. Wired via `aelf setup --commit-ingest`.
- **Transcript-ingest hook.** Four-event hook captures every conversation turn to `<root>/.git/aelfrice/transcripts/turns.jsonl`; `PreCompact` rotates and ingests. Closes the harness-conflict gap that previously starved the MCP of new beliefs from normal sessions.
- **Triple extractor.** Pure regex over six relation families (`SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`, `RELATES_TO`, `DERIVED_FROM`). Reusable by every prose-ingesting caller.
- **`agent_inferred → user_validated` promotion.** New `Belief.origin` column with seven tier values. `aelf validate <id>` graduates onboard-derived beliefs without requiring a lock.
- **SessionStart hook.** Injects locked beliefs as `<aelfrice-baseline>` once per session.
- **Ingest enrichment schema.** `DERIVED_FROM` edges, `anchor_text`, `session_id`, real `sessions` table. Forward-compatible with v1.0 stores.
- **`aelf ingest-transcript --batch DIR [--since DATE]`.** Backfill historical Claude Code session JSONLs into the local belief graph. Auto-detects transcript-logger and Claude Code session formats.
- **CLI consolidation + `INEDIBLE` per-file opt-out.** v1.3 prep: surface tightened, per-file privacy marker added.
- **Harness integration guide.** Three operational modes for coexisting with Claude Code's auto-memory directive.

## Planned

### v1.2.x — search-tool hook (planned patch)

Pulled forward from v1.3.0 to validate the `PreToolUse` retrieval surface ahead of the bigger retrieval wave. Ships against the v1.0 retrieval pipeline + v1.1.0 per-project DB resolution; no dependency on entity-index, BFS, or LLM classifier work.

- **Search-tool `PreToolUse` hook** ([search_tool_hook.md](search_tool_hook.md)). Fires before `Grep` and `Glob` tool calls, lifts the agent's search query out of `tool_input.pattern`, runs the same query against the per-project belief store, and emits results as `additionalContext` so the agent sees them *before* the tool runs. First retrieval-shaped hook on the agent's *own* tool intent (the v1.0.1 `UserPromptSubmit` hook covers user-initiated retrieval; this covers agent-initiated). Opt-in via `aelf setup --search-tool` at v1.2.x; default-on candidate at v1.3.0 once production telemetry confirms the latency budget (median ≤ 50 ms, p95 ≤ 200 ms on a 10k-belief store).

### v1.3.0 — retrieval wave

This is the release where retrieval moves beyond BM25-only.

- **Entity-index retrieval.** L2.5 entity-index ports forward, including the regex extraction patterns. Spec: [entity_index.md](entity_index.md).
- **BFS multi-hop graph traversal.** Edge-type-weighted graph walks layered on top of FTS5 hits. Bounded depth, bounded budget. Spec: [bfs_multihop.md](bfs_multihop.md).
- **LLM-classification onboard path.** Haiku-backed classifier as an opt-in alternative to the regex classifier. Default-off; opt in via `aelf onboard --llm-classify` or `[onboard.llm].enabled = true` in `.aelfrice.toml`. Boundary policy and prompt template specified in [llm_classifier.md](llm_classifier.md). PRIVACY note: this introduces the first outbound call in the install path that transmits user content; see [PRIVACY § Optional outbound calls](PRIVACY.md#optional-outbound-calls).
- **Posterior-weighted ranking (partial).** Retrieval scoring begins to incorporate `α / (α+β)` on top of BM25. Full feedback-into-ranking eval lands at v2.0.0.

### v1.4.0 — context rebuilder

Long-running sessions cheaper without a visible seam.

- **PreCompact-driven rebuild.** When the harness signals an approaching context limit, an aelfrice hook queries the brain graph for the highest-value beliefs against the session tail and emits them as `additionalContext`. Locked beliefs first, then session-scoped, then BM25 / posterior-weighted hits, packed to a configurable token budget.
- **Augment mode.** The hook augments the harness's compaction; both summaries land in the new context. Replace mode is parked for v2.x.
- **Trigger modes.** Manual (`/aelf-rebuild`) ships first; threshold mode ships with calibration data.
- **Continuation-fidelity eval.** `benchmarks/context-rebuilder/` replays fixture transcripts, forces a midpoint clear, runs the rebuilder, and measures continuation fidelity vs. the full-replay baseline. Fixture corpus policy (synthetic public for CI / headline number; captured corpus held lab-side for offline calibration only) is decided in [eval_fixture_policy.md](eval_fixture_policy.md).

Hard prerequisites: v1.2 transcript-ingest, v1.2 `session_id` schema. Alpha shipped in v1.2.0a0.

### v2.0.0 — feature parity and reproducibility

After v2.0.0, `benchmarks/` reproduces every published headline number on a fresh clone with `uv sync && aelf bench all`, within documented tolerance bands.

- HRR vocabulary bridge — closes the vocabulary-gap-recovery claim against a corpus checked into the repo.
- Type-aware compression — tokens-per-belief reductions on retrieved output.
- Intentional clustering — co-locating related beliefs for higher coherence on multi-fact queries.
- Correction-detection eval — five-codebase labeled fixture, scored by both the zero-LLM detector and the LLM-judge path.
- Posterior drives ranking, end to end. The 10-round MRR uplift eval and ECE calibration scorer ship with this release.
- Expanded surface: `wonder`, `reason`, `core`, `unlock`, `delete`, `confirm`, plus graph-metrics and document-linking.
- Reproducibility harness: `benchmarks/results/v2.0.0.json` is canonical; CI runs the academic suite nightly.

## Recovery inventory

What the research line had, when each piece returns:

| Capability | Public version |
|---|---|
| Triple extraction | v1.2.0 |
| Commit-ingest hook | v1.2.0 |
| Transcript-ingest + `ingest_jsonl()` | v1.2.0 |
| `agent_inferred → user_validated` promotion | v1.2.0 |
| Context rebuilder + continuation eval | v1.4.0 (alpha in v1.2) |
| Graph metrics + status/health split | v1.1.0 |
| Entity-index retrieval | v1.3.0 |
| BFS multi-hop graph traversal | v1.3.0 |
| LLM-Haiku onboard classifier | v1.3.0 |
| Posterior-weighted ranking | v1.3.0 (partial) / v2.0.0 (full) |
| HRR vocabulary bridge | v2.0.0 |
| Type-aware compression | v2.0.0 |
| Doc / semantic linker | v2.0.0 |
| `wonder` / `reason` / `core` / `unlock` / `delete` / `confirm` | v2.0.0 |

## Compatibility

aelfrice follows semver:

- **Patch (v1.0.x):** API and schema preserved.
- **Minor (v1.x.0):** API preserved; migrations are forward-compatible (new columns/tables only).
- **v2.0.0:** may break the v1 API only where a benchmark or eval justifies the break. Migrations are documented and tested before tag.

## Non-goals

- **Vector database / embedding retrieval.** aelfrice stays SQLite + FTS5 at every milestone. The HRR bridge in v2.0 is a structural retrieval layer, not an embedding layer.
- **Cloud sync.** The runtime stays local. No release introduces network I/O in the retrieval or write path.
- **Telemetry.** No release adds outbound network calls in the default install.
- **Brain-graph sharing or sync.** No release ships a mechanism for distributing memory contents between users, machines, or projects. See [LIMITATIONS § Sharing, sync, or federation](LIMITATIONS.md).

## Validation

Every release with a behavioural claim ships with a benchmark or test that demonstrates it. v1.0's central claim is that `apply_feedback` updates posteriors mathematically — proven by tests and the synthetic harness. v1.0 makes *no* claim that BM25 ranking moves under feedback, because it doesn't yet. v1.3 is the release where that claim becomes testable; v2.0 is where the published MRR uplift is reproduced or retracted.

That is what "early access" means: the surface is stable, the central feedback-into-ranking claim is a future deliverable, and the benchmark harness ships now so the eventual delivery is auditable.
