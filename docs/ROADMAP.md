# Roadmap

How aelfrice gets from v1.0.0 to v2.0.0, in user-facing terms.

For per-issue tracking see [LIMITATIONS.md](LIMITATIONS.md). For benchmark adapter activation status see [`benchmarks/README.md`](../benchmarks/README.md). The release log lives in [CHANGELOG.md](../CHANGELOG.md).

## Origin

aelfrice is a rewrite of [`agentmemory`](http://www.robotrocketscience.com/projects/agentmemory), an earlier research line on Bayesian + graph-backed memory for AI coding agents. The research codebase explored FTS5 keyword retrieval, HRR (Holographic Reduced Representations) vocabulary bridging, BFS multi-hop graph traversal, entity-indexed retrieval, type-aware compression, correction detection, and a multi-tool MCP surface, with published results against MAB, LoCoMo, LongMemEval, StructMemEval, and AMA-Bench.

Public aelfrice **v1.0.0 is the foundation for that surface, not the surface itself.** v1.0 ships a stable core (SQLite store, Beta-Bernoulli scoring, BM25 retrieval, 11-command CLI, 8 MCP tools, Claude Code wiring, a synthetic benchmark harness). The v1.x line recovers the remaining features incrementally with evidence required for each addition. v2.0.0 is the planned feature-parity release with the published claim set.

This is a rebuild, not a port. Structural issues that survived agentmemory are being fixed at the foundation layer where each was originally introduced rather than patched forward. The intent is to leave aelfrice in a state where every behavioural claim is backed by a test or a benchmark, with a transparent issue trail for items that are not.

## Versions at a glance

| Version | Status | Theme |
|---|---|---|
| v0.1 – v1.0 | shipped | core memory, CLI, MCP, hook wiring, synthetic benchmark, PyPI publish |
| **v1.0.1** | shipped | launch fix-up — hook→retrieval wiring, onboard noise, `aelf --version` |
| **v1.0.2** | shipped | per-project install routing, `aelf doctor`, release-docs CI gate |
| **v1.1.0** | shipped | project identity, edges→threads, status/health split, `aelf migrate`, git-recency onboard |
| **v1.2.0** | shipped | auto-capture pipeline, `user_validated` promotion, triple extractor, ingest-enrichment schema, `--batch` JSONL ingest, CLI consolidation, `INEDIBLE` per-file opt-out |
| **v1.3.0** | planned | retrieval wave — entity index + BFS multi-hop + LLM classification |
| **v1.4.0** | planned | context rebuilder — automatic PreCompact-driven retrieval-curated context |
| **v2.0.0** | planned | feature parity with the earlier research line + full benchmark reproducibility |

## v1.0.0 — shipped

See [CHANGELOG.md § v1.0.0](../CHANGELOG.md) for the full surface. In one paragraph: SQLite-backed Belief / Edge store with FTS5 BM25 search; Beta-Bernoulli confidence with type-specific decay and lock-floor short-circuit; L0 (locked) + L1 (FTS5) retrieval at a 2000-token budget; `apply_feedback` endpoint with audit log and demotion-pressure auto-demote; onboarding scanner with three extractors (filesystem walk, git log, Python AST); 11-command CLI; 8-tool MCP server (under the `[mcp]` optional extra); Claude Code `UserPromptSubmit` hook wiring; reproducible synthetic regression harness (`aelf bench`); academic benchmark adapter scaffolds (inert at v1.0 — see [`benchmarks/README.md`](../benchmarks/README.md) for the activation schedule).

## v1.0.1 — launch fix-up

The first patch closes the highest-visibility gaps from launch. Each item is tracked in [LIMITATIONS.md § known issues at v1.0](LIMITATIONS.md#known-issues-at-v10).

- **Hook layer.** Replace the `UserPromptSubmit` hook's inline FTS5 shim with a thin `hook_search` module (`search_for_prompt` + `record_retrieval`). The hook routes through `aelfrice.retrieval.retrieve()` and writes a `feedback_history` row tagged `source='hook'` for every retrieval. Closes the loop between hook-time retrievals and the Bayesian feedback math.
- **Onboard noise filter.** A dedicated module that removes Markdown headings, checklist items, three-word fragments, and license-header boilerplate before they enter as candidate beliefs.
- **Onboard performance.** A regression test verifying < 60s on a 50k-LOC project.
- **Contradiction tie-breaker.** Default precedence `user_stated > user_corrected > document_recent > agent_inferred`, with ISO timestamp as the deciding fallback. The loser is auto-superseded; the audit row records which rule fired.
- **`aelf --version`.** Reads from `aelfrice.__version__`. Trivial, but currently raises an argparse error.
- **Query result cache.** A bounded LRU cache wrapping `aelfrice.retrieval.retrieve()`, keyed on a canonicalized form of `(query, token_budget, l1_limit)` and invalidated on every store mutation. Skips a full L0+L1 pass when an agent loop re-issues the same query. Spec: [`lru_query_cache.md`](lru_query_cache.md).

## v1.0.2 — install routing + release guardrails

Second patch. Two threads:

- **Per-project install routing + `aelf doctor`.** `aelf setup` auto-detects scope (`project` if `cwd/.venv` matches the active interpreter, else `user`) and writes an absolute `aelf-hook` path so one machine can route per-project venvs to their own hook alongside a global `pipx`-installed fallback without bare-name `$PATH` collisions. `aelf doctor` is a settings linter that scans user- and project-scope `settings.json` for hook + statusline commands whose program token doesn't resolve, exits `1` on findings so it can gate CI. Closes [#81](https://github.com/robotrocketscience/aelfrice/issues/81).
- **Release-docs CI gate + post-release docs sweep.** The v1.0.1 cut shipped to PyPI with the README roadmap row still saying `v1.0.1 | next`. New `staging-gate.yml` `release-docs-check` job enforces, on any version-bump PR, that `CHANGELOG.md` has the matching `[X.Y.Z]` section + compare-link footnote and that `README.md` has no roadmap row marking the released version as `next` / `planned`. New `post-release-docs-issue.yml` opens a tracking issue on `release.published` for second-order docs the gate can't verify (RELEASING.md test counts, ROADMAP.md narrative).

## v1.0.3 — contradiction tie-breaker + onboard perf + CONFIG.md

Third patch. Three feature PRs and a docs PR landed between v1.0.2 and v1.0.3, surfaced to PyPI as v1.0.3. See [CHANGELOG § v1.0.3](../CHANGELOG.md#103---2026-04-27) for the full surface.

- **Contradiction tie-breaker.** New `aelfrice.contradiction` module (`resolve_contradiction`, `find_unresolved_contradictions`, `auto_resolve_all_contradictions`). When the graph holds a `CONTRADICTS` edge, the tie-breaker picks a winner via precedence (`user_stated > user_corrected > document_recent`; ties broken by recency, then id), creates a `SUPERSEDES` edge from winner to loser, and writes a `feedback_history` audit row tagged `source='contradiction_tiebreaker:<rule>'`. v1.0.x collapses to three precedence classes; the fourth `agent_inferred` class needs a `Belief.origin` field that lands in v1.1.0. PR [#75](https://github.com/robotrocketscience/aelfrice/pull/75).
- **`aelf resolve` CLI subcommand (12th).** Sweeps unresolved `CONTRADICTS` edges and runs the tie-breaker on each. Matching `slash_commands/resolve.md`; the 1:1 CLI ↔ slash invariant is preserved at 12/12. PR [#75](https://github.com/robotrocketscience/aelfrice/pull/75).
- **Onboard performance regression test.** `tests/regression/test_onboard_perf_50k_loc.py` asserts `scan_repo` finishes in under 60s on a synthetic ~55k-LOC project (250 .py + 60 doc files), held against the `:memory:` store. Current measured time ~0.8s on Apple Silicon; the 60s budget is a regression alarm, not a target. PR [#76](https://github.com/robotrocketscience/aelfrice/pull/76).
- **`docs/CONFIG.md`.** Power-user reference for `.aelfrice.toml` — full schema, worked examples, and what each setting affects. Linked from the noise-filter LIMITATIONS entry, the onboard CLI help, and ARCHITECTURE.md. PR [#74](https://github.com/robotrocketscience/aelfrice/pull/74).

Release cut: PR [#97](https://github.com/robotrocketscience/aelfrice/pull/97).

## v1.1.0 — project identity and cosmetic surface

Minor release. Eight PRs landed between v1.0.3 and v1.1.0. See [CHANGELOG § v1.1.0](../CHANGELOG.md) for the full surface.

- ✅ **Per-project DB resolution.** v1.0.x shipped a single global DB at `~/.aelfrice/memory.db` shared across every project on the machine. v1.1.0 lands a resolution chain: `$AELFRICE_DB` (override) → `<git-common-dir>/aelfrice/memory.db` (when `cwd` is inside a git work-tree) → `~/.aelfrice/memory.db` (non-git fallback). Worktrees of one repo share one DB via the git-common-dir. `.git/` is not git-tracked — the brain graph never crosses the git boundary. Shipped at [#88](https://github.com/robotrocketscience/aelfrice/issues/88) / PR [#96](https://github.com/robotrocketscience/aelfrice/pull/96).
- ✅ **`aelf migrate` from legacy global store.** One-shot copy from `~/.aelfrice/memory.db` (the v1.0 single global DB) into the active project's `.git/aelfrice/memory.db`. Dry-run by default; `--apply` writes. `--all` skips the project-mention filter. Idempotent. PR [#104](https://github.com/robotrocketscience/aelfrice/pull/104).
- ✅ **Worktree concurrency tests + `busy_timeout=5000`.** Multiple worktrees share one `.git/aelfrice/memory.db` without corruption under WAL mode + the new busy_timeout. PR [#102](https://github.com/robotrocketscience/aelfrice/pull/102).
- ✅ **`aelf health` rewritten as diagnostic auditor.** Replaces the v1.0 regime classifier output (preserved as `aelf regime`). Reports credal gap, orphan threads, thread type counts, feedback coverage, FTS5 sync, locked-belief contradictions. Exits 1 on structural failures (orphan threads, FTS5 mismatch, locked-belief contradictions); informational metrics stay exit 0. `aelf status` aliases `aelf health`. PR [#100](https://github.com/robotrocketscience/aelfrice/pull/100).
- ✅ **`edges` → `threads` user-facing rename.** All user-facing surfaces use "threads"; internal schema, `Edge` dataclass, and `EDGE_*` type constants unchanged. MCP `aelf:stats` emits both `edges` and `threads` keys for one minor; `edges` removed in v1.2.0. PR [#105](https://github.com/robotrocketscience/aelfrice/pull/105).
- ✅ **Onboard git-recency weighting.** Scanner records source file's most-recent git commit date as `belief.created_at`, so the existing decay mechanism penalises pre-migration content from old branches. One `git log --name-only --pretty=format:%aI` call per scan. PR [#103](https://github.com/robotrocketscience/aelfrice/pull/103).
- ✅ **`agent_inferred` → `user_validated` promotion path designed.** [docs/promotion_path.md](promotion_path.md) — schema bump, conservative backfill, flag-only flip mechanism, `aelf validate` surface, tie-breaker slot. Implementation lands in v1.2.0. PR [#101](https://github.com/robotrocketscience/aelfrice/pull/101).
- **Query result cache.** A bounded LRU cache wrapping `aelfrice.retrieval.retrieve()`, keyed on a canonicalized form of `(query, token_budget, l1_limit)` and invalidated on every store mutation. Skips a full L0+L1 pass when an agent loop re-issues the same query. Spec: [`lru_query_cache.md`](lru_query_cache.md). Shipped pre-v1.1.0 in [#69](https://github.com/robotrocketscience/aelfrice/pull/69).

## v1.2.0 — auto-capture and triple extraction

- ✅ **Commit-ingest `PostToolUse` hook** ([commit_ingest_hook.md](commit_ingest_hook.md)). PostToolUse:Bash hook that turns each successful `git commit` into a triple-extraction ingest under a deterministic git-derived session id. The first ingest path that densely populates `Edge.anchor_text`, `Belief.session_id`, and `DERIVED_FROM` edges in production data. Wired via `aelf setup --commit-ingest`; opt-in at v1.2 (default-on candidate at v1.3 once latency telemetry confirms the budget holds). Writes are local-only — the hook never crosses the git boundary or any network boundary.
- ✅ **Ingest enrichment schema** ([ingest_enrichment.md](ingest_enrichment.md)). `DERIVED_FROM` edge type, `anchor_text` on edges, `session_id` on beliefs plus a real `sessions` table. Forward-compatible with v1.0 stores via idempotent `ALTER TABLE`. Producers (transcript-ingest, commit-ingest, triple-extraction) populate the new fields.
- ✅ **Transcript-ingest hooks.** A four-event hook (`UserPromptSubmit` + `Stop` + `PreCompact` + `PostCompact`) appends every conversation turn to a per-project `<root>/.git/aelfrice/transcripts/turns.jsonl` log; the `PreCompact` rotation triggers `ingest_jsonl()` which lowers each turn into the brain graph as beliefs and edges. Closes the harness-conflict write-path gap (the MCP no longer depends on the harness's auto-memory directive to receive new beliefs from normal session activity) and densifies production data with `session_id` and `DERIVED_FROM` edges from real conversations. Writes land under `.git/aelfrice/`, which git does not track — transcripts and ingested beliefs never cross the git boundary. Wired via `aelf setup --transcript-ingest` (opt-in at v1.2; default flip pending telemetry). Required prerequisite for the v1.4.0 context rebuilder.
- ✅ **Triple-extraction port** ([triple_extractor.md](triple_extractor.md)). Pure-regex `extract_triples` + side-effecting `ingest_triples` over six relation families (active and passive forms). All triple-derived beliefs share a canonical id space; `anchor_text` is populated from the citing prose. Reusable by every prose-ingesting caller in v1.2+ and by the v1.3.0 entity-index path. Migration is forward-compatible against v1.0 stores — existing rows continue to read.
- ✅ **`agent_inferred → user_validated` promotion** ([promotion_path.md](promotion_path.md)). Onboard-derived beliefs can graduate to user-validated via `aelf validate <id>` (CLI) or `aelf:validate` (MCP) under explicit user action, without being re-locked. New `Belief.origin` column with seven tier values; v1.0/v1.1 stores migrate forward via `ALTER TABLE` plus a one-shot backfill (locked → `user_stated`, correction → `user_corrected`, rest stay `unknown`). Provenance flip only — alpha/beta unchanged on promotion. Reversible via `aelf demote` which flips `user_validated` back to `agent_inferred` (one tier per call: lock first, then validation). Contradiction tie-breaker expands to five classes (`user_stated > user_corrected > user_validated > document_recent > agent_inferred`); `lock_level=user` short-circuits to `user_stated` so locks always win regardless of origin. Audit rows under `promotion:` source prefix.
- ✅ **`docs/HARNESS_INTEGRATION.md`** ([HARNESS_INTEGRATION.md](HARNESS_INTEGRATION.md)). User-facing operational guide covering the auto-memory + aelfrice coexistence story. Documents three modes (default coexistence; aelfrice-canonical with a small `~/.claude/CLAUDE.md` edit; aelfrice-only after disabling auto-memory) and a migration recipe (`aelf onboard ~/.claude/projects/<slug>/memory`) for users who want existing `.md` content brought into the brain graph as `agent_inferred` beliefs. Rewrites [LIMITATIONS.md § harness conflict](LIMITATIONS.md) to point to the v1.2.0 hook mitigation instead of the v1.0/v1.1 manual workaround.
- **First academic benchmark activations.** `mab_triple_adapter` activates in `benchmarks/`.

## v1.3.0 — retrieval wave

This is the release where retrieval moves beyond BM25-only.

- **Entity-index retrieval.** The L2.5 entity-index from the earlier research line ports forward, including the regex extraction patterns. Reproduces the multi-hop chain-valid target from published baselines on the MAB benchmark.
- **BFS multi-hop graph traversal.** Edge-type-weighted graph walks layer on top of FTS5 hits. Bounded depth, bounded budget.
- **LLM-classification onboard path.** A Haiku-backed classifier becomes an opt-in alternative to the regex classifier. Cost is documented per-session in the published claims.
- **Bayesian-weighted ranking — partial.** Retrieval scoring begins to incorporate posterior confidence on top of BM25. The full feedback-into-ranking eval lands in v2.0.0.
- **Benchmark activations.** `mab_entity_index_adapter` and `mab_llm_entity_adapter` activate in `benchmarks/`.

## v1.4.0 — context rebuilder

The release that makes long-running sessions cheaper without a visible seam to the user.

- **`PreCompact`-driven context rebuild.** When the harness signals an approaching context limit, an aelfrice hook intercepts and queries the brain graph for the highest-value beliefs against the tail of the session transcript. The rebuild emits as `additionalContext`: locked beliefs first, then session-scoped beliefs from the same `session_id`, then BM25 / posterior-weighted hits, packed to a configurable token budget. The user's next prompt is answered against a leaner, retrieval-curated working set instead of the harness's generic compaction summary.
- **Augment mode at v1.4.0.** The hook augments the harness's default compaction; both summaries land in the new context. Suppress mode (replacing the harness's compaction entirely) is parked as a v2.x candidate gated on continuation-fidelity evidence.
- **Trigger modes.** Manual (`/aelf-rebuild` slash command) ships first as the explicit testing surface. Threshold mode (auto-fire at a configurable fraction of the model's window) ships with calibration data — the default threshold is derived from the eval harness, not hardcoded. Dynamic mode (heuristic-driven trigger) is gated on showing it tracks fidelity better than a fixed threshold.
- **Continuation-fidelity eval harness.** A new harness in `benchmarks/context-rebuilder/` replays captured transcripts, forces a midpoint clear, runs the rebuilder, and measures: (a) fraction of post-clear turns where the agent's answer matches the original session's answer, (b) rebuild block size as a fraction of the full-replay baseline, (c) PreCompact hook latency. Headline regression band: ≥80% continuation fidelity at ≤30% token cost on the v1.0.0 baseline. Higher targets land at v1.5.x and v2.0.0.
- **Hard prerequisites.** v1.2.0 transcript-ingest (the rebuilder reads `<root>/.git/aelfrice/transcripts/turns.jsonl`) and the v1.2.0 `session_id` schema addition.
- **Soft prerequisites.** v1.3.0 partial Bayesian-weighted ranking improves rebuild quality; without it the rebuilder ships against BM25-only retrieval.

The central claim of v1.4.0: long-running sessions use less context per steady-state turn without measurable continuation regression. The eval harness is what makes that claim falsifiable.

## v2.0.0 — feature parity and reproducibility

The release that re-anchors every published claim. After v2.0.0, the benchmark harness in `benchmarks/` reproduces the headline numbers on a fresh clone with `uv sync && aelf bench all`, within documented tolerance bands.

- **HRR vocabulary bridge.** Closes the vocabulary-gap-recovery claim against a corpus checked into the repository.
- **Type-aware compression.** Tokens-per-belief reductions on retrieved output. Reproduces the published compression ratio on a fixed input corpus.
- **Intentional clustering.** Co-locating related beliefs in the graph for higher retrieval coherence on multi-fact queries.
- **Correction-detection eval.** A five-codebase labeled fixture, scored by both the zero-LLM detector and the LLM-judge path. Reproduces the published 92% / 99% targets.
- **Bayesian feedback drives ranking.** The full feedback-into-retrieval loop. The 10-round MRR uplift eval and ECE calibration scorer are part of this release.
- **Expanded MCP surface and CLI.** `wonder`, `reason`, `core`, `unlock`, `delete`, `confirm`, plus graph-metrics and document-linking tools that exist in the earlier research line.
- **Reproducibility harness.** `benchmarks/results/v2.0.0.json` is the canonical results file; CI runs the academic suite nightly (Haiku) and on each tag (Opus, manual approval).

## What is being recovered, in detail

The earlier research line implemented these modules. They are not in v1.0.0; each is queued for the version listed below. The names are working titles for the public ports — the implementation is being re-validated rather than copied.

| Capability | Public version |
|---|---|
| Triple extraction (`triple_extraction`) | v1.2.0 |
| Commit-ingest `PostToolUse` integration (`commit_tracker`) | v1.2.0 |
| Transcript-ingest hooks + `ingest_jsonl()` (`transcript_ingest`) | v1.2.0 |
| Context rebuilder + continuation-fidelity eval (`context_rebuilder`) | v1.4.0 |
| Graph metrics + status/health split (`graph_metrics`) | v1.1.0 |
| Entity-index retrieval (regex + entity patterns) | v1.3.0 |
| BFS multi-hop graph traversal | v1.3.0 |
| LLM-Haiku onboard classifier | v1.3.0 |
| Hook-driven retrieval audit (`hook_search`) | v1.0.1 |
| Update-check / `aelf --version` (`update_check`) | v1.0.1 |
| Onboard noise filter (`noise_filter`) | v1.0.1 |
| HRR vocabulary bridge (`hrr`) | v2.0.0 |
| Type-aware compression (`compression`) | v2.0.0 |
| Doc / semantic linker (`doc_linker`, `semantic_linker`) | v2.0.0 |
| Full edge-type vocabulary (`SUPPORTS`, `TESTS`, `IMPLEMENTS`, `TEMPORAL_NEXT`, `POTENTIALLY_STALE`, `DERIVED_FROM`) | v1.2.0 partial; v2.0.0 complete |
| Relationship detector / supersession (`relationship_detector`, `supersession`) | v1.2.0 partial; v2.0.0 complete |
| Wonder / reason / core / unlock / delete / confirm CLI commands | v2.0.0 |
| Correction-detection module (production path) | v1.2.0 |
| Correction-detection eval corpus | v2.0.0 |

## Built fresh, not copied

The rewrite is fixing the following structural issues at the foundation rather than patching them forward.

- **No per-project DB identity.** v1.0 shipped a single global DB at `~/.aelfrice/memory.db` shared across every project on the machine; onboarding repo A and repo B mixed their beliefs in one file. Fixed in v1.1.0 ([#88](https://github.com/robotrocketscience/aelfrice/issues/88)) by introducing per-project resolution: the DB defaults to `.git/aelfrice/memory.db` for git repos (keyed off the git-common-dir so worktrees share one DB) and falls back to `~/.aelfrice/memory.db` for non-git directories. `.git/` is not git-tracked, so the brain graph never crosses the git boundary. A one-shot `aelf migrate` copies beliefs from the legacy global store into the active project's in-repo store.
- **Hook → retrieval coupling** missed the feedback-history audit row, so retrievals did not exercise posteriors and were not auditable. Fixed in v1.0.1 by routing the hook through the same `retrieval.retrieve()` codepath as the CLI / MCP, with one `feedback_history` row per retrieval.
- **Contradictions** were detected but never resolved — both beliefs remained equally retrievable, with a warning logged. Fixed in v1.0.1 by a default tie-breaker that auto-supersedes the loser and records the rule that fired.
- **Onboard noise** (Markdown headings, license boilerplate, three-word fragments) entered as first-class beliefs and depressed signal-to-noise on every subsequent retrieval. Fixed in v1.0.1 by the `noise_filter` module wired into the synchronous onboard path.
- **BFS multi-hop temporal coherence.** Each hop in a multi-hop chain currently resolves to the globally latest serial independently; this misses chains where intermediate hops should follow earlier serials. Open research as of v1.3.0; documented openly. Will not be patched into v1.x without a benchmark delta to justify the design.
- **`apply_feedback` write authority under Claude Code.** The harness directive currently routes "save a memory" intents to a separate file-based store, so the MCP receives no new beliefs from normal sessions. Documented in [LIMITATIONS.md § harness conflict](LIMITATIONS.md). v1.0.1 closes the read-side of the loop (hook-driven retrievals exercise posteriors); v1.2 publishes a documented procedure for users who want the MCP to be canonical.

## Validation commitment

Every release with a behavioural claim ships with a benchmark or test that demonstrates it. v1.0.0's central claim is that `apply_feedback` updates posteriors mathematically — proven by `tests/test_scoring.py` and the synthetic harness. v1.0.0 makes *no* claim that BM25 ranking moves under feedback, because it doesn't yet. v1.3.0 is the release where that claim becomes testable; v2.0.0 is where the published MRR uplift is reproduced or retracted.

This is what "early access" means in the v1.0 announcement: the surface is stable, the central feedback-into-ranking claim is a future deliverable, and the benchmark harness ships now so the eventual delivery is auditable.

## Compatibility commitment

aelfrice follows semver. Within the v1.x series:

- **Patch releases (v1.0.x)** preserve API and database schema.
- **Minor releases (v1.x.0)** preserve API; database migrations are forward-compatible (new columns / new tables only). Schema changes that read forward-compatibly require no user action.
- **v2.0.0** may break the v1 API only where a benchmark or eval justifies the break. Migrations from v1 to v2 are documented and tested before tag.

Each version above ships as a real tagged release on PyPI. Users on a working v1.x release are not blocked behind v2.0.0 development.

## Non-goals

- **Vector database / embedding-based retrieval.** aelfrice stays SQLite + FTS5 at every milestone in this roadmap. The HRR bridge in v2.0.0 is a structural retrieval layer, not an embedding layer.
- **Cloud sync.** The runtime stays local. No release in this roadmap introduces network IO in the retrieval path or the write path.
- **Telemetry.** No release in this roadmap adds outbound network calls in the default install. The optional `[mcp]` extra adds the MCP transport (local).
- **Brain-graph sharing, sync, or export between users / machines / projects.** No release in this roadmap ships a mechanism for distributing memory contents outside the machine they were written on. See [LIMITATIONS.md § Sharing or sync of brain-graph content](LIMITATIONS.md#sharing-or-sync-of-brain-graph-content) for the architectural rationale.

## Where this list grows

Issues filed against this repository drive the roadmap. Each release closes a list of issues mapped at [LIMITATIONS.md § known issues at v1.0](LIMITATIONS.md#known-issues-at-v10). Roadmap drift between this document and the issue tracker is itself a bug — file it.
