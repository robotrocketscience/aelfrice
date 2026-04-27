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
| **v1.1.0** | planned | project identity, edges→threads, status/health split |
| **v1.2.0** | planned | commit-ingest hook, seed files, triple-extraction port |
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

## v1.1.0 — project identity and cosmetic surface

- **In-repo store.** `.git/aelfrice/memory.db` becomes the default location. The current `SHA256(cwd)`-keyed path produces orphan databases on directory rename, machine move, or worktree creation; the new layout is portable and survives all three.
- **`.aelfrice.toml`.** Optional cross-machine project identity. A repo's memory follows the repo, not the absolute path.
- **Orphan-DB cleanup tooling.** A migration command finds and merges abandoned per-`cwd` databases into the in-repo store.
- **Worktree concurrency tests.** Multiple worktrees of the same repo share one `.git/aelfrice/memory.db` without corruption.
- **`aelf health` vs `aelf status` split.** `status` becomes a counts snapshot; `health` becomes a real graph auditor — orphan edges, isolated clusters, FTS5 sync, locked-belief contradictions, decay anomalies.
- **`edges` → `threads`.** User-facing rename in CLI output and MCP tool descriptions. Internal schema is unchanged.
- **Onboard improvements.** Git-recency weighting for source files; explicit `agent_inferred` → `user_validated` promotion path so onboard-derived beliefs can graduate without being re-locked.

## v1.2.0 — auto-capture and triple extraction

- **Commit-ingest `PostToolUse` hook.** The graph grows during normal sessions without explicit `onboard` / `remember` / `feedback` calls.
- **Transcript-ingest hooks.** A pair of `UserPromptSubmit` + `Stop` hooks append every conversation turn to a per-project `<root>/.git/aelfrice/transcripts/turns.jsonl` log; a `PreCompact` hook rotates the log and triggers `ingest_jsonl()` to pull turns into the brain graph as beliefs and edges. Closes the harness-conflict write-path gap (the MCP no longer depends on the harness's auto-memory directive to receive new beliefs from normal session activity) and densifies production data with `session_id` and `DERIVED_FROM` edges from real conversations. Required prerequisite for the v1.4.0 context rebuilder.
- **`.aelfrice/seed.md`.** A git-tracked seed file auto-ingested on first `onboard`. Lets a project author bootstrap collaborator memory directly from the repository.
- **Triple-extraction port.** The triple-extraction ingest module from the earlier research line ports forward. Migration is forward-compatible against v1.0 stores — existing rows continue to read.
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
- **Cross-project shared store.** Optional, opt-in. Beliefs scoped `shared` participate across projects; project-scoped beliefs do not. The default remains per-project isolation.
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
| Cross-project shared scopes (`shared_scopes`) | v2.0.0 |
| Full edge-type vocabulary (`SUPPORTS`, `TESTS`, `IMPLEMENTS`, `TEMPORAL_NEXT`, `POTENTIALLY_STALE`, `DERIVED_FROM`) | v1.2.0 partial; v2.0.0 complete |
| Relationship detector / supersession (`relationship_detector`, `supersession`) | v1.2.0 partial; v2.0.0 complete |
| Wonder / reason / core / unlock / delete / confirm CLI commands | v2.0.0 |
| Correction-detection module (production path) | v1.2.0 |
| Correction-detection eval corpus | v2.0.0 |

## Built fresh, not copied

The rewrite is fixing the following structural issues at the foundation rather than patching them forward.

- **Per-project DB identity** keyed off arbitrary `cwd` hashes silently produced orphan stores on directory rename, machine move, and worktree creation. Fixed in v1.1.0 via in-repo storage and `.aelfrice.toml`.
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
- **Cloud sync.** The runtime stays local. Cross-project sharing in v2.0.0 is opt-in and on-disk; it does not introduce network IO in the retrieval path.
- **Telemetry.** No release in this roadmap adds outbound network calls in the default install. The optional `[mcp]` extra adds the MCP transport (local).

## Where this list grows

Issues filed against this repository drive the roadmap. Each release closes a list of issues mapped at [LIMITATIONS.md § known issues at v1.0](LIMITATIONS.md#known-issues-at-v10). Roadmap drift between this document and the issue tracker is itself a bug — file it.
