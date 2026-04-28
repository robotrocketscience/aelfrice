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
| **v1.3.0** | shipped | retrieval wave — entity index (L2.5), BFS multi-hop (L3), LLM-Haiku onboard classifier (opt-in), partial posterior-weighted ranking |
| **v1.4.0** | shipped | context rebuilder — PreCompact retrieval-curated continuation (augment mode); manual + threshold trigger; continuation-fidelity scorer (exact-match) |
| **v1.5.0** | shipped | retrieval plumbing — composition plumbing + telemetry, BM25F anchor text, search-tool Bash matcher, v3 federation version-vector schema, dynamic-trigger re-park |
| **v1.6.0** | planned | graph signal wave — signed Laplacian + eigenbasis, heat kernel authority, posterior-weighted ranking (full), Plate FFT HRR primitives |
| **v1.7.0** | planned | structural retrieval lane + composition default-on flip — HRR bind/probe, `uri_baki` post-rank adjuster retest, benchmark-gate default-on flip (#154) |
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

### v1.3.0 — retrieval wave

The release where retrieval moves beyond BM25-only.

- **Entity-index retrieval.** L2.5 entity-index, including the regex extraction patterns. Spec: [entity_index.md](entity_index.md).
- **BFS multi-hop graph traversal.** Edge-type-weighted graph walks layered on top of FTS5 hits. Bounded depth, bounded budget. Spec: [bfs_multihop.md](bfs_multihop.md).
- **LLM-classification onboard path.** Haiku-backed classifier as an opt-in alternative to the regex classifier. Default-off; opt in via `aelf onboard --llm-classify` or `[onboard.llm].enabled = true` in `.aelfrice.toml`. Boundary policy and prompt template in [llm_classifier.md](llm_classifier.md). PRIVACY note: first outbound call in the install path that transmits user content; see [PRIVACY § Optional outbound calls](PRIVACY.md#optional-outbound-calls).
- **Posterior-weighted ranking (partial).** Retrieval scoring incorporates `α / (α+β)` on top of BM25, log-additively at weight 0.5. See [bayesian_ranking.md](bayesian_ranking.md) for the v1.3 contract. Full feedback-into-ranking eval — 10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — lands at v1.6.0 (#151) / re-runs for the canonical cut at v2.0.0.

### v1.4.0 — context rebuilder

Long-running sessions cheaper without a visible seam.

- **PreCompact-driven rebuild.** When the harness signals an approaching context limit, an aelfrice hook queries the brain graph for the highest-value beliefs against the session tail and emits them as `additionalContext`. Locked beliefs first, then session-scoped, then BM25 / posterior-weighted hits, packed to a configurable token budget.
- **Augment mode.** The hook augments the harness's compaction; both summaries land in the new context. Replace mode parked for v2.x.
- **Trigger modes** ([#141](https://github.com/robotrocketscience/aelfrice/issues/141)). Manual (`/aelf:rebuild`) and threshold shipped at v1.4; dynamic parked to v1.5 (the v1.4 ship-gate investigation in `benchmarks/context_rebuilder/dynamic_probe.py` did not produce ≥ 5% absolute fidelity uplift over threshold at same-or-lower token cost on the synthetic fixture; revisit tracked at #188). Threshold default fraction (0.6) sourced from eval-harness calibration in `benchmarks/context-rebuilder/calibration_v1_4_0.json`. Manual is the v1.4 ship default; threshold is opt-in until production telemetry.
- **Continuation-fidelity eval.** `benchmarks/context-rebuilder/` replays fixture transcripts, forces a midpoint clear, runs the rebuilder, and measures continuation fidelity vs. the full-replay baseline. Fixture corpus policy (synthetic public for CI / headline number; captured corpus held lab-side for offline calibration only) decided in [eval_fixture_policy.md](eval_fixture_policy.md).

Hard prerequisites: v1.2 transcript-ingest, v1.2 `session_id` schema. Alpha shipped in v1.2.0a0.

## Planned

### v1.2.x — search-tool hook (planned patch)

Pulled forward from v1.3.0 to validate the `PreToolUse` retrieval surface ahead of the bigger retrieval wave. Ships against the v1.0 retrieval pipeline + v1.1.0 per-project DB resolution; no dependency on entity-index, BFS, or LLM classifier work.

- **Search-tool `PreToolUse` hook** ([search_tool_hook.md](search_tool_hook.md)). Fires before `Grep` and `Glob` tool calls, lifts the agent's search query out of `tool_input.pattern`, runs the same query against the per-project belief store, and emits results as `additionalContext` so the agent sees them *before* the tool runs. First retrieval-shaped hook on the agent's *own* tool intent (the v1.0.1 `UserPromptSubmit` hook covers user-initiated retrieval; this covers agent-initiated). Opt-in via `aelf setup --search-tool` at v1.2.x; matcher extension to other tools tracked at #155 for v1.5.

### v1.5.0 — retrieval plumbing

Composition gate first, then cheap retrieval wins. No new ranking math in this minor; that's v1.6.

- **Pipeline composition tracker — unified `retrieve()` with feature-flag gate** ([#154](https://github.com/robotrocketscience/aelfrice/issues/154)). One entry point, every retrieval feature behind a config flag, telemetry per lane. This is the prerequisite for the v1.6 graph wave: `retrieve()` must be the only path before heat kernel and posterior-full can ship safely behind defaults.
- **Augmented BM25F (incoming-edge anchor text) + vectorized BM25 sparse matvec** ([#148](https://github.com/robotrocketscience/aelfrice/issues/148), merged on `main`). +0.06 NDCG @ +0 ms vs BM25 in the v1.6 component bake-off — adopt now since runtime cost is free.
- **Search-tool hook — extend matcher beyond `Grep|Glob`** ([#155](https://github.com/robotrocketscience/aelfrice/issues/155)). Carryover from v1.3. Widens the `PreToolUse` matcher list once telemetry from v1.2.x confirms latency budget on the existing two.
- **v1.4 dynamic-trigger revisit** ([#188](https://github.com/robotrocketscience/aelfrice/issues/188)). The dynamic mode parked at v1.4 ship-gate (no ≥ 5% absolute fidelity uplift over threshold on synthetic fixture) gets a second eval pass on captured-corpus calibration data. Keeps parking if the bar still isn't met.

### v1.6.0 — graph signal wave

The release where ranking moves beyond BM25 + L2.5 + BFS into graph-authority and full posterior-weighted territory.

- **Signed normalized Laplacian + offline eigenbasis (top-K=200) builder** ([#149](https://github.com/robotrocketscience/aelfrice/issues/149)). Offline-only build step; no runtime cost. Hard prerequisite for #150.
- **Heat kernel authority signal via precomputed eigenbasis** ([#150](https://github.com/robotrocketscience/aelfrice/issues/150)). +0.41 NDCG @ +7.8 ms p50 on a 50k-belief store — biggest single retrieval gain in the bake-off. Ships default-on; latency stays inside the v1.2.x search-tool hook's 50 ms median budget.
- **Posterior-weighted ranking via Beta-Bernoulli prior — full** ([#151](https://github.com/robotrocketscience/aelfrice/issues/151)). Closes the v1.3 partial. Log-additive, weight 0.5, with the 10-round MRR uplift eval and ECE calibration scorer that v1.3 deferred. The central feedback-into-ranking claim becomes fully testable here; v2.0 only re-runs it for the canonical reproducibility cut.
- **Plate FFT HRR primitives — port to public repo** ([#216](https://github.com/robotrocketscience/aelfrice/issues/216)). Hard prerequisite for the v1.7 HRR structural-query lane (#152); lifted to v1.6 to land the math + tests ahead of the lane wiring.

Hard prerequisite: v1.5 #154 composition tracker (heat kernel and posterior-full must ship behind the unified `retrieve()` entry point with telemetry per lane).

### v1.7.0 — structural retrieval lane and research retests

- **HRR structural-query lane (bind/probe over outgoing edges)** ([#152](https://github.com/robotrocketscience/aelfrice/issues/152)). A separate retrieval lane, not a projection — naive HRR projection into BM25 ranking was rejected at -0.10 NDCG (R9 in the bake-off). Bind/probe over outgoing edges is the structural-query path that survives. Persists `id_vec` per belief; `enable_hrr` config flag default-off until any belief has HRR vectors written.
- **`uri_baki` post-rank adjuster retest with relevance-aware locked set** ([#153](https://github.com/robotrocketscience/aelfrice/issues/153)). Named after Uri and Baki, the two Fire Aelfmaidens in Gene Wolfe's *The Wizard Knight* (2004) — bound attendants who operate after the main action to tilt outcomes for their bound knight. Locked-floor (Uri's protection), supersession demote (Baki's undermining), recency decay (the Aelfrice time-tilt). The pattern is publicly described as Google's "Twiddler" in Pandu Nayak's DOJ testimony (October 2023) and the May 2024 Content Warehouse API leak; aelfrice uses neutral naming to avoid trading on Google's term-of-art. A prior random-pinned synthetic regressed -0.05 NDCG due to a methodology bug (random pinning drowned signal). The relevance-aware retest uses `Belief.lock_state` and decides whether the lane is dead or just needed a fairer eval. Research item; ships only if the retest beats the BM25F + heat-kernel + posterior baseline.

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
| LLM-Haiku onboard classifier (opt-in) | v1.3.0 |
| LLM-Haiku onboard classifier (default-on, host-driven) | v1.5.0 |
| Posterior-weighted ranking | v1.3.0 (partial) / v1.6.0 (full) |
| BM25F anchor-text + vectorized BM25 | v1.5.0 |
| Signed Laplacian + heat-kernel authority | v1.6.0 |
| HRR structural-query lane | v1.7.0 |
| HRR vocabulary bridge | v2.0.0 |
| Type-aware compression | v2.0.0 |
| Doc / semantic linker | v2.0.0 |
| Graph-traversal store methods (`expand_graph`, `get_neighbors`, `edge_exists`) | v1.5 or v1.6 (substrate for `wonder` + `reason`; ships ahead of v2.0) |
| `ingest_turn(bulk=)` parameter | v2.0.0 ([#194](https://github.com/robotrocketscience/aelfrice/issues/194); mechanical, post-`wonder_ingest` port) |
| `scoring.uncertainty_score(α, β)` | v2.0.0 ([#195](https://github.com/robotrocketscience/aelfrice/issues/195); conditional on substrate decision) |
| Multi-axis uncertainty substrate (`UncertaintyVector`) | v2.0.0 substrate decision ([#196](https://github.com/robotrocketscience/aelfrice/issues/196); load-bearing — blocks `wonder` + `reason`) |
| Speculative-belief schema migration (3 columns + 1 belief type + 2 edge types) | v2.0.0 (depends on #196) |
| Speculative / causal edge types (`SPECULATES`, `DEPENDS_ON`, `RESOLVES`, `HIBERNATED`) | v2.0.0 (with `wonder`) |
| `wonder` (gap-analysis frontend) | v2.0.0 (depends on substrate + graph-traversal) |
| `wonder_ingest` + `wonder_gc` (speculative-belief lifecycle) | v2.0.0 (depends on substrate) |
| `reason` (graph-walk reasoning) | v2.0.0 (depends on graph-traversal) |
| `core` / `unlock` / `delete` / `confirm` (CLI surface) | v2.0.0 |
| Directive-detection + compliance-audit + selective-injection triad | v2.0.0 candidate ([#199](https://github.com/robotrocketscience/aelfrice/issues/199)) |
| Sentiment-from-prose feedback | v2.0.0 candidate ([#193](https://github.com/robotrocketscience/aelfrice/issues/193)) |
| Near-duplicate audit (`aelf doctor dedup`) | v1.x candidate ([#197](https://github.com/robotrocketscience/aelfrice/issues/197)) |
| Multi-model belief classifier (SIGNAL/NOISE/STALE/CONTESTED) | v2.0.0 candidate ([#198](https://github.com/robotrocketscience/aelfrice/issues/198)) |
| Automatic CONTRADICTS detection (semantic-divergence) | v1.x candidate ([#201](https://github.com/robotrocketscience/aelfrice/issues/201)) |

The four "candidate" lines are the orphaned research-line capabilities from the agentmemory parity audit — neither shipping today nor previously listed on this roadmap. They land if and only if a benchmark or experiment justifies the inclusion (per the validation discipline below); otherwise they stay parked.

### Deliberately not on this list

The research line also shipped the following capabilities that aelfrice does **not** plan to recover:

- **Research-artifact provenance metadata** (`produced_at` / `method` / `sample_size` / `data_source` / `independently_validated` per belief) and the **rigor-tier** classification layer (`hypothesis` / `simulated` / `empirically_tested` / `validated`). Motivated in the research line by a case study where a new agent miscalibrated project maturity from raw completion counts. aelfrice v1 stores provenance via the `Belief.origin` enum (7 source-tier values) only; epistemic-rigor metadata is **not** on the v2.0 surface. If status reporting needs this signal it lands as a separate feature with its own benchmark, not as a schema-wide migration.
- **Session-velocity tracking** (items/hour decay scaling). v1 ships per-belief decay with type-specific half-lives; velocity-scaled decay is the research-line refinement and is parked.
- **Calibrated status reporting** that surfaces rigor-tier distribution and velocity context to a new agent. Depends on the two items above.
- **Cross-project shared scopes** via SQLite ATTACH. Subsumed by the Multi-project query non-goal in [LIMITATIONS § Sharing, sync, or federation](LIMITATIONS.md). Named here so the research-line term ("shared scopes") doesn't read as an oversight.
- **Obsidian vault export** and **vault-as-source-of-truth** storage. Rejected at v1: SQLite is the source of truth, per-project isolation is a hard property. Subsumed by the same federation non-goal.

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
