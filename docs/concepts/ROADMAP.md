# Roadmap

Release history and forward-looking design cuts. As of v3.0 the v1.0→v2.0
parity arc is complete and v3.0 has shipped; see the per-version rows below.

Per-issue tracking: [LIMITATIONS.md](../user/LIMITATIONS.md). Release log: [CHANGELOG.md](../CHANGELOG.md).

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
| **v1.5.1** | shipped | corroboration tracking sibling table (#190); default-on host-driven LLM onboard classifier (#238) |
| **v1.6.0** | shipped | hardening + observability — hook-hardening framing-tag contract + per-turn audit log, `aelf tail`, belief retention class, rebuild diagnostic log, posterior-ranking eval harness + heat-kernel composition (default-flip gated), deferred-feedback sweeper, v2.0 corpus public scaffold + bench-gate, `replay_full_equality` probe, `session_id` propagation, reachable-install detection |
| **v1.7.0** | shipped | graph signal wave + structural retrieval lane — signed Laplacian + eigenbasis (#149), heat kernel authority (#150), Plate FFT HRR primitives (#216), HRR bind/probe (#152), `uri_baki` post-rank adjuster retest (#153). Heat-kernel + HRR-structural shipped opt-in; benchmark-gate default-on flip (#154) deferred to v2.1.0. |
| **v2.0.0** | shipped | feature parity with the research line + reproducibility-harness scaffolding (#437); wonder lifecycle + dispatch surface; sentiment-feedback module (hook integration pending); dedup read-path (audit-only) |
| **v2.1.0** | shipped | reproducibility-harness gate cleared 11/11 (#437); `use_heat_kernel` + `use_hrr_structural` defaults flipped on (#154); HRR `dim` default 2048→512 (#538); default-on transcript / commit / session-start hooks (#529); query-strategy uplift bench gate (#291); vocab-bridge bench gate (#433) |
| **v3.0.0** | shipped 2026-05-13 | wonder lifecycle complete (#542 umbrella + #547/#550/#552); wonder/reason parity #645 (Verdict/ImpasseKind, ConsequencePath fork on CONTRADICTS, VERDICT-driven dispatch + close-the-loop suggested-updates); HRR persistence default-ON + split-format save/load (#553); type-aware compression A2 bench gate (#434); eval-harness host-agent replay + LLM-judge stage + Cohen's-κ runner (#592, #600, #687); read-only federation — `scope` field + peer DB FTS5/BFS + foreign-id rejection (#650, #655, #688, #690, #713); `query_strategy` default flipped legacy-bm25 → stack-r1-r3 (#718); phantom-promotion Surface A + Surface B (#550, #616); sentiment-feedback UPS hook (#606); self-installing hook manifest (#623); merge-train label-driven serialized merger (#602). Ratified design decisions: PHILOSOPHY stays deterministic (#605); multimodel deferred (#607); federation read-only (#661). Milestone tracker: [#608](https://github.com/robotrocketscience/aelfrice/issues/608). |
| **v3.0.1** | shipped 2026-05-13 | install-surface collapse: pipx/pip/venv channels removed, `uv tool install` is the single supported path (#730); auto-migrate non-uv installs to `uv tool` on first 3.0.1 `aelf setup` (#733, follow-up #774 for the uv-not-found one-liner); `search-tool` + `search-tool-bash` `PreToolUse` hooks default-on so the agent's own Grep/Glob/Bash-search calls go through the belief store first (#738); cross-fire injection dedup ring so back-to-back UPS + PreToolUse fires don't re-surface the same belief in the same turn (#740); transitive `authlib` 1.7.0 → 1.7.2 for CVE-2026-44681 (zero-exposure surface — `[mcp]` extra, aelfrice has no OIDC authorization endpoint). |

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

- **Entity-index retrieval.** L2.5 entity-index, including the regex extraction patterns. Spec: [entity_index.md](../design/entity_index.md).
- **BFS multi-hop graph traversal.** Edge-type-weighted graph walks layered on top of FTS5 hits. Bounded depth, bounded budget. Spec: [bfs_multihop.md](../design/bfs_multihop.md).
- **LLM-classification onboard path.** Haiku-backed classifier as an opt-in alternative to the regex classifier. Default-off; opt in via `aelf onboard --llm-classify` or `[onboard.llm].enabled = true` in `.aelfrice.toml`. Boundary policy and prompt template in [llm_classifier.md](../design/llm_classifier.md). PRIVACY note: first outbound call in the install path that transmits user content; see [PRIVACY § Optional outbound calls](../user/PRIVACY.md#optional-outbound-calls).
- **Posterior-weighted ranking (partial).** Retrieval scoring incorporates `α / (α+β)` on top of BM25, log-additively at weight 0.5. See [bayesian_ranking.md](../design/bayesian_ranking.md) for the v1.3 contract. The MRR-uplift + ECE-calibration eval harness shipped at v1.6.0 (#151, #306) along with heat-kernel composition wiring (#310), both default-OFF; the default-flip lands at v1.7.0 once the harness clears thresholds against the v2.0 corpus, with a re-run for the canonical cut at v2.0.0.

### v1.4.0 — context rebuilder

Long-running sessions cheaper without a visible seam.

- **PreCompact-driven rebuild.** When the harness signals an approaching context limit, an aelfrice hook queries the brain graph for the highest-value beliefs against the session tail and emits them as `additionalContext`. Locked beliefs first, then session-scoped, then BM25 / posterior-weighted hits, packed to a configurable token budget.
- **Augment mode.** The hook augments the harness's compaction; both summaries land in the new context. Replace mode parked for v2.x.
- **Trigger modes** ([#141](https://github.com/robotrocketscience/aelfrice/issues/141)). Manual (`/aelf:rebuild`) and threshold shipped at v1.4; dynamic parked to v1.5 (the v1.4 ship-gate investigation in `benchmarks/context_rebuilder/dynamic_probe.py` did not produce ≥ 5% absolute fidelity uplift over threshold at same-or-lower token cost on the synthetic fixture; revisit tracked at #188). Threshold default fraction (0.6) sourced from eval-harness calibration in `benchmarks/context-rebuilder/calibration_v1_4_0.json`. Manual is the v1.4 ship default; threshold is opt-in until production telemetry.
- **Continuation-fidelity eval.** `benchmarks/context-rebuilder/` replays fixture transcripts, forces a midpoint clear, runs the rebuilder, and measures continuation fidelity vs. the full-replay baseline. Fixture corpus policy (synthetic public for CI / headline number; captured corpus held lab-side for offline calibration only) decided in [eval_fixture_policy.md](../design/eval_fixture_policy.md).

Hard prerequisites: v1.2 transcript-ingest, v1.2 `session_id` schema. Alpha shipped in v1.2.0a0.

### v1.5.0 — retrieval plumbing

Composition gate first, then cheap retrieval wins. No new ranking math in this minor.

- **Pipeline composition tracker — unified `retrieve()` with feature-flag gate** ([#154](https://github.com/robotrocketscience/aelfrice/issues/154)). One entry point, every retrieval feature behind a config flag, telemetry per lane. Prerequisite for the v1.7 graph wave: `retrieve()` must be the only path before heat kernel and posterior-full can ship safely behind defaults.
- **Augmented BM25F (incoming-edge anchor text) + vectorized BM25 sparse matvec** ([#148](https://github.com/robotrocketscience/aelfrice/issues/148)). +0.06 NDCG @ +0 ms vs BM25 in the component bake-off — adopted since runtime cost is free.
- **Search-tool hook — extend matcher beyond `Grep|Glob`** ([#155](https://github.com/robotrocketscience/aelfrice/issues/155)). Widens the `PreToolUse` matcher list now that telemetry from v1.2.x confirmed latency budget.
- **v1.4 dynamic-trigger revisit** ([#188](https://github.com/robotrocketscience/aelfrice/issues/188)). The dynamic mode parked at v1.4 ship-gate got a second eval pass on captured-corpus calibration data; bar still wasn't met, parked again.

### v1.5.1 — corroboration tracking + default-on host-driven onboard

- **Belief corroboration tracking — sibling table + ingest recorder** ([#190](https://github.com/robotrocketscience/aelfrice/issues/190)). New `belief_corroborations` table records each re-ingest of identical content without disturbing the existing dedup contract. Phantom-prereqs T1 of the #190 session-tracking story (T2 = #191 sweeper, T3 = #192 session_id propagation, both at v1.6.0).
- **Default-on host-driven LLM onboard classifier — no API key required** ([#238](https://github.com/robotrocketscience/aelfrice/issues/238)). `[onboard.llm].enabled` flips default `False → True` via the host model's own Task tool against the smallest model in its stack. Quality matches the v1.3.0 LLM-classifier ceiling; cost is sub-percent of a typical weekly host-plan allowance. Direct-API path (`aelf onboard --llm-classify`) remains the API-key fallback.

### v1.6.0 — hardening, observability, retention

A consolidation release rather than the originally-planned graph-signal wave. Ranking math (#149 / #150 / #216) was punted to v1.7 to land alongside #154's default-on flip; v1.6 instead absorbed the security-hardening surface that #280 surfaced and the observability + bench-gate scaffolding that the rebuild redesign (#288 / #289 / #291) needs.

- **Hook-hardening Phase 1 — framing-tag contract + content escape for memory blocks** ([#280](https://github.com/robotrocketscience/aelfrice/issues/280), [#297](https://github.com/robotrocketscience/aelfrice/pull/297)). Closes the prompt-injection surface where ingested belief content could forge or close the `<aelfrice-memory>` framing tag.
- **Hook-hardening mitigation 3 — per-turn audit log** ([#280](https://github.com/robotrocketscience/aelfrice/issues/280), [#314](https://github.com/robotrocketscience/aelfrice/pull/314)). `<git-common-dir>/aelfrice/hook_audit.jsonl` records the exact rendered hook block on every fire. 10 MB cap, single-slot rotation, fail-soft.
- **`aelf tail` — live observability for hook injections** ([#321](https://github.com/robotrocketscience/aelfrice/issues/321), [#322](https://github.com/robotrocketscience/aelfrice/pull/322)). `tail -f`-style pretty-printer over the audit log. The audit record itself is extended with `beliefs[]`, `latency_ms`, `tokens`.
- **Belief retention class + per-source aging policy** ([#290](https://github.com/robotrocketscience/aelfrice/issues/290)). Schema column on `beliefs`, per-ingest-source defaults wired into `derive()` and the scanner, promotion path via `aelf doctor --promote-retention`. Foundational layer for v2.0 aging / pruning; no automated retention-driven eviction yet.
- **Rebuild diagnostic log — phase-1a write + phase-1c audit script** ([#288](https://github.com/robotrocketscience/aelfrice/issues/288), [#302](https://github.com/robotrocketscience/aelfrice/pull/302)). JSONL records under `<git-common-dir>/aelfrice/rebuild_logs/` capture prompt, retrieval candidates per lane, dedupe stats, pack-rate. Unblocks the operator-week of in-tree evidence collection that gates the rebuild-redesign calibration work in #289 / #291.
- **Posterior-ranking eval harness + heat-kernel composition wiring** ([#151](https://github.com/robotrocketscience/aelfrice/issues/151), [#306](https://github.com/robotrocketscience/aelfrice/pull/306), [#310](https://github.com/robotrocketscience/aelfrice/pull/310)). `benchmarks/posterior_ranking.py` measures MRR uplift + ECE; heat-kernel composition is wired through `retrieve_v2` as a log-additive term. Default-flip is still gated on the harness clearing thresholds against the v2.0 corpus — full lane-default flip lands at v1.7.0 (#154).
- **Deferred-feedback sweeper — implicit retrieval-driven posterior signal** ([#191](https://github.com/robotrocketscience/aelfrice/issues/191), [#256](https://github.com/robotrocketscience/aelfrice/pull/256)). String-overlap signal between retrieved beliefs and the host's continuation; emits `helped` / `noise` posterior events. Replaces the v1.5 explicit-only feedback path with a default-on implicit signal.
- **v2.0 corpus public scaffold + bench-gate harness** ([#307](https://github.com/robotrocketscience/aelfrice/issues/307), [#311](https://github.com/robotrocketscience/aelfrice/pull/311), [#319](https://github.com/robotrocketscience/aelfrice/issues/319), [#320](https://github.com/robotrocketscience/aelfrice/pull/320)). Empty per-module directories under `tests/corpus/v2_0/` plus the bench-gate harness in `tests/bench_gate/`. Autouse `bench_gated` marker skips when `AELFRICE_CORPUS_ROOT` is unset, so public CI stays green while the labeled corpus lives in the lab repo.
- **`replay_full_equality` probe — flip-readiness gate for #262** ([#262](https://github.com/robotrocketscience/aelfrice/issues/262), [#304](https://github.com/robotrocketscience/aelfrice/pull/304)). Walks the append-only `ingest_log` (#205), replays every row through `derive()`, asserts byte-equal equality against the live store. Sentinel for the v2.0 view-flip.
- **Onboard / scanner / MCP `session_id` propagation to inserted beliefs** ([#192](https://github.com/robotrocketscience/aelfrice/issues/192)). Phantom-prereqs T3 of the #190 session-tracking story.
- **Reachable-install detection + multi-install upgrade warning** ([#345](https://github.com/robotrocketscience/aelfrice/issues/345)). `aelf upgrade` enumerates every reachable install before upgrading, so users on multi-install machines see what they're about to update.

## Planned

### v1.2.x — search-tool hook (planned patch)

Pulled forward from v1.3.0 to validate the `PreToolUse` retrieval surface ahead of the bigger retrieval wave. Ships against the v1.0 retrieval pipeline + v1.1.0 per-project DB resolution; no dependency on entity-index, BFS, or LLM classifier work.

- **Search-tool `PreToolUse` hook** ([search_tool_hook.md](../design/search_tool_hook.md)). Fires before `Grep` and `Glob` tool calls, lifts the agent's search query out of `tool_input.pattern`, runs the same query against the per-project belief store, and emits results as `additionalContext` so the agent sees them *before* the tool runs. First retrieval-shaped hook on the agent's *own* tool intent (the v1.0.1 `UserPromptSubmit` hook covers user-initiated retrieval; this covers agent-initiated). Opt-in via `aelf setup --search-tool` at v1.2.x; matcher extension to other tools tracked at #155 for v1.5.

### v1.7.0 — graph signal wave + structural retrieval lane

The release where ranking moves beyond BM25 + L2.5 + BFS into graph-authority and full posterior-weighted territory. Originally targeted at v1.6; lifted to v1.7 because the v1.6 cycle absorbed the hook-hardening + observability + retention surface ahead of it. The eval harness (#151) and heat-kernel composition wiring (#310) shipped at v1.6 in default-OFF form so the math could land in front of the bake-off; v1.7 flips the lane defaults once #154's gate criteria pass.

- **Signed normalized Laplacian + offline eigenbasis (top-K=200) builder** ([#149](https://github.com/robotrocketscience/aelfrice/issues/149)). Offline-only build step; no runtime cost. Hard prerequisite for #150.
- **Heat kernel authority signal via precomputed eigenbasis** ([#150](https://github.com/robotrocketscience/aelfrice/issues/150)). +0.41 NDCG @ +7.8 ms p50 on a 50k-belief store — biggest single retrieval gain in the bake-off. Ships default-on; latency stays inside the v1.2.x search-tool hook's 50 ms median budget.
- **Plate FFT HRR primitives — port to public repo** ([#216](https://github.com/robotrocketscience/aelfrice/issues/216)). Hard prerequisite for the HRR structural-query lane (#152).
- **HRR structural-query lane (bind/probe over outgoing edges)** ([#152](https://github.com/robotrocketscience/aelfrice/issues/152)). A separate retrieval lane, not a projection — naive HRR projection into BM25 ranking was rejected at -0.10 NDCG (R9 in the bake-off). Bind/probe over outgoing edges is the structural-query path that survives. Persists `id_vec` per belief; `enable_hrr` config flag default-off until any belief has HRR vectors written.
- **`uri_baki` post-rank adjuster retest with relevance-aware locked set** ([#153](https://github.com/robotrocketscience/aelfrice/issues/153)). Named after Uri and Baki, the two Fire Aelfmaidens in Gene Wolfe's *The Wizard Knight* (2004) — bound attendants who operate after the main action to tilt outcomes for their bound knight. Locked-floor (Uri's protection), supersession demote (Baki's undermining), recency decay (the Aelfrice time-tilt). The pattern is publicly described as Google's "Twiddler" in Pandu Nayak's DOJ testimony (October 2023) and the May 2024 Content Warehouse API leak; aelfrice uses neutral naming to avoid trading on Google's term-of-art. A prior random-pinned synthetic regressed -0.05 NDCG due to a methodology bug (random pinning drowned signal). The relevance-aware retest uses `Belief.lock_state` and decides whether the lane is dead or just needed a fairer eval. Research item; ships only if the retest beats the BM25F + heat-kernel + posterior baseline.
- **Posterior-weighted ranking — full default-flip** ([#151](https://github.com/robotrocketscience/aelfrice/issues/151)). Harness + composition shipped at v1.6.0 in default-OFF form. v1.7 flips the lane defaults once the harness clears the MRR-uplift / ECE thresholds against the v2.0 corpus.
- **Benchmark-gate default-on flip** ([#154](https://github.com/robotrocketscience/aelfrice/issues/154)). Composition tracker shipped at v1.5.0 with the per-lane gate; v1.7 promotes the heat-kernel and posterior lanes from default-OFF to default-ON.

### v2.0.0 — feature parity and reproducibility

After v2.0.0, `benchmarks/` reproduces every published headline number on a fresh clone with `uv sync && aelf bench all`, within documented tolerance bands.

- ~~HRR vocabulary bridge~~ — **closed by the structural-query lane (#152, default-on as of v2.1)**. The lab campaign (`exp/hrr-vocabulary-bridge`) reframed "vocabulary bridge" as "typed-edge structural retrieval" and that mechanism shipped via `src/aelfrice/hrr_index.py`. See [feature-hrr-integration.md](../design/feature-hrr-integration.md). #433 closed; #536 (the parallel `vocab_bridge.py` query-rewrite module) removed.
- Type-aware compression — tokens-per-belief reductions on retrieved output.
- Intentional clustering — co-locating related beliefs for higher coherence on multi-fact queries.
- Correction-detection eval — five-codebase labeled fixture, scored by both the zero-LLM detector and the LLM-judge path.
- Posterior drives ranking, end to end. The 10-round MRR uplift eval and ECE calibration scorer ship with this release.
- Expanded surface: `wonder`, `reason`, `core`, `unlock`, `delete`, `confirm`, plus graph-metrics and document-linking.
- Reproducibility harness: `benchmarks/results/v2.0.0.json` is canonical; CI runs the academic suite nightly.

### v3.0.0 — completion + design cut (shipped 2026-05-13)

v3.0 closed the wonder-lifecycle wave, shipped HRR persistence with split-format on-disk migration, shipped type-aware compression at the A2 recall@k bench gate, shipped read-only federation, and ratified four v3-level design decisions. Replaced the prior v2.2 row, whose three referenced issues turned out to be stale (#197 WONTFIX; #193 evaluation shipped without hook successor; #194 was `ingest_turn(bulk=)`, also shipped). Per-entry detail: [CHANGELOG.md § 3.0.0](../CHANGELOG.md). Milestone tracker: [#608](https://github.com/robotrocketscience/aelfrice/issues/608).

Substrate completion (all shipped):

- **HRR persistence default-ON + split-format save/load** ([#553](https://github.com/robotrocketscience/aelfrice/issues/553)). `HRRStructIndex.save()` writes a per-store directory with `struct.npy` + `meta.npz`; legacy `.npz` bundles still load with a deprecation warning. Persistence defaults on; opt out via `[retrieval] hrr_persist = false` or `AELFRICE_HRR_PERSIST=0`. Ephemeral paths auto-disable.
- **Wonder lifecycle completion** ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella). Phantom promotion Surface A + Surface B ([#550](https://github.com/robotrocketscience/aelfrice/issues/550) + [#616](https://github.com/robotrocketscience/aelfrice/issues/616)) per the 2026-05-11 ratification (no count-trigger). Skill-layer subagent dispatch → `wonder_ingest` ([#552](https://github.com/robotrocketscience/aelfrice/issues/552)). Wonder/reason parity ([#645](https://github.com/robotrocketscience/aelfrice/issues/645)): Verdict/ImpasseKind classifiers, ConsequencePath fork-on-CONTRADICTS, VERDICT-driven dispatch + suggested-updates close-the-loop. `aelf wonder QUERY` positional defaults to the axes flow.
- **Type-aware compression A2 bench gate** ([#434](https://github.com/robotrocketscience/aelfrice/issues/434)). `run_compression_a2_uplift` driver landed; strict positive `mean_recall@k(use_type_aware_compression=ON) > OFF` gate in place. Rebuilder continuation-fidelity (A4) remains the flip-default gate for the next minor.
- **Eval-harness completion** ([#592](https://github.com/robotrocketscience/aelfrice/issues/592), [#600](https://github.com/robotrocketscience/aelfrice/issues/600), [#687](https://github.com/robotrocketscience/aelfrice/issues/687)). Host-agent replay path writes/joins per-run JSONL; opt-in LLM-judge stage scores open-ended turns at the operator's anchor tier; Cohen's-κ runner gates inter-judge agreement ≥ 0.70 plus hot-start fidelity ≥ 0.80. Synthetic `hot_start` fixture covers post-compact "where were we?" prompts.
- **Read-only federation** ([#650](https://github.com/robotrocketscience/aelfrice/issues/650), [#655](https://github.com/robotrocketscience/aelfrice/issues/655), [#688](https://github.com/robotrocketscience/aelfrice/issues/688), [#689](https://github.com/robotrocketscience/aelfrice/issues/689), [#690](https://github.com/robotrocketscience/aelfrice/issues/690), [#713](https://github.com/robotrocketscience/aelfrice/issues/713)). `scope` field on beliefs (`project` / `global` / `shared:<name>`); peer DB FTS5 + BFS visible through `knowledge_deps.json`; `aelf promote --to-scope` flips visibility; mutations against foreign belief IDs raise `ForeignBeliefError`. `aelf reason` annotates peer hops with `[scope:<name>]`.
- **`query_strategy` default flip** ([#718](https://github.com/robotrocketscience/aelfrice/issues/718)). `DEFAULT_STRATEGY` flipped `legacy-bm25` → `stack-r1-r3` on bench evidence (+0.2851 absolute NDCG@k, +94.8%, p99 latency 13% of the 5 ms budget). `legacy-bm25` remains callable via explicit kwarg until PR-4 removes the code path one minor release out.

Design ratifications (all closed, doc-only follow-through):

- **NL-relatedness philosophy** ([#605](https://github.com/robotrocketscience/aelfrice/issues/605), ratified 2026-05-10). Option 1 — stay deterministic, narrow surface. Dedup, contradiction, and relatedness gates live in the consuming agent, not aelfrice. Memo: [`docs/design/v3_relatedness_philosophy.md`](../design/v3_relatedness_philosophy.md).
- **Sentiment-feedback hook production wire-up** ([#606](https://github.com/robotrocketscience/aelfrice/issues/606), ratified 2026-05-10). `UserPromptSubmit` lane, default-off opt-in, most-recent-window decay policy. Shipped behind `[feedback] sentiment_from_prose = true` / `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1`.
- **Multimodel scope** ([#607](https://github.com/robotrocketscience/aelfrice/issues/607), deferred 2026-05-11). No maintainer validation path for third-party LLM CLIs; the wonder-dispatch lane (#542/#551) covers the in-tree story.
- **Federation write model** ([#661](https://github.com/robotrocketscience/aelfrice/issues/661), ratified 2026-05-11). Option B — read-only federation. Per-project DB is sole writer; peers open foreign DBs read-only and UNION FTS5 results. Mutation tools reject foreign belief IDs at the API surface. CRDT-primitives sub-issues (#651-#654) closed WONTFIX. See [`docs/design/federation-primitives.md`](design/federation-primitives.md) §1 for the forward-compat version-vector substrate; §2-§5 are flagged as deferred multi-writer extension.

### v3.0.1 — install-surface collapse + default-on agent-side retrieval (shipped 2026-05-13)

Patch release on top of v3.0.0. No public API change; the user-facing surface narrows on the install side and broadens on the retrieval side. Per-entry detail: [CHANGELOG.md § 3.0.1](../CHANGELOG.md).

- **Install / upgrade surface collapsed to `uv tool` only** ([#730](https://github.com/robotrocketscience/aelfrice/issues/730)). `aelf upgrade-cmd` (and `/aelf:upgrade`) emit a single in-place form (`uv tool upgrade aelfrice`) for uv-managed installs and a migration chain (`pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent) for any other installer. `UpgradeAdvice.context` collapses `uv_tool` / `pipx` / `venv` / `system` → `uv_tool` / `non_uv`. README and `docs/user/INSTALL.md` rewritten around `uv tool install`; the pipx/venv/system helpers in `lifecycle.py` remain internal but no longer surface as supported channels.
- **Auto-migrate non-uv installs on first 3.0.1 setup** ([#733](https://github.com/robotrocketscience/aelfrice/issues/733), follow-up [#774](https://github.com/robotrocketscience/aelfrice/issues/774) for the uv-not-found runnable one-liner). `aelf setup` runs `lifecycle.maybe_migrate_to_uv()` before hook reconciliation; on a pipx/pip install with `uv` on `$PATH`, runs `uv tool install --force aelfrice` (120s timeout) once per machine behind a `~/.aelfrice/migrated-to-uv` sentinel. The fresh uv-tool shim overwrites the existing `~/.local/bin/aelf` so subsequent invocations resolve through the uv-tool venv. A pipx-only user without `uv` now sees a copy-pasteable installer hint (`curl -LsSf https://astral.sh/uv/install.sh | sh` and `brew install uv` on macOS), not just a docs URL.
- **`search-tool` and `search-tool-bash` hooks default-on** ([#738](https://github.com/robotrocketscience/aelfrice/issues/738)). `aelf setup` (no flags) wires both `PreToolUse:Grep|Glob` and `PreToolUse:Bash` (grep/rg/find/fd/ack) hooks. The README's "four parallel retrieval lanes" claim was previously aspirational on agent-initiated search; this fixes that. Flags follow the `--X / --no-X` BooleanOptionalAction convention; opt-out persists across upgrades via `~/.aelfrice/opt-out-hooks.json`. Worst-case latency at v3.0-typical 10k-belief corpus: ~22ms / turn for 5 Grep fires + ~12ms / turn for 3 Bash-search fires (Bash hook capped at 3 firings per turn).
- **Cross-fire injection dedup ring** ([#740](https://github.com/robotrocketscience/aelfrice/issues/740)). New per-session rolling-window FIFO ring at `<git-common-dir>/aelfrice/session_injected_ids.json`. UPS hook appends per-turn hit IDs after its `<aelfrice-memory>` block; `PreToolUse` search hook filters `retrieve()` results against the ring before emitting `<aelfrice-search>`. Three emit shapes: (1) all-recent → pointer block `note="answer already in prompt context"`; (2) mixed → render new beliefs with a trailing "(N more matching belief(s) already in prompt context from earlier this session)"; (3) all-new → original block unchanged. Locked beliefs pass through as `new` regardless of ring state. New session_id wipes the ring; `fcntl.LOCK_EX` serializes near-simultaneous UPS + PreToolUse read-modify-write. Cap defaults to 200 IDs (`AELFRICE_INJECTION_RING_MAX` env override). `aelf doctor` surfaces `injection ring: N/MAX ids (evicted K this session)`.
- **Transitive `authlib` 1.7.0 → 1.7.2 for CVE-2026-44681** ([GHSA-r95x-qfjj-fjj2](https://github.com/advisories/GHSA-r95x-qfjj-fjj2)). Medium-severity (CVSS 6.1) open-redirect in Authlib's OIDC server flows (`OpenIDImplicitGrant` / `OpenIDHybridGrant`). aelfrice's exposure: effectively zero — `authlib` is transitive via `fastmcp` (the optional `[mcp]` extra) and the aelfrice MCP server implements no OIDC authorization endpoint. Lockfile-only change.

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
| LLM onboard classifier (default-on, host-driven) | v1.5.1 |
| Posterior-weighted ranking | v1.3.0 (partial) / v1.6.0 (eval harness + composition wiring, default-OFF) / v1.7.0 (default-flip) |
| BM25F anchor-text + vectorized BM25 | v1.5.0 |
| Signed Laplacian + heat-kernel authority | v1.7.0 |
| HRR structural-query lane | v1.7.0 (default-on as of v2.1; closes vocabulary-gap-recovery claim per the lab campaign R5 reframe) |
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
- **Cross-project shared scopes** via SQLite ATTACH. Subsumed by the Multi-project query non-goal in [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md). Named here so the research-line term ("shared scopes") doesn't read as an oversight.
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
- **Brain-graph sync or multi-writer federation.** v3.0 ships *read-only* federation (#650 / #655 / #661): a project can declare peer DBs in `knowledge_deps.json` and surface their `global` / `shared:<name>` beliefs in FTS5 + BFS, but every per-project DB stays the sole writer for its own rows. Mutations against foreign belief IDs raise `ForeignBeliefError`. Multi-writer federation with CRDT primitives (#651-#654) was filed and closed WONTFIX in the same cut. See [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md) for the boundary.

## Validation

Every release with a behavioural claim ships with a benchmark or test that demonstrates it. v1.0's central claim is that `apply_feedback` updates posteriors mathematically — proven by tests and the synthetic harness. v1.0 makes *no* claim that BM25 ranking moves under feedback, because it doesn't yet. v1.3 is the release where that claim becomes testable; v2.0 is where the published MRR uplift is reproduced or retracted.

That is what "early access" means: the surface is stable, the central feedback-into-ranking claim is a future deliverable, and the benchmark harness ships now so the eventual delivery is auditable.
