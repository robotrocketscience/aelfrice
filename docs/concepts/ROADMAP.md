# Roadmap

This document holds the release history and the planned design cuts. As of
v3.0, the parity work from v1.0 to v2.0 is complete, and v3.0 has shipped. The
rows for each version are below.

Per-issue tracking: [LIMITATIONS.md](../user/LIMITATIONS.md). Release log: [CHANGELOG.md](../../CHANGELOG.md).

## Origin

aelfrice is a rebuild of an earlier research line. That research line studied Bayesian and graph-backed memory for AI coding agents. The research codebase explored these capabilities:

- full-text search version 5 (FTS5) retrieval;
- vocabulary bridging;
- breadth-first search (BFS) multi-hop traversal;
- entity-indexed retrieval;
- type-aware compression;
- correction detection;
- a Model Context Protocol (MCP) surface with several tools.

The research line has results against MAB, LoCoMo, LongMemEval, StructMemEval, and AMA-Bench.

v1.0 shipped the foundation: the SQLite store, Beta-Bernoulli scoring, Best Matching 25 (BM25) retrieval, the CLI, MCP, the host hook wiring, and the synthetic benchmark harness. The v1.x line recovered the remaining features one at a time. v2.0 reached feature parity with the research line. v2.0 also added the scaffolding for the reproducibility harness. v3.0 added the wonder lifecycle, wonder/reason parity, read-only federation, and the completion of the eval harness.

v4.0 converged the production hook path onto the unified `retrieve_v2` ranker. That change graduated the staged retrieval lanes onto live hosts. v4.1 completed the work for the Codex host: it shipped the slash-command bundle as `$aelf-*` agent skills. v4.2 added conditional, keyword-triggered belief categories and a phantom promotion-opportunity detector. v4.2 also restructured the retrieval and ingest hot paths for performance, and kept the outputs byte-identical. v4.3 made the tool usable on native Windows, and completed the Codex host surface. v4.3 also removed the MCP server, which had been broken for a long time. The current line is **v4.3.0**.

This project is a rebuild, not a port. The rebuild fixed the structural issues that survived the research line. It fixed those issues at the foundation layer. A test or a benchmark supports every behavioural claim. Items without such support have a transparent issue trail.

## Versions at a glance

| Version | Status | Theme |
|---|---|---|
| v1.0.x | shipped | core memory, CLI, MCP, hook wiring, install routing, contradiction tie-breaker |
| **v1.1.0** | shipped | per-project DBs, `aelf migrate`, `edges`→`threads` rename, `aelf health` rewrite |
| **v1.2.0** | shipped | auto-capture pipeline (transcript-ingest, commit-ingest, SessionStart), `agent_inferred → user_validated` promotion, triple extractor, `--batch` JSONL ingest, CLI consolidation, `INEDIBLE` per-file opt-out |
| **v1.2.x** | shipped (matcher widened v1.5.0; default-on v3.0.1 #738) | search-tool `PreToolUse` hook — memory-first context on Grep/Glob |
| **v1.3.0** | shipped | retrieval wave — entity index (L2.5), BFS multi-hop (L3), LLM-Haiku onboard classifier (opt-in), partial posterior-weighted ranking |
| **v1.4.0** | shipped | context rebuilder — PreCompact retrieval-curated continuation (augment mode); manual + threshold trigger; continuation-fidelity scorer (exact-match) |
| **v1.5.0** | shipped | retrieval plumbing — composition plumbing + telemetry, BM25F anchor text, search-tool Bash matcher, v3 federation version-vector schema, dynamic-trigger re-park |
| **v1.5.1** | shipped | corroboration tracking sibling table (#190); default-on host-driven LLM onboard classifier (#238) |
| **v1.6.0** | shipped | hardening + observability — hook-hardening framing-tag contract + per-turn audit log, `aelf tail`, belief retention class, rebuild diagnostic log, posterior-ranking eval harness + heat-kernel composition (default-flip gated), deferred-feedback sweeper, v2.0 corpus public scaffold + bench-gate, `replay_full_equality` probe, `session_id` propagation, reachable-install detection |
| **v1.7.0** | shipped | graph signal wave + structural retrieval lane — signed Laplacian + eigenbasis (#149), heat kernel authority (#150), Plate FFT primitives for holographic reduced representation (HRR) (#216), HRR bind/probe (#152), `uri_baki` post-rank adjuster retest (#153). Heat-kernel and HRR-structural shipped opt-in. The benchmark-gate default-on flip (#154) moved to v2.1.0. |
| **v2.0.0** | shipped | feature parity with the research line + reproducibility-harness scaffolding (#437); wonder lifecycle + dispatch surface; sentiment-feedback module (hook integration pending); dedup read-path (audit-only) |
| **v2.1.0** | shipped | reproducibility-harness gate cleared 11/11 (#437); `use_heat_kernel` + `use_hrr_structural` defaults flipped on (#154); HRR `dim` default 2048→512 (#538); default-on transcript / commit / session-start hooks (#529); query-strategy uplift bench gate (#291); vocab-bridge bench gate (#433) |
| **v3.0.0** | shipped 2026-05-13 | wonder lifecycle complete (#542 umbrella + #547/#550/#552); wonder/reason parity #645 (Verdict/ImpasseKind, ConsequencePath fork on CONTRADICTS, VERDICT-driven dispatch + close-the-loop suggested-updates); HRR persistence default-ON + split-format save/load (#553); type-aware compression A2 bench gate (#434); eval-harness host-agent replay + LLM-judge stage + Cohen's-κ runner (#592, #600, #687); read-only federation — `scope` field + peer DB FTS5/BFS + foreign-id rejection (#650, #655, #688, #690, #713); `query_strategy` default flipped legacy-bm25 → stack-r1-r3 (#718); phantom-promotion Surface A + Surface B (#550, #616); sentiment-feedback UPS hook (#606); self-installing hook manifest (#623); merge-train label-driven serialized merger (#602). Ratified design decisions: PHILOSOPHY stays deterministic (#605); multimodel deferred (#607); federation read-only (#661). Milestone tracker: [#608](https://github.com/robotrocketscience/aelfrice/issues/608). |
| **v3.0.1** | shipped 2026-05-13 | install-surface collapse: pipx/pip/venv channels removed, `uv tool install` is the single supported path (#730); auto-migrate non-uv installs to `uv tool` on first 3.0.1 `aelf setup` (#733, follow-up #774 for the uv-not-found one-liner); `search-tool` + `search-tool-bash` `PreToolUse` hooks default-on so the agent's own Grep/Glob/Bash-search calls go through the belief store first (#738); cross-fire injection dedup ring, so that consecutive UPS + PreToolUse fires do not re-surface the same belief in the same turn (#740); transitive `authlib` 1.7.0 → 1.7.2 for CVE-2026-44681 (zero-exposure surface — `[mcp]` extra, aelfrice has no OIDC authorization endpoint). |
| **v3.1.0** | shipped | `DEFAULT_TRIGGER_MODE` for the PreCompact rebuilder flipped from `manual` → `threshold` (#746). (The contradict-edge auto-demote removal below merged after the v3.1.0 tag and first shipped in v3.2.0.) Auto-demote machinery removed — `Belief.demotion_pressure` column dropped, `apply_feedback(propagate=)` kwarg dropped, `FeedbackResult.pressured_locks` / `.demoted_locks` fields gone (#814 / PR #820). Lock correction now goes through `aelf lock` overwrite per PHILOSOPHY #605. |
| **v3.2.0** | shipped | (see [CHANGELOG/v3.md § 3.2.0](../../CHANGELOG/v3.md) for the per-entry detail) |
| **v3.3.0** | shipped 2026-05-21 | `user_transcript` origin tier added (#888 — distinguishes user-stated content captured from transcript ingest from `user_stated` explicit-lock content and `agent_inferred` derivation); session-start recent-work block (#887 — surfaces current branch, recent commits, referenced issue numbers); P3 cadence policy + turn-density scoring (#876); aelf graph CLI for DOT/JSON subgraph emission (#629); aelf scope-out for federation denylist control (#856); aelf label CLI for relevance-corpus labelling (#859); aelf export-obsidian for one-way Obsidian rendering (#630). |
| **v3.4.0** | merged 2026-05-26; never tagged standalone — published with the v3.5.0 cut (see [CHANGELOG/v3.md § 3.4.0](../../CHANGELOG/v3.md)) | cadence-policy completion + conversation-aware retrieval — `p3_substantive` policy live, 4-policy shadow capture, offline cadence replay bench (#876, #920); UPS retrieval conversation-aware (#909 — folds recent turns into BM25 query) |
| **v3.5.0** | shipped 2026-06-04 | visibility/observability wave — per-project feed log + `aelf feed` (#931), SessionStart recap (#934), contradiction marker on `aelf search` (#938), `aelf speculative` (#937), `aelf stale` (#933), `aelf audit-claude-memory` (#935), `aelf review` (#936), duplicate-issue PreToolUse guard (#941), statusline counts (#932) |
| **v3.5.1** | shipped 2026-06-10 | patch — scanner skips `.claude/` on onboard (#955); INEDIBLE marker propagation to ingest-transcript paths (#958); v3.5.0 docs-vs-code reconciliation wave (#952, #957, #964) |
| **v3.6.0** | shipped 2026-06-19 | `project_context` provenance — repo-identity auto-stamp on write + idempotent backfill + migrate provenance preservation (#970, architecture decision record (ADR) 0003; decision 4 ratified keep-opt-in #973); transcript-logger consecutive-duplicate turn guard, so that duplicated hook registrations do not inflate turn density (#968); `hook_audit` sink moved out of `hook.py`, off the heavy import path for retrieval (#968) |
| **v3.7.0** | shipped 2026-06-23 | graph-substrate + phantom-lifecycle wave — `CONTRADICTS` semantic-edge substrate at ingest, default-off + incremental per-turn (#988, #1000); `claude-memory` write-through mirror into the belief graph (#985); phantom lifecycle made observable + self-maintaining — trigger-driven generation, opt-in SessionStart auto-GC, status/stats surface, soft-delete retrieval hygiene (#980); default-off HRR vocabulary-bridge expansion lane wired + ablated recall-neutral (#981); denser `SUPPORTS`/`SUPERSEDES` edges reverted per `CONTRADICTS`-only ratification (#998 A4, revert #999) |
| **v3.8.0** | shipped 2026-06-30 | belief-hygiene + doctor-observability wave — layered locks (#1016-B: `lock_tier` frozen/reference, reference-tier bounded manifest injection across all four injection paths, lock-dedup hygiene warning); relevance floor, so that a large lock set does not reduce the query-relevant results to zero (#1014/#1015); new `aelf doctor` checks (`lock_budget`, `ingest_gap`) + `--backfill-ingest` + `--prune-noise` GC (#1011, #1029); pollution-recovery benchmark (#1011); worktree `uv run aelf` no longer downgrades installed hooks (#1044) |
| **v4.0.0** | shipped 2026-07-07 | production-retrieval convergence — `retrieve()` becomes a thin adapter over `retrieve_v2` and four staged lanes graduate onto the live hook path (temporal-spine #1064, entity-persist demotion #1096, intentional clustering #436, HRR structural-query #152), two staged lanes held off (origin-tiebreak #1013, HRR-expand #1001) (#1107); belief-curation + junk-percolation wave — `aelf introspect` / `retire` / `restore`, exposure-is-not-endorsement (#1086), entity-persistence sink (#1096), hedged-float drop + stranded-noise prune (#1081); provenance-aware capture — claude-memory reconcile default-on + `transcript` as a full source type + origin tie-break default-off (#1089); dispatched-subagent memory inheritance (#1068); Codex host support for setup/doctor/unsetup (#1052–#1055); wide-retrieval knobs (#1045); valence propagation live (#1058) |
| **v4.1.0** | shipped 2026-07-09 | Codex host completion — `aelf setup --host codex` installs the full slash-command surface as `$aelf-*` agent skills under `~/.agents/skills/`, generated on install from the same `slash_commands/*.md` bundle so `/aelf:foo` and `$aelf-foo` never drift; idempotent + orphan-pruning behind an `AELFRICE-CODEX-SKILL` marker; `aelf doctor --host codex` reports the skill count, `aelf unsetup --host codex` removes skills + hooks together; hooks-only install via `--no-codex-skills` (see [CHANGELOG § 4.1.0](../../CHANGELOG/v4.md)) |
| **v4.2.0** | shipped 2026-07-21 | conditional-memory + hot-path wave — keyword-triggered belief categories rerank the `<aelfrice-memory>` block by whichever category the prompt activates, the conditional complement to a static `CLAUDE.md`/`AGENTS.md` (advisory, additive data definition language (DDL), no embeddings; new `aelf category` verb + `/aelf:category`; default-off) (#1126); phantom promotion-opportunity detector surfaces a corroborated phantom for the explicit #229 promotion act, closing the #1125-census gap — note-not-write, default-off (#1132); hot-path performance overhaul restructures the retrieval + ingest hot paths (persistent BM25F sidecar, single store-open per prompt, memoized `.aelfrice.toml` parse, marker-gated origin backfill, hot-path indexes) with byte-identical retrieval outputs (#1135), plus the correctness-smell follow-ups it surfaced — `now_ts` clock-seam pin (#1143), `RetrievalCache` cache-hit exposure exemption (#1144 — the class was later removed in #1418; this line records what shipped in v4.2.0), telemetry-ring inter-process lock + fsync-drop (#1145); codex host-management skills steer to `--host codex` (#1136); corroboration-driven retention promotion no longer reaches phantoms (#1132) (see [CHANGELOG § 4.2.0](../../CHANGELOG/v4.md)) |
| **v4.3.0** | shipped 2026-08-13 | native-Windows + Codex wave, and the largest v4 cut (966 commits) — v4.2.0 could not start on native Windows at all (module-scope `import fcntl` fatal at CLI entry, #1329), and the fixes behind it sat unreleased for three weeks; with that unblocked the rest of the platform surface is fixed here: console launchers resolved through `PATHEXT` instead of by bare name and `aelf setup` no longer pruning correctly-installed hooks (#1412), every subprocess and `Path` text boundary pinned to utf-8 rather than the ANSI code page (#1441), the hook stdin boundary read as bytes so a non-ascii turn is no longer dropped without a message and a piped archive password no longer derives an unreproducible key (#1426), and `setup --host codex` no longer overwriting a user's foreign Codex handlers (#1428). **Removed:** the `aelf mcp` subcommand, `mcp_server.py` and the `[mcp]` extra (#1422) — the server had not started on any version of its declared `fastmcp` range since 2026-05-08, so no working install can experience the break; `aelf migrate --remove-mcp-config` clears the leftover config. Also: an off-switch for the injected memory block (#1359), lock windows via `aelf lock --for/--until` (#1314), and the CI-integrity run after the finding that no on-push workflow had fired in 541 commits (#1423) (see [CHANGELOG § 4.3.0](../../CHANGELOG/v4.md)) |

## What shipped

### v1.0.x — surface

The stable core has these parts:

- the SQLite + FTS5 store;
- Beta-Bernoulli scoring;
- L0+L1 retrieval at a 2,400-token budget;
- `apply_feedback` with an audit log;
- the onboarding scanner (filesystem + git log + Python abstract syntax tree (AST));
- the CLI;
- the host hook wiring;
- the synthetic benchmark harness;
- the contradiction tie-breaker (`aelf resolve`);
- per-project install routing (`aelf doctor`);
- the release-docs CI gate.

### v1.1.0 — project identity

- Per-project DB resolution. Inside any git work-tree, the store is `<git-common-dir>/aelfrice/memory.db`. Outside a work-tree, the store falls back to `~/.aelfrice/memory.db`.
- `aelf migrate` ports beliefs from the legacy global store. The command is read-only on the source.
- Worktree concurrency is tested under a write-ahead log (WAL) and `busy_timeout=5000`.
- `aelf health` is rewritten as the structural auditor (orphan threads, FTS5 sync, locked contradictions). The v1.0 regime classifier stays available as `aelf regime`. `aelf status` is an alias of `aelf health`.
- The user-facing name `edges` becomes `threads`. The internal schema does not change. The deprecation window covered both keys.
- Onboard git-recency weighting. `belief.created_at` is the most recent commit of the source file. Decay therefore penalises stale branches.
- The `agent_inferred → user_validated` promotion path is designed here. v1.2 implements that path.

### v1.2.0 — auto-capture and triple extraction

- **Commit-ingest hook.** `PostToolUse:Bash` ingests every successful `git commit` through the triple extractor. The hook uses a deterministic session id that it derives from git. It populates `Edge.anchor_text`, `Belief.session_id`, and `DERIVED_FROM` edges densely. `aelf setup --commit-ingest` wires the hook.
- **Transcript-ingest hook.** A four-event hook captures every conversation turn to `<git-common-dir>/aelfrice/transcripts/turns.jsonl`. `PreCompact` rotates the file and ingests it. The hook closes the harness-conflict gap. Before this release, that gap kept new beliefs from normal sessions out of the MCP.
- **Triple extractor.** The extractor is pure regex over six relation families (`SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`, `RELATES_TO`, `DERIVED_FROM`). Every caller that ingests prose can reuse the extractor.
- **`agent_inferred → user_validated` promotion.** A new `Belief.origin` column holds seven tier values. `aelf validate <id>` graduates onboard-derived beliefs. The command requires no lock.
- **SessionStart hook.** The hook injects locked beliefs as `<aelfrice-baseline>` one time in each session.
- **Ingest enrichment schema.** The schema adds `DERIVED_FROM` edges, `anchor_text`, `session_id`, and a real `sessions` table. The schema is forward-compatible with v1.0 stores.
- **`aelf ingest-transcript --batch DIR [--since DATE]`.** Backfill historical Claude Code session JSONLs into the local belief graph. Auto-detects transcript-logger and Claude Code session formats.
- **CLI consolidation + `INEDIBLE` per-file opt-out.** This work prepares v1.3. It tightens the surface. It adds a per-file privacy marker.
- **Harness integration guide.** Three operational modes for coexisting with Claude Code's auto-memory directive.

### v1.3.0 — retrieval wave

This release moves retrieval beyond BM25 alone.

- **Entity-index retrieval.** The L2.5 entity index ships, including the regex extraction patterns. Specification: [entity_index.md](../design/entity_index.md).
- **BFS multi-hop graph traversal.** The lane runs edge-type-weighted graph walks over the FTS5 hits. The depth is bounded. The budget is bounded. Specification: [bfs_multihop.md](../design/bfs_multihop.md).
- **LLM-classification onboard path.** A classifier backed by a small LLM is an opt-in alternative to the regex classifier. The classifier is default-off. To opt in, run `aelf onboard --llm-classify`, or set `[onboard.llm].enabled = true` in `.aelfrice.toml`. The boundary policy and the prompt template are in [llm_classifier.md](../design/llm_classifier.md). PRIVACY: this call is the first outbound call in the install path that transmits user content. See [PRIVACY § Onboard-time outbound call](../user/PRIVACY.md#onboard-time-outbound-call).
- **Posterior-weighted ranking (partial).** Retrieval scoring adds `α / (α+β)` to BM25. The addition is log-additive at weight 0.5. For the v1.3 contract, see [bayesian_ranking.md](../design/bayesian_ranking.md). The eval harness for mean reciprocal rank (MRR) uplift and for expected calibration error (ECE) calibration shipped at v1.6.0 (#151, #306). The heat-kernel composition wiring (#310) shipped at the same time. Both shipped default-OFF. The default-flip lands at v1.7.0, once the harness clears the thresholds against the v2.0 corpus. The canonical cut at v2.0.0 re-runs the harness.

### v1.4.0 — context rebuilder

This release makes long-running sessions cheaper. The change leaves no visible seam.

- **PreCompact-driven rebuild.** When the harness signals an approaching context limit, an aelfrice hook queries the brain graph for the highest-value beliefs against the session tail. The hook emits those beliefs as `additionalContext`. The order is locked beliefs first, then session-scoped beliefs, then BM25 hits and posterior-weighted hits. The hook packs the beliefs to a configurable token budget.
- **Augment mode.** The hook augments the compaction of the harness. Both summaries appear in the new context. Replace mode is parked for v2.x.
- **Trigger modes** ([#141](https://github.com/robotrocketscience/aelfrice/issues/141)). The manual mode (`/aelf:rebuild`) and the threshold mode shipped at v1.4. The dynamic mode is parked to v1.5. The v1.4 ship-gate investigation ran in `benchmarks/context_rebuilder/dynamic_probe.py`. At the same token cost or a lower one, the dynamic mode did not produce a fidelity uplift of ≥ 5% absolute over the threshold mode on the synthetic fixture. #188 tracks the revisit. The default threshold fraction (0.6) comes from the eval-harness calibration in `benchmarks/context-rebuilder/calibration_v1_4_0.json`. Manual is the v1.4 ship default. Threshold is opt-in until production telemetry arrives.
- **Continuation-fidelity eval.** `benchmarks/context-rebuilder/` replays fixture transcripts. It forces a midpoint clear. It runs the rebuilder. It then measures continuation fidelity against the full-replay baseline. [eval_fixture_policy.md](../design/eval_fixture_policy.md) decides the fixture corpus policy. The synthetic public corpus serves CI and the headline number. The captured corpus stays lab-side, for offline calibration only.

Hard prerequisites: the v1.2 transcript-ingest and the v1.2 `session_id` schema. The alpha shipped in v1.2.0a0.

### v1.5.0 — retrieval plumbing

The composition gate comes first. Cheap retrieval improvements come after it. This minor release adds no new ranking math.

- **Pipeline composition tracker — unified `retrieve()` with feature-flag gate** ([#154](https://github.com/robotrocketscience/aelfrice/issues/154)). There is one entry point. Every retrieval feature is behind a config flag. Telemetry is per lane. This tracker is a prerequisite for the v1.7 graph wave: `retrieve()` must be the only path before the heat kernel and posterior-full can ship safely behind defaults.
- **Augmented BM25F (incoming-edge anchor text) + vectorized BM25 sparse matvec** ([#148](https://github.com/robotrocketscience/aelfrice/issues/148)). The lane gives +0.06 normalized discounted cumulative gain (NDCG) at +0 ms against BM25 in the component bake-off. The project adopted the lane because the runtime cost is free.
- **Search-tool hook — extend matcher beyond `Grep|Glob`** ([#155](https://github.com/robotrocketscience/aelfrice/issues/155)). This change widens the `PreToolUse` matcher list, because telemetry from v1.2.x confirmed the latency budget.
- **v1.4 dynamic-trigger revisit** ([#188](https://github.com/robotrocketscience/aelfrice/issues/188)). The dynamic mode was parked at the v1.4 ship-gate. That mode got a second eval pass on captured-corpus calibration data. The mode still did not meet the bar, so it is parked again.

### v1.5.1 — corroboration tracking + default-on host-driven onboard

- **Belief corroboration tracking — sibling table + ingest recorder** ([#190](https://github.com/robotrocketscience/aelfrice/issues/190)). The new `belief_corroborations` table records each re-ingest of identical content. The table does not disturb the existing dedup contract. This work is phantom-prereqs T1 of the #190 session-tracking story. T2 is the #191 sweeper. T3 is the #192 session_id propagation. Both T2 and T3 ship at v1.6.0.
- **Default-on host-driven LLM onboard classifier — no API key required** ([#238](https://github.com/robotrocketscience/aelfrice/issues/238)). The default of `[onboard.llm].enabled` flips `False → True`. The classifier runs through the host model's own Task tool, against the smallest model in the host's stack. The quality matches the v1.3.0 LLM-classifier ceiling. The cost is less than one percent of a typical weekly host-plan allowance. The direct-API path (`aelf onboard --llm-classify`) remains the fallback for a user with an API key.

### v1.6.0 — hardening, observability, retention

This release is a consolidation release, and not the graph-signal wave that the plan named. The ranking math (#149 / #150 / #216) moved to v1.7, so that it lands with the default-on flip of #154. v1.6 instead absorbed two other bodies of work. The first is the security-hardening surface that #280 surfaced. The second is the observability and bench-gate scaffolding that the rebuild redesign (#288 / #289 / #291) needs.

- **Hook-hardening Phase 1 — framing-tag contract + content escape for memory blocks** ([#280](https://github.com/robotrocketscience/aelfrice/issues/280), [#297](https://github.com/robotrocketscience/aelfrice/pull/297)). This work closes the prompt-injection surface. In that surface, ingested belief content could forge or close the `<aelfrice-memory>` framing tag.
- **Hook-hardening mitigation 3 — per-turn audit log** ([#280](https://github.com/robotrocketscience/aelfrice/issues/280), [#314](https://github.com/robotrocketscience/aelfrice/pull/314)). `<git-common-dir>/aelfrice/hook_audit.jsonl` records the exact rendered hook block on every fire. The file has a 10 MB cap. Rotation uses a single slot. The write is fail-soft.
- **`aelf tail` — live observability for hook injections** ([#321](https://github.com/robotrocketscience/aelfrice/issues/321), [#322](https://github.com/robotrocketscience/aelfrice/pull/322)). The command is a pretty-printer over the audit log, in the style of `tail -f`. The audit record itself gains the fields `beliefs[]`, `latency_ms`, and `tokens`.
- **Belief retention class + per-source aging policy** ([#290](https://github.com/robotrocketscience/aelfrice/issues/290)). A new schema column lands on `beliefs`. The per-ingest-source defaults are wired into `derive()` and into the scanner. The promotion path is `aelf doctor --promote-retention`. This work is the foundation layer for v2.0 aging and pruning. There is no automatic retention-driven eviction yet.
- **Rebuild diagnostic log — phase-1a write + phase-1c audit script** ([#288](https://github.com/robotrocketscience/aelfrice/issues/288), [#302](https://github.com/robotrocketscience/aelfrice/pull/302)). JSONL records under `<git-common-dir>/aelfrice/rebuild_logs/` capture the prompt, the retrieval candidates for each lane, the dedupe statistics, and the pack-rate. These records unblock the operator-week of in-tree evidence collection. That collection gates the rebuild-redesign calibration work in #289 / #291.
- **Posterior-ranking eval harness + heat-kernel composition wiring** ([#151](https://github.com/robotrocketscience/aelfrice/issues/151), [#306](https://github.com/robotrocketscience/aelfrice/pull/306), [#310](https://github.com/robotrocketscience/aelfrice/pull/310)). `benchmarks/posterior_ranking.py` measures the MRR uplift and the ECE. The heat-kernel composition is wired through `retrieve_v2` as a log-additive term. The default-flip still waits on the harness clearing the thresholds against the v2.0 corpus. The full lane-default flip lands at v1.7.0 (#154).
- **Deferred-feedback sweeper — implicit retrieval-driven posterior signal** ([#191](https://github.com/robotrocketscience/aelfrice/issues/191), [#256](https://github.com/robotrocketscience/aelfrice/pull/256)). The sweeper measures the string overlap between the retrieved beliefs and the host's continuation. It emits `helped` and `noise` posterior events. It replaces the explicit-only feedback path of v1.5 with an implicit signal. **Superseded as a default at #1086 (v4.0):** automatic retrieval exposure is now audit-only, because a retrieval is exposure and not endorsement. The sweeper therefore remains only as the manual `aelf sweep-feedback` path. Enqueue-on-retrieve stays default-on, but no consumer runs automatically.
- **v2.0 corpus public scaffold + bench-gate harness** ([#307](https://github.com/robotrocketscience/aelfrice/issues/307), [#311](https://github.com/robotrocketscience/aelfrice/pull/311), [#319](https://github.com/robotrocketscience/aelfrice/issues/319), [#320](https://github.com/robotrocketscience/aelfrice/pull/320)). Empty per-module directories land under `tests/corpus/v2_0/`. The bench-gate harness lands in `tests/bench_gate/`. The autouse `bench_gated` marker skips when `AELFRICE_CORPUS_ROOT` is unset. Public CI therefore stays green while the labeled corpus lives in the lab repo.
- **`replay_full_equality` probe — flip-readiness gate for #262** ([#262](https://github.com/robotrocketscience/aelfrice/issues/262), [#304](https://github.com/robotrocketscience/aelfrice/pull/304)). The probe walks the append-only `ingest_log` (#205). It replays every row through `derive()`. It asserts byte-equal equality against the live store. The probe is the sentinel for the v2.0 view-flip.
- **Onboard / scanner / MCP `session_id` propagation to inserted beliefs** ([#192](https://github.com/robotrocketscience/aelfrice/issues/192)). This work is phantom-prereqs T3 of the #190 session-tracking story.
- **Reachable-install detection + multi-install upgrade warning** ([#345](https://github.com/robotrocketscience/aelfrice/issues/345)). `aelf upgrade` enumerates every reachable install before it upgrades. A user on a machine with several installs therefore sees what the command will update.

### v1.7.0 — graph signal wave + structural retrieval lane (shipped)

This release moved ranking beyond BM25, L2.5 and BFS. Ranking now also uses graph authority and full posterior weighting. The eval harness (#151) and the heat-kernel composition wiring (#310) shipped at v1.6 in default-OFF form, so that the math landed before the bake-off. v1.7 flipped the lane defaults once the gate criteria of #154 passed.

- **Signed normalized Laplacian + offline eigenbasis (top-K=200) builder** ([#149](https://github.com/robotrocketscience/aelfrice/issues/149)). The build step is offline-only. It has no runtime cost. It is a hard prerequisite for #150.
- **Heat kernel authority signal via precomputed eigenbasis** ([#150](https://github.com/robotrocketscience/aelfrice/issues/150)). The signal gives +0.41 NDCG at +7.8 ms p50 on a 50k-belief store. This is the largest single retrieval gain in the bake-off.
- **Plate FFT HRR primitives — port to public repo** ([#216](https://github.com/robotrocketscience/aelfrice/issues/216)).
- **HRR structural-query lane (bind/probe over outgoing edges)** ([#152](https://github.com/robotrocketscience/aelfrice/issues/152)). This lane is a separate retrieval lane, and not a projection. The bake-off rejected the naive HRR projection into BM25 ranking at -0.10 NDCG (R9 in the bake-off). Bind/probe over outgoing edges is the structural-query path that survives.
- **`uri_baki` post-rank adjuster** ([#153](https://github.com/robotrocketscience/aelfrice/issues/153)). The name comes from Uri and Baki, the two Fire Aelfmaidens in Gene Wolfe's *The Wizard Knight* (2004). They are bound attendants who operate after the main action, and who tilt outcomes for their bound knight. The adjuster has three parts: the locked floor (Uri's protection), the supersession demote (Baki's undermining), and the recency decay (the Aelfrice time-tilt). Pandu Nayak's DOJ testimony (October 2023) and the May 2024 Content Warehouse API leak describe the same pattern publicly as Google's "Twiddler". aelfrice uses a neutral name, so that it does not trade on Google's term-of-art.
- **Posterior-weighted ranking — full default-flip** ([#151](https://github.com/robotrocketscience/aelfrice/issues/151)). The harness and the composition shipped at v1.6.0 in default-OFF form. v1.7 flipped the lane defaults after the harness cleared the MRR-uplift and ECE thresholds.
- **Benchmark-gate default-on flip** ([#154](https://github.com/robotrocketscience/aelfrice/issues/154)). The composition tracker shipped at v1.5.0 with the per-lane gate. v1.7 promoted the heat-kernel lane and the posterior lane from default-OFF to default-ON.

### v2.0.0 — feature parity and reproducibility (shipped)

`benchmarks/` reproduces every published headline number on a fresh clone. The command is `uv sync && uv run aelf bench all --out results.json`. The results stay within the documented tolerance bands.

- ~~HRR vocabulary bridge~~ — **closed by the structural-query lane (#152, default-on as of v2.1)**. The lab campaign (`exp/hrr-vocabulary-bridge`) restated "vocabulary bridge" as "typed-edge structural retrieval". That mechanism shipped in `src/aelfrice/hrr_index.py`. See [feature-hrr-integration.md](../design/feature-hrr-integration.md). #433 is closed. #536 (the parallel `vocab_bridge.py` query-rewrite module) is removed.
- Type-aware compression reduces the number of tokens for each belief in the retrieved output.
- Intentional clustering places related beliefs together. The intent is higher coherence on multi-fact queries.
- The correction-detection eval uses a labeled fixture over five codebases. Both the zero-LLM detector and the LLM-judge path score that fixture.
- The posterior drives ranking from end to end. The 10-round MRR uplift eval and the ECE calibration scorer ship with this release.
- The surface expands with `wonder`, `reason`, `core`, `unlock`, `delete` and `confirm`. It also expands with graph metrics and document linking.
- Reproducibility harness. `benchmarks/results/v2.0.0.json` is canonical. CI runs the academic suite every night.

### v3.0.0 — completion + design cut (shipped 2026-05-13)

v3.0 closed the wonder-lifecycle wave. It shipped HRR persistence with a split-format on-disk migration. It shipped type-aware compression at the A2 recall@k bench gate. It shipped read-only federation. It ratified four v3-level design decisions. This section replaced the prior v2.2 row, whose three referenced issues were stale: #197 is WONTFIX; the #193 evaluation shipped without a hook successor; #194 was `ingest_turn(bulk=)`, which also shipped. Per-entry detail: [CHANGELOG/v3.md § 3.0.0](../../CHANGELOG/v3.md). Milestone tracker: [#608](https://github.com/robotrocketscience/aelfrice/issues/608).

Substrate completion (all shipped):

- **HRR persistence default-ON + split-format save/load** ([#553](https://github.com/robotrocketscience/aelfrice/issues/553)). `HRRStructIndex.save()` writes a per-store directory that holds `struct.npy` + `meta.npz`. A legacy `.npz` bundle still loads, and prints a deprecation warning. Persistence is on by default. To opt out, set `[retrieval] hrr_persist = false` or `AELFRICE_HRR_PERSIST=0`. An ephemeral path disables persistence automatically.
- **Wonder lifecycle completion** ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella). Phantom promotion Surface A + Surface B ([#550](https://github.com/robotrocketscience/aelfrice/issues/550) + [#616](https://github.com/robotrocketscience/aelfrice/issues/616)) ship per the 2026-05-11 ratification (no count-trigger). Skill-layer subagent dispatch reaches `wonder_ingest` ([#552](https://github.com/robotrocketscience/aelfrice/issues/552)). Wonder/reason parity ([#645](https://github.com/robotrocketscience/aelfrice/issues/645)) covers the Verdict/ImpasseKind classifiers, the ConsequencePath fork-on-CONTRADICTS, and VERDICT-driven dispatch with suggested-updates close-the-loop. The positional form `aelf wonder QUERY` defaults to the axes flow.
- **Type-aware compression A2 bench gate** ([#434](https://github.com/robotrocketscience/aelfrice/issues/434)). The `run_compression_a2_uplift` driver landed. The strict positive gate `mean_recall@k(use_type_aware_compression=ON) > OFF` is in place. The rebuilder continuation-fidelity (A4) cleared in the same campaign. The default-on flip landed in [#769](https://github.com/robotrocketscience/aelfrice/issues/769), after the [#878](https://github.com/robotrocketscience/aelfrice/issues/878) compose-reconciliation with `use_intentional_clustering`.
- **Eval-harness completion** ([#592](https://github.com/robotrocketscience/aelfrice/issues/592), [#600](https://github.com/robotrocketscience/aelfrice/issues/600), [#687](https://github.com/robotrocketscience/aelfrice/issues/687)). The host-agent replay path writes and joins per-run JSONL. The opt-in LLM-judge stage scores open-ended turns at the operator's anchor tier. The Cohen's-κ runner gates inter-judge agreement at ≥ 0.70. It also gates hot-start fidelity at ≥ 0.80. The synthetic `hot_start` fixture covers post-compact "where were we?" prompts.
- **Read-only federation** ([#650](https://github.com/robotrocketscience/aelfrice/issues/650), [#655](https://github.com/robotrocketscience/aelfrice/issues/655), [#688](https://github.com/robotrocketscience/aelfrice/issues/688), [#689](https://github.com/robotrocketscience/aelfrice/issues/689), [#690](https://github.com/robotrocketscience/aelfrice/issues/690), [#713](https://github.com/robotrocketscience/aelfrice/issues/713)). Beliefs gain a `scope` field (`project` / `global` / `shared:<name>`). The FTS5 and BFS results of a peer DB are visible through `knowledge_deps.json`. `aelf promote --to-scope` changes the visibility. A mutation against a foreign belief ID raises `ForeignBeliefError`. `aelf reason` annotates a peer hop with `[scope:<name>]`.
- **`query_strategy` default flip** ([#718](https://github.com/robotrocketscience/aelfrice/issues/718)). `DEFAULT_STRATEGY` flipped `legacy-bm25` → `stack-r1-r3` on bench evidence. That evidence was +0.2851 absolute NDCG@k, which is +94.8%, at a p99 latency of +0.96 ms over legacy-bm25. That latency is 19% of the +5 ms delta-budget that [`tests/bench_gate/test_query_strategy.py`](../../tests/bench_gate/test_query_strategy.py) enforces. `legacy-bm25` remains callable through an explicit kwarg, until PR-4 removes the code path one minor release later. **Later reverted** ([#1501](https://github.com/robotrocketscience/aelfrice/issues/1501)): the +0.2851 held only while the FTS5 MATCH was conjunctive. [#1177](https://github.com/robotrocketscience/aelfrice/issues/1177) made that MATCH disjunctive. The change lifted `legacy-bm25` on the same corpus from 0.3006 to 0.9553, and turned the uplift into −0.1324. The default therefore returned to `legacy-bm25`, and PR-4 is moot.

Design ratifications (all closed, doc-only follow-through):

- **NL-relatedness philosophy** ([#605](https://github.com/robotrocketscience/aelfrice/issues/605), ratified 2026-05-10). The decision is Option 1: stay deterministic, with a narrow surface. The dedup gate, the contradiction gate and the relatedness gate live in the consuming agent, and not in aelfrice. Memo: [`docs/design/v3_relatedness_philosophy.md`](../design/v3_relatedness_philosophy.md).
- **Sentiment-feedback hook production wire-up** ([#606](https://github.com/robotrocketscience/aelfrice/issues/606), ratified 2026-05-10). The lane is `UserPromptSubmit`. The lane is default-off and opt-in. The decay policy uses the most recent window. The lane shipped behind `[feedback] sentiment_from_prose = true` / `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1`.
- **Multimodel scope** ([#607](https://github.com/robotrocketscience/aelfrice/issues/607), deferred 2026-05-11). There is no maintainer validation path for third-party LLM CLIs. The wonder-dispatch lane (#542/#551) covers the in-tree work.
- **Federation write model** ([#661](https://github.com/robotrocketscience/aelfrice/issues/661), ratified 2026-05-11). The decision is Option B: read-only federation. The per-project DB is the sole writer. A peer opens a foreign DB read-only, and combines the FTS5 results with a UNION. Mutation tools reject foreign belief IDs at the API surface. The sub-issues for conflict-free replicated data type (CRDT) primitives (#651-#654) closed WONTFIX. For the forward-compat version-vector substrate, see [`docs/design/federation-primitives.md`](../design/federation-primitives.md) §1. §2-§5 are flagged as a deferred multi-writer extension.

### v3.0.1 — install-surface collapse + default-on agent-side retrieval (shipped 2026-05-13)

This patch release follows v3.0.0. There is no public API change. The user-facing surface narrows on the install side. The same surface broadens on the retrieval side. Per-entry detail: [CHANGELOG/v3.md § 3.0.1](../../CHANGELOG/v3.md).

- **Install / upgrade surface collapsed to `uv tool` only** ([#730](https://github.com/robotrocketscience/aelfrice/issues/730)). `aelf upgrade-cmd` and `/aelf:upgrade` emit a single in-place form (`uv tool upgrade aelfrice`) for a uv-managed install. For any other installer, they emit a migration chain (`pipx uninstall aelfrice && uv tool install aelfrice`, or the pip equivalent). `UpgradeAdvice.context` collapses `uv_tool` / `pipx` / `venv` / `system` → `uv_tool` / `non_uv`. The README and `docs/user/INSTALL.md` are rewritten around `uv tool install`. The pipx, venv and system helpers in `lifecycle.py` remain internal, and no longer appear as supported channels.
- **Auto-migrate non-uv installs on first 3.0.1 setup** ([#733](https://github.com/robotrocketscience/aelfrice/issues/733), follow-up [#774](https://github.com/robotrocketscience/aelfrice/issues/774) for the uv-not-found runnable one-liner). `aelf setup` runs `lifecycle.maybe_migrate_to_uv()` before hook reconciliation. On a pipx or pip install with `uv` on `$PATH`, that function runs `uv tool install --force aelfrice` (120s timeout). It runs one time on each machine, behind a `~/.aelfrice/migrated-to-uv` sentinel. The fresh uv-tool shim overwrites the existing `~/.local/bin/aelf`, so a later invocation resolves through the uv-tool venv. A pipx-only user without `uv` now sees an installer hint that the user can copy and paste. The hint is `curl -LsSf https://astral.sh/uv/install.sh | sh`, and `brew install uv` on macOS. Before this release, that user saw only a docs URL.
- **`search-tool` and `search-tool-bash` hooks default-on** ([#738](https://github.com/robotrocketscience/aelfrice/issues/738)). `aelf setup` with no flags wires both the `PreToolUse:Grep|Glob` hook and the `PreToolUse:Bash` (grep/rg/find/fd/ack) hook. The README claim of "four parallel retrieval lanes" was aspirational on agent-initiated search. This release makes the claim true. The flags follow the `--X / --no-X` BooleanOptionalAction convention. The opt-out persists across upgrades through `~/.aelfrice/opt-out-hooks.json`. A 10k-belief corpus is typical for v3.0. The worst-case latency on that corpus is ~22ms per turn for 5 Grep fires, plus ~12ms per turn for 3 Bash-search fires. The Bash hook is capped at 3 firings per turn.
- **Cross-fire injection dedup ring** ([#740](https://github.com/robotrocketscience/aelfrice/issues/740)). A new per-session rolling-window first-in-first-out (FIFO) ring lives at `<git-common-dir>/aelfrice/session_injected_ids.json`. The UPS hook appends the per-turn hit IDs after its `<aelfrice-memory>` block. The `PreToolUse` search hook filters the `retrieve()` results against the ring before it emits `<aelfrice-search>`. That search hook has three emit shapes:

  1. all-recent → a pointer block `note="answer already in prompt context"`;
  2. mixed → render the new beliefs with a trailing "(N more matching belief(s) already in prompt context from earlier this session)";
  3. all-new → the original block, unchanged.

  A locked belief passes through as `new`, whatever the state of the ring. A new session_id clears the ring. `fcntl.LOCK_EX` serializes a near-simultaneous read-modify-write from UPS and PreToolUse. The cap defaults to 200 IDs. If `AELFRICE_INJECTION_RING_MAX` is set, that environment variable overrides the default. `aelf doctor` surfaces `injection ring: N/MAX ids (evicted K this session)`.
- **Transitive `authlib` 1.7.0 → 1.7.2 for CVE-2026-44681** ([GHSA-r95x-qfjj-fjj2](https://github.com/advisories/GHSA-r95x-qfjj-fjj2)). The advisory reports a medium-severity (CVSS 6.1) open-redirect in the OIDC server flows of Authlib (`OpenIDImplicitGrant` / `OpenIDHybridGrant`). The exposure of aelfrice is effectively zero. `authlib` is transitive through `fastmcp` (the optional `[mcp]` extra), and the aelfrice MCP server implements no OIDC authorization endpoint. The change touches only the lockfile.

## Recovery inventory

This table lists what the research line had, and when each part returns:

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
| Graph-traversal store methods (`edges_from`, `edges_to`, `get_edge`; BFS expansion in `bfs_multihop.py`) | shipped v1.3–v1.6 (substrate for `wonder` + `reason`; ships ahead of v2.0) |
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

The four "candidate" lines are the orphaned research-line capabilities from the agentmemory parity audit. They do not ship today, and this roadmap did not list them before. They land if and only if a benchmark or an experiment justifies the inclusion; see the validation discipline below. If no benchmark or experiment justifies the inclusion, they stay parked.

### Deliberately not on this list

The research line also shipped the following capabilities that aelfrice does **not** plan to recover:

- **Research-artifact provenance metadata** and the **rigor-tier** classification layer. The provenance metadata holds `produced_at` / `method` / `sample_size` / `data_source` / `independently_validated` for each belief. The rigor-tier layer holds `hypothesis` / `simulated` / `empirically_tested` / `validated`. A case study motivated this work in the research line. In that case study, a new agent miscalibrated the project maturity from raw completion counts. aelfrice v1 stores provenance only through the `Belief.origin` enum. That enum holds 7 source-tier values at v1, and 8 values since the `user_transcript` tier of v3.3.0 (#888). Epistemic-rigor metadata is **not** on the v2.0 surface. If status reporting needs this signal, the signal lands as a separate feature with its own benchmark. It does not land as a schema-wide migration.
- **Session-velocity tracking** (items/hour decay scaling). v1 ships per-belief decay with type-specific half-lives. Velocity-scaled decay is the research-line refinement, and it is parked.
- **Calibrated status reporting** that surfaces the rigor-tier distribution and the velocity context to a new agent. This reporting depends on the two items above.
- **Cross-project shared scopes** through SQLite ATTACH. The Multi-project query non-goal in [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md) subsumes this capability. This list names the capability, so that the research-line term ("shared scopes") does not read as an oversight.
- **Vault-as-source-of-truth storage** and **reverse sync (Obsidian → aelfrice)**. Both are hard non-goals. SQLite remains the source of truth. Per-project isolation is a hard property. The same federation non-goal subsumes both items. **Export-only** Obsidian rendering shipped in v3.1+ as `aelf export-obsidian` (#630). The export is one-way, from the DB to the vault. The vault is a regenerable artifact, and not a content store. The v1 rejection above covered the broader vault-as-source-of-truth surface and the reverse-sync surface. Both of those surfaces remain rejected.

## Compatibility

aelfrice follows semver:

- **Patch (v3.x.y):** the API and the schema stay the same.
- **Minor (v3.x.0):** the API stays the same. Migrations are forward-compatible, and add new columns or tables only.
- **Major (v4.0.0):** a major release may break the v3 API only where a benchmark or an eval justifies the break. Migrations are documented and tested before the tag.

The minor and major boundaries before v3 followed the same shape (v1.x.0 / v2.0.0 / v3.0.0). The v1→v2 transition documented all its breaking changes in `CHANGELOG/v2.md`. The v2→v3 transition documented all its breaking changes in `CHANGELOG/v3.md`.

## Non-goals

- **Vector database / embedding retrieval.** aelfrice stays on SQLite + FTS5 at every milestone. The HRR bridge in v2.0 is a structural retrieval layer, and not an embedding layer.
- **Cloud sync.** The runtime stays local. No release introduces network I/O in the retrieval path or in the write path.
- **Telemetry.** No release adds outbound network calls in the default install.
- **Brain-graph sync or multi-writer federation.** v3.0 ships *read-only* federation (#650 / #655 / #661). A project can declare peer DBs in `knowledge_deps.json`. The project can then surface the `global` / `shared:<name>` beliefs of those peers in FTS5 + BFS. Every per-project DB stays the sole writer for its own rows. A mutation against a foreign belief ID raises `ForeignBeliefError`. The same cut filed multi-writer federation with CRDT primitives (#651-#654), and closed it WONTFIX. For the boundary, see [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md).

## Validation

Every release with a behavioural claim ships with a benchmark or a test that demonstrates the claim. The central claim of v1.0 is that `apply_feedback` updates posteriors mathematically. Tests and the synthetic harness proved that claim at the tag. v1.3 added the claim that BM25 moves under feedback. v1.3 shipped that claim behind the partial posterior-weighted ranking lane. v2.0 reproduced the published MRR uplift against the academic corpus on a fresh clone. v3.0 added the eval-harness completion (host-agent replay, LLM-judge, Cohen's-κ gate) and the read-only federation determinism check. The canonical headline results are pinned at the benchmark-bearing cuts (`benchmarks/results/v2.0.0.json`, `v3.0.1.json`). The nightly reproducibility cron (`bench-canonical.yml`) writes ongoing snapshots to the `bench-canonical-results` branch.
