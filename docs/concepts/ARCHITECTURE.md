# Architecture

How aelfrice fits together. Maps directly to source under `src/aelfrice/`.

## Principles

1. **Determinism end to end.** Every retrieval result is bit-identical given the same write log and the same code. Every result traces to named beliefs and named rules. See [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property). **`edges` are the standing exception:** they are written outside the log as shipped, so two stores with identical `ingest_log` content can hold different graphs and the L3 edge-walk lane can return different results ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)).
2. **SQLite plus a small numeric stack.** No vector DB, no embeddings, no LLM in the hot path. Required deps beyond stdlib: `numpy` and `scipy` (added v1.5.0, #148, for the BM25 sparse-matvec lane; now also used by the HRR and spectral-graph lanes) and `snowballstemmer` (added v1.7.0, #154, for Porter stemming). Optional extras: `[onboard-llm]` (the direct-API onboard classifier SDK, for `aelf onboard --llm-classify`), `[archive]` (cryptography, for `aelf uninstall --archive`), `[benchmarks]` (dev-side adapters).
3. **Bayesian, not vibes.** Confidence is `α / (α + β)`. Every update has a closed-form rule. At v1.3.0+ the posterior is combined log-additively with BM25 on the L1 tier — see [LIMITATIONS](../user/LIMITATIONS.md) for what the partial ranking does and doesn't cover.
4. **`apply_feedback` is the central endpoint.** One writer of `(α, β)`. One audit row per successful update.
5. **Locks are user-asserted ground truth.** A user-locked belief short-circuits decay — a mechanism that does not currently run; see the posterior-decay note below. Lock correction is an explicit user act via `aelf lock` overwriting (PHILOSOPHY [#605](https://github.com/robotrocketscience/aelfrice/issues/605)); contradiction-driven auto-demotion was removed in [#814](https://github.com/robotrocketscience/aelfrice/issues/814).

### Enrichment-step boundary

The determinism contract applies to retrieval — every read is reproducible from the inputs. Some write-side operations (LLM-driven sentence classification on the polymorphic onboard path; future research-line capabilities) involve non-deterministic steps. The boundary is explicit:

- Inputs to enrichment (sentence, source) and the classifier's *outputs* (`raw_meta["route_overrides"]` —
  belief type, origin, alpha, beta) are recorded. **The model id, model version and prompt-template hash are
  not.** They live only on the transient router object and in stdout telemetry, so "which model produced this
  belief's type and prior?" is not answerable from the store. That bounds this carve-out less than it reads:
  it localises the non-determinism to a recorded step without making the step reproducible. Persisting them
  into `raw_meta` is tracked separately.
- Outputs (belief type, prior, derived edges) are stored as deterministic content with provenance — except that *derived edges* carry provenance only in part: until [#1354](https://github.com/robotrocketscience/aelfrice/issues/1354) all six `derive()` return paths emitted `edges=[]`, so `ingest_log.derived_edge_ids` was NULL on every row and no edge had a log row pointing at its origin ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)). `derive()` now emits `DERIVED_FROM` for the intra-turn case and the column populates forward-only — at most 1.93% of the live edge set. `TEMPORAL_NEXT` (88.3%) and the detector edges are still written outside the log.
- All retrieval and feedback math downstream of the enriched store is deterministic.

The contract is *deterministic substrate + bounded, audited enrichment layer*, not "no model ever touches the data."

## Modules

Imports are *intended* to be one-directional — modules lower in the table import from higher — but this is an
aspiration, not an enforced invariant. One known inversion is broken by a deferred (in-function) import:
`classification.py` imports from `scanner`, which the comment at the import site names as circular. A second
is avoided by duplication rather than by deferral — `store.py` reimplements `wonder.lifecycle`'s
constituent-key hash inline, citing the cycle (`store` ← `wonder.lifecycle`) as one of two reasons — so
counting deferred imports alone understates how often the ordering is worked around. Note the converse too:
not every deferred import is an inversion. `store.py` defers `federation` purely for import cost — the
comment at that site records that `federation` is a leaf module importing nothing from `store`, and the
deferral exists to keep `subprocess` + `json` out of every store consumer. The table is also a curated
subset — 31 modules against the 117 `.py` files under `src/aelfrice/` — not an exhaustive map.

| Module | Responsibility |
|---|---|
| `models.py` | `Belief`, `Edge`, `FeedbackEvent`, `OnboardSession` dataclasses; type / lock / origin constants. No I/O. |
| `scoring.py` | `posterior_mean`, `partial_bayesian_score`, and the gamma / zeta posterior rerank scorers — the functions retrieval actually imports. Also defines `decay` / `type_half_life` / `TYPE_HALF_LIFE_SECONDS` (lock-floor short-circuit, Jeffreys `(0.5, 0.5)` target), which **no module under `src/` calls**: posterior decay is designed but not wired ([#1218](https://github.com/robotrocketscience/aelfrice/issues/1218)); disposition tracked under [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162). |
| `store.py` | SQLite WAL + FTS5 + CRUD. `propagate_valence` BFS with broker-confidence attenuation — fired by `apply_feedback` on every direct feedback event (disable with `AELFRICE_VALENCE_PROPAGATION=0`). |
| `retrieval.py` | `retrieve(store, query, token_budget=2400)` — L0 locked + L2.5 entity-index (v1.3+) + L1 FTS5 BM25/BM25F (BM25F default-on since v1.7.0) with Bayesian log-additive reranking (v1.3+) + L3 BFS multi-hop (v1.3+, default-off) over the L0+L2.5+L1 seed set. L0 never trimmed. |
| `feedback.py` | `apply_feedback(store, belief_id, valence, source)` — only Bayesian-update path. Writes `feedback_history`. |
| `contradiction.py` | `resolve_contradiction` — picks a winner per precedence, inserts `SUPERSEDES`, writes audit row. Backs `aelf resolve`. |
| `correction.py` | No-LLM heuristic correction detector. |
| `classification.py` | Type priors + regex fallback. Polymorphic onboard state machine. |
| `noise_filter.py` | `is_noise(text, config)` — filters markdown headings, checklist blocks, three-word fragments, license boilerplate. Tunable via `.aelfrice.toml` — see [CONFIG](../user/CONFIG.md). |
| `scanner.py` | `scan_repo` — filesystem + git log + Python AST extractors. Idempotent on `content_hash`. |
| `health.py` | v1.0 regime classifier (`supersede` / `ignore` / `mixed` / `insufficient_data`). Surfaced via `aelf regime`. |
| `auditor.py` | Structural auditor: orphan threads, FTS5 sync, locked contradictions, corpus volume. Backs `aelf health`. Pure read-only. |
| `migrate.py` | One-shot port from the legacy global DB into the per-project DB. Reads source via SQLite `mode=ro`. Backs `aelf migrate`. |
| `doctor.py` | Settings-linter: walks every `command` in `settings.json` and verifies it resolves. Special-cases `bash <script>` wrappers. Backs `aelf doctor`. |
| `lifecycle.py` | Update notifier (PyPI background check), uninstall machinery, archive encryption. |
| `transcript_logger.py` | Hook entry-point for v1.2+ transcript capture. Writes one JSONL line per turn under `<git-common-dir>/aelfrice/transcripts/`. |
| `hook_commit_ingest.py` | `PostToolUse:Bash` hook — ingests commit messages after `git commit`. |
| `hook_search.py` | UserPromptSubmit retrieval helper that records every hit as a `feedback_history` row tagged `source='hook'`. Audit-only since #1086 (v4.0): the row logs exposure/recurrence but `record_retrieval` passes `update_posterior=False` by default, so a surfacing does **not** move α/β unless `AELFRICE_EXPOSURE_UPDATES_POSTERIOR=1` restores the legacy promote-on-exposure behaviour. |
| `triple_extractor.py` | Pure-regex `(subject, relation, object)` extraction over six relation families. Used by commit-ingest and transcript-ingest. |
| `context_rebuilder.py` | Post-compaction rebuilder that re-surfaces aelfrice retrieval after the harness summarises (delivered on `SessionStart(source="compact")`, #1031). |
| `benchmark.py` | Deterministic 16-belief × 16-query synthetic harness. Frozen `BenchmarkReport`. |
| `cli.py` | argparse multi-subcommand CLI. Entry: `aelf`. Everyday surface in `aelf --help`; full surface (diagnostic, hook, lifecycle verbs) in `aelf --help --advanced`. |
| `federation.py` | (v3.0+) Read-only peer-DB federation. `load_peer_deps()` parses `knowledge_deps.json`; `open_peer_connection(path)` opens a peer SQLite in `mode=ro` (honouring the peer's WAL, with an `immutable=1` fallback for read-only media); `ForeignBeliefError` rejects mutations against foreign belief ids at the API surface. See [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md). |
| `clamp_ghosts.py` | (v3.0+) `clamp_ghost_alphas(store, target_alpha, dry_run)` repair tool — clamps α on belief rows that have inflated posteriors without audit-trail backing (pre-migration artifacts only). Reversible via the negative-valence audit row written inside the same transaction. Backs `aelf clamp-ghosts` (hidden). |
| `reason.py` | (v2.0+, expanded v3.0) Graph-walk reasoning over the belief edge graph. v3.0 (#645, #658) adds Verdict / ImpasseKind classifiers, `ConsequencePath` fork-on-CONTRADICTS deriver, `dispatch_policy()` mapping impasses to Verifier/Gap-filler/Fork-resolver roles, and `suggested_updates()` close-the-loop feedback row derivation. Backs `aelf reason` + `/aelf:reason`. |
| `wonder/` | (v2.0+, expanded v3.0) Wonder lifecycle: gap analysis (`dispatch.py`), research-axes generation, phantom ingest/GC (`wonder_ingest`, `wonder_gc`), Skill-layer subagent integration (`skill_integration.py` per #552), structured `WonderResult` dataclass (#656). |
| `sentiment_feedback.py` | (v2.0 module, v3.0 hook wired) Regex sentiment detector. v3.0 (#606) wires it into `UserPromptSubmit` behind `[feedback] sentiment_from_prose = true`. |
| `auto_install.py` | (v3.0+, #623) Version-stamped manifest merger. First `aelf <cmd>` after a wheel upgrade merges any new default-on hooks from `data/hook_manifest.json` into `~/.claude/settings.json`. `fcntl`-locked; honors `~/.aelfrice/opt-out-hooks.json`. |
| `working_state.py` | (v3.0+, #587) Post-compact `<working-state>` projector (current branch, bounded `git status`, last HEAD log entries, last K user prompts, session commits). Each git invocation has a 1.5s timeout + return-empty fallback. |
| `setup.py` | Idempotent install/uninstall of all hooks + statusline. Atomic write via tempfile + `os.replace`. |
| `hook.py` | `aelfrice.hook:main` — process Claude Code spawns on each prompt. Reads stdin, calls `retrieve()`, emits `<aelfrice-memory>` on stdout. Non-blocking. Entry: `aelf-hook`. |
| `slash_commands/` | One markdown file per CLI subcommand surfaced in `/aelf:*`. |

## Data model

**Belief** — `id, content, content_hash, alpha, beta, type, lock_level, locked_at, origin, session_id, created_at, last_retrieved_at, corroboration_count, hibernation_score, activation_condition, retention_class, valid_to, scope, project_context` (v3.2+, #858)`, last_confirmed_at` (v3.5+, #936)`, lock_tier` (v3.7+, #1016).

- `type ∈ {factual, correction, preference, requirement, speculative}` (`speculative` added with the v3.0 wonder lifecycle, #548, for phantom beliefs)
- `retention_class ∈ {fact, snapshot, transient, unknown}` — drives type-aware compression (#769)
- `lock_level ∈ {none, user}`
- `lock_tier ∈ {frozen, reference}` (v3.7+, #1016-B) — orthogonal to `lock_level`, only meaningful when `lock_level = user`. `frozen` (the default for every lock) is always injected verbatim; `reference` is bounded — injected as a one-line manifest entry, full text read on demand via `aelf locked` / `aelf search`. Demote bulky locks with `aelf lock <text> --reference`.
- `origin ∈ {user_stated, user_corrected, user_validated, user_transcript, agent_inferred, agent_remembered, document_recent, speculative, unknown}` (v1.2+; `user_transcript` added with the v2.1 transcript-ingest lane; `speculative` added with the v2.0 wonder substrate for phantom beliefs and now written by `wonder/lifecycle.py`)
- `scope ∈ {project, global, shared:<name>}` (v3.0+, #688). `project` is the default and local-only; `global` is surfaced to any peer DB that declares this DB in its `knowledge_deps.json`; `shared:<name>` is surfaced only to peers that also list `shared:<name>` as a dep.

**Edge** — `src, dst, type, weight, anchor_text`. Ten edge types in `EDGE_VALENCE`:

| Type | Valence | |
|---|---|---|
| `SUPPORTS` | +1.0 | full positive |
| `IMPLEMENTS` | +0.65 | code-implements-spec link |
| `TESTS` | +0.55 | test-covers link |
| `CITES` | +0.5 | half positive |
| `DERIVED_FROM` | +0.5 | half positive (turn-to-turn provenance) |
| `RELATES_TO` | +0.3 | weak positive |
| `TEMPORAL_NEXT` | +0.2 | session-time successor |
| `SUPERSEDES` | 0.0 | structural; no propagation |
| `RESOLVES` | 0.0 | structural; closes a `CONTRADICTS` thread |
| `CONTRADICTS` | -0.5 | half negative |

A separate `POTENTIALLY_STALE` edge type exists as a producer-only signal from `aelf doctor` (#387) and is deliberately not in `EDGE_TYPES` — it does not participate in valence propagation. The research line carried 17 edge types — additional speculative/causal markers (`SPECULATES`, `DEPENDS_ON`, `HIBERNATED`) and additional structural extractors (`CALLS`, `CO_CHANGED`, `CONTAINS`, `COMMIT_TOUCHES`) remain parked until the extractors that produce them ship. The current ten-type set covers the v2.0 wonder lifecycle (`RESOLVES`, `SUPERSEDES`, `CONTRADICTS`) and the v1.x code/test linkage (`IMPLEMENTS`, `TESTS`); see [ROADMAP § Recovery inventory](ROADMAP.md#recovery-inventory) for the deferred set.

**Core SQLite tables include:** `beliefs` (with `scope` column since v3.0), `beliefs_fts` (virtual, porter unicode61), `edges` PK `(src, dst, type)`, `feedback_history`, `sessions`, `onboard_sessions`, `belief_corroborations` (sibling table, v1.5.1+), `ingest_log` (append-only, v1.6+), `belief_versions` + `edge_versions` (per-scope version vectors, v1.5+), `belief_entities` (L2.5 entity index, v1.3+), `deferred_feedback_queue` (v1.6+, #191), `belief_documents` (wonder research docs, v3.0), `injection_events` (relevance signal, v3.x), `belief_touches` (hot-path ring, v3.x #748), `schema_meta`. The `scope` column has an `idx_beliefs_scope` index; both column and index land idempotently via the migration runner.

## Bayesian update

`apply_feedback(store, belief_id, valence, source)`:

1. Load belief. Reject zero valence and empty source.
2. **Positive valence:** `α += valence`.
3. **Negative valence:** `β += |valence|`.
4. Persist atomically.
5. Append a `FeedbackEvent` row. Always.

## Retrieval

L0 (locked beliefs) is the **always-injected pool**: every lock ships on every retrieval, no scoring, no top-K. Lock count is the operator's baseline-context budget knob. `frozen`-tier locks (the default) ship **in full**; `reference`-tier locks (v3.7+, #1016-B) ship as a **one-line manifest entry** and are budgeted at that size, so a large lock set stays bounded — demote bulky locks with `aelf lock <text> --reference` and read their full text on demand via `aelf locked` / `aelf search`. Only the non-locked pool (L1/L2.5/L3) is subject to relevance ranking and budget trim. When locks alone approach the budget, a reserved relevance floor (#1015) keeps query-relevant results from being starved, and `aelf doctor` warns (#1016-D).

```
L0: store.list_locked()              always loaded; never trimmed
        ↓
L2.5: entity-index lookup (v1.3+)    NER-extracted entities → exact + stem match;
        ↓                             default-on; disable via [retrieval] entity_index_enabled = false
L1: FTS5 BM25 / BM25F                limit l1_limit, query escaped;
        ↓                             v1.3+: score = log(bm25) + 0.5*log(posterior_mean)
                                      v1.7+: BM25F anchor-augmented sparse matvec, default-on (#148/#154)
Temporal spine (#1064)               TEMPORAL_NEXT chain traversal from the top-5 L1 seeds,
        ↓                             appended after L1; default-ON since the #1107 Phase-2
                                      cutover, opt out via [retrieval] use_temporal_spine = false
L3: BFS multi-hop expansion (v1.3+)  edge-weighted graph walk from L0+L2.5+L1 seed set;
        ↓                             default-OFF; enable via [retrieval] bfs_enabled = true
Dedupe L1+L2.5+L3 against L0 ids
        ↓
Trim from tail until sum(estimated_tokens) ≤ token_budget
```

Two **rerank modifiers** refine the ranked (L1 / L2.5) tiers without adding a lane. Since v4.0 the production `retrieve()` hook path is a thin adapter over `retrieve_v2` (the #1107 cutover), so both are exposed on the live path — entity-persistence demotion is **default-on and live on production**, while origin tie-break is exposed but **held off** by default:

- **Entity-persistence demotion** (#1096, `[retrieval] use_entity_persist_demote` / `AELFRICE_ENTITY_PERSIST_DEMOTE`) — the organic sink for #1086's junk-percolation problem. A log-additive **demotion** term `min(0, log(S1 + ε))`, where `S1 = durable / (durable + transient + 1)` is read from the `belief_entities` index (one batched query), down-weights candidates that ground only to *transient* coordination tokens (bare PR/issue numbers, version/branch tags) relative to *durable* entities (file paths, error codes, symbols). Pure demotion (well-grounded candidates are neutral, never boosted); applied only to entity-bearing candidates, so entity-free prose is untouched. Note the sink is **content-referential, not temporal** — a time/cold-decay sink was measured empirically inert here (the junk is *hot*, not stale). **Default-ON and live on the production `retrieve()` path since v4.0** (flipped once the #1096 G2 mixed-corpus eval (#1103) cleared the no-regression gate, then graduated onto the hook path by the #1107 cutover); opt out with a falsy env/kwarg/TOML rung.
- **Origin-priority tie-break** (#1089, `[retrieval] use_origin_tiebreak` / `AELFRICE_ORIGIN_TIEBREAK`) — a within-tier tie-break (not a rerank term): when two candidates tie on relevance, the higher-trust *origin* wins, sitting between the relevance score and the id tie-break so relevance always dominates. Byte-identical when off.

Both are deterministic (#605) and byte-identical to the prior pipeline when their flags resolve falsy (origin tie-break is default-off; entity-persistence demotion is default-on, so opt out explicitly for parity).

Token estimate: `(len(content) + 3) // 4`. Empty query: L0 only. L0 always wins overflow.

Spec docs: [entity_index.md](../design/entity_index.md) (L2.5), [bfs_multihop.md](../design/bfs_multihop.md) (L3), [bayesian_ranking.md](../design/bayesian_ranking.md) (L1 Bayesian reranking).

**BFS temporal-coherence caveat:** L3 resolves each hop to the globally latest serial of its target belief. For recall queries this is correct. For audit queries (what did the agent believe at decision-time?) a post-seed supersession can appear mid-chain. The temporal-coherence fix was originally targeted at v2.0.0 but slipped and is not scheduled on a current milestone — see [LIMITATIONS § BFS multi-hop temporal coherence](../user/LIMITATIONS.md#bfs-multi-hop-temporal-coherence).

## Onboarding

`scan_repo(store, path)`:

1. **Filesystem walk** over `*.md`, `*.rst`, `*.txt`, `*.adoc` → `factual` / `requirement` candidates.
2. **Git log** → `factual` candidates with file recency (v1.1.0: `belief.created_at` = file's most recent commit, so decay penalises old branches).
3. **Python AST** → function/class names + docstrings → `factual` candidates.

Classification via priors + regex fallback. Idempotent on `content_hash`.

**LLM onboard classifier (v1.3+, default-OFF):** `aelf onboard --llm-classify` routes each candidate through the vendor's small model instead of the regex path. Four consent gates enforce the privacy boundary: the `[onboard-llm]` extra installed, `ANTHROPIC_API_KEY` present, the `--llm-classify` flag (or `[onboard.llm].enabled` TOML key), and a one-time interactive consent prompt recorded in a sentinel file. `--dry-run` previews candidates without calling the API. Spec: [llm_classifier.md](../design/llm_classifier.md). This is the only path in aelfrice that transmits user content outbound — see [PRIVACY § Onboard-time outbound call](../user/PRIVACY.md#onboard-time-outbound-call).

## Claude Code hook

```
settings.json  hooks.UserPromptSubmit: [{command: "aelf-hook"}]
                          ↓ written by aelf setup
                  Claude Code spawns aelf-hook on each prompt
                          ↓ JSON payload on stdin
                  aelfrice.hook:main
                          ↓ retrieve(store, prompt)
                  .git/aelfrice/memory.db
                          ↓ <aelfrice-memory> block on stdout
                  Claude Code injects above your prompt
```

Non-blocking contract: every failure path exits 0, and a hook problem must never block your prompt. Two
qualifications, both deliberate and both reachable today:

- **One hook blocks by design.** `aelf-pre-issue-hook` exits `2` on a successful duplicate match
  (`pre_issue_create_hook.py`) — see its row below. It never exits non-zero on *error*; the blocking exit is
  the feature, not a failure path.
- **Partial stdout is possible.** The UserPromptSubmit lane writes the cadence-checkpoint block before
  retrieval runs (`hook.py`), so a failure in a later stage exits 0 with that block already flushed. The
  guarantee is "exits 0", not "emits nothing".

## Default-on hooks (v2.1+ / v3.0+)

| Hook | Event | Purpose | Default since |
|---|---|---|---|
| `aelf-hook` | `UserPromptSubmit` | Retrieval injection — the core mechanism. | v1.0 |
| `aelf-transcript-logger` | `UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact` | One JSONL line per conversation turn; PreCompact rotates and re-ingests. | v2.1 (#529) |
| `aelf-commit-ingest` | `PostToolUse:Bash` | After `git commit`, ingest the commit message via the triple extractor. | v2.1 (#529) |
| `aelf-session-start-hook` | `SessionStart` | Inject locked beliefs as `<aelfrice-baseline>` once per session; emit `<recent-work>` sub-block (#887). | v2.1 (#529) |
| `aelf-stop-hook` | `Stop` | End-of-session lock prompt — surfaces session-scoped unlocked correction-class (#582) and directive (#1315) beliefs as `<aelfrice-session-end>` with pre-filled `aelf lock` commands, the directive ones carrying `--for` when a memory verb governs a stated window; also hosts the default-off cadence checkpoint dispatch (#749 / #871 / #876). | v3.0 |
| `aelf-search-tool-hook` | `PreToolUse:Grep|Glob` | Surface relevant beliefs adjacent to tool-driven search (#674). | v3.0.1 (#738) |
| `aelf-search-tool-hook` | `PreToolUse:Bash` | Same entry point, separate settings.json matcher, for bash search invocations. | v3.0.1 (#738) |
| `aelf-pre-issue-hook` | `PreToolUse:Bash` | Duplicate-detection guard before `gh issue create` — blocks (exit 2) on Jaccard title overlap ≥ 0.5 against open issues and shipped commits (#941). | v3.5.0 |
| `aelf-claude-memory-mirror` | `PostToolUse:Write\|Edit\|MultiEdit` | One-way mirror of host claude-memory fact-file writes into the belief graph (#985). Consent-gated since v4.0 (#1089): runs once the first-setup reconcile records per-project consent, or when `AELFRICE_MIRROR_CLAUDE_MEMORY` / `[memory] mirror_claude_memory` enables it explicitly; an explicit falsy value is the opt-out. | v3.7.0 |
| `aelf-agent-context-hook` | `PreToolUse:Agent\|Task` | Worker-context injection — dispatched workers inherit L0 locked + task-relevant beliefs via the harness `updatedInput` channel; fail-open passthrough, kill switch `AELFRICE_AGENT_CONTEXT=0` (#1068). | v4.0.0 |
| `aelf-pre-compact-hook` | `PreCompact` | Rebuilder trigger-mode bookkeeping only — never injects (#1031). Opt-in via `--rebuilder`; default trigger flipped from `manual` to `threshold` at v3.1 (#746). | opt-in |
| `aelf-session-start-hook` | `SessionStart(source="compact")` | Emits the rebuild block after compaction — the channel the harness actually honors (#1031). | opt-in |

Each lane is opt-out via `aelf setup --no-<lane>`. All exit 0 on failure — with the two qualifications stated
under the non-blocking contract above: `aelf-pre-issue-hook` exits `2` on a duplicate match by design, and
partial stdout is possible when an early stage has already emitted.

The `PostToolUseFailure:<tool_name>` event-name namespace inside
`~/.aelfrice/hook-activity.jsonl` is reserved for raw tool-failure
observation produced by a HOME-side hook (tracked separately). See
[hook_activity_schema](../design/hook_activity_schema.md) for the field schema
and the consumer-side dedupe-by-fingerprint warning.

## Post-compaction rebuilder

> **Delivery channel (corrected by [#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)):** the rebuild block ships on **SessionStart** with `source == "compact"`, *after* compaction — not on `PreCompact`. The harness rejects `additionalContext` emitted from a `PreCompact` hook (`PreCompact` is absent from the events that support it), so a block written there is discarded with a validation error. `pre_compact()` therefore emits nothing on stdout; it is retained for trigger-mode parity only. `rebuild_v14` and the block content are unchanged.

Compaction is reached two ways — the harness hitting its context limit, or the user compacting explicitly — and both take the same path: `PreCompact` fires, the harness compacts, then `SessionStart` fires with `source == "compact"`. The rebuilder does its bookkeeping on the first event and its injection on the last. Do not read the flow below as an auto-compaction-only path: on a measured local corpus 72 of 73 scoreable compactions were explicit rather than automatic ([#1252](https://github.com/robotrocketscience/aelfrice/issues/1252)), so the explicit route is the one an operator debugging the rebuilder is most likely on:

```
PreCompact fires
      ↓
aelf-pre-compact-hook resolves trigger_mode and surfaces the
dynamic-mode parked trace on stderr — no stdout, nothing injected
      ↓
the harness compacts the conversation
      ↓
SessionStart fires with source == "compact"
      ↓
aelf-session-start-hook reads the last N turns from turns.jsonl
      ↓
rebuild_v14(recent_turns, store, token_budget)
      → L0 locked beliefs (always first)
      → session-scoped beliefs matching recent content
      → BM25+posterior hits against the session tail
      packed to token_budget (default: [rebuilder].token_budget in .aelfrice.toml)
      ↓
emitted as additionalContext — the aelfrice block lands in the
new context alongside the harness's own summary (augment mode)
```

`aelf rebuild [--transcript PATH] [--n N] [--budget N]` runs the same codepath manually (prints block to stdout). Install via `aelf setup --rebuilder`. Default `DEFAULT_TRIGGER_MODE` flipped from `"manual"` to `"threshold"` at v3.1 ([#746](https://github.com/robotrocketscience/aelfrice/issues/746)). Spec: [context_rebuilder.md](../design/context_rebuilder.md). Eval fixture policy: [eval_fixture_policy.md](../design/eval_fixture_policy.md).

## Tests

| Layer | Marker | Coverage |
|---|---|---|
| Unit | default | One property per test. Pyright strict. |
| Property | default | Pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, token-budget invariant, broker-attenuation. |
| Regression | `@pytest.mark.regression` | Cross-module scenarios: retrieval round-trip, feedback loop, onboarding, setup→hook→unsetup, `aelf bench` end-to-end. |

`uv run pytest` (7,300+ tests at v4.2.0).

## Out of scope through v1.x

These remain parked until a benchmark, experiment, or concrete failure mode justifies them:

- Sentence-transformer embeddings (HRR primitives shipped at v1.7.0 as a structural lane, not a learned-embedding lane; v3.0 PHILOSOPHY ratification #605 keeps determinism as the property — no embedding lane planned)
- Multi-writer federation / CRDT primitives. v3.0 ships *read-only* federation (#650 / #655 / #688) — peers open foreign DBs read-only and UNION FTS5 results. The multi-writer extension (#651-#654 CRDT primitives) closed WONTFIX at the v3.0 cut per the #661 ratification.
- Full composition tracker — 10-round MRR uplift, ECE calibration, BM25F × heat-kernel × HRR-structural composition eval (#154). Heat-kernel and HRR-structural defaults flipped on at v2.1; the joint-composition bench gate as such was not run separately, but the #437 reproducibility-harness cleared 11/11 **at the v2.1.0 cut (2026-05-08)** and covers the
substrate as of that measurement. The README badge currently reads `partial (6/11 adapters)` — a later
regression in the nightly canonical run, not a retraction of the v2.1.0 gate. Read the 11/11 as the
evidence standing at the time of the flip, not as the harness's present state.

The following were previously listed here and have since shipped:
- Posterior-aware retrieval ranking → **shipped v1.3.0** (partial; [bayesian_ranking.md](../design/bayesian_ranking.md))
- BFS multi-hop graph retrieval → **shipped v1.3.0** ([bfs_multihop.md](../design/bfs_multihop.md))
- Entity index / NER → **shipped v1.3.0** ([entity_index.md](../design/entity_index.md))
- LLM in the hot path (optional onboard classifier) → **shipped v1.3.0** ([llm_classifier.md](../design/llm_classifier.md))
- BM25F anchor-text retrieval → **shipped v1.7.0**, default-on (#148/#154; +0.6650 NDCG@k uplift on the v0.1 retrieve_uplift fixture)
- HRR primitives + structural lane → **shipped v1.7.0**, default-on as of v2.1 ([feature-hrr-integration.md](../design/feature-hrr-integration.md); source at `src/aelfrice/hrr_index.py`; closes the vocabulary-gap-recovery claim, #154 composition tracker, #437 reproducibility-harness 11/11 at the v2.1.0 cut, 2026-05-08 — see the note above on the current badge)
- Heat-kernel authority scorer → **shipped v1.6.0** (opt-in), default-on as of v2.1 (#154 composition tracker)
- HRR persistence (split-format `.npy` + `.npz` save/load, default-on) → **shipped v3.0** (#553)
- Wonder lifecycle (graph-walk + axes-dispatch + phantom promotion Surfaces A+B) → **shipped v2.0/v3.0** ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella)
- Read-only cross-project federation → **shipped v3.0** (#650 / #655 / #688)
- Eval-harness LLM-judge + Cohen's-κ calibration → **shipped v3.0** (#592 / #600 / #687)
- Type-aware compression A2 bench gate → **shipped v3.0** (#434)
- `query_strategy` stack-r1-r3 default → **shipped v3.0** (#718)
