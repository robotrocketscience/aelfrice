# Architecture

How aelfrice fits together. Maps directly to source under `src/aelfrice/`.

## Principles

1. **Determinism end to end.** Every retrieval result is bit-identical given the same write log and the same code. Every result traces to named beliefs and named rules. See [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property).
2. **Stdlib + SQLite only.** No vector DB, no embeddings, no LLM in the hot path. The `[mcp]` extra (`fastmcp`) is the only optional runtime dep.
3. **Bayesian, not vibes.** Confidence is `α / (α + β)`. Every update has a closed-form rule. At v1.3.0+ the posterior is combined log-additively with BM25 on the L1 tier — see [LIMITATIONS](LIMITATIONS.md) for what the partial ranking does and doesn't cover.
4. **`apply_feedback` is the central endpoint.** One writer of `(α, β)`. One audit row per successful update.
5. **Locks are user-asserted ground truth.** A user-locked belief short-circuits decay. Contradicting positive feedback accumulates `demotion_pressure`; ≥5 ⇒ auto-demote.

### Enrichment-step boundary

The determinism contract applies to retrieval — every read is reproducible from the inputs. Some write-side operations (LLM-driven sentence classification on the polymorphic onboard path; future research-line capabilities) involve non-deterministic steps. The boundary is explicit:

- Inputs to enrichment (sentence, source, model id + version, prompt template hash) are recorded.
- Outputs (belief type, prior, derived edges) are stored as deterministic content with provenance.
- All retrieval and feedback math downstream of the enriched store is deterministic.

The contract is *deterministic substrate + bounded, audited enrichment layer*, not "no model ever touches the data."

## Modules

Imports are one-directional — modules lower in the table import from higher.

| Module | Responsibility |
|---|---|
| `models.py` | `Belief`, `Edge`, `FeedbackEvent`, `OnboardSession` dataclasses; type / lock / origin constants. No I/O. |
| `scoring.py` | `posterior_mean`, `decay`, `relevance_combiner`. Type half-lives. Lock-floor short-circuit. Decay target: Jeffreys `(0.5, 0.5)`. |
| `store.py` | SQLite WAL + FTS5 + CRUD. `propagate_valence` BFS with broker-confidence attenuation. |
| `retrieval.py` | `retrieve(store, query, token_budget=2000)` — L0 locked + L2.5 entity-index (v1.3+) + L1 FTS5 BM25/BM25F (BM25F default-on since v1.7.0) with Bayesian log-additive reranking (v1.3+) + L3 BFS multi-hop (v1.3+, default-off) over the L0+L2.5+L1 seed set. L0 never trimmed. |
| `feedback.py` | `apply_feedback(store, belief_id, valence, source)` — only Bayesian-update path. Writes `feedback_history`. Drives demotion-pressure + auto-demote. |
| `contradiction.py` | `resolve_contradiction` — picks a winner per precedence, inserts `SUPERSEDES`, writes audit row. Backs `aelf resolve`. |
| `correction.py` | No-LLM heuristic correction detector. |
| `classification.py` | Type priors + regex fallback. Polymorphic onboard state machine. |
| `noise_filter.py` | `is_noise(text, config)` — filters markdown headings, checklist blocks, three-word fragments, license boilerplate. Tunable via `.aelfrice.toml` — see [CONFIG](CONFIG.md). |
| `scanner.py` | `scan_repo` — filesystem + git log + Python AST extractors. Idempotent on `content_hash`. |
| `health.py` | v1.0 regime classifier (`supersede` / `ignore` / `mixed` / `insufficient_data`). Surfaced via `aelf regime`. |
| `auditor.py` | Structural auditor: orphan threads, FTS5 sync, locked contradictions, corpus volume. Backs `aelf health`. Pure read-only. |
| `migrate.py` | One-shot port from the legacy global DB into the per-project DB. Reads source via SQLite `mode=ro`. Backs `aelf migrate`. |
| `doctor.py` | Settings-linter: walks every `command` in `settings.json` and verifies it resolves. Special-cases `bash <script>` wrappers. Backs `aelf doctor`. |
| `lifecycle.py` | Update notifier (PyPI background check), uninstall machinery, archive encryption. |
| `transcript_logger.py` | Hook entry-point for v1.2+ transcript capture. Writes one JSONL line per turn under `<git-common-dir>/aelfrice/transcripts/`. |
| `hook_commit_ingest.py` | `PostToolUse:Bash` hook — ingests commit messages after `git commit`. |
| `hook_search.py` | UserPromptSubmit retrieval helper that records every hit as a `feedback_history` row tagged `source='hook'`. |
| `triple_extractor.py` | Pure-regex `(subject, relation, object)` extraction over six relation families. Used by commit-ingest and transcript-ingest. |
| `context_rebuilder.py` | PreCompact alpha that surfaces aelfrice retrieval before Claude Code summarises. |
| `benchmark.py` | Deterministic 16-belief × 16-query synthetic harness. Frozen `BenchmarkReport`. |
| `cli.py` | argparse multi-subcommand CLI. Entry: `aelf`. Everyday surface in `aelf --help`; full surface (diagnostic, hook, lifecycle verbs) in `aelf --help --advanced`. |
| `mcp_server.py` | FastMCP server, 15 tools (12 v2.0 surface + `aelf_wonder` / `aelf_wonder_persist` / `aelf_wonder_gc` added in v3.0). `[mcp]` optional extra. See [MCP](MCP.md) for the full tool list. |
| `federation.py` | (v3.0+) Read-only peer-DB federation. `load_peer_deps()` parses `knowledge_deps.json`; `open_peer_connection(path)` opens a peer SQLite in `mode=ro&immutable=1`; `ForeignBeliefError` rejects mutations against foreign belief ids at the API surface. See [LIMITATIONS § Sharing, sync, or federation](LIMITATIONS.md). |
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

**Belief** — `id, content, content_hash, alpha, beta, type, lock_level, locked_at, demotion_pressure, origin, session_id, created_at, last_retrieved_at`.

- `type ∈ {factual, correction, preference, requirement}`
- `lock_level ∈ {none, user}`
- `origin ∈ {user_stated, user_corrected, user_validated, agent_inferred, agent_remembered, document_recent, speculative, unknown}` (v1.2+; `speculative` added with the v2.0 wonder substrate for phantom beliefs)
- `scope ∈ {project, global, shared:<name>}` (v3.0+, #688). `project` is the default and local-only; `global` is surfaced to any peer DB that declares this DB in its `knowledge_deps.json`; `shared:<name>` is surfaced only to peers that also list `shared:<name>` as a dep.

**Edge** — `src, dst, type, weight, anchor_text, created_at`. Ten edge types in `EDGE_VALENCE`:

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

**SQLite tables:** `beliefs` (with `scope` column since v3.0), `beliefs_fts` (virtual, porter unicode61), `edges` PK `(src, dst, type)`, `feedback_history`, `sessions`, `onboard_sessions`, `belief_corroborations` (sibling table, v1.5.1+), `ingest_log` (append-only, v1.6+), `belief_versions` + `edge_versions` (per-scope version vectors, v1.5+), `schema_meta`. The `scope` column has an `idx_beliefs_scope` index; both column and index land idempotently via the migration runner.

## Bayesian update

`apply_feedback(store, belief_id, valence, source)`:

1. Load belief. Reject zero valence and empty source.
2. **Positive valence:** `α += valence`. Walk outbound `CONTRADICTS` edges; user-locked destinations get `demotion_pressure += 1`. If pressure ≥ 5 (default), demote.
3. **Negative valence:** `β += |valence|`. No pressure walk — the β increment already handles it.
4. Persist atomically.
5. Append a `FeedbackEvent` row. Always.

Walk is 1-hop only. Multi-hop pressure is deferred.

## Retrieval

L0 (locked beliefs) is the **always-injected pool**: every lock ships on every retrieval, in full, no scoring, no top-K. Lock count is the operator's baseline-context budget knob — if you lock 200 things, every retrieval opens with all 200, by design. Only the non-locked pool (L1/L2.5/L3) is subject to relevance ranking and budget trim.

```
L0: store.list_locked()              always loaded; never trimmed
        ↓
L2.5: entity-index lookup (v1.3+)    NER-extracted entities → exact + stem match;
        ↓                             default-on; disable via [retrieval] entity_index_enabled = false
L1: FTS5 BM25 / BM25F                limit l1_limit, query escaped;
        ↓                             v1.3+: score = log(bm25) + 0.5*log(posterior_mean)
                                      v1.7+: BM25F anchor-augmented sparse matvec, default-on (#148/#154)
L3: BFS multi-hop expansion (v1.3+)  edge-weighted graph walk from L0+L2.5+L1 seed set;
        ↓                             default-OFF; enable via [retrieval] bfs_enabled = true
Dedupe L1+L2.5+L3 against L0 ids
        ↓
Trim from tail until sum(estimated_tokens) ≤ token_budget
```

Token estimate: `(len(content) + 3) // 4`. Empty query: L0 only. L0 always wins overflow.

Spec docs: [entity_index.md](entity_index.md) (L2.5), [bfs_multihop.md](bfs_multihop.md) (L3), [bayesian_ranking.md](bayesian_ranking.md) (L1 Bayesian reranking).

**BFS temporal-coherence caveat:** L3 resolves each hop to the globally latest serial of its target belief. For recall queries this is correct. For audit queries (what did the agent believe at decision-time?) a post-seed supersession can appear mid-chain. The temporal-coherence fix is targeted at v2.0.0 — see [LIMITATIONS § BFS multi-hop temporal coherence](LIMITATIONS.md#bfs-multi-hop-temporal-coherence).

## Onboarding

`scan_repo(store, path)`:

1. **Filesystem walk** over `*.md`, `*.rst`, `*.txt`, `*.adoc` → `factual` / `requirement` candidates.
2. **Git log** → `factual` candidates with file recency (v1.1.0: `belief.created_at` = file's most recent commit, so decay penalises old branches).
3. **Python AST** → function/class names + docstrings → `factual` candidates.

Classification via priors + regex fallback. Idempotent on `content_hash`.

**LLM-Haiku onboard classifier (v1.3+, default-OFF):** `aelf onboard --llm-classify` routes each candidate through Claude Haiku instead of the regex path. Four consent gates enforce the privacy boundary: flag presence, `ANTHROPIC_API_KEY` present, stored sentinel, interactive prompt. `--dry-run` previews candidates without calling the API. Spec: [llm_classifier.md](llm_classifier.md). This is the only path in aelfrice that transmits user content outbound — see [PRIVACY § Optional outbound calls](PRIVACY.md#optional-outbound-calls).

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

Non-blocking contract: every failure path exits 0 with no stdout. A hook problem must never block your prompt.

## v1.2+ hooks

| Hook | Event | Purpose |
|---|---|---|
| `aelf-transcript-logger` | `UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact` | One JSONL line per conversation turn; PreCompact rotates and re-ingests. |
| `aelf-commit-ingest` | `PostToolUse:Bash` | After `git commit`, ingest the commit message via the triple extractor. |
| `aelf-session-start-hook` | `SessionStart` | Inject locked beliefs as `<aelfrice-baseline>` once per session. |
| `aelf-pre-compact-hook` | `PreCompact` | Context rebuilder alpha — surfaces retrieval before Claude Code summarises. |

All opt-in via `aelf setup --<flag>`. All non-blocking.

The `PostToolUseFailure:<tool_name>` event-name namespace inside
`~/.aelfrice/hook-activity.jsonl` is reserved for raw tool-failure
observation produced by a HOME-side hook (tracked separately). See
[hook_activity_schema](hook_activity_schema.md) for the field schema
and the consumer-side dedupe-by-fingerprint warning.

## PreCompact rebuilder (v1.4)

When Claude Code approaches its context limit it fires `PreCompact`. The `aelf-pre-compact-hook` intercepts this event and injects a curated retrieval block before the harness summarises:

```
PreCompact fires
      ↓
aelf-pre-compact-hook reads the last N turns from turns.jsonl
      ↓
rebuild_v14(recent_turns, store, token_budget)
      → L0 locked beliefs (always first)
      → session-scoped beliefs matching recent content
      → BM25+posterior hits against the session tail
      packed to token_budget (default: [rebuilder].token_budget in .aelfrice.toml)
      ↓
emitted as additionalContext — both the aelfrice block
and the harness's own summary land in the new context (augment mode)
```

`aelf rebuild [--transcript PATH] [--n N] [--budget N]` runs the same codepath manually (prints block to stdout). Install via `aelf setup --rebuilder`. Spec: [context_rebuilder.md](context_rebuilder.md). Eval fixture policy: [eval_fixture_policy.md](eval_fixture_policy.md).

## Tests

| Layer | Marker | Coverage |
|---|---|---|
| Unit | default | One property per test. Pyright strict. |
| Property | default | Pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, token-budget invariant, broker-attenuation. |
| Regression | `@pytest.mark.regression` | Cross-module scenarios: retrieval round-trip, feedback loop, onboarding, setup→hook→unsetup, `aelf bench` end-to-end. |

`uv run pytest` (2,700+ tests at v2.0.0).

## Out of scope through v1.x

These remain parked until a benchmark, experiment, or concrete failure mode justifies them:

- Sentence-transformer embeddings (HRR primitives shipped at v1.7.0 as a structural lane, not a learned-embedding lane; v3.0 PHILOSOPHY ratification #605 keeps determinism as the property — no embedding lane planned)
- Multi-writer federation / CRDT primitives. v3.0 ships *read-only* federation (#650 / #655 / #688) — peers open foreign DBs read-only and UNION FTS5 results. The multi-writer extension (#651-#654 CRDT primitives) closed WONTFIX at the v3.0 cut per the #661 ratification.
- Full composition tracker — 10-round MRR uplift, ECE calibration, BM25F × heat-kernel × HRR-structural composition eval (#154). Heat-kernel and HRR-structural defaults flipped on at v2.1; the joint-composition bench gate as such was not run separately, but the #437 reproducibility-harness 11/11 covers the substrate.

The following were previously listed here and have since shipped:
- Posterior-aware retrieval ranking → **shipped v1.3.0** (partial; [bayesian_ranking.md](bayesian_ranking.md))
- BFS multi-hop graph retrieval → **shipped v1.3.0** ([bfs_multihop.md](bfs_multihop.md))
- Entity index / NER → **shipped v1.3.0** ([entity_index.md](entity_index.md))
- LLM in the hot path (optional onboard classifier) → **shipped v1.3.0** ([llm_classifier.md](llm_classifier.md))
- BM25F anchor-text retrieval → **shipped v1.7.0**, default-on (#148/#154; +0.6650 NDCG@k uplift on the v0.1 retrieve_uplift fixture)
- HRR primitives + structural lane → **shipped v1.7.0**, default-on as of v2.1 ([feature-hrr-integration.md](feature-hrr-integration.md); source at `src/aelfrice/hrr_index.py`; closes the vocabulary-gap-recovery claim, #154 composition tracker, #437 reproducibility-harness 11/11)
- Heat-kernel authority scorer → **shipped v1.7.0**, default-on as of v2.1 (#154 composition tracker)
- HRR persistence (split-format `.npy` + `.npz` save/load, default-on) → **shipped v3.0** (#553)
- Wonder lifecycle (graph-walk + axes-dispatch + phantom promotion Surfaces A+B) → **shipped v2.0/v3.0** ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella)
- Read-only cross-project federation → **shipped v3.0** (#650 / #655 / #688)
- Eval-harness LLM-judge + Cohen's-κ calibration → **shipped v3.0** (#592 / #600 / #687)
- Type-aware compression A2 bench gate → **shipped v3.0** (#434)
- `query_strategy` stack-r1-r3 default → **shipped v3.0** (#718)
