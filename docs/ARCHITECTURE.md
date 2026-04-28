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
| `retrieval.py` | `retrieve(store, query, token_budget=2000)` — L0 locked + L2.5 entity-index (v1.3+) + L3 BFS multi-hop (v1.3+, default-off) + L1 FTS5 BM25 with Bayesian log-additive reranking (v1.3+). L0 never trimmed. |
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
| `cli.py` | argparse 24-subcommand CLI. Entry: `aelf`. |
| `mcp_server.py` | FastMCP server, 9 tools. `[mcp]` optional extra. |
| `setup.py` | Idempotent install/uninstall of all hooks + statusline. Atomic write via tempfile + `os.replace`. |
| `hook.py` | `aelfrice.hook:main` — process Claude Code spawns on each prompt. Reads stdin, calls `retrieve()`, emits `<aelfrice-memory>` on stdout. Non-blocking. Entry: `aelf-hook`. |
| `slash_commands/` | One markdown file per CLI subcommand surfaced in `/aelf:*`. |

## Data model

**Belief** — `id, content, content_hash, alpha, beta, type, lock_level, locked_at, demotion_pressure, origin, session_id, created_at, last_retrieved_at`.

- `type ∈ {factual, correction, preference, requirement}`
- `lock_level ∈ {none, user}`
- `origin ∈ {user_stated, user_corrected, user_validated, agent_inferred, agent_remembered, document_recent, unknown}` (v1.2+)

**Edge** — `src, dst, type, weight, anchor_text, created_at`. Six edge types with valence multipliers:

| Type | Valence | |
|---|---|---|
| `SUPPORTS` | +1.0 | full positive |
| `CITES` | +0.5 | half positive |
| `DERIVED_FROM` | +0.5 | half positive (turn-to-turn provenance) |
| `RELATES_TO` | +0.3 | weak positive |
| `CONTRADICTS` | -0.5 | half negative |
| `SUPERSEDES` | 0.0 | structural; no propagation |

The research line carried 17 edge types — 12 core (the six above plus `CALLS`, `TESTS`, `IMPLEMENTS`, `TEMPORAL_NEXT`, `CO_CHANGED`, `CONTAINS`, `COMMIT_TOUCHES`) and 5 speculative/causal (`SPECULATES`, `DEPENDS_ON`, `RESOLVES`, `HIBERNATED`, `DERIVED_FROM`). The narrowing to six is deliberate: the speculative/causal set hangs on the deferred `wonder` / multi-axis-uncertainty substrate (see [ROADMAP § Recovery inventory](ROADMAP.md#recovery-inventory)), and the additional core types (`CALLS`, `TESTS`, `IMPLEMENTS`, etc.) come back with the additional onboarding extractors that produce them — not by extending this enum in isolation.

**SQLite tables:** `beliefs`, `beliefs_fts` (virtual, porter unicode61), `edges` PK `(src, dst, type)`, `feedback_history`, `sessions`, `onboard_sessions`, `schema_meta`.

## Bayesian update

`apply_feedback(store, belief_id, valence, source)`:

1. Load belief. Reject zero valence and empty source.
2. **Positive valence:** `α += valence`. Walk outbound `CONTRADICTS` edges; user-locked destinations get `demotion_pressure += 1`. If pressure ≥ 5 (default), demote.
3. **Negative valence:** `β += |valence|`. No pressure walk — the β increment already handles it.
4. Persist atomically.
5. Append a `FeedbackEvent` row. Always.

Walk is 1-hop only. Multi-hop pressure is deferred.

## Retrieval

```
L0: store.list_locked()              always loaded; never trimmed
        ↓
L2.5: entity-index lookup (v1.3+)    NER-extracted entities → exact + stem match;
        ↓                             default-on; disable via [retrieval] entity_index_enabled = false
L3: BFS multi-hop expansion (v1.3+)  edge-weighted graph walk from L0+L2.5 seeds;
        ↓                             default-OFF; enable via [retrieval] bfs_enabled = true
L1: FTS5 BM25 keyword search         limit l1_limit, query escaped;
        ↓                             v1.3+: score = log(bm25) + 0.5*log(posterior_mean)
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

`uv run pytest` (~1,414 tests at v1.3/v1.4, ~15s on Apple Silicon).

## Out of scope through v1.x

These land at v2.0 with evidence (a benchmark, an experiment, a clear case where the existing operations don't suffice):

- HRR / sentence-transformer embeddings
- Cross-project knowledge federation
- Full posterior-driven ranking eval (10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — v2.0.0; the partial Bayesian reranking shipped at v1.3.0)

The following were previously listed here and have since shipped:
- Posterior-aware retrieval ranking → **shipped v1.3.0** (partial; [bayesian_ranking.md](bayesian_ranking.md))
- BFS multi-hop graph retrieval → **shipped v1.3.0** ([bfs_multihop.md](bfs_multihop.md))
- Entity index / NER → **shipped v1.3.0** ([entity_index.md](entity_index.md))
- LLM in the hot path (optional onboard classifier) → **shipped v1.3.0** ([llm_classifier.md](llm_classifier.md))
