# ARCHITECTURE

Module map, data flow, design decisions. Maps directly to source under `src/aelfrice/`.

## Principles

1. **Determinism end to end.** Every retrieval result is bit-identical given the same write log and the same code. Every result traces to named beliefs and named rules. The four-property commitment (bit-level reproducibility, named-rule traceability, write-log historical reconstruction, non-technical audit) is documented in [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property). New components must preserve it; non-deterministic steps either run as one-time enrichment whose outputs are stored as deterministic content (see § Enrichment-step boundary below) or are clearly marked as opt-in non-deterministic paths.
2. **Stdlib + SQLite only.** No vector DB, no embeddings, no cloud, no LLM in the hot path. The `[mcp]` extra is the only non-stdlib runtime dependency, and it's optional.
3. **Bayesian, not vibes.** Confidence is `α / (α + β)`. Every update has a closed-form rule. (Note: at v1.0 this score does not yet drive retrieval ranking — see [LIMITATIONS](LIMITATIONS.md#known-issues-at-v10).)
4. **`apply_feedback` is the central endpoint.** One writer of `(α, β)`. One audit row per successful update.
5. **Locks are user-asserted ground truth.** A `user`-locked belief short-circuits decay (lock floor) and bypasses L1 budgeting on retrieval. Contradicting positive feedback accumulates `demotion_pressure`; ≥ 5 ⇒ auto-demote.
6. **Clean by construction, not clean by audit.** Filtering is a tripwire, not a gate.

### Enrichment-step boundary

The determinism contract applies to retrieval — every read is reproducible from the inputs. Some write-side operations (LLM-driven sentence classification on the v1.3 polymorphic onboard path; future `wonder` / `reason` capabilities in v2.0) involve non-deterministic upstream steps. The boundary is explicit:

- **Inputs to enrichment** (raw sentence, source path, host-LLM model id and version, prompt template hash) are recorded deterministically.
- **Outputs of enrichment** (assigned belief type, type prior, derived edges) are stored as deterministic content with provenance: classifier version, rule-set hash, timestamp.
- **All retrieval and feedback math downstream** of the enriched store is deterministic. Replaying retrieval against the same enriched corpus produces bit-identical results.

The contract is *deterministic substrate + bounded, audited enrichment layer*, not "no model ever touches the data." A reviewer can identify, for any belief, exactly which step was bounded-non-deterministic and inspect the model id, prompt, and output. Replay-from-raw-text reproducibility requires re-running the enrichment with the same model version; replay-from-enriched-corpus reproducibility is unconditional.

## Modules

One-directional imports — lower in the table imports from higher.

| Module | Responsibility |
|---|---|
| `models.py` | `Belief`, `Edge`, `FeedbackEvent`, `OnboardSession` dataclasses; belief / edge / lock / state constants. No I/O. |
| `scoring.py` | `posterior_mean`, `decay`, `relevance_combiner`. Type-specific half-lives (factual 14d / requirement 30d / preference 12w / correction 24w). Lock-floor short-circuit. Decay target: Jeffreys prior `(0.5, 0.5)`. |
| `store.py` | SQLite WAL + FTS5 + CRUD. `propagate_valence` BFS with broker-confidence attenuation. |
| `retrieval.py` | `retrieve(store, query, token_budget=2000)` — L0 locked auto-load + L1 FTS5 BM25. L0 never trimmed. |
| `feedback.py` | `apply_feedback(store, belief_id, valence, source)` — the only Bayesian-update path. Writes `feedback_history`. Drives demotion-pressure increment + auto-demote. |
| `correction.py` | No-LLM heuristic correction detector. |
| `classification.py` | `TYPE_PRIORS` + regex fallback. Polymorphic onboard state machine. |
| `scanner.py` | `scan_repo` — filesystem + git log + Python AST extractors, classification, persistence. Idempotent against `content_hash`. |
| `health.py` | Regime classifier (`insufficient_data` / `early-onboarding` / `steady` / `lock-heavy` / `over-confident`) over confidence, mass, lock density, edge density. |
| `benchmark.py` | `seed_corpus(store)` + `run_benchmark(store, *, aelfrice_version, top_k=5)` — deterministic 16-belief × 16-query synthetic harness. Frozen `BenchmarkReport` with `hit_at_1` / `hit_at_3` / `hit_at_5` / `mrr` + `p50_latency_ms` / `p99_latency_ms`. |
| `cli.py` | argparse 11-subcommand CLI. DB resolves from `$AELFRICE_DB` or `~/.aelfrice/memory.db`. Entry: `aelf`. |
| `mcp_server.py` | FastMCP server, 8 tools. `[mcp]` optional extra. |
| `setup.py` | Idempotent install/uninstall of the Claude Code `UserPromptSubmit` hook. Atomic write via tempfile + `os.replace`. |
| `hook.py` | `aelfrice.hook:main` — process Claude Code spawns when the hook fires. Reads JSON from stdin, calls `retrieve()`, emits `<aelfrice-memory>...</aelfrice-memory>` on stdout. Non-blocking by contract. Entry: `aelf-hook`. |
| `slash_commands/` | 11 markdown files, 1:1 with CLI subcommands. |

## Data model

**Belief** — `id, content, content_hash, alpha, beta, type, lock_level, locked_at, demotion_pressure, created_at, last_retrieved_at`. `type ∈ {factual, correction, preference, requirement}`. `lock_level ∈ {none, user}`.

**Edge** — `src, dst, type, weight`. Five types with valence multipliers:

| Type | Mult | |
|---|---|---|
| `SUPPORTS` | +1.0 | full positive |
| `CITES` | +0.5 | half positive |
| `RELATES_TO` | +0.3 | weak positive |
| `CONTRADICTS` | -0.5 | half negative |
| `SUPERSEDES` | 0.0 | structural; no propagation |

**SQLite tables:** `beliefs`, `beliefs_fts` (FTS5 virtual, porter unicode61), `edges` PK `(src, dst, type)`, `feedback_history`, `onboard_sessions`, `schema_meta`.

## Bayesian update path

`apply_feedback(store, belief_id, valence, source)`:

1. Load belief. Reject zero valence; reject empty source.
2. **Positive valence:** `α += valence`. Then walk source's outbound `CONTRADICTS` edges; for each `dst` that is `user`-locked, `dst.demotion_pressure += 1`. If pressure ≥ `DEMOTION_THRESHOLD` (default 5), demote (`lock_level → none`, `locked_at → None`, `demotion_pressure → 0`).
3. **Negative valence:** `β += |valence|`. No pressure walk — a negative signal weakens the contradictor itself, which the β increment already handles.
4. Persist atomically.
5. Append a `FeedbackEvent` row. Always.

Walk is **1-hop only**. Multi-hop pressure is deferred to v1.x.

> **v1.0 note:** updated `(α, β)` does not currently affect retrieval ranking. `store.search_beliefs` orders L1 hits by `bm25(beliefs_fts)`. The v1.x roadmap consumes the posterior in ranking.

## Retrieval

```
L0: store.list_locked()         always loaded; never trimmed
        ↓
L1: FTS5 BM25 keyword search    limit l1_limit, query escaped
        ↓
Dedupe L1 against L0 ids
        ↓
Trim L1 from tail until sum(estimated_tokens) ≤ token_budget
```

Token estimate: `(len(content) + 3) // 4`. Empty query: L0 only. L0 always wins overflow.

## Onboarding

`scan_repo(store, path)`:

1. **Filesystem walk** over `*.md`, `*.rst`, `*.txt`, `*.adoc` → `factual` / `requirement` candidates.
2. **Git log** → `factual` candidates and edges to touched-file beliefs.
3. **Python AST** → function/class names + docstrings → `factual` candidates.

Classification via `TYPE_PRIORS` + regex fallback. Idempotent: `content_hash` dedupe.

## Claude Code hook

```
~/.claude/settings.json  hooks.UserPromptSubmit: [{command: "aelf-hook"}]
                                  ↓ written by aelf setup
                          Claude Code (LLM)
                                  ↓ JSON payload on stdin
                          aelf-hook subprocess (aelfrice.hook:main)
                                  ↓ retrieve(store, prompt)
                          ~/.aelfrice/memory.db
                                  ↓ <aelfrice-memory> on stdout
                          Claude Code injects above prompt
```

Non-blocking contract: every failure path exits 0 with no stdout.

## Benchmark harness

`aelfrice.benchmark` ships a deterministic 16-belief × 16-query synthetic corpus.

```bash
aelf bench                 # in-memory store, fresh seed each run
aelf bench --db PATH       # against an existing DB
aelf bench --top-k 5       # override hit-depth
```

Output is a single JSON document with `BenchmarkReport` fields. Reproducible across runs against fresh in-memory stores. The harness is the **measurement instrument**, not yet a proof of the feedback claim — see the v1.0 note above.

## Tests

| Layer | Marker | Coverage |
|---|---|---|
| Unit | (default) | One property per test. Pyright strict. ~5s suite timeout. |
| Property | (default) | Pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, token-budget invariant, broker-attenuation. |
| Regression | `@pytest.mark.regression` | Cross-module scenarios: retrieval round-trip, feedback loop, onboarding, setup→hook→unsetup, `aelf bench` end-to-end. |

`uv run pytest` (~530 tests, ~7s on Apple Silicon). `uv run pytest -m regression` for integration only.

## Out of scope through v1.0 (lands in v1.x with evidence)

Posterior-aware retrieval ranking. HRR retrieval. BFS multi-hop graph retrieval. Entity index/NER. LLM in the hot path. Sentence-transformer embeddings. Multi-source provenance tagging. Bitemporal `event_time`. Rigor-tier metadata. Cross-project knowledge federation.
