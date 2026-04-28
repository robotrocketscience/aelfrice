# Architecture

How aelfrice fits together. Maps directly to source under `src/aelfrice/`.

## Principles

1. **Determinism end to end.** Every retrieval result is bit-identical given the same write log and the same code. Every result traces to named beliefs and named rules. See [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property).
2. **Stdlib + SQLite only.** No vector DB, no embeddings, no LLM in the hot path. The `[mcp]` extra (`fastmcp`) is the only optional runtime dep.
3. **Bayesian, not vibes.** Confidence is `α / (α + β)`. Every update has a closed-form rule. (At v1.0–v1.2 the score does not yet drive ranking — see [LIMITATIONS](LIMITATIONS.md).)
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
| `retrieval.py` | `retrieve(store, query, token_budget=2000)` — L0 locked + L1 FTS5 BM25. L0 never trimmed. |
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
| `cli.py` | argparse 22-subcommand CLI. Entry: `aelf`. |
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
L0: store.list_locked()         always loaded; never trimmed
        ↓
L1: FTS5 BM25 keyword search    limit l1_limit, query escaped
        ↓
Dedupe L1 against L0 ids
        ↓
Trim L1 from tail until sum(estimated_tokens) ≤ token_budget
```

Token estimate: `(len(content) + 3) // 4`. Empty query: L0 only. L0 always wins overflow.

The v1.3.0 retrieval wave inserts an L2.5 entity-index tier between L0 and L1 — spec lives at [entity_index.md](entity_index.md).

## Onboarding

`scan_repo(store, path)`:

1. **Filesystem walk** over `*.md`, `*.rst`, `*.txt`, `*.adoc` → `factual` / `requirement` candidates.
2. **Git log** → `factual` candidates with file recency (v1.1.0: `belief.created_at` = file's most recent commit, so decay penalises old branches).
3. **Python AST** → function/class names + docstrings → `factual` candidates.

Classification via priors + regex fallback. Idempotent on `content_hash`.

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

## Tests

| Layer | Marker | Coverage |
|---|---|---|
| Unit | default | One property per test. Pyright strict. |
| Property | default | Pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, token-budget invariant, broker-attenuation. |
| Regression | `@pytest.mark.regression` | Cross-module scenarios: retrieval round-trip, feedback loop, onboarding, setup→hook→unsetup, `aelf bench` end-to-end. |

`uv run pytest` (~1,150 tests at v1.2, ~15s on Apple Silicon).

## Out of scope through v1.x

These land at v2.0 with evidence (a benchmark, an experiment, a clear case where the existing operations don't suffice):

- Posterior-aware retrieval ranking (gated on the v1.3 retrieval wave)
- HRR / sentence-transformer embeddings
- BFS multi-hop graph retrieval
- Entity index / NER
- LLM in the hot path
- Cross-project knowledge federation
