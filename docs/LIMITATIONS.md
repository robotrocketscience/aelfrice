# Limitations

aelfrice at v1.0 ships the surface but has gaps the v1.x line is closing. Library, CLI, MCP server, and benchmark harness are stable (~530 tests passing). The end-to-end Claude Code injection path and the feedback-into-retrieval loop are partially complete.

## Known issues at v1.0

Tracked openly. Each item is mapped to its target version below.

| Bucket | Theme | Lands |
|---|---|---|
| **v1.0.1** | launch fix-up — hook layer + onboard noise + regression baseline | patch |
| **v1.1.0** | project identity + onboard behavior + cosmetic surface | minor |
| **v1.2.0** | commit-ingest hook + hook perf + seed files | minor |
| **v1.3** | retrieval wave (HRR + LLM classification + cross-project federation) | minor, 6-9 wks |
| **v2.0** | full benchmark suite (LoCoMo, MAB, LongMemEval) + feature parity with the legacy v3 codebase | major |

### Feedback in retrieval (the big one) — v1.3

**Posterior `α` / `β` does not currently drive retrieval ranking.** `store.search_beliefs` orders L1 hits by `bm25(beliefs_fts)` only. So `apply_feedback` updates the math (and the audit log) but doesn't yet move what the agent sees. The synthetic benchmark harness ships at v1.0 as the measurement instrument; the v1.3 retrieval wave wires the posterior into ranking, which is the precondition for claiming feedback drives accuracy.

### Hook layer — v1.0.1

- ✅ **`aelf --version` (v1.0.1).** Prints `aelf <__version__>` and exits 0. Short-circuits the required-subcommand check via argparse's standard version action.
- **The `aelf-hook` UserPromptSubmit hook may return empty results despite a populated FTS5 index.** Hook fires; no `<aelfrice-memory>` block appears. Workaround: drop to CLI.
- ✅ **Hook records retrievals as feedback events (v1.0.1).** The `aelf-hook` UserPromptSubmit entry-point now routes through `aelfrice.hook_search.search_for_prompt`, which wraps `aelfrice.retrieval.retrieve()` and writes one `feedback_history` row per returned belief, tagged `source='hook'` with valence `HOOK_RETRIEVAL_VALENCE` (0.1) and `propagate=False` so implicit retrieval-time exposure does not pressure-walk locked beliefs. The "feedback updates the math" loop is closed on the retrieval side; ranking still uses BM25 only at v1.0.1 (the v1.3 retrieval wave wires posterior into ranking).

### Onboarding — v1.0.1 + v1.1.0

- ✅ **Noise filter wired into synchronous onboard (v1.0.1).** New `aelfrice.noise_filter.is_noise(text, config)` predicate runs on every candidate before classification. Drops markdown heading blocks, checklist blocks, three-word fragments, and license-header boilerplate (seven canonical signatures). `ScanResult.skipped_noise` reports how many candidates were filtered per scan. A paragraph that *mixes* heading-or-checklist with prose is preserved — only pure structural runs are filtered. Power-user configurable via a single `.aelfrice.toml` file at the project root (or any ancestor): toggle individual categories, override the fragment threshold, or add `exclude_words` (whole-word, case-insensitive — "jso" drops "jso" but not "json") and `exclude_phrases` (literal substring, case-insensitive). No regex anywhere in the user-facing schema.
- **Onboard performance.** Believed batched in v0.6+ via WAL + transaction batching; v1.0.1 adds a regression benchmark verifying < 60s on a 50k-LOC project.
- **No git-recency weighting.** Pre-migration content from old branches surfaces as first-class beliefs. v1.1.0.
- **No promotion path from `agent_inferred` to `user_validated`.** Beliefs from onboard stay `agent_inferred` unless explicitly locked. v1.1.0.

### Feedback semantics — v1.0.1

- **Contradictions detected but not auto-resolved.** Search output prints `WARNING: CONTRADICTS [X] vs [Y]` and continues. v1.0.1 adds a default tie-breaker (`user_stated > user_corrected > document_recent > agent_inferred`, then ISO timestamp) that auto-supersedes the loser. Until then: review with `aelf locked --pressured`.

### Project identity — v1.1.0

- **The default DB is keyed by `SHA256(cwd)`.** Worktree, directory move, fresh clone — each produces a different orphan DB. Workaround today: pin with `AELFRICE_DB=/abs/path/.aelfrice.db`. v1.1.0 lands `.git/aelfrice/memory.db` (in-repo storage) and `.aelfrice.toml` (cross-machine identity), plus orphan-DB cleanup tooling and worktree concurrency tests.

### Cosmetic — v1.1.0

- **`edges` not yet renamed to `threads`** in user-facing CLI output and MCP tool descriptions. Decision recorded; not implemented.
- **`aelf health` and `aelf status` are aliased.** v1.1.0 splits them: `status` = counts snapshot, `health` = real graph auditor (orphan edges, isolated clusters, FTS5 sync, locked-belief contradictions, decay anomalies).

### Auto-capture — v1.2.0

- **No commit-ingest PostToolUse hook.** Beliefs accumulate only on explicit `onboard` / `lock` / `feedback`. v1.2.0 adds an automatic capture path so the graph grows during normal sessions.
- **Seed files for git-tracked knowledge bootstrapping.** `.aelfrice/seed.md` committed to a repo, auto-ingested on first onboard. v1.2.0.

### Harness conflict — Claude Code auto-memory write path

**Behavior.** When `aelfrice` is installed alongside Claude Code's built-in file-based auto-memory system, the harness directive routes any "save a memory" intent to its own file-based store (under `.claude/projects/.../memory/*.md` plus a `MEMORY.md` index), not to the `aelfrice` MCP server. The MCP server stays connected and remains queryable for retrieval, but it does not receive new beliefs from normal session activity. New beliefs only enter the `aelfrice` store via explicit tool calls (`aelf remember`, `aelf onboard`, MCP `aelf:remember`) or bulk import.

**Why this is a limitation.** The README's central claim is that `apply_feedback` is the endpoint that makes `aelfrice` distinct from plain RAG: a memory which actually applies feedback should outperform one that doesn't. If the MCP receives no new beliefs during a conversation, then `apply_feedback` is firing against a snapshot written at install time (or at the most recent explicit `aelf onboard` / `aelf remember` call), not against beliefs the agent forms during current work. The feedback loop is intact mathematically but starved of fresh inputs.

**v1.0.1 partial mitigation.** v1.0.1 closes the retrieval-side of the loop: the `UserPromptSubmit` hook records every retrieval as a `feedback_history` row tagged `source='hook'`, and `apply_feedback` moves posteriors based on actual hook-driven retrievals. This means even without a new write path, beliefs already in the store are exercised by feedback during normal use. The write path itself remains gated by the harness directive at v1.0.1.

**Workaround today.** To make the `aelfrice` MCP the canonical write path, edit your `~/.claude/CLAUDE.md` to remove or rephrase the auto-memory harness directive, and rely on `aelf remember` (CLI) or the MCP `aelf:remember` tool for new beliefs. This is a user-side configuration change; `aelfrice` does not attempt to override the harness in code.

**Tracked.** v1.2 will publish `docs/HARNESS_INTEGRATION.md` with a documented procedure for users who want the MCP to be canonical without manually editing `CLAUDE.md`.

## Deferred to v1.3 / v2.0 (with evidence required)

These existed in the legacy v2.0 codebase. Each will be reintroduced with a benchmark, an experiment, or a clear use case justifying inclusion.

- **v1.3 retrieval wave:** HRR / vocabulary bridging, multi-hop graph retrieval, LLM-Haiku classification (~$0.005/session), cross-project knowledge federation (#109).
- **v2.0:** full academic benchmark suite (LoCoMo, MAB, LongMemEval, StructMemEval, AmaBench) + the `wonder`, `reason`, `core`, `unlock`, `delete`, `confirm` commands. Snapshot / timeline / evolution / diff tools. Obsidian export/sync. Vector embeddings and ANN are NOT planned for v2.0 — `aelfrice` stays SQLite + FTS5 by default at every milestone in this list.

## Out of scope by design

Some properties are **not** v1.x gaps to close — they are scope choices that follow from aelfrice's architectural commitments. Listed here so users evaluating the project know what it is *not* trying to be.

### Multi-session aggregation / counting / fuzzy semantic recall

aelfrice's retrieval is BM25 lexical plus a typed graph walk. This is the right substrate for known-item search over behavioural directives — the core design target — and it has known limitations on aggregative or counting queries spanning many sessions ("how many times did the user mention X across last quarter?", "summarise all my preferences about testing"). On benchmarks like LongMemEval multi-session, embedding-based systems will outperform aelfrice on this query category, and that is a real performance gap on those queries.

The principled response is **not** to add embeddings as a fallback retrieval path. Doing so would break the determinism contract documented in [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property): one non-deterministic retrieval step destroys bit-level reproducibility, named-rule traceability, and write-log historical reconstruction for every result the system returns, not just the embedding-routed ones. The trade is structurally bad — the deterministic guarantees are the property that makes the entire system valuable for high-stakes deployment, and they hold compositionally only.

The principled response **is** one of:

1. **Improve the lexical layer for aggregative cases.** Detect aggregative query intent at Layer 0 (structural query analysis) and route to a specialised handler that operates over the deterministic store: SQL aggregations over `feedback_history`, scoped graph walks, time-bucketed COUNT queries.
2. **Document the scope choice.** This section. aelfrice is for behavioural directive recall, lock enforcement, and correction memory — not for general-purpose long-context QA over conversational history.
3. **Pair aelfrice with a separate tool when the use case demands fuzzy semantic recall.** The two have different trust profiles and different deployment stories; pairing them is fine, blending them inside a single retrieval pipeline is not.

### General-purpose long-context QA

aelfrice is not optimised for "answer arbitrary questions over a large conversation history." That is the LLM-with-RAG-and-summary-buffer task. aelfrice is optimised for *the agent doesn't forget the rule you gave it*. These are different problems with different success criteria; separating them is the cleaner architectural stance.

### Probabilistic retrieval-relevance

Retrieval relevance is computed deterministically from BM25 scores plus typed-edge weights. There is no learned re-ranker, no neural relevance model, and no fine-tuning path. Relevance quality is improved by: better tokenisation, better Layer 0 query analysis, better edge inference at write time. None of these introduce non-determinism into the retrieval path.

## Surface limits at v1.0

- **Retrieval is keyword only.** No stemming beyond porter+unicode61. No synonym expansion. "deploy" won't surface "publish to prod" without a tokenizable substring overlap.
- **One DB at a time.** No multi-project query. Use `AELFRICE_DB` to scope per-project.
- **Onboarding is shallow.** Three sources: prose, git log, Python AST. JS/TS/Rust/Go AST parsing not yet wired.
- **Classifier is regex-based on the CLI path.** Only the polymorphic MCP onboard uses host-LLM classification.
- **No bulk operations.** No batch lock, no `delete <pattern>`, no merge.
- **No edit.** Wrong belief → insert corrected one with `SUPERSEDES` edge; original stays.
- **No graph viz.** Inspect with `sqlite3 ~/.aelfrice/memory.db`.

## Performance caveats

- **`aelf bench` ships, but it's a small harness** (16 beliefs × 16 queries). Useful for regression on retrieval mechanics; not yet a real-world workload.
- **FTS5 ranking is BM25 only.** Confidence does not currently weight retrieval order — see the "feedback in retrieval" gap above.
- **Valence propagation walks BFS** to 3 hops, threshold 0.05. Cheap on thousands of beliefs; not measured at much larger scale.

## Sharp edges

- **Locks are stronger than you may expect.** Fresh lock = `(α, β) = (9.0, 0.5)`. Five contradictions required to auto-demote. If you lock a wrong rule and only correct it once or twice, the lock keeps winning. `aelf demote <id>` drops it immediately.
- **CONTRADICTS edges drive demotion pressure.** If your graph has no CONTRADICTS edges (the regex classifier rarely produces them), pressure won't accumulate — only manual contradicting feedback does.
- **Jeffreys prior reads as 0.5.** A new belief with no feedback reports posterior mean exactly `0.5` — that means "no evidence yet," not "coin-flip true."
- **`aelf onboard` is non-incremental on duplicates.** Re-runs are idempotent; existing beliefs aren't re-scored or refreshed.

## Compatibility

- **Python 3.12+ only.** Uses `Final`, `Self`, structural pattern matching, PEP 695.
- **No Windows-native testing.** Should work; not exercised on every release.
- **uv recommended; pip works.**

## Where the list grows

This doc updates each milestone. File issues tagged `limitations` for additions.
