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

- **`aelf --version` does not exist.** Argparse error today. ~5 LOC + unit test.
- **The `aelf-hook` UserPromptSubmit hook may return empty results despite a populated FTS5 index.** Hook fires; no `<aelfrice-memory>` block appears. Workaround: drop to CLI.
- **Hook bypasses `aelfrice.retrieval.retrieve()` and does not record retrievals as feedback events.** Highest-impact gap. The hook uses an inline FTS5 shim, skipping L0 locked-belief auto-load and feedback-history logging. The "feedback-driven learning" claim depends on this loop closing. v1.0.1 introduces `src/aelfrice/hook_search.py` (`search_for_prompt` + `record_retrieval`) and replaces the shim.

### Onboarding — v1.0.1 + v1.1.0

- **Noise filtering not wired into the synchronous onboard path.** Markdown headings, checklist items, three-word fragments, license-header boilerplate currently surface as candidate beliefs. Use the polymorphic MCP onboard (host-LLM classification) for higher signal. Filters land in v1.0.1 as a separate `noise_filter` module.
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

### Harness conflict — out of scope

- **Claude Code's built-in auto-memory directive competes for write authority with the aelfrice MCP.** Writes go to flat `.md` files instead of beliefs. CLAUDE.md / harness-level fix, not an aelfrice code change. Tracked separately.

## Deferred to v1.3 / v2.0 (with evidence required)

These existed in the legacy v2.0 codebase. Each will be reintroduced with a benchmark, an experiment, or a clear use case justifying inclusion.

- **v1.3 retrieval wave:** HRR / vocabulary bridging, multi-hop graph retrieval, LLM-Haiku classification (~$0.005/session), cross-project knowledge federation (#109).
- **v2.0:** full academic benchmark suite (LoCoMo, MAB, LongMemEval, StructMemEval, AmaBench) + the `wonder`, `reason`, `core`, `unlock`, `delete`, `confirm` commands. Snapshot / timeline / evolution / diff tools. Obsidian export/sync. Vector embeddings and ANN are NOT planned for v2.0 — `aelfrice` stays SQLite + FTS5 by default at every milestone in this list.

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
