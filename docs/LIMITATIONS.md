# Limitations

aelfrice at v1.0 ships the surface but has gaps the v1.x line is closing. Library, CLI, MCP server, and benchmark harness are stable (~530 tests passing). The end-to-end Claude Code injection path and the feedback-into-retrieval loop are partially complete.

## Known issues at v1.0

Tracked openly. The v1.x line picks them up.

### Feedback in retrieval (the big one)

**Posterior `α` / `β` does not currently drive retrieval ranking.** `store.search_beliefs` orders L1 hits by `bm25(beliefs_fts)` only. So `apply_feedback` updates the math (and the audit log) but doesn't yet move what the agent sees. The benchmark harness ships at v1.0 as the measurement instrument; a v1.x retrieval upgrade that consumes the posterior is the precondition for using it to claim feedback drives accuracy.

### Hook layer

- **`aelf --version` does not exist.** Argparse error today. Trivial fix; hasn't landed.
- **The `aelf-hook` UserPromptSubmit hook may return empty results despite a populated FTS5 index.** Hook fires; no `<aelfrice-memory>` block appears. Workaround: drop to CLI (`aelf search`, `aelf locked`).
- **The hook now calls `aelfrice.retrieval.retrieve()` directly** — that's a v0.7→v1.0 upgrade from the inline FTS5 fallback that earlier versions used.

### Onboarding

- **Noise filtering is not wired into the synchronous onboard path.** Markdown headings, checklist items, three-word fragments, and license-header boilerplate currently surface as candidate beliefs. Use the polymorphic MCP onboard (host-LLM classification) for higher signal until the filters land.
- **No git-recency weighting.** Pre-migration content from old branches surfaces as first-class beliefs alongside current code.
- **Onboard performance.** Believed batched in v0.6+ via WAL + transaction batching, but not yet measured at scale on a 50k-LOC project for v1.0 sign-off.

### Feedback semantics

- **Contradictions are detected but not auto-resolved.** Search output prints `WARNING: CONTRADICTS [X] vs [Y]` and continues. There is no default tie-breaker (timestamp, source-tier, git-blame recency). Until v1.x: review contradictions manually with `aelf locked --pressured`.
- **No promotion path from `agent_inferred` to `user_validated`.** Beliefs created during onboard are tagged `agent_inferred` and stay that way unless explicitly locked. The math is sound at the `(α, β)` level; the tier-promotion mechanism isn't in the data flow yet.

### Project identity

- **The default DB is keyed by `SHA256(cwd)`.** Worktree, directory move, fresh clone — each produces a different orphan DB. The local-only promise breaks the moment you move the directory. Workaround today: pin with `AELFRICE_DB=/abs/path/.aelfrice.db`. Long-term plan: `.git/aelfrice/memory.db` (in-repo storage) plus `.aelfrice.toml` (cross-machine identity).

### Cosmetic

- **`edges` not yet renamed to `threads`** in user-facing CLI output and MCP tool descriptions. Decision recorded; not implemented.
- **`aelf health` and `aelf status` are aliased.** They should be split: `status` = counts snapshot, `health` = a real graph auditor (orphan edges, isolated clusters, FTS5 sync, locked-belief contradictions, decay anomalies).

### Harness conflict

- **Claude Code's built-in auto-memory directive competes for write authority with the aelfrice MCP.** When the harness instructs the model to "save a memory," the write goes to a flat `.md` file rather than to a belief. Result: the graph stops growing after explicit `onboard` / `lock` / `feedback` calls. This is a CLAUDE.md / harness-level fix, not a code change in aelfrice itself.

## Deferred to v1.x (with evidence required)

These existed in the legacy v2.0 codebase. Each will be reintroduced with a benchmark, an experiment, or a clear use case justifying inclusion.

`wonder`, `reason`, `core`, `unlock`, `delete`, `confirm` commands. Obsidian export/sync. Cross-project memory federation. Vector embeddings, ANN, semantic similarity. HRR / structural binding. Multi-hop graph retrieval. Snapshot / timeline / evolution / diff tools.

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
