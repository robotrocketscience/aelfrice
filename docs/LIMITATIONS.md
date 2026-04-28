# Limitations

aelfrice at v1.0 ships the surface but has gaps the v1.x line is closing. Library, CLI, MCP server, and benchmark harness are stable (~530 tests passing at v1.0; ~1090 at v1.1.0). The end-to-end Claude Code injection path and the feedback-into-retrieval loop are partially complete.

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

- ✅ **Noise filter wired into synchronous onboard (v1.0.1).** New `aelfrice.noise_filter.is_noise(text, config)` predicate runs on every candidate before classification. Drops markdown heading blocks, checklist blocks, three-word fragments, and license-header boilerplate (seven canonical signatures). `ScanResult.skipped_noise` reports how many candidates were filtered per scan. A paragraph that *mixes* heading-or-checklist with prose is preserved — only pure structural runs are filtered. Power-user configurable via a single `.aelfrice.toml` file at the project root (or any ancestor) — full schema, worked examples, and what each setting affects in [CONFIG.md](CONFIG.md). No regex anywhere in the user-facing schema.
- ✅ **Onboard performance regression test (v1.0.1).** New `tests/regression/test_onboard_perf_50k_loc.py` generates a synthetic ~55k-LOC project (250 .py files of 200 lines + 60 .md/.rst/.txt docs of 50 lines + 10 `__init__.py` files) and asserts `scan_repo` finishes in under 60s wall clock. Marked `regression` so the default fast suite runs it; per-test `timeout(120)` overrides the default 5s pytest timeout. Held against the `:memory:` store so disk-write contention does not skew the measurement. Current measured time: ~0.8s on Apple Silicon — the budget is a regression alarm, not a target.
- ✅ **Onboard git-recency weighting (v1.1.0, [#94](https://github.com/robotrocketscience/aelfrice/issues/94)).** Scanner records each source file's most-recent git commit date as the belief's `created_at`, so the existing decay mechanism penalises pre-migration content from old branches without a separate ranking pass. One `git log --name-only --pretty=format:%aI` call per scan. PR [#103](https://github.com/robotrocketscience/aelfrice/pull/103).
- **No promotion path from `agent_inferred` to `user_validated`.** Beliefs from onboard stay `agent_inferred` unless explicitly locked. Designed in v1.1.0 ([promotion_path.md](promotion_path.md), [#95](https://github.com/robotrocketscience/aelfrice/issues/95)); implemented in v1.2.0.

### Feedback semantics — v1.0.1

- ✅ **Contradiction tie-breaker (v1.0.1).** New `aelfrice.contradiction` module: `resolve_contradiction(store, a_id, b_id)` picks a winner per precedence — `user_stated` (lock_level=user) > `user_corrected` (type=correction) > `document_recent` (everything else). Ties within a class break by `created_at` recency; final tiebreak by belief id (deterministic). Inserts a SUPERSEDES edge from winner to loser and writes one `feedback_history` row tagged `source='contradiction_tiebreaker:<rule>'`. `aelf resolve` (CLI) sweeps all unresolved CONTRADICTS edges in the store. Idempotent — re-running the command on a resolved store does no edge work but writes no audit rows either. The fourth class named in the original spec (`agent_inferred`) requires a `Belief.origin` field not in the v1.0 schema; in practice no v1.0 path produces beliefs that map to it, so v1.0.1 collapses to three classes. v1.1.0 adds the `origin` field alongside project-identity work and the four-class split becomes faithful.

### Project identity — v1.1.0

- ✅ **Per-project DB resolution (v1.1.0, [#88](https://github.com/robotrocketscience/aelfrice/issues/88)).** v1.0.x shipped a single global DB at `~/.aelfrice/memory.db` shared across every project on the machine — onboarding repo A and repo B writes both projects' beliefs into one file. v1.1.0 introduces a resolution chain in `cli.db_path()`: `$AELFRICE_DB` (override, honoured even inside a git repo) → `<git-common-dir>/aelfrice/memory.db` (when `cwd` is inside a git work-tree; resolved via `git rev-parse --path-format=absolute --git-common-dir`, so worktrees of one repo share one DB) → `~/.aelfrice/memory.db` (legacy global fallback for non-git dirs). `.git/` is not git-tracked, so the brain graph never crosses the git boundary. `aelf migrate` ([#93](https://github.com/robotrocketscience/aelfrice/issues/93)) ports beliefs from the legacy global store into per-project stores. Fresh clones intentionally start with an empty store — the brain graph stays on the machine it was written on (see [§ Sharing or sync of brain-graph content](#sharing-or-sync-of-brain-graph-content)); bootstrap a clone with `aelf onboard .`.

### Cosmetic — v1.1.0

- ✅ **`edges` → `threads` user-facing rename (v1.1.0, [#92](https://github.com/robotrocketscience/aelfrice/issues/92)).** All user-facing surfaces (CLI output labels, slash command descriptions, COMMANDS.md, MCP.md prose) now use `threads`. The internal SQLite schema, the `Edge` Python dataclass, and the `EDGE_*` type constants are unchanged. The MCP `aelf:stats` tool emits **both** `edges` and `threads` keys with the same integer value for the v1.1.0 deprecation window; `edges` is removed in v1.2.0. Same for `aelf:health.features.edge_per_belief` / `thread_per_belief`. Clients should migrate to `threads` now. The `auditor.CHECK_ORPHAN_EDGES` constant is kept as a deprecated alias of `CHECK_ORPHAN_THREADS` for v1.0 importer compatibility; removed in v1.2.0.
- ✅ **`aelf health` rewritten as graph auditor (v1.1.0, [#100](https://github.com/robotrocketscience/aelfrice/pull/100)).** `health` now runs a real auditor (orphan threads, FTS5 sync, locked-belief contradictions; corpus-volume warning added in [#116](https://github.com/robotrocketscience/aelfrice/issues/116)) and exits 1 on structural failures so CI can gate on it. The v1.0 regime classifier is preserved as `aelf regime`. `aelf status` is an alias for `aelf health` (not a separate "counts snapshot" — that responsibility lives in `aelf stats`).

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

### Sharing or sync of brain-graph content

aelfrice ships no mechanism for exporting, syncing, or distributing memory contents between users, machines, or projects. The brain graph stays on the machine it was written on. There is no `aelf seed export`, no `.aelfrice/seed/` git-tracked directory, no cross-machine sync, no team-shared store, and no cross-project federation in the default install or any planned release.

This is a scope choice, not a missing feature. The local-only contract is what makes aelfrice's privacy, determinism, and audit guarantees meaningful:

- **Privacy.** A brain graph derived from real session activity contains absolute filesystem paths, hostnames, internal URLs, identifiers from git config, project-internal architecture details, names from commit history, and content the agent inferred from chat context. None of that is suitable for cross-machine or cross-user distribution by default, and no per-belief allowlist mechanism is reliable enough to make automated export safe by construction.
- **Determinism.** Every state of the store must be reproducible from its local write log. Importing curated content from another store breaks the property that the local write log is the single source of truth.
- **Audit.** Every retrieval result must be traceable to the locally-recorded events that produced it. Imported content has no local provenance and breaks the audit chain.

The principled response **is** one of:

1. **Bootstrap new collaborators with `aelf onboard .`** — re-extract from the repo's prose, git log, and AST on each new clone. This produces a graph derived from the publicly visible repo content only; nothing private leaks.
2. **Lock rules in CLAUDE.md, CONTRIBUTING.md, or other repo-tracked prose.** The onboarding scanner picks them up. The repo file is the source of truth; the brain graph is one machine's local index of it.
3. **Pair aelfrice with a separate tool when the use case demands shared memory.** A tool that distributes memory contents has a fundamentally different trust profile and deployment story. Mixing the two inside aelfrice would weaken aelfrice's guarantees for the use cases it does serve well.

This applies to the v2.0.0 release also: there is no "cross-project shared store," "shared scopes," or federation primitive on any planned release.

## Surface limits at v1.0

- **Retrieval is keyword only.** No stemming beyond porter+unicode61. No synonym expansion. "deploy" won't surface "publish to prod" without a tokenizable substring overlap.
- **One DB at a time.** No multi-project query. Use `AELFRICE_DB` to scope per-project.
- **Onboarding is shallow.** Three sources: prose, git log, Python AST. JS/TS/Rust/Go AST parsing not yet wired.
- **Classifier is regex-based on the CLI path.** Only the polymorphic MCP onboard uses host-LLM classification.
- **No bulk operations.** No batch lock, no `delete <pattern>`, no merge.
- **No edit.** Wrong belief → insert corrected one with `SUPERSEDES` edge; original stays.
- **No graph viz.** Inspect with `sqlite3 "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"`.

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
