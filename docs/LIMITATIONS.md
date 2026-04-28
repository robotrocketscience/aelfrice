# Limitations

What aelfrice doesn't do yet. Tracked openly.

## The big one: feedback doesn't drive ranking

`apply_feedback` updates `(α, β)` and writes an audit row. The posterior mean is exposed via `aelf stats` and the MCP. But L1 retrieval still orders hits by `bm25(beliefs_fts)` alone, not by posterior.

Concretely: marking a belief `harmful` weakens its math, but the next retrieval that matches its keywords will still surface it. Ranking that consumes the posterior lands in the v1.3 retrieval wave.

The benchmark harness ships at v1.0 as the measurement instrument. It is not yet a proof of the feedback claim.

## No semantic similarity

Retrieval is BM25 keyword search over FTS5 (porter unicode61 stemming). "deploy" will not surface "publish to prod" without tokenizable substring overlap.

This is a deliberate scope choice, not a roadmap item. Adding embeddings would break determinism end-to-end — see [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property). The principled response to fuzzy semantic recall queries is to pair aelfrice with a separate tool, not blend embeddings into the retrieval path.

## Onboarding scope

The CLI scanner walks three sources: prose files (`*.md`, `*.rst`, `*.txt`, `*.adoc`), `git log`, and Python AST. Not yet wired: JavaScript / TypeScript / Rust / Go ASTs.

Classification on the CLI path is regex-based. Higher-quality classification requires the MCP `aelf:onboard` polymorphic flow, which routes through the host LLM.

## Sharp edges

- **Locks are durable.** Fresh lock = `(α, β) = (9.0, 0.5)`. Five independent contradicting feedback events are required to auto-demote. If you lock a wrong rule and correct it only once or twice, the lock keeps winning. `aelf demote <id>` drops it immediately.
- **`CONTRADICTS` edges drive demotion pressure.** The regex classifier rarely produces them, so demotion pressure accumulates only from manual feedback unless you wire commit-ingest or transcript-ingest.
- **Jeffreys prior reads as 0.5.** A belief with no feedback reports posterior mean exactly `0.5`. That means "no evidence yet," not "coin-flip true."
- **`aelf onboard` is non-incremental on duplicates.** Re-runs are idempotent; existing beliefs are not re-scored or refreshed.
- **No bulk operations.** No batch lock, no `delete <pattern>`, no merge.
- **No edit.** A wrong belief is corrected by inserting a new one with a `SUPERSEDES` edge; the original stays.
- **No graph viz.** Inspect with `sqlite3 "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"`.
- **JSONL batch ingest has no PII scrubber.** `aelf ingest-transcript --batch ~/.claude/projects/` will pull whatever you typed in chat into the local belief graph. Review before backfilling.

## Out of scope

These are scope choices that follow from aelfrice's commitments. They are *not* roadmap items.

### Sharing, sync, or federation

aelfrice ships no mechanism for exporting, syncing, or distributing memory contents between users, machines, or projects. The brain graph stays on the machine it was written on.

This is a privacy and audit choice. A graph derived from real session activity contains filesystem paths, hostnames, internal URLs, names from git config, project architecture details, and content the agent inferred from chat. None of that is suitable for cross-machine distribution by default. Per-belief allowlists are not reliable enough to make automated export safe.

To bootstrap a new clone or collaborator: run `aelf onboard .`. The graph is re-extracted from publicly-visible repo content. To share rules: lock them in CLAUDE.md, CONTRIBUTING.md, or other repo-tracked prose, and the onboard scanner picks them up.

### Multi-session aggregation

aelfrice is not optimised for "how many times did the user mention X across last quarter?" queries. That is the LLM-with-RAG-and-summary-buffer task, not a behavioural-directive recall task. On benchmarks like LongMemEval multi-session, embedding systems will outperform aelfrice for that query category.

The principled response is to add aggregative-query routing at the structural-analysis layer (SQL aggregations over `feedback_history`, scoped graph walks, time-bucketed COUNT queries) — not to add embeddings.

### Multi-project query

One DB at a time. Beliefs from project A do not surface in project B (different `.git/` directories). Use `AELFRICE_DB` to scope per-project explicitly.

## Compatibility

- Python 3.12 or 3.13.
- macOS and Linux are routinely tested. Windows should work but is not exercised on every release.
- `uv` recommended; `pip` works.

## Reporting

File an issue tagged `limitations` for additions or corrections.
