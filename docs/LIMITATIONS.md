# Limitations

What aelfrice doesn't do yet. Tracked openly.

## The big one: feedback doesn't drive ranking (lifted at v1.3.0, partially)

Through v1.2.x: `apply_feedback` updates `(α, β)` and writes an audit row. The posterior mean is exposed via `aelf stats` and the MCP. But L1 retrieval orders hits by `bm25(beliefs_fts)` alone, not by posterior. Marking a belief `harmful` weakens its math, but the next retrieval that matches its keywords still surfaces it.

The benchmark harness ships at v1.0 as the measurement instrument. It is not yet a proof of the feedback claim.

**At v1.3.0:** ranking begins to consume the posterior. L1 score becomes `log(bm25) + 0.5 * log(posterior_mean)`; locked beliefs (L0) bypass scoring as before; the cache invalidates correctly through the existing store-mutation hook. **Partial** because the full feedback-into-ranking eval — 10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — lands at v2.0.0. See [`docs/bayesian_ranking.md`](bayesian_ranking.md) for the v1.3 contract.

## No semantic similarity

Retrieval is BM25 keyword search over FTS5 (porter unicode61 stemming). "deploy" will not surface "publish to prod" without tokenizable substring overlap.

This is a deliberate scope choice, not a roadmap item. Adding embeddings would break determinism end-to-end — see [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property). The principled response to fuzzy semantic recall queries is to pair aelfrice with a separate tool, not blend embeddings into the retrieval path.

## Onboarding scope

The CLI scanner walks three sources: prose files (`*.md`, `*.rst`, `*.txt`, `*.adoc`), `git log`, and Python AST. Not yet wired: JavaScript / TypeScript / Rust / Go ASTs.

Classification on the CLI path defaults to regex-based priors. Higher-quality classification is available via two paths:
- **MCP `aelf:onboard`** polymorphic flow, which routes through the host LLM.
- **`aelf onboard --llm-classify`** (v1.3+, default-off) — routes through Claude Haiku directly. Requires `ANTHROPIC_API_KEY`. Four consent gates enforce the privacy boundary. See [llm_classifier.md](llm_classifier.md).

## BFS multi-hop temporal coherence

The v1.3.0 BFS multi-hop expansion (see [bfs_multihop.md](bfs_multihop.md)) resolves each hop to the globally latest serial of its target belief independently. For audit-shaped queries ("what did the agent believe at the time it made this decision?") this can surface a chain whose intermediate hops postdate the seed's session — a fidelity loss against a corpus where supersessions accumulated after the seed was written.

The default retrieval mode (recall, not audit) is correctly served by latest-serial-per-hop. The temporal-coherence fix is targeted at v2.0.0 alongside the published-numbers reproducibility harness; the spec § Open question: temporal coherence documents why deferring is the right call for v1.3.

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
