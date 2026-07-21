# Philosophy

Design principles. Receipts in the code.

## From low-context to high-context

A fresh Claude session is low-context. Every conversation starts from zero: your stack, your conventions, the rule you set last week, the decision you reversed last month. The agent is a new hire on day one, every day. You spend the first ten minutes of every session restating things you've already said.

aelfrice changes that. You correct the agent once. You lock the constraint that matters. The next session starts with that context already attached. After a few weeks of small corrections the agent is operating on months of accumulated rules, and at some point you notice you've stopped repeating yourself.

In linguistics this is [high-context communication](https://en.wikipedia.org/wiki/High-context_and_low-context_cultures): three words convey a full procedure because the listener already carries the background. The more sessions you work together, the less you need to explain.

## Files don't solve this

The standard workaround is more markdown. `STATE.md`. `DECISIONS.md`. A CLAUDE.md that cross-references runbooks. Some projects have seven files the agent is *supposed* to follow.

The failure modes are predictable:

- **The agent reads but doesn't follow.** You write "always use the publish script, never push directly." It reads it, acknowledges it, runs `git push` anyway. The rule was in context. The agent treated it as a suggestion.
- **Cross-references break silently.** "See `docs/deploy-runbook.md` for deployment." The agent skips the reference, or reads the wrong section, or reads it but loses it after compaction. You don't find out until production.
- **State rots.** State files require discipline. One missed update and downstream sessions operate on stale information.
- **Files multiply.** What starts as one config becomes five, then ten. Each new failure mode begets a new file.

aelfrice is a different mechanism. The hook injects matched beliefs *as part of your prompt*, before the agent sees it. Nothing voluntary, nothing the agent can skip.

## Determinism is the property

Every retrieval result is a deterministic function of beliefs and rules. Given the same write log and the same code, every retrieval returns the same answer, bit-for-bit, across runs and machines and time.

Four commitments interlock to make that hold:

1. **Bit-level reproducibility.** No embeddings, no learned re-rankers, no LLM in the retrieval path. Replay the write log on the same code; you get the same result.
2. **Named-rule traceability.** *"Why did this rule surface for this query?"* has a finite answer. It bottoms out in named beliefs, with timestamps, created by named user actions, via named extraction patterns.
3. **Write-log reconstruction.** The log is the source of truth; the queryable structure is derived. *"What would the agent have retrieved last March, before that lock was set?"* is an answerable question.
4. **Audit comprehensible to a non-technical reviewer.** "BM25 matched these terms, returned these beliefs, filtered by these locked directives, created by these user actions at these timestamps." The chain holds for an auditor, not just a developer.

These hold compositionally. A single non-deterministic step in the retrieval path destroys the property for the whole pipeline. There is no "mostly deterministic" — either it holds end-to-end or it does not.

The trade-off is real. Embedding systems beat aelfrice on fuzzy semantic recall and multi-session aggregation. We treat that as a clarification of what aelfrice is for, not a gap to close at the cost of the property. Two bench-gate runs ([#197](https://github.com/robotrocketscience/aelfrice/issues/197), [#201](https://github.com/robotrocketscience/aelfrice/issues/201)) hit the natural-language-relatedness wall and closed wontfix; the successor [#422](https://github.com/robotrocketscience/aelfrice/issues/422) closed by shipping a deterministic value-comparison shape instead of the embedding shape — the same boundary, resolved the same way; the v3.0 ratification of *paraphrase / synonymy gates live in the consuming agent, not in aelfrice* is in [v3_relatedness_philosophy.md](../design/v3_relatedness_philosophy.md).

What it buys, beyond the obvious: debugging is bounded — a wrong retrieval traces to one specific rule, where an embedding system answers "wrong result" with similarity scores. Provenance composes — every belief carries it, so every retrieval inherits it. Counterfactual evaluation becomes tractable: replay history with and without a given correction, and determinism is what makes that differential meaningful. And domains that need explainability to survive an audit — medical, legal, financial — can actually use this; most agent-memory systems are locked out of them by construction.

## Bayesian, not vector

Vector RAG remembers documents. It doesn't remember *outcomes* — which retrievals helped, which hurt, which were corrected. Two beliefs with similar embeddings rank equally even when one has been right ten times and the other was harmful once.

aelfrice scores every belief with a Beta-Bernoulli posterior. Twenty lines:

```
posterior_mean = α / (α + β)
used    ⟹ α += 1
harmful ⟹ β += 1
```

No embedding model. No hyperparameter search. No opaque ranking. Every score is one division and the audit trail is one table. The Bayesian update is itself one of the named rules traceability bottoms out in.

**Exposure is not endorsement.** The posterior moves only on *evidence* — an explicit `used`/`harmful` signal, a user lock or correction, a contradiction. Merely *surfacing* a belief in retrieval does not. Since #1086 (v4.0) the hook retrieval path is audit-only by default (`AELFRICE_EXPOSURE_UPDATES_POSTERIOR`, default off): each surfacing is logged so its frequency stays recoverable, but α/β are untouched. This is deliberate — counting every re-surfacing as truth let whatever *recurs* float above genuine knowledge (measured on a real 24,883-belief store, recurring session scaffolding scored a higher mean μ than clean beliefs). **Recurrence is therefore a separate axis, not the posterior:** `corroboration_count` tracks how often a belief was re-asserted and is surfaced alongside μ (see `aelf introspect`), but it never feeds the truth-posterior. μ answers "is this true/useful"; recurrence answers "how often did this come up" — the two are kept orthogonal.

The posterior is **single-axis**. One `(α, β)` pair per belief, not a vector. A wrong belief is wrong overall; a useful belief is useful overall. The research line shipped a multi-axis `UncertaintyVector` (per-aspect `(α_i, β_i)` across existence / semantics / mechanism / cost) used by the speculative-belief surface (`wonder`, `reason`). aelfrice stays single-axis at v3.x; the multi-axis substrate question tracked at [#196](https://github.com/robotrocketscience/aelfrice/issues/196) was not reopened after v2.0 shipped — assume single-axis when porting.

The cost: dense semantic similarity is gone. The benefit: a learning loop that converges on what works *for you*, not what's textually similar — and a retrieval pipeline that preserves determinism end to end.

What's intentionally absent: an exploration term in retrieval. The research-line requirement was that ≥15% of retrievals surface high-uncertainty beliefs to keep the feedback loop from collapsing into a filter bubble — confident beliefs reinforced, uncertain beliefs never re-tested. aelfrice does not yet address that requirement. Any exploration mechanism (bandit-style, entropy-weighted, sampling-based) breaks the "same query, same beliefs" property, which aelfrice prioritises higher. Posterior reranking has shipped since v1.3 with no exploration term; if a future benchmark shows filter-bubble cost outweighs the determinism gain, exploration ships behind a flag in a future version.

> Historical note: at v1.0–v1.2 the posterior was computed and stored but L1 retrieval ranked by BM25 alone. The v1.3 retrieval wave wired the posterior into ranking; v1.7 made BM25F default-on; v3.0 made intentional clustering default-on; v3.3 made type-aware compression default-on. Feedback now moves what the agent sees end-to-end. See [LIMITATIONS](../user/LIMITATIONS.md).

## Locks, not just decay

Pure decay drifts trusted ground-truth toward the prior unless you keep restating it — the same failure mode as files.

A user lock short-circuits decay. The function returns `(α, β)` unchanged regardless of age. You don't ping it monthly to keep it alive.

Hard locks ossify, though. The earlier v2.x design accumulated `demotion_pressure` on `CONTRADICTS` edges and auto-demoted at a threshold; v3.2.0 removed that mechanism ([#814](https://github.com/robotrocketscience/aelfrice/issues/814) / PR #820, landed just after the v3.1.0 tag — see #833; it dropped the `demotion_pressure` column, the `apply_feedback(propagate=)` kwarg, and the `FeedbackResult.pressured_locks` / `.demoted_locks` fields). Lock correction now goes through `aelf lock` overwriting (per [#605](https://github.com/robotrocketscience/aelfrice/issues/605)) or explicit `aelf unlock` / `aelf delete`. Durability is the property, and the substrate trusts the user to be the one who flips a stale rule.

## Trust boundary at the hook surface

The `UserPromptSubmit` hook injects retrieved beliefs into the model's
input on every turn. That makes the hook a privileged channel: anything
inside or *adjacent to* the emitted block reads as elevated, system-trusted
context to the model.

The hook layer's job is to make that trust boundary structurally legible,
not to police what the model does on the other side of it. Three structural
defenses ship today: a fixed framing tag (`<belief id="…" lock="…">` inside
`<aelfrice-memory>`) tells the model the contents are *retrieved memory,
not instructions*; a render-time escape pass neutralises any tag-substring
that lands in stored belief content; a per-turn audit log
(`hook_audit.jsonl`) records the exact rendered block so post-hoc forensics
can answer "what was injected on turn N." See
[hook_hardening.md](../design/hook_hardening.md) for the design memo.

What the hook layer cannot do, by design: enforce that the model verifies
named session artifacts before acting, or guarantee the model treats
escaped tag-substrings as data. Those are model-behavior contracts. They
belong in CLAUDE.md / AGENTS.md, not in `aelfrice` Python. A model that
chooses to act on belief content as instruction *despite* the framing tag
is a model-layer problem; aelfrice exposes the boundary, the model honors
it.

## Local, always

Your corrections live in one SQLite file on your machine. No cloud sync, no telemetry, no API calls in the retrieval path. The cloud LLM at the other end of your prompt sees whatever aelfrice injects — that's inherent — but aelfrice limits the slice (default 1,500 tokens for hook-injected context, scoped to the current query; the `retrieve()` API default is 2,400) rather than dumping the whole memory.

The 2,400-token retrieval-API default is a calibrated choice, not an arbitrary one. The hypothesis from the research line is that focused context beats exhaustive context: a ~2.4K-token retrieval that selects the right beliefs should match or exceed a 10K-token full-memory dump on response quality, while burning ~4× fewer tokens on memory plumbing. The reproducibility cut at v2.0 was where that curve got re-measured against the public retrieval pipeline; the 2,400-token budget is the post-v1.3 default, configurable per-call.

[PRIVACY.md](../user/PRIVACY.md) for verifiable specifics.

## Small surface, on purpose

The v1 surface is small. Every belief mutation goes through `apply_feedback`. Every lock through one path. There is one writer of `(α, β)`, one place that runs decay, one entry point for retrieval. When the system misbehaves, there is one place to look.

The earlier research line had a much bigger surface — twenty-nine MCP tools, `wonder`, `reason`, snapshot/diff. It delivered value but also delivered ambiguity. The rebuild started narrow on purpose; v1.x–v3.x have reintroduced breadth (15 MCP tools at v3.3, plus `/aelf:wonder` / `/aelf:reason` / `/aelf:graph` slash surfaces), each addition gated on evidence — a benchmark, an experiment, a clear case where the existing operations don't suffice.

## Lean dependencies, on purpose

Three hard runtime dependencies, each one argued in: `numpy` and `scipy` (v1.5, #148 — the BM25 sparse-matvec retrieval lane, now also the HRR and spectral-graph math) and `snowballstemmer` (v1.7, #154 — Porter stemming). Everything else is Python stdlib plus SQLite (the stdlib already wraps it). Optional extras add capability without entering the default install: `[mcp]` (fastmcp), `[onboard-llm]` (the direct-API classifier SDK), `[archive]` (cryptography), `[benchmarks]` (dev-side adapters).

Every dependency is maintenance debt and attack surface. Heavier machinery — vector indices, embedding services, neural rerankers — earns its way in only when an experiment shows the existing stack is the bottleneck.

## What we can and can't guarantee

aelfrice is a memory substrate, not an LLM. The honest decomposition for any "the agent will follow this rule" claim:

| Tier | Mechanism | Guarantee |
|---|---|---|
| 1. Storage | SQLite WAL + locked belief | The rule is durably written and never lost. |
| 2. Injection | L0 always-loaded into every prompt | The rule is in the model's context on every retrieval. |
| 3. Compression survival | Per-prompt L0 re-injection (locks re-enter context on the first prompt after compaction); strengthened by the opt-in PreCompact rebuilder (`aelf setup --rebuilder`) | The rule survives a context-window compaction. |
| 4. Violation detection | Not implemented | — |
| 5. Violation blocking | Not implemented | — |
| 6. LLM compliance | The model actually obeys the injected rule | **Not under aelfrice's control.** |

Tiers 1–3 hold mechanically. Tiers 4–5 (post-execution detection, pre-execution blocking) are research-line capabilities with no current roadmap entry. Tier 6 is the LLM's own training and decoding, which aelfrice cannot constrain. If the model ignores an injected lock, the failure mode is in the model, not in aelfrice — but that distinction does not console a user whose agent just ran `git push` despite a clear directive.

A few more properties fall out of the same substrate:

- **Session recovery, not just write durability.** SQLite WAL guarantees that every acknowledged write survives a crash. That is the storage-engine claim. The product-level claim is that the *working context* of an interrupted session is reconstructable on restart — not from a snapshot file, but from the same belief store the next session retrieves against, augmented by a `<recent-work>` SessionStart sub-block carrying the current branch, the last N commits, and any issue numbers referenced in the recent work (#887). Re-open the terminal next week, ask "where were we?", and the locks plus the per-project working state are still there.
- **Confidence does not auto-flag.** A belief whose posterior drifts below 0.5 is not surfaced as a warning. No automatic state change is driven by negative evidence — locked beliefs hold by design (the v2.x auto-demote mechanism was removed at v3.2.0 [#814](https://github.com/robotrocketscience/aelfrice/issues/814)). If you want to know which beliefs are losing the feedback loop, you ask `aelf speculative` (non-locked beliefs ranked by posterior evidence) or `aelf review --generate` (the oldest-unconfirmed review queue); `aelf status` shows only aggregate counts, and the system does not interrupt to tell you.
- **Append-only substrate.** The research line had a separate `observations` table — insert-only, every observation that produced a belief permanently recorded. v1.x kept only `feedback_history` immutable (every `apply_feedback` writes a row, rows are never updated). v1.5 (#205) added the append-only `ingest_log` table; v1.7 (#264) routed every ingest entry point through the derivation worker, and the v2.0.1 view-flip (#265) made `beliefs` and `edges` a materialized projection of the log. Beliefs are still mutated for decay and feedback, but the full write log is durable and replay-capable. See [`design/write-log-as-truth.md`](../design/write-log-as-truth.md) for the architectural memo.

## What this design buys

- **Continuity.** Close the terminal, come back next week, "where were we?" — the memory restores it.
- **Compounding.** The graph fills on explicit `onboard` / `lock` / `feedback` (since v1.0), the default-on transcript-ingest / commit-ingest / session-start hooks (since v2.1 #529), and posterior-aware ranking (since v1.3, with BM25F default-on since v1.7 and type-aware compression default-on since #769). Every layer compounds: more sessions → more feedback → better ranking → more accurate retrieval.
- **Self-correction.** Stale unlocked rules decay. Wrong locks are corrected by overwriting via `aelf lock`, or removed via `aelf unlock` / `aelf delete` — the v2.x auto-demote mechanism was removed at v3.2.0 ([#814](https://github.com/robotrocketscience/aelfrice/issues/814)).
- **Auditability.** Every belief has a content hash and timestamp. Every feedback event has an audit row. Every score is `α / (α + β)`. Read the database; reproduce the system's claims about itself.
- **Locality.** No service to fail, no account to lose, no quota to exceed.

## What it isn't

- Not a notebook. Not a team knowledge base.
- Not a vector store. Semantic similarity isn't in v1 retrieval.
- Not a planner. Decisions belong in your head.
- Not a replacement for documentation, conventions, or runbooks. A complement.

A small, sharp tool for one job: keeping the agent that helps you build software from forgetting what you said.
