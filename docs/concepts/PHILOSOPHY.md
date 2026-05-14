# Philosophy

Design principles. Receipts in the code.

## From low-context to high-context

A fresh Claude session is low-context. Every conversation starts from zero: your stack, your conventions, the rule you set last week, the decision you reversed last month. The agent is a new hire on day one, every day. You spend the first ten minutes of every session restating things you've already said.

aelfrice changes that. You correct the agent once. You lock the constraint that matters. The next session starts with that context already attached. After a few weeks of small corrections, the agent operates with months of accumulated rules, and you stop noticing the friction is gone.

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

The trade-off is real. Embedding systems beat aelfrice on fuzzy semantic recall and multi-session aggregation. We treat that as a clarification of what aelfrice is for, not a gap to close at the cost of the property. Three independent bench-gate runs against three different corpora ([#197](https://github.com/robotrocketscience/aelfrice/issues/197), [#422](https://github.com/robotrocketscience/aelfrice/issues/422), [#201](https://github.com/robotrocketscience/aelfrice/issues/201)) hit the natural-language-relatedness wall and closed wontfix on the same boundary; the v3.0 ratification of *paraphrase / synonymy gates live in the consuming agent, not in aelfrice* is in [v3_relatedness_philosophy.md](../design/v3_relatedness_philosophy.md).

What it buys, beyond the obvious:

- **Debugging is bounded.** A wrong retrieval is one specific rule. Compare with embedding systems where "wrong result" gets answered with similarity scores.
- **Provenance composes.** Every belief carries provenance; every retrieval inherits it.
- **Counterfactual evaluation is tractable.** Replay history with and without specific corrections. Determinism is what makes the differential meaningful.
- **High-stakes deployment is structurally admitted.** Medical, legal, financial — domains that need explainability that survives an audit. Most agent-memory systems are structurally locked out; aelfrice is structurally admitted.

## Bayesian, not vector

Vector RAG remembers documents. It doesn't remember *outcomes* — which retrievals helped, which hurt, which were corrected. Two beliefs with similar embeddings rank equally even when one has been right ten times and the other was harmful once.

aelfrice scores every belief with a Beta-Bernoulli posterior. Twenty lines:

```
posterior_mean = α / (α + β)
used    ⟹ α += 1
harmful ⟹ β += 1
```

No embedding model. No hyperparameter search. No opaque ranking. Every score is one division and the audit trail is one table. The Bayesian update is itself one of the named rules traceability bottoms out in.

The posterior is **single-axis**. One `(α, β)` pair per belief, not a vector. A wrong belief is wrong overall; a useful belief is useful overall. The research line shipped a multi-axis `UncertaintyVector` (per-aspect `(α_i, β_i)` across existence / semantics / mechanism / cost) used by the speculative-belief surface (`wonder`, `reason`). aelfrice v1.x is single-axis only. Whether v2.0 adopts the multi-axis substrate is a load-bearing architectural decision tracked at [#196](https://github.com/robotrocketscience/aelfrice/issues/196); until that lands, assume single-axis when porting.

The cost: dense semantic similarity is gone. The benefit: a learning loop that converges on what works *for you*, not what's textually similar — and a retrieval pipeline that preserves determinism end to end.

What's intentionally absent: an exploration term in retrieval. The research-line requirement was that ≥15% of retrievals surface high-uncertainty beliefs to keep the feedback loop from collapsing into a filter bubble — confident beliefs reinforced, uncertain beliefs never re-tested. aelfrice does not yet address that requirement. Any exploration mechanism (bandit-style, entropy-weighted, sampling-based) breaks the "same query, same beliefs" property, which v1.x prioritises higher. v1.3 ships posterior reranking with no exploration term. If a future benchmark shows filter-bubble cost outweighs the determinism gain, exploration ships behind a flag in v2.x.

> At v1.0–v1.2 the posterior is computed and stored, but L1 retrieval still ranks by BM25 alone. The v1.3 retrieval wave wires the posterior into ranking. Until then, feedback updates the audit trail but doesn't yet move what the agent sees. See [LIMITATIONS](../user/LIMITATIONS.md).

## Locks, not just decay

Pure decay drifts trusted ground-truth toward the prior unless you keep restating it — the same failure mode as files.

A user lock short-circuits decay. The function returns `(α, β)` unchanged regardless of age. You don't ping it monthly to keep it alive.

Hard locks ossify, though. So locks accumulate **demotion pressure** when contradicting evidence arrives; after enough independent contradictions (default 5), the lock auto-demotes to a regular belief. Durable when stable, self-correcting when wrong.

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

Your corrections live in one SQLite file on your machine. No cloud sync, no telemetry, no API calls in the retrieval path. The cloud LLM at the other end of your prompt sees whatever aelfrice injects — that's inherent — but aelfrice limits the slice (default 2,400 tokens, scoped to the current query) rather than dumping the whole memory.

The 2,400-token default is a calibrated choice, not an arbitrary one. The hypothesis from the research line is that focused context beats exhaustive context: a ~2.4K-token retrieval that selects the right beliefs should match or exceed a 10K-token full-memory dump on response quality, while burning ~4× fewer tokens on memory plumbing. The reproducibility cut at v2.0 was where that curve got re-measured against the public retrieval pipeline; the 2,400-token budget is the post-v1.3 default, configurable per-call.

[PRIVACY.md](../user/PRIVACY.md) for verifiable specifics.

## Small surface, on purpose

The v1 surface is small. Every belief mutation goes through `apply_feedback`. Every lock through one path. There is one writer of `(α, β)`, one place that runs decay, one entry point for retrieval. When the system misbehaves, there is one place to look.

The earlier research line had a much bigger surface — twenty-nine MCP tools, `wonder`, `reason`, snapshot/diff. It delivered value but also delivered ambiguity. The rebuild starts narrow on purpose. v1.x reintroduces breadth, each addition gated on evidence — a benchmark, an experiment, a clear case where the existing operations don't suffice.

## Pure stdlib

Zero hard runtime dependencies. Python stdlib plus SQLite (the stdlib already wraps it). The optional `[mcp]` extra adds `fastmcp`; nothing else.

Every dependency is maintenance debt and attack surface. Heavier machinery — vector indices, embedding services, neural rerankers — earns its way in only when an experiment shows the existing stack is the bottleneck.

## What we can and can't guarantee

aelfrice is a memory substrate, not an LLM. The honest decomposition for any "the agent will follow this rule" claim:

| Tier | Mechanism | Guarantee |
|---|---|---|
| 1. Storage | SQLite WAL + locked belief | The rule is durably written and never lost. |
| 2. Injection | L0 always-loaded into every prompt | The rule is in the model's context on every retrieval. |
| 3. Compression survival | PreCompact rebuilder + locks-first ordering | The rule survives a context-window compaction. |
| 4. Violation detection | Not implemented at v1.x | — |
| 5. Violation blocking | Not implemented at v1.x | — |
| 6. LLM compliance | The model actually obeys the injected rule | **Not under aelfrice's control.** |

Tiers 1–3 hold mechanically. Tiers 4–5 (post-execution detection, pre-execution blocking) are research-line capabilities deferred to v2.x. Tier 6 is the LLM's own training and decoding, which aelfrice cannot constrain. If the model ignores an injected lock, the failure mode is in the model, not in aelfrice — but that distinction does not console a user whose agent just ran `git push` despite a clear directive.

Two recovery angles fall out of the same substrate:

- **Session recovery, not just write durability.** SQLite WAL guarantees that every acknowledged write survives a crash. That is the storage-engine claim. The product-level claim is that the *working context* of an interrupted session is reconstructable on restart — not from a snapshot file, but from the same belief store the next session retrieves against. Re-open the terminal next week, ask "where were we?", and the locks plus recent retrieval-relevant beliefs are still there.
- **Confidence does not auto-flag.** A belief whose posterior drifts below 0.5 is not surfaced as a warning at v1.x. Only locked-belief demotion-pressure (≥5 contradictions → auto-demote) produces a visible state change. If you want to know which beliefs are losing the feedback loop, you ask `aelf stats`; the system does not interrupt to tell you.
- **The append-only substrate at v1 is `feedback_history`, not observations.** The research line had a separate `observations` table that was insert-only — every observation that produced a belief was permanently recorded. aelfrice v1 does not have that table. Beliefs are the substrate, and beliefs *are* mutated (decay adjusts age weighting; feedback updates `(α, β)`). What aelfrice v1 *does* keep append-only is `feedback_history`: every `apply_feedback` event writes a row, and rows are never updated. That is the immutable substrate at v1, and it is sufficient for "did the user actually correct this?" audit. Full ingest-log immutability — recording every observation that produced or refreshed a belief, not only every feedback event — is the v2.0 contract; see [`design/write-log-as-truth.md`](design/write-log-as-truth.md) for the proposed table and the migration story.

## What this design buys

- **Continuity.** Close the terminal, come back next week, "where were we?" — the memory restores it.
- **Compounding.** At v1.0 the graph fills on explicit `onboard`/`lock`/`feedback`. v1.1 closes the hook→feedback loop. v1.2 adds commit-ingest and transcript-ingest hooks. v1.3 wires the posterior into ranking.
- **Self-correction.** Stale rules decay. Wrong locks demote.
- **Auditability.** Every belief has a content hash and timestamp. Every feedback event has an audit row. Every score is `α / (α + β)`. Read the database; reproduce the system's claims about itself.
- **Locality.** No service to fail, no account to lose, no quota to exceed.

## What it isn't

- Not a notebook. Not a team knowledge base.
- Not a vector store. Semantic similarity isn't in v1 retrieval.
- Not a planner. Decisions belong in your head.
- Not a replacement for documentation, conventions, or runbooks. A complement.

A small, sharp tool for one job: keeping the agent that helps you build software from forgetting what you said.
