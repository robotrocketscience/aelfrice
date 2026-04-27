# Philosophy

Design principles. Receipts in the code.

## Files aren't enough

Power users converge on the same workaround: externalize state into markdown files. A config for rules. A state file. A roadmap. A decisions log. Then forty-six more, with cross-references that hold only when the agent decides to read them.

The failure modes are predictable: the agent reads the rule and ignores it; cross-references break silently after compaction; manual state rots after one missed update; files multiply faster than discipline can maintain them.

aelfrice is a different mechanism, not a tidier file layout. It finds context itself and injects it into the prompt before the agent answers. The agent never has to remember to check a file or follow a cross-reference. Knowledge is already there.

## Determinism is the property

The architectural commitment that organizes everything else.

aelfrice commits to four things, and the commitments interlock:

1. **Bit-level reproducibility.** Given the same write log and the same code, every retrieval result is bit-identical across runs, machines, and time. No model checkpoint, no learned representation that drifts between releases.
2. **Named-rule traceability.** Every retrieval result is a function of named beliefs and named rules. Asking *"why did this directive surface for this query?"* produces a chain that bottoms out in specific user actions at specific timestamps via specific extraction patterns. The chain is finite, inspectable, and human-readable at every step.
3. **Write-log historical reconstruction.** Because the log of writes is the source of truth and the queryable structure is derived, any past state of the system is reconstructible. *"What would the agent have retrieved on this query last March, before that lock was set?"* is an answerable question.
4. **Audit comprehensible to a non-technical reviewer.** "BM25 matched these terms, which retrieved these beliefs, which were filtered by these locked directives, which were created by these user actions at these timestamps" — the chain holds for a regulator, an auditor, or a user, not just for the developer.

These properties hold compositionally only if every component preserves them. A single non-deterministic step in any retrieval path destroys the property for the whole pipeline. There is no "mostly deterministic" — either the property holds end to end or it does not. Embeddings, learned re-rankers, and LLM-driven retrieval steps trade this property for retrieval-quality gains on certain query categories. aelfrice trades the other way, deliberately.

We are not aware of another agent-memory system that combines all four of these guarantees as a single system property.

What it buys, beyond the obvious:

- **Debugging is bounded.** A wrong retrieval is the result of a specific rule that can be identified and fixed. Compare with embedding systems where "wrong result" gets answered with similarity scores.
- **Provenance composes.** Every belief carries provenance; every retrieval result is a deterministic function of beliefs and rules; every retrieval result therefore inherits provenance from its sources.
- **Acceptance tests are precise behavioural specifications, not regression detectors with tolerances.**
- **Counterfactual evaluation becomes tractable.** Replay history with and without specific corrections; measure the differential. Determinism is what makes the differential meaningful.
- **High-stakes deployment is structurally admitted.** Medical, legal, financial contexts need explainability that survives an audit. Most agent-memory systems are structurally locked out of these settings; aelfrice is structurally admitted.

There are categories of query — multi-session aggregation, fuzzy semantic recall — where the deterministic stack underperforms an embedding system. We treat that as a clarification of what aelfrice is for, not a gap to close at the cost of the property. See [LIMITATIONS](LIMITATIONS.md) for the current scope.

## Bayesian, not vector

Vector RAG remembers documents. It doesn't remember _outcomes_ — which retrievals helped, which hurt, which were corrected. Two beliefs with similar embeddings rank equally even when one has been right ten times and the other was retrieved once and harmful.

aelfrice scores every belief with a Beta–Bernoulli posterior over feedback events. Twenty lines in `scoring.py`:

```
posterior_mean = α / (α + β)
used    ⟹ α += 1
harmful ⟹ β += 1
```

No embedding model. No hyperparameter search. No opaque ranking. Every score is one division and the audit trail is one table. The Bayesian update is itself one of the named rules that traceability bottoms out in.

The cost: dense semantic similarity is gone. The benefit: a learning loop that converges on what _works for you_, not what's textually similar — and a retrieval pipeline that preserves the determinism property end to end.

## Locks, not just decay

Pure decay drifts trusted ground-truth toward the prior unless you keep restating it — the same failure mode as files.

A user lock short-circuits decay. The function returns `(α, β)` unchanged regardless of age. You don't ping it monthly to keep it alive.

But hard locks ossify. So locks accumulate _demotion pressure_ when contradicting evidence arrives; after enough independent contradictions (default 5), the lock auto-demotes back to a normal belief. Durable when stable, self-correcting when wrong.

## Local, always

Your corrections live in one SQLite file on your machine. No cloud sync, no telemetry, no API calls in the retrieval path. The cloud LLM at the other end of your prompt sees whatever aelfrice injects — that's inherent. aelfrice limits the slice (default 2000 tokens, scoped to the current query) rather than dumping the whole memory.

## Small surface, on purpose

The v1.0 surface is ten operations. Every belief mutation goes through `apply_feedback`. Every lock through one path. There is one writer of `(α, β)`, one place that runs decay, one entry point for retrieval. When the system misbehaves, there is one place to look.

The previous codebase (v2.0) had a much bigger surface — thirty MCP tools, `wonder`, `reason`, snapshot/diff. It delivered value but also delivered ambiguity. The rebuild starts narrow on purpose. v1.x reintroduces breadth, each addition with evidence — a benchmark, an experiment, a clear case where the existing operations don't suffice.

## Pure stdlib

Zero hard dependencies. Python stdlib plus SQLite (which the stdlib already wraps). The optional `[mcp]` extra adds `fastmcp`; nothing else.

Every dependency is maintenance debt and attack surface. Heavier machinery — vector indices, embedding services, neural rerankers — earns its way in only when an experiment shows the existing stack is the bottleneck.

## What this design buys

- **Continuity.** Close the terminal, come back next week, "where were we?" — the memory restores it.
- **Compounding (by design).** At v1.0 the graph fills during explicit `onboard`/`lock`/`feedback` calls. v1.0.1 closes the hook→feedback-history loop so retrievals leave an audit trail. v1.2.0 adds the commit-ingest hook for automatic capture. v1.3 wires the posterior into retrieval ranking. See [LIMITATIONS](LIMITATIONS.md#known-issues-at-v10).
- **Self-correction.** Stale rules decay. Wrong locks demote. The mechanism notices when reality has moved.
- **Auditability.** Every belief has a content hash and timestamp. Every feedback event has an audit row. Every score is `α / (α + β)`. Every retrieval result is reproducible bit-for-bit from the write log and the code. Read the database; reproduce the system's claims about itself. See [Determinism is the property](#determinism-is-the-property).
- **Locality.** No service to fail, no account to lose, no quota to exceed.

## What it isn't

- Not a notebook. Not a team knowledge base.
- Not a vector store. Semantic similarity is not in v1.0 retrieval.
- Not a planner. Decisions belong in your head.
- Not a replacement for documentation, conventions, or runbooks. Complement.

A small, sharp tool for one job: keeping the agent that helps you build software from forgetting what you said.
