# Philosophy

Design principles. The code holds the evidence.

## From low-context to high-context

A new Claude session has low context. Every conversation starts with no knowledge of your work. The agent does not know your stack, your conventions, the rule you set last week, or the decision you reversed last month. The agent starts every day with no knowledge from the days before it. You restate information that you gave before. This restatement takes the first ten minutes of every session.

aelfrice changes this condition. You correct the agent one time. You lock the constraint that is important. The next session starts with that context attached. After some weeks of small corrections, the agent operates on months of accumulated rules. At some time you see that you no longer repeat yourself.

Linguistics calls this condition [high-context communication](https://en.wikipedia.org/wiki/High-context_and_low-context_cultures). Three words communicate a full procedure, because the listener already has the background. When you and the agent work together for more sessions, you must explain less.

## Files don't solve this

The usual alternative is more markdown. Examples are `STATE.md` and `DECISIONS.md`. Another example is a CLAUDE.md file that refers to runbooks. Some projects have seven files that the agent is *supposed* to follow.

The failure modes are predictable:

- **The agent reads the rule but does not obey it.** You write "always use the publish script, never push directly." The agent reads the rule. The agent confirms the rule. The agent then runs `git push`. The rule was in the context. The agent treated the rule as a suggestion.
- **Cross-references break, and nothing reports the failure.** One file says "See `docs/deploy-runbook.md` for deployment." The agent skips the reference. Or the agent reads the wrong section. Or the agent reads the reference and then loses the content after a compaction. You find the failure only in production.
- **State files become stale.** State files require discipline. One missed update makes the sessions after it operate on stale information.
- **The number of files increases.** One configuration file becomes five files, then ten files. Each new failure mode causes a new file.

aelfrice uses a different mechanism. The hook injects the matched beliefs *as part of your prompt*, before the agent receives the prompt. The agent does not choose to read the beliefs. The agent cannot skip them.

## Determinism is the property

Every retrieval result is a deterministic function of the beliefs and the rules. Two retrievals with the same write log and the same code return the same answer. The answer is identical bit for bit across runs, across machines and across time.

Four commitments together make that property hold:

1. **Bit-level reproducibility.** The retrieval path contains no embeddings, no learned re-rankers and no large language model (LLM). Replay the write log on the same code. You get the same result.
2. **Named-rule traceability.** The question *"Why did this rule surface for this query?"* has a finite answer. The answer ends at named beliefs with timestamps. Named user actions created those beliefs, through named extraction patterns.
3. **Write-log reconstruction.** The log is the source of truth. aelfrice derives the queryable structure from the log. The question *"What would the agent have retrieved last March, before that lock was set?"* has an answer.
4. **Audit comprehensible to a non-technical reviewer.** "BM25 matched these terms, returned these beliefs, filtered by these locked directives, created by these user actions at these timestamps." The chain holds for an auditor, and not only for a developer.

> **The edges are the standing exception to commitment (1) and commitment (3).** The write log as shipped does not cover the edges. A replay therefore reconstructs the beliefs. The replay does not reconstruct the graph. The contract ratified for the edges is *log-derived*. The shipped state is not log-derived. For the measured detail, see the `edges` note under *Append-only substrate* below ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)).

> **Point-in-time reconstruction is a design property. It is not a shipped surface.** Commitment (3) states the goal that the substrate is built for. Commitment (3) does not describe an existing command. What ships is `aelf doctor --replay`. That command re-derives the log against the *current* store and reports full-column equality. It answers the question *"does the projection still match the log?"*. It does not answer the question *"what did the store look like on a given date?"*. The retrieval path has no `as-of` parameter, so you cannot run the *"last March"* question today. The same qualification applies to the counterfactual-evaluation claim below ([#1163](https://github.com/robotrocketscience/aelfrice/issues/1163)).

These commitments hold compositionally. One non-deterministic step in the retrieval path destroys the property for the whole pipeline. There is no "mostly deterministic" condition. The property holds from end to end, or it does not hold.

The trade-off is real. Embedding systems give better results than aelfrice for fuzzy semantic recall and for multi-session aggregation. We treat that result as a clarification of what aelfrice is for. We do not close that gap at the cost of the determinism property.

Two bench-gate runs ([#197](https://github.com/robotrocketscience/aelfrice/issues/197), [#201](https://github.com/robotrocketscience/aelfrice/issues/201)) reached the limit of natural-language relatedness. Both runs closed as wontfix. The successor issue [#422](https://github.com/robotrocketscience/aelfrice/issues/422) closed with a deterministic value-comparison shape instead of the embedding shape. That is the same boundary, resolved in the same way. At v3.0 the project ratified this rule: *paraphrase / synonymy gates live in the consuming agent, not in aelfrice*. The document [v3_relatedness_philosophy.md](../design/v3_relatedness_philosophy.md) records the ratification.

Determinism gives more than the obvious benefits:

- **Debugging is bounded.** A wrong retrieval traces to one specific rule. An embedding system answers a wrong result with similarity scores.
- **Provenance composes.** Every belief carries its provenance, so every retrieval inherits it.
- **Counterfactual evaluation becomes tractable.** Replay the history with a given correction. Then replay the history without that correction. Determinism is what makes that difference meaningful.
- **Some domains need explainability to survive an audit.** Medical, legal and financial work are examples. Those domains can use aelfrice. Most agent-memory systems cannot enter those domains, because of how they are built.

## Bayesian, not vector

Vector retrieval-augmented generation (RAG) remembers documents. Vector RAG does not remember *outcomes*. It does not record which retrievals helped, which retrievals hurt, and which retrievals somebody corrected. Two beliefs with similar embeddings get an equal rank. The ranks stay equal when one belief was right ten times and the other belief was harmful one time.

aelfrice scores every belief with a Beta-Bernoulli posterior. The implementation is twenty lines. The core of the implementation is:

```
posterior_mean = α / (α + β)
used    ⟹ α += 1
harmful ⟹ β += 1
```

There is no embedding model. There is no hyperparameter search. There is no opaque ranking. Every score is one division. The audit trail is one table. The Bayesian update is one of the named rules where the traceability ends.

**Exposure is not endorsement.** The posterior moves only on *evidence*. The evidence is an explicit `used` or `harmful` signal, a user lock, a user correction, or a contradiction. A retrieval that *surfaces* a belief does not move the posterior. Since #1086 (v4.0) the hook retrieval path is audit-only by default. The environment variable `AELFRICE_EXPOSURE_UPDATES_POSTERIOR` controls the path and defaults to off. aelfrice logs each surfacing, so the frequency of the surfacing stays recoverable. aelfrice does not touch α or β. This behaviour is deliberate. When every re-surfacing counted as truth, the beliefs that *recur* ranked above genuine knowledge. A measurement on a real store of 24,883 beliefs showed the effect: recurring session scaffolding scored a higher mean μ than clean beliefs. **Recurrence is therefore a separate axis from the posterior.** The field `corroboration_count` counts how often a belief was re-asserted. aelfrice shows that count together with μ (see `aelf introspect`). The count never feeds the truth-posterior. μ answers the question "is this true/useful". Recurrence answers the question "how often did this come up". aelfrice keeps the two measures orthogonal.

**A store that predates #1086 still carries the old writes.** The change stopped new exposure evidence. The change did not remove the evidence that aelfrice had already written. Nothing after the change removed that evidence either. The maintainer's store shows the residue ([#1270](https://github.com/robotrocketscience/aelfrice/issues/1270)). **Roughly one active belief in ten** sits at exactly `ingest_prior + 0.1 × hook_retrieval_count`. That value is the arithmetic signature of the old channel. Several hundred beliefs were exposed only *after* the change. Those beliefs sit unmoved at their ingest prior. That difference confirms the cutover, rather than an ongoing write. Read those posteriors as a historical artefact of a policy that this project abandoned. Do not read them as evidence of the current policy.

The project measured the residue first. Then the project left the residue alone. The measurement replays real prompts through `retrieve()` against the store, first with the residue and then without it. The returned pack changes for a substantial minority of the prompts. A byte-identical control changes the pack for none of them. Retrieval is therefore deterministic, and the residue is the cause of the difference. The project left the residue in place anyway. The argument for a correction was that inflated posteriors keep the cold tail out of the results. That argument failed its own test. The beliefs that the correction promotes are *under*-represented among the never-retrieved beliefs, relative to the store. The residue is therefore not what keeps the cold tail out. aelfrice retains every `feedback_history` row, so `0.1n` stays recomputable. The correction remains available if better evidence ever argues for it. The figures live on [#1270](https://github.com/robotrocketscience/aelfrice/issues/1270) with the procedure that produced them. The figures come from one store on one machine. They belong next to their script, and not in this document.

The posterior is **single-axis**. Each belief has one `(α, β)` pair, and not a vector. A wrong belief is wrong overall. A useful belief is useful overall. The research line shipped a multi-axis `UncertaintyVector`. That vector holds one `(α_i, β_i)` pair for each aspect: existence, semantics, mechanism and cost. The speculative-belief surface (`wonder`, `reason`) uses that vector. aelfrice stays single-axis at v3.x. Nobody reopened the multi-axis substrate question at [#196](https://github.com/robotrocketscience/aelfrice/issues/196) after v2.0 shipped. Assume single-axis when you port code.

The cost is the loss of dense semantic similarity. The first benefit is a learning loop that converges on what works *for you*, and not on what is textually similar. The second benefit is a retrieval pipeline that keeps determinism from end to end.

aelfrice has no exploration term in the default retrieval path. The research line required that ≥15% of retrievals surface high-uncertainty beliefs. That requirement stops one failure mode of the feedback loop: the loop reinforces the confident beliefs and never re-tests the uncertain beliefs. Any exploration mechanism (bandit-style, entropy-weighted, sampling-based) breaks the "same query, same beliefs" property. aelfrice gives that property a higher priority. Posterior reranking has shipped since v1.3 with no exploration term.

An exploration slot does ship, and it is **off by default**. `src/aelfrice/exploration.py` holds the mechanism, and `retrieval.py` resolves the `[retrieval] exploration_enabled` flag ([#1279](https://github.com/robotrocketscience/aelfrice/issues/1279), [#1176](https://github.com/robotrocketscience/aelfrice/issues/1176) proposal 5). The default stays off because the slot changes what goes into a live conversation, which is the determinism cost above. The reason the slot exists is coverage, and not ranking: most of the store has never been injected, so it can never earn evidence, and the slot is the intervention that breaks that loop. To change the default is a separate operator decision. That decision is gated on measured coverage growth, and not on a relevance score.

> Historical note: at v1.0–v1.2 aelfrice computed and stored the posterior, but L1 retrieval ranked by Best Matching 25 (BM25) alone. The v1.3 retrieval wave connected the posterior to the ranking. v1.7 made BM25F default-on. v3.0 made intentional clustering default-on. v3.3 made type-aware compression default-on. Feedback now changes what the agent sees, from end to end. See [LIMITATIONS](../user/LIMITATIONS.md).

## Locks, not just decay

Pure decay moves trusted ground truth toward the prior. To stop that movement, you must restate the ground truth again and again. That is the same failure mode as the failure mode of files. A locked belief must be exempt from every forgetting mechanism that the substrate runs. Without that exemption, a lock gives you nothing.

**What actually ships is narrower than that framing suggests. The distinction matters** ([#1218](https://github.com/robotrocketscience/aelfrice/issues/1218)). There is **no posterior decay**. Nothing moves a stored `(α, β)` toward the prior, at any time. The module `scoring.decay` implements decay. That module includes the lock short-circuit, which returns `(α, β)` unchanged for any age. No module under `src/` calls `scoring.decay`. The exemption therefore protects locks from a mechanism that does not run. Treat the decay code as design intent, and not as shipped behaviour. The disposition of that code is open under [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162). The two options are to connect it or to delete it.

The forgetting that does run acts on the **ranking position**. It never acts on the posterior. A lock ranks above that forgetting. **Entity-persistence demotion** ([#1096](https://github.com/robotrocketscience/aelfrice/issues/1096), resolver default-ON) moves short-lived coordination text with low grounding down the L1 rerank. The effect is mild. Low-value content therefore ranks below durable content. That is the one such mechanism on the default production path.

Two other mechanisms are narrower than they look. The first is the ranking-time temporal decay by age (`_apply_temporal_decay`, [#473](https://github.com/robotrocketscience/aelfrice/issues/473)). Only `retrieve_v2` reaches that decay, and only behind `temporal_sort`. `temporal_sort` defaults to off. Nothing in `src/` sets it. The second is the marker-edge demotion of the beliefs that `aelf doctor --detect-stale` tagged `POTENTIALLY_STALE`. That demotion is opt-in, because its producer is opt-in ([#1207](https://github.com/robotrocketscience/aelfrice/issues/1207)).

Two mechanisms keep a lock durable today, and decay-exemption is not one of them. The mechanisms are the **lock floor in the rerank path** and the overwrite semantics of `aelf lock`. You still do not restate a lock every month to keep it alive. Nothing decreases the strength of a lock in the first place.

Hard locks are rigid, though. The earlier v2.x design accumulated `demotion_pressure` on `CONTRADICTS` edges. That design demoted a lock automatically at a threshold. v3.2.0 removed that mechanism ([#814](https://github.com/robotrocketscience/aelfrice/issues/814) / PR #820, landed just after the v3.1.0 tag — see #833). The removal dropped the `demotion_pressure` column, the `apply_feedback(propagate=)` kwarg, and the `FeedbackResult.pressured_locks` and `.demoted_locks` fields. Lock correction now goes through an overwrite with `aelf lock` (per [#605](https://github.com/robotrocketscience/aelfrice/issues/605)). The other paths are the explicit `aelf unlock` and `aelf delete` commands. Durability is the property. The substrate trusts the user to be the one who changes a stale rule.

## Trust boundary at the hook surface

The `UserPromptSubmit` hook injects the retrieved beliefs into the
input of the model on every turn. That makes the hook a privileged channel.
The model reads any content inside the emitted block as elevated,
system-trusted context. The model reads content *adjacent to* the block in
the same way.

The task of the hook layer is to make that trust boundary structurally
clear. The task is not to police what the model does after it reads the
block. Four structural defenses ship today:

- A fixed framing tag marks every injected belief. The tag is
  `<belief id="…" lock="…">` inside `<aelfrice-memory>`. The tag splits the
  injected beliefs into **two trust tiers**. It frames the user-locked items
  as the standing instructions of the user. It frames every other belief as
  *retrieved data, not instructions: context to verify, not directives*
  ([#1163](https://github.com/robotrocketscience/aelfrice/issues/1163)).
- A `speculative="1"` attribute separates the beliefs that the memory system
  *synthesised* from the beliefs that somebody *asserted*. Machine conjecture
  therefore cannot present itself as an observation
  ([#1171](https://github.com/robotrocketscience/aelfrice/issues/1171)).
- A render-time escape pass neutralises any tag-substring that lands in
  stored belief content.
- A per-turn audit log (`hook_audit.jsonl`) records the exact rendered block.
  Post-hoc forensics can therefore answer "what was injected on turn N."

See [hook_hardening.md](../design/hook_hardening.md) for the design memo.

**Why the locked tier is framed as instructions, and not as data.** The
obvious hardening is to frame *everything* as data and never as instruction.
The project tried that framing. The project rejected it on measurement, and
not on taste. A blanket "data, not instructions" framing makes models refuse
to obey the user's own locked rules. That result defeats `aelf lock` as a
rules mechanism. A user who locks "never force-push to main" wants the agent
to obey that rule, not to evaluate it. The two-tier header exists because of
that result.

This document records the trade, because the code does not make it
self-evident. An audit that reads "standing instructions" without this
context will correctly identify the header as a prompt-injection surface.
The auditor will then revert the header. That revert breaks lock compliance
again, and nothing reports the break.

The instruction tier is safe because it is **user-authored by
construction**. Explicit user acts set the locks. Those acts are `aelf lock`,
the `/aelf:lock` slash form, and an `aelf review` verdict. `aelf promote`
moves the *origin*, and never the `lock_level`. Nothing ingested, inferred,
or synthesised reaches that tier on its own.

There is one exception. This document states it, because the property above
is exactly the property worth protecting. The setting
`AELF_AUTOLOCK_CORRECTIONS=1` makes the Stop hook lock part of the
lock-candidates of this session without asking. That population is wider
than the name of the flag suggests. `hook._belief_is_correction_class`
admits any belief whose `origin` is `agent_inferred` or `agent_remembered`.
It does not admit only the beliefs whose type is `type=correction`. The hook
also rewrites the `origin` to `user_stated`. Under that opt-in setting, a
belief that the agent inferred can enter the instruction tier. Nobody
asserted that belief.

The flag is off by default. The path that ships prompts the user instead of
locking. The property above holds for every default install. This setting is
the one setting that suspends the property.

Audit `_belief_is_correction_class`, **not** `_belief_is_lock_candidate`.
The two predicates separated in #1315. Candidacy is deliberately the wider
predicate. It admits every directive belief in the session, and that is
3,003 beliefs on this repository's own store. Everything that candidacy
admits beyond the correction class is *proposed only*. aelfrice never writes
those proposals. The filter that maintains that separation is at the
`_autolock_candidates` call site in `hook.stop()`. A document that named
candidacy here would report those 3,003 beliefs as auto-lockable. That is a
false alarm. A false alarm in that direction causes a reviewer to revert a
correct filter.

The hook layer cannot do two things, by design. It cannot enforce that the
model verifies named session artifacts before it acts. It cannot guarantee
that the model treats escaped tag-substrings as data. Those two items are
model-behaviour contracts. They belong in CLAUDE.md / AGENTS.md, and not in
`aelfrice` Python. A model can act on belief content as instruction *despite*
the framing tag. That behaviour is a model-layer problem. aelfrice exposes
the boundary. The model is responsible for honoring the boundary.

## Local, always

aelfrice keeps your corrections in one SQLite file on your machine. There is no cloud sync, no telemetry and no API call in the retrieval path. The cloud LLM at the other end of your prompt sees whatever aelfrice injects. That exposure is inherent. aelfrice limits the quantity of injected context instead of sending the whole memory. The default for hook-injected context is 1,500 tokens, scoped to the current query. The default for the `retrieve()` API is 2,400 tokens.

The 2,400-token default of the retrieval API is a calibrated choice, not an arbitrary one. The research line proposed this hypothesis: focused context gives better results than exhaustive context. A retrieval of approximately 2.4K tokens that selects the right beliefs should match or exceed a 10K-token dump of the full memory on response quality. That retrieval also uses approximately 4× fewer tokens for the memory mechanism. The v2.0 reproducibility cut re-measured that curve against the public retrieval pipeline. The 2,400-token budget is the default after v1.3. You configure it per call.

See [PRIVACY.md](../user/PRIVACY.md) for verifiable specifics.

## Small surface, on purpose

The v1 surface is small. Feedback-driven belief mutation goes through `apply_feedback`. Every lock goes through one path. When the system misbehaves, there is one place to look.

`apply_feedback` is the *primary* writer of `(α, β)`, and not the only one. This document is precise about the writers, because the aspiration and the code have separated before ([#1168](https://github.com/robotrocketscience/aelfrice/issues/1168)). Two other paths write `(α, β)`. Both paths are deliberate, and both leave an audit trail. The first is `clamp_ghosts.clamp_ghost_alpha`, a one-shot migration clamp. The second is the consolidation dedup pass. That pass sums the existing evidence when it collapses a duplicate group. The pass does not add new evidence.

A third path wrote `(α, β)` in the past. That path is `deferred_feedback.sweep_deferred_feedback`, the implicit retrieval lane, which [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162) made audit-only. The lane had no counterweight, because `scoring.decay` has no production caller. Exposure alone therefore moved the posterior of a frequently-retrieved belief upward without bound. The lane also contradicted [#1086](https://github.com/robotrocketscience/aelfrice/issues/1086), which had already decided that exposure is not evidence.

The invariants that matter hold across all of these paths:

- A user lock is a floor. No passive signal moves that floor.
- The belief of a federated peer is read-only on the local machine.
- Every posterior move that is not a merge writes a `feedback_history` row.

The posterior write itself is a single atomic SQL increment. It runs inside one `BEGIN IMMEDIATE` transaction together with its audit row. Concurrent writers therefore cannot lose the evidence of another writer. The log therefore cannot disagree with the projection.

The earlier research line had a much bigger surface: twenty-nine Model Context Protocol (MCP) tools, `wonder`, `reason`, and snapshot/diff. It delivered value. It also delivered ambiguity. The rebuild started narrow on purpose. Versions v1.x to v3.x have reintroduced breadth. That breadth is 15 MCP tools at v3.3, plus the `/aelf:wonder`, `/aelf:reason` and `/aelf:graph` slash surfaces. Each addition depends on evidence. That evidence is a benchmark, an experiment, or a clear case where the existing operations do not suffice.

## Lean dependencies, on purpose

aelfrice has three hard runtime dependencies. An argument justifies each dependency. `numpy` and `scipy` came in at v1.5 (#148) for the BM25 sparse-matvec retrieval lane. They now also do the math for holographic reduced representations (HRR) and the spectral-graph math. `snowballstemmer` came in at v1.7 (#154) for Porter stemming. Everything else is the Python standard library plus SQLite. The standard library already wraps SQLite. Optional extras add capability without entering the default install. `[onboard-llm]` is the classifier SDK for the direct API. `[archive]` is cryptography. `[benchmarks]` holds the dev-side adapters.

Every dependency is maintenance cost and attack surface. Larger components are a vector index, an embedding service and a neural reranker. Such a component enters aelfrice only when an experiment shows that the existing stack is the limiting part.

## What we can and can't guarantee

aelfrice is a memory substrate, not an LLM. This table gives the honest decomposition for any "the agent will follow this rule" claim:

| Tier | Mechanism | Guarantee |
|---|---|---|
| 1. Storage | SQLite write-ahead log (WAL) + locked belief | aelfrice writes the rule durably and never loses it. |
| 2. Injection | L0 always-loaded into every prompt | The rule is in the context of the model on every retrieval. |
| 3. Compression survival | Per-prompt L0 re-injection. The locks re-enter the context on the first prompt after a compaction. The opt-in post-compaction rebuilder (`aelf setup --rebuilder`) strengthens the mechanism. The host runs the rebuilder on `SessionStart(source="compact")` ([#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)). | The rule survives a compaction of the context window. |
| 4. Violation detection | Not implemented | — |
| 5. Violation blocking | Not implemented | — |
| 6. LLM compliance | The model actually obeys the injected rule | **Not under aelfrice's control.** |

Tiers 1–3 hold mechanically. Tiers 4–5 are post-execution detection and pre-execution blocking. Both are research-line capabilities, and neither has a current roadmap entry. Tier 6 is the LLM's own training and decoding, which aelfrice cannot constrain. If the model ignores an injected lock, the failure mode is in the model, and not in aelfrice. That distinction does not console a user whose agent just ran `git push` despite a clear directive.

The same substrate gives a few more properties:

- **Session recovery, not just write durability.** The SQLite WAL guarantees that every acknowledged write survives a crash. That is the storage-engine claim. The product-level claim is different: aelfrice can reconstruct the *working context* of an interrupted session on restart. The reconstruction does not use a snapshot file. It uses the same belief store that the next session retrieves against. A `<recent-work>` SessionStart sub-block augments that store. The sub-block carries the current branch, the last N commits, and any issue numbers that the recent work references (#887). Re-open the terminal next week. Ask "where were we?". The locks and the per-project working state are still there.
- **Confidence does not raise an automatic warning.** aelfrice does not surface a warning for a belief whose posterior drifts below 0.5. Negative evidence drives no automatic state change. Locked beliefs hold by design. v3.2.0 removed the v2.x auto-demote mechanism ([#814](https://github.com/robotrocketscience/aelfrice/issues/814)). To find out which beliefs are losing the feedback loop, run `aelf speculative` or `aelf review --generate`. `aelf speculative` lists the non-locked beliefs ranked by posterior evidence. `aelf review --generate` gives the review queue of the oldest unconfirmed beliefs. `aelf status` shows only aggregate counts. The system does not interrupt you to tell you.
- **Append-only substrate.** The research line had a separate `observations` table. That table was insert-only. It recorded permanently every observation that produced a belief. v1.x kept only `feedback_history` immutable. Every `apply_feedback` call writes a row there. Nothing ever updates a row in `feedback_history`. v1.5 (#205) added the append-only `ingest_log` table. v1.7 (#264) routed every ingest entry point through the derivation worker. The v2.0.1 view-flip (#265) made `beliefs` a materialized projection of the log. Feedback still mutates beliefs. Lifecycle operations such as retire and restore also mutate beliefs. The full write log is durable and replay-capable. See [`design/write-log-as-truth.md`](../design/write-log-as-truth.md) for the architectural memo.

  **The log does not cover the `edges` today. This is the one place where the substrate claim overreaches.** The contract ratified for the edges ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283), 2026-08-01) is *log-derived*. The edge set is to be recomputable from `ingest_log`, ordered by `(created_at, ingest_log ULID)`. That contract is the decision, and not the current state. As shipped, the contract is now partly true and mostly not true.

  [#1354](https://github.com/robotrocketscience/aelfrice/issues/1354) made `derive()` emit the intra-turn `DERIVED_FROM` edges from the `raw_meta` of the log row itself. `derived_edge_ids` therefore populates forward-only. The replay probe now counts an edge-set divergence as drift. The probe no longer files that divergence in the informational bucket. That change covers **at most 1.93%** of the live edge set.

  Writers outside the log still write the remainder. Those writers are `temporal_spine.py` (`TEMPORAL_NEXT`, 88.3%), the inter-turn `DERIVED_FROM` writer, and the relationship and contradiction detectors. The practical cost is unchanged in kind. A replay from empty still cannot reconstruct the L3 breadth-first search (BFS) graph, the temporal spine or the `CONTRADICTS` substrate.

  The writer uses the order `(created_at, rowid)`. The `rowid` is implicit here. VACUUM may renumber the `rowid`. For that reason the ratified key is the ULID of the log, and not a value read off the belief table.

## What this design buys

- **Continuity.** Close the terminal. Come back next week. Ask "where were we?". The memory restores the context.
- **Compounding.** Three sources fill the graph. The first is the explicit `onboard`, `lock` and `feedback` operations (since v1.0). The second is the default-on transcript-ingest, commit-ingest and session-start hooks (since v2.1 #529). The third is posterior-aware ranking (since v1.3, with BM25F default-on since v1.7 and type-aware compression default-on since #769). Every layer compounds: more sessions → more feedback → better ranking → more accurate retrieval.
- **Self-correction.** Entity-persistence demotion (default-on) demotes stale unlocked beliefs in the *ranking*. Their posteriors do not decay. Nothing ages a stored `(α, β)` (see *Locks, not just decay* above and [#1218](https://github.com/robotrocketscience/aelfrice/issues/1218)). Removal is explicit: use `aelf retire` (reversible) or `aelf delete`. Correct a wrong lock by overwriting it with `aelf lock`. Remove a wrong lock with `aelf unlock` or `aelf delete`. v3.2.0 removed the v2.x auto-demote mechanism ([#814](https://github.com/robotrocketscience/aelfrice/issues/814)).
- **Auditability.** Every belief has a content hash and a timestamp. Every feedback event has an audit row. Every score is `α / (α + β)`. Read the database. Reproduce the claims that the system makes about itself.
- **Locality.** There is no service that can fail. There is no account that you can lose. There is no quota that you can exceed.

## What it isn't

- aelfrice is not a notebook. aelfrice is not a team knowledge base.
- aelfrice is not a vector store. v1 retrieval has no semantic similarity.
- aelfrice is not a planner. The decisions stay in your head.
- aelfrice is not a replacement for documentation, conventions or runbooks. aelfrice is a complement to them.

aelfrice is a small tool for one job: it keeps the agent that helps you build software from forgetting what you said.
