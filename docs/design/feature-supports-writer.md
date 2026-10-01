# Feature spec: a deterministic SUPPORTS writer for conversation (#1653)

**Status:** spec. No code yet. The decisions below were made with the operator on 2026-09-30.
**Issue:** [#1653](https://github.com/robotrocketscience/aelfrice/issues/1653). Its consumer is [#1650](https://github.com/robotrocketscience/aelfrice/issues/1650), phantom promotion.
**Substrate prereqs:**
- `EDGE_SUPPORTS` in `models.py`;
- `belief_corroborations`, which records a session id per corroboration;
- the #1647 sentiment lane;
- the #1649 ingest gate;
- the relevance detector (`relevance_detection.score_references`), whose matcher must be fixed first (#1655).

---

## Purpose

Almost no beliefs carry meaning-based edges. Measured on 2026-09-30 (method and data in the #1653 comments):
- On the cleaned store (248 active beliefs), the CONTRADICTS detector finds no pairs. On the polluted store, two blind graders put its 303 pairs at about 0% precision.
- The store holds 0 SUPPORTS edges (`SELECT type, COUNT(*) FROM edges GROUP BY type`). SUPPORTS is written only by `triple_extractor`'s phrase match on "supports" and "is supported by".

So no conversational evidence reaches a belief through an edge, and its confidence stays at its ingest prior.

Promotion counting exists but isn't safe to automate. `store.find_promotable_phantoms` already finds phantoms with at least 3 corroborations from at least 2 sessions and no CONTRADICTS edge, and `phantom_promotion_opportunity` shows them to you as a note. That count accepts every corroboration source type, including `wonder_ingest`, which wonder writes when it creates each phantom, and `claude_memory_mirror`, which mirrors text the agent writes into its own memory files. New transcript ingest skips assistant turns (#785), but assistant rows written before that were never purged. Automating promotion on that count would let the model promote its own claims.

This spec defines a deterministic writer for SUPPORTS edges, and the confidence change each edge carries. It adds no model, network call, or embedding.

## What counts as support

There are four sources, each recorded with its kind and weighted by tier:

| Source | Tier | What happens |
|---|---|---|
| You restate a belief in a **later session**, in text you typed | strong | **Exact restatement:** ingest already resolves it to the existing belief and writes a `belief_corroborations` row with the new session id. That row is the support; no edge is written, because the supporter and the target are the same belief. **Reworded restatement:** it becomes a new belief, and gets a SUPPORTS edge from the new belief to the existing one. |
| A test or check passes and its result matches a belief | weak | The check result is stored as an **evidence belief** under a dedicated origin (see "Storage"), with a SUPPORTS edge to the belief it matches. |
| A commit implements a belief | weak | The commit is ingested as today. A SUPPORTS edge goes from the commit's belief to each belief it matches. |
| The user praises the previous answer | weak | A SUPPORTS edge goes from the praise turn's belief to each injected belief whose words appear in the praised answer. It's never written to every injected belief. |

A restatement within the **same** session isn't support. Repeating yourself in one conversation is emphasis, not new evidence.

Only the strong tier comes from you alone. The agent writes the tests it passes and the commits it makes, so those sources are weak: they add an edge and a confidence move, but no promotion credit.

## Confidence

Writing a support moves the target's confidence (α) by tier:
- **Strong:** +1.0.
- **Weak, check or commit:** +0.5.
- **Weak, praise:** nothing more. The #1647 sentiment lane already applied the praise.

These deltas are starting values with no measurement behind them. The precision-gate measurement for each source sets the final value.

Rules for the move:
- **No propagation.** Support moves only its target, never neighbors along edges. One praise mustn't ripple across beliefs that merely sit near each other.
- **Lock floor.** A locked belief isn't moved, per the #1168 floor.
- **One channel.** The move goes through `apply_feedback` with `propagate=False` (its default is `True`) and a source string per tier, so it's audited.
- **CI audit.** This is a new automatic posterior channel. `benchmarks/posterior_channel_audit.py` must list it, along with its default.

## Phantom promotion (consumer: #1650)

This reverses part of the #229 rule, which made a corroboration count a non-trigger for promotion. The operator ruled on 2026-09-30 that phantoms need an automatic path, on evidence that the model can't produce itself.

A speculative (phantom) belief is promoted automatically when all of the following hold:
- it has at least **3 strong supports**. A strong support is either a corroboration row whose speaker is you (an exact restatement) or a SUPPORTS edge from a belief whose text you typed (a reworded restatement);
- those supports come from at least **2 different sessions**;
- it has no CONTRADICTS edge;
- it has no `feedback_history` row with negative valence.

The thresholds match `find_promotable_phantoms`' defaults (`_DEFAULT_MIN_CORROBORATIONS=3`, `_DEFAULT_MIN_SESSIONS=2`). The writer reuses that query with two changes:
- **An allowlist, not a denylist.** A corroboration row counts only when its source type is `transcript_ingest` **and** its speaker is the user. Every other source type never counts: `wonder_ingest`, `commit_ingest`, `cli_remember` and `mcp_remember` (the agent can run both), `filesystem_ingest`, `consolidation_migration`, and `claude_memory_mirror`. A source type added later doesn't count until it's added to the allowlist.
- **A speaker column.** A `transcript_ingest` row doesn't record who spoke today, so the implementation adds it first. A row with no speaker, including every row written before the column exists, never counts.

It also adds the reworded-restatement edges to the count, with each edge's session taken from the supporting belief's ingest session.

A promoted belief gets a new origin, `evidence_promoted`. It's added to `ORIGINS` and gets `ORIGIN_RETRIEVAL_PRIORITY` 3, tying with `user_transcript`: above a phantom (default 2) and below `user_validated` (4), so it never claims that you validated it. The tie fits, because its evidence is your own typed restatements. `promote()` stamps `user_validated` today, so it needs an origin parameter or a sibling function.

On promotion, the next injection of that belief carries a one-line note saying it was promoted automatically, so you can retire it if it's wrong. Promotion writes a `feedback_history` row, so it's audited and can be undone. The #1650 experiment may tune the thresholds.

## Storage

- **Evidence beliefs** from check results get their own origin, for example `check_evidence`, and they are **never injected**. They exist to be supporters. That keeps #1649's rule intact: tool output never reaches the model as if it were the user's belief.
- **Edges** are written with a stable direction (supporter → target), and duplicates are skipped. `anchor_text` records the source kind and the matched span, so every edge can be explained.

## Precision gate

Each source's matcher (reworded restatement, commit match, check match, praise targeting) has to pass the project's standing two-grader rule, as used for #1647, before it writes by default:
- a pre-registered sample of at most 200 edges it would write;
- two blind graders;
- at least 70% precision on **both** graders.

Until it passes, a source stays behind its own opt-in flag. Edges it would have written are logged, not applied, so it can be measured again later. That's the same pattern as #1647's `negative_disabled`.

## Dependencies to resolve first

1. **Praise targeting needs the relevance detector to work.** It has marked 0 of 1,384 injections as referenced (measurement in #1655). Until #1655 is fixed, praise writes no SUPPORTS edges: the fallback is nothing, not every injected belief.
2. **Check results need a capture point.** Since #1649, tool output is never ingested as user text. The writer needs a narrow hook on test and CI results that produces evidence beliefs only.

## Left to the implementation

- **Matching:** the algorithm and thresholds for "restates", "implements", and "matches". It has to be deterministic, and its precision is gated as above.
- **Timing and cost:** when the writer runs (at ingest, at the Stop hook, or as a sweep), with performance bounds and batching. #1674 is a reminder that hook-path costs compound.

## Out of scope

- **Ranking:** retrieval that reads SUPPORTS edges, such as a boost for supported beliefs. It's separate work after the writer exists.
- **Model pass:** a model proposing edges. It's measured only after the deterministic writer ships (operator ruling, 2026-09-30).
- **CONTRADICTS:** it stays off. See #1653 for the measurements.
