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

Almost no beliefs carry meaning-based edges. On the cleaned store (248 active beliefs on 2026-09-30), the CONTRADICTS detector finds no pairs. On the polluted store, its 303 pairs were about 0% precise (#1653). SUPPORTS has one writer, a commit-message phrase match, and on this store it has written 0 edges. So no conversational evidence ever reaches a belief:
- its confidence stays at its ingest prior;
- #1650 has nothing to count when it decides whether to promote a phantom.

This spec defines a deterministic writer for SUPPORTS edges, and the confidence change each edge carries. It adds no model, network call, or embedding.

## What counts as support

There are four sources, each recorded with its kind and weighted by tier:

| Source | Tier | What happens |
|---|---|---|
| The user restates a belief in a **later session** | strong | **Exact restatement:** ingest already resolves it to the existing belief and writes a `belief_corroborations` row with the new session id. That row is the support; no edge is written, because the supporter and the target are the same belief. **Reworded restatement:** it becomes a new belief, and gets a SUPPORTS edge from the new belief to the existing one. |
| A test or check passes and its result matches a belief | strong | The check result is stored as an **evidence belief** under a dedicated origin (see "Storage"), with a SUPPORTS edge to the belief it matches. |
| A commit implements a belief | medium | The commit is ingested as today. A SUPPORTS edge goes from the commit's belief to each belief it matches. |
| The user praises the previous answer | weak | A SUPPORTS edge goes from the praise turn's belief to each injected belief whose words appear in the praised answer. It's never written to every injected belief. |

A restatement within the **same** session isn't support. Repeating yourself in one conversation is emphasis, not new evidence.

## Confidence

Writing a support moves the target's confidence (α) by tier:
- **Strong:** +1.0.
- **Medium:** +0.5.
- **Weak:** nothing more. The #1647 sentiment lane already applied the praise.

Rules for the move:
- **No propagation.** Support moves only its target, never neighbors along edges. One praise mustn't ripple across beliefs that merely sit near each other.
- **Lock floor.** A locked belief isn't moved, per the #1168 floor.
- **One channel.** The move goes through `apply_feedback`, with a source string per tier, so it's audited.
- **CI audit.** This is a new automatic posterior channel. `benchmarks/posterior_channel_audit.py` must list it, along with its default.

## Phantom promotion (consumer: #1650)

A speculative (phantom) belief is promoted to a trusted origin automatically when all of the following hold:
- it has at least **3 strong supports**;
- those supports come from at least **2 different sessions**;
- it has no CONTRADICTS edge and no `harmful` feedback.

These numbers are the starting point. The #1650 experiment may tune them.

On promotion, the next injection of that belief carries a one-line note saying it was promoted automatically, so you can retire it if it's wrong. Promotion writes a `feedback_history` row, so it's audited and can be undone.

## Storage

- **Evidence beliefs** from check results get their own origin, for example `check_evidence`, and they are **never injected**. They exist to be supporters. That keeps #1649's rule intact: tool output never reaches the model as if it were the user's belief.
- **Edges** are written with a stable direction (supporter → target), and duplicates are skipped. `anchor_text` records the source kind and the matched span, so every edge can be explained.

## Precision gate

Each source's matcher (reworded restatement, commit match, check match, praise targeting) has to pass the standing rule before it writes by default:
- a pre-registered sample of at most 200 edges it would write;
- two blind graders;
- at least 70% precision on **both** graders.

Until it passes, a source stays behind its own opt-in flag. Edges it would have written are logged, not applied, so it can be measured again later. That's the same pattern as #1647's `negative_disabled`.

## Dependencies to resolve first

1. **Praise targeting needs the relevance detector to work.** It has marked 0 of 1,384 injections as referenced (#1655). Until #1655 is fixed, praise writes no SUPPORTS edges: the fallback is nothing, not every injected belief.
2. **Check results need a capture point.** Since #1649, tool output is never ingested as user text. The writer needs a narrow hook on test and CI results that produces evidence beliefs only.

## Left to the implementation

- **Matching:** the algorithm and thresholds for "restates", "implements", and "matches". It has to be deterministic, and its precision is gated as above.
- **Timing and cost:** when the writer runs (at ingest, at the Stop hook, or as a sweep), with performance bounds and batching. #1674 is a reminder that hook-path costs compound.

## Out of scope

- **Ranking:** retrieval that reads SUPPORTS edges, such as a boost for supported beliefs. It's separate work after the writer exists.
- **Model pass:** a model proposing edges. It's measured only after the deterministic writer ships (operator ruling, 2026-09-30).
- **CONTRADICTS:** it stays off. See #1653 for the measurements.
