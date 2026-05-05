---
name: aelf:confirm
description: Affirm an existing belief — bumps the Beta-Bernoulli posterior toward truth without freezing it (use aelf:lock for ground-truth freeze).
argument-hint: The belief ID to confirm
allowed-tools:
  - Bash
---
<objective>
Explicitly affirm a belief: apply one unit of positive feedback (α += 1.0)
via the `user_confirmed` source. The belief's posterior mean moves toward 1
without being locked. Distinct from `aelf:lock`, which freezes the belief as
ground-truth; use `confirm` when you want to nudge the posterior without the
commitment of a lock. Not persisted to `belief_corroborations` — that table
tracks duplicate re-ingests; `confirm` is an explicit user signal written to
`feedback_history`.
</objective>

<process>
Run: `uv run aelf confirm "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
