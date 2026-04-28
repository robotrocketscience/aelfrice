---
name: aelf:validate
description: Promote an agent_inferred belief to origin=user_validated.
argument-hint: The belief id (16-hex prefix as shown by aelf:search)
allowed-tools:
  - Bash
---
<objective>
Acknowledge an onboard-derived belief as correct without locking it.
A validated belief outranks unvalidated peers in contradiction
resolution but stays demotable on negative feedback. Lock instead
(`aelf:lock`) when the statement is non-negotiable.
</objective>

<process>
Run: `uv run aelf validate "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
