---
name: aelf:promote
description: Promote an agent_inferred belief to user_validated origin. Alias of aelf:validate with identical semantics.
argument-hint: The belief ID to promote
allowed-tools:
  - Bash
---
<objective>
Promote an agent_inferred belief to the user_validated origin tier. This is
a provenance change only — alpha/beta posteriors are unchanged. Idempotent:
promoting an already-validated belief exits 0 with an "already validated"
message. This command is a first-class alias of `aelf validate`.
</objective>

<process>
Run: `uv run aelf promote "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
