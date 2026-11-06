---
name: aelf:speculative
description: List non-user-locked (L1) beliefs sorted by alpha descending — the agent-inferred and ingested layer.
allowed-tools:
  - Bash
---
<objective>
Inspect the L1 (non-user-locked) belief tier — beliefs that were ingested,
inferred, or wonder-generated but not yet explicitly asserted by the user.
</objective>

<process>
Run: `aelf speculative`

Optionally:
- `aelf speculative --origin <tag>` to restrict to a single origin tag.
- `aelf speculative --limit N` to cap the row count.
- `aelf speculative --json` for machine-readable JSONL output.

Display the output verbatim. Do not add commentary.
</process>
