---
name: aelf:health
description: Run the structural auditor and report orphan threads, FTS5 sync, and locked-belief contradictions plus informational metrics.
allowed-tools:
  - Bash
---
<objective>
Run the v1.1.0 structural auditor against the local memory store. Each
of three checks (orphan threads, FTS5 sync drift, locked-belief
CONTRADICTS pairs) reports ok or FAIL; the command exits 1 if any
check fails. Informational metrics (counts, average confidence, credal
gap) are printed alongside but do not affect exit status. For the v1.0
regime classifier output, use `aelf regime`.
</objective>

<process>
Run: `uv run aelf health`
Display the output verbatim. Do not add commentary.
</process>
