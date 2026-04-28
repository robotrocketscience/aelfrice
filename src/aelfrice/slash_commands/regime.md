---
name: aelf:regime
description: Print the v1.0 regime classifier output (supersede / ignore / mixed / insufficient_data).
allowed-tools:
  - Bash
---
<objective>
Print the brain's current regime label as computed by the v1.0
classifier. Five features (confidence mean / median, mass mean, lock
density, thread density) are scored against published anchors and
aggregated into one of four labels. Informational only; never affects
exit status. For structural auditing (orphan threads, FTS5 sync, locked
contradictions), use `aelf health`.
</objective>

<process>
Run: `uv run aelf regime`
Display the output verbatim. Do not add commentary.
</process>
