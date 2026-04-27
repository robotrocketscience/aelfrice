---
name: aelf:health
description: Run the regime classifier and report the brain's current operating mode.
allowed-tools:
  - Bash
---
<objective>
Diagnose the brain's regime — supersede vs. ignore vs. balanced — based
on aggregate confidence, lock density, and edge density.
</objective>

<process>
Run: `uv run aelf health`
Display the output verbatim. Do not add commentary.
</process>
