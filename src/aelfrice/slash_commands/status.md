---
name: aelf:status
description: Alias for `aelf:health` — runs the structural auditor.
allowed-tools:
  - Bash
---
<objective>
`aelf status` is an alias for `aelf health`. Same output, same exit
codes. Use whichever name fits your muscle memory.
</objective>

<process>
Run: `uv run aelf status`
Display the output verbatim. Do not add commentary.
</process>
