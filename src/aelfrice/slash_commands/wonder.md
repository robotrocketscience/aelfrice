---
name: aelf:wonder
description: Surface consolidation candidates and phantom-belief suggestions over the belief graph; integration with store deferred to v2.x.
argument-hint: Optional flags (e.g. --top 5, --emit-phantoms, --seed <id>)
allowed-tools:
  - Bash
---
<objective>
Walk the belief graph from a deterministically-picked seed (or `--seed <id>`) and surface ranked consolidation candidates with suggested actions. Use `--emit-phantoms` to print Phantom JSON for offline review.
</objective>

<process>
Run: `uv run aelf wonder $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
