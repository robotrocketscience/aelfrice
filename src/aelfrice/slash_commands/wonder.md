---
name: aelf:wonder
description: Surface consolidation candidates and phantom-belief suggestions over the belief graph; with a query, run the axes-spawn-ingest research flow.
argument-hint: Optional query (e.g. "about X as it relates to Y") or graph-walk flags (--top 5, --emit-phantoms, --seed <id>, --gc, --persist)
allowed-tools:
  - Bash
---
<objective>
Two modes, picked from the arguments:

1. **No query → graph-walk consolidation.** Walk the belief graph from a deterministically-picked seed (or `--seed <id>`) and surface ranked consolidation candidates with suggested actions. Use `--emit-phantoms` to print Phantom JSON for offline review or `--persist` to write them to the store via `wonder_ingest`.
2. **With a positional query → axes / research flow.** Run gap analysis against the query, generate research axes, and emit a dispatch-payload JSON suitable for the skill layer's research-agent fan-out (see `slash_commands/aelf:wonder` agentmemory-parity flow, #645). The query may carry agent-count shorthand: `quick N-agent`, `deep N-agent`, or bare `N-agent` (e.g. `aelf wonder "quick 2-agent wonder about indentation"` → `agent_count=2`, query `"about indentation"`).
</objective>

<process>
Run: `uv run aelf wonder $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
