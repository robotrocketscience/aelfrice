---
name: aelf:wonder
description: Surface consolidation candidates and phantom-belief suggestions over the belief graph; with a query, run the axes-spawn-ingest research flow.
argument-hint: Optional query (e.g. "about X as it relates to Y") or graph-walk flags (--top 5, --emit-phantoms, --seed <id>, --gc, --persist)
allowed-tools:
  - Bash
  - Task
  - Write
---
<objective>
Two modes, picked from the arguments:

1. **No query → graph-walk consolidation.** Walk the belief graph from a deterministically-picked seed (or `--seed <id>`) and surface ranked consolidation candidates with suggested actions. Use `--emit-phantoms` to print Phantom JSON for offline review or `--persist` to write them to the store via `wonder_ingest`.
2. **With a positional query (or `--axes "<query>"`) → axes-spawn-ingest research flow.** Run gap analysis against the query, generate research axes, fan out one subagent per axis to produce research documents, then hand the documents back through `wonder_ingest` so each subagent's research lands as a speculative phantom belief anchored to the gap-surface seeds. This is the wonder consolidation dispatch loop (#542 E4 / #552 / #645). The query may carry agent-count shorthand: `quick N-agent`, `deep N-agent`, or bare `N-agent` (e.g. `aelf wonder "quick 2-agent wonder about indentation"` → `agent_count=2`, query `"about indentation"`).
</objective>

<process>
**If `$ARGUMENTS` does NOT contain `--axes`:**

Run: `uv run aelf wonder $ARGUMENTS`. Display the output verbatim. Do not add commentary.

**If `$ARGUMENTS` contains `--axes "<query>"`:**

1. **Get the dispatch payload.** Run `uv run aelf wonder $ARGUMENTS`. Stdout is JSON of shape `{gap_analysis, research_axes, agent_count, speculative_anchor_ids}`. Parse it. If `research_axes` is empty, print the payload and stop — there is nothing to dispatch.

2. **Fan out one subagent per axis.** For each axis in `research_axes`, spawn a Task subagent in parallel (send a single assistant message containing one Task tool use per axis). Each subagent's prompt should include:

   * The originating user query (`gap_analysis.query`).
   * The axis `name`, `description`, `search_hints`, and `gap_context`.
   * Instruction: produce a focused research document (a few paragraphs) that summarises what the subagent found about that axis. Plain text. No need for the subagent to commit code or write files.

3. **Collect responses into a JSONL file.** When all subagents have returned, write `/tmp/aelf-wonder-dispatch-<unix-ts>.jsonl`. One row per axis, shape:

   ```json
   {"axis_name": "<axis.name>", "content": "<subagent response>", "anchor_ids": [<speculative_anchor_ids verbatim from step 1>]}
   ```

   The `anchor_ids` array is identical across all rows — it is the gap-analysis seed set.

4. **Persist.** Run `uv run aelf wonder --persist-docs /tmp/aelf-wonder-dispatch-<unix-ts>.jsonl`. Display the resulting `inserted=N skipped=N edges_created=N` summary verbatim.

**Dedup behaviour:** `wonder_ingest` keys idempotency on the sorted constituent belief IDs **and** the generator string (key prefix `wonder_ingest:v2:`, shipped in #644). An N-axis dispatch over the same `speculative_anchor_ids` therefore persists as N distinct phantoms (one per axis-derived generator), not one. Re-running the same dispatch is still a no-op.
</process>
