---
name: aelf:wonder
description: Research a topic and persist findings as new speculative beliefs the next /aelf:reason can use, or (no topic) run a graph-walk consolidation pass.
argument-hint: Topic (e.g. "about session id propagation" or "indentation rules"), or graph-walk flags (--top 5, --emit-phantoms, --seed <id>, --gc, --persist)
allowed-tools:
  - Bash
  - Task
  - Write
---
<objective>
The research surface. Pair to `/aelf:reason`: where reason walks the graph you already have, wonder *grows* the graph by going off and learning things, so the next reason pass is richer.

Two modes, picked from the shape of the arguments:

1. **Bare positional topic** (e.g. `/aelf:wonder about indentation`) **or `--axes "<query>"` → axes-spawn-ingest research flow.** Run gap analysis against the topic, generate 2–6 orthogonal research axes, fan out one subagent per axis to produce research documents, then hand the documents back through `wonder_ingest` so each subagent's research lands as a speculative phantom belief anchored to the gap-surface seeds. This is the wonder consolidation dispatch loop (#542 E4 / #552 / #645). The topic may carry agent-count shorthand: `quick N-agent`, `deep N-agent`, or bare `N-agent` (e.g. `quick 2-agent wonder about indentation` → `agent_count=2`, query `"about indentation"`).
2. **No topic / flag-only** (`/aelf:wonder` with no args, or only flags like `--top 5`, `--emit-phantoms`, `--seed <id>`, `--gc`, `--persist`) **→ graph-walk consolidation.** Walk the belief graph from a deterministically-picked seed (or `--seed <id>`) and surface ranked consolidation candidates with suggested actions. Use `--emit-phantoms` to print Phantom JSON for offline review or `--persist` to write them to the store via `wonder_ingest`.
</objective>

<process>
**Dispatch on the shape of `$ARGUMENTS`:**

- **Bare positional topic** — `$ARGUMENTS` is non-empty and its first non-whitespace character is **not** `-`. The user typed something like `/aelf:wonder about indentation`. Run with the topic quoted as one CLI argument:

  ```bash
  uv run aelf wonder "$ARGUMENTS"
  ```

  The quotes are load-bearing — the CLI's positional `query` is a single argument; without them argparse rejects multi-word topics (`error: unrecognized arguments: ...`).

- **Empty or flag-only** — `$ARGUMENTS` is empty, or its first non-whitespace character is `-` (e.g. `--gc`, `--persist`, `--emit-phantoms`, `--axes "<query>"`). Run unquoted so the shell tokenises flags correctly:

  ```bash
  uv run aelf wonder $ARGUMENTS
  ```

**What to do with the output:**

- If stdout is JSON of shape `{gap_analysis, research_axes, agent_count, speculative_anchor_ids}` (the axes-spawn-ingest flow ran — either via a bare positional topic or via `--axes "<query>"`), proceed to the dispatch loop below.
- Otherwise (graph-walk consolidation output, `--gc` summary, etc.), display the output verbatim and stop.

**Dispatch loop** (axes flow only):

1. **Parse the JSON payload.** If `research_axes` is empty, print the payload and stop — there is nothing to dispatch.

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
