---
name: aelf:onboard
description: Scan a project directory and ingest beliefs into aelfrice memory.
argument-hint: Path to the project directory (e.g. . or ~/projects/myapp). Append --no-subagents to force the regex classifier.
allowed-tools:
  - Bash
  - Task
---
<objective>
Onboard a project into aelfrice using LLM-quality classification driven
by Haiku Task subagents — no `ANTHROPIC_API_KEY` required, no MCP
roundtrip, no extra billing. Falls back to the regex classifier on
`--no-subagents` or when Task is unavailable.
</objective>

<process>
1. Parse `$ARGUMENTS`. If it contains `--no-subagents`, run
   `uv run aelf onboard "<path>"` (regex path) and display the output
   verbatim. Stop. Otherwise extract `<path>` (everything that isn't
   `--no-subagents`).

2. **Emit candidates.** Run:
   `uv run aelf onboard "<path>" --emit-candidates`
   Parse the JSON. Capture `session_id`, `n_already_present`, and
   `sentences` (a list of `{index, text, source}` objects).
   If `sentences` is empty, run accept with an empty classification
   list to close the session, print a one-line summary
   (`onboarded <path>: 0 added, <n_already_present> already present`),
   and stop.

3. **Batch.** Split `sentences` into batches of at most 50 entries.

4. **Classify each batch via Haiku Task subagents.** For each batch,
   invoke the Task tool with:
   - `subagent_type`: `general-purpose`
   - `model`: `haiku`
   - `description`: e.g. `Classify onboard batch 1/N`
   - `prompt`: the classification template below, with `<BATCH_JSON>`
     replaced by the JSON-encoded batch.

   Classification template:
   ```
   You are classifying candidate sentences extracted from a code repo
   into beliefs for a memory system. For each sentence, decide:

   1. `belief_type` — one of:
      - "factual"      — a statement of fact about the repo or project
      - "preference"   — a stated preference or convention
      - "requirement"  — a "must" / "should" / hard rule
      - "correction"   — a fix-up of a prior fact ("not X, actually Y")
   2. `persist` — true if this sentence should be stored as a belief,
      false if it is too noisy / generic / off-topic to keep.

   Respond with ONLY a JSON array, no prose, no markdown fences:
   [{"index": <int>, "belief_type": "<one of the four>", "persist": <bool>}, ...]

   Include one entry per input sentence, preserving the original
   `index` value. Do not invent indices.

   Sentences:
   <BATCH_JSON>
   ```

   The subagent must return a JSON array. Parse it; if parsing fails,
   retry once with a "respond with JSON only, no prose" reminder. If
   it still fails, drop that batch (the sentences will be counted as
   `skipped_unclassified`).

5. **Aggregate.** Concatenate all per-batch arrays into one list of
   `{index, belief_type, persist}` entries.

6. **Apply.** Pipe the aggregated JSON into:
   `uv run aelf onboard --accept-classifications --session-id <id> --classifications-file -`
   The CLI prints a JSON summary with `inserted`,
   `skipped_non_persisting`, `skipped_existing`,
   `skipped_unclassified`.

7. **Display.** Print one human-readable line summarising the result,
   e.g.:
   `onboarded <path>: <inserted> added, <skipped_non_persisting>
   skipped (filtered out), <skipped_existing> skipped (already
   present), <skipped_unclassified> skipped (unclassified),
   <n_already_present> pre-existing`.

If the Task tool is unavailable in this host, fall back to step 1's
regex path with a one-line notice: `aelf: Task tool unavailable;
falling back to regex classifier.`
</process>

<notes>
- Zero direct calls to `https://api.anthropic.com/`. Classification is
  performed by Haiku Task subagents drawing from the user's existing
  Claude Code session — no API key, no separate billing.
- All beliefs land typed (`factual` / `preference` / `requirement` /
  `correction`); none are `pending_classification=True`. The store-
  layer dedup keys by `(text, source)` so re-running is idempotent.
- `--no-subagents` is the deterministic-fallback escape hatch: same
  semantics as `aelf onboard <path>` from a plain shell.
</notes>
