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

2. **Pre-scan and emit candidates.** First run the read-only pre-scan:
   `uv run aelf onboard "<path>" --check`
   Print the output verbatim so the user sees the idempotency state
   before any session opens. Parse the `new since last onboard: <N>
   candidates` line; if `N == 0`, stop — no session is opened, no
   classification work is dispatched (#761).

   Otherwise run:
   `uv run aelf onboard "<path>" --emit-candidates`
   Parse the JSON. Capture `session_id`, `n_already_present`, and
   `sentences` (a list of `{index, text, source}` objects).
   If `sentences` is empty, run accept with an empty classification
   list to close the session, print a one-line summary
   (`onboarded <path>: 0 added, <n_already_present> already present`),
   and stop.

3. **Batch.** Split `sentences` into batches of at most 50 entries.

   Then, **before dispatching any subagents in step 4**, print a single
   human-readable block that explains what is about to happen, how long
   it will take, and what it costs. Without this block, the wall of
   "12 background agents launched" log lines reads as scary, even
   though classification is cheap and finishes in a few minutes.

   Compute everything from the actual batched payload (not fixed
   numbers). When you fill in the block, name the actual classifier
   model from the `model:` field of step 4's dispatch — render it
   literally as `<model>` from that field so the user sees the
   concrete id in the printed output.

   **Token estimate.** For each batch:
   - `input_tokens ≈ ceil((1200 + sum(len(s["text"]) + len(s["source"]) + 30 for s in batch)) / 4)`
     where `1200` is the rough character length of the classification
     prompt template in step 4, and `/4` is the standard "~4 chars per
     token" approximation from the provider's public token glossary.
   - `output_tokens ≈ 40 * len(batch)` — one short JSON entry per
     sentence (`{"index": N, "belief_type": "...", "persist": ...}`).

   Sum across all batches.

   **Cost estimate.** Price at the classifier model's currently
   published per-token rates from the provider's public pricing page.
   As of 2026-05 the model named in step 4 priced at
   `$1.00/MTok input, $5.00/MTok output`; refresh from the pricing
   page if pricing has moved.

   **Time estimate.** Subagents dispatch in parallel waves of ~12.
   Each wave takes ~30-90 seconds wall-clock — the classifier itself
   is fast, but subagent spawn / queue overhead dominates per-token
   latency. So:
   Let `waves = ceil(n_batches / 12)`.
   `wall_time_range ≈ [waves * 30s, waves * 90s]`
   Render as a range, e.g. "~2-6 min" for 46 batches, "~30-90s" for
   ≤12 batches.

   **Print exactly one block**, in this shape:
   ```text
   ═ aelf onboard: parallel classification ═
   what  <S> candidate sentences extracted from <path>. Each one needs
         to be typed (factual / preference / requirement / correction)
         or filtered as noise, so retrieval later can prioritize real
         facts over boilerplate.
   how   <N> batches of ≤50 sentences, dispatched to `<model>`
         subagents in parallel waves of ~12 — that's why you'll see
         "12 background agents launched" repeated several times below.
   time  ~<X>-<Y> min  (subagent dispatch + classifier latency)
   cost  ~<Ti>K input + ~<To>K output tokens
         ≈ $<X.XX> equivalent API spend
         billing: runs against your existing subagent quota
                  (Max / Pro / Team) — no separate API charge.
   ════════════════════════════════════════
   ```
   Round token counts to nearest K and cost to two decimals.
   `<path>` is the path passed to the slash command; `<model>` is the
   model id from step 4's `model:` field.

   This is an upper-bound order-of-magnitude estimate, not a guarantee;
   actual usage depends on per-batch retries and the subagent's system
   prompt overhead, which the host can't observe. The point is to
   replace "how much is this going to cost? how long will it take?"
   with numbers up front.

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
