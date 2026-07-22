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
by host subagent tasks — no separate API key, no MCP roundtrip, no extra
billing. Classification runs on a low-cost model by default and is
user-selectable at run time (step 3a). Falls back to the regex
classifier on `--no-subagents` or when the subagent tool is unavailable.
</objective>

<process>
1. Parse `$ARGUMENTS`. If it contains `--no-subagents`, run
   `uv run aelf onboard "<path>" --llm-classify=false` (deterministic regex path) and display the output
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
   numbers). When you fill in the block, name the classifier model the
   user selected in the model-choice step below — render the string
   verbatim (alias or full model id) so the printed `<model>` matches
   the step 4 dispatch and the per-tier pricing you showed.

   **Token estimate.** For each batch:
   - `input_tokens ≈ ceil((1200 + sum(len(s["text"]) + len(s["source"]) + 30 for s in batch)) / 4)`
     where `1200` is the rough character length of the classification
     prompt template in step 4, and `/4` is the standard "~4 chars per
     token" approximation from the provider's public token glossary.
   - `output_tokens ≈ 40 * len(batch)` — one short JSON entry per
     sentence (`{"index": N, "belief_type": "...", "persist": ...}`).

   Sum across all batches.

   **Cost estimate (per tier).** The token counts above are fixed; only
   the per-token rate changes with the model. Compute the run's cost for
   each of your host's model tiers — a low-cost tier, a mid tier, and a
   top tier — at each model's currently published per-token rates from
   the provider's public pricing page. (As a low-cost-tier anchor: a
   rate around `$1/MTok input, $5/MTok output` puts a few-thousand-
   sentence run in the low-cents range; refresh from the pricing page,
   and price the mid/top tiers from their own current rates.)

3a. **Choose the classifier model.** Before dispatching, let the user
   pick which model tier classifies the batch. Present the three tiers —
   naming the concrete model your host offers for each — with the
   per-tier run cost from the estimate above:

   - **low-cost tier — default, recommended.** The current behaviour.
   - **mid tier.**
   - **top tier.**

   State the trade-off plainly: this task is short-label classification
   (typing each sentence as `factual` / `preference` / `requirement` /
   `correction` plus a keep/drop bit), and higher-tier models show
   **strongly diminishing returns** on quality for it — the extra spend
   rarely changes the labels. Recommend the low-cost tier.

   Ask via the host's question mechanism which tier to use. **If the
   user does not answer, or the run is non-interactive / scripted,
   proceed with the low-cost tier** — the prompt is an optional
   override, never a hard block, so unattended onboarding still works.
   Carry the chosen model id into step 4's dispatch and the estimate
   block's `<model>` / cost line.

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

4. **Classify each batch via Task subagents on the chosen model.** For
   each batch, invoke the Task tool with:
   - `subagent_type`: `general-purpose`
   - `model`: the model id chosen in step 3a (the low-cost tier by
     default)
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
- Zero direct calls to the model provider's HTTP API. Classification is
  performed by host subagent tasks drawing from the user's existing
  session (on the model chosen in step 3a) — no API key, no separate
  billing.
- All beliefs land typed (`factual` / `preference` / `requirement` /
  `correction`); none are `pending_classification=True`. The store-
  layer dedup keys by `(text, source)` so re-running is idempotent.
- `--no-subagents` is the deterministic-fallback escape hatch: same
  semantics as `aelf onboard <path>` from a plain shell.
</notes>
