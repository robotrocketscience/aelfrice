### Fixed

- **The seven injection lanes charge the line their own renderer emits, at unchanged budgets ([#1526](https://github.com/robotrocketscience/aelfrice/issues/1526)).** The seven are the ones `benchmarks/injection_budget_bytes.py` enumerates and packs, and they are the table below. The PreCompact rebuilder is not among them and is the one deliberate exception; it has its own paragraph. The injection budgets were denominated in belief-content characters while what reaches the model is a rendered line. A packed belief is never emitted as its content: on most lanes it is a `<belief …>` element on its own line, and that wrapper does not shrink when the content does — 52 characters for the per-turn line shape, with a 16-character belief id. So each block overran its cap by the wrapper's share of the line. The packers now charge the line, so each block comes in at or under the cap it claims.

  **Every budget constant is unchanged.** An earlier draft of this change re-tuned all seven so the emitted bytes would not move; two adversarial reviews refuted that, because byte-neutrality is not achievable with a fixed constant. What the correction removes from a pack is the ratio of a rendered line to its content — largest where beliefs are shortest, shrinking towards parity as they grow, and changing sign on the lanes that truncate — so a constant chosen to hold bytes flat holds them flat near one belief length and nowhere else. Re-tuning the *amount* of context each lane gets is deferred to its own issue, gated on retrieval quality rather than on a byte count.

  **What a budget buys, measured.** `benchmarks/injection_budget_bytes.py` is the producer for every figure here. It packs 300-belief synthetic stores — carrying user locks and speculative-origin beliefs, so the arms this change rewrote for those cases are exercised — under the pre-#1526 cost functions and under the shipped ones, **at the same unchanged budget**, and renders both through each lane's own renderer with compression at its production default of ON. The row below is each lane at the committed replay-soak corpus's 92-character median belief length. Run `uv run python benchmarks/injection_budget_bytes.py --curve` for the whole length grid; the deviation is not one multiplier and is not quoted as one.

  | Lane | Budget (unchanged) | Emitted bytes, before → after |
  |---|---|---|
  | `hook.DEFAULT_HOOK_TOKEN_BUDGET` (per-turn) | 1500 | 9259 → 6763 |
  | `hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` (`<core>`) | 1500 | 10335 → 5883 |
  | `hook.DEFAULT_SESSION_START_TOKEN_BUDGET` | 1500 | 1408 → 1408 (see below) |
  | `hook_search_tool.INJECTED_TOKEN_BUDGET` (Grep\|Glob) | 600 | 3305 → 2609 |
  | `hook_search_tool.BASH_INJECTED_TOKEN_BUDGET` | 300 | 1910 → 1678 |
  | `hook_agent_context.INJECTED_TOKEN_BUDGET` | 600 | 4586 → 3114 |
  | `retrieval.DEFAULT_TOKEN_BUDGET` (`aelf search`) | 2400 | 6960 → 6960 |

  <!-- derived: benchmarks/injection_budget_bytes.py#belief_line_wrapper_chars = 52 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#store_beliefs = 300 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#ups_budget = 1500 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#ups_bytes_before = 9259 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#ups_bytes_after = 6763 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#core_bytes_before = 10335 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#core_bytes_after = 5883 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#session_start_bytes_before = 1408 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#session_start_bytes_after = 1408 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_budget = 600 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bytes_before = 3305 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bytes_after = 2609 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bash_budget = 300 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bash_bytes_before = 1910 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bash_bytes_after = 1678 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#agent_context_bytes_before = 4586 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#agent_context_bytes_after = 3114 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#cli_search_budget = 2400 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#cli_search_bytes_before = 6960 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#cli_search_bytes_after = 6960 -->

  Two rows in that table are equalities, and neither is a null result. `retrieval.DEFAULT_TOKEN_BUDGET`'s two arms both end on the **L2.5 sub-budget** at this length, not on the budget in the row — the producer names which cap ended each pack rather than reporting a bare byte count, and away from 92 characters this lane moves. `hook.DEFAULT_SESSION_START_TOKEN_BUDGET`'s two arms both end on the **candidate pool**, at every length in the grid, which is the next item.

  **A seventh renderer, which does not emit `<belief …>` at all ([#1526](https://github.com/robotrocketscience/aelfrice/issues/1526)).** `hook_search_tool` emits `[L0] <id-prefix>: <content>` truncated to `PER_LINE_CHAR_CAP = 200`. Charging it the `<belief>` element made a 500-character belief cost 138 tokens for a line that is 200 characters — 51 tokens — long, and left the 23-character `[L0] <id-prefix>: ` prefix charged to nothing. `retrieve()` now takes a `belief_cost_fn`, and that lane passes one built from `_belief_line`, the same function the renderer calls, so the truncation and the escaping are charged because they happened rather than transcribed as a width. Because the old accounting charged *more* than this lane emits past the per-line cap, this is the one lane where the correction changes sign: it emits fewer bytes on short beliefs and more on long ones.

  **What changed, enumerated rather than summarised ([#1526](https://github.com/robotrocketscience/aelfrice/issues/1526)).** `retrieval._belief_tokens` — the base cost reached by the L1 tail pack, the L2.5 sub-budget pack, the HRR structural pack and `retrieve()`'s own cost closure — charges the rendered line, including the `speculative="1"` marker ([#1171](https://github.com/robotrocketscience/aelfrice/issues/1171)) on the beliefs that carry it. `retrieval.lock_injection_tokens` charges the same line on its verbatim arm and the manifest entry's indent and newline on its [#1016-B](https://github.com/robotrocketscience/aelfrice/issues/1016) reference-lock arm. `clustering._belief_tokens`, the default `cost_fn` for the cluster and max-coverage packs, is a deliberate duplicate of the first and moved with it — the claim that it mirrors is now a test rather than a docstring. `retrieve_with_tiers`'s cost closure adds the wrapper to a compressed render through `retrieval._render_wrapper_tokens`, a module-level name so a before/after measurement can reach it. `retrieval._l25_hits` takes the caller's `cost_fn` too, so the L2.5 sub-pack is no longer selected against a line shape the Grep\|Glob lane never emits. And the `<core>` packer costs its own rendered line exactly, by building it, because that line's width genuinely varies: `corr` and `posterior` are interpolated numbers.

  **What deliberately does not change ([#1526](https://github.com/robotrocketscience/aelfrice/issues/1526)).** `retrieval.LEGACY_TOKEN_BUDGET` (2000) exists to reproduce the v1.2 caller's budget byte-for-byte on the disabled-entity-index fallback, and re-denominating it would destroy the thing it is for. `retrieval.DEFAULT_L25_TOKEN_SUBBUDGET` stays the literal 400 it has always been. `aelfrice.benchmark.DEFAULT_TOKEN_BUDGET` is a harness constant, not an injection path. And the fixed framing header — 502 characters, 126 tokens, emitted ahead of the first belief by four formatters — is still charged to no budget: reserving it is a separable change, and an uncompensated one at unchanged budgets. How much it would take off each block is not quoted here, because the arm that measured it was reverted along with the reservation and no producer on this tree can re-derive it; the figures go to the follow-up issue, where they describe work not yet done.

  <!-- derived: benchmarks/injection_budget_bytes.py#framing_header_chars = 502 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#l25_token_subbudget = 400 -->

  **The rebuilder's self-report counts what the block emits ([#1526](https://github.com/robotrocketscience/aelfrice/issues/1526)).** `<retrieved-beliefs budget_used="N/M">` was counting belief text rather than rendered lines — a number no reader of the block could check against the block — and now reports what the section emits, so the numerator can exceed the denominator. `context_rebuilder._estimate_belief_tokens` is still on the old currency, and the reason given in an earlier draft — "correcting it would move this block's emitted bytes" — was not true: the block's bytes already move under this change, because `rebuild_v14` takes its non-locked candidates from `retrieve()`. Measured on the producer's stores: 9011 → 9011 bytes at 92 content characters, and 19293 → 16421 at 300. The real reason it is left alone is that `DEFAULT_REBUILDER_TOKEN_BUDGET` **is** reachable from `.aelfrice.toml` as `[rebuilder] token_budget`, unlike `[retrieval] token_budget`, which the hook shadows with an explicit kwarg. Re-denominating the currency a user's configured value is written in would cut their rebuild block without a word; that needs a migration or a currency marker on the key, not a constant edit.

  <!-- derived: benchmarks/injection_budget_bytes.py#rebuild_bytes_before_92 = 9011 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#rebuild_bytes_after_92 = 9011 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#rebuild_bytes_before_300 = 19293 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#rebuild_bytes_after_300 = 16421 -->

  One knob still does not reach the hook: `[retrieval] token_budget` in `.aelfrice.toml`. The resolver ranks an explicit argument above TOML and the hook always passes one (`aelf search` shadows it the same way, its `--budget` having a default). `docs/user/CONFIG.md` says so at the key.
