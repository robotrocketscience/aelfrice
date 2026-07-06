# Configuration: `.aelfrice.toml`

Most users never need this file. The defaults are tuned so `uv tool install aelfrice && aelf onboard .` does the right thing.

This is the reference for power users whose project has a documentation idiom or naming convention the default filter mishandles.

## What it does

A single optional TOML file at the root of a project (or any ancestor). It exposes the following power-user surfaces:

- `[noise]` — onboard-time belief filter. Changes how `aelf onboard` ingests beliefs; nothing else.
- `[retrieval]` (v1.3+) — retrieval-time tier toggles + ranking. Knobs: `entity_index_enabled` (L2.5), `bfs_enabled` (L3), `posterior_weight` (partial Bayesian-weighted L1 ranking), `l1_limit` + `token_budget` (the #1045 wide-retrieval knobs — BM25 candidate cap + token budget, default 50/2400; raise both together for multi-hop recall), `use_bm25f_anchors` (BM25F-with-anchor-text since v1.7), `use_heat_kernel` (authority scoring lane, default-on since v2.1), `use_hrr_structural` (HRR structural-query lane, default-on since v2.1), `hrr_persist` (HRR structural-index on-disk persistence, default-on since v3.0), `use_type_aware_compression` (per-belief retention-class compression, default-on since #769), `use_intentional_clustering` (co-locating related beliefs, default-on since v3.0), `expansion_gate_enabled`, `use_gamma_posterior_temperature` (default off), and `use_zeta_posterior_rerank` (default off; mutually exclusive with the γ flag — `retrieve()` raises `ValueError` when both are on), `use_temporal_spine` + `temporal_spine_budget` (the #1064 chronological-adjacency lane, default off/32; pairs with `[ingest] write_temporal_spine`), `use_entity_persist_demote` (the #1096 entity-persistence demotion / organic-sink rerank modifier, default off), `use_origin_tiebreak` (the #1089 origin-priority within-tier tie-break, default off). Two placeholder flags (`use_signed_laplacian`, `use_posterior_ranking`) are recognised but emit a deprecation warning if set — their lanes have not yet shipped.
- `[rebuilder]` (v1.4+) — context-rebuilder knobs: `turn_window_n` (default 50), `token_budget` (default 4000), `trigger_mode` (`manual`|`threshold`|`dynamic`, default `threshold`), `threshold_fraction` (default 0.6), and `query_strategy` (v1.7+, default `stack-r1-r3` since v3.0). `[rebuild_floor]` (v1.7+) sets the token-budget floors for the session-scoped and L1 belief lanes (`[rebuild_floor] session` and `[rebuild_floor] l1`).
- `[onboard.llm]` (v1.3.0+) — direct-API onboard classifier gate; documented under [Keys § `[onboard.llm]`](#onboardllm-v130) below.
- `[cadence]`, `[implicit_feedback]`, and `[hook_audit]` — feedback-cadence scoring, deferred retrieval-exposure feedback, and the per-turn hook audit log. Recognised here but documented in their module docstrings (`src/aelfrice/cadence.py`, `src/aelfrice/deferred_feedback.py`, `src/aelfrice/hook.py`).
- `[feedback]` (v3.0+) — feedback-lane opt-ins. `sentiment_from_prose` (default `false`) wires the sentiment-feedback detector into `UserPromptSubmit` (#606).
- `[user_prompt_submit_hook]` (v3.0+) — UPS hook knobs. `prompt_shape_gate_enabled` (default `true`) gates trivial-prompt and system-envelope short-circuits before BM25 retrieval runs (#674). `conversation_aware_query_enabled` (default `true`, v3.x #909) folds a small window of recent dialog turns into the BM25 query so paraphrase / pronoun / numeric-reference follow-ups still surface the load-bearing thread; tuned by `conversation_aware_turn_window` (default `4`) and `conversation_aware_prompt_weight` (default `3`).

Locks and the MCP tool surface are not affected. Hook behavior IS configurable here (`[user_prompt_submit_hook]`, `[feedback]`, `[cadence]`, `[hook_audit]`); the Bayesian update math itself is not.

`scan_repo` walks up from the scan root looking for `.aelfrice.toml`. The first one found wins. The walk stops at the filesystem root — there is no global / per-user config.

If the file does not exist, the noise filter uses defaults. That is the recommended state.

## Schema

```toml
# .aelfrice.toml
[noise]
# Turn off any of: headings | checklists | fragments | license
disable = []

# Drop paragraphs with fewer than this many whitespace tokens.
# Default 4. Set to 0 to disable the fragment check entirely.
min_words = 4

# Drop paragraphs containing any of these whole words.
# Word-bounded, case-insensitive. "jso" does NOT match "json".
exclude_words = []

# Drop paragraphs containing any of these substrings.
# Literal substring match, case-insensitive.
exclude_phrases = []

[retrieval]
# v1.3+. Default-on at v1.3.0. Enables the L2.5 entity-index tier
# between L0 locked beliefs and L1 FTS5 BM25. Set to false to
# disable (alongside the AELFRICE_ENTITY_INDEX=0 env-var off-switch).
entity_index_enabled = true

# v1.3+. Default-OFF at v1.3.0. Enables the L3 BFS multi-hop graph
# traversal layered on top of L0+L2.5+L1. Set to true to opt in
# (alongside the AELFRICE_BFS=1 env-var on-switch). Bounded by
# max_depth=2, nodes_per_hop=16, total_budget_nodes=32, and a
# 0.10 path-score floor; shares the unified token budget.
bfs_enabled = false

# v1.3+. Default 0.5. Posterior-weighted ranking on the L1 BM25
# tier: score = log(-bm25) + posterior_weight * log(posterior_mean).
# Set to 0.0 to reproduce v1.0.x BM25-only ordering byte-for-byte.
# AELFRICE_POSTERIOR_WEIGHT env var overrides; explicit kwargs on
# retrieve() / retrieve_v2() override TOML in turn. Locked beliefs
# (L0) bypass scoring entirely.
posterior_weight = 0.5

# #1045. Wide-retrieval knobs — the multi-hop RECALL lever. `l1_limit`
# is the BM25 candidate cap (default 50); `token_budget` is the retrieval
# token budget (default 2400). Raising l1_limit recovers multi-session /
# temporal answers a 50-candidate slice misses (LongMemEval-S 58.8% ->
# 68.6% at l1_limit=200 / token_budget=8000), but ONLY when the budget is
# raised too — candidates cap at l1_limit BEFORE the budget trim, so
# budget alone is inert. Both default to the latency-sensitive hot-path
# values; raising them widens retrieval (more recall, more tokens, more
# latency), best for retrieval-heavy / large-context callers rather than
# the per-prompt injection hook. AELFRICE_L1_LIMIT and
# AELFRICE_RETRIEVAL_TOKEN_BUDGET env vars override; explicit kwargs on
# retrieve() / retrieve_v2() override TOML in turn.
#
# Measured characterization (LongMemEval oracle, 364 questions across dev
# + held-out confirmation; per-turn gold labels; deterministic reruns):
#   - The knobs are a PAIR. l1_limit=200 + token_budget=8000 lifts
#     whole-set recall by +8.6pp (dev) / +9.5pp (held-out) over the
#     defaults. l1_limit alone at the default budget is nearly inert
#     (~+1pp): the extra candidates are trimmed before they can matter.
#   - Recovery plateaus at l1_limit=200; 400 adds ~nothing for 40% more
#     packed tokens.
#   - Top-rank ordering is UNAFFECTED (MRR / recall@1 identical to three
#     decimals in every measured cell): widening only adds beliefs deep
#     in the packed set. Consumers that read the whole retrieval block
#     benefit; consumers that act on the top few results will see no
#     change.
#   - Cost at 200/8000: ~4-5.7x injected tokens vs defaults. This is why
#     the defaults stay put and wide retrieval is opt-in.
l1_limit = 50
token_budget = 2400

# v1.7+. Default `true` since v1.7.0 (#154 bench gate). Enables the
# BM25F sparse-matvec L1 path that augments belief content with
# anchor text (#142) under Porter-stemmed FTS5 indexing. Set to
# false to fall back to the v1.5/v1.6 FTS5-BM25 path.
# AELFRICE_BM25F=0 env var overrides.
use_bm25f_anchors = true

# Default `true` since the #154 composition tracker flipped the
# default after the #437 reproducibility-harness gate cleared at
# 11/11. Enables the heat-kernel authority scoring lane (#150). Set
# to `false` for parity with the pre-flip ranking.
# AELFRICE_HEAT_KERNEL=0 env var overrides.
use_heat_kernel = true

# Default `true` since the #154 composition tracker flipped the
# default after the #437 reproducibility-harness gate cleared at
# 11/11. Enables the HRR structural-query lane (#152). Set to
# `false` for parity with the pre-flip ranking.
# AELFRICE_HRR_STRUCTURAL=0 env var overrides.
use_hrr_structural = true

# v3.0+. Default `true`. Persists the HRR structural-index
# (struct.npy + meta.npz) to <store_dir>/.hrr_struct_index/ so
# warm starts mmap the matrix instead of rebuilding (~38s at
# N=50k → ~1s warm-load per #553). Auto-disabled when the store
# root resolves under /tmp/, /var/tmp/, /dev/shm/, or /run/.
# AELFRICE_HRR_PERSIST env var overrides (truthy/falsy match);
# AELFRICE_HRR_PERSIST=1 forces persistence even on ephemeral
# paths.
hrr_persist = true

# v2.1+ (#434), default `true` since #769 (A2 + A4 bench gates
# cleared on the lab-side compression_a* corpora). Type-aware
# compression: populates RetrievalResult.compressed_beliefs with
# per-belief renderings keyed by retention_class (snapshot →
# headline, transient → stub, fact + locked → verbatim). The
# pack-loop budget rewrite accounts in compressed rendered_tokens
# so a tight budget admits more transient/snapshot beliefs at
# their stub/headline cost. Composes with use_intentional_clustering
# since #878. AELFRICE_TYPE_AWARE_COMPRESSION=0 reverts.
use_type_aware_compression = true

# v3.0+ (#436). Default `true` since the multi-store production sweep
# cleared 60/60 PASS at p99 0.328 ms (~15-30x margin under the 5 ms A4
# latency budget). Co-locates related beliefs in the packed retrieval
# output so multi-fact queries surface a coherent neighborhood. Set to
# `false` for v2.0.x parity. AELFRICE_INTENTIONAL_CLUSTERING=0 env
# var overrides.
use_intentional_clustering = true

# v4.0.0+ (#1064). Default `false` — landing posture, not the end state:
# the default-ON flip is gated on the pre-registered #1064 criteria
# (see docs/design/feature-temporal-spine.md). When true, the
# temporal-spine lane traverses TEMPORAL_NEXT chronological chains from
# the top-5 packed L1 seeds (both directions, depth 1) and appends the
# neighbours after the L1 candidates. Reaches gold that shares zero
# salient terms with the question through chronological adjacency —
# confirmed +14.6pp gold-coverage on LoCoMo, 10x its shuffled control.
# No-op on stores with zero TEMPORAL_NEXT edges (run `aelf spine
# backfill` to build the spine on an existing store, and enable the
# [ingest] write_temporal_spine writer to keep it growing).
# AELFRICE_TEMPORAL_SPINE env var overrides.
use_temporal_spine = false

# v4.0.0+ (#1064). Node budget for the temporal-spine lane traversal
# (default 32). The confirmatory budget curve is monotone (~+2.5pp
# coverage per doubling at 32/64/128, no plateau) — this is the knob to
# raise for retrieval-heavy callers with the token budget to hold the
# extra candidates. AELFRICE_TEMPORAL_SPINE_BUDGET env var overrides.
temporal_spine_budget = 32

# Placeholder flags reserved by #154 — recognised so callers can
# write forward-compat config, but their lanes have not yet
# shipped. Setting either to true emits a one-shot stderr
# deprecation warning via warn_placeholder_flags() and is
# otherwise a no-op.
# use_signed_laplacian = false
# use_posterior_ranking = false

[ingest]
# v4.0.0+ (#1064). Default `false` (same flip gate as use_temporal_spine
# above — the two flags flip together at release time but resolve
# independently). When true, every belief insert chains to its session
# predecessor with a TEMPORAL_NEXT edge (src = successor, weight 0.8),
# building the per-session temporal spine the retrieval lane traverses.
# One edge per belief, O(1) per insert. Off-path ingest is
# byte-identical. AELFRICE_TEMPORAL_SPINE_WRITE env var overrides.
write_temporal_spine = false

[rebuilder]
# v3.0+ / #718 (PR #719). Selects the query-rewriting stack used by
# the context rebuilder. Default `"stack-r1-r3"` since v3.0; runs
# entity expansion + per-store IDF clipping via aelfrice.query_understanding.
# Set to `"legacy-bm25"` for the v1.4-byte-identical escape hatch.
# `"legacy-bm25"` remains available as an escape hatch; its removal
# (PR-4 of #291) is deferred and currently unscheduled.
query_strategy = "stack-r1-r3"

[rebuild_floor]
# v1.7+ (#289 / #364). Token-budget composite-score floors applied
# during context rebuilding. Malformed values (wrong type, negative)
# fall back to the default with a stderr trace; the rebuild never
# raises on a bad floor value.
#
# Minimum composite score for a session-scoped (L2) belief to be
# packed into the rebuilt block. 0.0 = no floor (pack everything).
# Default 0.10.
session = 0.10

# Minimum composite score for an L1 / L2.5 belief to be packed.
# 0.0 = no floor. Default 0.40.
l1 = 0.40

[feedback]
# v3.0+ (#606). Default `false`, opt-in. When true, the
# UserPromptSubmit hook runs the regex sentiment detector against
# each prompt and applies +/- valence feedback against the prior
# turn's retrieved beliefs (single-session window — cross-session
# propagation is explicit follow-up work). Fail-soft: any internal
# error returns 0 and never surfaces into the UPS hook contract.
# AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1 env var overrides. See
# docs/design/v3_sentiment_feedback_hook.md.
sentiment_from_prose = false

[user_prompt_submit_hook]
# v3.0+ (#674). Default `true`. Short-circuits BM25 retrieval on two
# prompt shapes: system-envelope echoes (prompts that start with a
# <task-notification> / <system-*> / <tool-result> tag) and trivial
# acks (stripped length < 12, <= 2 words after punctuation strip,
# or a normalized match against a fixed 16-entry ack set: "yes",
# "ok", "continue", "keep going", etc.). When the gate fires,
# hits = [] and the hook-audit row records the reason
# (prompt_shape_gate_skip="trivial:ack:yes",
# "system-tag:<task-notification>", etc.). The session-start
# sub-block is preserved unaffected. Set to false to disable.
prompt_shape_gate_enabled = true
# v3.x (#909). Default `true`. Conditions the per-prompt BM25 query on
# a small window of recent dialog turns, so paraphrase / pronoun /
# numeric-reference follow-ups still surface the load-bearing thread
# (the topic vocabulary the prompt lacks lives in the conversation
# history). The current prompt is repeated `conversation_aware_prompt_weight`
# times so its terms stay dominant; the last `conversation_aware_turn_window`
# turns are appended once. Fail-soft: any error reading turns falls back
# to the prompt-only query. Set enabled to false for v3.2-and-earlier
# prompt-only behaviour.
conversation_aware_query_enabled = true
# Number of trailing turns folded into the query (default 4). Kept small
# on purpose: a large window re-buries the thread on topic-drift.
conversation_aware_turn_window = 4
# Prompt repeat count for BM25 term-frequency weighting (default 3,
# minimum 1). Higher = the current prompt dominates the appended turns.
conversation_aware_prompt_weight = 3

[onboard.llm]
# v1.3.0+; default flipped to true in v1.5.1 (#238). Host-driven
# classification routes through the host model's Task tool — no API
# key required for the default path. The direct-API path (when the
# host has no Task tool reachable) requires the [onboard-llm] extra
# and the ANTHROPIC_API_KEY env var. To opt out entirely, set this
# to false or pass --llm-classify=false. See docs/design/llm_classifier.md
# and docs/user/PRIVACY.md § Optional outbound calls.
enabled = true

# Hard cap on total input+output tokens per onboard run.
# Default: 200_000. Run aborts mid-stream if exceeded; already-
# classified candidates remain in the store and an idempotent
# re-run resumes from the cap-hit point. 0 disables the cap.
max_tokens = 200_000

# Model id. Pinned by default to keep classification stable
# across releases. Override only if you have a reason.
model = "claude-haiku-4-5-20251001"
```

Unknown keys and unknown tables are ignored; the file is forward-compatible.

## Keys

### `disable`

| Token | Disables | Effect |
|---|---|---|
| `headings` | the "every line is a markdown heading" filter | pure heading blocks pass through |
| `checklists` | the "every line is `- [ ]`" filter | task-list items become belief candidates |
| `fragments` | the `min_words` short-paragraph filter | short labels like `DRAFT` pass to the classifier |
| `license` | the seven-signature license-preamble filter | LICENSE.md text becomes belief candidates |

A disabled category is silent — `ScanResult.skipped_noise` will not count anything from it. Other categories still fire. Unrecognised tokens are silently ignored.

### `min_words`

Integer, default `4`. Paragraphs shorter than this are dropped.

| Setting | Use when |
|---|---|
| `4` (default) | Most projects. |
| `3` or lower | You lock terse rules ("prefer composition", "no global state"). |
| `0` | Disables the check entirely. |

A non-integer value is rejected with a stderr warning; default applies.

### `exclude_words`

List of whole-word matches. Word boundaries: `"jso"` matches the standalone token but not `json`, `jsonify`, etc. Useful for initials, codenames, status keywords.

### `exclude_phrases`

List of literal substring matches. Case-insensitive but otherwise verbatim. Useful for templated header lines (`Last updated:`, `Generated by`) and inline status flags (`TODO:`, `FIXME`).

The trade-off vs. `exclude_words`: phrase match is a literal substring, no word boundaries. `["foo"]` here would drop a paragraph containing `foobar`.

### `[onboard.llm]` (v1.3.0+)

Host-driven LLM classifier for onboard ingest. Replaces the default regex `classify_sentence` path with the host model's Task tool (no API key required) — direct-API fallback gated by the `[onboard-llm]` extra. Default at v1.5.1+ (#238): on (`enabled = true`); soft-fallback to the regex classifier when no host Task tool is reachable. Boundary policy: [`docs/design/llm_classifier.md`](../design/llm_classifier.md). Privacy: [`docs/user/PRIVACY.md § Optional outbound calls`](PRIVACY.md#optional-outbound-calls).

| Key | Type | Default | Effect |
|---|---|---|---|
| `enabled` | bool | `true` (since v1.5.1, #238; was `false` v1.3.0–v1.5.0) | Turns on the LLM path when no `--llm-classify` flag is passed. The CLI flag wins on conflict; pass `--llm-classify=false` to force regex even when this is `true`. |
| `max_tokens` | int | `200_000` | Hard cap on total input+output tokens per onboard run. The classifier aborts mid-stream when exceeded; already-classified candidates persist and the deterministic belief id ensures a re-run resumes idempotently. `0` disables the cap (power-user). |
| `model` | str | `"claude-haiku-4-5-20251001"` | Anthropic model id. Pinned by default. Override only if you have a reason — classification recall and the few-shot block are calibrated against the pinned model. |

All three keys are optional; missing keys take their defaults.

The four-gate boundary policy is non-negotiable. Setting `enabled = true` is one of the four gates; the others are: `[onboard-llm]` extra installed, `ANTHROPIC_API_KEY` set in the environment, and a one-time per-machine consent prompt accepted (sentinel at `~/.aelfrice/llm-classify-consented`). The consent prompt re-fires when the model id changes or when aelfrice's MAJOR version changes.

```toml
# Example: opt in for this project, raise the cap, leave model pinned.
[onboard.llm]
enabled = true
max_tokens = 500_000
```

Auth, model selection, and provider choice are NOT configurable here. `ANTHROPIC_API_KEY` is read only from the environment, never from this file. There is no provider abstraction layer; only Anthropic's Haiku is supported in v1.3.0.

## Worked examples

```toml
# Filter a contributor's initials without breaking `json` mentions
[noise]
exclude_words = ["jso"]
```

```toml
# Let terse beliefs through on a dense rule project
[noise]
min_words = 2
```

```toml
# Filter templated boilerplate the default doesn't catch
[noise]
exclude_phrases = ["Last updated:", "Generated by tool-x", "DO NOT EDIT"]
```

```toml
# License-heavy project (a legal-tech tool, an OSS-compliance app)
[noise]
disable = ["license"]
```

## `[retrieval]` (v1.3+)

### `entity_index_enabled`

Boolean, default `true` at v1.3.0. Toggles the L2.5 entity-index retrieval tier.

When enabled:
- `retrieve()` extracts entities (file paths, identifiers, branch names, version strings, URLs, error codes, noun phrases) from the query.
- Looks them up in the `belief_entities` SQL table.
- Returns matched beliefs above L1 BM25, ranked by entity-overlap count.
- Default token budget rises from 2,000 to 2,400 to make room for the L2.5 sub-budget (400 tokens) on top of the unchanged L1 sub-budget (2,000 tokens).

When disabled (TOML `false`, or `AELFRICE_ENTITY_INDEX=0`, or explicit `entity_index_enabled=False` kwarg on `retrieve()`):
- L2.5 does not fire; output is byte-identical to v1.2's L0+L1 path.
- Default token budget snaps back to 2,000 if the caller did not pass an explicit budget.

Precedence (first decisive wins): env var `AELFRICE_ENTITY_INDEX=0` > explicit Python kwarg > TOML > default `true`.

The on-write index is always populated regardless of this flag — disabling only affects reads. Re-enabling sees an up-to-date index without a backfill pass.

### `posterior_weight`

Float ≥ 0, default `0.5` at v1.3.0. Combines the L1 BM25 score with the Beta-Bernoulli posterior mean log-additively:

```
score = log(-bm25_raw) + posterior_weight * log(posterior_mean(α, β))
```

`-bm25_raw` flips SQLite FTS5's signed score to positive (smaller-magnitude-negative is better in SQLite; we negate before taking `log`). `posterior_mean(α, β) = α / (α+β)` reuses the existing scoring helper — Jeffreys prior, reads `0.5` for unobserved beliefs.

Behaviour at the boundaries:

- **`0.0`** — score collapses to `log(-bm25_raw)`, byte-identical to v1.0.x `ORDER BY bm25(beliefs_fts)` ordering. Use for diff-tooling and bisection.
- **`0.5`** (default) — synthetic-graph optimum from the v1.3 calibration. Posterior moves rank without overwhelming BM25.
- **`> 1.0`** — posterior dominates; high-confidence beliefs surface even on weak keyword matches. Useful when feedback density is high and BM25 noise is the limiting factor.

Locked beliefs (L0) bypass scoring entirely; the weight only reranks the L1 BM25 candidate set. L2.5 entity-index hits and L3 BFS expansions are unaffected.

Precedence (first decisive wins): env var `AELFRICE_POSTERIOR_WEIGHT=<float>` > explicit Python kwarg `posterior_weight=<float>` on `retrieve()` / `retrieve_v2()` > TOML `[retrieval] posterior_weight` > default `0.5`.

### `use_entity_persist_demote`

Boolean, default `false` (v4.0+, [#1096](https://github.com/robotrocketscience/aelfrice/issues/1096)). Enables the **entity-persistence demotion lane** — a deterministic *organic sink* for the #1086 junk-percolation problem (junk ranks up, not down), applied as a log-additive rerank modifier over the ranked candidate tiers.

For each entity-bearing candidate it reads a grounding score `S1 = durable / (durable + transient + 1)` from the `belief_entities` index (one batched query over the candidate set), then applies the penalty `min(0, log(S1 + ε))`. Beliefs that ground only to *transient* coordination tokens (bare PR/issue numbers, version/branch tags) are demoted below those that ground to *durable* entities (file paths, error codes, symbol identifiers). It is a **pure demotion** — well-grounded beliefs are neutral, never boosted — and touches only entity-bearing candidates, so entity-free durable content (docstrings, formulae) is never penalised. Measured separation on a 118-belief hand-labelled set: durable vs ephemeral S1 mean 0.56 vs 0.06, lifting the durable-above-ephemeral ranking AUC from 0.48 to 0.87.

The sink is **content-referential, not temporal**: a time/recency decay sink was measured empirically inert for this workload (the junk is *hot*, not stale), so this lane — not cold-hibernation — is the organic sink. Deterministic per #605 (an entity-index join, no embeddings), byte-identical when unset.

Precedence (first decisive wins): env var `AELFRICE_ENTITY_PERSIST_DEMOTE=1` > explicit Python kwarg `use_entity_persist_demote=<bool>` on `retrieve()` / `retrieve_v2()` > TOML `[retrieval] use_entity_persist_demote` > default `false`. The default-ON flip is gated on a retrieval-bench no-regression check and is a separate operator call.

### `use_origin_tiebreak`

Boolean, default `false` (v4.0+, [#1089](https://github.com/robotrocketscience/aelfrice/issues/1089)). Enables the **origin-priority tie-break**: when two ranked candidates tie on relevance, the higher-trust *origin* wins (e.g. a belief curated from a `user`/`feedback` fact file outranks one auto-captured from a chat transcript).

This is a within-tier **tie-break**, never a primary rerank term — the origin key sits *between* the relevance score and the id tie-break, so relevance always dominates and byte-identical behaviour is preserved when the flag is off. It applies in both ranked tiers (the L1 FTS rerank and the L2.5 entity-index overlap). Deliberately *not* an origin *rerank lane* — that was refuted on LoCoMo in #1013 (the failure there was a BM25 recall limit, which reranking cannot fix). Deterministic per #605.

Precedence (first decisive wins): env var `AELFRICE_ORIGIN_TIEBREAK=1` > explicit Python kwarg `use_origin_tiebreak=<bool>` on `retrieve()` / `retrieve_v2()` > TOML `[retrieval] use_origin_tiebreak` > default `false`. Note a single-provenance corpus (e.g. LoCoMo) shares one origin tier, so the tie-break is inert there; the default-ON flip is a separate operator call.

Negative values clamp to `0.0`. Non-numeric env values trace to stderr and fall through. The cache key is extended with the resolved weight (rounded to four decimals), so two callers passing different weights against the same store do not collide on a shared `RetrievalCache`.

BM25F-only L1 shipped default-on at v1.7.0 (see `use_bm25f_anchors`); the heat-kernel and HRR-structural lanes shipped default-on at v2.1.0 once the #154 composition-tracker bench gate cleared 11/11 against the #437 reproducibility-harness corpus (see `use_heat_kernel` and `use_hrr_structural` below). See [`docs/design/bayesian_ranking.md`](../design/bayesian_ranking.md) for the v1.3 contract and the rejected-alternatives analysis.

### `bfs_enabled`

Boolean, default `false` at v1.3.0. Toggles the L3 BFS multi-hop graph traversal retrieval tier.

When enabled:
- After L0+L2.5+L1 are packed, `retrieve()` walks outbound edges from those seeds.
- Each visited belief scores `product(BFS_EDGE_WEIGHTS[edge.type])` along its path.
- Bounded by `max_depth=2`, `nodes_per_hop=16`, `total_budget_nodes=32`, `min_path_score=0.10`.
- Edge-type weights bias the frontier toward decisional edges: SUPERSEDES 0.90, CONTRADICTS 0.85, DERIVED_FROM 0.70, SUPPORTS 0.60, CITES 0.40, RELATES_TO 0.30.
- BFS expansions append to the same packed output, consuming the same `token_budget` as the prior tiers in score-descending order.
- `RetrievalResult.bfs_chains` exposes the edge-type path that reached each L3 expansion.

When disabled (the v1.3.0 default):
- L3 does not fire; output is byte-identical to the L0+L2.5+L1 baseline.

Precedence (first decisive wins): env var `AELFRICE_BFS=1`/`0` > explicit Python kwarg > TOML > default `false`.

The flag ships default-OFF at v1.3.0 because the literature-default edge weights have not yet been calibrated against the v1.2 corpus. A v1.3.x patch may re-tune them; the default-on flip is deferred until benchmark uplift is confirmed. See [bfs_multihop.md](../design/bfs_multihop.md) for the full spec, including the temporal-coherence limitation.

### `use_bm25f_anchors`

Boolean, default `true` since v1.7.0 (#154 bench gate). Enables the BM25F sparse-matvec L1 path that augments belief content with anchor text (#142) under Porter-stemmed FTS5 indexing.

When enabled (the v1.7.0+ default):
- L1 retrieval uses the BM25F implementation in `retrieval.py`, indexing belief text alongside its anchor terms (entity mentions, source paths, identifier captures).
- `LaneTelemetry.bm25f_used = True` for the call.
- Composition-tracker (#154) bench measured **+0.6650 NDCG@k uplift** versus the all-flags-off baseline on the `tests/corpus/v2_0/retrieve_uplift/v0_1.jsonl` lab fixture (30 rows, 6 categories).

When disabled:
- L1 falls back to the v1.5/v1.6 FTS5-BM25 path. `LaneTelemetry.bm25f_used = False`.

Precedence (first decisive wins): env var `AELFRICE_BM25F=0`/`1` > explicit Python kwarg `use_bm25f_anchors=<bool>` > TOML `[retrieval] use_bm25f_anchors` > default `true`.

### `use_heat_kernel`

Boolean, default `true` since the #154 composition tracker flipped the default after the #437 reproducibility-harness gate cleared at 11/11. Enables the heat-kernel authority-scoring lane (#150). Set to `false` for parity with the v2.0.x ranking.

Precedence (first decisive wins): env var `AELFRICE_HEAT_KERNEL=0`/`1` > explicit Python kwarg > TOML `[retrieval] use_heat_kernel` > default `true`.

### `use_hrr_structural`

Boolean, default `true` since the #154 composition tracker flipped the default after the #437 reproducibility-harness gate cleared at 11/11. Enables the HRR structural-query lane (#152). Wired into `retrieve_v2` as a parallel routing branch (per spec: not blended with the textual lane). When on, `retrieve_v2` parses the query for a structural marker before any other rewrite or lane fans out:

```
query string -> parse_structural_marker
              hit:  HRRStructIndex.probe(kind, target_id) -> RetrievalResult
              miss: textual lane (BM25F + heat-kernel + BFS)
```

A marker is a leading uppercase edge-type token followed by `:` and a non-empty target belief id. Recognised kinds match `aelfrice.models.EDGE_TYPES` — the full current set is `SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`, `RELATES_TO`, `DERIVED_FROM`, `IMPLEMENTS`, `TEMPORAL_NEXT`, `TESTS`, `RESOLVES`; treat the constant as the source of truth. Case-sensitive: `contradicts:b/abc` does not match and falls through to the textual lane on the literal string. Whitespace inside the target is preserved; leading/trailing whitespace on the query is stripped.

Examples:

| Query | Routes to | Returns |
|---|---|---|
| `CONTRADICTS:b/abc` | structural lane | beliefs whose outgoing edge of kind `CONTRADICTS` targets `b/abc`, ranked by HRR probe score |
| `SUPPORTS:b/xyz` | structural lane | beliefs that `SUPPORTS` `b/xyz` |
| `contradicts everything` | textual lane | BM25 over the literal string |
| `CONTRADICTS: ` (empty target) | textual lane (marker rejected by regex) | BM25 over the literal string |
| `CONTRADICTS:nonexistent_id` | textual lane (marker parsed but probe finds no edges) | BM25 over the literal string |

On structural lane hit, locked beliefs (when `include_locked=True`) pin to the head of the result and bypass the budget per the existing public-API contract; HRR-ranked beliefs are appended in score-descending order until the token budget is exhausted. Beliefs already in the locked set are de-duped from the HRR tail.

Long-running consumers should pass an explicit `hrr_struct_index_cache: HRRStructIndexCache | None` to amortise the per-belief HRR encode cost across queries. None falls through to a fresh build per call. The cache subscribes to the store's invalidation registry so any belief / edge mutation drops the index transparently.

Precedence (first decisive wins): env var `AELFRICE_HRR_STRUCTURAL=0`/`1` > explicit Python kwarg `use_hrr_structural=<bool>` > TOML `[retrieval] use_hrr_structural` > default `true`. The flip landed when the #437 reproducibility-harness reached 11/11 (see #154). Set the flag to `false` for parity with the v2.0.x ranking.

### `hrr_persist`

Boolean, default `true` (v3.0+, #698). Toggles HRR structural-index persistence. When enabled, `HRRStructIndexCache` writes the built `(N, dim)` matrix to `<store_dir>/.hrr_struct_index/struct.npy` (plus the `meta.npz` metadata blob) on first build and `np.load(..., mmap_mode='r')`s it on every subsequent cold start — turning the ~38 s rebuild at N=50k into a ~1 s warm-load per `docs/design/feature-hrr-integration.md`. The save is atomic via temp-file + `os.replace` so readers never observe a partial write.

**Ephemeral-path auto-disable** (#695). When the store root resolves under one of `/tmp/`, `/var/tmp/`, `/dev/shm/`, or `/run/`, the cache treats `hrr_persist` as if it were explicitly `false` and logs once per process:

```
aelfrice: HRR persistence disabled on ephemeral path <path>; set AELFRICE_HRR_PERSIST=1 to force.
```

Set `AELFRICE_HRR_PERSIST=1` to override the auto-disable. The TOML key cannot override (TOML lives at the store root which is itself the path being checked); the env var is the only escape hatch.

Precedence (first decisive wins): env var `AELFRICE_HRR_PERSIST` (truthy `"1"`/`"true"`/`"yes"`/`"on"` forces on; falsy `"0"`/`"false"`/`"no"`/`"off"` disables) > explicit `persist_enabled=<bool>` on `HRRStructIndexCache(...)` > TOML `[retrieval] hrr_persist` > default `true`. Non-boolean TOML values trace to stderr and fall through to the default. The canonical construction site is `aelfrice.retrieval.make_hrr_struct_cache(...)`, which threads the resolved value into the cache for callers that don't manage flag resolution themselves.

**When to disable.** Disk-constrained deployments (the on-disk blob is 8·N·dim bytes — ~41 MB at N=10k and ~200 MB at N=50k at the default dim=512, ~800 MB at N=50k with the dim=2048 escape hatch; federation × multiple stores amplifies) and read-only filesystems are the two cases the opt-out exists for. Operators see the resolved state via `aelf doctor` (`hrr.persist_enabled` row) and `aelf status` (`hrr.persist_state` summary line).

### `use_type_aware_compression`

Boolean, default `true` since #769 (v2.1+, #434). Populates `RetrievalResult.compressed_beliefs` with per-belief renderings dispatched by `belief.retention_class`:

| Retention class | Locked | Unlocked | Notes |
|---|---|---|---|
| `fact` | verbatim | verbatim | Stable codebase state. |
| `snapshot` | verbatim | **headline** | First sentence (split outside ``` fences) + `…`. |
| `transient` | verbatim | **stub** | `[stub: belief={id} class=transient]` marker; full text via `store.get_belief(id)`. |
| `unknown` | verbatim | verbatim | Migration safety. |

Compression is pure and deterministic — no store, clock, env, or random reads. The `compressed_beliefs` field is parallel to `beliefs` (same length, same order); consumers that want the raw belief read `.beliefs[i]`, consumers that want the compressed render read `.compressed_beliefs[i].rendered`.

Enabled by default: `compressed_beliefs` is parallel to `beliefs` (same length, same order). To disable for v2.x parity, set the env var or TOML key to `false`; with that, `compressed_beliefs` is empty and the pack accounts in raw `_belief_tokens`.

Precedence (first decisive wins): env var `AELFRICE_TYPE_AWARE_COMPRESSION=0`/`1` > explicit Python kwarg `use_type_aware_compression=<bool>` > TOML `[retrieval] use_type_aware_compression` > default `true`. The default-on flip landed in #769 after the A2 + A4 bench gates (`docs/design/feature-type-aware-compression.md` §"Bench-gate / ship-or-defer policy") cleared on the lab-side `compression_a*` corpora. Composes with `use_intentional_clustering` since #878.

### `use_temporal_spine` / `temporal_spine_budget`

v4.0.0+ (#1064). Default `false` / `32`. The temporal-spine retrieval lane:
an additive candidate source after L1 that traverses `TEMPORAL_NEXT`
chronological chains from the top-5 packed L1 seeds (both directions,
depth 1) and appends the neighbours — never displacing L1 pre-packing.
The mechanism is complementary to lexical matching: gold beliefs sharing
zero salient terms with the question become reachable through
chronological adjacency to beliefs that do match. No-op guard: stores
with zero `TEMPORAL_NEXT` edges get byte-identical output at ~zero cost.
Precedence: `AELFRICE_TEMPORAL_SPINE` / `AELFRICE_TEMPORAL_SPINE_BUDGET`
env → explicit kwarg → TOML → default. Default-off is the landing
posture; the default-ON flip is gated on the pre-registered criteria in
[docs/design/feature-temporal-spine.md](../design/feature-temporal-spine.md).

### Placeholder flags

`use_signed_laplacian` and `use_posterior_ranking` are reserved by #154 but their owning lanes have not yet shipped. The flags are recognised by `warn_placeholder_flags()` so writing them in `.aelfrice.toml` does not error; setting either to `true` emits a one-shot stderr deprecation warning and is otherwise a no-op. Source of truth: `PLACEHOLDER_FLAGS` in `src/aelfrice/retrieval.py`.

## `[ingest]` (v4.0.0+)

### `write_temporal_spine`

Default `false`. When enabled, every belief insert links to the previous
belief in the same session (`created_at` order, insertion-order
tie-break) with a `TEMPORAL_NEXT` edge — the per-session temporal spine
the `use_temporal_spine` retrieval lane traverses. One edge per belief,
O(1) per insert, idempotent; the off-path is byte-identical to today.
Existing stores predate the writer: `aelf spine backfill` builds their
chains (idempotent, `--dry-run` supported), and `aelf doctor` reports
spine presence + edge count. `AELFRICE_TEMPORAL_SPINE_WRITE` env var
overrides.

## `[rebuilder]` and `[rebuild_floor]` (v1.7+)

Malformed values (wrong type, out-of-range, unrecognised strategy string) in either section fall back to the field default with a `aelfrice rebuilder: ignoring …` trace to stderr. The rebuild never raises on a bad config value.

### `query_strategy`

String, one of `"stack-r1-r3"` or `"legacy-bm25"`. Default `"stack-r1-r3"` since v3.0 (#718, PR #719).

| Value | Effect |
|---|---|
| `"stack-r1-r3"` (default since v3.0) | Runs the R1+R3 query-understanding stack: entity expansion followed by per-store IDF clipping. See `aelfrice.query_understanding` for the rewriter contract. Bench evidence (2026-05-12, 30-row corpus): mean NDCG@k 0.3006 → 0.5858 (+0.2851 absolute). |
| `"legacy-bm25"` | Byte-identical to the v1.4 raw-BM25 path. Opt-in escape hatch for operators who need the exact pre-v3.0 retrieval shape. Removal (PR-4 of #291) is deferred and currently unscheduled; the escape hatch remains available. |

Unrecognised values trace to stderr and fall back to `"stack-r1-r3"`.

### `[rebuild_floor] session`

Float ≥ 0, default `0.10` (v1.7+, #289 / #364). Minimum composite score for a session-scoped (L2) belief to be packed into the rebuilt block. Beliefs whose composite score falls below this floor are skipped with a `below_floor_session:…` reason tag in the rebuild log. Set to `0.0` to disable the floor and pack all session-scoped candidates.

### `[rebuild_floor] l1`

Float ≥ 0, default `0.40` (v1.7+, #289 / #364). Minimum composite score for an L1 / L2.5 belief to be packed. Beliefs below this floor are skipped with a `below_floor_l1:…` reason tag. Set to `0.0` to pack all L1 / L2.5 candidates regardless of score.

Negative values and non-numeric values are rejected; the default applies and the rejection is traced to stderr.

## `[phantom_generation]` (v3.6+)

Opt-in trigger-driven phantom generation (#980). On every `UserPromptSubmit` turn aelfrice deterministically detects whether the turn is a *phantom-generation opportunity* and, if so, appends a small `<aelfrice-phantom-opportunity>` note to the injected context suggesting `/aelf:wonder`. Per the #605 determinism boundary, aelfrice only **flags** the opportunity; the LLM synthesis stays a host-agent action on the existing `/aelf:wonder` path — aelfrice never dispatches an LLM. Default-off; the lane is inert until enabled.

Three signals are ORed under one flag and one per-session budget (the note's `reason` names which fired):
- **gap** — the prompt retrieved zero stored beliefs.
- **new_entity** — a *named* entity (identifier, file path, URL, error code, version, branch — loose noun-phrases excluded) resolves to zero stored beliefs.
- **contradiction** — a CONTRADICTS pair appeared since the per-session snapshot (poll + set-diff; inert unless the #988 semantic-edge substrate is also enabled to mint the edges).

### `enabled`

Boolean, default `false`. Master opt-in. Precedence (first decisive wins): env var `AELFRICE_PHANTOM_GENERATION=1`/`0` (truthy/falsy normalised) > explicit Python kwarg > TOML `[phantom_generation] enabled` > default `false`. Mirrors the `bfs_enabled` resolver shape; a fresh install is unaffected.

### `max_fires_per_session`

Integer ≥ 1, default `3`. Per-session cap on opportunity notes, shared across all three signals and tracked in `session_ring` state. Per-signal dedup (normalised prompt-topic for gap, entity string for new_entity, sorted belief-id pair for contradiction) prevents re-surfacing the same opportunity within a session. TOML-only (no env override), matching the cadence-config precedent.

### `auto_dispatch`

Boolean, default `false`. When `false` (default) the note is a passive surface — it states the opportunity and the agent or user decides. When `true` the note instructs the agent to run the `/aelf:wonder` dispatch on the listed topics. TOML-only.

The trigger is skipped on prompt-shape-gated turns (#674) and is fully fail-soft: any error yields no note and never breaks the hook. Full spec: [phantom_trigger_generation.md](../design/phantom_trigger_generation.md).

## `[memory]` (v3.7.0+)

Controls the claude-memory mirror (#985) — a one-way `PostToolUse:Write|Edit|MultiEdit` hook that ingests host claude-memory fact-file writes into the belief graph so the two stores do not drift. The hook is installed default-on by `aelf setup` but **inert until the flag below is set**; when off it returns after three cheap checks (tool name, path shape, flag) and never imports the store. aelfrice is never authoritative over the memory files; the mirror never locks (L0 stays reserved for explicit `aelf lock`).

### `mirror_claude_memory`

Boolean, default `false`. Master opt-in. Precedence (first decisive wins): env var `AELFRICE_MIRROR_CLAUDE_MEMORY` (truthy/falsy normalised) > TOML `[memory] mirror_claude_memory` > default `false`. When enabled, a `metadata.type` of `user`/`feedback` ingests as `origin=user_validated` (undeflated prior); `project`/`reference`/absent ingests as `origin=agent_inferred` (deflated prior). Belief ids are content-derived, so a byte-identical re-write corroborates rather than duplicates.

## When changes apply

Edits apply on the next `aelf onboard` run for `[noise]` keys. `[retrieval] entity_index_enabled` applies on the next `retrieve()` call. They do not retroactively re-filter beliefs already in the store — config controls ingestion and retrieval, not retention.

To remove existing noise: drop and re-onboard.

```bash
rm "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"
aelf onboard /path/to/project
```

Locks, manually inserted beliefs, and feedback history will be lost. For a less destructive cleanup, query the store with `sqlite3` and `DELETE` rows that match.

## What this file does not do

- The `[noise]` table does not affect retrieval — the noise filter only runs at onboard time. (Retrieval-time behavior is governed by `[retrieval]`, `[rebuilder]`, and `[user_prompt_submit_hook]` above.)
- Does not affect `aelf lock` or `aelf:lock`. Manually-asserted beliefs bypass the noise filter.
- Does not redefine the four built-in categories. You can disable them, not modify what they match. Use `exclude_words` / `exclude_phrases` for custom rules.
- Does not load from `pyproject.toml`, env vars, or CLI flags.

## Resilience

If the file is malformed, unreadable, or contains wrong-typed values, the filter degrades silently to defaults rather than failing the onboard. Failures trace to stderr.

| Failure | Behaviour |
|---|---|
| Malformed TOML | defaults loaded, `malformed TOML in <path>` to stderr |
| Wrong-typed field | that field defaults, `ignoring [noise] <field>` to stderr |
| Non-string entry in a list field | that entry skipped, list still loads |
| Unknown field | silently ignored (forward-compat) |
| Missing file | defaults loaded, no warning |

## See also

- [COMMANDS § `onboard`](COMMANDS.md) — the CLI surface.
- [ARCHITECTURE § Modules](../concepts/ARCHITECTURE.md) — where `noise_filter.py` sits.
- [LIMITATIONS § Onboarding scope](LIMITATIONS.md) — what's still on the horizon for onboard behaviour.

## Pre-issue-create guard (`aelf-pre-issue-hook`, v3.5.0+)

A `PreToolUse:Bash` hook that fires when the agent is about to run `gh issue create`. It
checks the proposed title against open and closed issues (via `gh issue list --state all`)
and recent commit messages (via `git log --grep`) and blocks the call (exit 2) if any
candidate's Jaccard token-overlap with the title is >= 0.5.

**Default:** on. Installed automatically by `aelf setup` and the auto-install manifest.

**Opt-out per-call:** there is no inline per-call bypass — an `ALLOW_DUP_ISSUE=1` prefix on the `gh` command itself never reaches the hook (the guard reads the env var from the *host process's* environment, and its command parser strips leading `KEY=VAL` assignments before matching). To bypass once, set `ALLOW_DUP_ISSUE=1` in the host's environment (e.g. launch the host with it set), run the command, then unset it.

**Opt-out globally (persists across upgrades):**

```bash
AELFRICE_NO_PRE_ISSUE_GUARD=1   # set in shell profile to disable entirely
aelf setup --no-pre-issue-guard  # persist opt-out (~/.aelfrice/opt-out-hooks.json) so upgrades skip it;
                                 # does NOT remove an already-installed settings.json entry — use
                                 # `aelf unsetup` (removes all aelfrice hook entries) for that
```

The guard is deterministic (no embeddings, no LLM calls). Tokenization strips the
conventional-commit prefix (`feat(scope):`, `fix:`, etc.), lowercases, splits on
non-alphanumeric runs, and drops a small stop-word set before scoring.
