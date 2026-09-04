# Configuration: `.aelfrice.toml`

Most users never need this file. The defaults are tuned, and `uv tool install aelfrice && aelf onboard .` gives you correct behavior without it.

This document is the reference for power users. Reach for it when your project has a documentation idiom, or a naming convention, that the default filter handles incorrectly.

## What it does

`.aelfrice.toml` is a single optional TOML file at the root of your project, or in any ancestor directory. It opens up the power-user surfaces below.

- `[noise]` — the belief filter that runs at onboard time. It changes how `aelf onboard` ingests beliefs, and nothing else.
- `[retrieval]` (v1.3+) — the tier toggles and ranking controls that apply at retrieval time:
  - `entity_index_enabled` — the L2.5 tier.
  - `bfs_enabled` — the L3 tier.
  - `posterior_weight` — partial Bayesian-weighted L1 ranking.
  - `l1_limit` and `token_budget` — the #1045 keys for wide retrieval. `l1_limit` caps the Best Matching 25 (BM25) candidate set, and `token_budget` caps the tokens. The defaults are 50 and 2400. Raise both together for multi-hop recall.
  - `use_bm25f_anchors` — the BM25F path with anchor text, since v1.7.
  - `bm25f_per_field` and `bm25_b_anchor` — the #1180 two-field BM25F scorer. It normalizes the content and the anchor text separately instead of concatenating them. The default is off, pending its bench.
  - `use_heat_kernel` — the authority-scoring lane. The default is **off** again since #1162: the lane needs an eigenbasis, no production caller builds one, and a default-on flag therefore advertised a lane that can't fire. `LaneTelemetry.heat_used` now reports at runtime whether the lane fired.
  - `use_hrr_structural` — the structural-query lane that uses a holographic reduced representation (HRR). The default is on since v2.1, and the #1107 Phase-5 cutover put the lane on the production `retrieve()` path. The lane is marker-routed, so on a query without a marker it falls through and does nothing.
  - `hrr_persist` — on-disk persistence of the HRR structural index. The default is on since v3.0.
  - `use_type_aware_compression` — per-belief compression by retention class. The default is on since #769.
  - `use_intentional_clustering` — co-location of related beliefs. The default is on since v3.0, and the #1107 Phase-4 cutover put the lane on the production `retrieve()` path.
  - `expansion_gate_enabled`.
  - `use_gamma_posterior_temperature` — the default is off.
  - `use_zeta_posterior_rerank` — the default is off. This flag is mutually exclusive with the γ flag, and `retrieve()` raises `ValueError` when you turn both on.
  - `use_temporal_spine` and `temporal_spine_budget` — the #1064 lane for chronological adjacency. The defaults are **on** and 32 since v4.0, and the #1107 cutover put the lane on the production `retrieve()` path. The lane works together with `[ingest] write_temporal_spine`.
  - `use_entity_persist_demote` — the #1096 rerank modifier for entity-persistence demotion, also called the organic sink. The default is **on** since v4.0, and the #1107 cutover put the lane on the production `retrieve()` path.
  - `use_origin_tiebreak` — the #1089 tie-break on origin priority inside one tier. The default is off. **This key has no TOML tier, and the kwarg tier is unreachable from `retrieve()`. The environment variable does reach it.** See its section below.
  - `use_supersession_demote`, `supersession_treatment`, and `supersession_demote_factor` — the #1187 supersession lane. The lane demotes or excludes the beliefs that a `SUPERSEDES` edge retires. The defaults are off, `demote`, and 0.5, pending the three-arm bench.

  aelfrice also recognizes two placeholder flags: `use_signed_laplacian` and `use_posterior_ranking`. Setting either one emits a deprecation warning, and neither lane has shipped yet.
- `[rebuilder]` (v1.4+) — the context rebuilder's keys: `turn_window_n` (default 50), `token_budget` (default 4000), `trigger_mode` (`manual`|`threshold`|`dynamic`, default `threshold`), `threshold_fraction` (default 0.6), and `query_strategy` (v1.7+, default `legacy-bm25`). `stack-r1-r3` was the `query_strategy` default from v3.0 until #1501. `[rebuild_floor]` (v1.7+) sets the token-budget floors for the session-scoped belief lane and the L1 belief lane, through the keys `[rebuild_floor] session` and `[rebuild_floor] l1`.
- `[onboard.llm]` (v1.3.0+) — the gate for the onboard classifier that calls the direct API. For the full table, see [Keys § `[onboard.llm]`](#onboardllm-v130) below.
- `[cadence]`, `[implicit_feedback]`, and `[hook_audit]` — three more recognized tables. They hold the feedback-cadence scoring, the deferred feedback for retrieval exposure, and the per-turn hook audit log. Their module docstrings document them (`src/aelfrice/cadence.py`, `src/aelfrice/deferred_feedback.py`, `src/aelfrice/hook.py`).
- `[feedback]` (v3.0+) — the opt-in keys for the feedback lanes. `sentiment_from_prose` (default `false`) connects the sentiment-feedback detector to `UserPromptSubmit` (#606).
- `[belief_categories]` (v4.x+) — the belief categories that a keyword triggers. `enabled` (default `false`) connects the category-injection lane to `UserPromptSubmit` (#1126). Manage the categories with `aelf category`.
- `AELFRICE_TURN_DIFFERENTIAL` (v4.x+, #1382) — an environment variable with no TOML key. **The default is off.** To turn it on, export `AELFRICE_TURN_DIFFERENTIAL=1`. Once it is on, and once a belief has gone into this session's context **verbatim**, a later turn writes a one-line `seen <id>: "<topic>"` reference in the locks manifest instead of repeating the block. The text is already above in the same window, so the reference points at it.

  Each SessionStart opens a new epoch and clears the record, because a new context window, or a compacted one, no longer holds the earlier text. The PreCompact hook also opens a new epoch, **but that hook is opt-in** (`aelf setup --rebuilder`) and a default install doesn't have it, so on a default install SessionStart is the only reset. A boundary that carries no session identifier deletes the record instead, because aelfrice cannot resolve which epoch it belongs to.

  Two configurations remove that last reset, and you must not turn this feature on under either one: `aelf setup --no-session-start`, and a host that compacts the context without starting a new session. In both cases the record survives, and a belief in it stays a one-line reference for the rest of the session.

  **The default is off for two reasons, and both are measurements.**

  First, the feature can make the block **larger**. A `seen` entry opens the `<aelfrice-locks-manifest>` wrapper on a block that had none, and that wrapper costs approximately 237 characters, once per block. A block with one belief therefore has to hold more than approximately 310 characters of content before the change saves anything, and the recorded content distribution has a median of 86 characters. A small block of short beliefs grows; a large block, or a block of long beliefs, shrinks a lot.

  Second, the earlier argument for defaulting to on was that the mechanism can only add text, never hide it. That argument was wrong: the epoch didn't reset when SessionStart wrote an empty block, so a belief could stay a one-line reference for the rest of a session. The fault is fixed, but the argument that supported the default went with it.

- `[memory_block]` (v4.x+, #1359) — the switch that turns off the injected `<aelfrice-memory>` block. The key is `enabled` (default `true`). To stop `UserPromptSubmit` writing the block into your prompt, set the key to `false` or export `AELFRICE_MEMORY_BLOCK=0`. The environment variable overrides the TOML key in both directions, so `AELFRICE_MEMORY_BLOCK=1` re-enables the block for a project that disabled it. The switch stops the whole `<aelfrice-memory>` envelope, and two sub-blocks stop with it:
  - the session-start sub-block of the first prompt (`<locked>`, `<core>`, `<recent-work>` — #578);
  - the `<cadence-resume>` "pick up where you left off" block (#871).

  aelfrice writes the in-session `<cadence-checkpoint>` block (#870) outside the envelope, so the switch does **not** suppress it. The session ring's `next_fire_idx` counter also advances on a suppressed fire, and that advance is what keeps the checkpoint block alive under the `p1_every_k_turns` and `p3_velocity` policies: each policy's firing predicate reads that counter, the counter counts *fires*, and a suppressed fire is still a fire. aelfrice withholds only the per-fire list of injected ids in the ring.

  These parts keep running:
  - retrieval;
  - the correction lane and the relevance lane;
  - `hook_audit.jsonl`. It still records what retrieval returned, with `tokens: 0`, because aelfrice injected nothing.
  - the UserPromptSubmit (UPS) hook's telemetry JSONL. It still gets one row per fire, with `n_returned` / `n_l0` / `n_l1` intact and `total_chars: 0`. There is no `suppressed` field, so read the two values together.
  - the per-turn `rebuild_logs/<session-id>.jsonl` row, which records what retrieval scored on the fire. Don't confuse this row with the `aelf rebuild` CLI command listed below.
  - `session_injected_ids.json` and its `.session-ring.lock`. aelfrice creates both files even on a suppressed fire, carrying `ring: []` and a live `next_fire_idx`.
  - `aelf rebuild`;
  - the SessionStart `<aelfrice-baseline>` block.

  Two other notes sit outside the envelope, like `<cadence-checkpoint>`, so the switch does **not** suppress them either: `<aelfrice-phantom-opportunity>` (#980) and `<aelfrice-phantom-promotion-opportunity>` (#1132 Q2). Both are default-off, and they reach your prompt only when you opt in with `[phantom_generation] enabled = true` / `[phantom_promotion] enabled = true`, or with their `AELFRICE_PHANTOM_GENERATION` / `AELFRICE_PHANTOM_PROMOTION` environment variables.

  A suppressed fire deliberately records no evidence of exposure, because the model never saw those beliefs. It writes none of these records:
  - an `injection_events` row;
  - a `belief_touches` row;
  - a ring entry for an injected id;
  - an `exploration_events` row. aelfrice skips the #1279 slot outright, including its counter, rather than drawing a belief into a pack that nobody reads.
  - a `feedback_history` row with `source='hook'`.

  The last record matters most. `store.exploration_pool` (#1176) reads that exposure record to find the beliefs aelfrice has *never shown*, so a write on a suppressed fire would drop a belief out of that pool permanently, without aelfrice ever having shown it.

  **This switch does not reach two records of retrieval exposure, and both exclusions are deliberate.** Under the opt-in key `[implicit_feedback] enqueue_on_retrieve = true`, `retrieve()` still enqueues one `deferred_feedback_queue` row per hit, because that write sits inside retrieval and retrieval keeps running. The row is inert today, since `sweep_deferred_feedback` has been audit-only and writes nothing since #1162, but it is still a record that retrieval returned a belief. The PreToolUse hook's agent-context lane also writes its own `source='hook'` rows for the beliefs that *it* injects; that lane emits a separate envelope, which this switch does not control.

  **The `source='hook'` row and the belief's `last_retrieved_at` stamp share one transaction, and aelfrice suppresses them together. With the block off permanently, `aelf stale --cold-for` and every other consumer of recency therefore read those beliefs as never retrieved.** That is the intended trade: splitting the pair would make the store assert an exposure that it also denies. The consequence is that `--cold-for` measures "cold since you turned the block off", not "cold".

  `aelf review` is the consumer of recency that proposes a *destructive* action. `store.list_review_candidates` orders `last_retrieved_at` NULLS FIRST, so a belief with a suppressed stamp sorts to the top of the weekly keep/remove/lock checkpoint. `review._cold_days` finds both `last_retrieved_at` and `last_confirmed_at` NULL, falls back to the age since creation, and prints the belief at its maximum coldness.

  A 70-day-old belief that retrieval returned yesterday under a suppressed block reads `70d cold`, not `1d cold`. With the block off permanently, the checkpoint therefore opens with exactly the beliefs that retrieval still finds, and the remove box is one keystroke away. Confirm each entry before you tick it, or leave the block on.

  Flipping the switch *in the middle of a session* has one more consequence. `is_session_first_prompt` runs before aelfrice resolves the switch, so a suppressed fire consumes the session's first-prompt slot. That is deliberate, because `aelf scope-out` resolves against the `session_id` key of the same file. Flipping back to on part-way through a session therefore does not restore the #578 session-start sub-block for that session; start a new session to get it.
- `[user_prompt_submit_hook]` (v3.0+) — the UPS hook's keys. `prompt_shape_gate_enabled` (default `true`) controls the short-circuits for a trivial prompt and for a system envelope, which run before BM25 retrieval (#674). `conversation_aware_query_enabled` (default `true`, v3.x #909) folds a small window of recent dialog turns into the BM25 query, so a follow-up that uses a paraphrase, a pronoun, or a numeric reference still surfaces the thread that carries the answer. Two keys tune this behavior: `conversation_aware_turn_window` (default `4`) and `conversation_aware_prompt_weight` (default `3`).

This file doesn't affect locks, and it doesn't configure the mathematics of the Bayesian update. It DOES configure hook behavior, through `[user_prompt_submit_hook]`, `[feedback]`, `[cadence]`, and `[hook_audit]`.

`scan_repo` walks up from the scan root looking for `.aelfrice.toml`, and the first file it finds is the one that applies. The walk stops at the filesystem root. There is no global configuration and no per-user configuration.

If the file doesn't exist, the noise filter uses the defaults, which is the recommended state.

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

# #1180. Default `false`. Scores content and anchor text as two BM25F
# fields, each normalised by its own length and `avgdl`, instead of
# concatenating the anchor text into the belief's document. With it
# off, a belief's own content terms are length-penalised in proportion
# to how much text its citers wrote: on a synthetic corpus where the
# anchor text never mentions the query term, the cited belief scores
# 0.27x an otherwise identical uncited one. With it on, that case is
# exactly 1.00x, and anchor text only ever helps.
#
# Off by default because it replaces the scoring functional form
# rather than re-parameterising it — no constants make on and off
# agree once an anchor stream exists, so the flip is bench-gated.
# `anchor_weight` becomes a field weight rather than a replication
# count; its shipped value of 3 was tuned as the latter.
# AELFRICE_BM25F_PER_FIELD=1 env var overrides.
bm25f_per_field = false

# #1180. Default `0.75` (the content stream's `b`). Length-
# normalisation strength for the anchor field; consulted only when
# `bm25f_per_field` is on. `0.0` disables anchor length normalisation
# so the contribution tracks raw citation volume rather than term
# density — exposed for ablation, not recommended.
# AELFRICE_BM25_B_ANCHOR env var overrides.
bm25_b_anchor = 0.75

# Default `false` again since #1162. #154 flipped it on after the
# #437 gate cleared at 11/11, but the heat-kernel lane (#150) also
# needs a built eigenbasis and nothing in the shipped pipeline
# constructs one — so the flag advertised a lane that could not
# fire. Setting it `true` is still the opt-in, and still requires
# passing an `eigenbasis_cache` for anything to happen.
# AELFRICE_HEAT_KERNEL=1 env var overrides.
use_heat_kernel = false

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
# latency budget). Live on the production retrieve() path since the #1107
# Phase-4 cutover. Co-locates related beliefs in the packed retrieval
# output so multi-fact queries surface a coherent neighborhood. Set to
# `false` for v2.0.x parity. AELFRICE_INTENTIONAL_CLUSTERING=0 env
# var overrides.
use_intentional_clustering = true

# v4.0.0+ (#1064). When true, the
# temporal-spine lane traverses TEMPORAL_NEXT chronological chains from
# the top-5 packed L1 seeds (both directions, depth 1) and appends the
# neighbours after the L1 candidates. Reaches gold that shares zero
# salient terms with the question through chronological adjacency —
# confirmed +14.6pp gold-coverage on LoCoMo, 10x its shuffled control.
# Default `true` since the v4.0 lane flip (#1064, #1107 Phase 2) — live
# on the production retrieve() hook path. No-op on stores with zero
# TEMPORAL_NEXT edges (run `aelf spine backfill` to build the spine on an
# existing store; the [ingest] write_temporal_spine writer keeps it
# growing). Opt out with `AELFRICE_TEMPORAL_SPINE=0` or this key = false.
use_temporal_spine = true

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
# v4.0.0+ (#1064). Default `true` since the writer flip. Every belief
# insert chains to its session predecessor with a TEMPORAL_NEXT edge
# (src = successor, weight 0.8), building the per-session temporal spine.
# The retrieval lane (use_temporal_spine above) is also default-on since
# the #1107 Phase-2 cutover; the two flags resolve independently. One edge
# per belief, O(1) per insert; explicit opt-out
# is byte-identical. AELFRICE_TEMPORAL_SPINE_WRITE env var overrides.
write_temporal_spine = true

[relationship_detector]
# #988 / #1299. Deterministic contradiction detector. `auto_detect` is
# default-OFF: when false, ingest writes no CONTRADICTS edges and the
# section only affects the read-only `aelf doctor --relationships` /
# `--detect-stale` audits. AELFRICE_AUTO_RELATIONSHIPS env var overrides
# `auto_detect`; the three thresholds have no env override.
auto_detect = false
# Minimum token Jaccard for a belief pair to enter the classifier.
# Default 0.4. Unrelated to `dedup`'s 0.8 — different module, different
# consumer.
jaccard_min = 0.4
# Minimum verdict score for a `contradicts` pair to be auto-emitted as
# an edge. Default 0.5. Sub-confidence pairs are the POTENTIALLY_STALE
# writer's domain.
confidence_min = 0.5
# Cap on candidate pairs scored per audit run. Default 5000.
max_candidate_pairs = 5000

[hook]
# #1326 / #1177 proposal 18. Group the injected per-turn block into
# trust-tier sections (<user-locked> / <observed> / <inferred>) and emit
# each belief's origin, evidence count and posterior. Default-OFF: with
# it off the block is byte-identical to the pre-#1326 output, including
# the validated framing header. AELFRICE_PROVENANCE_RENDER overrides.
provenance_render = false

[rebuilder]
# v1.7+ / #291. Selects the query-rewriting stack used by the context
# rebuilder. Default `"legacy-bm25"`: the raw query is passed through.
# `"stack-r1-r3"` runs entity expansion + per-store IDF clipping via
# aelfrice.query_understanding. It was the default from v3.0 (#718)
# and was reverted under #1501 — see that entry below before selecting it.
query_strategy = "legacy-bm25"

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

[belief_categories]
# v4.x+ (#1126). Default `false`, opt-in. When true, the
# UserPromptSubmit hook reranks the retrieval output so a fired
# category's rules lead the block (with a <category-focus> label), for
# every category that is always-on or has a keyword phrase in the
# prompt. Reranks rather than injecting a second block (a separate block
# double-injects what retrieval already returns — see the #1126 R&D).
# Advisory only — it never blocks a tool call. Deterministic (case-
# insensitive word-boundary literal-phrase matching, no embeddings),
# fail-soft. AELFRICE_BELIEF_CATEGORIES=1 env var overrides. Manage
# categories with `aelf category`. See docs/design/belief_categories.md.
enabled = false

[memory_block]
# v4.x+ (#1359). Default `true`. Set to false to stop the
# UserPromptSubmit hook writing the <aelfrice-memory> block into your
# prompt. The whole envelope goes, including the first-prompt
# session-start sub-block (#578) and <cadence-resume> (#871); the
# in-session <cadence-checkpoint> block (#870) is written outside it
# and is not suppressed, and neither are the two default-off phantom
# notes (#980, #1132 Q2). Retrieval, the correction and relevance
# lanes, hook_audit.jsonl, the UPS telemetry JSONL (total_chars 0,
# n_returned intact), the per-turn rebuild_logs/<session-id>.jsonl
# row, `aelf rebuild` and the SessionStart <aelfrice-baseline> block
# all keep working, and the session ring's next_fire_idx keeps
# advancing so the cadence policies that read it still fire; a
# suppressed fire records no injection_events / belief_touches /
# injected-id ring entry / exploration_events row and no
# source='hook' feedback_history row, since the model never saw
# those beliefs. That last one also suppresses the belief's
# last_retrieved_at stamp, which shares its transaction, so
# `aelf stale --cold-for` reads those beliefs as never retrieved
# and `aelf review` sorts them to the top of the weekly
# keep/remove/lock checkpoint.
# The AELFRICE_MEMORY_BLOCK env
# var overrides this key in both directions (=0/false/no/off forces
# off, =1/true/yes/on forces on); any other value falls through to
# this key. Every emitted <aelfrice-memory> block carries a one-line
# pointer to `aelf tail` and to this switch.
enabled = true

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

aelfrice ignores unknown keys and unknown tables. The file is forward-compatible.

## Keys

### `disable`

| Token | Disables | Effect |
|---|---|---|
| `headings` | The "every line is a markdown heading" filter | Pure heading blocks pass through |
| `checklists` | The "every line is `- [ ]`" filter | Task-list items become belief candidates |
| `fragments` | The `min_words` short-paragraph filter | Short labels like `DRAFT` pass to the classifier |
| `license` | The seven-signature license-preamble filter | LICENSE.md text becomes belief candidates |

A disabled category is silent: `ScanResult.skipped_noise` counts nothing from it, and the other categories still fire. aelfrice ignores an unrecognized token without reporting it.

### `min_words`

Integer, default `4`. aelfrice drops any paragraph shorter than this value.

| Setting | Use when |
|---|---|
| `4` (default) | Most projects. |
| `3` or lower | You lock terse rules ("prefer composition", "no global state"). |
| `0` | Nothing should be dropped for length. |

aelfrice rejects a non-integer value, writes a warning to stderr, and applies the default.

### `exclude_words`

A list of whole-word matches, matched with respect for word boundaries. `"jso"` matches the standalone token, but not `json` or `jsonify`. Use this key for initials, codenames, and status keywords.

### `exclude_phrases`

A list of literal substring matches. The match is case-insensitive, but otherwise verbatim. Use this key for templated header lines such as `Last updated:` and `Generated by`, and for inline status flags such as `TODO:` and `FIXME`.

There is a trade-off against `exclude_words`: a phrase match is a literal substring with no word boundaries, so `["foo"]` in this key drops a paragraph that contains `foobar`.

### `[onboard.llm]` (v1.3.0+)

The host-driven large language model (LLM) classifier for onboard ingest. It replaces the default regex path `classify_sentence` with the host model's Task tool, which needs no API key; the `[onboard-llm]` extra gates the fallback to the direct API. The default at v1.5.1+ (#238) is on (`enabled = true`). When no host Task tool is reachable, the classifier falls back softly to the regex classifier. For the boundary policy, see [`docs/design/llm_classifier.md`](../design/llm_classifier.md); for privacy, see [`docs/user/PRIVACY.md § Onboard-time outbound call`](PRIVACY.md#onboard-time-outbound-call).

| Key | Type | Default | Effect |
|---|---|---|---|
| `enabled` | bool | `true` (since v1.5.1, #238; was `false` v1.3.0–v1.5.0) | Turns on the LLM path when you don't pass a `--llm-classify` flag. The CLI flag overrides this key, so pass `--llm-classify=false` to force the regex path while this key is `true`. |
| `max_tokens` | int | `200_000` | Hard cap on the combined input and output tokens for each onboard run. If a run goes over the cap, the classifier stops mid-stream, and whatever it already classified stays in the store. Belief ids are deterministic, so a re-run resumes idempotently. `0` turns the cap off. For power users. |
| `model` | str | `"claude-haiku-4-5-20251001"` | Anthropic model id, pinned by default. Override it only if you have a reason: classification recall and the few-shot block are calibrated against the pinned model. |

All three keys are optional, and a missing key takes its default.

The four-gate boundary policy is non-negotiable. `enabled = true` is one gate; the other three are:

- the `[onboard-llm]` extra is installed;
- `ANTHROPIC_API_KEY` is set in the environment;
- you accepted the one-time consent prompt for this machine (the sentinel is at `~/.aelfrice/llm-classify-consented`).

The consent prompt fires again when the model id changes, and again when aelfrice's MAJOR version changes.

```toml
# Example: opt in for this project, raise the cap, leave model pinned.
[onboard.llm]
enabled = true
max_tokens = 500_000
```

Auth, model selection, and provider choice are NOT configurable here. aelfrice reads `ANTHROPIC_API_KEY` only from the environment, never from this file. There is no provider abstraction layer, and v1.3.0 supports only Anthropic's Haiku.

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

When you enable the tier:
- `retrieve()` extracts entities from the query. The entity kinds are file paths, identifiers, branch names, version strings, URLs, error codes, and noun phrases.
- `retrieve()` looks up those entities in the `belief_entities` SQL table.
- `retrieve()` returns the matched beliefs above the L1 BM25 results, ranked by the count of entity overlaps.
- The default token budget rises from 2,000 to 2,400, making room for the L2.5 sub-budget of 400 tokens. The L1 sub-budget of 2,000 tokens doesn't change.

When you disable the tier (TOML `false`, or `AELFRICE_ENTITY_INDEX=0`, or an explicit `entity_index_enabled=False` kwarg on `retrieve()`):
- L2.5 does not fire, and the output is byte-identical to the L0+L1 path of v1.2.
- The default token budget returns to 2,000, unless the caller passed an explicit budget.

Precedence (the first decisive tier applies): environment variable `AELFRICE_ENTITY_INDEX=0` > explicit Python kwarg > TOML > default `true`.

aelfrice always populates the index on write, whatever this flag holds; disabling the flag changes only the reads. When you enable it again, the index is already up to date and needs no backfill pass.

### `posterior_weight`

Float ≥ 0, default `0.5` at v1.3.0. This key combines the L1 BM25 score with the Beta-Bernoulli posterior mean, log-additively:

```
score = log(-bm25_raw) + posterior_weight * log(posterior_mean(α, β))
```

`-bm25_raw` flips the signed score of SQLite full-text search version 5 (FTS5) to a positive number: in SQLite, a negative score of smaller magnitude is the better score, so aelfrice negates it before taking `log`. `posterior_mean(α, β) = α / (α+β)` reuses the existing scoring helper, which uses the Jeffreys prior and reads `0.5` for an unobserved belief.

Behavior at the boundaries:

- **`0.0`** — the score becomes `log(-bm25_raw)`, an ordering byte-identical to the v1.0.x `ORDER BY bm25(beliefs_fts)` ordering. Use this value for diff tooling and bisection.
- **`0.5`** (default) — the optimum on the synthetic graph from the v1.3 calibration. The posterior moves the rank without overwhelming BM25.
- **`> 1.0`** — the posterior dominates, so a high-confidence belief surfaces even on a weak keyword match. Use this range when the feedback density is high and BM25 noise is the limiting factor.

Locked beliefs (L0) bypass scoring completely. The weight reranks only the L1 BM25 candidate set; it leaves the L2.5 entity-index hits and the L3 breadth-first search (BFS) expansions alone.

A negative value clamps to `0.0`. A non-numeric value in an environment variable traces to stderr and falls through.

Precedence (the first decisive tier applies): environment variable `AELFRICE_POSTERIOR_WEIGHT=<float>` > explicit Python kwarg `posterior_weight=<float>` on `retrieve()` / `retrieve_v2()` > TOML `[retrieval] posterior_weight` > default `0.5`.

### `use_entity_persist_demote`

Boolean, default `true` in `retrieve_v2` since v4.0 ([#1096](https://github.com/robotrocketscience/aelfrice/issues/1096)), and off before that. The default flipped once the G2 mixed-corpus eval [#1103] cleared the no-regression gate. The **entity-persistence demotion lane** is a deterministic *organic sink* for the #1086 junk-percolation problem, in which junk ranks up rather than down. aelfrice applies the lane as a log-additive rerank modifier over the ranked candidate tiers.

**Scope:** the lane is default-ON on the production `retrieve()` path since the #1107 Phase-3 cutover, because the shim passes the flag in the resolver-driven form. The live `UserPromptSubmit` hook and `context_rebuilder` therefore both run the demotion, which is what carries the #1086 junk-percolation fix to real hosts rather than only to the consumers of `retrieve_v2`.

For each candidate that carries an entity, the lane reads a grounding score `S1 = durable / (durable + transient + 1)` from the `belief_entities` index, in one batched query over the whole candidate set. It then applies the penalty `min(0, log(S1 + ε))`. Some beliefs ground only to *transient* coordination tokens, such as a bare pull-request number, a bare issue number, a version tag, or a branch tag; the lane demotes those below the beliefs that ground to *durable* entities, such as a file path, an error code, or a symbol identifier. The lane is a **pure demotion**: a well-grounded belief stays neutral, and the lane never boosts it. It touches only the candidates that carry an entity, so durable content without one, such as a docstring or a formula, is never penalised.

Measurement on a hand-labeled set of 118 beliefs gave this separation: mean S1 was 0.56 for durable beliefs and 0.06 for ephemeral ones, and the ranking area under the curve (AUC) for durable above ephemeral rose from 0.48 to 0.87.

The sink is **content-referential, not temporal**. Measurement showed that a decay sink keyed on time or recency is empirically inert for this workload, because the junk is *hot* rather than stale. The organic sink is therefore this lane, not cold hibernation. The lane is deterministic per #605, because it is an entity-index join and uses no embeddings. Its output is byte-identical when you leave the flag unset.

Precedence (the first decisive tier applies): environment variable `AELFRICE_ENTITY_PERSIST_DEMOTE=1`/`0` > explicit Python kwarg `use_entity_persist_demote=<bool>` on `retrieve_v2()` > TOML `[retrieval] use_entity_persist_demote` > default `true`. The production `retrieve()` shim takes no per-call kwarg, so on that path you opt out with the environment variable or the TOML key to get the pre-flip ranking.

### `use_supersession_demote`

Boolean, default `false` ([#1187](https://github.com/robotrocketscience/aelfrice/issues/1187)). This key enables the **supersession lane**. A `SUPERSEDES` edge points *at* a belief, and that belief is the claim you retired; the lane pushes it down the ranking, or drops it from the candidate set.

Without this lane, retrieval has no concept of supersession. Consider this sequence: you correct "deploy target is heroku" to "fly.io", contradiction resolution records the supersession, and the next prompt still injects the heroku belief **ahead of** the fly.io belief. `aelf resolve` writes the edge, and so does the triple extractor's "X supersedes Y" rule. Since the #1005 revert, the auto-relationship detector writes CONTRADICTS edges and nothing else, so this lane affects the explicit paths.

Two arms, selected by `supersession_treatment`:

| Value | Behavior |
|---|---|
| `demote` (default) | Adds `log(supersession_demote_factor)` to the candidate's rerank score. |
| `exclude` | Drops the superseded belief from the candidate set entirely, before aelfrice computes the heat-kernel seeds. |

`demote` is the default because it is the recoverable arm. The triple extractor can write a `SUPERSEDES` edge from prose that only *looks* like a supersession, and a wrong exclusion then hides a belief with no ranking signal left to make it visible again. Exclusion is the stronger reading of "the user retired this claim". **A three-arm bench gates the choice of default**, across demote, exclusion, and control. Unlike the #1170 BFS-direction fix, this lane changes the output of `retrieve()` on the default path, so both arms ship behind the flag and neither is presumed.

`supersession_demote_factor` (float, default `0.5`) keeps its multiplicative meaning in the log domain: the penalty is `log(factor)`. aelfrice clamps the factor to `(0, 1]`, so a value above 1 cannot promote a retired belief, and `0` gives a finite penalty rather than `-inf`. The penalty is **additive**, not `score * factor`. The composite rerank score is a log-domain quantity and routinely negative, so multiplying it by `0.5` would *raise* the score and promote the very belief the lane is demoting.

Calibration note if you run the bench: at `factor = 0.5` the penalty is `-0.69`. That is the same order of magnitude as the default-ON entity-persistence penalty, and much weaker than that penalty's `log(ε)` floor of `-6.9`. The two penalties compose additively and can cancel each other, so sweep the factor rather than testing `0.5` alone.

The lane runs one batched query per retrieval: `SELECT DISTINCT dst … WHERE type = 'SUPERSEDES'` over the candidate set. With the flag off, aelfrice skips that query completely. The lane is deterministic per #605, because it is an edge join that reads no clock and no embeddings.

Precedence for all three keys (the first decisive tier applies): the environment variables > the explicit Python kwarg on `retrieve_v2()` > the TOML keys > the defaults. The environment variables are `AELFRICE_SUPERSESSION_DEMOTE=1`/`0`, `AELFRICE_SUPERSESSION_TREATMENT=demote|exclude`, and `AELFRICE_SUPERSESSION_FACTOR=<float>`. The TOML keys are `[retrieval] use_supersession_demote` / `supersession_treatment` / `supersession_demote_factor`, with defaults `false` / `demote` / `0.5`. An unrecognized treatment traces to stderr and falls through to the default, and a non-numeric factor does the same. Neither one raises.

### `order_policy`

String, default `lane` (v4.3+, [#1274](https://github.com/robotrocketscience/aelfrice/issues/1274)). This key selects the order in which aelfrice **renders the retrieved beliefs into the injected block**. It belongs to the render layer, not to retrieval: it permutes the hits that `retrieve()` already returned, and it changes neither which beliefs are selected nor how many tokens they cost.

Without this key, a belief's position in the block is a side effect of lane concatenation (`locked + l25 + l1 + hrr + spine + bfs`). Nobody chose that order as a policy, which makes it untestable. Named policies turn the question into a configuration change.

| Value | Behavior |
|---|---|
| `lane` (default) | Identity permutation — the block is byte-identical to pre-#1274 output. |
| `locks_last` | Non-locked hits first, the user-locked tier last. |
| `score_desc` | The user-locked tier first, then the non-locked hits in descending rerank-score order. **This key cannot reach that policy today.** See below. |

Every policy is a **stable and total** permutation, with ties broken on the hit's original index. The rendered order is therefore a pure function of the hits, the policy, and the scores, and a replay reproduces it exactly. No policy adds a hit, and none drops one.

`score_desc` needs the L1 rerank scores, and `Belief` doesn't carry them. When the caller supplies no scores, the policy falls back to `lane` **and traces to stderr**. It never silently substitutes a proxy such as the posterior, because a silent downgrade of an explicit setting is the failure [#1271](https://github.com/robotrocketscience/aelfrice/issues/1271) documents. Rerank scores are log-domain and negative, so an unscored hit sorts *last* rather than first.

**No shipped call site supplies those scores**, so "not supplied" means "always" today. `AELFRICE_ORDER_POLICY=score_desc` renders the `lane` permutation and emits the stderr trace on every hook fire. The hook-audit row records the policy that *applied*, so it reads `lane`: the key reports the arm that ran, not an arm that didn't. To reach `score_desc`, call `retrieval.order_for_injection(hits, "score_desc", scores=...)` directly; that call is the only route to `score_desc`. Until the scores reach the render boundary, an ordering A/B has **two** arms, `lane` and `locks_last`, not three.

The hook-audit row records the policy that produced a block, in the field `order_policy`, so an ordering A/B can attribute a block to its arm from the audit alone.

Two cautions apply if you measure this key, both from the #1274 pre-flight over 341 real `user_prompt_submit` blocks.

- **A policy that relocates the locks is not attention-neutral.** Position 1 holds a user lock in 100% of the live blocks, and a median of 50 locks comes before the first non-locked belief. Score any arm that moves the locked tier on lock-following, not on answer accuracy alone.
- **The benchmark harness has no lock tier.** `benchmarks/longmemeval_adapter.py` retrieves with `include_locked=False`, and every bench adapter ingests at `LOCK_NONE`. `lane`, `locks_last`, and `score_desc` are therefore the *same permutation* in that harness, so a null from it shows an inert instrument, not a null result.

Precedence (the first decisive tier applies): environment variable `AELFRICE_ORDER_POLICY=lane|locks_last|score_desc` > explicit argument > TOML `[retrieval] order_policy` > default `lane`. An unrecognized value traces to stderr and falls through to the default; it does not raise.

### `use_origin_tiebreak`

Boolean, default `false` (v4.0+, [#1089](https://github.com/robotrocketscience/aelfrice/issues/1089)). This key enables the **origin-priority tie-break**: when two ranked candidates tie on relevance, the *origin* with the higher trust takes the higher rank. For example, when two such candidates tie, a belief curated from a `user` or `feedback` fact file outranks a belief captured automatically from a chat transcript.

This is a **tie-break** inside one tier, never a primary rerank term. The origin key sits *between* the relevance score and the id tie-break, so relevance always dominates and the behavior stays byte-identical when the flag is off. The tie-break applies in both ranked tiers: the L1 FTS rerank and the L2.5 entity-index overlap. It is deliberately *not* an origin *rerank lane*; #1013 refuted that lane on LoCoMo, because the failure there was a BM25 recall limit, and a rerank cannot fix a recall limit. The tie-break is deterministic per #605.

Precedence (the first decisive tier applies): environment variable `AELFRICE_ORIGIN_TIEBREAK=1` > explicit Python kwarg `use_origin_tiebreak=<bool>` > default `false`. **There is no TOML tier.** Unlike every sibling resolver, `is_origin_tiebreak_enabled` does not read `.aelfrice.toml`, so aelfrice accepts a `[retrieval] use_origin_tiebreak` key in silence and that key has no effect.

**The environment variable reaches `retrieve()`. The kwarg does not.** Unlike the four graduated lanes, the production shim does not spell the origin tie-break in the resolver-driven form: `retrieve()` passes the literal `use_origin_tiebreak=False` rather than `None`, and that literal does **not** disable the environment tier. `retrieve()` is the #1107 thin adapter over `retrieve_v2()`, and `retrieve_v2()` resolves the flag with `is_origin_tiebreak_enabled(use_origin_tiebreak)`, a resolver that checks the environment *first*. `AELFRICE_ORIGIN_TIEBREAK=1` therefore overrides the literal `False`, exactly as it does for `use_fan_effect`. The measurement below runs end to end on a store of two beliefs whose contents tie, so the tie-break is the only thing that can decide the order:

```
AELFRICE_ORIGIN_TIEBREAK unset  ->  retrieve() order: aaa1,zzz9   (id ASC)
AELFRICE_ORIGIN_TIEBREAK=1      ->  retrieve() order: zzz9,aaa1   (origin priority)
```

The literal `False` costs the **kwarg** tier. `retrieve()` exposes no `use_origin_tiebreak` parameter, so a caller has nothing to pass through, the same shape as `use_fan_effect`. Use the environment variable on that path. The lane is still the one staged rerank that v4.0 did not graduate; graduating it is a separate operator call, and so is flipping the default to ON. A corpus with a single provenance, such as LoCoMo, shares one origin tier, so the tie-break is always inert there.

BM25F-only L1 shipped default-on at v1.7.0 (see `use_bm25f_anchors`). The heat-kernel lane and the HRR-structural lane shipped default-on at v2.1.0, after the #154 composition-tracker bench gate cleared 11/11 against the #437 reproducibility-harness corpus (see `use_heat_kernel` and `use_hrr_structural` below). #1162 returned `use_heat_kernel` to default-off, and left the HRR-structural lane alone. For the v1.3 contract and the analysis of the rejected alternatives, see [`docs/design/bayesian_ranking.md`](../design/bayesian_ranking.md).

### `use_fan_effect`

Boolean, default `false` (v4.x+, [#1176](https://github.com/robotrocketscience/aelfrice/issues/1176)). This key ranks the **L2.5 entity tier** by the fan-weighted activation of Adaptive Control of Thought—Rational (ACT-R). Without the key, that tier ranks by a raw count of entity overlaps.

The shipped lane orders by `COUNT(DISTINCT entity_lower)`, which prices every matched entity the same. A real corpus doesn't behave that way: on a store of 44,584 beliefs, `tmp` appears in 1,480 beliefs and `and` appears in 1,026, while 86% of the entities appear in exactly one belief. A match on a token that occurs everywhere in the corpus therefore earns the same rank as a match on a unique symbol, on the one tier that holds unconditional precedence for the budget.

With the flag on, a belief scores `A_i = Σ_j ln((N + 1) / (fan_j + 1))` over the query entities it carries, where `fan_j` counts the *active* beliefs carrying entity `j` and `N` is the count of active beliefs. Written as a log ratio, this score is algebraically the inverse document frequency (IDF), so it reuses a calibration the system already has. Every term is non-negative, so an extra match can never demote a belief. **When all the fans are equal, the ordering degenerates exactly to the overlap count it replaces.**

The returned tuples keep their shape: the second element is still the overlap count, and the row **count** is `min(pool, limit)` either way. The **order** differs, and under truncation a different order means a different returned **set**. At a top-*k*, at a token budget, or at any `[:n]`, *which* beliefs you get depends on whether this lane ran. The [#1434](https://github.com/robotrocketscience/aelfrice/issues/1434) fixture verified this element for element: the counts were identical at every limit, the sets disagreed from limit 1 through 6, and they converged at 7. If you truncate, treat the selected set as dependent on the lane ([#1462](https://github.com/robotrocketscience/aelfrice/issues/1462)).

The lane needs no new table and no migration. It counts the fan inline over the query's own keys, and takes the logarithm in Python, because SQL `LN()` requires `SQLITE_ENABLE_MATH_FUNCTIONS` and the support matrix doesn't guarantee that option. The cost is at parity with the lane it replaces: 0.039 ms at p50 against 0.045 ms. Parity holds because `store_generation()` memoizes the count of active beliefs; recomputing that count for each query costs 1.315 ms, which dominates everything else. The lane is deterministic per #605: the activation sum iterates the entities in sorted key order, so two beliefs carrying the same entity set land on bit-identical activations.

Precedence (the first decisive tier applies): environment variable `AELFRICE_FAN_EFFECT=1`/`0` > explicit Python kwarg `use_fan_effect=<bool>` > default `false`. There is no TOML tier yet; one should arrive with any flip of the default. **The kwarg tier exists on `retrieve_v2()` and `retrieve_with_tiers()` only. `retrieve()` takes no `use_fan_effect` argument.** On the production entry point, the environment variable is therefore the only control. **The default stays off until the A/B runs.** The kill gate cleared, and the cost is lower than that of the lane this one replaces, but whether the reorder ranks *better* is a separate measurement. Flipping the default is an operator call.

### `exploration_enabled` / `exploration_cadence` / `exploration_slots`

One boolean and two integers, defaults `false` / `20` / `1` (v4.x+, [#1279](https://github.com/robotrocketscience/aelfrice/issues/1279), [#1294](https://github.com/robotrocketscience/aelfrice/issues/1294), [#1176](https://github.com/robotrocketscience/aelfrice/issues/1176) proposal 5). These keys give a **never-injected** belief a slot in the injected block on every *n*-th turn.

**The cadence counts the turns globally, across all sessions** ([#1294](https://github.com/robotrocketscience/aelfrice/issues/1294)). `fire_idx` is a monotonic counter at store level, held in `schema_meta`, so `exploration_cadence = 20` means one turn in twenty and the realised rate doesn't depend on how you segment your sessions. The value was **3** in #1279, when the counter counted per session: the index came from the session ring, which holds exactly one session, so it restarted constantly. A cadence of 20 then reached a firing turn on **8 of 956** turns for all time, and since the regime break of 2026-06-30, on **0 of 259** turns. The lane would have been enabled and would never have run.

> **Regime break for `exploration_events`.** The rows aelfrice wrote before #1294 drew `fire_idx` from the per-session sequence; the rows after #1294 draw it from the global sequence. The form of the seed derivation doesn't change: it is blake2b over `(scope_id, fire_idx, query)`. The two series are not comparable, and you **must not pool them**. #1016-B partitions the injection-pack series the same way. The rows describe themselves: an index from before #1294 restarts from a low value repeatedly, and an index from after #1294 never decreases.

The slot targets a loop the ranker cannot leave on its own: a belief that starts underranked is never retrieved, so it never earns evidence, so it stays underranked. That loop covers most of the store. On a live store of 44,586 beliefs, **37,489 active unlocked beliefs (84.1%) carry neither a `feedback_history` row nor an `injection_events` row**. Only **1,352 beliefs (3.0%) have ever been injected**.

On a firing turn the slot does three things:

1. It claims the next `fire_idx` at store level. The claim is a read-modify-write inside an immediate transaction, so two sister sessions sharing the store cannot receive the same index.
2. It draws from `MemoryStore.exploration_pool` with a seed derived from `(session, fire_idx, query)`.
3. It appends one `exploration_events` row, recording the seed, the candidate list, the drawn ids, **and the displaced ids**.

A single row therefore answers the question "why was this belief in my context?", and the counterfactual pack is reconstructable.

Three properties are contractual rather than incidental:

- **The slot substitutes, it never appends, and it accounts in tokens rather than slots.** A drawn belief can be longer than the hit it replaces, so a one-for-one swap would still grow the block. Instead, the slot frees at least as many tokens from the lowest-ranked non-locked tail as it spends, and when that tail cannot fund the draw it **skips the draw rather than growing the block**. A slot that grew the block would be a budget increase under a different name, and it would confound the coverage measurement the slot exists to produce.
- **The slot never displaces a user lock.** aelfrice injects L0 unconditionally, the pool already excludes the locks, and the displacement scan skips them. A pack holding only locks therefore produces no action rather than an eviction.
- **aelfrice records the exposure.** The substitution runs before the injection ledger, so the ledger records an explored belief as injected, like any other hit. A substitution without a record would leave the loop as closed as it was.

The path is fail-soft end to end: any error in it, such as a pool query that raises, leaves the pack exactly as retrieval produced it. The path is deterministic per #605, since the same `(session, fire_idx, query, pool)` draws the same belief, and that is what makes the ledger replayable.

Precedence (the first decisive tier applies): environment variable `AELFRICE_EXPLORATION` / `AELFRICE_EXPLORATION_CADENCE` / `AELFRICE_EXPLORATION_SLOTS` > explicit Python kwarg > `[retrieval] exploration_enabled` / `exploration_cadence` / `exploration_slots` in `.aelfrice.toml` > the defaults above. A cadence of `0` or less disables exploration; it does not raise.

**The default is off, and the correct measurement gates the flip.** This slot is *not* a ranking change, and you must not run an A/B on it as one. Its outcome is the **coverage of the never-injected pool over time**, which you can count from `exploration_events` and `injection_events` without a judge and without a gold set. The draw is deliberately **uniform**, because the A-Res weighting the original specification keyed on `scoring.uncertainty_score` is invalid: that function is the Beta *differential* entropy, which is `<= 0` on `[0, 1]`, so the reservoir key divides by zero. After either natural repair of the sign, that weighting is indistinguishable from a uniform draw at a total-variation distance of 0.0586.

### `utterance_prior_weight`

Float, default `0.0` (v4.x+, [#1174](https://github.com/robotrocketscience/aelfrice/issues/1174)). This key sets the weight of the **document prior for utterance against knowledge** in the L1 rerank. The prior is a term that doesn't depend on the query, and it demotes a belief that looks like *something someone said* rather than *something that is true*.

The prior is a naive-Bayes log-odds over two classes, which aelfrice reads directly from `ingest_log`: the transcript rows form one class, and the filesystem and git rows form the other. It therefore uses no hand labels and no embeddings. It targets a measured failure: the store ingests its own query log, so the nearest lexical neighbor of a query is frequently an earlier query.

The penalty is **log-additive, and aelfrice clamps it at 0**, so knowledge-shaped content stays neutral and the lane never promotes it. The rerank score is a log-domain quantity and routinely negative, so an unclamped term would reorder the documents the lane has no opinion about. `score()` returns a *mean* over the document's known stems rather than a sum, so the term doesn't scale with document length, which the per-field normalization of BM25F already handles. The lane is deterministic per #605, because the mean sums the stems in sorted order.

At `0.0` the lane short-circuits and **nothing reads the ingest log**, so the behavior is byte-identical to running without the flag. A malformed value falls through to `0.0`, and so does a negative value; neither inverts the lane. aelfrice builds the prior once per store and caches it.

Precedence (the first decisive tier applies): environment variable `AELFRICE_UTTERANCE_PRIOR_WEIGHT=<float>` > explicit Python kwarg > default `0.0`. There is no TOML tier yet. As with `use_fan_effect`, the kwarg tier exists on `retrieve_v2()` and `retrieve_with_tiers()` only: `retrieve()` exposes no `utterance_prior_weight` parameter and honors the environment variable alone, and passing that parameter to `retrieve()` raises `TypeError`.

**The default stays off until the W-sweep runs.** Proving that a non-zero weight ranks *better* needs a relevance gold set, and the store's observed-utility signal cannot supply one, because it holds 5 positives across 16,355 resolved `injection_events`. Score the sweep below the locked block: aelfrice injects the L0 locks ahead of the ranked candidates and never trims them, so a top-k metric measures the lock tier and stays constant in this weight.

### `bfs_enabled`

Boolean, default `false` at v1.3.0. This key toggles the L3 retrieval tier, a multi-hop BFS traversal of the graph.

When you enable the tier:
- After aelfrice packs L0, L2.5, and L1, `retrieve()` walks the outbound edges from those seeds.
- Each visited belief scores `product(BFS_EDGE_WEIGHTS[edge.type])` along its path.
- Four bounds apply: `max_depth=2`, `nodes_per_hop=16`, `total_budget_nodes=32`, and `min_path_score=0.10`.
- The edge-type weights move the frontier toward the decisional edges: SUPERSEDES 0.90, CONTRADICTS 0.85, DERIVED_FROM 0.70, SUPPORTS 0.60, CITES 0.40, RELATES_TO 0.30.
- The BFS expansions append to the same packed output, consuming the same `token_budget` as the earlier tiers, in order of descending score.
- `RetrievalResult.bfs_chains` exposes the edge-type path that reached each L3 expansion.

When you disable the tier (the v1.3.0 default):
- L3 does not fire, and the output is byte-identical to the L0+L2.5+L1 baseline.

Precedence (the first decisive tier applies): environment variable `AELFRICE_BFS=1`/`0` > explicit Python kwarg > TOML > default `false`.

The flag ships default-OFF at v1.3.0, because nobody has calibrated the default edge weights from the literature against the v1.2 corpus. A v1.3.x patch might tune those weights again, and the default flips to on only once a benchmark confirms an uplift. For the full specification, including the limitation on temporal coherence, see [the BFS multi-hop design note](../design/bfs_multihop.md).

### `use_bm25f_anchors`

Boolean, default `true` since v1.7.0 (#154 bench gate). This key enables the BM25F sparse-matvec L1 path, which adds anchor text to the belief content (#142) under Porter-stemmed FTS5 indexing.

When you enable the path (the v1.7.0+ default):
- L1 retrieval uses the BM25F implementation in `retrieval.py`, which indexes the belief text together with its anchor terms: the entity mentions, the source paths, and the identifier captures.
- `LaneTelemetry.bm25f_used = True` for the call.
- The composition-tracker bench (#154) measured an uplift of **+0.6650 in normalized discounted cumulative gain at k (NDCG@k)** against the baseline with all the flags off. That bench ran on the lab fixture `tests/corpus/v2_0/retrieve_uplift/v0_1.jsonl`, which holds 30 rows in 6 categories.

When you disable the path:
- L1 falls back to the FTS5-BM25 path of v1.5 and v1.6, and `LaneTelemetry.bm25f_used = False`.

Precedence (the first decisive tier applies): environment variable `AELFRICE_BM25F=0`/`1` > explicit Python kwarg `use_bm25f_anchors=<bool>` > TOML `[retrieval] use_bm25f_anchors` > default `true`.

### `use_heat_kernel`

Boolean, default `false` since #1162. This key enables the heat-kernel authority-scoring lane (#150).

#154 flipped this default to `true` at v2.1.0, once the #437 reproducibility-harness gate cleared 11/11, and it stayed at `true` for two minor versions. The lane is guarded on a `GraphEigenbasisCache` that is not stale, as well as on this flag, and nothing in the shipped pipeline constructs such a cache: `retrieve()` accepts a cache and defaults it to `None`, and only the tests pass one. The flag therefore advertised an active lane that could not fire. The flip back is inert rather than a ranking change, because every call already took the path with the heat kernel off.

A value of `true` is still the opt-in, and the lane is still connected, but the flag on its own does nothing. You must also pass an `eigenbasis_cache`. For whether the branch actually rewrote an ordering, read `LaneTelemetry.heat_used`; the flag cannot answer that question.

Precedence (the first decisive tier applies): environment variable `AELFRICE_HEAT_KERNEL=0`/`1` > explicit Python kwarg > TOML `[retrieval] use_heat_kernel` > default `false`.

### `use_hrr_structural`

Boolean, default `true` since the #154 composition tracker flipped the default, after the #437 reproducibility-harness gate cleared at 11/11. This key enables the HRR structural-query lane (#152). `retrieve_v2` connects the lane as a parallel routing branch, and per the specification that branch does not blend the lane with the textual lane. The lane is **live on the production `retrieve()` path** since the #1107 Phase-5 cutover. Before that cutover the resolver defaulted the lane to ON on `retrieve_v2`, which left it inert on the live hook path, because that path called the legacy `retrieve()`.

The lane is marker-routed. On a query without a marker, which is the shape of a free-text hook prompt, the lane falls through as a byte-identical no-op, so the graduation reaches only the callers that issue `<KIND>:<target_id>` queries. When the key is on, `retrieve_v2` parses the query for a structural marker before any other rewrite runs and before any lane fans out:

```
query string -> parse_structural_marker
              hit:  HRRStructIndex.probe(kind, target_id) -> RetrievalResult
              miss: textual lane (BM25F + heat-kernel + BFS)
```

A marker is a leading uppercase edge-type token, followed by `:` and a non-empty target belief id. The recognized kinds match `aelfrice.models.EDGE_TYPES`, and the current set is `SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`, `RELATES_TO`, `DERIVED_FROM`, `IMPLEMENTS`, `TEMPORAL_NEXT`, `TESTS`, `RESOLVES`. Treat that constant as the source of truth. The match is case-sensitive, so `contradicts:b/abc` does not match and falls through to the textual lane on the literal string. aelfrice preserves whitespace inside the target, and strips leading and trailing whitespace on the query.

Examples:

| Query | Routes to | Returns |
|---|---|---|
| `CONTRADICTS:b/abc` | Structural lane | Beliefs whose outgoing edge of kind `CONTRADICTS` targets `b/abc`, ranked by HRR probe score |
| `SUPPORTS:b/xyz` | Structural lane | Beliefs that `SUPPORTS` `b/xyz` |
| `contradicts everything` | Textual lane | BM25 over the literal string |
| `CONTRADICTS: ` (empty target) | Textual lane; the regex rejects the marker | BM25 over the literal string |
| `CONTRADICTS:nonexistent_id` | Textual lane; the marker parses, but the probe finds no edges | BM25 over the literal string |

On a hit in the structural lane, the locked beliefs pin to the head of the result when `include_locked=True`, bypassing the budget as the existing public-API contract requires. aelfrice then appends the HRR-ranked beliefs in order of descending score until the token budget is exhausted, removing from the HRR tail any belief already in the locked set.

A long-running consumer should pass an explicit `hrr_struct_index_cache: HRRStructIndexCache | None`, which spreads the per-belief cost of the HRR encoding across the queries. A value of None falls through to a fresh build on each call. The cache subscribes to the store's invalidation registry, so mutating a belief or an edge drops the index without further action.

Precedence (the first decisive tier applies): environment variable `AELFRICE_HRR_STRUCTURAL=0`/`1` > explicit Python kwarg `use_hrr_structural=<bool>` > TOML `[retrieval] use_hrr_structural` > default `true`. The flip landed when the #437 reproducibility-harness reached 11/11 (see #154). For parity with the v2.0.x ranking, set the flag to `false`.

### `hrr_persist`

Boolean, default `true` (v3.0+, #698). This key toggles the persistence of the HRR structural index. When you enable persistence, `HRRStructIndexCache` writes the built `(N, dim)` matrix to `<store_dir>/.hrr_struct_index/struct.npy` on the first build, along with the metadata blob `meta.npz`. On every later cold start the cache reads the matrix with `np.load(..., mmap_mode='r')`, which turns a rebuild of about 38 s at N=50k into a warm load of about 1 s, per `docs/design/feature-hrr-integration.md`. The save is atomic: the cache writes a temporary file and then calls `os.replace`, so a reader never sees a partial write.

**Automatic disable on an ephemeral path** (#695). When the store root resolves under `/tmp/`, `/var/tmp/`, `/dev/shm/`, or `/run/`, the cache treats `hrr_persist` as an explicit `false` and logs this line once per process:

```
aelfrice: HRR persistence disabled on ephemeral path <path>; set AELFRICE_HRR_PERSIST=1 to force.
```

To override the automatic disable, set `AELFRICE_HRR_PERSIST=1`. The TOML key cannot override it, because the TOML file lives at the store root and that root is the path the cache checks, so the environment variable is the only override.

Precedence (the first decisive tier applies): environment variable `AELFRICE_HRR_PERSIST` > explicit `persist_enabled=<bool>` on `HRRStructIndexCache(...)` > TOML `[retrieval] hrr_persist` > default `true`. For that environment variable, a truthy `"1"`/`"true"`/`"yes"`/`"on"` forces persistence on, and a falsy `"0"`/`"false"`/`"no"`/`"off"` disables it. A non-boolean TOML value traces to stderr and falls through to the default. The canonical construction site is `aelfrice.retrieval.make_hrr_struct_cache(...)`, which passes the resolved value into the cache for callers that don't resolve the flag themselves.

**When to disable.** The opt-out exists for two cases. The first is a deployment with limited disk space: the blob on disk is 8·N·dim bytes, so at the default dim=512 it is ~41 MB at N=10k and ~200 MB at N=50k, and with the dim=2048 option it is ~800 MB at N=50k. Federation over several stores raises the total further. The second case is a read-only filesystem. You can read the resolved state in `aelf doctor`, in the `hrr.persist_enabled` row, and in `aelf status`, in the `hrr.persist_state` summary line.

### `use_type_aware_compression`

Boolean, default `true` since #769 (v2.1+, #434). This key populates `RetrievalResult.compressed_beliefs` with one rendering per belief, and `belief.retention_class` selects the rendering:

| Retention class | Locked | Unlocked | Notes |
|---|---|---|---|
| `fact` | verbatim | verbatim | Stable codebase state. |
| `snapshot` | verbatim | **headline** | First sentence (split outside ``` fences) + `…`. |
| `transient` | verbatim | **stub** | `[stub: belief={id} class=transient]` marker; full text via `store.get_belief(id)`. |
| `unknown` | verbatim | verbatim | Migration safety. |

The compression is pure and deterministic: it reads no store, no clock, no environment variable, and no random source. The `compressed_beliefs` field is parallel to `beliefs`, with the same length and order. For the raw belief, read `.beliefs[i]`; for the compressed rendering, read `.compressed_beliefs[i].rendered`.

The key is enabled by default, and `compressed_beliefs` is then parallel to `beliefs`, with the same length and order. For v2.x parity, set the environment variable or the TOML key to `false`; `compressed_beliefs` is then empty, and the pack accounts in the raw `_belief_tokens`.

Precedence (the first decisive tier applies): environment variable `AELFRICE_TYPE_AWARE_COMPRESSION=0`/`1` > explicit Python kwarg `use_type_aware_compression=<bool>` > TOML `[retrieval] use_type_aware_compression` > default `true`. The default flipped to on in #769, after the A2 and A4 bench gates (`docs/design/feature-type-aware-compression.md` §"Bench-gate / ship-or-defer policy") cleared on the lab-side `compression_a*` corpora. This key composes with `use_intentional_clustering` since #878.

### `use_temporal_spine` / `temporal_spine_budget`

v4.0.0+ (#1064). Defaults `true` and `32`. These two keys control the
temporal-spine retrieval lane, an additive source of candidates after
L1. The lane traverses the chronological `TEMPORAL_NEXT` chains from the
top-5 packed L1 seeds, in both directions, at depth 1, and appends the
neighbors without displacing L1 before the packing. The mechanism
complements lexical matching: a gold belief that shares no salient term
with the question becomes reachable through its chronological adjacency
to a belief that does match.

A no-op guard applies, so a store with zero `TEMPORAL_NEXT` edges gets
byte-identical output at ~zero cost. Precedence: the
`AELFRICE_TEMPORAL_SPINE` / `AELFRICE_TEMPORAL_SPINE_BUDGET` environment
variables → the explicit kwarg → TOML → the default. The default is
**ON** since the #1107 Phase-2 cutover, so the lane is live on the
production `retrieve()` hook path, not only on `retrieve_v2`. That
cutover followed every pre-registered gate in
[the temporal-spine feature design note](../design/feature-temporal-spine.md).
To opt out, set `AELFRICE_TEMPORAL_SPINE=0` or `[retrieval]
use_temporal_spine = false`.

### Placeholder flags

#154 reserves `use_signed_laplacian` and `use_posterior_ranking`, and neither one's lane has shipped. `warn_placeholder_flags()` recognizes both, so writing either in `.aelfrice.toml` raises no error. Setting either to `true` emits one deprecation warning to stderr and does nothing else. Source of truth: `PLACEHOLDER_FLAGS` in `src/aelfrice/retrieval.py`.

## `[ingest]` (v4.0.0+)

### `write_temporal_spine`

Default `true` since the writer flip of v4.0 (#1064). To opt out, set
`AELFRICE_TEMPORAL_SPINE_WRITE=0` or this key to `false`. Every belief
insert links that belief to the previous belief in the same session with
a `TEMPORAL_NEXT` edge, in `created_at` order, with insertion order
breaking a tie. Those edges form the per-session temporal spine that the
`use_temporal_spine` retrieval lane traverses. That lane is default-on
since the #1107 Phase-2 cutover, so a fresh install writes the spine and
reads it end to end.

The writer adds one edge per belief, at O(1) per insert, and the write
is idempotent. The opt-out path is byte-identical to the path today. For
an existing store that predates the writer, `aelf spine backfill` builds
the chains; it is idempotent and supports `--dry-run`. `aelf doctor`
reports whether the spine is present, along with the edge count. The
`AELFRICE_TEMPORAL_SPINE_WRITE` environment variable overrides this key.

## `[relationship_detector]` (v4.x+)

The deterministic contradiction detector (#201 / #988). Two consumers read
this section, and they are **not** equivalent, so this document gives the
reach of each key. The consumers are:

* the **ingest write path**. It runs on every ingested turn and inserts
  `CONTRADICTS` edges. `auto_detect` gates it completely.
* the **audit commands**. `aelf doctor --relationships` prints a read-only
  report, and `aelf doctor --detect-stale` writes `POTENTIALLY_STALE` edges.
  Both run on demand, and both ignore `auto_detect`.

| Key | Type | Default | Ingest write path | `aelf doctor` audits |
|---|---|---|---|---|
| `auto_detect` | bool | `false` | **honored** — the on/off switch | Not read; the flag gates ingest only |
| `jaccard_min` | float `[0.0, 1.0]` | `0.4` | **honored** (since [#1299](https://github.com/robotrocketscience/aelfrice/issues/1299); ignored without warning before that) | **honored**; `--relationships-jaccard` overrides it |
| `confidence_min` | float `[0.0, 1.0]` | `0.5` | **honored** (since #1299; ignored without warning before that) | **honored**; `--relationships-confidence` overrides it |
| `max_candidate_pairs` | int `>= 1` | `5000` | **honored** (since #1299; ignored without warning before that) | **honored**; `--relationships-max-pairs` overrides it |
| `residual_overlap_min` | — | `0.4` | **No TOML key** — nothing parses it | **No TOML key** |
| `max_edges_per_belief` | — | `8` | **No TOML key** by design (the Exp-48 write gate takes a caller kwarg only) | Not applicable; the audits never write `CONTRADICTS` |

The section therefore still doesn't mean exactly one thing for every key. The
last two rows are module constants with no configuration surface. #1299
changed one thing: the three keys that *do* parse now reach the path that
mutates the graph, where before #1299 they reached only the read-only audit.

`auto_detect` resolves in this order: environment variable > TOML > the
default of off. `AELFRICE_AUTO_RELATIONSHIPS` (`1`/`true`/`yes`/`on` against
`0`/`false`/`no`/`off`) overrides the file. The three thresholds have no
environment override.

The default of off is important, because a fresh install writes no semantic
edges. Once you turn `auto_detect` on, every ingested turn runs an
incremental contradiction audit over the beliefs that turn inserted. A value
of the wrong type falls back to the default and writes a trace to stderr; it
never raises.

## `[hook] provenance_render` (v4.x+)

This key adds trust-tier grouping and evidence attributes to each turn's
injected block (#1326, decomposed from #1177 proposal 18).

With the key off, every belief renders as `<belief id="…" lock="user|none">`,
plus the `speculative="1"` marker from #1171. With the key on, aelfrice groups
the block:

| Section | Membership | What the framing tells the model |
|---|---|---|
| `<user-locked>` | `lock_level == 'user'`, whatever the origin | Standing instructions; verify factual claims against the project first |
| `<observed>` | `user_stated`, `user_corrected`, `user_validated`, `user_transcript`, `document_recent` | Recorded from what the user said, or from what the repo contains; weigh by `n` and `mu` |
| `<inferred>` | `agent_inferred`, `agent_remembered`, `speculative`, `unknown` | The system's own hypotheses; check them, and never treat them as fact |

A non-locked line gains four attributes: `origin`, `n` (= `alpha + beta`),
`mu` (the posterior, to 3 decimal places), and `seen` (the corroboration
count). Every value is already on the belief object at render time, so this
key adds no query. A measurement on a live pack found all four attributes
populated on 74 of 74 retrieved hits.

`n` is the attribute that matters. `mu = 0.6 at n = 2` is byte-identical to
`mu = 0.6 at
n = 200` at every scoring site. The spread is not academic: one live
pack carried 25 distinct `n` values, from 1.6 to 363.2, across 74 hits. A
ranker has to collapse that spread, but a model that sees the number can
weigh it against the question you asked.

Membership is a **total** function of `lock_level` and `origin`, and every
`models.ORIGIN_*` constant has a classification. An unrecognized origin falls
back to `<inferred>` rather than being dropped. That direction is deliberate:
nobody classified such an origin, so nobody established its trustworthiness.
A test enumerates the constants from `models`, so a new origin without a
classification fails the suite.

The default of off is important. The framing header is validated wording
(rule-compliance 0/3 → 5/5), and this key, when on, changes every belief line.
`AELFRICE_PROVENANCE_RENDER` (`1`/`true`/`yes`/`on` against `0`/`false`/`no`/
`off`) overrides the file. A value of the wrong type, or a malformed value,
degrades to off and writes a trace to stderr. Neither one raises.

## `[implicit_feedback]` (v1.6.0+)

This table controls the feedback from retrieval exposure. The queue records
which beliefs a `retrieve()` call surfaced, and `aelf sweep-feedback` reports
on that queue. **Since
[#1162](https://github.com/robotrocketscience/aelfrice/issues/1162)
the sweep is audit-only and writes nothing.** No `alpha` value moves,
aelfrice writes no `feedback_history` row, and no queue status changes.
`epsilon` and `grace_window_seconds` therefore shape only what the audit
*reports*; neither can alter a posterior. Turning implicit exposure back into
real feedback is a separate proposal, and these keys cannot do it.

All three keys resolve in this order: **environment variable > explicit
kwarg > TOML > default**. The **environment** and **TOML** tiers are
fail-soft: each discards a value it cannot use, and the next tier decides.
Exactly one case reports itself. A malformed **environment variable** for
`epsilon` or `grace_window_seconds` prints an
`aelfrice implicit_feedback: ignoring …` trace to stderr before it falls
through. Every other rejection at those two tiers is silent, including a TOML
value of the wrong type for any of the three keys.

`[implicit_feedback]
epsilon = "0.1"` (quoted, therefore a string) resolves to `0.05` with no
message at all, and `AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE=enabled` resolves to
`false`. Check the resolved state, and don't read the absence of a warning
as acceptance.

The **explicit kwarg** tier is the exception: it is strict rather than
fail-soft, and it never falls through. A value that doesn't match the
declared type raises `TypeError` at the call site instead of deferring to
TOML. `grace_window_seconds` takes an `int`, `epsilon` takes a `float` or an
`int`, and `enqueue_on_retrieve` takes a `bool`. The two numeric keys reject
`bool`, as their TOML tiers already do; without that rejection,
`resolve_epsilon(True)` would be an exploration rate of 100%, and
`resolve_grace_seconds(True)` would be a window of one second. Pass a value
that already has the correct type: the tier also rejects a string that parses,
such as `resolve_grace_seconds("900")`.

The type check runs **before** the environment tier, so it holds even when an
environment variable would have decided the value. Only the check moves
earlier; the environment tier still has the higher precedence, and a
correctly typed kwarg still loses to it. A check inside the kwarg branch would
make the same bad call raise on a machine where the variable is unset, and
pass in silence on a machine where it is set.

The split is deliberate. The environment and TOML tiers carry the
configuration you write, and a typo there must not stop a session, so those
tiers discard the value and continue. The kwarg comes from the calling code,
where a discard would hide the caller's bug behind whatever the next tier
returned.

### `enqueue_on_retrieve`

Boolean, default `false` since
[#1162](https://github.com/robotrocketscience/aelfrice/issues/1162). The key
has existed since `v1.6.0+` (#191/#256), and its default was `true` through
v4.2. When the key is true, every `retrieve()` call
writes one queue row per surfaced belief.

The default was `true` on this reasoning: the queue is additive, and nothing
reads a row until the sweeper runs. That held only because the sweeper is a
manual command that nothing schedules, which is a fact about deployment
rather than a property of the design, so real stores collected six-figure row
counts. The key was also a second route to the posterior bump, default-on and
carrying no flag, and
[#1086](https://github.com/robotrocketscience/aelfrice/issues/1086) had
already turned that bump off when it decided that retrieval exposure is
deliberately not evidence.

Leave the key off unless you are measuring exposure specifically. The
environment variable `AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE` accepts
`1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off`, and **only** those values.
It discards anything else without a warning, including a near-miss such as
`enabled` or `y`, and the next tier then decides. `…ENQUEUE=enabled`
therefore resolves to `false`, not to `true`.

### `epsilon`

Float, default `0.05`. The **pre-#1162** sweeper would have added this
increment to `alpha` for each row. The audit reports
`alpha_withheld = would_apply *
epsilon`, so this key scales a reported total and nothing else. A negative
value clamps to `0.0`. The environment variable is
`AELFRICE_IMPLICIT_FEEDBACK_EPSILON`, and the CLI option is
`aelf sweep-feedback --epsilon`.

### `grace_window_seconds`

Integer, default `1800` (30 min). A queue row becomes eligible only when
`enqueued_at + grace_window_seconds <= now`, so a wider window moves rows out
of the eligible count and into `pending_in_grace`. This key sets the
*eligibility* only. The cancellation check that follows spans
`[enqueued_at, now]`, the whole life of the row rather than only the grace
window, so an explicit correction arriving long after the window still counts
as a signal that would have cancelled the row. The environment variable is
`AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS`, and the CLI option is
`aelf sweep-feedback --grace-seconds`.

### Draining a queue of collected rows

A store that ran with the old default carries rows the audit-only sweeper
cannot act on. `aelf sweep-feedback --gc` deletes the `status='enqueued'` rows
**that the same run reported on**, so `--limit` bounds the report and the
deletion together. The command leaves the rows past that limit alone and
reports them separately, and it never touches a row that is already
`applied` or `cancelled`, since those rows are the record of the sweeps that
did run.

## `[rebuilder]` and `[rebuild_floor]` (v1.7+)

A malformed value in either section falls back to the field's default and writes an `aelfrice rebuilder: ignoring …` trace to stderr. A malformed value is one of the wrong type, out of range, or an unrecognized strategy string. The rebuild never raises on a bad configuration value.

### `query_strategy`

String, one of `"legacy-bm25"` or `"stack-r1-r3"`, defaulting to `"legacy-bm25"`. The default was `"stack-r1-r3"` from v3.0 (#718, PR #719) until #1501 reverted it.

| Value | Effect |
|---|---|
| `"stack-r1-r3"` (default v3.0 → #1501) | Runs the R1+R3 query-understanding stack: entity expansion, then per-store IDF clipping. See `aelfrice.query_understanding` for the contract of the rewriter. The stack raised recall only while the FTS5 MATCH was conjunctive. #1177 made that MATCH disjunctive, so the clip now deletes query terms and widens nothing. Measurement across that one commit, on the same 30-row labeled corpus: legacy-bm25 0.3006 → 0.9553, stack-r1-r3 0.5858 → 0.8229, uplift +0.2851 → −0.1324. On one store of 16,454 beliefs, the clip dropped 1,284 of the 1,859 terms that reached it over 200 in-domain queries. Those two measurements are the whole evidence base. Nobody has measured the behavior on a store of a different size or a different shape. |
| `"legacy-bm25"` (default) | Byte-identical to the v1.4 raw-BM25 path: the query reaches `retrieve()` unchanged. |

An unrecognized value traces to stderr and falls back to `"legacy-bm25"`.

### `[rebuild_floor] session`

Float ≥ 0, default `0.10` (v1.7+, #289 / #364). This key sets the minimum composite score for a session-scoped (L2) belief to enter the rebuilt block. The rebuilder skips a belief scoring below this floor and writes a `below_floor_session:…` reason tag in the rebuild log. To disable the floor and pack every session-scoped candidate, set the key to `0.0`.

### `[rebuild_floor] l1`

Float ≥ 0, default `0.40` (v1.7+, #289 / #364). This key sets the minimum composite score for an L1 / L2.5 belief to enter the block. The rebuilder skips a belief below this floor and writes a `below_floor_l1:…` reason tag. To pack every L1 / L2.5 candidate whatever its score, set the key to `0.0`.

The rebuilder rejects negative and non-numeric values, applies the default, and traces the rejection to stderr.

## `[phantom_generation]` (v3.6+)

Opt-in, trigger-driven phantom generation (#980). On every `UserPromptSubmit` turn, aelfrice determines deterministically whether the turn is a *phantom-generation opportunity*, and on such a turn it appends a short `<aelfrice-phantom-opportunity>` note to the injected context suggesting `/aelf:wonder`. Under the #605 determinism boundary, aelfrice only **flags** the opportunity: the LLM synthesis stays an action of the host agent, on the existing `/aelf:wonder` path, and aelfrice never dispatches an LLM. The default is off, and the lane is inert until you enable it.

One flag and one per-session budget cover three signals, which aelfrice combines with OR. The note's `reason` field names the signal that fired.
- **gap** — the prompt retrieved zero stored beliefs.
- **new_entity** — a *named* entity resolves to zero stored beliefs. A named entity is an identifier, a file path, a URL, an error code, a version, or a branch; a loose noun phrase is not.
- **contradiction** — a CONTRADICTS pair appeared after the per-session snapshot. The detector polls and then takes a set difference. This signal is inert unless the #988 semantic-edge substrate is also enabled to write the edges.

### `enabled`

Boolean, default `false`. This key is the top-level opt-in. Precedence (the first decisive tier applies): environment variable `AELFRICE_PHANTOM_GENERATION=1`/`0` (aelfrice normalizes the truthy and falsy values) > explicit Python kwarg > TOML `[phantom_generation] enabled` > default `false`. The resolver has the same shape as the `bfs_enabled` resolver, and a fresh install is unaffected.

### `max_fires_per_session`

Integer ≥ 1, default `3`. This key caps the opportunity notes for each session. All three signals share the cap, and the `session_ring` state tracks it. Each signal also removes its own duplicates, which stops the same opportunity from appearing twice inside one session; the deduplication key is the normalized prompt topic for gap, the entity string for new_entity, and the sorted pair of belief ids for contradiction. This key is TOML-only, with no environment override, matching the precedent of the cadence configuration.

### `auto_dispatch`

Boolean, default `false`. With `false` (the default) the note is passive: it states the opportunity, and you or the agent then decide what to do. With `true` the note instructs the agent to run the `/aelf:wonder` dispatch on the listed topics. This key is TOML-only.

aelfrice skips the trigger on a turn that the prompt-shape gate stopped (#674). The trigger is fail-soft end to end: any error produces no note, and no error breaks the hook. For the full specification, see [the phantom-generation trigger design note](../design/phantom_trigger_generation.md).

## `[phantom_promotion]` (v4.x+)

Opt-in, trigger-driven detection of a **promotion opportunity** for a phantom (#1132). This table is the promotion-side mirror of `[phantom_generation]`. On each `UserPromptSubmit` turn, aelfrice checks deterministically whether a phantom (`origin='speculative'`) has collected enough cross-session corroboration to be worth confirming, and when one has, it appends a short `<aelfrice-phantom-promotion-opportunity>` note naming the candidates and their `aelf validate <id>` / `aelf lock` surface. Promoting an origin stays exactly where the ratified #229 rule put it: it is an explicit act by you. A corroboration count is a **non-trigger** for that write, and this lane decides only *when to prompt* you. It never promotes a phantom on its own. The default is off, and the lane is inert until you enable it.

The detector answers a finding of the #1125 census, which found that phantoms are essentially never promoted, with 0 promotions across seven real stores. The cause is not a broken promotion path; it is that nothing surfaces a corroborated phantom for the explicit act #229 requires.

### `enabled`

Boolean, default `false`. This key is the top-level opt-in. Precedence (the first decisive tier applies): environment variable `AELFRICE_PHANTOM_PROMOTION=1`/`0` (aelfrice normalizes the truthy and falsy values) > explicit Python kwarg > TOML `[phantom_promotion] enabled` > default `false`. The resolver has the same shape as the `[phantom_generation]` resolver, and a fresh install is unaffected.

### `max_fires_per_session`

Integer ≥ 1, default `3`. This key caps the promotion-opportunity notes for each session, and the `session_ring` state tracks the cap independently of the `[phantom_generation]` budget. Per-candidate deduplication, keyed on the phantom's belief id, stops the same candidate from appearing twice inside one session. This key is TOML-only, with no environment override.

### `min_corroborations` / `min_sessions`

Integers ≥ 1, defaults `3` / `2`. aelfrice surfaces a phantom only when three conditions hold together: the phantom has at least `min_corroborations` corroborations; those corroborations come from at least `min_sessions` distinct sessions, excluding the NULL sessions; and the phantom has no inbound CONTRADICTS edge. These thresholds have the same shape as those in the retention-promotion rule (`belief_retention_class.md` §4). To surface fewer candidates at a higher confidence, raise the two keys. Both are TOML-only.

aelfrice skips the trigger on a turn that the prompt-shape gate stopped. The trigger is fail-soft end to end: any error produces no note, and no error breaks the hook. For the full specification, see [the phantom-generation sources design note](../design/phantom_generation_sources.md) §6 (issue #1132).

## `[belief_categories]` (v4.x+)

Opt-in belief categories that a keyword triggers (#1126). A *category* groups beliefs, such as repo-rules, git-workflow, and prose-and-docs, and binds them to an activation trigger. A category fires when it is always-on, or when one of its keyword phrases appears in the prompt. When the lane is enabled and a category fires, the `UserPromptSubmit` hook does three things: it **reranks the retrieval output**, so that category's member rules lead the `<aelfrice-memory>` block; it adds a one-line `<category-focus>` note in front, naming the categories that fired; and it surfaces a bounded set of members that retrieval missed. This lane is the conditional complement to a static `CLAUDE.md` / `AGENTS.md`, because it brings the right rule at the right moment.

The lane **reranks the output rather than injecting a second block**. The #1126 research found that a separate block injects a second copy of what retrieval (L0 + BM25) already returns, and that the category members are almost always already in the tail of the retrieval output. The value is therefore prioritizing and labeling the one block, not adding content.

The lane is **advisory, not enforcement**: it never blocks a tool call. Under the enforcement history (#199) and the #605 determinism boundary, the matching uses the standard library only: it is case-insensitive, respects word boundaries, matches a literal phrase, and uses no embeddings and no model call. The hook is fail-soft, so any error leaves the hits unchanged and returns exit 0. Because the lane reorders existing hits, a locked member is still injected exactly once, with its L0 ground-truth framing; the lane only lifts it to the top.

### `enabled`

Boolean, default `false`. This key is the top-level opt-in. Precedence (the first decisive tier applies): environment variable `AELFRICE_BELIEF_CATEGORIES=1`/`0` (aelfrice normalizes the truthy and falsy values) > TOML `[belief_categories] enabled` > default `false`. A fresh install is unaffected until you enable the lane.

Manage the categories and their membership with the `aelf category` CLI (`init`/`add`/`list`/`show`/`set-trigger`/`assign`/`unassign`/`delete`), or with `aelf lock "<rule>" --category <name>`. `aelf category init` creates a starter set of 5 categories: repo-rules, git-workflow, secrets-and-safety, prose-and-docs, and testing. You drive every category assignment; there is no automatic classification. For the full specification, see [the belief-categories design note](../design/belief_categories.md).

## `[memory]` (v3.7.0+)

This table controls the claude-memory mirror (#985), a one-way `PostToolUse:Write|Edit|MultiEdit` hook. It ingests the host's claude-memory fact-file writes into the belief graph, so the two stores don't drift. `aelf setup` installs the hook, and the hook is default-on. Since v4.0 (#1089) a consent gates the mirror, rather than a flag: the mirror runs when an explicit configuration value enables it, and also when the per-project consent sentinel exists. The one-shot claude-memory reconcile writes that sentinel at the first `aelf setup` for a project, so on a project that has run `aelf setup` the mirror is in effect on without any flag.

When the resolved value is off, the hook returns after three cheap checks — the tool name, the shape of the path, and the flag — and never imports the store. aelfrice is never authoritative over the memory files, and the mirror never locks a belief, because L0 stays reserved for an explicit `aelf lock`.

### `mirror_claude_memory`

Boolean. Precedence (the first decisive tier applies): environment variable `AELFRICE_MIRROR_CLAUDE_MEMORY` > explicit caller kwarg > TOML `[memory] mirror_claude_memory` > the #1089 per-project consent sentinel > default `false`. aelfrice normalizes the truthy and falsy values of that environment variable, and a present sentinel means `true`. **Opt-out:** the environment and TOML tiers outrank the sentinel, so an explicit `AELFRICE_MIRROR_CLAUDE_MEMORY=0` or `mirror_claude_memory = false` disables the mirror even after a consent. The sentinel lives beside the belief store, so an uninstall or a rebuild removes it with the store, and a fresh store asks for consent again at its next `aelf setup`. When the mirror is enabled, a `metadata.type` of `user` or `feedback` ingests as `origin=user_validated` with an undeflated prior, while a `metadata.type` of `project` or `reference`, or an absent `metadata.type`, ingests as `origin=agent_inferred` with a deflated prior. The belief ids come from the content, so a byte-identical rewrite corroborates a belief rather than duplicating it.

## When changes apply

An edit to a `[noise]` key applies on the next `aelf onboard` run, and an edit to `[retrieval] entity_index_enabled` applies on the next `retrieve()` call. An edit never re-filters the beliefs already in the store: the configuration controls ingestion and retrieval, not retention.

To remove existing noise, drop the store and onboard again. The two commands below lose your locks, the beliefs you inserted manually, and the feedback history.

```bash
rm "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"
aelf onboard /path/to/project
```

For a less destructive cleanup, query the store with `sqlite3` and `DELETE` the rows that match.

## What this file does not do

- The `[noise]` table does not affect retrieval. The noise filter runs at onboard time only; the `[retrieval]`, `[rebuilder]`, and `[user_prompt_submit_hook]` tables above control the behavior at retrieval time.
- This file does not affect `aelf lock` or `aelf:lock`. A belief you assert manually bypasses the noise filter.
- This file does not redefine the four built-in categories. You can disable a category, but you cannot change what it matches. Use `exclude_words` or `exclude_phrases` for your own rules.
- This file does not load values from `pyproject.toml`, from environment variables, or from CLI flags.

## Resilience

The file can be malformed, unreadable, or hold values of the wrong type. In each of those cases the filter degrades to the defaults in silence and the onboard does not fail. Every failure traces to stderr.

| Failure | Behavior |
|---|---|
| Malformed TOML | Defaults load, and `malformed TOML in <path>` goes to stderr |
| Wrong-typed field | That field takes its default, and `ignoring [noise] <field>` goes to stderr |
| Non-string entry in a list field | That entry is skipped, and the list still loads |
| Unknown field | Ignored without warning, for forward compatibility |
| Missing file | Defaults load, with no warning |

## See also

- [The `onboard` command reference](COMMANDS.md) — the CLI surface.
- [The modules section of the architecture overview](../concepts/ARCHITECTURE.md) — where `noise_filter.py` sits.
- [Onboarding scope in the limitations list](LIMITATIONS.md) — what is still to come for onboard behavior.

## Pre-issue-create guard (`aelf-pre-issue-hook`, v3.5.0+)

A `PreToolUse:Bash` hook that fires when the agent is about to run `gh issue create`. The
hook checks the proposed title against the open and closed issues, through
`gh issue list --state all`, and against the recent commit messages, through
`git log --grep`. It blocks the call with exit 2 when the Jaccard token
overlap between a candidate and the title is >= 0.5.

**Default:** on. `aelf setup` and the auto-install manifest install the hook automatically.

**Opt-out per call:** there is no inline bypass for a single call. An `ALLOW_DUP_ISSUE=1` prefix on the `gh` command never reaches the hook, because the guard reads the environment variable from the environment of the *host process*, and its command parser strips the leading `KEY=VAL` assignments before it matches. To bypass the guard once:

1. Set `ALLOW_DUP_ISSUE=1` in the host's environment, for example by launching the host with that variable set.
2. Run the command.
3. Unset the variable.

**Opt-out globally (persists across upgrades):**

```bash
AELFRICE_NO_PRE_ISSUE_GUARD=1   # set in shell profile to disable entirely
aelf setup --no-pre-issue-guard  # persist opt-out (~/.aelfrice/opt-out-hooks.json) so upgrades skip it;
                                 # does NOT remove an already-installed settings.json entry — use
                                 # `aelf unsetup` (removes all aelfrice hook entries) for that
```

The guard is deterministic: it uses no embeddings and makes no LLM calls. The
tokenization strips the conventional-commit prefix, such as `feat(scope):` or `fix:`, then
lowercases the text, splits it on runs of non-alphanumeric characters, and
drops a small set of stop words before scoring.
