# Configuration: `.aelfrice.toml`

Most users never need this file. The defaults are tuned so `pip install aelfrice && aelf onboard .` does the right thing.

This is the reference for power users whose project has a documentation idiom or naming convention the default filter mishandles.

## What it does

A single optional TOML file at the root of a project (or any ancestor). It exposes two power-user surfaces:

- `[noise]` — onboard-time belief filter. Changes how `aelf onboard` ingests beliefs; nothing else.
- `[retrieval]` (v1.3+) — retrieval-time tier toggles + ranking. Knobs: `entity_index_enabled` (L2.5), `bfs_enabled` (L3), `posterior_weight` (partial Bayesian-weighted L1 ranking), `use_bm25f_anchors` (BM25F-with-anchor-text since v1.7), `use_heat_kernel` (authority scoring lane, default-on since v2.1), `use_hrr_structural` (HRR structural-query lane, default-on since v2.1), `hrr_persist` (HRR structural-index on-disk persistence, default-on since v3.0), `use_type_aware_compression` (per-belief retention-class compression, opt-in since v2.1). Two placeholder flags (`use_signed_laplacian`, `use_posterior_ranking`) are recognised but emit a deprecation warning if set — their lanes have not yet shipped.

Locks, hooks, MCP tools, and the Bayesian feedback math are not affected.

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

# v2.1+. Default `false`, opt-in. Enables type-aware compression
# (#434) — populates RetrievalResult.compressed_beliefs with per-
# belief renderings keyed by retention_class (snapshot → headline,
# transient → stub, fact + locked → verbatim). The pack-loop budget
# rewrite that turns the parallel field into recall@k uplift is a
# follow-up; this flag at v2.1 just exposes the mechanism behind a
# default-OFF gate. AELFRICE_TYPE_AWARE_COMPRESSION=1 env var
# overrides.
use_type_aware_compression = false

# Placeholder flags reserved by #154 — recognised so callers can
# write forward-compat config, but their lanes have not yet
# shipped. Setting either to true emits a one-shot stderr
# deprecation warning via warn_placeholder_flags() and is
# otherwise a no-op.
# use_signed_laplacian = false
# use_posterior_ranking = false

[onboard.llm]
# v1.3.0+; default flipped to true in v1.5.0 (#238). Host-driven
# classification routes through the host model's Task tool — no API
# key required for the default path. The direct-API path (when the
# host has no Task tool reachable) requires the [onboard-llm] extra
# and the ANTHROPIC_API_KEY env var. To opt out entirely, set this
# to false or pass --llm-classify=false. See docs/llm_classifier.md
# and docs/PRIVACY.md § Optional outbound calls.
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

Host-driven LLM classifier for onboard ingest. Replaces the default regex `classify_sentence` path with the host model's Task tool (no API key required) — direct-API fallback gated by the `[onboard-llm]` extra. Default at v1.5.1+ (#238): on (`enabled = true`); soft-fallback to the regex classifier when no host Task tool is reachable. Boundary policy: [`docs/llm_classifier.md`](llm_classifier.md). Privacy: [`docs/PRIVACY.md § Optional outbound calls`](PRIVACY.md#optional-outbound-calls).

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

Negative values clamp to `0.0`. Non-numeric env values trace to stderr and fall through. The cache key is extended with the resolved weight (rounded to four decimals), so two callers passing different weights against the same store do not collide on a shared `RetrievalCache`.

BM25F-only L1 shipped default-on at v1.7.0 (see `use_bm25f_anchors`); the heat-kernel and HRR-structural lanes shipped default-on at v2.1.0 once the #154 composition-tracker bench gate cleared 11/11 against the #437 reproducibility-harness corpus (see `use_heat_kernel` and `use_hrr_structural` below). See [`docs/bayesian_ranking.md`](bayesian_ranking.md) for the v1.3 contract and the rejected-alternatives analysis.

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

The flag ships default-OFF at v1.3.0 because the literature-default edge weights have not yet been calibrated against the v1.2 corpus. A v1.3.x patch may re-tune them; the default-on flip is deferred until benchmark uplift is confirmed. See [bfs_multihop.md](bfs_multihop.md) for the full spec, including the temporal-coherence limitation.

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

A marker is a leading uppercase edge-type token followed by `:` and a non-empty target belief id. Recognised kinds match `aelfrice.models.EDGE_TYPES` (currently `SUPPORTS`, `CITES`, `RELATES_TO`, `SUPERSEDES`, `CONTRADICTS`, `DERIVED_FROM`). Case-sensitive: `contradicts:b/abc` does not match and falls through to the textual lane on the literal string. Whitespace inside the target is preserved; leading/trailing whitespace on the query is stripped.

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

Boolean, default `true` (v3.0+, #698). Toggles HRR structural-index persistence. When enabled, `HRRStructIndexCache` writes the built `(N, dim)` matrix to `<store_dir>/.hrr_struct_index/struct.npy` (plus the `meta.npz` metadata blob) on first build and `np.load(..., mmap_mode='r')`s it on every subsequent cold start — turning the ~38 s rebuild at N=50k into a ~1 s warm-load per `docs/feature-hrr-integration.md`. The save is atomic via temp-file + `os.replace` so readers never observe a partial write.

**Ephemeral-path auto-disable** (#695). When the store root resolves under one of `/tmp/`, `/var/tmp/`, `/dev/shm/`, or `/run/`, the cache treats `hrr_persist` as if it were explicitly `false` and logs once per process:

```
aelfrice: HRR persistence disabled on ephemeral path <path>; set AELFRICE_HRR_PERSIST=1 to force.
```

Set `AELFRICE_HRR_PERSIST=1` to override the auto-disable. The TOML key cannot override (TOML lives at the store root which is itself the path being checked); the env var is the only escape hatch.

Precedence (first decisive wins): env var `AELFRICE_HRR_PERSIST` (truthy `"1"`/`"true"`/`"yes"`/`"on"` forces on; falsy `"0"`/`"false"`/`"no"`/`"off"` disables) > explicit `persist_enabled=<bool>` on `HRRStructIndexCache(...)` > TOML `[retrieval] hrr_persist` > default `true`. Non-boolean TOML values trace to stderr and fall through to the default. The canonical construction site is `aelfrice.retrieval.make_hrr_struct_cache(...)`, which threads the resolved value into the cache for callers that don't manage flag resolution themselves.

**When to disable.** Disk-constrained deployments (the on-disk blob is 328 MB at N=10k, 1.64 GB at N=50k; federation × multiple stores amplifies) and read-only filesystems are the two cases the opt-out exists for. Operators see the resolved state via `aelf doctor` (`hrr.persist_enabled` row) and `aelf status` (`hrr.persist_state` summary line).

### `use_type_aware_compression`

Boolean, default `false`, opt-in (v2.1+, #434). Populates `RetrievalResult.compressed_beliefs` with per-belief renderings dispatched by `belief.retention_class`:

| Retention class | Locked | Unlocked | Notes |
|---|---|---|---|
| `fact` | verbatim | verbatim | Stable codebase state. |
| `snapshot` | verbatim | **headline** | First sentence (split outside ``` fences) + `…`. |
| `transient` | verbatim | **stub** | `[stub: belief={id} class=transient]` marker; full text via `store.get_belief(id)`. |
| `unknown` | verbatim | verbatim | Migration safety. |

Compression is pure and deterministic — no store, clock, env, or random reads. The `compressed_beliefs` field is parallel to `beliefs` (same length, same order); consumers that want the raw belief read `.beliefs[i]`, consumers that want the compressed render read `.compressed_beliefs[i].rendered`.

When disabled (default), `compressed_beliefs` is empty and `beliefs` is byte-identical to the v1.x return shape.

Precedence (first decisive wins): env var `AELFRICE_TYPE_AWARE_COMPRESSION=0`/`1` > explicit Python kwarg `use_type_aware_compression=<bool>` > TOML `[retrieval] use_type_aware_compression` > default `false`. The default-on flip is gated on the lab-side bench in `tests/bench_gate/test_compression_uplift.py` plus the pack-loop budget rewrite (follow-up).

### Placeholder flags

`use_signed_laplacian` and `use_posterior_ranking` are reserved by #154 but their owning lanes have not yet shipped. The flags are recognised by `warn_placeholder_flags()` so writing them in `.aelfrice.toml` does not error; setting either to `true` emits a one-shot stderr deprecation warning and is otherwise a no-op. Source of truth: `PLACEHOLDER_FLAGS` in `src/aelfrice/retrieval.py`.

## When changes apply

Edits apply on the next `aelf onboard` run for `[noise]` keys. `[retrieval] entity_index_enabled` applies on the next `retrieve()` call. They do not retroactively re-filter beliefs already in the store — config controls ingestion and retrieval, not retention.

To remove existing noise: drop and re-onboard.

```bash
rm "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"
aelf onboard /path/to/project
```

Locks, manually inserted beliefs, and feedback history will be lost. For a less destructive cleanup, query the store with `sqlite3` and `DELETE` rows that match.

## What this file does not do

- Does not affect retrieval. The filter only runs at onboard time.
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
- [ARCHITECTURE § Modules](ARCHITECTURE.md) — where `noise_filter.py` sits.
- [LIMITATIONS § Onboarding scope](LIMITATIONS.md) — what's still on the horizon for onboard behaviour.
