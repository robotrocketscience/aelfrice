"""Four-layer retrieval: L0 locked beliefs, L2.5 entity-index, L1 FTS5
BM25, L3 BFS multi-hop graph traversal.

Token-budgeted output (default 2400 tokens at v1.3.0, ~4 chars/token
estimate). L0 beliefs always present in the output above any non-locked
result and never trimmed by the budget — locks are user-asserted ground
truth and must survive retrieval.

L2.5 (v1.3.0) is a deterministic entity-index lookup that runs between
L0 and L1. It extracts entities from the query using
`aelfrice.entity_extractor.extract_entities`, looks them up in the
`belief_entities` table, ranks by entity-overlap count (tie-break:
belief_id ASC), and feeds a `DEFAULT_L25_TOKEN_SUBBUDGET`-sized slice
into the output ahead of L1. L1 fills the remaining budget.

L3 (v1.3.0) is `aelfrice.bfs_multihop.expand_bfs` — edge-type-weighted
BFS over outbound edges from the L0+L2.5+L1 seed set. Bounded depth /
fanout / total-budget; multiplicative path-score over the table in
`bfs_multihop.BFS_EDGE_WEIGHTS`. Default-OFF at v1.3.0; opt in via the
`bfs_enabled` flag in `[retrieval]` of `.aelfrice.toml`, the
`AELFRICE_BFS=1` env var, or an explicit kwarg. See
`docs/design/bfs_multihop.md` for the spec.

Default-on at v1.3.0 via the config flag `entity_index_enabled` in
`[retrieval]` of `.aelfrice.toml`. Two off-switches:

  - `AELFRICE_ENTITY_INDEX=0` env var (emergency disable; matches the
    v1.2.x `AELFRICE_SEARCH_TOOL=0` convention).
  - `entity_index_enabled=False` kwarg on `retrieve()` /
    `retrieve_v2()`.

When BOTH flags are off — for any reason — `retrieve()` reproduces the
v1.2 byte-identical L0 + L1 path with the v1.0 default budget of 2000
tokens. When only `bfs_enabled` is off (the v1.3.0 default) and L2.5
is on (also default), `retrieve()` is byte-identical to the entity-
index-enabled v1.3.0 baseline. Both invariants are guarded by
regression tests.

NO HRR in v1.3.0. That lands at v2.0.0.

A `RetrievalCache` wrapper provides bounded LRU memoization. Cache
invalidation is wired through the store's callback registry, which
fires on every belief / edge / entity-row mutation (the entity rows
mutate inside `insert_belief` / `update_belief` / `delete_belief`,
so the existing callback semantics already cover them). The v1.0.1
wipe-on-write policy on edge mutators (`insert_edge`, `update_edge`,
`delete_edge`) is exactly what makes the v1.3 BFS cache correctness
zero-effort — see docs/design/bfs_multihop.md § Cache invalidation.
"""
from __future__ import annotations

import functools
import math
import os
import re
import sys
import time
import tomllib
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Final

from aelfrice.bfs_multihop import (
    DEFAULT_MAX_DEPTH as BFS_DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_PATH_SCORE as BFS_DEFAULT_MIN_PATH_SCORE,
    DEFAULT_NODES_PER_HOP as BFS_DEFAULT_NODES_PER_HOP,
    DEFAULT_TOTAL_BUDGET_NODES as BFS_DEFAULT_TOTAL_BUDGET_NODES,
    expand_bfs,
)
from aelfrice.bm25 import BM25IndexCache
from aelfrice.clustering import (
    DEFAULT_CLUSTER_DIVERSITY_TARGET,
    DEFAULT_CLUSTER_EDGE_FLOOR,
    cluster_candidates,
    pack_max_coverage,
    pack_with_clusters,
)
from aelfrice.compression import CompressedBelief, compress_for_retrieval
from aelfrice.doc_linker import DocAnchor
from aelfrice.hrr import DEFAULT_DIM
from aelfrice.hrr_index import (
    HRRStructIndex,
    HRRStructIndexCache,
    parse_structural_marker,
)
from aelfrice.entity_extractor import extract_entities
from aelfrice.graph_spectral import (
    DEFAULT_BM25_SEED_TOP_K,
    DEFAULT_HEAT_BANDWIDTH,
    DEFAULT_HEAT_KERNEL_WEIGHT,
    DEFAULT_POSTERIOR_LOG_WEIGHT,
    HEAT_SCORE_FLOOR,
    GraphEigenbasisCache,
    combine_log_scores,
    heat_kernel_score,
    seeds_from_bm25,
)
from aelfrice.models import (
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    ORIGIN_RETRIEVAL_PRIORITY,
    ORIGIN_RETRIEVAL_PRIORITY_DEFAULT,
    Belief,
)
from aelfrice.scoring import (
    DEFAULT_POSTERIOR_WEIGHT,
    ZETA_ALPHA_DEFAULT,
    ZETA_BETA_DEFAULT,
    ZETA_SCALE_DEFAULT,
    gamma_posterior_score,
    partial_bayesian_score,
    posterior_mean,
    zeta_posterior_score,
)
from aelfrice.store import MemoryStore
from aelfrice.utterance_prior import (
    UtterancePrior,
    resolve_utterance_prior_weight,
    utterance_logodds,
    utterance_prior_penalty,
)

# v1.0 / v1.2 baseline. Used by the disabled-flag fallback so the
# byte-identical regression test sees the same budget the v1.2 caller
# would have seen.
LEGACY_TOKEN_BUDGET: Final[int] = 2000

# v1.3.0 expanded default. L2.5 fills against a sub-budget of 400 and
# L1 fills against the remaining 2000 — preserves the v1.0 L1
# behaviour byte-for-byte on queries where L2.5 returns nothing.
DEFAULT_TOKEN_BUDGET: Final[int] = 2400

# Reserved relevance-budget floor. L0 locked beliefs are injected
# unconditionally and never trimmed (#379), so a store whose locks alone
# meet or exceed `effective_budget` left ZERO budget for query-relevant
# L2.5/L1 hits — every retrieval returned only the locks, regardless of the
# prompt (observed on a real 12-lock store: 2485 lock tokens vs a 2400
# budget → 0 relevance tokens). This reserves at least
# `floor(effective_budget * RELEVANCE_BUDGET_FLOOR_FRACTION)` tokens for
# L2.5+L1. It is a strict no-op (byte-identical) whenever the locks leave at
# least that much room — i.e. it only fires once locks consume more than
# `(1 - fraction)` of the budget — so lock-light corpora (e.g. LoCoMo) are
# unaffected. Locks are still never trimmed; in that regime total output may
# exceed the nominal budget by up to the floor, the intended trade for never
# going blind.
#
# #1023: raised 0.25 -> 0.50. On a real lock-saturated store (24 locks =
# 3491 tok vs a 1500 budget) this doubles surfaced relevance hits (8 -> 16)
# for ~9% more total tokens (3825 -> 4182) — cheap because the locks already
# dominate the injection. It also widens engagement to locks > 50% of budget
# (was > 75%); diminishing BM25-relevance past ~0.5 makes it the knee.
RELEVANCE_BUDGET_FLOOR_FRACTION: Final[float] = 0.50

_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_L1_LIMIT: Final[int] = 50

# v1.3.0 entity-index defaults (docs/design/entity_index.md § Budget split).
DEFAULT_L25_LIMIT: Final[int] = 20
DEFAULT_L25_TOKEN_SUBBUDGET: Final[int] = 400
DEFAULT_QUERY_ENTITY_CAP: Final[int] = 16

DEFAULT_CACHE_CAPACITY: Final[int] = 256

# Section / key names in `.aelfrice.toml`. Public so consumers can
# reference them in their own config.
CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
RETRIEVAL_SECTION: Final[str] = "retrieval"
ENTITY_INDEX_FLAG: Final[str] = "entity_index_enabled"
# #1279 exploration slot — `[retrieval] exploration_*` TOML tier.
EXPLORATION_FLAG: Final[str] = "exploration_enabled"
EXPLORATION_CADENCE_FLAG: Final[str] = "exploration_cadence"
EXPLORATION_SLOTS_FLAG: Final[str] = "exploration_slots"
BFS_FLAG: Final[str] = "bfs_enabled"
POSTERIOR_WEIGHT_FLAG: Final[str] = "posterior_weight"
# v1.5.0 BM25F flag. Default-ON since v1.7.0: the #154 composition-
# tracker bench cleared with +0.6650 NDCG@k uplift on the v0.1
# retrieve_uplift fixture, so the FTS5 BM25 path is no longer the L1
# default. Opt out via the kwarg, AELFRICE_BM25F=0, or
# `[retrieval] use_bm25f_anchors = false`.
BM25F_FLAG: Final[str] = "use_bm25f_anchors"

# #1166 query-term-frequency saturation on the BM25F lane. Default
# `bm25.DEFAULT_K3` = 0.0, which discards qf and so reproduces the
# pre-#1166 ranking byte-for-byte. Raising it is a retrieval-quality
# change, gated on its own bench; see `resolve_bm25_k3`.
BM25_K3_FLAG: Final[str] = "bm25_k3"

# #1180 per-field BM25F. When on, the anchor stream is normalised by its
# own length and `avgdl` instead of being concatenated into the content
# document, and `anchor_weight` acts as a field weight rather than a
# replication count. Default **False**: this changes the scoring
# functional form, so it cannot be made parity-exact by any choice of
# constants and the flip is gated on a bench, not on a parity test.
BM25F_PER_FIELD_FLAG: Final[str] = "bm25f_per_field"

# #1180 anchor-stream length-normalisation strength. Only consulted when
# `bm25f_per_field` is on. See `bm25.DEFAULT_B_ANCHOR` for why it
# defaults to the content stream's `b`.
BM25_B_ANCHOR_FLAG: Final[str] = "bm25_b_anchor"

# v1.5.0 #154 composition-tracker placeholder flags. The components
# ship across v1.6 / v1.7. Each was a no-op placeholder at v1.5.0;
# `HEAT_KERNEL_FLAG` is the first to leave the placeholder set as the
# heat-kernel scorer (#150) lands here. Listed in `PLACEHOLDER_FLAGS`
# only while the lane is still unwired — once the wiring lands, the
# flag is removed from the placeholder tuple so the deprecation
# warning stops firing for users who set it.
SIGNED_LAPLACIAN_FLAG: Final[str] = "use_signed_laplacian"
HEAT_KERNEL_FLAG: Final[str] = "use_heat_kernel"
POSTERIOR_RANKING_FLAG: Final[str] = "use_posterior_ranking"
HRR_STRUCTURAL_FLAG: Final[str] = "use_hrr_structural"
# #981 HRR vocabulary-bridge *expansion* lane flag. Distinct from the #152
# structural lane (HRR_STRUCTURAL_FLAG): structural is a marker-routed query
# path that replaces the textual lane; expansion is an additive candidate
# source that probes the same struct index for single-hop semantic
# neighbours of the FTS5 seeds and merges them before scoring. Default-OFF
# (resolver default False) — landing the lane + ablation only; a default
# flip reverses locked #605 and is routed to a re-opened #897.
HRR_EXPAND_FLAG: Final[str] = "use_hrr_expand"
# #1096 entity-persistence demotion lane flag. Default-ON (resolver
# default True since the v4.0 flip, after the #1103 G2 mixed-corpus eval
# cleared the no-regression gate) — a mild log-additive demotion of
# ephemeral coordination hits (low referential grounding) in the L1
# rerank, so junk percolates down.
ENTITY_PERSIST_DEMOTE_FLAG: Final[str] = "use_entity_persist_demote"
# Mild weight: G3 ablation showed 1.0 == 2.0 in AUC, so a tie-breaking
# term that never dominates BM25 relevance (a relevant low-grounding
# belief is still rescued by its relevance score).
ENTITY_PERSIST_DEMOTE_WEIGHT: Final[float] = 1.0
# #1089 axis-2 origin-priority tie-break flag. Default-OFF (resolver
# default False) — breaks a bm25 tie in favour of the higher-priority
# origin (curated user_validated over conversational user_transcript),
# never a primary rerank term (that lane was refuted in #1013). The
# default flip is gated on the LoCoMo bench and is a separate operator
# call.
ORIGIN_TIEBREAK_FLAG: Final[str] = "use_origin_tiebreak"
# Floor so a durable-free belief (S1 = 0) gets a bounded, not infinite,
# log penalty.
ENTITY_PERSIST_DEMOTE_EPS: Final[float] = 1e-3
# #1187 supersession lane. Default-OFF (resolver default False) and it
# stays that way: unlike the #1170 BFS fix this changes `retrieve()` output
# on the **default** path, so the flip waits on the three-arm bench
# (demote vs exclusion vs control) the operator ratified on 2026-07-29.
# Both arms ship; the bench picks the default, nothing here presumes it.
SUPERSESSION_DEMOTE_FLAG: Final[str] = "use_supersession_demote"
# Treatment selector. `demote` reduces the superseded belief's rerank
# score; `exclude` drops it from the candidate set outright. The
# trade-off the bench has to settle: exclusion is the stronger reading of
# "the user retired this claim", but a SUPERSEDES edge can be written by
# the triple extractor from prose that merely looks like a supersession,
# and a wrong exclusion hides a belief with no ranking signal to notice.
SUPERSESSION_TREATMENT_FLAG: Final[str] = "supersession_treatment"
SUPERSESSION_TREATMENT_DEMOTE: Final[str] = "demote"
SUPERSESSION_TREATMENT_EXCLUDE: Final[str] = "exclude"
SUPERSESSION_TREATMENTS: Final[tuple[str, ...]] = (
    SUPERSESSION_TREATMENT_DEMOTE,
    SUPERSESSION_TREATMENT_EXCLUDE,
)
# #1274 injection-block ordering policy (proposal 14 on #1177). Position in
# the rendered block is currently a side effect of lane concatenation order,
# not a choice anyone made. This names the policies so the question is a
# config flip rather than a code change. Default `lane` is the identity
# permutation, so the shipped block stays byte-identical.
ORDER_POLICY_FLAG: Final[str] = "order_policy"
ORDER_POLICY_LANE: Final[str] = "lane"
ORDER_POLICY_SCORE_DESC: Final[str] = "score_desc"
ORDER_POLICY_LOCKS_LAST: Final[str] = "locks_last"
ORDER_POLICIES: Final[tuple[str, ...]] = (
    ORDER_POLICY_LANE,
    ORDER_POLICY_SCORE_DESC,
    ORDER_POLICY_LOCKS_LAST,
)

# #1279: defaults live in `aelfrice.exploration`, which is kept free of
# config/IO imports; the resolvers around them live here with every other
# `[retrieval]` lane flag.
from aelfrice.exploration import (  # noqa: E402, PLC0415
    DEFAULT_EXPLORATION_CADENCE,
    DEFAULT_EXPLORATION_SLOTS,
)

SUPERSESSION_FACTOR_FLAG: Final[str] = "supersession_demote_factor"
# 0.5 per the issue spec, the same default the uri_baki primitive carries.
SUPERSESSION_DEMOTE_FACTOR: Final[float] = 0.5
# Floor so `factor = 0` is a bounded penalty rather than `log(0) = -inf`,
# which would make the score non-comparable (and NaN once summed).
SUPERSESSION_FACTOR_EPS: Final[float] = 1e-6
TEMPORAL_SPINE_FLAG: Final[str] = "use_temporal_spine"
TEMPORAL_SPINE_BUDGET_FLAG: Final[str] = "temporal_spine_budget"
# v2.1 #434 type-aware compression flag. Default-ON since the #769 flip
# (the A2 + A4 bench gates in docs/design/feature-type-aware-compression.md
# cleared). ON populates RetrievalResult.compressed_beliefs with per-belief
# CompressedBelief renderings; OFF leaves the field empty for byte-identical
# behavior with v1.x adapters.
TYPE_AWARE_COMPRESSION_FLAG: Final[str] = "use_type_aware_compression"
# v2.0 #436 intentional-clustering flag. Default-ON since v3.0: the
# A4 latency gate cleared on the multi-store production sweep (#436
# R6, 60/60 PASS at p99 0.328ms — ~15-30x margin under the 5ms
# budget). When ON, the L1 pack loop is replaced with a diversity-
# aware greedy fill that biases the top-K toward distinct graph-
# connected clusters; locked + L2.5 are pre-included unchanged.
# Composes with use_type_aware_compression since #878 — the cluster
# pack accepts a cost callable and reads compressed rendered_tokens
# when the compression flag is also ON. Opt out via the kwarg,
# AELFRICE_INTENTIONAL_CLUSTERING=0, or
# `[retrieval] use_intentional_clustering = false`.
INTENTIONAL_CLUSTERING_FLAG: Final[str] = "use_intentional_clustering"

# #1176 proposal 2. Budgeted maximum coverage as the L1 pack selector,
# replacing the cluster pack. Default OFF: the stage-1 measurement showed
# the objective has headroom (every replayed pack carries near-duplicate
# pairs, and the shipped cluster pass does not reduce them), but "has
# headroom" is not "is better end to end" -- that needs the A/B.
MAX_COVERAGE_PACK_FLAG: Final[str] = "use_max_coverage_pack"
ENV_MAX_COVERAGE_PACK: Final[str] = "AELFRICE_MAX_COVERAGE_PACK"
# v3.x #796 γ rerank flag. Default-OFF until a labeled relevance corpus
# exists and the bench panel (PR@5 + ρ + ordered_top_k_overlap +
# rank_biased_overlap) demonstrates uplift over log-additive. When ON,
# `_l1_hits` routes its rerank through `gamma_posterior_score(...)` with
# `T = resolve_posterior_temperature_with_meta(...)` (defaults to 1.0
# when the meta-belief is absent — byte-identical to
# `partial_bayesian_score` at `posterior_weight = 1.0`).
USE_GAMMA_POSTERIOR_TEMPERATURE_FLAG: Final[str] = (
    "use_gamma_posterior_temperature"
)
# v3.x #817 / #800 ζ rerank flag. Default-OFF, mirrors γ's posture:
# the flag is plumbing until a labeled relevance corpus exists and
# the bench panel demonstrates uplift over either log-additive or γ.
# When ON, `_l1_hits` routes its rerank through `zeta_posterior_score(...)`
# with pinned `(ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT)`
# from the #800 R&D campaign verdict. ζ + γ are mutually exclusive on
# any given retrieval call — `retrieve()` / `retrieve_with_tiers()`
# raise `ValueError` at flag resolution when both are on (the operator
# decision per issue #817 §"Out of scope" deferred composition).
USE_ZETA_POSTERIOR_RERANK_FLAG: Final[str] = "use_zeta_posterior_rerank"

PLACEHOLDER_FLAGS: Final[tuple[str, ...]] = (
    SIGNED_LAPLACIAN_FLAG,
    POSTERIOR_RANKING_FLAG,
)

# Env var override. Set to "0", "false", or "no" to force-disable
# the index. Unset / any other value falls through to the TOML
# config (which defaults to True at v1.3.0). Same convention as the
# v1.2.x `AELFRICE_SEARCH_TOOL=0` off-switch.
ENV_ENTITY_INDEX: Final[str] = "AELFRICE_ENTITY_INDEX"
# BFS env override. Symmetric to ENV_ENTITY_INDEX but with default
# OFF at v1.3.0 — set to "1", "true", "yes", "on" to opt in. The
# default-off contract means the env-var omission is the same as
# the explicit-off case.
ENV_BFS: Final[str] = "AELFRICE_BFS"
# v1.5.0 BM25F env override. Tri-state like ENV_BFS — unset means
# "fall through" rather than "force off", because the default at
# v1.5.0 is already off.
ENV_BM25F: Final[str] = "AELFRICE_BM25F"
# v1.7.0 heat-kernel env override. Tri-state like ENV_BM25F.
ENV_HEAT_KERNEL: Final[str] = "AELFRICE_HEAT_KERNEL"
# v1.7.0 HRR structural-query env override. Tri-state like ENV_BM25F.
ENV_HRR_STRUCTURAL: Final[str] = "AELFRICE_HRR_STRUCTURAL"
# #981 HRR expansion-lane env override. Tri-state like ENV_BM25F; default-OFF.
ENV_HRR_EXPAND: Final[str] = "AELFRICE_HRR_EXPAND"
# #1096 entity-persistence demotion env override. Tri-state; default-OFF.
ENV_ENTITY_PERSIST_DEMOTE: Final[str] = "AELFRICE_ENTITY_PERSIST_DEMOTE"
# #1089 axis-2 origin-priority tie-break env override. Tri-state; default-OFF.
ENV_ORIGIN_TIEBREAK: Final[str] = "AELFRICE_ORIGIN_TIEBREAK"
# #1176 proposal 3 — ACT-R fan-effect ranker on the L2.5 entity lane.
# Tri-state like ENV_ORIGIN_TIEBREAK; default-OFF pending the A/B.
ENV_FAN_EFFECT: Final[str] = "AELFRICE_FAN_EFFECT"
# #1279 exploration slot (#1176 proposal 5). Default OFF.
ENV_EXPLORATION: Final[str] = "AELFRICE_EXPLORATION"
ENV_EXPLORATION_CADENCE: Final[str] = "AELFRICE_EXPLORATION_CADENCE"
ENV_EXPLORATION_SLOTS: Final[str] = "AELFRICE_EXPLORATION_SLOTS"
# #1187 supersession lane env overrides. Tri-state; default-OFF.
ENV_SUPERSESSION_DEMOTE: Final[str] = "AELFRICE_SUPERSESSION_DEMOTE"
ENV_SUPERSESSION_TREATMENT: Final[str] = "AELFRICE_SUPERSESSION_TREATMENT"
ENV_SUPERSESSION_FACTOR: Final[str] = "AELFRICE_SUPERSESSION_FACTOR"
# #1274 injection-block ordering policy. Default `lane` (identity).
ENV_ORDER_POLICY: Final[str] = "AELFRICE_ORDER_POLICY"
# #1064 temporal-spine lane flag + node-budget knob.
ENV_TEMPORAL_SPINE: Final[str] = "AELFRICE_TEMPORAL_SPINE"
ENV_TEMPORAL_SPINE_BUDGET: Final[str] = "AELFRICE_TEMPORAL_SPINE_BUDGET"
# #698 HRR persist env override. "0" disables; "1" forces on.
# Mirrors _ENV_PERSIST in hrr_index (same value; imported at call site).
ENV_HRR_PERSIST: Final[str] = "AELFRICE_HRR_PERSIST"
# #698 `[retrieval] hrr_persist` TOML key.
HRR_PERSIST_FLAG: Final[str] = "hrr_persist"
# v2.1 #434 type-aware compression env override. Tri-state.
ENV_TYPE_AWARE_COMPRESSION: Final[str] = "AELFRICE_TYPE_AWARE_COMPRESSION"
# v2.0 #436 intentional-clustering env override. Tri-state.
ENV_INTENTIONAL_CLUSTERING: Final[str] = "AELFRICE_INTENTIONAL_CLUSTERING"
# v3.x #796 γ rerank env override. Tri-state, default-OFF.
ENV_USE_GAMMA_POSTERIOR_TEMPERATURE: Final[str] = (
    "AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE"
)
# v3.x #817 ζ rerank env override. Tri-state, default-OFF.
ENV_USE_ZETA_POSTERIOR_RERANK: Final[str] = (
    "AELFRICE_USE_ZETA_POSTERIOR_RERANK"
)
# v1.3.0 posterior-weight env override. Float-typed; "0.0" is the
# only value that fully disables (collapsing to BM25-only ordering).
# Empty / non-numeric values fall through to the next precedence
# layer (kwarg → TOML → DEFAULT_POSTERIOR_WEIGHT) and trace to
# stderr. Same shape as `_read_toml_flag_for` tolerance.
ENV_POSTERIOR_WEIGHT: Final[str] = "AELFRICE_POSTERIOR_WEIGHT"
# #1166 BM25F query-term-frequency saturation. Float >= 0; 0.0 keeps
# qf discarded (the shipped default). Empty / non-numeric values fall
# through to the next precedence layer and trace to stderr. A *negative*
# value does not fall through — it is decisive at this layer and clamps
# to 0.0, matching `resolve_posterior_weight`, so `AELFRICE_BM25_K3=-1`
# means "qf off" rather than "consult the TOML".
ENV_BM25_K3: Final[str] = "AELFRICE_BM25_K3"
# #1180 per-field BM25F env override. Truthy/falsy normalised; an
# unrecognised value falls through to the kwarg → TOML → default chain.
ENV_BM25F_PER_FIELD: Final[str] = "AELFRICE_BM25F_PER_FIELD"
# #1180 anchor-stream `b` env override. Float; empty / non-numeric falls
# through. Negative clamps to 0.0, matching `ENV_BM25_K3`'s posture.
ENV_BM25_B_ANCHOR: Final[str] = "AELFRICE_BM25_B_ANCHOR"
# v2.1 #473 temporal-decay half-life env override. Float seconds.
# Empty / non-numeric values fall through (kwarg → TOML → default).
ENV_TEMPORAL_HALF_LIFE: Final[str] = "AELFRICE_TEMPORAL_HALF_LIFE_SECONDS"
# `[retrieval] temporal_half_life_seconds` TOML key (#473).
TEMPORAL_HALF_LIFE_FLAG: Final[str] = "temporal_half_life_seconds"
# v2.1 #473 default half-life: 7 days. Ratified A1=A by the operator
# (issue #473 comment IC_kwDOSM7PXc8AAAABBokOsA). Conservative default
# tunable via `[retrieval] temporal_half_life_seconds` in
# `.aelfrice.toml`. A bench-evidence sweep harness is queued as a
# follow-up issue (A3=A).
DEFAULT_TEMPORAL_HALF_LIFE_SECONDS: Final[float] = 7.0 * 24.0 * 3600.0
# v3.x #756 meta-belief consumer for the temporal half-life. See
# umbrella #480 + the operator ratification on #756 (2026-05-13)
# for the design call: encoding lives in the consumer, the substrate
# stays pattern-uniform across B–F, and bounds are picked at the
# consumer's natural scale. Log-linear with [3d, 14d] bounds; v=0.5
# (cold start) → ~6.5d, close to the #473 ratified 7d static.
META_HALF_LIFE_KEY: Final[str] = "meta:retrieval.temporal_half_life_seconds"
HALF_LIFE_FLOOR_SECONDS: Final[float] = 3.0 * 24.0 * 3600.0
HALF_LIFE_CEIL_SECONDS: Final[float] = 14.0 * 24.0 * 3600.0
# Static-default `value` for the meta-belief. Mid-range under the
# log-linear bounds chosen above so the cold-start surfaced half-life
# lands near the #473 ratified 7d.
META_HALF_LIFE_STATIC_DEFAULT: Final[float] = 0.5
# Sub-posterior decay half-life — how fast the meta-belief itself
# forgets old evidence. 30d so a one-off latency spike doesn't shift
# the half-life; sustained evidence does. Slower than the surfaced
# half-life value's own [3d, 14d] band, per the #756 init spec.
META_HALF_LIFE_POSTERIOR_DECAY_SECONDS: Final[int] = 30 * 24 * 3600
# Default-OFF feature flag. Until the bench-gate clears, every
# retrieval still uses the v2.1 #473 static path; setting this env
# var truthy switches `resolve_temporal_half_life_with_meta` to
# read the meta-belief from the store first.
ENV_META_BELIEF_HALF_LIFE: Final[str] = "AELFRICE_META_BELIEF_HALF_LIFE"
# Target wall-time per the #437 bench latency floor. Evidence per
# retrieval event is `clip(target / observed, 0, 1)` — fast retrieval
# saturates at 1.0, slow retrieval shrinks toward 0. Per the ratified
# wiring (D4: latency signal only, relevance deferred to #779).
LATENCY_TARGET_SECONDS: Final[float] = 0.080
# v3.x #757 meta-belief consumer for the BM25F anchor-weight knob.
# Sub-task C of umbrella #480, reuses the #756 pattern (encoding in the
# consumer, substrate pattern-uniform). BM25F in aelfrice has a single
# tunable: `anchor_weight` (DEFAULT_ANCHOR_WEIGHT = 3 in bm25.py), the
# weight on the incoming-anchor token stream relative to the content
# stream. The issue's per-field framing assumed a multi-field BM25F
# that aelfrice does not have — scope ratified down to anchor_weight
# only (2026-05-13 review).
META_BM25F_ANCHOR_WEIGHT_KEY: Final[str] = "meta:retrieval.bm25f_anchor_weight"
# anchor_weight is an integer >= 0 (see bm25.py:217). Bounds `[1, 10]`
# avoid the degenerate `0` case (anchors fully disabled, equivalent to
# vanilla BM25 — silently turning off a feature flagged ON by default
# would be a regression vector) and cap at 10 to keep the L1 lane within
# the operator-reasoned range where the #148 R3 default of 3 was chosen.
BM25F_ANCHOR_WEIGHT_FLOOR: Final[int] = 1
BM25F_ANCHOR_WEIGHT_CEIL: Final[int] = 10
# Static-default `value` for the meta-belief. Mid-range so cold-start
# decodes to ~3.16 → round to 3, matching DEFAULT_ANCHOR_WEIGHT exactly
# and preserving byte-identical retrieval order on first install.
META_BM25F_ANCHOR_WEIGHT_STATIC_DEFAULT: Final[float] = 0.5
# Sub-posterior decay. 30d matches #756 — same rationale: a one-off
# noisy signal shouldn't shift the surfaced weight; sustained evidence
# should. The whole #480 family converges on this number so each
# sub-task is comparable.
META_BM25F_ANCHOR_WEIGHT_POSTERIOR_DECAY_SECONDS: Final[int] = 30 * 24 * 3600
# Default-OFF feature flag. Until the #437 A/B bench gate clears, every
# retrieval still uses the v1.5/#148 R3 static `DEFAULT_ANCHOR_WEIGHT=3`
# path through `BM25IndexCache`; setting this env var truthy switches
# `resolve_bm25f_anchor_weight_with_meta` to read the meta-belief from
# the store first.
ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT: Final[str] = (
    "AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT"
)
# ---------------------------------------------------------------------------
# #759 BFS depth-budget meta-belief consumer (sub-task E of umbrella #480)
# ---------------------------------------------------------------------------
# `expand_bfs` takes `int max_depth`. Bounds `[1, 6]` honour the
# single-hop floor (a depth-0 expansion is a no-op) and the latency
# safety ceiling documented in `docs/design/bfs_multihop.md`. BFS depth
# dominates p95 retrieval latency, so this is load-bearing for safety.
META_BFS_DEPTH_BUDGET_KEY: Final[str] = "meta:retrieval.bfs_depth_budget"
BFS_DEPTH_BUDGET_FLOOR: Final[int] = 1
BFS_DEPTH_BUDGET_CEIL: Final[int] = 6
# Static-default `value` for the meta-belief. Mid-range under the
# log-linear bounds: `decode_bfs_depth_budget(0.5)` ≈ sqrt(1*6) ≈ 2.45
# → rounds to 2. This is intentionally one hop below
# `BFS_DEFAULT_MAX_DEPTH` (2): shallow-only finds → posterior pulls
# budget down, consistent with the #759 spec rationale.
META_BFS_DEPTH_BUDGET_STATIC_DEFAULT: Final[float] = 0.5
# Sub-posterior decay — 30d, matching #756 and #757 so all #480 sub-
# tasks are comparable and a single one-off spike doesn't shift the
# surfaced depth.
META_BFS_DEPTH_BUDGET_POSTERIOR_DECAY_SECONDS: Final[int] = 30 * 24 * 3600
# Default-OFF feature flag. Ships behind the #437 A/B bench-gate
# clause, same as #756 and #757. Setting this env var truthy switches
# `resolve_bfs_depth_budget_with_meta` to read the meta-belief first.
ENV_META_BELIEF_BFS_DEPTH_BUDGET: Final[str] = (
    "AELFRICE_META_BELIEF_BFS_DEPTH_BUDGET"
)
# ---------------------------------------------------------------------------
# #760 expansion-gate token-threshold meta-belief consumer (sub-task F of #480)
# ---------------------------------------------------------------------------
# `should_run_expansion` tests `len(tokens) > threshold`. Bounds `[20, 320]`
# give a factor-of-4 band around the v1 default of 80; the geometric mean
# of 20 and 320 is exactly 80 (`sqrt(20*320) = sqrt(6400) = 80`), so the
# cold-start decode at `static_default=0.5` is byte-identical to the
# hardcoded `BROAD_PROMPT_TOKEN_THRESHOLD` value. The `relevance` signal
# (close-the-loop #779) is the sole subscribed class: an injected belief
# that is referenced by the assistant in a subsequent turn is evidence that
# the expansion-gate threshold should let more broad prompts through
# (higher threshold = fewer gates). The direction is correct because
# expansion gate outcomes directly feed retrieval quality, not latency.
EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR: Final[int] = 20
EXPANSION_GATE_TOKEN_THRESHOLD_CEIL: Final[int] = 320
META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY: Final[str] = (
    "meta:retrieval.expansion_gate.token_threshold"
)
# Static-default `value` for the meta-belief. decode(0.5) == 80 exactly
# (geometric mean of floor and ceil), matching `BROAD_PROMPT_TOKEN_THRESHOLD`
# so a cold-start install with the meta-belief flag on is byte-identical.
META_EXPANSION_GATE_TOKEN_THRESHOLD_STATIC_DEFAULT: Final[float] = 0.5
# Sub-posterior decay — 30d, matching #756/#757/#759 so all #480 sub-tasks
# are comparable on the same evidence time-scale.
META_EXPANSION_GATE_TOKEN_THRESHOLD_POSTERIOR_DECAY_SECONDS: Final[int] = (
    30 * 24 * 3600
)
# Default-OFF feature flag. Setting this env var truthy switches
# `resolve_expansion_gate_token_threshold_with_meta` to read the meta-belief
# first, replacing the hardcoded `BROAD_PROMPT_TOKEN_THRESHOLD` in
# `should_run_expansion`. Same default-OFF / bench-gate posture as
# #756/#757/#759.
ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD: Final[str] = (
    "AELFRICE_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD"
)
# ---------------------------------------------------------------------------
# #796 γ-rerank posterior-temperature meta-belief consumer
# ---------------------------------------------------------------------------
# Boltzmann temperature `T` on the posterior log term, consumed by
# `scoring.gamma_posterior_score`. Bounds `[0.5, 2.0]`; geometric mean
# is exactly 1.0, so the cold-start decode at `static_default=0.5` is
# `T = 1.0` — byte-identical to `partial_bayesian_score` with
# `posterior_weight = 1.0`. Adaptive learning of `T` (the #758 follow-
# up) is out of scope for #796: this issue ships the surface and the
# default-OFF flag, and the bench panel records γ vs log-additive
# under a hardcoded `T = 1.0`. The meta-belief substrate is installed
# here so the #758 wiring can drop in without a second config flip.
META_POSTERIOR_TEMPERATURE_KEY: Final[str] = (
    "meta:retrieval.posterior_temperature"
)
POSTERIOR_TEMPERATURE_FLOOR: Final[float] = 0.5
POSTERIOR_TEMPERATURE_CEIL: Final[float] = 2.0
# Mid-range so cold-start decodes to `T = 1.0` exactly.
META_POSTERIOR_TEMPERATURE_STATIC_DEFAULT: Final[float] = 0.5
# Sub-posterior decay — 30d, matching the rest of the #480 family.
META_POSTERIOR_TEMPERATURE_POSTERIOR_DECAY_SECONDS: Final[int] = 30 * 24 * 3600
# Default-OFF feature flag for #758 adaptive-learning delivery.
# Distinct from :data:`ENV_USE_GAMMA_POSTERIOR_TEMPERATURE` (#796): that
# flag gates whether the γ-rerank uses ``T`` at all; this flag gates
# whether the sweeper delivers relevance evidence to the meta-belief so
# ``T`` can learn. The two axes are independent — running with only the
# gamma flag on gives a fixed ``T=1.0`` cold-start; running with both on
# lets ``T`` adapt based on which top-K beliefs the assistant references.
ENV_META_BELIEF_POSTERIOR_TEMPERATURE: Final[str] = (
    "AELFRICE_META_BELIEF_POSTERIOR_TEMPERATURE"
)

# Number of decimal places used to round `posterior_weight` before
# inclusion in the cache key. Two callers passing weights that
# differ by less than this granularity collapse to the same key.
POSTERIOR_WEIGHT_KEY_PRECISION: Final[int] = 4
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

_CANONICALIZE_PUNCT: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")

# #677 retrieval-time literal boost for `#N` issue/PR references.
# Audit-log survey of 88 substantive prompts containing a literal
# `#NNN` token showed ~20% mean topical-match rate in the L1 BM25
# block: the BM25 tokenizer drops `#` so `#627` collides on the
# bare digit `627` with every other `#NNN` in the corpus. The boost
# compares prompt-extracted `#N` literals against belief content as
# a substring and adds `log(HASH_N_BOOST_MULTIPLIER)` to the final
# log score on a hit — log-additive in the same space
# `partial_bayesian_score` / `combine_log_scores` produce, so the
# shift is equivalent to multiplying the underlying BM25 relevance
# magnitude by the multiplier.
_HASH_N_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"#\d+")
HASH_N_BOOST_MULTIPLIER: Final[float] = 2.0
_HASH_N_BOOST_LOG: Final[float] = math.log(HASH_N_BOOST_MULTIPLIER)


def _extract_hash_n_literals(query: str) -> list[str]:
    """Return every `#N` literal in `query` (e.g. ``['#627', '#280']``).

    Empty list when none present — the caller treats empty as "no
    boost, keep the byte-identical FTS5 short-circuit".
    """
    return _HASH_N_LITERAL_RE.findall(query)


def _hash_n_boosted(score: float, content: str, literals: list[str]) -> float:
    """Add `log(HASH_N_BOOST_MULTIPLIER)` to `score` when `content`
    contains any literal from `literals`. No-op for empty `literals`.

    The check is literal `lit in content` rather than a tokenized
    match — the whole reason the boost exists is that the BM25
    tokenizer strips the leading `#`, which is what causes the
    disambiguation failure on plain `#NNN` queries (#677). The
    literals carry the `#` anchor.
    """
    if not literals:
        return score
    if any(lit in content for lit in literals):
        return score + _HASH_N_BOOST_LOG
    return score


def canonicalize_query(query: str) -> str:
    """Return a deterministic key for cache lookup.

    Lowercase, replace punctuation with whitespace, split on whitespace,
    sort tokens, rejoin with single spaces. Two queries that differ only
    in word order or punctuation map to the same key — correct for FTS5
    BM25, which is bag-of-words.
    """
    cleaned = _CANONICALIZE_PUNCT.sub(" ", query.lower()).strip()
    tokens = sorted(cleaned.split())
    return " ".join(tokens)


@dataclass(frozen=True)
class RetrievalResult:
    """Wrapper object for retrieve_v2 callers (academic-suite adapters).

    Public v1.0.x retrieve() returns list[Belief] directly. Lab v2.0.0
    adapters expect `result.beliefs` plus auxiliary diagnostics fields
    that aren't yet computed in public — those are placeholders here so
    adapter code that reads them does not crash.

    `entity_hits` (v1.3.0) exposes the L2.5 belief ids surfaced by the
    last call. The benchmark adapter consumes it for the L0/L1/L2.5
    counts surface; default `[]` for backwards-compat with adapters
    that only inspect `beliefs`.

    `doc_anchors` (#435) is a parallel list to `beliefs`: same length,
    same order. `doc_anchors[i]` lists every `belief_documents` row for
    `beliefs[i]`. Empty when the caller did not opt in via
    `with_doc_anchors=True`; also empty for beliefs that have no
    anchors.
    """

    beliefs: list[Belief]
    hrr_expansions: list[str] = field(default_factory=lambda: [])
    bfs_chains: list[list[str]] = field(default_factory=lambda: [])
    entity_hits: list[str] = field(default_factory=lambda: [])
    locked_ids: list[str] = field(default_factory=lambda: [])
    l1_ids: list[str] = field(default_factory=lambda: [])
    doc_anchors: list[list[DocAnchor]] = field(default_factory=lambda: [])
    # v2.1 #434 type-aware compression. Populated when
    # use_type_aware_compression resolves True. Same length and order as
    # `beliefs` (parallel field — consumers that want compressed render
    # read this; consumers that want raw Belief keep reading `beliefs`).
    # Default-empty preserves byte-identical v1.x adapter behavior when
    # the flag is OFF.
    compressed_beliefs: list[CompressedBelief] = field(default_factory=lambda: [])


def _estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate. Conservative (rounds up)."""
    if not text:
        return 0
    return int((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _belief_tokens(b: Belief) -> int:
    return _estimate_tokens(b.content)


# --- #1016-B layered locks: reference-tier manifest -------------------

# Cap on the manifest topic. A reference lock is surfaced as one line —
# `ref <id>: "<topic>"` — so injection stays ~constant regardless of the
# lock's full length; the agent reads full text on demand.
_LOCK_TOPIC_MAX: Final[int] = 80


def _lock_topic(content: str) -> str:
    """Deterministic one-line topic for a reference lock's manifest entry.

    Whitespace-collapsed; the first sentence if it ends within the cap,
    else a hard char-cap with an ellipsis. No ML — a pure string
    transform so the manifest is reproducible. Internal double-quotes are
    flattened to single so the `"<topic>"` wrapper stays unambiguous.
    """
    collapsed = " ".join(content.split()).replace('"', "'")
    if not collapsed:
        return ""
    for sep in (". ", "? ", "! "):
        idx = collapsed.find(sep)
        if 0 <= idx < _LOCK_TOPIC_MAX:
            return collapsed[: idx + 1].strip()
    if len(collapsed) <= _LOCK_TOPIC_MAX:
        return collapsed
    return collapsed[:_LOCK_TOPIC_MAX].rstrip() + "…"


def lock_manifest_line(b: Belief) -> str:
    """One-line manifest entry for a reference-tier lock (#1016-B)."""
    return f'ref {b.id}: "{_lock_topic(b.content)}"'


def is_reference_lock(b: Belief) -> bool:
    """True iff `b` is a user lock demoted to the bounded reference tier."""
    return b.lock_level == LOCK_USER and b.lock_tier == LOCK_TIER_REFERENCE


def lock_injection_tokens(b: Belief, *, manifest_reference_locks: bool) -> int:
    """Token cost of injecting a locked belief.

    When `manifest_reference_locks` is on, a reference lock costs only its
    one-line manifest entry (the #1016-B bound); otherwise — and for every
    frozen lock — it costs full content, identical to `_belief_tokens`. So
    the default (off) is byte-identical to pre-#1016 budgeting.
    """
    if manifest_reference_locks and is_reference_lock(b):
        return _estimate_tokens(lock_manifest_line(b))
    return _belief_tokens(b)


def effective_order_policy(
    policy: str, *, scores: dict[str, float] | None = None
) -> str:
    """The policy `order_for_injection` will actually apply (#1274).

    `score_desc` degrades to `lane` when the rerank scores are absent, and an
    unrecognised value is the identity. Both are correct behaviours, but they
    make *requested* and *applied* two different things, and anything that
    reports which arm produced a block has to report the applied one — a row
    labelled with an arm that did not run turns an inert instrument into what
    reads as a null result.

    This is the single source of truth for that rule: `order_for_injection`
    dispatches through it, and the hook audit records its output. Keeping one
    function means the two cannot drift into disagreeing about what happened.
    """
    if policy == ORDER_POLICY_LOCKS_LAST:
        return ORDER_POLICY_LOCKS_LAST
    if policy == ORDER_POLICY_SCORE_DESC and scores is not None:
        return ORDER_POLICY_SCORE_DESC
    return ORDER_POLICY_LANE


def order_for_injection(
    hits: list[Belief],
    policy: str,
    *,
    scores: dict[str, float] | None = None,
) -> list[Belief]:
    """Permute retrieved hits into their rendered order (#1274, #1177 p14).

    Position in the injected block is currently a side effect of lane
    concatenation (`locked + l25 + l1 + hrr + spine + bfs`), not a policy
    anyone chose. This makes it one, so the ordering question is answerable
    by a config flip instead of a rewrite.

    Policies:

    - `lane` — identity. The default, so the shipped block is byte-identical.
    - `locks_last` — non-locked hits first, the locked tier last.
    - `score_desc` — locked tier first, non-locked hits by descending
      `scores[belief.id]`.

    `score_desc` needs the rerank scores, which are not carried on `Belief`.
    When they are missing it degrades to the identity permutation **and says
    so on stderr** rather than silently substituting a proxy such as the
    posterior — a silent downgrade is exactly the failure #1271 documents,
    where an explicit setting was quietly replaced by a different one and the
    measurement it fed went unnoticed for a release.

    Because of that degradation the *requested* policy and the *applied* one
    can differ; :func:`effective_order_policy` is the shared rule for which
    is which, and anything recording what produced a block must record the
    applied one.

    Every policy is a stable, total permutation of `hits`: ties break on the
    original index, so the result is a pure function of (hits, policy,
    scores) and replay reproduces it exactly. No policy drops or adds a hit.
    """
    if policy == ORDER_POLICY_SCORE_DESC and scores is None:
        print(
            "aelfrice: order_policy=score_desc needs rerank scores; "
            "none supplied, falling back to lane order",
            file=sys.stderr,
        )

    applied = effective_order_policy(policy, scores=scores)

    if applied == ORDER_POLICY_LOCKS_LAST:
        locked = [b for b in hits if b.lock_level == LOCK_USER]
        rest = [b for b in hits if b.lock_level != LOCK_USER]
        return rest + locked

    if applied == ORDER_POLICY_SCORE_DESC:
        assert scores is not None  # guaranteed by `effective_order_policy`
        locked = [b for b in hits if b.lock_level == LOCK_USER]
        rest = [b for b in hits if b.lock_level != LOCK_USER]
        # `-score` with the original index as the tiebreak keeps the sort
        # total and stable; scores are log-domain and negative, so a missing
        # id must sort last, not first (-inf, not 0.0).
        ranked = sorted(
            enumerate(rest),
            key=lambda t: (-scores.get(t[1].id, -math.inf), t[0]),
        )
        return locked + [b for _, b in ranked]

    # `lane`, an unrecognised policy, and a degraded `score_desc` all land
    # here — identity, matching the resolver's fallback rather than raising
    # inside the render path.
    return list(hits)


# --- Config flag resolution ----------------------------------------------


def _env_disabled() -> bool:
    """Return True if AELFRICE_ENTITY_INDEX is set to a falsy value."""
    raw = os.environ.get(ENV_ENTITY_INDEX)
    if raw is None:
        return False
    return raw.strip().lower() in _ENV_FALSY


def _env_bfs_override() -> bool | None:
    """Return True/False if AELFRICE_BFS is set to a recognised
    truthy/falsy value, else None.

    Symmetric to `_env_disabled` but tri-state because the BFS flag
    ships default-OFF at v1.3.0 — an unset env var is "fall through
    to the next precedence layer", not "force off". The config-flag
    semantics for BFS are: env > kwarg > TOML > False.
    """
    raw = os.environ.get(ENV_BFS)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_bm25f_override() -> bool | None:
    """Return True/False if AELFRICE_BM25F is set to a recognised
    truthy/falsy value, else None. Symmetric to `_env_bfs_override`.
    """
    raw = os.environ.get(ENV_BM25F)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_heat_kernel_override() -> bool | None:
    """Return True/False if AELFRICE_HEAT_KERNEL is set to a recognised
    truthy/falsy value, else None. Symmetric to `_env_bm25f_override`.
    """
    raw = os.environ.get(ENV_HEAT_KERNEL)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_hrr_structural_override() -> bool | None:
    """Return True/False if AELFRICE_HRR_STRUCTURAL is set to a
    recognised truthy/falsy value, else None. Symmetric to
    `_env_bm25f_override`."""
    raw = os.environ.get(ENV_HRR_STRUCTURAL)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_hrr_expand_override() -> bool | None:
    """Return True/False if AELFRICE_HRR_EXPAND is set to a recognised
    truthy/falsy value, else None. Symmetric to
    `_env_hrr_structural_override`."""
    raw = os.environ.get(ENV_HRR_EXPAND)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_bm25f_per_field_override() -> bool | None:
    """Return True/False if AELFRICE_BM25F_PER_FIELD is set to a
    recognised truthy/falsy value, else None (#1180). Symmetric to
    `_env_hrr_expand_override`."""
    raw = os.environ.get(ENV_BM25F_PER_FIELD)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_entity_persist_demote_override() -> bool | None:
    """Return True/False if AELFRICE_ENTITY_PERSIST_DEMOTE is set to a
    recognised truthy/falsy value, else None (#1096). Symmetric to
    `_env_hrr_expand_override`."""
    raw = os.environ.get(ENV_ENTITY_PERSIST_DEMOTE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def is_entity_persist_demote_enabled(
    kwarg: bool | None = None, *, start: Path | None = None
) -> bool:
    """Resolve the entity-persistence demotion flag (#1096).

    Precedence (first decisive wins):
      1. AELFRICE_ENTITY_PERSIST_DEMOTE env var (truthy / falsy normalised).
      2. Explicit `kwarg` from the caller.
      3. `[retrieval] use_entity_persist_demote` in `.aelfrice.toml`.
      4. Default: **True** — the demotion lane flipped default-ON at v4.0
         once the #1096 G2 mixed-corpus eval (#1103) cleared the
         no-regression gate: durable recall held (20→20), ephemeral
         coordination demoted (20→3 at the tight pack budget), MRR
         0.883→1.000; recall-safe / inert on LoCoMo (all `noun_phrase`).
         Opt out via the env var, kwarg, or TOML key for parity with the
         pre-flip ranking.

    Live on the production path too: since the #1107 cutover `retrieve()`
    is a thin adapter over `retrieve_v2()` and passes this lane
    resolver-driven, so hook/rebuilder/MCP callers all observe it."""
    env = _env_entity_persist_demote_override()
    if env is not None:
        return env
    if kwarg is not None:
        return kwarg
    toml_value = _read_toml_flag_for(ENTITY_PERSIST_DEMOTE_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def _env_supersession_demote_override() -> bool | None:
    """Return True/False if AELFRICE_SUPERSESSION_DEMOTE is set to a
    recognised truthy/falsy value, else None (#1187). Symmetric to
    `_env_entity_persist_demote_override`."""
    raw = os.environ.get(ENV_SUPERSESSION_DEMOTE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def is_supersession_demote_enabled(
    kwarg: bool | None = None, *, start: Path | None = None
) -> bool:
    """Resolve the supersession lane flag (#1187).

    Precedence (first decisive wins):
      1. AELFRICE_SUPERSESSION_DEMOTE env var (truthy / falsy normalised).
      2. Explicit `kwarg` from the caller.
      3. `[retrieval] use_supersession_demote` in `.aelfrice.toml`.
      4. Default: **False**.

    The default stays False until the ratified three-arm bench (demote vs
    exclusion vs control) exists and the operator reads it. Unlike the
    #1170 BFS direction fix, this lane changes `retrieve()` output on the
    default path, which is precisely why the arms ship behind a flag
    instead of one of them shipping as the new behaviour.
    """
    env = _env_supersession_demote_override()
    if env is not None:
        return env
    if kwarg is not None:
        return kwarg
    toml_value = _read_toml_flag_for(SUPERSESSION_DEMOTE_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def resolve_supersession_treatment(
    kwarg: str | None = None, *, start: Path | None = None
) -> str:
    """Resolve which arm of the supersession lane runs (#1187).

    Same precedence as `is_supersession_demote_enabled`; default
    `"demote"`, the safer arm — a wrong demote leaves a ranking signal
    to notice, a wrong exclusion does not. Unrecognised values fall back
    to the default rather than raising, matching the tolerance the TOML
    readers already apply, and trace to stderr so a typo is visible.
    """
    candidates = (
        os.environ.get(ENV_SUPERSESSION_TREATMENT),
        kwarg,
        _read_toml_str_for(SUPERSESSION_TREATMENT_FLAG, start),
    )
    for raw in candidates:
        if raw is None:
            continue
        norm = raw.strip().lower()
        if norm in SUPERSESSION_TREATMENTS:
            return norm
        print(
            f"aelfrice retrieval: ignoring supersession treatment {raw!r} "
            f"(expected one of {', '.join(SUPERSESSION_TREATMENTS)})",
            file=sys.stderr,
        )
    return SUPERSESSION_TREATMENT_DEMOTE


def resolve_order_policy(
    kwarg: str | None = None, *, start: Path | None = None
) -> str:
    """Resolve the injection-block ordering policy (#1274).

    Same env -> kwarg -> TOML precedence as the other lane resolvers.
    Default `lane`, the identity permutation, so nothing about the rendered
    block changes until an experiment sets this deliberately. Unrecognised
    values fall back to the default and trace to stderr, matching
    `resolve_supersession_treatment`.
    """
    candidates = (
        os.environ.get(ENV_ORDER_POLICY),
        kwarg,
        _read_toml_str_for(ORDER_POLICY_FLAG, start),
    )
    for raw in candidates:
        if raw is None:
            continue
        norm = raw.strip().lower()
        if norm in ORDER_POLICIES:
            return norm
        print(
            f"aelfrice: ignoring unknown order_policy {raw!r} "
            f"(expected one of {', '.join(ORDER_POLICIES)})",
            file=sys.stderr,
        )
    return ORDER_POLICY_LANE


def resolve_supersession_factor(
    kwarg: float | None = None, *, start: Path | None = None
) -> float:
    """Resolve the demote arm's multiplicative factor (#1187).

    Same precedence; default `SUPERSESSION_DEMOTE_FACTOR` (0.5). Clamped
    to `(0, 1]`: above 1 would *promote* a retired belief, which is never
    the intent, and 0 is floored by `SUPERSESSION_FACTOR_EPS` so the log
    penalty stays finite. Out-of-range values clamp rather than raise.
    """
    raw: float | None = None
    env_raw = os.environ.get(ENV_SUPERSESSION_FACTOR)
    if env_raw is not None:
        try:
            raw = float(env_raw)
        except ValueError:
            print(
                f"aelfrice retrieval: ignoring {ENV_SUPERSESSION_FACTOR}"
                f"={env_raw!r} (expected a number)",
                file=sys.stderr,
            )
    if raw is None:
        raw = kwarg
    if raw is None:
        raw = _read_toml_float_for(SUPERSESSION_FACTOR_FLAG, start)
    if raw is None:
        raw = SUPERSESSION_DEMOTE_FACTOR
    return min(1.0, max(SUPERSESSION_FACTOR_EPS, float(raw)))


def _supersession_penalty(
    superseded: frozenset[str] | None, belief_id: str, factor: float
) -> float:
    """Log-additive demote for a superseded belief (#1187).

    Returns 0.0 when the lane is off (`superseded is None`) or this
    belief has not been retired by anything.

    **Additive `log(factor)`, not multiplicative `score * factor`.** The
    composite rerank score here is a log-domain quantity from
    `combine_log_scores` / `partial_bayesian_score` and is routinely
    negative — measured at ~-13 on a two-belief store. Multiplying a
    negative score by 0.5 *raises* it, so the multiplicative primitive in
    `uri_baki.apply_supersession_demote` (written against a non-negative
    score scale) would promote the superseded belief to the top of the
    pack: the exact inversion this lane exists to fix. Adding
    `log(factor)` is the log-domain equivalent of scaling a probability
    by `factor`, so the issue's "factor 0.5" semantics are preserved and
    the demote is unconditional. Same shape as
    `_entity_persist_penalty`, which is log-additive for the same reason.
    """
    if superseded is None or belief_id not in superseded:
        return 0.0
    return min(0.0, math.log(max(factor, SUPERSESSION_FACTOR_EPS)))


def _env_origin_tiebreak_override() -> bool | None:
    """Return True/False if AELFRICE_ORIGIN_TIEBREAK is set to a
    recognised truthy/falsy value, else None (#1089). Symmetric to
    `_env_entity_persist_demote_override`."""
    raw = os.environ.get(ENV_ORIGIN_TIEBREAK)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def is_origin_tiebreak_enabled(kwarg: bool | None = None) -> bool:
    """Resolve the #1089 axis-2 origin-priority tie-break flag.

    Precedence: AELFRICE_ORIGIN_TIEBREAK env var → explicit kwarg →
    default False. Mirrors the entity-persist resolution; default-OFF
    until the bench-gated flip."""
    env = _env_origin_tiebreak_override()
    if env is not None:
        return env
    if kwarg is not None:
        return kwarg
    return False


def _env_fan_effect_override() -> bool | None:
    """Return True/False if AELFRICE_FAN_EFFECT is set to a recognised
    truthy/falsy value, else None (#1176). Symmetric to
    `_env_origin_tiebreak_override`."""
    raw = os.environ.get(ENV_FAN_EFFECT)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def is_fan_effect_enabled(kwarg: bool | None = None) -> bool:
    """Resolve the #1176 proposal-3 ACT-R fan-effect lane flag.

    Precedence: AELFRICE_FAN_EFFECT env var -> explicit kwarg -> default
    False. Default-OFF: the kill gate cleared and the cost is lower than
    the lane it replaces, but *that the reorder ranks better* is what the
    A/B decides, and flipping the default is a separate operator call.
    """
    env = _env_fan_effect_override()
    if env is not None:
        return env
    if kwarg is not None:
        return kwarg
    return False


def _env_int_allowing_disable(env_name: str) -> int | None:
    """Read `env_name` as an int, keeping non-positive values (#1279).

    The sibling `_env_positive_int` discards `<= 0` as invalid, which is
    right where a non-positive value has no meaning. For the exploration
    cadence it has one — it disables the lane — so discarding it turns an
    explicit "off" into the default. Only a non-numeric value falls
    through here, and it traces like its sibling.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {env_name}={raw!r} (expected int)",
            file=sys.stderr,
        )
        return None


def _env_exploration_override() -> bool | None:
    """Return True/False if AELFRICE_EXPLORATION is set to a recognised
    truthy/falsy value, else None (#1279). Symmetric to
    `_env_fan_effect_override`."""
    raw = os.environ.get(ENV_EXPLORATION)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def is_exploration_enabled(
    kwarg: bool | None = None, *, start: Path | None = None
) -> bool:
    """Resolve the #1279 exploration-slot flag (#1176 proposal 5).

    Precedence: `AELFRICE_EXPLORATION` env var -> explicit kwarg ->
    `[retrieval] exploration_enabled` in `.aelfrice.toml` -> default
    **False**.

    Default-OFF because the slot changes what is injected into a live
    conversation. Its purpose is not ranking: 84.1% of the store has never
    been injected and therefore can never earn evidence, and the slot is the
    intervention that breaks that loop. Flipping the default is a separate
    operator call, gated on coverage growth measured from
    `exploration_events`, not on a relevance score.
    """
    env = _env_exploration_override()
    if env is not None:
        return env
    if kwarg is not None:
        return kwarg
    toml_value = _read_toml_flag_for(EXPLORATION_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def resolve_exploration_cadence(
    kwarg: int | None = None, *, start: Path | None = None
) -> int:
    """Turns between exploration fires (#1279).

    Defaults to `exploration.DEFAULT_EXPLORATION_CADENCE`, named rather than
    restated so this cannot drift from it again — it said 20 after the
    constant moved to 3.

    `<= 0` disables exploration rather than raising — `should_explore`
    guards the modulus, so a misconfigured cadence degrades to "never
    explore" instead of a `ZeroDivisionError` inside a retrieval.

    That contract has to hold on the **env tier too**, which is why this
    does not use `_env_positive_int`: that helper rejects `<= 0` and
    returns None, so `AELFRICE_EXPLORATION_CADENCE=0` fell through to the
    default and an operator asking for "off" silently got a cadence of 20
    — the opposite of the request, on the highest-precedence tier. The
    shared helper is right for the resolvers where a non-positive value is
    merely invalid (`l1_limit`); here it is meaningful.
    """
    env = _env_int_allowing_disable(ENV_EXPLORATION_CADENCE)
    if env is not None:
        return env
    if kwarg is not None:
        return int(kwarg)
    toml_value = _read_toml_float_for(EXPLORATION_CADENCE_FLAG, start)
    if toml_value is not None:
        return int(toml_value)
    return DEFAULT_EXPLORATION_CADENCE


def resolve_exploration_slots(
    kwarg: int | None = None, *, start: Path | None = None
) -> int:
    """Pack slots given to exploration on a firing turn (#1279). Default 1.

    Slots are *substituted*, never appended — see
    `hook._substitute_exploration_slots`. A slot that grew the block would
    be a budget increase wearing an exploration costume, and would confound
    the coverage measurement the slot exists to produce.
    """
    env = _env_positive_int(ENV_EXPLORATION_SLOTS)
    if env is not None:
        return env
    if kwarg is not None:
        return int(kwarg)
    toml_value = _read_toml_float_for(EXPLORATION_SLOTS_FLAG, start)
    if toml_value is not None:
        return int(toml_value)
    return DEFAULT_EXPLORATION_SLOTS


def _env_temporal_spine_override() -> bool | None:
    """Return True/False if AELFRICE_TEMPORAL_SPINE is set to a
    recognised truthy/falsy value, else None. Symmetric to
    `_env_hrr_expand_override`."""
    raw = os.environ.get(ENV_TEMPORAL_SPINE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_type_aware_compression_override() -> bool | None:
    """Return True/False if AELFRICE_TYPE_AWARE_COMPRESSION is set to a
    recognised truthy/falsy value, else None. Symmetric to
    `_env_bm25f_override`."""
    raw = os.environ.get(ENV_TYPE_AWARE_COMPRESSION)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_intentional_clustering_override() -> bool | None:
    """Return True/False if AELFRICE_INTENTIONAL_CLUSTERING is set to a
    recognised truthy/falsy value, else None. Symmetric to
    `_env_bm25f_override`."""
    raw = os.environ.get(ENV_INTENTIONAL_CLUSTERING)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_use_gamma_posterior_temperature_override() -> bool | None:
    """Return True/False if AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE is
    set to a recognised truthy/falsy value, else None. Symmetric to
    `_env_type_aware_compression_override`."""
    raw = os.environ.get(ENV_USE_GAMMA_POSTERIOR_TEMPERATURE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_use_zeta_posterior_rerank_override() -> bool | None:
    """Return True/False if AELFRICE_USE_ZETA_POSTERIOR_RERANK is set to a
    recognised truthy/falsy value, else None. Symmetric to
    `_env_use_gamma_posterior_temperature_override`."""
    raw = os.environ.get(ENV_USE_ZETA_POSTERIOR_RERANK)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _env_hrr_persist_override() -> bool | None:
    """Return True/False if AELFRICE_HRR_PERSIST is set to a recognised
    truthy/falsy value, else None. Symmetric to `_env_bm25f_override`.

    "0" → False (disable); "1" → True (force on). Unset or unrecognised
    values return None so the next precedence rung (TOML → default) wins.
    """
    raw = os.environ.get(ENV_HRR_PERSIST)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


# #1135: memoized `.aelfrice.toml` parse. ~24 resolver call sites fall
# through to the TOML rung per retrieve(), and each used to re-read +
# re-parse the file. The directory walk (a handful of stat calls)
# still runs per call — so a config file created, deleted, or moved
# mid-process is honoured — but the read + tomllib parse is cached per
# file until its (mtime_ns, size) changes. Value None records a
# malformed / unreadable / section-less parse, so the error path is
# cached too (and its stderr trace prints once per file version
# instead of once per resolver call).
_TOML_SECTION_CACHE: dict[str, tuple[int, int, dict[str, Any] | None]] = {}

# Discovery memo for `_discover_config`, scoped to one retrieval call.
#
# The *parse* above is memoized on the file, but the *walk* that finds the
# file was not, so every `[retrieval]` resolver re-walked from cwd to the
# first `.aelfrice.toml`. With ~22 resolvers that is O(flags x path depth)
# filesystem work per `retrieve()` — 22 walks and 173 `posix.stat` calls
# per retrieval, measured on a 5-belief store (#1289) — and it grew with
# every lane flag added.
#
# Staleness semantics, stated rather than left implicit: the memo lives for
# the duration of a single retrieval and is discarded at the end of it, so a
# `.aelfrice.toml` created or deleted between two retrievals is picked up by
# the next one exactly as before. Only a change made *during* one retrieval
# is missed, which no caller can observe. This is deliberately narrower than
# a process-lifetime cache, which would have made a config file created
# mid-process invisible until restart.
#
# A ContextVar rather than a plain dict so concurrent retrievals in threads
# or async tasks cannot see each other's memo.
_CONFIG_DISCOVERY_MEMO: ContextVar[dict[Path, Path | None] | None] = ContextVar(
    "aelfrice_config_discovery_memo",
    default=None,
)

# Memo key standing for "the caller passed no `start`", i.e. resolve from
# cwd. Not a real path, and cannot collide with one: every other key is an
# absolute resolved directory.
_CWD_KEY: Final[Path] = Path("\x00cwd")


@contextmanager
def config_discovery_scope() -> Iterator[None]:
    """Memoize `.aelfrice.toml` discovery for the duration of the block.

    Entering is what turns the memo on; outside a scope every resolver
    walks, preserving the original behaviour for direct callers. Nesting is
    safe — an inner scope reuses the outer memo rather than shadowing it, so
    `retrieve` calling `retrieve_v2` does not re-walk.
    """
    if _CONFIG_DISCOVERY_MEMO.get() is not None:
        yield
        return
    token = _CONFIG_DISCOVERY_MEMO.set({})
    try:
        yield
    finally:
        _CONFIG_DISCOVERY_MEMO.reset(token)


def _memoize_config_discovery(fn: Any) -> Any:
    """Run `fn` inside a `config_discovery_scope`.

    Applied to the retrieval entry points so the ~22 `[retrieval]` resolvers
    reached during one call share a single `.aelfrice.toml` walk. Nesting is
    handled by the scope itself, so an entry point calling another one costs
    nothing extra.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with config_discovery_scope():
            return fn(*args, **kwargs)

    return wrapper


def _discover_config(start: Path | None) -> Path | None:
    """Return the nearest `.aelfrice.toml` at or above `start`, else None.

    This is the walk every `[retrieval]` resolver used to do for itself.
    Inside a `config_discovery_scope` the result is memoized per resolved
    start directory, so N resolvers cost one walk instead of N.
    """
    memo = _CONFIG_DISCOVERY_MEMO.get()
    if memo is not None and start is None and _CWD_KEY in memo:
        # Resolve the key before `Path.cwd().resolve()`, which is itself a
        # syscall pair — the default `start=None` is what every one of the
        # ~22 resolvers passes, so it is the case worth short-circuiting.
        return memo[_CWD_KEY]
    base = (start if start is not None else Path.cwd()).resolve()
    if memo is not None and base in memo:
        return memo[base]
    located: Path | None = None
    current = base
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            located = candidate
            break
        if current.parent == current:
            break
        current = current.parent
    if memo is not None:
        memo[base] = located
        if start is None:
            memo[_CWD_KEY] = located
    return located


def _parsed_retrieval_section(candidate: Path) -> dict[str, Any] | None:
    """Return `candidate`'s `[retrieval]` table, or None when the file
    is unreadable, malformed, or has no such table. Memoized on the
    file's (mtime_ns, size); tolerant — never raises."""
    serr: IO[str] = sys.stderr
    try:
        st = candidate.stat()
    except OSError as exc:
        print(
            f"aelfrice retrieval: cannot read {candidate}: {exc}",
            file=serr,
        )
        return None
    cache_key = str(candidate)
    hit = _TOML_SECTION_CACHE.get(cache_key)
    if (
        hit is not None
        and hit[0] == st.st_mtime_ns
        and hit[1] == st.st_size
    ):
        return hit[2]
    section: dict[str, Any] | None = None
    try:
        raw = candidate.read_bytes()
        parsed: dict[str, Any] = tomllib.loads(
            raw.decode("utf-8", errors="replace"),
        )
        section_obj: Any = parsed.get(RETRIEVAL_SECTION, {})
        if isinstance(section_obj, dict):
            section = section_obj
    except OSError as exc:
        print(
            f"aelfrice retrieval: cannot read {candidate}: {exc}",
            file=serr,
        )
    except tomllib.TOMLDecodeError as exc:
        print(
            f"aelfrice retrieval: malformed TOML in {candidate}: {exc}",
            file=serr,
        )
    _TOML_SECTION_CACHE[cache_key] = (st.st_mtime_ns, st.st_size, section)
    return section


def _read_toml_flag_for(
    key: str,
    start: Path | None = None,
) -> bool | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] <key>`. Returns the boolean value when found, or
    None when no file / no key.

    Tolerant: a malformed TOML or wrong-typed value returns None
    (let the default win) and traces to stderr without raising.
    Mirrors `noise_filter.NoiseConfig.discover` semantics.
    """
    serr: IO[str] = sys.stderr
    candidate = _discover_config(start)
    if candidate is None:
        return None
    section = _parsed_retrieval_section(candidate)
    if section is None or key not in section:
        return None
    value: Any = section[key]
    if isinstance(value, bool):
        return value
    print(
        f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
        f"{key} in {candidate} (expected bool)",
        file=serr,
    )
    return None


def _read_toml_str_for(
    key: str,
    start: Path | None = None,
) -> str | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] <key>` typed as a string (#1187). Returns the value when
    found, or None when no file / no key.

    Same tolerance as `_read_toml_flag_for`: a wrong-typed value traces to
    stderr and returns None so the caller's default wins.
    """
    serr: IO[str] = sys.stderr
    candidate = _discover_config(start)
    if candidate is None:
        return None
    section = _parsed_retrieval_section(candidate)
    if section is None or key not in section:
        return None
    value: Any = section[key]
    if isinstance(value, str):
        return value
    print(
        f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
        f"{key} in {candidate} (expected str)",
        file=serr,
    )
    return None


def _read_toml_float_for(
    key: str,
    start: Path | None = None,
) -> float | None:
    """Walk up from `start` looking for a `.aelfrice.toml` with
    `[retrieval] <key>` typed as int or float. Returns the float
    value when found, or None when no file / no key.

    Tolerant: a malformed TOML or wrong-typed value returns None
    and traces to stderr without raising. Mirrors
    `_read_toml_flag_for` semantics but accepts numeric types.
    """
    serr: IO[str] = sys.stderr
    candidate = _discover_config(start)
    if candidate is None:
        return None
    section = _parsed_retrieval_section(candidate)
    if section is None or key not in section:
        return None
    value: Any = section[key]
    # bool is a subclass of int -- reject it explicitly so
    # `posterior_weight = true` reads as malformed rather
    # than silently coercing to 1.0.
    if isinstance(value, bool):
        print(
            f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
            f"{key} in {candidate} (expected number, got bool)",
            file=serr,
        )
        return None
    if isinstance(value, (int, float)):
        return float(value)
    print(
        f"aelfrice retrieval: ignoring [{RETRIEVAL_SECTION}] "
        f"{key} in {candidate} (expected number)",
        file=serr,
    )
    return None


def _env_posterior_weight() -> float | None:
    """Return the AELFRICE_POSTERIOR_WEIGHT env value as a float,
    or None when unset / non-numeric.

    Non-numeric values trace to stderr and fall through (same
    fail-soft contract as the TOML readers).
    """
    raw = os.environ.get(ENV_POSTERIOR_WEIGHT)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {ENV_POSTERIOR_WEIGHT}={raw!r} "
            f"(expected float)",
            file=sys.stderr,
        )
        return None


def resolve_posterior_weight(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the posterior weight per v1.3 precedence:

      1. AELFRICE_POSTERIOR_WEIGHT env var (float, including 0.0).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] posterior_weight` in `.aelfrice.toml`.
      4. Default: DEFAULT_POSTERIOR_WEIGHT (0.5 at v1.3.0).

    A weight of `0.0` is treated as "BM25-only" (the byte-identical-
    with-v1.0.x ordering case); negative weights are clamped to
    0.0 since the spec defines the contract for weight ≥ 0 only.
    """
    env = _env_posterior_weight()
    if env is not None:
        weight = env
    elif explicit is not None:
        weight = float(explicit)
    else:
        toml_value = _read_toml_float_for(POSTERIOR_WEIGHT_FLAG, start)
        weight = float(toml_value) if toml_value is not None else (
            DEFAULT_POSTERIOR_WEIGHT
        )
    if weight < 0.0:
        return 0.0
    return weight


def _env_bm25_k3() -> float | None:
    """Return `AELFRICE_BM25_K3` as a float, or None when unset /
    non-numeric. Same fail-soft contract as `_env_posterior_weight`.
    """
    raw = os.environ.get(ENV_BM25_K3)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {ENV_BM25_K3}={raw!r} "
            f"(expected float)",
            file=sys.stderr,
        )
        return None


def resolve_bm25_k3(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the BM25F query-term-frequency saturation constant (#1166).

    Precedence (first decisive wins):
      1. ``AELFRICE_BM25_K3`` env var (float, including 0.0).
      2. Explicit `explicit` kwarg from the caller.
      3. ``[retrieval] bm25_k3`` in `.aelfrice.toml`.
      4. Default: ``bm25.DEFAULT_K3`` (0.0).

    ``k3 = 0`` weights every query term by its idf alone regardless of
    how many times it appears, which is exactly what the pre-#1166 code
    did by assigning rather than accumulating. It stays the default so
    this bug fix does not move anyone's ranking: three shipped
    components express their boost as a duplicated token
    (`query_understanding.entity_expand`, `query_understanding.idf_clip`,
    `hook._build_conversation_aware_query`) and have been inert on this
    lane, but their multipliers were tuned against the FTS5 lane and do
    not transfer unexamined. Raising `k3` is therefore a separate,
    bench-gated flip.

    Negative values clamp to 0.0 — the saturation form is defined for
    ``k3 >= 0`` only, and `BM25Index.build` rejects negatives outright.
    """
    from aelfrice.bm25 import DEFAULT_K3

    env = _env_bm25_k3()
    if env is not None:
        k3 = env
    elif explicit is not None:
        k3 = float(explicit)
    else:
        toml_value = _read_toml_float_for(BM25_K3_FLAG, start)
        k3 = float(toml_value) if toml_value is not None else DEFAULT_K3
    if k3 < 0.0:
        return 0.0
    return k3


def resolve_bm25f_per_field(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the per-field BM25F flag (#1180).

    Precedence (first decisive wins):
      1. ``AELFRICE_BM25F_PER_FIELD`` env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. ``[retrieval] bm25f_per_field`` in `.aelfrice.toml`.
      4. Default: **False**.

    Off by default because the field split replaces the scoring
    functional form rather than re-parameterising it: the saturation
    denominator becomes the constant `k1` instead of `tf + k1*B`, so no
    choice of constants makes flag-on and flag-off agree once an anchor
    stream exists. There is therefore no parity test that could gate the
    flip — only a bench.

    What the flip changes, measured on a synthetic two-belief corpus
    (identical content, one belief carrying 200 tokens of incoming
    anchor text, queried by a term the two share):

    - anchor text that never mentions the query term: the cited belief
      scores **0.45x** the uncited one today, and **1.00x** under
      per-field. Today's demotion is the defect — a belief is punished
      for what its citers wrote about, not for anything about itself.
    - anchor text that does mention it: today's boost tops out near
      1.23x; per-field reaches ~1.96x.

    That second figure is the reason for the bench gate rather than a
    straight flip: per-field roughly doubles the weight real anchor
    evidence carries, and `anchor_weight`'s shipped value of 3 was tuned
    as a replication count against the old form, not as a field weight
    against this one.

    Passing the flag (any rung) never raises — an unset / unrecognised
    value falls through to the next rung and ultimately to False.
    """
    env = _env_bm25f_per_field_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(BM25F_PER_FIELD_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def _env_bm25_b_anchor() -> float | None:
    """Return `AELFRICE_BM25_B_ANCHOR` as a float, or None when unset /
    non-numeric. Same fail-soft contract as `_env_bm25_k3`."""
    raw = os.environ.get(ENV_BM25_B_ANCHOR)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {ENV_BM25_B_ANCHOR}={raw!r} "
            f"(expected float)",
            file=sys.stderr,
        )
        return None


def resolve_bm25_b_anchor(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the anchor stream's length-normalisation strength (#1180).

    Precedence (first decisive wins):
      1. ``AELFRICE_BM25_B_ANCHOR`` env var (float, including 0.0).
      2. Explicit `explicit` kwarg from the caller.
      3. ``[retrieval] bm25_b_anchor`` in `.aelfrice.toml`.
      4. Default: ``bm25.DEFAULT_B_ANCHOR`` (equal to the content `b`).

    Only consulted when `resolve_bm25f_per_field` is on; the legacy
    single-field path has no second `b` to set.

    Negative values clamp to 0.0 — `B_f(d) = (1-b) + b*dl/avgdl` is
    defined for ``b >= 0`` only, and `BM25Index.build` rejects negatives
    outright. Note that 0.0 is a *permitted* setting that disables
    length normalisation on the anchor stream, which lets the anchor
    contribution grow linearly with citation volume; it is exposed for
    ablation, not recommended.
    """
    from aelfrice.bm25 import DEFAULT_B_ANCHOR

    env = _env_bm25_b_anchor()
    if env is not None:
        b_anchor = env
    elif explicit is not None:
        b_anchor = float(explicit)
    else:
        toml_value = _read_toml_float_for(BM25_B_ANCHOR_FLAG, start)
        b_anchor = (
            float(toml_value) if toml_value is not None else DEFAULT_B_ANCHOR
        )
    if b_anchor < 0.0:
        return 0.0
    return b_anchor


# --- #1045 wide-retrieval knobs (l1_limit + retrieval token budget) ------
# The BM25 candidate cap (`l1_limit`) is the multi-hop RECALL lever: raising
# it — WITH a token budget large enough to hold the extra candidates —
# recovers multi-session / temporal answers a 50-candidate slice misses
# (LongMemEval-S 58.8% → 68.6% at l1=200 / budget=8000). Budget alone is
# inert: candidates cap at `l1_limit` BEFORE the pack trim. Both stay at
# their latency-sensitive hot-path defaults (50 / 2400) unless a caller
# opts in via kwarg, env, or `.aelfrice.toml`. See #1045.
ENV_L1_LIMIT: Final[str] = "AELFRICE_L1_LIMIT"
L1_LIMIT_FLAG: Final[str] = "l1_limit"
ENV_RETRIEVAL_TOKEN_BUDGET: Final[str] = "AELFRICE_RETRIEVAL_TOKEN_BUDGET"
TOKEN_BUDGET_FLAG: Final[str] = "token_budget"


def _env_positive_int(env_name: str) -> int | None:
    """Return `env_name` as a positive int, or None when unset / invalid.

    Non-numeric and non-positive values trace to stderr and fall through
    (same fail-soft contract as `_env_posterior_weight`).
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = int(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {env_name}={raw!r} (expected int)",
            file=sys.stderr,
        )
        return None
    if value <= 0:
        print(
            f"aelfrice retrieval: ignoring {env_name}={raw!r} (must be > 0)",
            file=sys.stderr,
        )
        return None
    return value


def resolve_l1_limit(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the L1 (BM25) candidate cap per the standard precedence:

      1. ``AELFRICE_L1_LIMIT`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[retrieval] l1_limit`` in ``.aelfrice.toml``.
      4. Default: ``DEFAULT_L1_LIMIT`` (50).

    Mirrors `resolve_posterior_weight`. The default keeps the
    latency-sensitive per-prompt hook narrow; opting into a larger cap is
    the multi-hop recall lever (#1045) and only takes effect together with
    a ``token_budget`` large enough to hold the extra candidates.
    """
    env = _env_positive_int(ENV_L1_LIMIT)
    if env is not None:
        return env
    if explicit is not None:
        return int(explicit)
    toml_value = _read_toml_float_for(L1_LIMIT_FLAG, start)
    return int(toml_value) if toml_value is not None else DEFAULT_L1_LIMIT


def resolve_token_budget(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the retrieval token budget per the standard precedence:

      1. ``AELFRICE_RETRIEVAL_TOKEN_BUDGET`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[retrieval] token_budget`` in ``.aelfrice.toml``.
      4. Default: ``DEFAULT_TOKEN_BUDGET`` (2400).

    Companion to `resolve_l1_limit`: raising ``l1_limit`` only helps when
    the budget is large enough to keep the extra candidates past the pack
    trim (#1045).
    """
    return resolve_token_budget_with_provenance(explicit, start=start)[0]


def resolve_token_budget_with_provenance(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> tuple[int, bool]:
    """Resolve the budget and report whether any tier actually set it.

    Returns ``(budget, defaulted)``. ``defaulted`` is True only when env,
    the explicit kwarg and TOML were all silent, so the value is the
    built-in default rather than anyone's choice.

    The provenance exists here and used to be discarded one frame down
    (#1271). `retrieve_with_tiers` needs "the caller expressed no
    preference" to decide the legacy-budget downgrade, and was
    reconstructing it as ``budget == DEFAULT_TOKEN_BUDGET`` — a different
    predicate that agrees most of the time and is wrong for exactly the
    caller who asks for the default value on purpose. Returning the fact
    is cheaper than inferring it, and cannot drift from the precedence
    above because it is computed by the same branches.
    """
    env = _env_positive_int(ENV_RETRIEVAL_TOKEN_BUDGET)
    if env is not None:
        return env, False
    if explicit is not None:
        return int(explicit), False
    toml_value = _read_toml_float_for(TOKEN_BUDGET_FLAG, start)
    if toml_value is not None:
        return int(toml_value), False
    return DEFAULT_TOKEN_BUDGET, True


def _env_temporal_half_life() -> float | None:
    """Return AELFRICE_TEMPORAL_HALF_LIFE_SECONDS as a positive float,
    or None when unset / non-numeric / non-positive.

    Non-numeric and non-positive values trace to stderr and fall through
    (same fail-soft contract as `_env_posterior_weight`). A half-life of
    zero or negative is structurally meaningless for an exponential
    decay, so we reject rather than clamp.
    """
    raw = os.environ.get(ENV_TEMPORAL_HALF_LIFE)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        print(
            f"aelfrice retrieval: ignoring {ENV_TEMPORAL_HALF_LIFE}={raw!r} "
            f"(expected float)",
            file=sys.stderr,
        )
        return None
    if value <= 0.0:
        print(
            f"aelfrice retrieval: ignoring {ENV_TEMPORAL_HALF_LIFE}={raw!r} "
            f"(must be > 0)",
            file=sys.stderr,
        )
        return None
    return value


def decode_meta_half_life(value: float) -> float:
    """Decode a `[0, 1]` meta-belief value into seconds via log-linear
    interpolation between :data:`HALF_LIFE_FLOOR_SECONDS` (3 days) and
    :data:`HALF_LIFE_CEIL_SECONDS` (14 days).

    `v=0.0` → 3d, `v=1.0` → 14d, `v=0.5` → ~6.5d (close to the #473
    ratified 7d static). Values outside `[0, 1]` are clamped — the
    substrate's `posterior_mean` is mathematically bounded to `[0, 1]`
    but `value` may be the `static_default` cold-start fallback on a
    misconfigured row, so we defend.

    Per the #756 operator ratification (2026-05-13): the encoding
    lives in the consumer, not in the substrate. This keeps the
    substrate pattern-uniform across umbrella #480 sub-tasks B–F.
    """
    v = max(0.0, min(1.0, value))
    ln_floor = math.log(HALF_LIFE_FLOOR_SECONDS)
    ln_ceil = math.log(HALF_LIFE_CEIL_SECONDS)
    return math.exp(ln_floor + v * (ln_ceil - ln_floor))


def is_meta_belief_half_life_enabled() -> bool:
    """Return True iff :data:`ENV_META_BELIEF_HALF_LIFE` is set to a
    recognised truthy value.

    Ships default-OFF per the #756 bench-gate clause: until #437 A/B
    corpus evidence clears, every retrieval still uses the v2.1 #473
    static-7d path. Operators flip this on per-shell to opt into the
    adaptive half-life. The issue's `=enabled` spelling is honoured
    alongside the codebase-standard truthy tokens (`1`, `true`, `yes`,
    `on`) — both forms map to the same state, no precedence wrinkles.
    """
    raw = os.environ.get(ENV_META_BELIEF_HALF_LIFE)
    if raw is None:
        return False
    norm = raw.strip().lower()
    return norm in _ENV_TRUTHY or norm == "enabled"


def decode_meta_bm25f_anchor_weight(value: float) -> int:
    """Decode a `[0, 1]` meta-belief value into an integer anchor_weight
    via log-linear interpolation between
    :data:`BM25F_ANCHOR_WEIGHT_FLOOR` (1) and
    :data:`BM25F_ANCHOR_WEIGHT_CEIL` (10).

    `v=0.0` → 1, `v=1.0` → 10, `v=0.5` → 3 (rounded from ~3.16, matching
    `DEFAULT_ANCHOR_WEIGHT` exactly so the cold-start install preserves
    byte-identical retrieval order until the meta-belief actually moves).

    Values outside `[0, 1]` are clamped — the substrate's
    `posterior_mean` is mathematically bounded to `[0, 1]` but `value`
    may be the `static_default` cold-start fallback on a misconfigured
    row, so we defend. The result is rounded to the nearest int because
    `BM25Index.build` accepts only integer anchor_weight (it replicates
    the anchor token stream `anchor_weight` times — see bm25.py:243).

    Per the same #756 (2026-05-13) ratification: encoding lives in the
    consumer, substrate stays pattern-uniform across #480 B–F.
    """
    v = max(0.0, min(1.0, value))
    ln_floor = math.log(BM25F_ANCHOR_WEIGHT_FLOOR)
    ln_ceil = math.log(BM25F_ANCHOR_WEIGHT_CEIL)
    return round(math.exp(ln_floor + v * (ln_ceil - ln_floor)))


def is_meta_belief_bm25f_anchor_weight_enabled() -> bool:
    """Return True iff :data:`ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT` is set
    to a recognised truthy value.

    Ships default-OFF per the #757 bench-gate clause: until #437 A/B
    corpus evidence clears, BM25F still uses `DEFAULT_ANCHOR_WEIGHT = 3`
    from bm25.py. Operators flip this on per-shell to opt into the
    adaptive anchor-weight. The `=enabled` spelling is honoured
    alongside the codebase-standard truthy tokens (`1`, `true`, `yes`,
    `on`), mirroring :func:`is_meta_belief_half_life_enabled`.
    """
    raw = os.environ.get(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT)
    if raw is None:
        return False
    norm = raw.strip().lower()
    return norm in _ENV_TRUTHY or norm == "enabled"


def decode_bfs_depth_budget(value: float) -> int:
    """Decode a `[0, 1]` meta-belief value into an integer BFS max-depth
    via log-linear interpolation between
    :data:`BFS_DEPTH_BUDGET_FLOOR` (1) and
    :data:`BFS_DEPTH_BUDGET_CEIL` (6).

    `v=0.0` → 1, `v=1.0` → 6, `v=0.5` → 2 (rounded from ~2.45,
    matching :data:`BFS_DEFAULT_MAX_DEPTH` exactly, so a cold-start
    install with the meta-belief on is byte-identical to the
    hardcoded default until evidence accrues).

    Values outside `[0, 1]` are clamped — the substrate's
    `posterior_mean` is mathematically bounded to `[0, 1]` but
    ``value`` may be the static_default fallback on a misconfigured
    row. The result is rounded to the nearest int because
    ``expand_bfs`` accepts only ``int max_depth``.

    Per the 2026-05-13 #756 ratification: encoding lives in the
    consumer, substrate stays pattern-uniform across #480 B–F.
    """
    v = max(0.0, min(1.0, value))
    ln_floor = math.log(BFS_DEPTH_BUDGET_FLOOR)
    ln_ceil = math.log(BFS_DEPTH_BUDGET_CEIL)
    return int(round(math.exp(ln_floor + v * (ln_ceil - ln_floor))))


def is_meta_belief_bfs_depth_budget_enabled() -> bool:
    """Return True iff :data:`ENV_META_BELIEF_BFS_DEPTH_BUDGET` is set
    to a recognised truthy value.

    Ships default-OFF per the #759 bench-gate clause: until #437 A/B
    corpus evidence clears, BFS still uses :data:`BFS_DEFAULT_MAX_DEPTH`
    from ``bfs_multihop``. Operators flip this on per-shell to opt into
    the adaptive depth budget. The ``=enabled`` spelling is honoured
    alongside the codebase-standard truthy tokens (``1``, ``true``,
    ``yes``, ``on``), mirroring :func:`is_meta_belief_half_life_enabled`.
    """
    raw = os.environ.get(ENV_META_BELIEF_BFS_DEPTH_BUDGET)
    if raw is None:
        return False
    norm = raw.strip().lower()
    return norm in _ENV_TRUTHY or norm == "enabled"


def decode_expansion_gate_token_threshold(value: float) -> int:
    """Decode a `[0, 1]` meta-belief value into an integer token threshold
    via log-linear interpolation between
    :data:`EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR` (20) and
    :data:`EXPANSION_GATE_TOKEN_THRESHOLD_CEIL` (320).

    `v=0.0` → 20, `v=1.0` → 320, `v=0.5` → 80 exactly.

    The `v=0.5` decode equals 80 because the geometric mean of 20 and
    320 is ``sqrt(20 * 320) = sqrt(6400) = 80`` — a precise integer.
    Cold-start with the meta-belief on is therefore byte-identical to
    the pre-#760 hardcoded :data:`aelfrice.expansion_gate.BROAD_PROMPT_TOKEN_THRESHOLD`.

    Values outside `[0, 1]` are clamped — the substrate's
    ``posterior_mean`` is mathematically bounded to `[0, 1]` but
    ``value`` may be the static_default fallback on a misconfigured
    row. The result is rounded to the nearest int because
    ``should_run_expansion`` compares ``len(tokens) > threshold``
    against a whole number.

    Per the 2026-05-13 #756 ratification: encoding lives in the
    consumer, substrate stays pattern-uniform across #480 B–F.
    """
    v = max(0.0, min(1.0, value))
    ln_floor = math.log(EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR)
    ln_ceil = math.log(EXPANSION_GATE_TOKEN_THRESHOLD_CEIL)
    return int(round(math.exp(ln_floor + v * (ln_ceil - ln_floor))))


def is_meta_belief_expansion_gate_token_threshold_enabled() -> bool:
    """Return True iff :data:`ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD`
    is set to a recognised truthy value.

    Ships default-OFF per the #760 bench-gate clause: until #437 A/B
    corpus evidence clears, ``should_run_expansion`` still uses the
    hardcoded :data:`aelfrice.expansion_gate.BROAD_PROMPT_TOKEN_THRESHOLD`
    (80). Operators flip this on per-shell to opt into the adaptive
    token threshold. The ``=enabled`` spelling is honoured alongside the
    codebase-standard truthy tokens (``1``, ``true``, ``yes``, ``on``),
    mirroring :func:`is_meta_belief_half_life_enabled`.
    """
    raw = os.environ.get(ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD)
    if raw is None:
        return False
    norm = raw.strip().lower()
    return norm in _ENV_TRUTHY or norm == "enabled"

def is_meta_belief_posterior_temperature_enabled() -> bool:
    """Return True iff :data:`ENV_META_BELIEF_POSTERIOR_TEMPERATURE`
    is set to a recognised truthy value.

    Ships default-OFF per the #758 adaptive-delivery gate: until relevance
    evidence accumulates in a bench corpus, the gamma-rerank temperature stays
    at its cold-start value of T = 1.0. Operators flip this on per-shell
    to opt into the adaptive learning signal. The ``=enabled`` spelling is
    honoured alongside the codebase-standard truthy tokens (``1``, ``true``,
    ``yes``, ``on``), mirroring :func:`is_meta_belief_half_life_enabled`.

    Note: this flag controls only the sweeper delivery path. The gamma-rerank
    itself is separately gated by :func:`resolve_use_gamma_posterior_temperature`
    and :data:`ENV_USE_GAMMA_POSTERIOR_TEMPERATURE`.
    """
    raw = os.environ.get(ENV_META_BELIEF_POSTERIOR_TEMPERATURE)
    if raw is None:
        return False
    norm = raw.strip().lower()
    return norm in _ENV_TRUTHY or norm == "enabled"


def get_active_meta_belief_consumers() -> list[str]:
    """Return the canonical-sorted list of meta-belief keys whose
    retrieval consumer is currently env-gated ON.

    Used by the #779 UPS-hook write-path to populate
    ``injection_events.active_consumers`` per turn. The sweeper later
    iterates this list when scoring `referenced` evidence so each
    enabled consumer's `relevance` sub-posterior gets updated.

    Sort order is alphabetical so a determinism-replay test that pins
    env state sees the same column-bytes across runs. #756 half-life,
    #757 bm25f_anchor_weight, #758 posterior_temperature, #760
    expansion_gate subscribe to signals covered by the sweeper. #759
    bfs_depth_budget uses latency-only and is not swept for relevance.
    """
    active: list[str] = []
    if is_meta_belief_half_life_enabled():
        active.append(META_HALF_LIFE_KEY)
    if is_meta_belief_bm25f_anchor_weight_enabled():
        active.append(META_BM25F_ANCHOR_WEIGHT_KEY)
    if is_meta_belief_expansion_gate_token_threshold_enabled():
        active.append(META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY)
    if is_meta_belief_posterior_temperature_enabled():
        active.append(META_POSTERIOR_TEMPERATURE_KEY)
    return sorted(active)


def resolve_temporal_half_life(
    explicit: float | None = None,
    *,
    start: Path | None = None,
) -> float:
    """Resolve the temporal-decay half-life (seconds) per v2.1 #473:

      1. AELFRICE_TEMPORAL_HALF_LIFE_SECONDS env var (positive float).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] temporal_half_life_seconds` in `.aelfrice.toml`.
      4. Default: DEFAULT_TEMPORAL_HALF_LIFE_SECONDS (7 days).

    Non-positive values at any layer fall through to the next layer.
    The decay is `2 ** (-age_seconds / half_life)` so half_life=0 is
    undefined; treat it as missing.

    The meta-belief consumer (#756) lives in a separate function
    (:func:`resolve_temporal_half_life_with_meta`) because it requires
    a store handle and a `now_ts`. Bare retrieve-time consumers without
    a store still resolve through this static-config chain.
    """
    env = _env_temporal_half_life()
    if env is not None:
        return env
    if explicit is not None and explicit > 0.0:
        return float(explicit)
    toml_value = _read_toml_float_for(TEMPORAL_HALF_LIFE_FLAG, start)
    if toml_value is not None and toml_value > 0.0:
        return float(toml_value)
    return DEFAULT_TEMPORAL_HALF_LIFE_SECONDS


def install_temporal_half_life_meta_belief(
    store: MemoryStore,
    *,
    now_ts: int,
) -> bool:
    """Idempotent install of the #756 meta-belief on ``store``.

    Returns True on first install, False if the row already exists.
    Mirrors :func:`store.MemoryStore.install_meta_belief`'s idempotency
    contract — existing rows are not overwritten because the surfaced
    value would silently shift under retrieval consumers.

    The install signature pins the v3.x ratified defaults: latency
    signal only (relevance deferred to #779 per D4), 30d posterior
    decay (slower than the surfaced half-life's own [3d, 14d] band),
    cold-start ``value`` ≈ 6.5d via :func:`decode_meta_half_life` on
    the 0.5 static default.
    """
    from aelfrice.meta_beliefs import SIGNAL_LATENCY
    return store.install_meta_belief(
        META_HALF_LIFE_KEY,
        static_default=META_HALF_LIFE_STATIC_DEFAULT,
        half_life_seconds=META_HALF_LIFE_POSTERIOR_DECAY_SECONDS,
        signal_weights={SIGNAL_LATENCY: 1.0},
        now_ts=now_ts,
    )


def resolve_temporal_half_life_with_meta(
    store: MemoryStore | None,
    *,
    now_ts: int,
    explicit: float | None = None,
    start: Path | None = None,
) -> float:
    """Meta-aware variant of :func:`resolve_temporal_half_life`.

    Precedence (first decisive wins):
      1. AELFRICE_TEMPORAL_HALF_LIFE_SECONDS env var (positive float).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[retrieval] temporal_half_life_seconds`` in `.aelfrice.toml`.
      4. **Meta-belief** (#756) — only when :data:`ENV_META_BELIEF_HALF_LIFE`
         resolves truthy AND ``store`` has the meta-belief installed.
         Decodes the substrate's `[0, 1]` value through
         :func:`decode_meta_half_life` into the `[3d, 14d]` band.
      5. Default: :data:`DEFAULT_TEMPORAL_HALF_LIFE_SECONDS` (7 days).

    The meta-belief precedence sits *below* the explicit-config layers
    on purpose: an operator setting the env var or TOML key is an
    intentional override that should bypass the adaptive layer.
    Without that ordering, the meta-belief would compete with explicit
    operator intent. ``None`` ``store`` collapses to the static
    :func:`resolve_temporal_half_life` chain unchanged.
    """
    env = _env_temporal_half_life()
    if env is not None:
        return env
    if explicit is not None and explicit > 0.0:
        return float(explicit)
    toml_value = _read_toml_float_for(TEMPORAL_HALF_LIFE_FLAG, start)
    if toml_value is not None and toml_value > 0.0:
        return float(toml_value)
    if store is not None and is_meta_belief_half_life_enabled():
        meta_value = store.read_meta_belief_value(
            META_HALF_LIFE_KEY, now_ts=now_ts,
        )
        if meta_value is not None:
            return decode_meta_half_life(meta_value)
    return DEFAULT_TEMPORAL_HALF_LIFE_SECONDS


def install_bm25f_anchor_weight_meta_belief(
    store: MemoryStore,
    *,
    now_ts: int,
) -> bool:
    """Idempotent install of the #757 meta-belief on ``store``.

    Returns True on first install, False if the row already exists.
    Mirrors :func:`install_temporal_half_life_meta_belief`'s contract —
    existing rows are not overwritten because the surfaced anchor_weight
    would silently shift under BM25F.

    The install signature pins the v3.x ratified defaults: bm25_l0_ratio
    signal only (relevance deferred to #779 per the same D4 split that
    #756 followed), 30d posterior decay, cold-start ``value`` = 0.5
    which decodes through :func:`decode_meta_bm25f_anchor_weight` to 3,
    matching ``bm25.DEFAULT_ANCHOR_WEIGHT`` exactly.
    """
    from aelfrice.meta_beliefs import SIGNAL_BM25_L0_RATIO
    return store.install_meta_belief(
        META_BM25F_ANCHOR_WEIGHT_KEY,
        static_default=META_BM25F_ANCHOR_WEIGHT_STATIC_DEFAULT,
        half_life_seconds=META_BM25F_ANCHOR_WEIGHT_POSTERIOR_DECAY_SECONDS,
        signal_weights={SIGNAL_BM25_L0_RATIO: 1.0},
        now_ts=now_ts,
    )


def resolve_bm25f_anchor_weight_with_meta(
    store: MemoryStore | None,
    *,
    now_ts: int,
    explicit: int | None = None,
) -> int:
    """Resolve the BM25F anchor_weight knob with meta-belief consultation.

    Precedence (first decisive wins):
      1. Explicit ``explicit`` kwarg from the caller (lets the bench
         harness and tests pin a value).
      2. **Meta-belief** (#757) — only when
         :data:`ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT` resolves truthy AND
         ``store`` has the meta-belief installed. Decodes the substrate's
         `[0, 1]` value through :func:`decode_meta_bm25f_anchor_weight`
         into the `[1, 10]` band.
      3. Default: ``bm25.DEFAULT_ANCHOR_WEIGHT`` (3).

    Unlike #756's resolver, this one has no env-var or TOML override
    layer — anchor_weight has never had a user-facing config knob (it
    has been a code constant since v1.5/#148 R3). Explicit-kwarg stays
    the only operator-side override path so the meta-belief surface
    doesn't introduce a new public config that we'd have to keep
    stable. ``None`` ``store`` collapses to the static default.
    """
    if explicit is not None:
        return int(explicit)
    if store is not None and is_meta_belief_bm25f_anchor_weight_enabled():
        meta_value = store.read_meta_belief_value(
            META_BM25F_ANCHOR_WEIGHT_KEY, now_ts=now_ts,
        )
        if meta_value is not None:
            return decode_meta_bm25f_anchor_weight(meta_value)
    from aelfrice.bm25 import DEFAULT_ANCHOR_WEIGHT
    return DEFAULT_ANCHOR_WEIGHT


def install_bfs_depth_budget_meta_belief(
    store: MemoryStore,
    *,
    now_ts: int,
) -> bool:
    """Idempotent install of the #759 meta-belief on ``store``.

    Returns True on first install, False if the row already exists.
    Mirrors :func:`install_temporal_half_life_meta_belief`'s contract —
    existing rows are not overwritten because the surfaced depth budget
    would silently shift under BFS.

    The install signature pins the v3.x ratified defaults: latency
    signal only (`bfs_depth` signal deferred to #779 per the same D4
    split that #756 and #757 followed), 30d posterior decay, cold-start
    ``value`` = 0.5 which decodes to 2 via :func:`decode_bfs_depth_budget`.
    """
    from aelfrice.meta_beliefs import SIGNAL_LATENCY
    return store.install_meta_belief(
        META_BFS_DEPTH_BUDGET_KEY,
        static_default=META_BFS_DEPTH_BUDGET_STATIC_DEFAULT,
        half_life_seconds=META_BFS_DEPTH_BUDGET_POSTERIOR_DECAY_SECONDS,
        signal_weights={SIGNAL_LATENCY: 1.0},
        now_ts=now_ts,
    )


def resolve_bfs_depth_budget_with_meta(
    store: MemoryStore | None,
    *,
    now_ts: int,
    explicit: int | None = None,
    start: None = None,  # reserved for future TOML layer; unused by #759 MVP
) -> int:
    """Resolve the BFS max-depth knob with meta-belief consultation.

    Returns an ``int`` because ``expand_bfs`` takes ``int max_depth``.

    Precedence (first decisive wins):
      1. Explicit ``explicit`` kwarg from the caller (positive int) —
         for the bench harness override path.
      2. **Meta-belief** (#759) — only when
         :data:`ENV_META_BELIEF_BFS_DEPTH_BUDGET` resolves truthy AND
         ``store`` has the meta-belief installed. Decodes the substrate's
         `[0, 1]` value through :func:`decode_bfs_depth_budget` into
         the `[1, 6]` band, rounded to int.
      3. Default: :data:`BFS_DEFAULT_MAX_DEPTH` from
         ``aelfrice.bfs_multihop``.

    No env-var or TOML override layer — bfs_max_depth has never had a
    user-facing config knob, so we do not synthesize one. Operators
    override via explicit kwarg (bench) or meta-belief (production).
    ``None`` ``store`` collapses to the static default.

    The ``explicit`` clause is so a caller explicitly passing the
    :data:`BFS_DEFAULT_MAX_DEPTH` default doesn't disable the
    meta-belief; only a non-default explicit override bypasses the
    adaptive layer. Callers that do not override should pass
    ``explicit=bfs_max_depth if bfs_max_depth != BFS_DEFAULT_MAX_DEPTH
    else None`` so the default falls through to the meta-belief.
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    if store is not None and is_meta_belief_bfs_depth_budget_enabled():
        meta_value = store.read_meta_belief_value(
            META_BFS_DEPTH_BUDGET_KEY, now_ts=now_ts,
        )
        if meta_value is not None:
            return decode_bfs_depth_budget(meta_value)
    return BFS_DEFAULT_MAX_DEPTH


def install_expansion_gate_token_threshold_meta_belief(
    store: MemoryStore,
    *,
    now_ts: int,
) -> bool:
    """Idempotent install of the #760 meta-belief on ``store``.

    Returns True on first install, False if the row already exists.
    Mirrors :func:`install_bfs_depth_budget_meta_belief`'s contract —
    existing rows are not overwritten because the surfaced threshold
    would silently shift under the expansion gate.

    The install signature pins the v3.x ratified defaults: relevance
    signal only (the close-the-loop #779 layer is the right signal for
    expansion quality — a referenced injected belief is evidence that
    expansion was useful, so raising the threshold gate is warranted),
    30d posterior decay, cold-start ``value`` = 0.5 which decodes to 80
    via :func:`decode_expansion_gate_token_threshold`, matching
    :data:`aelfrice.expansion_gate.BROAD_PROMPT_TOKEN_THRESHOLD` exactly.
    """
    from aelfrice.meta_beliefs import SIGNAL_RELEVANCE
    return store.install_meta_belief(
        META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY,
        static_default=META_EXPANSION_GATE_TOKEN_THRESHOLD_STATIC_DEFAULT,
        half_life_seconds=META_EXPANSION_GATE_TOKEN_THRESHOLD_POSTERIOR_DECAY_SECONDS,
        signal_weights={SIGNAL_RELEVANCE: 1.0},
        now_ts=now_ts,
    )


def resolve_expansion_gate_token_threshold_with_meta(
    store: MemoryStore | None,
    *,
    now_ts: int,
    explicit: int | None = None,
) -> int:
    """Resolve the expansion-gate token-threshold knob with meta-belief
    consultation.

    Returns an ``int`` because ``should_run_expansion`` compares
    ``len(tokens) > threshold`` against a whole number.

    Precedence (first decisive wins):
      1. Explicit ``explicit`` kwarg from the caller (positive int) —
         for test and bench harness overrides.
      2. **Meta-belief** (#760) — only when
         :data:`ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD` resolves
         truthy AND ``store`` has the meta-belief installed. Decodes the
         substrate's `[0, 1]` value through
         :func:`decode_expansion_gate_token_threshold` into the `[20, 320]`
         band, rounded to int.
      3. Default: :data:`aelfrice.expansion_gate.BROAD_PROMPT_TOKEN_THRESHOLD`
         (80).

    No env-var or TOML override layer — the expansion-gate token threshold
    has never had a user-facing config knob outside the gate itself, so we
    do not synthesize one. Operators override via explicit kwarg or the
    meta-belief. ``None`` ``store`` collapses to the static default.
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    if store is not None and is_meta_belief_expansion_gate_token_threshold_enabled():
        meta_value = store.read_meta_belief_value(
            META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY, now_ts=now_ts,
        )
        if meta_value is not None:
            return decode_expansion_gate_token_threshold(meta_value)
    from aelfrice.expansion_gate import BROAD_PROMPT_TOKEN_THRESHOLD
    return BROAD_PROMPT_TOKEN_THRESHOLD


def _belief_age_seconds(b: Belief, now: datetime) -> float:
    """Seconds between `now` and `b.created_at` (clamped at 0).

    `created_at` is an ISO-8601 string (`datetime.now(timezone.utc)
    .isoformat()` per `store.insert_belief`). Malformed timestamps fall
    through as age=0 — no decay penalty rather than a hard crash. This
    matches the rest of `retrieval.py`'s fail-soft posture on user data.
    """
    raw = b.created_at
    if not raw:
        return 0.0
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = (now - ts).total_seconds()
    return delta if delta > 0.0 else 0.0


def _apply_temporal_decay(
    beliefs: list[Belief],
    half_life_seconds: float,
    *,
    now: datetime | None = None,
) -> list[Belief]:
    """Re-rank `beliefs` by an exponential recency decay.

    Locked beliefs (lock_level != LOCK_NONE) are pinned at the head of
    the output in their original relative order — L0 is user-asserted
    ground truth and is never re-ordered by recency.

    The remaining beliefs are scored by `(1 / (rank + 1)) * 2 ** (-age
    / half_life)`, where `rank` is the belief's pre-decay position in
    `beliefs` and `age` is the seconds since `created_at`. Sort is
    stable on the proxy score: ties keep the upstream pipeline's order.

    The proxy `1 / (rank + 1)` is a borderline design call (issue #473
    closing comment): `retrieve_v2` does not surface per-belief scores
    out of `retrieve_with_tiers`, so the wrapper has only the merged
    order to work from. Treating rank-position as the proxy score keeps
    the decay multiplicative on a meaningful baseline (1.0 at the head,
    diminishing) while leaving the upstream pipeline as the score
    authority.
    """
    if not beliefs or half_life_seconds <= 0.0:
        return list(beliefs)
    when = now if now is not None else datetime.now(timezone.utc)
    locked: list[Belief] = []
    rest: list[tuple[int, Belief]] = []
    for i, b in enumerate(beliefs):
        if b.lock_level != LOCK_NONE:
            locked.append(b)
        else:
            rest.append((i, b))
    if not rest:
        return list(beliefs)

    def keyfn(item: tuple[int, Belief]) -> float:
        i, b = item
        rank_score = 1.0 / float(i + 1)
        age = _belief_age_seconds(b, when)
        decay = 2.0 ** (-age / half_life_seconds)
        return rank_score * decay

    rest_sorted = sorted(rest, key=keyfn, reverse=True)
    return locked + [b for _, b in rest_sorted]


def is_entity_index_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the entity-index flag.

    Precedence (first decisive wins):
      1. AELFRICE_ENTITY_INDEX=0 (env override).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] entity_index_enabled` in `.aelfrice.toml`.
      4. Default: True (v1.3.0 default-on).
    """
    if _env_disabled():
        return False
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(ENTITY_INDEX_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def resolve_use_bm25f_anchors(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the BM25F (anchor-augmented sparse matvec) flag.

    Precedence (first decisive wins):
      1. AELFRICE_BM25F env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_bm25f_anchors` in `.aelfrice.toml`.
      4. Default: True (v1.7.0 default-ON per #154 bench evidence).

    The composition-tracker (#154) bench gate ran on the
    `tests/corpus/v2_0/retrieve_uplift/v0_1.jsonl` lab fixture and
    measured **+0.6650 NDCG@k uplift** for `use_bm25f_anchors=True`
    versus the all-flags-off baseline (30 rows, 6 categories) under
    Porter stemming. No regression on any row. See #154 for the
    per-flag table; the stemming addition (#428) closed the
    `q="banana"` vs content `"bananas"` gap that briefly blocked
    the flip.

    Callers that need the v1.5/v1.6 FTS5 path can still set
    `AELFRICE_BM25F=0`, pass `use_bm25f_anchors=False`, or write
    `[retrieval] use_bm25f_anchors = false` in `.aelfrice.toml`.
    """
    env = _env_bm25f_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(BM25F_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def is_hrr_structural_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the HRR structural-query lane flag (#152).

    Precedence (first decisive wins):
      1. AELFRICE_HRR_STRUCTURAL env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_hrr_structural` in `.aelfrice.toml`.
      4. Default: True — the structural lane is on by default. The
         composition tracker (#154) flipped the default after the
         #437 reproducibility-harness gate cleared at 11/11. Opt out
         via the env var, kwarg, or TOML key for parity with the
         pre-flip ranking.

    Reuses `HRR_STRUCTURAL_FLAG` (the placeholder constant from
    #232). Now that the lane has shipped, the flag is no longer in
    `PLACEHOLDER_FLAGS` so `warn_placeholder_flags()` does not flag
    it as unwired.
    """
    env = _env_hrr_structural_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(HRR_STRUCTURAL_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def is_hrr_expand_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the HRR vocabulary-bridge expansion-lane flag (#981).

    Precedence (first decisive wins):
      1. AELFRICE_HRR_EXPAND env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_hrr_expand` in `.aelfrice.toml`.
      4. Default: **False** — the expansion lane is OFF by default. #981
         lands the lane + its ablation arm only; flipping the default
         reverses the locked #605 determinism philosophy and is routed to a
         re-opened #897. Opt in for the ablation via the env var, kwarg, or
         TOML key.

    Passing the flag (any rung) never raises — an unset / unrecognised value
    falls through to the next rung and ultimately to False (AC1).
    """
    env = _env_hrr_expand_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(HRR_EXPAND_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def is_temporal_spine_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the temporal-spine retrieval-lane flag (#1064).

    Precedence (first decisive wins):
      1. AELFRICE_TEMPORAL_SPINE env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_temporal_spine` in `.aelfrice.toml`.
      4. Default: **True** — the lane is default-ON since the #1064 lane
         flip (v4.0, #1107 Phase 2). Every pre-registered gate cleared
         (G1 +14.6pp LoCoMo coverage, G2 trim survival + top-rank
         invariance, G3 latency delta_p95 in-band, G5 determinism); the
         production `retrieve()` hook path exposes it via the #1107 shim.
         Opt out with `AELFRICE_TEMPORAL_SPINE=0` or `[retrieval]
         use_temporal_spine = false`. Distinct from
         `AELFRICE_TEMPORAL_SPINE_WRITE` (the ingest-time writer flag in
         `aelfrice.temporal_spine`) — they resolve independently.

    Passing the flag (any rung) never raises — an unset / unrecognised
    value falls through to the next rung and ultimately to the default (True).
    """
    env = _env_temporal_spine_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(TEMPORAL_SPINE_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def resolve_temporal_spine_budget(
    explicit: int | None = None,
    *,
    start: Path | None = None,
) -> int:
    """Resolve the temporal-spine lane's node budget (#1064).

      1. ``AELFRICE_TEMPORAL_SPINE_BUDGET`` env var (positive int).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[retrieval] temporal_spine_budget`` in ``.aelfrice.toml``.
      4. Default: ``temporal_spine.DEFAULT_SPINE_NODE_BUDGET`` (32).

    The confirmatory budget curve is monotone (~+2.5pp coverage per
    doubling at 32/64/128 with no plateau) — the effect is
    budget-limited, so this is the knob the flip release revisits.
    """
    from aelfrice.temporal_spine import DEFAULT_SPINE_NODE_BUDGET

    env = _env_positive_int(ENV_TEMPORAL_SPINE_BUDGET)
    if env is not None:
        return env
    if explicit is not None:
        return int(explicit)
    toml_value = _read_toml_float_for(TEMPORAL_SPINE_BUDGET_FLAG, start)
    return (
        int(toml_value) if toml_value is not None
        else DEFAULT_SPINE_NODE_BUDGET
    )


def is_hrr_persist_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the HRR structural-index persistence flag (#698).

    Precedence (first decisive wins):
      1. AELFRICE_HRR_PERSIST env var ("0" disables; "1" forces on).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] hrr_persist` in `.aelfrice.toml`.
      4. Default: True — persistence is on by default. Set
         `[retrieval] hrr_persist = false` or `AELFRICE_HRR_PERSIST=0`
         to disable. In-memory stores (`store_path=None`) are never
         persisted regardless of this flag.
    """
    env = _env_hrr_persist_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(HRR_PERSIST_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def make_hrr_struct_cache(
    store: MemoryStore,
    *,
    store_path: str | None = None,
    dim: int = DEFAULT_DIM,
    seed: int | None = None,
    start: Path | None = None,
) -> HRRStructIndexCache:
    """Construct an :class:`HRRStructIndexCache` with persistence wired
    to the resolved :func:`is_hrr_persist_enabled` flag (#698).

    This is the canonical construction site for long-running callers
    (interactive shells, bench harnesses) that want config-driven
    persistence behaviour without manually resolving the flag.
    The ``persist_enabled`` field on the returned cache reflects the
    env → TOML → default precedence chain so callers do not need to
    import or call :func:`is_hrr_persist_enabled` directly.
    """
    persist = is_hrr_persist_enabled(start=start)
    return HRRStructIndexCache(
        store=store,
        dim=dim,
        store_path=store_path,
        seed=seed,
        persist_enabled=persist,
    )


def _route_structural_query(
    store: MemoryStore,
    query: str,
    cache: HRRStructIndexCache | None,
    *,
    top_k: int,
    include_locked: bool,
    budget: int,
) -> RetrievalResult | None:
    """Probe the HRR structural lane and pack results to budget.

    Returns ``None`` when the query is not a structural marker, or
    when the marker resolves to an unknown ``(kind, target)`` pair on
    the index. The caller must fall through to the textual lane in
    both cases — the structural lane is parallel, never blended.

    On hit, locks (when ``include_locked=True``) are pinned at the
    head of the result and bypass the budget per the existing public-
    API contract; HRR-ranked beliefs are appended in score-descending
    order until the budget is exhausted. Beliefs already present
    among the locks are de-duped from the HRR tail so the locked
    pin-to-head invariant is preserved.
    """
    parsed = parse_structural_marker(query)
    if parsed is None:
        return None
    kind, target_id = parsed
    idx: HRRStructIndex
    if cache is None:
        idx = HRRStructIndex()
        idx.build(store)
    else:
        idx = cache.get()
    hits = idx.probe(kind, target_id, top_k=top_k)
    if not hits:
        # Marker parsed but the (kind, target) pair is unknown to the
        # index (no edges of that type touch target_id). Fall through
        # so the caller can try the textual lane on the literal
        # marker string — better than returning an empty result.
        return None

    locked: list[Belief] = (
        list(store.list_locked_beliefs()) if include_locked else []
    )
    locked_ids: set[str] = {b.id for b in locked}
    used: int = sum(_belief_tokens(b) for b in locked)
    out: list[Belief] = list(locked)

    for belief_id, _score in hits:
        if belief_id in locked_ids:
            continue
        belief = store.get_belief(belief_id)
        if belief is None:
            continue
        cost = _belief_tokens(belief)
        if used + cost > budget:
            break
        out.append(belief)
        used += cost

    return RetrievalResult(
        beliefs=out,
        locked_ids=[b.id for b in locked],
    )


def resolve_use_type_aware_compression(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the type-aware compression flag (#434).

    Precedence (first decisive wins):
      1. AELFRICE_TYPE_AWARE_COMPRESSION env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_type_aware_compression` in `.aelfrice.toml`.
      4. Default: True — flipped from False after the A2 + A4 bench
         gates (docs/design/feature-type-aware-compression.md) cleared
         on the lab-side compression_a* corpora (#769: A2 mean recall@k
         uplift +0.3267 on n=25 with zero per-row regressions; A4 mean
         fidelity delta +0.0085 on n=15). Compose-compatibility with
         `use_intentional_clustering` shipped in #878 (pack_with_clusters
         cost_fn seam); both flags resolving True is supported.
    """
    env = _env_type_aware_compression_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(TYPE_AWARE_COMPRESSION_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def is_max_coverage_pack_enabled(
    kwarg: bool | None = None, *, start: Path | None = None
) -> bool:
    """Resolve the max-coverage pack-selector flag (#1176 proposal 2).

    Precedence (first decisive wins):
      1. ``AELFRICE_MAX_COVERAGE_PACK`` env var (truthy / falsy normalised).
      2. Explicit `kwarg` from the caller.
      3. ``[retrieval] use_max_coverage_pack`` in `.aelfrice.toml`.
      4. Default: **False**.

    Off until the three-arm A/B (cluster pack vs rank-greedy vs coverage)
    is read by the operator. This changes which beliefs reach the agent on
    the default path, which is exactly why it ships behind a flag rather
    than as the new behaviour.

    Takes precedence over `use_intentional_clustering` when both are on --
    they are two answers to the same question and running both would mean
    packing twice.
    """
    raw = os.environ.get(ENV_MAX_COVERAGE_PACK)
    if raw is not None and raw.strip():
        norm = raw.strip().lower()
        if norm in ("1", "true", "yes", "on"):
            return True
        if norm in ("0", "false", "no", "off"):
            return False
    if kwarg is not None:
        return kwarg
    toml_value = _read_toml_flag_for(MAX_COVERAGE_PACK_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def _coverage_inputs(
    query: str,
    candidates: list[Belief],
    bm25f_cache: "BM25IndexCache | None",
) -> tuple[dict[str, frozenset[str]], dict[str, float]]:
    """Build `(coverage, term_weights)` for `pack_max_coverage` (#1176).

    Coverage is the query's stems intersected with each belief's stems,
    computed with the same `tokenize_stemmed` the BM25 lane indexes with
    so a term matches here exactly when it matched there.

    Weights are `idf` from the built BM25F index when one is available.
    Without an index every term weighs 1.0, which degrades the objective
    to plain term-count coverage rather than to nothing -- the FTS5 lane
    still gets redundancy suppression, just unweighted by rarity.

    **Assumes the pack renders `b.content` verbatim.** Coverage is
    computed from `content`, but the pack emits
    `compress_for_retrieval(b).rendered`, and `use_type_aware_compression`
    defaults on. For a retention class that does not render verbatim --
    `snapshot` (headline) or `transient` (stub) -- the objective can
    credit a term the agent never receives, and worse, mark it covered
    and suppress a later belief that would have delivered it. This is the
    seam #878 closed for the *cost* currency; `cost_fn=_cost` inherits
    that fix, coverage has no counterpart. Latent, not live: over 5,150
    replayed L1 candidates none rendered non-verbatim, because the corpus
    is 86.0% `fact` and 13.9% `unknown` against 0.09% `snapshot` and zero
    `transient`. Whoever raises that share, or turns on a more aggressive
    compression strategy, has to reconcile coverage with the rendered
    text here.
    """
    from aelfrice.bm25 import tokenize_stemmed

    q_stems = set(tokenize_stemmed(query))
    if not q_stems:
        return {}, {}
    coverage = {
        b.id: frozenset(q_stems & set(tokenize_stemmed(b.content)))
        for b in candidates
    }
    weights: dict[str, float] = {t: 1.0 for t in q_stems}
    if bm25f_cache is not None:
        try:
            index = bm25f_cache.get()
            for t in q_stems:
                j = index.vocabulary.get(t)
                if j is not None:
                    weights[t] = float(index.idf[j])
        except Exception:  # noqa: BLE001 - diagnostic weighting only
            # An unavailable index is not a reason to fail retrieval; the
            # uniform weights above remain a valid coverage objective.
            pass
    return coverage, weights


def resolve_use_intentional_clustering(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the intentional-clustering flag (#436).

    Precedence (first decisive wins):
      1. AELFRICE_INTENTIONAL_CLUSTERING env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_intentional_clustering` in `.aelfrice.toml`.
      4. Default: True — flipped from False after the A4 latency bench
         gate cleared on the multi-store production sweep (#436 R6, 60/60
         PASS at p99 0.328ms ~ 15-30x margin under the 5ms budget). See
         docs/design/feature-intentional-clustering.md A2 + A4.
    """
    env = _env_intentional_clustering_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(INTENTIONAL_CLUSTERING_FLAG, start)
    if toml_value is not None:
        return toml_value
    return True


def resolve_use_gamma_posterior_temperature(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the γ rerank flag (#796).

    Precedence (first decisive wins):
      1. AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE env var.
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_gamma_posterior_temperature` in `.aelfrice.toml`.
      4. Default: False — ships behind the flag at v3.x. Adoption verdict
         (flip default) is deferred until a labeled relevance corpus
         exists, per `docs/feature-posterior-temperature.md` §
         "Bench-gate / ship-or-defer policy".
    """
    env = _env_use_gamma_posterior_temperature_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(
        USE_GAMMA_POSTERIOR_TEMPERATURE_FLAG, start,
    )
    if toml_value is not None:
        return toml_value
    return False


def install_posterior_temperature_meta_belief(
    store: MemoryStore,
    *,
    now_ts: int,
) -> bool:
    """Idempotent install of the #758 meta-belief on ``store``.

    Returns True on first install, False if the row already exists.
    Mirrors :func:`install_expansion_gate_token_threshold_meta_belief`'s
    contract — existing rows are not overwritten because the surfaced
    temperature would silently shift under the gamma-rerank consumer.

    Ships with relevance signal only. Single-signal subscription is
    intentional: temperature changes the rerank distribution shape; the
    only natural feedback is whether the top-K beliefs surfaced by the
    rerank were actually used. Latency does not characterise distribution
    quality — a fast but poorly-ranked result set is not evidence that
    T should change. 30d posterior decay, matching the rest of the #480
    family. Cold-start ``static_default = 0.5`` decodes via
    :func:`resolve_posterior_temperature_with_meta` to ``T = 1.0`` (the
    geometric mean of FLOOR=0.5 and CEIL=2.0 in log space), which is
    byte-identical to the log-additive partial_bayesian_score baseline.
    """
    from aelfrice.meta_beliefs import SIGNAL_RELEVANCE
    return store.install_meta_belief(
        META_POSTERIOR_TEMPERATURE_KEY,
        static_default=META_POSTERIOR_TEMPERATURE_STATIC_DEFAULT,
        half_life_seconds=META_POSTERIOR_TEMPERATURE_POSTERIOR_DECAY_SECONDS,
        signal_weights={SIGNAL_RELEVANCE: 1.0},
        now_ts=now_ts,
    )


def resolve_posterior_temperature_with_meta(
    store: "MemoryStore | None",
    *,
    now_ts: int,
) -> float:
    """Resolve the γ-rerank Boltzmann temperature `T` (#796).

    Reads `meta:retrieval.posterior_temperature` from the store via
    `read_meta_belief_value`. Returns:

      * `T = 1.0` when `store` is None or the meta-belief is not
        installed — the byte-identical-to-log-additive case (γ at
        `T = 1.0` equals `partial_bayesian_score(..., 1.0)`).
      * Log-linear decode of the meta-belief value to
        `[POSTERIOR_TEMPERATURE_FLOOR, POSTERIOR_TEMPERATURE_CEIL]`
        otherwise. With the static_default of 0.5 the decode lands
        at the geometric mean 1.0 exactly, so a cold-start install
        is still byte-identical until evidence accumulates.

    Adaptive learning of `T` is #758's scope; #796 ships the surface
    and the decoder only. The store read is best-effort — any error
    falls back to `T = 1.0` rather than raising.
    """
    if store is None:
        return 1.0
    try:
        raw = store.read_meta_belief_value(
            META_POSTERIOR_TEMPERATURE_KEY, now_ts=now_ts,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "aelfrice retrieval: posterior-temperature meta-belief "
            f"read failed: {exc}",
            file=sys.stderr,
        )
        return 1.0
    if raw is None:
        return 1.0
    # Clamp the [0, 1] posterior surface value to its valid band before
    # the log-linear decode. The store contract should already keep it
    # in-range; the clamp is defensive against future code paths that
    # write raw values.
    v = max(0.0, min(1.0, float(raw)))
    log_floor = math.log(POSTERIOR_TEMPERATURE_FLOOR)
    log_ceil = math.log(POSTERIOR_TEMPERATURE_CEIL)
    return math.exp(log_floor + v * (log_ceil - log_floor))


def resolve_use_zeta_posterior_rerank(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the ζ rerank flag (#817 / #800).

    Precedence (first decisive wins):
      1. AELFRICE_USE_ZETA_POSTERIOR_RERANK env var.
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_zeta_posterior_rerank` in `.aelfrice.toml`.
      4. Default: False — ships behind the flag at v3.x. Flip-default
         is gated on the same labeled relevance corpus as γ's flip,
         per `docs/feature-zeta-posterior-rerank.md` §
         "Bench-gate / ship-or-defer policy".

    Mirror of `resolve_use_gamma_posterior_temperature`. The mutual-
    exclusion with γ is enforced at `retrieve()` /
    `retrieve_with_tiers()` entry via `_assert_gamma_zeta_mutual_exclusion`
    after both flags have been resolved — this resolver returns a bool
    independent of γ's state.
    """
    env = _env_use_zeta_posterior_rerank_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(
        USE_ZETA_POSTERIOR_RERANK_FLAG, start,
    )
    if toml_value is not None:
        return toml_value
    return False


def _assert_gamma_zeta_mutual_exclusion(
    gamma_on: bool, zeta_on: bool,
) -> None:
    """Raise `ValueError` if both the γ and ζ rerank flags resolve True.

    Composition semantics are deferred per issue #817 §"Out of scope" —
    the operator decision is "raise at flag resolution, operator picks
    one". This helper is called from `retrieve()` and
    `retrieve_with_tiers()` after each flag is resolved independently.
    """
    if gamma_on and zeta_on:
        raise ValueError(
            "γ and ζ posterior rerank flags are mutually exclusive on a "
            "given retrieval call. "
            f"AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE / "
            f"{USE_GAMMA_POSTERIOR_TEMPERATURE_FLAG} resolved True AND "
            f"AELFRICE_USE_ZETA_POSTERIOR_RERANK / "
            f"{USE_ZETA_POSTERIOR_RERANK_FLAG} resolved True. "
            "Pick one. See docs/feature-zeta-posterior-rerank.md § "
            "'Composition with γ' (issue #817)."
        )


_PLACEHOLDER_WARNED: set[str] = set()


@dataclass(frozen=True)
class LaneTelemetry:
    """Per-lane counters from the most recent `retrieve()` /
    `retrieve_with_tiers()` call. v1.5.0 #154 surface; consumed
    by `aelf doctor` and the v1.6+ benchmark gates.

    Counts are post-dedupe (a belief that L0 surfaced is not
    counted again by L2.5 or L1). `bm25f_used` records whether
    the BM25F sparse-matvec lane was the L1 implementation
    (True) or the FTS5 path (False) for the call.
    """

    locked: int = 0
    l25: int = 0
    l1: int = 0
    bfs: int = 0
    bm25f_used: bool = False
    posterior_weight: float = 0.0
    # #741 adaptive expansion-gate. ``expansion_gate_reason`` is the
    # tag returned by :func:`aelfrice.expansion_gate.should_run_expansion`
    # (e.g. ``"narrow"``, ``"broad:long,no-markers"``,
    # ``"env-force-expansion"``). ``expansion_gate_skipped_bfs`` is True
    # when the gate forced ``bfs_on=False`` on this call. Both fields
    # default to safe values so callers built against the pre-#741
    # LaneTelemetry surface keep working.
    expansion_gate_reason: str = ""
    expansion_gate_skipped_bfs: bool = False
    # #857 coverage-line support. ``l1_candidates`` is the total number
    # of L1 hits after BM25 scoring and dedup against L0/L2.5, before
    # token-budget trimming. ``l1`` tracks what was packed; the delta
    # is what the hook's coverage line surfaces to the user.
    l1_candidates: int = 0
    # #981 HRR expansion lane. Count of beliefs the vocabulary-bridge
    # expansion lane merged into the candidate set (0 when the lane is off,
    # the default). Lets the ablation read the lane's contribution per call.
    hrr_expand: int = 0
    # #1064 temporal-spine lane. ``temporal_spine`` is the packed survivor
    # count (what the lane actually added to the output within budget);
    # ``temporal_spine_candidates`` is what the traversal discovered before
    # dedup + token-budget packing. The delta is the lane's trim loss —
    # the G2 flip-gate question ("does it survive the production trim").
    temporal_spine: int = 0
    temporal_spine_candidates: int = 0
    # #1162 heat-kernel reachability. True only when the heat branch
    # actually rewrote the L1 ordering on this call — i.e. the flag was
    # on AND a non-stale eigenbasis was supplied AND its rows overlapped
    # the L1 hits. Deliberately *not* the resolved flag (unlike
    # `bm25f_used`): the lane's defect was that the flag reported an
    # active lane nothing in `src/` can reach, because no production
    # caller constructs a `GraphEigenbasisCache`. This field is what
    # turns that from a grep into a runtime fact.
    heat_used: bool = False


# Per-process snapshot of the most recent retrieval call. Test-
# friendly and zero-overhead (one assignment per retrieve()).
# Not thread-safe; callers that share a store across threads
# should consume the per-call return values from
# `retrieve_with_tiers` instead.
_LAST_TELEMETRY: LaneTelemetry = LaneTelemetry()


def last_lane_telemetry() -> LaneTelemetry:
    """Return the LaneTelemetry of the most recent retrieve() call
    in this process. Used by `aelf doctor` and benchmark gates."""
    return _LAST_TELEMETRY


def _reset_last_telemetry(tel: LaneTelemetry) -> None:
    """Overwrite the process-level LaneTelemetry snapshot.

    Exposed for the hook's pre-retrieval reset so that callers reading
    `last_lane_telemetry()` after a mocked `_retrieve` see a fresh
    zero-initialized snapshot rather than a stale one from a prior
    real call.
    """
    global _LAST_TELEMETRY
    _LAST_TELEMETRY = tel


# #1162. Whether the heat branch rewrote the ordering on the most recent
# `_l1_hits` call. Written only where the heat map is computed, so there
# is exactly one place the answer comes from — recomputing the predicate
# at the telemetry site would let a re-wired lane keep reporting the old
# answer, which is the failure this field exists to catch.
_LAST_HEAT_USED: bool = False


def _record_heat_used(used: bool) -> None:
    """Record whether the heat-kernel branch fired on this L1 pass."""
    global _LAST_HEAT_USED
    _LAST_HEAT_USED = used


def warn_placeholder_flags(start: Path | None = None) -> list[str]:
    """Read every `[retrieval] use_<lane>` placeholder flag from
    `.aelfrice.toml` and emit a stderr warning per flag set to
    True. Returns the list of placeholder names that were warned
    on (mostly for the test suite; callers can ignore the return
    value).

    Placeholder flags correspond to retrieval lanes that are
    spec'd by #154 but ship across v1.6 / v1.7 (signed Laplacian,
    heat kernel, posterior-full, HRR structural). Setting a
    placeholder True at v1.5.0 is a no-op; the warning tells the
    user the flag was recognised but the lane is not yet wired.

    Fail-soft: an unreadable / malformed TOML produces no warning.
    The intent is a forward-compat receipt, not a config gate.
    """
    warned: list[str] = []
    for flag in PLACEHOLDER_FLAGS:
        if flag in _PLACEHOLDER_WARNED:
            continue
        value = _read_toml_flag_for(flag, start)
        if value is True:
            print(
                f"aelfrice retrieval: [{RETRIEVAL_SECTION}] {flag} = true "
                f"recognised but the corresponding lane has not yet "
                f"shipped (v1.5.0 placeholder; tracked under #154). "
                f"No-op until the owning component lands.",
                file=sys.stderr,
            )
            _PLACEHOLDER_WARNED.add(flag)
            warned.append(flag)
    return warned


def _reset_placeholder_warnings() -> None:
    """Test-only helper: clear the once-per-process warning set so
    a test that toggles a placeholder flag and re-invokes the
    warner sees the warning again. Not part of the public API."""
    _PLACEHOLDER_WARNED.clear()


def is_heat_kernel_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the heat-kernel authority-scoring flag (#150).

    Precedence (first decisive wins):
      1. AELFRICE_HEAT_KERNEL env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] use_heat_kernel` in `.aelfrice.toml`.
      4. Default: False (#1162). The composition tracker (#154) flipped
         this to True once the #437 reproducibility-harness gate cleared
         at 11/11, and it has reported an active lane ever since — but
         the lane cannot fire on any production path, because nothing in
         `src/` constructs a `GraphEigenbasisCache`. `retrieve()` takes
         one as a parameter defaulting to None in four signatures and
         never builds it; only tests do. A True default therefore
         asserted something untrue about the shipped pipeline.

    **The flip is inert, not a behaviour change.** The heat branch is
    guarded on a non-stale eigenbasis being available as well as on this
    flag, so with no eigenbasis it was already falling through to the
    heat-off path on every call. γ and ζ already run their
    "no eigenbasis" branch today for the same reason. `heat_used` on
    `LaneTelemetry` is what makes that checkable at runtime rather than
    by grep.

    Turning it back on is still a one-line opt-in, and the lane is left
    wired for the day someone benches it (#1113 closed graph-theory-as-
    a-lever negative, so that would be a bench proposal, not a fix).

    Reuses the `HEAT_KERNEL_FLAG` constant that #232 introduced as a
    placeholder. Now that the lane has shipped, the flag is no
    longer in `PLACEHOLDER_FLAGS` so `warn_placeholder_flags()` will
    not flag it as unwired.
    """
    env = _env_heat_kernel_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(HEAT_KERNEL_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


def is_bfs_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the BFS multi-hop flag.

    Precedence (first decisive wins):
      1. AELFRICE_BFS env var (truthy / falsy normalised).
      2. Explicit `explicit` kwarg from the caller.
      3. `[retrieval] bfs_enabled` in `.aelfrice.toml`.
      4. Default: False (v1.3.0 default-OFF).

    The default-off contract is part of the v1.3.0 acceptance
    criteria: a fresh install must not change retrieval output
    against the v1.2 baseline.
    """
    env = _env_bfs_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_toml_flag_for(BFS_FLAG, start)
    if toml_value is not None:
        return toml_value
    return False


# --- Retrieval -----------------------------------------------------------


def _l25_hits(
    store: MemoryStore,
    query: str,
    *,
    locked_ids: set[str],
    l25_limit: int,
    l25_token_subbudget: int,
    query_entity_cap: int,
    use_origin_tiebreak: bool = False,
    use_fan_effect: bool = False,
) -> list[Belief]:
    """Run L2.5: query-side extraction, entity lookup, materialise
    beliefs, dedupe vs L0, trim to `l25_token_subbudget`.

    Returns at most `l25_limit` beliefs whose summed token estimate
    is at or below `l25_token_subbudget`. The trim is from the
    tail (lowest-overlap matches drop first).

    `use_fan_effect` (#1176) swaps the lane's raw overlap ordering for
    ACT-R fan-weighted activation. It reorders *which* beliefs the trim
    keeps, not how many, so the budget arithmetic below is unchanged.

    A `l25_token_subbudget <= 0` short-circuits to []. The outer
    `retrieve()` enforces that the L2.5 sub-budget never exceeds
    the remaining `token_budget`, so passing 0 is the correct
    expression of "no L2.5 budget left".
    """
    if l25_token_subbudget <= 0:
        return []
    q_entities = extract_entities(query, max_entities=query_entity_cap)
    if not q_entities:
        return []
    keys = [e.lower for e in q_entities]
    hits = store.lookup_entities(
        keys, limit=l25_limit, origin_tiebreak=use_origin_tiebreak,
        fan_effect=use_fan_effect,
    )
    out: list[Belief] = []
    used = 0
    for bid, _overlap in hits:
        if bid in locked_ids:
            continue
        b = store.get_belief(bid)
        if b is None:
            # Race: belief was deleted between lookup and fetch.
            # Skip; the index will be cleaned up by the next mutation
            # cycle (delete_belief cascades to belief_entities).
            continue
        cost = _belief_tokens(b)
        if used + cost > l25_token_subbudget:
            break
        out.append(b)
        used += cost
    return out


def _heat_by_id(
    cache: GraphEigenbasisCache,
    bm25_pos_by_id: dict[str, float],
) -> dict[str, float] | None:
    """Run one heat-kernel propagation pass and return per-belief
    authority scores keyed by belief id.

    `cache` must already hold a non-stale eigenbasis (caller checks
    `cache.is_stale()` and `cache.eigvals is not None`). `bm25_pos_by_id`
    is a `{belief_id: bm25_pos}` slice of the L1 hits, where `bm25_pos`
    is the same positive-relevance magnitude used by
    `partial_bayesian_score` (FTS5 path passes `-bm25_raw`; BM25F path
    passes `raw` directly).

    Returns `None` when the cache rows don't intersect the L1 hit set
    (every L1 belief was inserted after the eigenbasis build) or when
    the seed sum is zero — caller falls back to the heat-off path. The
    explicit None signal lets the caller short-circuit the matvec when
    propagation is guaranteed to be a no-op.
    """
    import numpy as np

    if cache.eigvals is None or cache.eigvecs is None or not cache.belief_ids:
        return None
    n = len(cache.belief_ids)
    bm25_arr = np.zeros(n, dtype=np.float64)
    hit_indices: list[int] = []
    for i, bid in enumerate(cache.belief_ids):
        v = bm25_pos_by_id.get(bid)
        if v is not None and v > 0.0:
            bm25_arr[i] = v
            hit_indices.append(i)
    if not hit_indices:
        return None
    seeds = seeds_from_bm25(bm25_arr, top_k=DEFAULT_BM25_SEED_TOP_K)
    if not float(seeds.sum()) > 0.0:
        return None
    heat = heat_kernel_score(
        cache.eigvals, cache.eigvecs, seeds, t=DEFAULT_HEAT_BANDWIDTH,
    )
    return {bid: float(heat[i]) for i, bid in enumerate(cache.belief_ids)}


def _entity_persist_penalty(
    ep: dict[str, float] | None, belief_id: str
) -> float:
    """Log-additive entity-persistence demotion for one belief (#1096).

    Returns 0.0 (no demotion) when the lane is off (`ep is None`) or the
    belief has no extracted entities (absent from `ep`) — so entity-free
    durable content is never penalised. Otherwise returns
    `WEIGHT * log(S1 + EPS)`, which is ~0 for well-grounded beliefs
    (S1 → 1) and increasingly negative for ephemeral coordination
    (S1 → 0). Deterministic."""
    if ep is None:
        return 0.0
    s1 = ep.get(belief_id)
    if s1 is None:
        return 0.0
    # Clamp at 0 so this is a pure demotion, never a promotion — a
    # well-grounded belief (S1 → 1) is neutral, not boosted.
    return min(
        0.0,
        ENTITY_PERSIST_DEMOTE_WEIGHT * math.log(s1 + ENTITY_PERSIST_DEMOTE_EPS),
    )


def _store_scoped_utterance_prior(store: MemoryStore) -> UtterancePrior:
    """One process-lifetime utterance prior per store (#1174 item 3).

    Building costs a full pass over `ingest_log`, so it must not happen
    per query. Cached on the store, like `_store_scoped_bm25f_cache`, so
    its lifetime is the store's.

    Deliberately *not* invalidated on store mutation. The table is a
    corpus-level vocabulary statistic over tens of thousands of ingest
    rows; a handful of new rows cannot move a mean log-odds enough to
    reorder anything, and subscribing to the invalidation hook would
    rebuild the whole table on every write. A long-running process picks
    up new vocabulary on restart.
    """
    cached = getattr(store, "_utterance_prior_cache", None)
    if not isinstance(cached, UtterancePrior):
        cached = utterance_logodds(store)
        store._utterance_prior_cache = cached
    return cached


def _origin_priority(origin: str) -> int:
    """Retrieval tie-break priority for an `origin` (higher sorts first).

    #1089 axis 2. Reads the canonical `ORIGIN_RETRIEVAL_PRIORITY` map
    (mirrors `contradiction.precedence_class`); origins absent from it
    fall to the default bucket. Used only as a tie-break on the composite
    rerank score, never as a primary term (the origin *rerank lane* was
    refuted in #1013)."""
    return ORIGIN_RETRIEVAL_PRIORITY.get(
        origin, ORIGIN_RETRIEVAL_PRIORITY_DEFAULT,
    )


def _store_scoped_bm25f_cache(
    store: MemoryStore,
    *,
    anchor_weight: int,
    k3: float,
    per_field: bool = False,
    b_anchor: float | None = None,
) -> BM25IndexCache:
    """One process-lifetime `BM25IndexCache` per store (#1135).

    Replaces the pre-#1135 per-retrieve construction, which leaked one
    invalidation-callback subscription per query and threw away the
    built index between calls on long-running processes (MCP server).
    The cache lives on `store._bm25f_shared_cache` so its lifetime is
    the store's. A changed `anchor_weight` (the meta-belief consumer
    can move it between calls) drops the cached index; the sidecar
    check in `BM25IndexCache.get()` compares weights independently.
    """
    from aelfrice.bm25 import DEFAULT_B_ANCHOR

    effective_b_anchor = (
        DEFAULT_B_ANCHOR if b_anchor is None else b_anchor
    )
    cache = store._bm25f_shared_cache  # noqa: SLF001 — slot owned here
    if not isinstance(cache, BM25IndexCache):
        cache = BM25IndexCache(
            store, anchor_weight=anchor_weight, k3=k3,
            per_field=per_field, b_anchor=effective_b_anchor,
        )
        store._bm25f_shared_cache = cache  # noqa: SLF001
    elif (
        cache.anchor_weight != anchor_weight
        or cache.k3 != k3
        or cache.per_field != per_field
        # #1180: b_anchor only participates in scoring under per_field,
        # so changing it while off must not force a rebuild.
        or (per_field and cache.b_anchor != effective_b_anchor)
    ):
        # #1166: k3 rides the same invalidation path as anchor_weight.
        # It is carried on the built index, so a cached index built
        # under a different k3 would keep scoring with the stale value.
        # #1180: per_field / b_anchor ride it for the same reason.
        cache.anchor_weight = anchor_weight
        cache.k3 = k3
        cache.per_field = per_field
        cache.b_anchor = effective_b_anchor
        cache.invalidate()
    return cache


# #1187 exclusion-arm refetch. The candidate limit is applied by the
# search (SQL `LIMIT` on the FTS5 path, `top_k` on BM25F), so filtering
# superseded beliefs afterwards SHRINKS the pack instead of backfilling
# from below the cutoff: at `l1_limit=4` with three of the top four
# retired, the arm returned one belief while three current ones sat at
# ranks 5-7. That is the same shape as the lock-budget starvation fixed
# in #1014/#1015 — a filter applied after the budget starves the pack —
# and it would also confound the demote-vs-exclude bench, since the two
# arms would differ in pack size as well as in treatment.
#
# So the exclusion arm widens the fetch and retries, stopping as soon as
# it has `l1_limit` survivors or the search runs out of matches
# (`len(rows) < limit` — a further round would return the same rows).
#
# The round cap is a backstop, not the normal termination (#1205). At
# three rounds it was the normal termination on any store where retired
# beliefs dominate a query's matches: 200 of 300 matching beliefs
# retired at `l1_limit=50` returned an EMPTY pack while a hundred
# current ones sat below the widest fetch attempted — the starvation
# this helper exists to remove, moved out by 4x rather than removed. It
# is now high enough that binding means something pathological, and
# binding TRACES rather than truncating silently: a short pack must be
# distinguishable from an empty store, or the demote-vs-exclude bench
# reads a floor as a measurement on exactly the corpus slice where
# supersession is most active.
SUPERSESSION_REFETCH_ROUNDS: Final[int] = 8


def _fetch_excluding_superseded(
    store: MemoryStore,
    fetch: Any,
    l1_limit: int,
) -> tuple[list[Any], frozenset[str]]:
    """Fetch `l1_limit` candidates that survive supersession exclusion.

    `fetch(limit)` returns a list of `(belief, raw)` pairs. Returns the
    kept pairs (at most `l1_limit`) and the superseded id set observed on
    the widest fetch, so the caller can reuse it without re-querying.
    """
    limit = l1_limit
    kept: list[Any] = []
    sup: frozenset[str] = frozenset()
    exhausted = False
    for _ in range(SUPERSESSION_REFETCH_ROUNDS):
        rows = fetch(limit)
        if not rows:
            return [], frozenset()
        sup = frozenset(store.superseded_belief_ids([b.id for b, _ in rows]))
        kept = [(b, raw) for b, raw in rows if b.id not in sup]
        if len(kept) >= l1_limit or len(rows) < limit:
            # Either the pack is full, or the search has no more matches
            # to widen into — a further round would return the same rows.
            exhausted = True
            break
        limit *= 2
    if not exhausted:
        # The cap bound with the search still yielding a full page, so
        # there may be survivors below `limit` that were never read.
        # Say so: a caller cannot otherwise tell this pack from one the
        # store genuinely could not fill, and a silent floor biases the
        # exclusion arm precisely where supersession is most active.
        print(
            f"aelfrice retrieval: supersession exclusion stopped at "
            f"{SUPERSESSION_REFETCH_ROUNDS} widening rounds (limit "
            f"{limit}) with {len(kept)} of {l1_limit} survivors; there "
            f"may be more below the cutoff",
            file=sys.stderr,
        )
    return kept[:l1_limit], sup



def _l1_hits(
    store: MemoryStore,
    query: str,
    *,
    l1_limit: int,
    posterior_weight: float,
    use_bm25f_anchors: bool = False,
    bm25f_cache: BM25IndexCache | None = None,
    eigenbasis_cache: GraphEigenbasisCache | None = None,
    heat_kernel_on: bool = False,
    gamma_temperature: float | None = None,
    zeta_params: tuple[float, float, float] | None = None,
    use_entity_persist_demote: bool = False,
    use_origin_tiebreak: bool = False,
    use_supersession_demote: bool = False,
    supersession_treatment: str = SUPERSESSION_TREATMENT_DEMOTE,
    supersession_factor: float = SUPERSESSION_DEMOTE_FACTOR,
    utterance_prior_weight: float = 0.0,
    now_ts: int | None = None,
) -> list[Belief]:
    """Run L1: FTS5 BM25 search (default) or BM25F sparse-matvec
    (v1.5.0 opt-in), optionally reranked by partial-Bayesian score.

    `use_bm25f_anchors = True` swaps the FTS5 lane for `BM25Index.score`
    over the augmented (content + W * incoming-anchor) document set.
    The posterior rerank still applies on top of the BM25F score.
    The cache is rebuilt on store mutation via the BM25IndexCache
    invalidation hook.

    `posterior_weight = 0.0` and FTS5 path short-circuits to the
    v1.0.x byte-identical contract. BM25F + posterior_weight = 0.0
    returns the BM25F top-K in score-descending, tie-break id-ASC
    order — the byte-identical guarantee against FTS5 only holds
    when use_bm25f_anchors is False.

    `posterior_weight > 0` reranks via `partial_bayesian_score`.

    `gamma_temperature` (v3.x #796): when not None AND `heat_kernel_on`
    is False (or no eigenbasis is available), the rerank loop swaps
    `partial_bayesian_score(bm25_raw, α, β, posterior_weight)` for
    `gamma_posterior_score(bm25_raw, α, β, gamma_temperature)`. At
    `T = 1.0` γ is byte-identical to `partial_bayesian_score` at
    `posterior_weight = 1.0`. None falls through to the log-additive
    path. When `heat_kernel_on` is True and the heat-rerank fires γ is
    a no-op on this call — the two scoring paths are mutually
    exclusive by design (the operator decision deferred composition
    to a later issue).

    `zeta_params` (v3.x #817 / #800): when not None AND `heat_kernel_on`
    is False (or no eigenbasis is available), the rerank loop swaps
    `partial_bayesian_score(...)` for
    `zeta_posterior_score(-raw, ζα, ζβ, ζscale, posterior_mean(α, β))`.
    The tuple is `(alpha, beta, scale)` in ζ-parameter space (not
    Beta-Bernoulli α/β). None falls through to the log-additive or γ
    path; γ and ζ are caller-enforced mutually exclusive — see
    `_assert_gamma_zeta_mutual_exclusion`. Heat-rerank dominates ζ
    just as it does γ.

    `heat_kernel_on` (v1.7.0): when True AND `eigenbasis_cache` holds a
    non-stale eigenbasis whose `belief_ids` intersect the L1 hit set,
    the rerank uses `combine_log_scores(bm25, heat, posterior_mean)`
    instead of `partial_bayesian_score`. Heat propagation cost is the
    `eigvecs.T @ seeds` matvec (~7-8 ms at N=50k, K=200; see
    docs/design/bayesian_ranking.md § "Heat-kernel cost"). When the cache is
    None, stale, empty, or carries no overlap with the L1 hit ids, the
    path degrades to `partial_bayesian_score` — byte-identical to the
    heat-off contract. AC4 / AC8 of #151 are preserved by this fall-
    through.
    """
    heat_active = (
        heat_kernel_on
        and eigenbasis_cache is not None
        and not eigenbasis_cache.is_stale()
        and eigenbasis_cache.eigvals is not None
    )
    # #1162. Default the reachability signal to False for this call;
    # the two heat-map sites below overwrite it once they know whether
    # propagation produced anything. `heat_active` alone is not the
    # answer — `_heat_by_id` still returns None when the eigenbasis
    # rows do not intersect the L1 hits, and then the ordering is the
    # heat-off ordering.
    _record_heat_used(False)
    # #677 retrieval-time `#N` literal boost. When the prompt names
    # one or more `#NNN` tokens, bypass the byte-identical FTS5 and
    # BM25F short-circuits and go through the rerank loop so the
    # boost can take effect; for prompts without literals the gate
    # short-circuits as before and the byte-identical contract holds.
    hash_n_literals = _extract_hash_n_literals(query)

    # #1143 clock seam: one wall-clock read per call, and none at all
    # when the caller pins `now_ts` — the anchor-weight resolver and
    # the bm25_l0_ratio signal write see the same timestamp.
    effective_now_ts = now_ts if now_ts is not None else int(time.time())

    if use_bm25f_anchors:
        # The cache lazy-builds the index on first call and is
        # invalidated by store mutations. The rerank below uses the
        # raw BM25F score in the same `bm25_raw` slot the FTS5 path
        # uses for ``bm25(beliefs_fts)``; see partial_bayesian_score
        # for the log-additive composition.
        #
        # #757 wire-in: when no cache is supplied, resolve the
        # anchor_weight through the meta-belief consumer; an explicit
        # caller-supplied cache is honoured as-is (the bench harness
        # and unit tests pin specific anchor_weights via the cache).
        # #1135: the fallback cache is store-scoped and reused across
        # calls — constructing one per retrieve leaked an invalidation
        # callback per query and rebuilt the index every time on
        # long-running processes; with the sidecar (bm25.py) a fresh
        # hook process loads the persisted index instead of building.
        if bm25f_cache is None:
            cache = _store_scoped_bm25f_cache(
                store,
                anchor_weight=resolve_bm25f_anchor_weight_with_meta(
                    store, now_ts=effective_now_ts,
                ),
                k3=resolve_bm25_k3(),
                per_field=resolve_bm25f_per_field(),
                b_anchor=resolve_bm25_b_anchor(),
            )
        else:
            cache = bm25f_cache
        index = cache.get()
        scored_pairs = index.score(query, top_k=l1_limit)
        # #757 bm25_l0_ratio signal update. Only fires when the
        # meta-belief feature flag is on; the env check matches the
        # resolver above so the read/write paths flip together.
        # update_meta_belief is a no-op when the row isn't installed
        # (returns False), so this is safe without an explicit
        # existence check. Fail-soft: a store error here must never
        # break retrieval — mirrors the #756 latency-signal posture
        # in retrieve_v2.
        if is_meta_belief_bm25f_anchor_weight_enabled():
            try:
                locked_at_query = list(store.list_locked_beliefs())
                if locked_at_query:
                    bm25f_top_ids = {bid for bid, _ in scored_pairs}
                    hits = sum(
                        1 for b in locked_at_query
                        if b.id in bm25f_top_ids
                    )
                    evidence = hits / len(locked_at_query)
                    from aelfrice.meta_beliefs import SIGNAL_BM25_L0_RATIO
                    store.update_meta_belief(
                        META_BM25F_ANCHOR_WEIGHT_KEY,
                        SIGNAL_BM25_L0_RATIO,
                        evidence=evidence,
                        now_ts=effective_now_ts,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    "aelfrice retrieval: meta-belief bm25_l0_ratio "
                    f"update failed: {exc}",
                    file=sys.stderr,
                )
        def _bm25f_candidates(top_k: int) -> list[tuple[Belief, float]]:
            """Materialise the top-`top_k` BM25F hits as (belief, raw).

            Factored out so the #1187 exclusion arm can widen `top_k`
            and refetch instead of filtering a fixed-size slice. At
            `top_k == l1_limit` it reuses the scores already computed
            above rather than re-scoring the index.
            """
            pairs = (
                scored_pairs if top_k == l1_limit
                else index.score(query, top_k=top_k)
            )
            out: list[tuple[Belief, float]] = []
            for bid, raw in pairs:
                b = store.get_belief(bid)
                if b is None:
                    continue
                out.append((b, raw))
            return out

        beliefs: list[tuple[Belief, float]] = _bm25f_candidates(l1_limit)
        # γ / ζ are opt-in; when either is set it forces the rerank
        # loop so the byte-identical short-circuit can't bypass the
        # posterior reweighting.
        if (
            posterior_weight == 0.0
            and not heat_active
            and not hash_n_literals
            and gamma_temperature is None
            and zeta_params is None
            and not use_entity_persist_demote
            and not use_supersession_demote
            and utterance_prior_weight == 0.0
        ):
            return [b for b, _ in beliefs]
        # BM25F scores are non-negative; the rerank uses `raw` as the
        # positive-magnitude relevance signal directly (the FTS5 path
        # has to negate first because SQLite returns smaller-negative
        # for stronger matches; BM25F doesn't).
        # #1187: resolve supersession before anything downstream reads the
        # candidate set, so the exclusion arm also keeps superseded beliefs
        # out of the heat-kernel seeds rather than only out of the ranking.
        sup: frozenset[str] | None = None
        if use_supersession_demote:
            if supersession_treatment == SUPERSESSION_TREATMENT_EXCLUDE:
                # Widen and retry rather than filtering in place, so the
                # pack backfills from below the cutoff instead of
                # shrinking. See `_fetch_excluding_superseded`.
                beliefs, sup = _fetch_excluding_superseded(
                    store, _bm25f_candidates, l1_limit,
                )
                if not beliefs:
                    return []
            else:
                sup = frozenset(
                    store.superseded_belief_ids([b.id for b, _ in beliefs])
                )
        bm25_pos_by_id: dict[str, float] = {b.id: float(raw) for b, raw in beliefs}
        heat_map = (
            _heat_by_id(eigenbasis_cache, bm25_pos_by_id)  # type: ignore[arg-type]
            if heat_active else None
        )
        _record_heat_used(heat_map is not None)
        ep = (
            store.entity_persistence_scores([b.id for b, _ in beliefs])
            if use_entity_persist_demote else None
        )
        up = (
            _store_scoped_utterance_prior(store)
            if utterance_prior_weight != 0.0 else None
        )
        keyed: list[tuple[float, str, Belief]] = []
        for b, raw in beliefs:
            if heat_map is not None:
                s = combine_log_scores(
                    bm25f=max(float(raw), 1e-9),
                    heat=heat_map.get(b.id, HEAT_SCORE_FLOOR),
                    posterior=posterior_mean(b.alpha, b.beta),
                    heat_weight=DEFAULT_HEAT_KERNEL_WEIGHT,
                    posterior_weight=(
                        posterior_weight if posterior_weight > 0.0
                        else DEFAULT_POSTERIOR_LOG_WEIGHT
                    ),
                )
            elif gamma_temperature is not None:
                s = gamma_posterior_score(
                    -raw, b.alpha, b.beta, gamma_temperature,
                )
            elif zeta_params is not None:
                ζα, ζβ, ζscale = zeta_params
                s = zeta_posterior_score(
                    -raw, ζα, ζβ, ζscale,
                    posterior_mean(b.alpha, b.beta),
                )
            else:
                s = partial_bayesian_score(
                    -raw, b.alpha, b.beta, posterior_weight,
                )
            s = _hash_n_boosted(s, b.content, hash_n_literals)
            s += _entity_persist_penalty(ep, b.id)
            s += _supersession_penalty(sup, b.id, supersession_factor)
            s += utterance_prior_penalty(
                up, b.content, utterance_prior_weight,
            )
            keyed.append((s, b.id, b))
        if use_origin_tiebreak:
            keyed.sort(
                key=lambda x: (-x[0], -_origin_priority(x[2].origin), x[1])
            )
        else:
            keyed.sort(key=lambda x: (-x[0], x[1]))
        return [b for _, _, b in keyed]

    if (
        posterior_weight == 0.0
        and not heat_active
        and not hash_n_literals
        and gamma_temperature is None
        and zeta_params is None
        and not use_entity_persist_demote
        and not use_supersession_demote
        and utterance_prior_weight == 0.0
    ):
        return store.search_beliefs(
            query, limit=l1_limit, origin_tiebreak=use_origin_tiebreak,
        )
    scored = store.search_beliefs_scored(
        query, limit=l1_limit, origin_tiebreak=use_origin_tiebreak,
    )
    if not scored:
        return []
    # #1187: as on the BM25F path, resolve and apply exclusion before the
    # candidate set is read downstream.
    sup: frozenset[str] | None = None
    if use_supersession_demote:
        if supersession_treatment == SUPERSESSION_TREATMENT_EXCLUDE:
            # Widen and retry rather than filtering in place, so the pack
            # backfills from below the cutoff instead of shrinking. See
            # `_fetch_excluding_superseded`.
            scored, sup = _fetch_excluding_superseded(
                store,
                lambda lim: store.search_beliefs_scored(
                    query, limit=lim, origin_tiebreak=use_origin_tiebreak,
                ),
                l1_limit,
            )
            if not scored:
                return []
        else:
            sup = frozenset(
                store.superseded_belief_ids([b.id for b, _ in scored])
            )
    # FTS5 path: bm25_raw is non-positive (SQLite convention). Negate
    # to get a positive relevance magnitude, same convention used by
    # `partial_bayesian_score` internally.
    bm25_pos_by_id = {b.id: max(-bm25_raw, 1e-9) for b, bm25_raw in scored}
    heat_map = (
        _heat_by_id(eigenbasis_cache, bm25_pos_by_id)  # type: ignore[arg-type]
        if heat_active else None
    )
    _record_heat_used(heat_map is not None)
    ep = (
        store.entity_persistence_scores([b.id for b, _ in scored])
        if use_entity_persist_demote else None
    )
    up = (
        _store_scoped_utterance_prior(store)
        if utterance_prior_weight != 0.0 else None
    )
    keyed: list[tuple[float, str, Belief]] = []
    for b, bm25_raw in scored:
        if heat_map is not None:
            s = combine_log_scores(
                bm25f=max(-bm25_raw, 1e-9),
                heat=heat_map.get(b.id, HEAT_SCORE_FLOOR),
                posterior=posterior_mean(b.alpha, b.beta),
                heat_weight=DEFAULT_HEAT_KERNEL_WEIGHT,
                posterior_weight=(
                    posterior_weight if posterior_weight > 0.0
                    else DEFAULT_POSTERIOR_LOG_WEIGHT
                ),
            )
        elif gamma_temperature is not None:
            s = gamma_posterior_score(
                bm25_raw, b.alpha, b.beta, gamma_temperature,
            )
        elif zeta_params is not None:
            ζα, ζβ, ζscale = zeta_params
            s = zeta_posterior_score(
                bm25_raw, ζα, ζβ, ζscale,
                posterior_mean(b.alpha, b.beta),
            )
        else:
            s = partial_bayesian_score(
                bm25_raw, b.alpha, b.beta, posterior_weight,
            )
        s = _hash_n_boosted(s, b.content, hash_n_literals)
        s += _entity_persist_penalty(ep, b.id)
        s += _supersession_penalty(sup, b.id, supersession_factor)
        s += utterance_prior_penalty(
            up, b.content, utterance_prior_weight,
        )
        keyed.append((s, b.id, b))
    # Higher score = more relevant. Tie-break on id ASC for
    # determinism (matches the convention in bfs_multihop and L2.5).
    # #1089 axis 2: when enabled, break a composite-score tie by origin
    # priority first (curated user_validated over conversational
    # user_transcript), then id — a pure tie-break, never a primary term.
    if use_origin_tiebreak:
        keyed.sort(
            key=lambda x: (-x[0], -_origin_priority(x[2].origin), x[1])
        )
    else:
        keyed.sort(key=lambda x: (-x[0], x[1]))
    return [b for _, _, b in keyed]


@_memoize_config_discovery
def retrieve(
    store: MemoryStore,
    query: str,
    token_budget: int | None = None,
    l1_limit: int | None = None,
    *,
    entity_index_enabled: bool | None = None,
    l25_limit: int = DEFAULT_L25_LIMIT,
    l25_token_subbudget: int = DEFAULT_L25_TOKEN_SUBBUDGET,
    query_entity_cap: int = DEFAULT_QUERY_ENTITY_CAP,
    bfs_enabled: bool | None = None,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
    use_bm25f_anchors: bool | None = None,
    bm25f_cache: BM25IndexCache | None = None,
    heat_kernel_enabled: bool | None = None,
    eigenbasis_cache: GraphEigenbasisCache | None = None,
    use_type_aware_compression: bool | None = None,
    manifest_reference_locks: bool = False,
) -> list[Belief]:
    """Return L0 locked + L2.5 entity + L1 BM25 + L3 BFS expansions.

    Output is token-budgeted: results are trimmed from the tail
    until the estimated total token count is at or below
    `token_budget`. L0 beliefs are never trimmed.

    Exception (#1014): because L0 locks are never trimmed, a store whose
    locks alone meet or exceed `token_budget` reserves a relevance floor
    (`RELEVANCE_BUDGET_FLOOR_FRACTION` of the budget) for L2.5/L1 so locks
    can't starve query-relevant hits to zero. In that lock-saturated
    regime the returned total may exceed `token_budget` by up to that
    floor; outside it the budget cap holds exactly (byte-identical).

    L2.5 (v1.3.0): entity-index lookup. Default-on; gated by
    `is_entity_index_enabled()` (env override → kwarg → TOML →
    default True). When disabled the path collapses to v1.2's L0+L1
    behaviour byte-for-byte, with the legacy budget if the caller
    didn't pass an explicit one.

    L3 (v1.3.0): BFS multi-hop expansion. Default-OFF; gated by
    `is_bfs_enabled()` (env override → kwarg → TOML → default
    False). Seeds are the L0+L2.5+L1 set that survived the prior
    tiers' filtering. Expansions are appended in score-descending
    order until the shared token budget is exhausted. When
    disabled, output is byte-identical to the L0+L2.5+L1 path.

    `posterior_weight` (v1.3.0): float ≥ 0. Combines the L1 BM25
    score with the Beta-Bernoulli posterior_mean log-additively:
    `score = log(-bm25) + posterior_weight * log(posterior_mean)`.
    `0.0` collapses to v1.0.x BM25-only ordering (byte-identical
    regression-tested). Default `0.5` per docs/design/bayesian_ranking.md
    § Defaults; resolved via `resolve_posterior_weight()` (env →
    kwarg → TOML → 0.5). L0 locks bypass the score entirely; L2.5
    and L3 are unaffected.

    Empty / whitespace-only query: returns L0 only (no L2.5, L1, or
    L3).

    Dedupe: L1 hits whose id appears in L0 or L2.5 are dropped
    before budget accounting. L2.5 hits whose id appears in L0 are
    likewise dropped. L3 expansions whose id appears in any prior
    tier are dropped (the visited-set in `expand_bfs` prevents
    seeds from being re-surfaced; we additionally guard against
    overlap with L1 hits the seeds didn't include).

    `use_type_aware_compression` (#776, v3.1): when True, L2.5 / L1 /
    BFS pack accounting uses each belief's compressed
    `rendered_tokens` (via `compress_for_retrieval`) instead of
    `_belief_tokens(b)`. Locks render verbatim per the strategy
    table, so L0 accounting is unchanged. Default-ON since the #769
    flip — the resolver returns True unless env / kwarg / TOML opts
    out. Mirrors the wiring
    `retrieve_with_tiers` has carried since v2.0 (#434); added to
    bare `retrieve()` so `rebuild_v14`'s call site observes the
    toggle that A4 (#775) measures.
    """
    # #1107 cutover: `retrieve()` is a thin adapter over `retrieve_v2`, the
    # single retrieval implementation the production hook path shares with
    # the benchmark/eval surface. Lanes light up in production one at a time
    # as each clears its gate; a graduated lane is passed through as `None`
    # so the production path honours its resolver (env -> TOML -> default)
    # exactly like the eval surface. Graduated so far: **temporal spine**
    # (#1064, Phase 2), **entity-persist demotion** (#1096, Phase 3 — the
    # #1086 junk-percolation sink), **intentional clustering** (#436, Phase 4 —
    # multi-fact cluster-coverage, resolver default-ON since v3.0's A2/A3/A4
    # gate) and the **HRR structural-query lane** (#152, Phase 5 — marker-routed
    # `<KIND>:<target_id>` routing, resolver default-ON since v2.1's #154/#437
    # gate; a no-op fall-through on non-marker queries). The remaining two
    # staged lanes (origin tie-break, HRR-expand) stay forced OFF, because
    # `retrieve()`'s historical pack loop never ran them; equivalence for those
    # is pinned by tests/test_retrieve_v2_equivalence.py.
    out = retrieve_v2(
        store,
        query,
        budget=token_budget,
        l1_limit=l1_limit,
        use_entity_index=entity_index_enabled,
        l25_limit=l25_limit,
        l25_token_subbudget=l25_token_subbudget,
        query_entity_cap=query_entity_cap,
        use_bfs=bfs_enabled,
        bfs_max_depth=bfs_max_depth,
        bfs_nodes_per_hop=bfs_nodes_per_hop,
        bfs_total_budget_nodes=bfs_total_budget_nodes,
        bfs_min_path_score=bfs_min_path_score,
        posterior_weight=posterior_weight,
        use_bm25f=use_bm25f_anchors,
        bm25f_cache=bm25f_cache,
        heat_kernel_enabled=heat_kernel_enabled,
        eigenbasis_cache=eigenbasis_cache,
        use_type_aware_compression=use_type_aware_compression,
        manifest_reference_locks=manifest_reference_locks,
        # Graduated lanes (#1107): resolver-driven (env -> TOML -> default)
        # rather than hard-off, so a production host gets the lane the moment
        # its resolver says on. temporal spine #1064 (Phase 2), entity-persist
        # demotion #1096 (Phase 3), intentional clustering #436 (Phase 4),
        # HRR structural-query lane #152 (Phase 5) — all resolver default-ON.
        use_temporal_spine=None,
        use_entity_persist_demote=None,
        # #1187: resolver-driven so the lane is reachable from the
        # production path, but the resolver default is OFF pending the bench.
        use_supersession_demote=None,
        supersession_treatment=None,
        supersession_factor=None,
        use_intentional_clustering=None,
        use_hrr_structural=None,
        # #1176 fan effect: spelled `None` to match the resolver-driven
        # lanes above, not because `False` would break it — the resolver
        # is env-first, so `AELFRICE_FAN_EFFECT=1` overrides either
        # spelling and the A/B runs against this path regardless (that
        # is measured, not assumed: hard-coding `False` here leaves
        # tests/test_fan_effect_1176.py wholly green). `None` is the
        # right spelling anyway — it is what a `.aelfrice.toml` tier
        # would have to read through, and that tier is the natural
        # companion to a default flip. Resolver default stays OFF.
        use_fan_effect=None,
        use_origin_tiebreak=False,
        use_hrr_expand=False,
    ).beliefs
    # v1.6.0 #191: enqueue one retrieval_exposure row per surfaced belief
    # for the deferred-feedback sweeper. Default-on; opt-out via
    # [implicit_feedback] enqueue_on_retrieve = false. Fail-soft: any DB
    # error here is logged but never breaks retrieval. This is a `retrieve()`
    # (production-hook) side effect, kept here rather than in `retrieve_v2`
    # so the benchmark / eval surface stays side-effect-free.
    if out:
        try:
            from aelfrice.deferred_feedback import (
                enqueue_retrieval_exposures,
                is_enqueue_on_retrieve_enabled,
            )
            if is_enqueue_on_retrieve_enabled():
                enqueue_retrieval_exposures(store, [b.id for b in out])
        except Exception as exc:  # noqa: BLE001 - retrieval must never raise
            print(
                f"aelfrice retrieval: deferred-feedback enqueue failed: {exc}",
                file=sys.stderr,
            )
    return out


@_memoize_config_discovery
def retrieve_with_tiers(
    store: MemoryStore,
    query: str,
    token_budget: int | None = None,
    l1_limit: int | None = None,
    *,
    budget_defaulted: bool | None = None,
    entity_index_enabled: bool | None = None,
    l25_limit: int = DEFAULT_L25_LIMIT,
    l25_token_subbudget: int = DEFAULT_L25_TOKEN_SUBBUDGET,
    query_entity_cap: int = DEFAULT_QUERY_ENTITY_CAP,
    bfs_enabled: bool | None = None,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
    use_bm25f_anchors: bool | None = None,
    bm25f_cache: BM25IndexCache | None = None,
    heat_kernel_enabled: bool | None = None,
    eigenbasis_cache: GraphEigenbasisCache | None = None,
    use_type_aware_compression: bool | None = None,
    use_intentional_clustering: bool | None = None,
    hrr_expand_enabled: bool | None = None,
    hrr_struct_index_cache: HRRStructIndexCache | None = None,
    temporal_spine_enabled: bool | None = None,
    temporal_spine_depth: int | None = None,
    temporal_spine_node_budget: int | None = None,
    use_entity_persist_demote: bool = False,
    use_origin_tiebreak: bool = False,
    use_fan_effect: bool = False,
    use_supersession_demote: bool = False,
    supersession_treatment: str = SUPERSESSION_TREATMENT_DEMOTE,
    supersession_factor: float = SUPERSESSION_DEMOTE_FACTOR,
    utterance_prior_weight: float | None = None,
    manifest_reference_locks: bool = False,
    now_ts: int | None = None,
) -> tuple[
    list[Belief], list[str], list[str], list[str], list[list[str]],
]:
    """Same logic as `retrieve()` but returns the per-tier id lists
    alongside the merged output.

    Used by the v1.3.0 benchmark adapter to surface L0 / L1 / L2.5
    counts in the per-question JSON without making a second call.
    Returns
    `(merged_output, locked_ids, l25_ids, l1_ids, bfs_chains)`.
    `bfs_chains[i]` is the edge-type path that reached the i-th
    L3-tier expansion belief in `merged_output` (empty list when
    BFS is off / produced nothing / bfs hits collide with prior
    tiers).

    When `use_type_aware_compression` resolves True (#434 v2.1), the
    pack loops account for L2.5/L1/BFS beliefs at their compressed
    `rendered_tokens` rather than `_belief_tokens(b)`. Locks always
    render verbatim per the strategy table, so locked accounting is
    unchanged. Default-ON since the #769 flip; resolving the flag
    False restores byte-identical pre-compression output.

    When `use_intentional_clustering` resolves True (#436 v2.0), the L1
    score-ranked greedy fill is replaced with `pack_with_clusters` over
    the candidate-induced edge subgraph; locked + L2.5 are pre-included
    unchanged, BFS expansion runs after as before. Since #878 the
    cluster pack accepts a `cost_fn` and composes with
    `use_type_aware_compression`: both arms account in the same
    currency (raw token estimate or compressed `rendered_tokens`).
    """
    global _LAST_TELEMETRY
    # #1143 clock seam: read the wall clock at most once per call.
    # `retrieve_v2` threads its own pinned `now_ts` through here, so a
    # pinned outer call performs no wall-clock read anywhere in the
    # tiered path (γ resolver, expansion gate, L1 meta-resolvers).
    effective_now_ts = now_ts if now_ts is not None else int(time.time())
    enabled = is_entity_index_enabled(entity_index_enabled)
    bfs_on = is_bfs_enabled(bfs_enabled)
    bm25f_on = resolve_use_bm25f_anchors(use_bm25f_anchors)
    weight = resolve_posterior_weight(posterior_weight)
    # #1045 wide-retrieval knobs — same resolution as `retrieve()`.
    l1_limit = resolve_l1_limit(l1_limit)
    token_budget, _self_defaulted = resolve_token_budget_with_provenance(
        token_budget,
    )
    # `retrieve()` resolves the budget before delegating, so by the time
    # the sentinel arrives here it is always a concrete int and the
    # provenance computed above is always False. It passes the fact down
    # explicitly instead; None means "nobody told me, work it out".
    if budget_defaulted is None:
        budget_defaulted = _self_defaulted
    heat_on = is_heat_kernel_enabled(heat_kernel_enabled)
    # #796 γ rerank — same resolution as `retrieve()`.
    gamma_on = resolve_use_gamma_posterior_temperature()
    gamma_t = (
        resolve_posterior_temperature_with_meta(
            store, now_ts=effective_now_ts,
        )
        if gamma_on else None
    )
    # #817 ζ rerank — same resolution as `retrieve()`.
    zeta_on = resolve_use_zeta_posterior_rerank()
    _assert_gamma_zeta_mutual_exclusion(gamma_on, zeta_on)
    zeta_params = (
        (ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT)
        if zeta_on else None
    )
    # #741 adaptive expansion-gate. Same shape as retrieve(): short-
    # circuit BFS on broad prompts; L0 / L1 / L2.5-entity unaffected.
    # #760: pass store + now_ts for the meta-belief token-threshold
    # resolver; same fallback posture as retrieve().
    from aelfrice.expansion_gate import should_run_expansion
    gate_decision = should_run_expansion(
        query, store=store, now_ts=effective_now_ts,
    )
    gate_skipped_bfs = bfs_on and not gate_decision.run_bfs
    bfs_on = bfs_on and gate_decision.run_bfs
    compress_on = resolve_use_type_aware_compression(
        use_type_aware_compression,
    )
    cluster_on = resolve_use_intentional_clustering(
        use_intentional_clustering,
    )
    # #1176 proposal 2. Resolved next to `cluster_on` because it is the
    # alternative answer to the same question, and takes precedence.
    max_coverage_on = is_max_coverage_pack_enabled()
    # #878: compose-reconciliation. The cluster pack reads the same
    # _cost closure the non-cluster L1 path uses; both arms account
    # in the same currency (raw token estimate or compressed
    # rendered_tokens depending on compress_on). The v2.0.0 mutex
    # (clustering + compression → ValueError) is dropped now that the
    # currencies match.
    warn_placeholder_flags()

    def _cost(b: Belief) -> int:
        """Per-belief pack cost. Compressed render when flag ON,
        else raw token estimate. Locks render verbatim either way."""
        if not compress_on:
            return _belief_tokens(b)
        cb = compress_for_retrieval(
            b, locked=(b.lock_level == LOCK_USER),
        )
        return cb.rendered_tokens

    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids_list: list[str] = [b.id for b in locked]
    locked_ids: set[str] = set(locked_ids_list)

    # #1271: the legacy downgrade is for "nobody asked", not "the number
    # happens to be 2400". Keying on the value silently turned an
    # explicit `token_budget=2400` into 2000 while 2399 and 2401 both
    # survived — discontinuous at precisely the default, and it
    # invalidated a measurement on #1269 before it was found.
    effective_budget = (
        token_budget
        if (enabled or not budget_defaulted)
        else LEGACY_TOKEN_BUDGET
    )
    # #1016-B parity with retrieve(): when manifest_reference_locks is on
    # (the hook injection paths), reference-tier locks cost only their
    # one-line manifest entry, freeing relevance budget. Frozen locks (the
    # default) still cost full content, so this is byte-identical unless a
    # lock is demoted to the reference tier.
    locked_used: int = sum(
        lock_injection_tokens(
            b, manifest_reference_locks=manifest_reference_locks
        )
        for b in locked
    )
    # #379 locks are uncapped + never trimmed; reserve a relevance floor so
    # they can't starve L2.5/L1 to zero. No-op (byte-identical) unless locks
    # leave less than the floor — see RELEVANCE_BUDGET_FLOOR_FRACTION.
    relevance_budget: int = max(
        int(effective_budget * RELEVANCE_BUDGET_FLOOR_FRACTION),
        effective_budget - locked_used,
    )
    l25_room: int = max(0, relevance_budget)
    effective_l25_subbudget: int = min(l25_token_subbudget, l25_room)

    if enabled and query.strip():
        l25 = _l25_hits(
            store,
            query,
            locked_ids=locked_ids,
            l25_limit=l25_limit,
            l25_token_subbudget=effective_l25_subbudget,
            query_entity_cap=query_entity_cap,
            use_origin_tiebreak=use_origin_tiebreak,
            use_fan_effect=use_fan_effect,
        )
    else:
        l25 = []
    l25_ids_list: list[str] = [b.id for b in l25]
    l25_ids: set[str] = set(l25_ids_list)

    l1: list[Belief] = []
    # #1162. False when the L1 lane did not run at all, which is also
    # the honest answer: no L1 pass, no heat propagation.
    heat_used = False
    if query.strip():
        raw_l1: list[Belief] = _l1_hits(
            store, query,
            l1_limit=l1_limit, posterior_weight=weight,
            use_bm25f_anchors=bm25f_on, bm25f_cache=bm25f_cache,
            eigenbasis_cache=eigenbasis_cache, heat_kernel_on=heat_on,
            gamma_temperature=gamma_t,
            zeta_params=zeta_params,
            use_entity_persist_demote=use_entity_persist_demote,
            use_origin_tiebreak=use_origin_tiebreak,
            use_supersession_demote=use_supersession_demote,
            supersession_treatment=supersession_treatment,
            supersession_factor=supersession_factor,
            utterance_prior_weight=resolve_utterance_prior_weight(
                utterance_prior_weight,
            ),
            now_ts=effective_now_ts,
        )
        l1 = [
            b for b in raw_l1
            if b.id not in locked_ids and b.id not in l25_ids
        ]
        heat_used = _LAST_HEAT_USED

    used: int = locked_used + sum(_cost(b) for b in l25)
    out: list[Belief] = list(locked) + list(l25)
    l1_ids_list: list[str] = []
    l1_packed: list[Belief] = []
    if max_coverage_on and l1:
        # #1176 proposal 2. Budgeted maximum coverage over query terms,
        # ahead of the cluster branch because the two are alternative
        # answers to the same question and running both would pack twice.
        l1_remaining_budget = max(0, locked_used + relevance_budget - used)
        cov_map, cov_weights = _coverage_inputs(query, l1, bm25f_cache)
        l1_packed = pack_max_coverage(
            l1,
            token_budget=l1_remaining_budget,
            coverage=cov_map,
            term_weights=cov_weights,
            cost_fn=_cost,
        )
        for b in l1_packed:
            out.append(b)
            used += _cost(b)
            l1_ids_list.append(b.id)
    elif cluster_on and l1:
        # Diversity-aware fill over the candidate-induced edge subgraph
        # (#436). Rank-position-as-score: l1 is already sorted descending
        # by the rerank, so position is a monotone proxy for score and
        # only the ordering matters to cluster_candidates / Stage 1.
        cluster_scores: dict[str, float] = {
            b.id: float(len(l1) - i) for i, b in enumerate(l1)
        }
        l1_edges = store.edges_for_beliefs([b.id for b in l1])
        clusters = cluster_candidates(
            l1, cluster_scores, edges=l1_edges,
            edge_weight_floor=DEFAULT_CLUSTER_EDGE_FLOOR,
        )
        l1_by_id: dict[str, Belief] = {b.id: b for b in l1}
        l1_remaining_budget = max(0, locked_used + relevance_budget - used)
        l1_packed = pack_with_clusters(
            clusters, l1_by_id,
            token_budget=l1_remaining_budget,
            cluster_diversity_target=DEFAULT_CLUSTER_DIVERSITY_TARGET,
            cost_fn=_cost,
        )
        for b in l1_packed:
            out.append(b)
            used += _cost(b)
            l1_ids_list.append(b.id)
    else:
        for b in l1:
            cost: int = _cost(b)
            if used + cost > locked_used + relevance_budget:
                break
            out.append(b)
            l1_packed.append(b)
            used += cost
            l1_ids_list.append(b.id)

    # #981 HRR vocabulary-bridge expansion lane (additive, default-OFF).
    # Probes the shared struct index for single-hop semantic neighbours of
    # the FTS5 seeds (the packed L1 hits) and merges them into the candidate
    # set before BFS. On a miss — flag off (the default), no index cache, no
    # seeds, or an empty index — it is a no-op and the output is byte-
    # identical to the pre-#981 path. The lane is deterministic (numpy FFT /
    # matvec over the seeded struct index; no random / betavariate).
    hrr_expand_ids_list: list[str] = []
    hrr_expanded: list[Belief] = []
    if (
        is_hrr_expand_enabled(hrr_expand_enabled)
        and query.strip()
        and hrr_struct_index_cache is not None
        and l1_packed
    ):
        from aelfrice.hrr_expand import expand_seeds

        seen_pre: set[str] = locked_ids | l25_ids | set(l1_ids_list)
        for b in expand_seeds(
            store,
            hrr_struct_index_cache.get(),
            [b.id for b in l1_packed],
        ):
            if b.id in seen_pre:
                continue
            cost = _cost(b)
            if used + cost > locked_used + relevance_budget:
                break
            out.append(b)
            hrr_expanded.append(b)
            hrr_expand_ids_list.append(b.id)
            seen_pre.add(b.id)
            used += cost

    # #1064 temporal-spine lane (additive, default-ON since #1107 Phase 2).
    # Traverses TEMPORAL_NEXT chains from the top-5 packed L1 seeds, both
    # directions, depth 1 by default, and appends the neighbours after
    # the L1 candidates — never displacing them pre-packing. No-op
    # guard: a store with zero TEMPORAL_NEXT edges skips the traversal
    # entirely (LIMIT-1 existence probe), so spineless stores get byte-
    # identical output at ~zero cost. Spine hits deliberately do NOT
    # seed the BFS expansion below: the #1064 confirmatory evidence
    # measured the lane as a depth-1 append-after-L1 source with BFS
    # untouched, and feeding BFS is unmeasured surface (#977 keeps
    # generic BFS off anyway).
    temporal_spine_ids_list: list[str] = []
    n_spine_candidates = 0
    if (
        is_temporal_spine_enabled(temporal_spine_enabled)
        and query.strip()
        and l1_packed
        and store.has_edge_type(EDGE_TEMPORAL_NEXT)
    ):
        from aelfrice.temporal_spine import (
            DEFAULT_SPINE_DEPTH,
            DEFAULT_SPINE_SEED_COUNT,
            spine_neighbors,
        )

        spine_hits = spine_neighbors(
            store,
            [b.id for b in l1_packed[:DEFAULT_SPINE_SEED_COUNT]],
            depth=(
                temporal_spine_depth if temporal_spine_depth is not None
                else DEFAULT_SPINE_DEPTH
            ),
            node_budget=resolve_temporal_spine_budget(
                temporal_spine_node_budget,
            ),
        )
        n_spine_candidates = len(spine_hits)
        seen_spine: set[str] = (
            locked_ids | l25_ids | set(l1_ids_list)
            | set(hrr_expand_ids_list)
        )
        for b in spine_hits:
            if b.id in seen_spine:
                continue
            cost = _cost(b)
            if used + cost > locked_used + relevance_budget:
                break
            out.append(b)
            temporal_spine_ids_list.append(b.id)
            seen_spine.add(b.id)
            used += cost

    bfs_chains: list[list[str]] = []
    if bfs_on and query.strip():
        seeds: list[Belief] = (
            list(locked) + list(l25) + list(l1_packed) + list(hrr_expanded)
        )
        if seeds:
            hops = expand_bfs(
                seeds,
                store,
                max_depth=bfs_max_depth,
                nodes_per_hop=bfs_nodes_per_hop,
                total_budget=bfs_total_budget_nodes,
                min_path_score=bfs_min_path_score,
            )
            seen_ids: set[str] = (
                locked_ids | l25_ids | set(l1_ids_list)
                | set(hrr_expand_ids_list)
                | set(temporal_spine_ids_list)
            )
            for hop in hops:
                if hop.belief.id in seen_ids:
                    continue
                cost = _cost(hop.belief)
                if used + cost > locked_used + relevance_budget:
                    break
                out.append(hop.belief)
                bfs_chains.append(list(hop.path))
                seen_ids.add(hop.belief.id)
                used += cost
    _LAST_TELEMETRY = LaneTelemetry(
        locked=len(locked_ids_list),
        l25=len(l25_ids_list),
        l1=len(l1_ids_list),
        bfs=len(bfs_chains),
        bm25f_used=bm25f_on,
        posterior_weight=weight,
        expansion_gate_reason=gate_decision.reason,
        expansion_gate_skipped_bfs=gate_skipped_bfs,
        l1_candidates=len(l1),
        hrr_expand=len(hrr_expand_ids_list),
        temporal_spine=len(temporal_spine_ids_list),
        temporal_spine_candidates=n_spine_candidates,
        heat_used=heat_used,
    )
    return out, locked_ids_list, l25_ids_list, l1_ids_list, bfs_chains


@_memoize_config_discovery
def retrieve_v2(
    store: MemoryStore,
    query: str,
    budget: int | None = None,
    include_locked: bool = True,
    use_bfs: bool | None = None,
    use_entity_index: bool | None = None,
    l1_limit: int | None = None,
    bfs_max_depth: int = BFS_DEFAULT_MAX_DEPTH,
    bfs_nodes_per_hop: int = BFS_DEFAULT_NODES_PER_HOP,
    bfs_total_budget_nodes: int = BFS_DEFAULT_TOTAL_BUDGET_NODES,
    bfs_min_path_score: float = BFS_DEFAULT_MIN_PATH_SCORE,
    posterior_weight: float | None = None,
    use_bm25f: bool | None = None,
    bm25f_cache: BM25IndexCache | None = None,
    temporal_sort: bool = False,
    temporal_half_life_seconds: float | None = None,
    use_type_aware_compression: bool | None = None,
    use_intentional_clustering: bool | None = None,
    use_hrr_structural: bool | None = None,
    use_hrr_expand: bool | None = None,
    use_entity_persist_demote: bool | None = None,
    use_origin_tiebreak: bool | None = None,
    use_fan_effect: bool | None = None,
    use_supersession_demote: bool | None = None,
    supersession_treatment: str | None = None,
    supersession_factor: float | None = None,
    utterance_prior_weight: float | None = None,
    use_temporal_spine: bool | None = None,
    temporal_spine_depth: int | None = None,
    temporal_spine_node_budget: int | None = None,
    hrr_struct_index_cache: HRRStructIndexCache | None = None,
    with_doc_anchors: bool = False,
    manifest_reference_locks: bool = False,
    # #1107 shim pass-throughs: params `retrieve()` exposes that route
    # straight to `retrieve_with_tiers`, added so `retrieve()` can delegate
    # to `retrieve_v2` without dropping caller overrides. Default to the
    # same values `retrieve_with_tiers` uses, so omitting them is a no-op.
    l25_limit: int = DEFAULT_L25_LIMIT,
    l25_token_subbudget: int = DEFAULT_L25_TOKEN_SUBBUDGET,
    query_entity_cap: int = DEFAULT_QUERY_ENTITY_CAP,
    heat_kernel_enabled: bool | None = None,
    eigenbasis_cache: GraphEigenbasisCache | None = None,
    now_ts: int | None = None,
    now: datetime | None = None,
) -> RetrievalResult:
    """Lab-compatible retrieval wrapper for academic-suite adapters.

    Wraps the public `retrieve()` in the signature lab v2.0.0 adapters
    expect:

    - `budget` (lab kwarg) maps to `token_budget` (public kwarg).
    - `include_locked=False` filters out lock_level != LOCK_NONE post-retrieval
      (public always returns L0 first; this wrapper drops them on demand).
    - `use_bfs` (v1.3.0) maps to `retrieve()`'s `bfs_enabled` kwarg.
      None falls through to the default-OFF resolution (env / TOML
      / False at v1.3.0). Setting it True opts a single retrieve_v2
      call into BFS regardless of process-wide config.
    - `use_entity_index` (v1.3.0) maps to `retrieve()`'s
      `entity_index_enabled` kwarg. None falls through to the default
      (env / TOML / True). The v1.3.0 benchmark adapter sets it
      explicitly.
    - `bfs_max_depth`, `bfs_nodes_per_hop`, `bfs_total_budget_nodes`,
      `bfs_min_path_score` — pass-through tuning knobs for the L3
      tier. Defaults match `bfs_multihop.DEFAULT_*`.
    - `temporal_sort` (v2.1 #473) — when True, applies a multiplicative
      half-life decay to the merged output as a final re-rank pass
      (post-BM25F + posterior + edge-type rerank). Default False keeps
      the v1.3-v1.7 ordering byte-identical for adapters that don't opt
      in. Locked (L0) beliefs are pinned at the head and never re-
      ordered. The decay is `2 ** (-age / half_life)` against
      `created_at`. Half-life resolution: explicit kwarg →
      `AELFRICE_TEMPORAL_HALF_LIFE_SECONDS` env →
      `[retrieval] temporal_half_life_seconds` in `.aelfrice.toml` →
      `DEFAULT_TEMPORAL_HALF_LIFE_SECONDS` (7 days, ratified per
      issue #473 A1=A).
    - `temporal_half_life_seconds` (v2.1 #473) — explicit override for
      the half-life when `temporal_sort=True`. None falls through to
      `resolve_temporal_half_life_with_meta()`'s precedence chain
      (env → kwarg → TOML → meta-belief → default-7d). Ignored when
      `temporal_sort=False`.
    - `now_ts` (#756) — UTC seconds, used by the meta-belief consumer
      to time-stamp posterior decay and the latency-signal update.
      Defaults to ``int(time.time())`` so production callers don't
      need to pass it. Tests and replay tooling pin it for
      determinism (locked ``c06f8d575fad71fb`` PHILOSOPHY). Since
      #1143 the pin threads through `retrieve_with_tiers` and
      `_l1_hits`, so a pinned call reads no wall clock anywhere in
      the tiered path.
    - `use_hrr_structural` (#152) — when True AND the query parses as
      a `<KIND>:<target_id>` structural marker, the HRR structural
      lane fires and returns instead of the textual lane. Parallel,
      not blended (per spec): on marker hit the BM25F + heat-kernel
      stack is bypassed entirely; on miss the call falls through to
      the textual lane unchanged. Default-ON post #154 composition
      tracker — the #437 reproducibility-harness gate cleared at
      11/11 and `is_hrr_structural_enabled()` resolves to True when
      no env / kwarg / TOML override is set. Opt out via
      `AELFRICE_HRR_STRUCTURAL=0`, `use_hrr_structural=False`, or
      `[retrieval] use_hrr_structural = false` in `.aelfrice.toml`.
    - `use_hrr_expand` (#981) — when True, the HRR vocabulary-bridge
      *expansion* lane runs as an additive candidate source: it probes
      the struct index for single-hop semantic neighbours of the FTS5
      seeds and merges them into the candidate set before BFS. Distinct
      from `use_hrr_structural` (a marker-routed replacement lane).
      Default-OFF — #981 lands the lane + ablation only; a default flip
      reverses locked #605 and is routed to a re-opened #897. Opt in via
      `AELFRICE_HRR_EXPAND=1`, the kwarg, or
      `[retrieval] use_hrr_expand = true`. The lane is a no-op (byte-
      identical output) unless a `hrr_struct_index_cache` is available;
      when the flag is on and no cache is passed, an ephemeral in-memory
      cache is built so the lane still fires (callers running the lane
      hot should pass an explicit cache to amortise the build).
    - `use_temporal_spine` (#1064) — when True, the temporal-spine lane
      runs as an additive candidate source after L1: it traverses
      TEMPORAL_NEXT chains from the top-5 packed L1 seeds (both
      directions, depth 1 by default, node budget 32 by default) and
      appends the chronological neighbours to the candidate set. The
      lane is a no-op (byte-identical output) when the store has zero
      TEMPORAL_NEXT edges. Default-ON since the #1107 Phase-2 cutover
      (every pre-registered #1064 gate cleared). Opt out via
      `AELFRICE_TEMPORAL_SPINE=0`, the kwarg, or
      `[retrieval] use_temporal_spine = false`.
      `temporal_spine_depth` / `temporal_spine_node_budget` tune the
      traversal (budget also via `AELFRICE_TEMPORAL_SPINE_BUDGET` env
      or `[retrieval] temporal_spine_budget` TOML).
    - `hrr_struct_index_cache` (#152) — explicit
      `HRRStructIndexCache` to reuse an already-built index across
      calls. None falls through to a fresh build per call.
      Long-running consumers (interactive shells, bench harnesses)
      should pass an explicit cache to amortise the per-belief HRR
      encode cost.
    - Returns a `RetrievalResult` wrapper so adapters can read
      `result.beliefs` (and stub diagnostics fields, plus the new
      v1.3 `entity_hits` and `bfs_chains`).
    """
    # #1045 wide-retrieval knobs: resolve here so both the HRR-structural
    # lane below and the retrieve_with_tiers delegation see the same
    # env/TOML-resolved values (default 50 / 2400 keep the hot path narrow).
    l1_limit = resolve_l1_limit(l1_limit)
    budget, budget_defaulted = resolve_token_budget_with_provenance(budget)
    # v2.1 #152 HRR structural-query routing. Returns early on marker
    # hit; falls through on miss (non-marker query, marker-with-
    # unknown-target, or flag OFF) so the textual lane handles the
    # call.
    if is_hrr_structural_enabled(use_hrr_structural):
        struct_result = _route_structural_query(
            store, query, hrr_struct_index_cache,
            top_k=l1_limit,
            include_locked=include_locked,
            budget=budget,
        )
        if struct_result is not None:
            return struct_result

    effective_now_ts = now_ts if now_ts is not None else int(time.time())

    # #759 BFS depth-budget resolver. Fires before retrieve_with_tiers so
    # the effective max_depth is locked in before the BFS expansion runs.
    # The explicit clause: only pass the caller's bfs_max_depth when it
    # deviates from the default — a caller that passes the default
    # deliberately shouldn't suppress the meta-belief layer.
    bfs_max_depth = resolve_bfs_depth_budget_with_meta(
        store,
        now_ts=effective_now_ts,
        explicit=(
            bfs_max_depth if bfs_max_depth != BFS_DEFAULT_MAX_DEPTH else None
        ),
    )

    # #981 expansion-lane index cache. When the lane resolves ON but no cache
    # was passed, build an ephemeral in-memory cache so the lane still fires
    # (no persistence — store_path is not threaded into this wrapper). When
    # the lane is OFF (the default) the cache stays None and the build is
    # skipped entirely, so the default path is unchanged.
    expand_cache = hrr_struct_index_cache
    if is_hrr_expand_enabled(use_hrr_expand) and expand_cache is None:
        expand_cache = HRRStructIndexCache(store=store)

    retrieve_start = time.perf_counter()
    (
        out,
        locked_ids_list,
        l25_ids_list,
        l1_ids_list,
        bfs_chains,
    ) = retrieve_with_tiers(
        store, query,
        token_budget=budget,
        budget_defaulted=budget_defaulted,
        entity_index_enabled=use_entity_index,
        l25_limit=l25_limit,
        l25_token_subbudget=l25_token_subbudget,
        query_entity_cap=query_entity_cap,
        heat_kernel_enabled=heat_kernel_enabled,
        eigenbasis_cache=eigenbasis_cache,
        l1_limit=l1_limit,
        bfs_enabled=use_bfs,
        bfs_max_depth=bfs_max_depth,
        bfs_nodes_per_hop=bfs_nodes_per_hop,
        bfs_total_budget_nodes=bfs_total_budget_nodes,
        bfs_min_path_score=bfs_min_path_score,
        posterior_weight=posterior_weight,
        use_bm25f_anchors=use_bm25f,
        bm25f_cache=bm25f_cache,
        now_ts=effective_now_ts,
        use_type_aware_compression=use_type_aware_compression,
        use_intentional_clustering=use_intentional_clustering,
        hrr_expand_enabled=use_hrr_expand,
        use_entity_persist_demote=is_entity_persist_demote_enabled(
            use_entity_persist_demote
        ),
        use_origin_tiebreak=is_origin_tiebreak_enabled(use_origin_tiebreak),
        use_fan_effect=is_fan_effect_enabled(use_fan_effect),
        use_supersession_demote=is_supersession_demote_enabled(
            use_supersession_demote
        ),
        supersession_treatment=resolve_supersession_treatment(
            supersession_treatment
        ),
        supersession_factor=resolve_supersession_factor(supersession_factor),
        utterance_prior_weight=utterance_prior_weight,
        manifest_reference_locks=manifest_reference_locks,
        hrr_struct_index_cache=expand_cache,
        temporal_spine_enabled=use_temporal_spine,
        temporal_spine_depth=temporal_spine_depth,
        temporal_spine_node_budget=temporal_spine_node_budget,
    )
    retrieve_elapsed = time.perf_counter() - retrieve_start
    if include_locked:
        beliefs = out
    else:
        beliefs = [b for b in out if b.lock_level == LOCK_NONE]
    if temporal_sort:
        half_life = resolve_temporal_half_life_with_meta(
            store,
            now_ts=effective_now_ts,
            explicit=temporal_half_life_seconds,
        )
        # `now` pins the temporal-decay clock so retrieve_v2-level temporal
        # tests are deterministic; default None -> _apply_temporal_decay uses
        # datetime.now(timezone.utc), leaving the production path unchanged.
        # Distinct from `now_ts`, which pins the meta-belief *resolver* clock
        # (half-life / BFS-depth resolvers). (#986)
        beliefs = _apply_temporal_decay(beliefs, half_life, now=now)
        # #756 latency-signal update. Only fires when the meta-belief
        # feature flag is on; the env check matches the resolver above
        # so the read/write paths flip together. update_meta_belief is
        # a no-op when the meta-belief isn't installed (returns False),
        # so this is safe without an explicit existence check. Fail-
        # soft: a store error here must never break retrieval — that
        # invariant mirrors the deferred-feedback enqueue at the end
        # of `retrieve()`.
        if is_meta_belief_half_life_enabled() and retrieve_elapsed > 0.0:
            try:
                from aelfrice.meta_beliefs import SIGNAL_LATENCY
                evidence = max(0.0, min(
                    1.0, LATENCY_TARGET_SECONDS / retrieve_elapsed,
                ))
                store.update_meta_belief(
                    META_HALF_LIFE_KEY,
                    SIGNAL_LATENCY,
                    evidence=evidence,
                    now_ts=effective_now_ts,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "aelfrice retrieval: meta-belief latency update "
                    f"failed: {exc}",
                    file=sys.stderr,
                )

    # #759 latency-signal update for bfs_depth_budget meta-belief. This
    # is a SECOND meta-belief update, independent of the #756 half-life
    # update above. Both fire on every retrieve_v2 call when their
    # respective flags are on. Same evidence formula and fail-soft
    # try/except posture as #756: store errors print to stderr and are
    # swallowed; retrieval must never raise on a meta-belief write failure.
    if is_meta_belief_bfs_depth_budget_enabled() and retrieve_elapsed > 0.0:
        try:
            from aelfrice.meta_beliefs import SIGNAL_LATENCY as _SIGNAL_LATENCY
            evidence = max(0.0, min(
                1.0, LATENCY_TARGET_SECONDS / retrieve_elapsed,
            ))
            store.update_meta_belief(
                META_BFS_DEPTH_BUDGET_KEY,
                _SIGNAL_LATENCY,
                evidence=evidence,
                now_ts=effective_now_ts,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "aelfrice retrieval: meta-belief bfs_depth_budget latency "
                f"update failed: {exc}",
                file=sys.stderr,
            )

    compressed: list[CompressedBelief] = []
    if resolve_use_type_aware_compression(use_type_aware_compression):
        compressed = [
            compress_for_retrieval(b, locked=(b.lock_level == LOCK_USER))
            for b in beliefs
        ]

    # #435 doc-linker post-rank, pre-pack projection. Default OFF keeps
    # the adapter wire bytes-identical for callers that don't opt in.
    # When ON, one batched `belief_id IN (...)` SELECT joins anchors
    # onto the result. Anchors are metadata for the consumer; they do
    # NOT count against the token budget pack.
    doc_anchors_list: list[list[DocAnchor]] = []
    if with_doc_anchors and beliefs:
        anchors_by_id = store.get_doc_anchors_batch([b.id for b in beliefs])
        doc_anchors_list = [anchors_by_id.get(b.id, []) for b in beliefs]

    return RetrievalResult(
        beliefs=beliefs,
        entity_hits=l25_ids_list,
        locked_ids=locked_ids_list,
        l1_ids=l1_ids_list,
        bfs_chains=bfs_chains,
        compressed_beliefs=compressed,
        doc_anchors=doc_anchors_list,
    )


class RetrievalCache:
    """Bounded LRU cache wrapping `retrieve()` for an attached store.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge / entity-row mutation wipes
    the cache. Per-instance: two `RetrievalCache` objects pointing
    at different stores never share state.

    Cache key includes the entity-index flag (v1.3.0 default-on),
    the BFS flag (v1.3.0 default-off), `use_bm25f_anchors`
    (v1.7.0 default-on per #154), and `posterior_weight`
    (v1.3.0 default 0.5, rounded to `POSTERIOR_WEIGHT_KEY_PRECISION`
    decimals so floating-point jitter does not fragment the cache).
    Two queries that differ in any of these are distinct entries.
    BFS knobs (`bfs_max_depth` etc.) are NOT in the key — per
    docs/design/bfs_multihop.md § Cache invalidation, callers that toggle
    them per call would defeat the cache anyway.

    The `posterior_weight` cache-key extension is a structural fix
    against cross-caller collisions per docs/design/bayesian_ranking.md §
    "Cache invalidation". Posterior-write staleness is handled by
    the existing store-mutation callback (apply_feedback ->
    update_belief -> _fire_invalidation -> cache wipe).
    """

    def __init__(
        self,
        store: MemoryStore,
        capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._store = store
        self._capacity = capacity
        self._entries: OrderedDict[
            tuple[
                str, int, int,
                bool | None, bool | None, float | None, bool | None,
            ],
            list[Belief],
        ] = OrderedDict()
        store.add_invalidation_callback(self.invalidate)

    def retrieve(
        self,
        query: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        l1_limit: int = DEFAULT_L1_LIMIT,
        *,
        entity_index_enabled: bool | None = None,
        bfs_enabled: bool | None = None,
        posterior_weight: float | None = None,
        use_bm25f_anchors: bool | None = None,
        bm25f_cache: BM25IndexCache | None = None,
    ) -> list[Belief]:
        """Cached `retrieve()`. Same returned beliefs as the free
        function; **not** its side effects on a hit (#1144).

        A cache hit returns the memoized beliefs without re-running the
        pipeline, so it also does not repeat `retrieve()`'s
        retrieval-exposure enqueue (#191): exposure is recorded once, on
        the miss that populates the entry, not on every subsequent hit.
        This is deliberate — the enqueue is a `deferred_feedback_queue`
        write, and a DB write per hit would blow the AC2 cache-hit
        latency budget (≤ 50 µs; see `test_ac2_cache_hit_under_fifty_microseconds`).
        Exposure is audit-only recurrence data (#1086), never evidence,
        so under-counting repeat hits of a cached query is a bounded,
        documented approximation. The production hook path calls the free
        `retrieve()` directly (no cache), so its exposure stream — the one
        the #1086 recurrence consumer reads — is unaffected; this class
        has no production call site.

        Cache key keeps `posterior_weight` in its caller-supplied
        form (None or a float) — `None` is its own bucket and
        deferred env / TOML resolution happens once on the miss
        path. Resolving on every hit would walk Path.cwd().resolve()
        each time and blow the same AC2 cache-hit latency budget.
        """
        if posterior_weight is None:
            key_weight: float | None = None
        else:
            key_weight = round(
                float(posterior_weight),
                POSTERIOR_WEIGHT_KEY_PRECISION,
            )
        key = (
            canonicalize_query(query),
            token_budget,
            l1_limit,
            entity_index_enabled,
            bfs_enabled,
            key_weight,
            use_bm25f_anchors,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return list(cached)
        result = retrieve(
            self._store, query,
            token_budget=token_budget, l1_limit=l1_limit,
            entity_index_enabled=entity_index_enabled,
            bfs_enabled=bfs_enabled,
            posterior_weight=posterior_weight,
            use_bm25f_anchors=use_bm25f_anchors,
            bm25f_cache=bm25f_cache,
        )
        self._entries[key] = list(result)
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return result

    def invalidate(self) -> None:
        """Drop every cached entry. Wired to the store's mutation hook."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
