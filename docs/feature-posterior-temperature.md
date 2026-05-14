# Feature spec: γ rerank — Boltzmann posterior temperature (#796)

**Status:** implemented behind default-OFF flag at v3.x; bench panel widened (`ordered_top_k_overlap`, `rank_biased_overlap`); adoption verdict (flip default) **deferred** until a labeled relevance corpus exists.
**Issue:** #796
**Substrate prereqs:** #756 / #757 / #759 / #760 (meta-belief consumer pattern, shipped v3.x). Operator path-B decision 2026-05-14T16:48Z, refined to γ″ after the R&D campaign 2026-05-14T18:04Z.

---

## Purpose

The v1.3 retrieval rerank is **log-additive**:

```
score = log(max(-bm25_raw, EPS)) + posterior_weight · log(posterior_mean)
```

`posterior_weight` is a process-wide scalar tuned at v1.3.0 (default `0.5`). Two independent operator concerns motivate the γ surface:

1. The intended consumer for `meta:retrieval.posterior_temperature` (a softmax / Boltzmann temperature `T`) does not exist on `github/main` — pre-claim analysis on #758 confirmed `git grep -E 'softmax|boltzmann|temperature' github/main -- 'src/aelfrice/'` returns zero hits in src. Issue #796 is the load-bearing precursor: ship the temperature surface so `meta:retrieval.posterior_temperature` has something to bind to.
2. PR@5 + Spearman ρ alone cannot discriminate top-K reorderings from middle-of-list churn (#796 R4). The bench panel needs `ordered_top_k_overlap` and `rank_biased_overlap` to make γ-vs-log-additive comparisons load-bearing.

γ is **reparametrised log-additive**, not a new family — see *Contract* below. At `T = 1.0` it is byte-identical to `partial_bayesian_score(..., posterior_weight=1.0)`. Lower `T` sharpens the posterior contribution; higher `T` flattens toward BM25-only ranking.

---

## Contract

```python
from aelfrice.scoring import gamma_posterior_score

score: float = gamma_posterior_score(
    bm25_raw, alpha, beta, temperature,
)
```

Formula:

```
score = log(max(-bm25_raw, EPS)) + (1 / T) · log(posterior_mean(α, β))
```

Equivalent to `partial_bayesian_score(bm25_raw, α, β, posterior_weight=1.0/T)` exactly — γ is a reparametrisation, not a new function family. Pure, deterministic, no store reads, no clock reads.

`T <= GAMMA_TEMPERATURE_FLOOR` (1e-6) clamps upward so a misconfigured meta-belief or env override never raises at retrieval time. Negative temperatures are likewise clamped (the Boltzmann reading is undefined for `T <= 0`).

---

## Flag + meta-belief

Resolved at `retrieve()` / `retrieve_with_tiers()` entry, once per call.

| Layer | Surface | Resolver |
|---|---|---|
| Env | `AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE` | `_env_use_gamma_posterior_temperature_override()` |
| TOML | `[retrieval] use_gamma_posterior_temperature` | `_read_toml_flag_for(...)` |
| Default | False | `resolve_use_gamma_posterior_temperature()` |

When the flag resolves True, the temperature is resolved against the meta-belief substrate:

```python
T = resolve_posterior_temperature_with_meta(store, now_ts=...)
```

Bounds: `T ∈ [POSTERIOR_TEMPERATURE_FLOOR, POSTERIOR_TEMPERATURE_CEIL] = [0.5, 2.0]`. Log-linear decode from the meta-belief's `[0, 1]` posterior value; geometric mean is exactly 1.0, so the cold-start `static_default = 0.5` decodes to `T = 1.0` and a fresh install with the flag on is byte-identical to `partial_bayesian_score(..., 1.0)`. Adaptive learning of `T` (the evidence-signal loop that moves the meta-belief away from its prior) is out of scope for #796 — that is issue #758.

When the flag is False, `gamma_temperature` is `None` and `_l1_hits` skips the γ branch entirely. The pre-#796 log-additive contract holds byte-for-byte.

### Heat-rerank composition

γ and the heat-kernel rerank (`use_heat_kernel`) are **mutually exclusive** on a given call. When both flags are on and a non-stale eigenbasis is available, the heat-rerank fires and γ is a no-op for that call. Composition was deferred to a later issue by the operator decision (R&D campaign verdict, 2026-05-14T18:04Z).

---

## Where γ sits

`src/aelfrice/retrieval.py::_l1_hits` — the rerank loop, both branches:

```python
elif gamma_temperature is not None:
    s = gamma_posterior_score(
        bm25_raw, b.alpha, b.beta, gamma_temperature,
    )
else:
    s = partial_bayesian_score(
        bm25_raw, b.alpha, b.beta, posterior_weight,
    )
s = _hash_n_boosted(s, b.content, hash_n_literals)
```

`_hash_n_boost` runs after the rerank, unchanged from the log-additive path — the R2 / R2b finding that path-B-literal divergence was entirely the boost interaction is informational, not a code change.

The byte-identical short-circuit (`posterior_weight == 0.0 and not heat_active and not hash_n_literals`) extends to require `gamma_temperature is None` so γ-on always exercises the rerank loop.

---

## Bench-gate / ship-or-defer policy

| Gate | Status | Notes |
|---|---|---|
| **G1** — surface lands behind a default-OFF flag | shipped | this PR |
| **G2** — bench panel widened with `ordered_top_k_overlap` + `rank_biased_overlap` | shipped | `src/aelfrice/calibration_metrics.py`, `src/aelfrice/eval_harness.py::compare_ranking_panel` |
| **G3** — labeled relevance corpus exists | **pending** | corpus authoring tracked separately |
| **G4** — γ@T=1.0 vs γ@T=0.5 / T=2.0 on labeled corpus shows discriminable rank-overlap deltas | **pending G3** | adoption verdict gate |
| **G5** — flip default to True if G4 clears with effect size ≥ 1σ | **pending G3, G4** | follow-up PR |

The bench-gate / ship-or-defer policy is the same shape as `feature-type-aware-compression.md` § *"Bench-gate / ship-or-defer policy"* and `feature-bfs-multihop.md` § *"Bench-gate posture"*. Until G3 lands, the flag is off and γ is plumbing — no behavioural change on any default code path.

---

## Out of scope (separate issues)

- **Adaptive `T`** — the evidence-signal loop that moves the meta-belief away from its `static_default = 0.5` prior. That is #758. Until #758 ships, the meta-belief is never updated; flag-on cold installs decode to `T = 1.0` and stay there.
- **ζ follow-up** — a bounded / sigmoid posterior-contribution parametrisation that gives `T` subtler dynamics than γ's global re-weighting. Filed as a separate R&D campaign after R&D refuted the α/β/γ trichotomy (see #800).
- **Composition with heat-rerank** — both flags can be on but γ is a no-op on heat-active calls. Composing the two scoring paths is a separate scoping decision.
- **Composition with `_hash_n_boost`** — boost interaction is informational, not a code change. R2 / R2b finding.

---

## Refs

- #796 — this issue (operator path-B decision 2026-05-14T16:48Z; γ″ refinement 2026-05-14T18:04Z).
- #758 — adaptive `T` follow-up; gated on #796 shipping.
- #800 — ζ parametrisation R&D campaign.
- #605 — PHILOSOPHY (deterministic, narrow surface). γ inherits.
- #661 — federation read-only; meta-belief is local-only write state.
- `src/aelfrice/scoring.py:gamma_posterior_score` — entry point.
- `src/aelfrice/retrieval.py:resolve_use_gamma_posterior_temperature` — flag resolver.
- `src/aelfrice/retrieval.py:resolve_posterior_temperature_with_meta` — decoder.
- `src/aelfrice/retrieval.py:_l1_hits` — call site.
- `src/aelfrice/calibration_metrics.py:ordered_top_k_overlap`, `rank_biased_overlap` — bench primitives.
- `src/aelfrice/eval_harness.py:compare_ranking_panel` — bench-panel aggregator.
- `tests/test_scoring_gamma.py`, `tests/test_retrieve_gamma_flag.py`, `tests/test_rank_overlap_metrics.py` — contract tests.
