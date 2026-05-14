# Feature spec: ζ rerank — bounded sigmoid posterior contribution (#817 / #800)

**Status:** implemented behind default-OFF flag at v3.x; flip-default deferred until a labeled relevance corpus exists (same gate as γ — see `feature-posterior-temperature.md`).
**Issues:** [#817](https://github.com/robotrocketscience/aelfrice/issues/817) (implementation), [#800](https://github.com/robotrocketscience/aelfrice/issues/800) (R&D campaign, closed by this PR).
**R&D verdict:** ADOPT at (α=1.0, β=0.25, scale=14.5), per R0–R4 campaign at `experiments/zeta-posterior/` (lab-side). R2 head-to-head: ζ dominates γ on `rank_biased_overlap` for similar `rank_changed_fraction`.

---

## Purpose

γ (`gamma_posterior_score`, #796) shipped a Boltzmann reparametrisation of v1.3's log-additive rerank:

```
γ:  score = log(max(-bm25_raw, EPS)) + (1 / T) · log(posterior_mean)
```

At moderate `T`, γ produces **global re-weighting** — measured `rank_changed_fraction` ≈ 0.92–0.96, ~96% of beliefs change rank. The #800 operator decision was that this is "more aggressive than the intended semantics" of `meta:retrieval.posterior_temperature` (#758).

ζ replaces γ's unbounded `(1/T)·log(p)` with a **bounded sigmoid contribution**:

```
ζ:  score = log(max(-bm25_raw, EPS))
          + α · (σ(β · (log(p) − log(0.5))) − 0.5) · scale
```

Where `σ(x) = 1 / (1 + exp(-x))` is the logistic. The posterior contribution lies strictly in `(-α·scale/2, +α·scale/2)`, centred so `posterior_mean = 0.5` (no evidence) contributes nothing. No belief gets unboundedly leveraged by extreme posteriors — the structural difference from γ.

ζ is **not** a continuous extension of γ. At the pinned defaults `(α=1, β=0.25, scale=14.5)`, ζ is rank-equivalent to log-BM25 alone on a uniform-posterior=0.5 store (both contributions = 0) but otherwise produces a different score sequence. The note at issue #817 § "Note re: cold-start byte-identity" makes this explicit; the unit tests in `tests/test_scoring_zeta.py::test_zeta_not_byte_identical_to_gamma_at_T_one` enforce it.

---

## Contract

```python
from aelfrice.scoring import (
    ZETA_ALPHA_DEFAULT,
    ZETA_BETA_DEFAULT,
    ZETA_SCALE_DEFAULT,
    zeta_posterior_score,
)

score: float = zeta_posterior_score(
    bm25_raw,
    alpha,    # ζ magnitude — defaults to 1.0
    beta,     # ζ sharpness — defaults to 0.25
    scale,    # ζ global multiplier — defaults to 14.5
    posterior_mean,  # the Beta-Bernoulli posterior mean (already computed)
)
```

Pure-function, deterministic, no store reads, no clock reads. `posterior_mean ≤ ZETA_POSTERIOR_FLOOR` clamps upward — a corrupted store row (degenerate posterior) never raises `math domain error` at retrieval time. Negative posteriors clamp the same way (the sigmoid is undefined as `p → 0`).

The `(alpha, beta, scale)` arguments are the **ζ math parameters**, not the Beta-Bernoulli `(α, β)` of the originating belief. The caller computes `posterior_mean` externally (typically `posterior_mean(b.alpha, b.beta)`) and passes the scalar. This is the structural difference from `gamma_posterior_score`, which takes the Beta-Bernoulli `(α, β)` directly and computes the posterior internally.

### Bounded property

For any (`α`, `β`, `scale`) and any `posterior_mean ∈ (0, 1)`:

```
-α·scale/2 < (score - log(max(-bm25, EPS))) < +α·scale/2
```

At the pinned defaults, the contribution is bounded by ±7.25 — comparable in magnitude to a strong BM25 spread but unable to dwarf it on any single belief. The bound is exact in the limit; the open interval holds at every finite posterior. Tests: `test_sigma_bound`, `test_sigma_bound_at_extreme_posteriors`.

### Posterior-neutral point

At `posterior_mean = 0.5`:

```
log(p) − log(0.5) = 0
σ(β · 0) = 0.5
(σ − 0.5) · α · scale = 0
score = log(max(-bm25, EPS))     # identical to BM25-only ordering
```

A store of all-uniform posteriors is rank-equivalent to log-BM25 alone — the same edge-case property `partial_bayesian_score` has at `posterior_weight = 0.0`. Tests: `test_posterior_half_is_log_bm_only`.

---

## Flag

Resolved at `retrieve()` / `retrieve_with_tiers()` entry, once per call. Five-path precedence (first decisive wins):

| Layer | Surface | Resolver |
|---|---|---|
| Env | `AELFRICE_USE_ZETA_POSTERIOR_RERANK` | `_env_use_zeta_posterior_rerank_override()` |
| Kwarg | `explicit=True/False` to `resolve_use_zeta_posterior_rerank` | — |
| TOML | `[retrieval] use_zeta_posterior_rerank` | `_read_toml_flag_for(...)` |
| Default | False | `resolve_use_zeta_posterior_rerank()` |

When the flag resolves True, `retrieve()` / `retrieve_with_tiers()` package the pinned defaults as a 3-tuple:

```python
zeta_params = (ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT)
```

and pass it into `_l1_hits(..., zeta_params=zeta_params)`. The rerank loop swaps `partial_bayesian_score(bm25_raw, b.alpha, b.beta, posterior_weight)` for `zeta_posterior_score(bm25_raw, ζα, ζβ, ζscale, posterior_mean(b.alpha, b.beta))` on the non-heat path.

The byte-identical short-circuits (`posterior_weight == 0.0 and not heat_active and not hash_n_literals`) extend to require `zeta_params is None`, so ζ-on always exercises the rerank loop.

### Tunability

Tunability via the `(alpha, beta, scale)` kwargs is preserved for testability — the unit tests sweep parameters to verify the σ-bound and monotonicity properties. **Env / TOML knobs for α, β, scale are deferred** per #817 § "Out of scope". Operator intent is to ship the fixed-default and revisit tunability if the labeled-corpus bench surfaces a need for per-deployment tuning.

---

## Composition with γ

γ and ζ are **mutually exclusive on a given retrieval call**. When both flags resolve True, `_assert_gamma_zeta_mutual_exclusion(gamma_on, zeta_on)` raises `ValueError` at flag resolution — both `retrieve()` and `retrieve_with_tiers()` perform the check immediately after resolving each flag independently:

```python
gamma_on = resolve_use_gamma_posterior_temperature()
gamma_t = (... if gamma_on else None)
zeta_on = resolve_use_zeta_posterior_rerank()
_assert_gamma_zeta_mutual_exclusion(gamma_on, zeta_on)
zeta_params = ((ζα, ζβ, ζscale) if zeta_on else None)
```

The operator decision per #817 § "Out of scope" was: composition (γ then ζ, or weighted sum, or hierarchical) requires a separate scoping issue. Raise loudly on collision so a misconfigured deployment is caught at retrieval-time, not at silent rank-divergence time.

Tests: `test_retrieve_raises_when_both_flags_on`, `test_retrieve_with_tiers_raises_when_both_flags_on`, `test_mutex_raises_when_both_on`.

## Composition with heat-rerank

ζ and the heat-kernel rerank (`use_heat_kernel`) are mutually exclusive on a given call, same as γ. When both ζ and heat are on AND a non-stale eigenbasis is available, the heat-rerank fires and ζ is a no-op for that call. The byte-identical short-circuit detects this and routes through the existing `combine_log_scores` path. Composition with heat is the same deferred-scoping question as γ-with-heat.

---

## Where ζ sits

`src/aelfrice/scoring.py::zeta_posterior_score` — the math primitive.
`src/aelfrice/retrieval.py`:

- `USE_ZETA_POSTERIOR_RERANK_FLAG`, `ENV_USE_ZETA_POSTERIOR_RERANK` — surface names.
- `_env_use_zeta_posterior_rerank_override()` — env decoder.
- `resolve_use_zeta_posterior_rerank(explicit=None, *, start=None)` — five-path resolver.
- `_assert_gamma_zeta_mutual_exclusion(gamma_on, zeta_on)` — mutex helper.
- `_l1_hits` — rerank loop; `zeta_params: tuple[float, float, float] | None` kwarg.
- `retrieve()` / `retrieve_with_tiers()` — call sites.

---

## Bench-gate / ship-or-defer policy

| Gate | Status | Notes |
|---|---|---|
| **G1** — surface lands behind a default-OFF flag | shipped | this PR |
| **G2** — R&D campaign R0–R4 verdict ADOPT | shipped | `experiments/zeta-posterior/` (lab-side) |
| **G3** — labeled relevance corpus exists | **pending** | corpus authoring tracked separately; same gate as γ's G3 |
| **G4** — ζ@defaults vs γ@T=1.0 on labeled corpus shows discriminable rank-overlap deltas | **pending G3** | adoption verdict gate |
| **G5** — flip default to True if G4 clears with effect size ≥ 1σ | **pending G3, G4** | follow-up PR |

Same shape as `feature-posterior-temperature.md` § "Bench-gate / ship-or-defer policy". Until G3 lands, ζ is plumbing — no behavioural change on any default code path. The R&D campaign already cleared the "ζ is plausibly better than γ on synthetic corpora" gate; G3+G4 are the production-fidelity gate.

---

## Out of scope (separate issues)

- **Adaptive `β` via meta-belief.** Per #800: "Deferred until ζ is verified." If ζ clears G4 the eventual #758 retarget shifts from adaptive γ-T to adaptive ζ-β. Until then `β` is pinned at 0.25.
- **Env / TOML tunability of α, β, scale.** The fixed-default ships first. Operator surfaces for the shape parameters require campaign evidence that single-deployment tuning matters.
- **Composition with γ.** Both flags ON raises. Designing a composed rerank (linear combination, gating, or hierarchical) is a separate scoping decision per the operator note.
- **Composition with heat-rerank.** Heat dominates ζ on heat-active calls; composing the two is the same deferred-scoping question as γ-with-heat.
- **Adaptive learning of α / scale.** The R&D verdict at fixed (α=1, scale=14.5) is what shipped. Whether α and scale should be meta-belief-driven is a follow-up campaign.

---

## Refs

- **#800** — R&D campaign (closes via this PR).
- **#796 / PR #807** — γ predecessor.
- **#758** — adaptive `posterior_temperature` consumer. Retargets to adaptive ζ-β if ζ clears G4.
- **#605** — PHILOSOPHY (deterministic, narrow surface, stdlib only). ζ inherits.
- **#661** — federation read-only; score computation is read-time.
- `src/aelfrice/scoring.py:zeta_posterior_score` — entry point.
- `src/aelfrice/retrieval.py:resolve_use_zeta_posterior_rerank` — flag resolver.
- `src/aelfrice/retrieval.py:_assert_gamma_zeta_mutual_exclusion` — γ/ζ mutex.
- `src/aelfrice/retrieval.py:_l1_hits` — call site.
- `tests/test_scoring_zeta.py`, `tests/test_retrieve_zeta_flag.py`, `tests/test_zeta_vs_gamma_panel.py` — contract tests.
- `experiments/zeta-posterior/RUNNING_DOC.md` (lab-side) — R0–R4 + verdict.
