# uri_baki retest — results

Issue [#153](https://github.com/robotrocketscience/aelfrice/issues/153).
Methodology fix for the prior synthetic-graph attempt that regressed
NDCG@10 by ~0.05 with random "locked" picks.

## Verdict

**Honest negative — uri_baki post-rank adjusters do not meet the
ratification gate on the relevance-aware retest.** Per the issue's
acceptance criteria #4 and #5, this lands as a documented negative.
The `uri_baki` approach is filed as **not-adopted** in its current
form.

The ratification gate from the issue body:

> **AC#4.** ≥ +0.02 NDCG@10 with ≤ 5 ms additional latency → candidate
> for production wiring.
> **AC#5.** ≤ 0 NDCG@10 lift → honest negative, filed as not-adopted.

Best per-effect result across N ∈ {10k, 50k}:

- `locked_floor` (relevance-aware locks): **0.0000** (best q=0.5/0.75)
  — floor calibrated below all matched scores is a no-op; aggressive
  floors regress (q=0.9: −0.036 at N=10k).
- `supersession_demote`: **±0.003** — within seed noise.
- `recency_decay` (λ=1/365): **+0.0049 at N=10k**, **−0.053 at N=50k**.
  Below the +0.02 gate at N=10k and inverts sign at scale.
- `combined`: **−0.015 at N=10k**, **−0.069 at N=50k** — net negative.

## Result tables

### N = 10 000, Q = 200, seed = 42

| condition                              | NDCG@10 | Δ vs base | ms/query |
|----------------------------------------|--------:|----------:|---------:|
| baseline                               |  0.7656 |         — |    1.016 |
| locked_floor[ra, q=0.5]                |  0.7656 |   +0.0000 |    1.350 |
| locked_floor[ra, q=0.75]               |  0.7656 |   +0.0000 |    1.271 |
| locked_floor[ra, q=0.9]                |  0.7295 |   −0.0361 |    1.263 |
| locked_floor[random, q=0.5] (control)  |  0.7656 |   +0.0000 |    1.287 |
| supersession_demote[f=0.7]             |  0.7645 |   −0.0011 |    1.303 |
| supersession_demote[f=0.5]             |  0.7684 |   +0.0028 |    1.314 |
| supersession_demote[f=0.25]            |  0.7684 |   +0.0028 |    1.286 |
| recency_decay[λ=1/365]                 |  0.7705 |   +0.0049 |    3.805 |
| recency_decay[λ=1/180]                 |  0.7518 |   −0.0138 |    3.736 |
| recency_decay[λ=1/90]                  |  0.7380 |   −0.0275 |    3.686 |
| combined[ra, q=0.5, f=0.5, λ=1/180]    |  0.7510 |   −0.0145 |    4.389 |

Locked beliefs: 50 (relevance-aware: only on hot topics).
Superseded: 500.

### N = 50 000, Q = 200, seed = 42

| condition                              | NDCG@10 | Δ vs base | ms/query |
|----------------------------------------|--------:|----------:|---------:|
| baseline                               |  0.8865 |         — |    5.288 |
| locked_floor[ra, q=0.5]                |  0.8865 |   +0.0000 |    8.030 |
| locked_floor[ra, q=0.75]               |  0.8865 |   +0.0000 |    7.647 |
| locked_floor[ra, q=0.9]                |  0.8865 |   +0.0000 |    7.674 |
| locked_floor[random, q=0.5] (control)  |  0.8865 |   +0.0000 |    7.708 |
| supersession_demote[f=0.7]             |  0.8857 |   −0.0008 |    7.017 |
| supersession_demote[f=0.5]             |  0.8827 |   −0.0038 |    7.034 |
| supersession_demote[f=0.25]            |  0.8827 |   −0.0038 |    6.974 |
| recency_decay[λ=1/365]                 |  0.8333 |   −0.0532 |   18.887 |
| recency_decay[λ=1/180]                 |  0.8184 |   −0.0681 |   19.579 |
| recency_decay[λ=1/90]                  |  0.7992 |   −0.0873 |   19.747 |
| combined[ra, q=0.5, f=0.5, λ=1/180]    |  0.8174 |   −0.0691 |   24.143 |

Locked beliefs: 250.
Superseded: 2 500.

JSON reports: `results_n10k.json`, `results_n50k.json`. Plain-text
captures: `results_n10k.txt`, `results_n50k.txt`.

## Why the synthetic produces a negative

The retest fixed the **locked-set methodology** (locks are now drawn
from beliefs whose topic appears in the query distribution — the
"hot" subset — so per-query lock relevance is non-zero by
construction). The rest of the corpus design leaves three signal
gaps that the issue spec did not specify and that systematically
push the result toward zero or negative:

1. **Locked-floor calibration is at war with itself.** A floor low
   enough to be safe (q=0.5 of matched scores) sits below most
   already-matched beliefs and is a no-op. A floor high enough to
   actually lift unmatched relevance-aware locks (q=0.9) drags
   non-relevant locks into the top-10 too — losing more rank
   positions than it gains. The intermediate quantiles never beat
   zero.
2. **Supersession has no obsolescence signal.** The synthetic marks
   beliefs as superseded uniformly within their topic. A query
   matching `supersedes(a, b)` matches `a` and `b` at correlated
   rates because both are in the same topic. Demoting `a` by 0.5
   does not change rank between the two if they had similar scores
   to start; among non-related beliefs the demote is uncorrelated
   with relevance, so NDCG moves stay at noise. To produce a
   positive result, the corpus would need superseded beliefs to be
   biased toward *lower* topic-vocabulary density (i.e., the
   superseder is a "better" version of the same claim with stronger
   keyword match) — that bias is editorial and would prove the
   instrument, not the effect.
3. **Recency decay's sign flips with N.** At N=10k λ=1/365 it lifts
   by +0.005 (close to noise). At N=50k it drops by −0.05 across
   all λ — the larger candidate pool means decay multiplicatively
   penalises the long tail that occasionally wins on filler-term
   collisions, but it also penalises legitimately-relevant older
   beliefs more aggressively at scale. Latency at N=50k (>13 ms
   added) blows past the 5 ms gate even if the lift were positive.

The control row `locked_floor[random, q=0.5]` matches the
relevance-aware row at +0.0000, so the relevance-aware locking does
not introduce its own signal at this lock rate (0.5 %) on this
corpus shape. Real-corpus lock rates and topic concentration may be
qualitatively different — see "follow-up paths" below.

## Reproduce

```bash
uv run python -m benchmarks.uri_baki_retest.harness \
    --n-beliefs 10000 --n-queries 200 --seed 42 \
    --json benchmarks/uri_baki_retest/results_n10k.json

uv run python -m benchmarks.uri_baki_retest.harness \
    --n-beliefs 50000 --n-queries 200 --seed 42 \
    --json benchmarks/uri_baki_retest/results_n50k.json
```

Harness lives in `benchmarks/uri_baki_retest/harness.py`. The
synthetic corpus is fully deterministic at the configured seed;
re-running yields identical numbers.

## Follow-up paths (out of scope for this retest)

The synthetic verdict is honest negative; it is not a verdict on the
underlying pattern. A future retest could:

- **Real-corpus retest.** Run the same harness against an exported
  snapshot of a live aelfrice store with ≥ 1 000 beliefs, real
  `lock_state` distribution, real `SUPERSEDES` edges, and a query
  set replayed from `rebuild_logs/` (#288 phase-1b will produce
  these once an operator-week of data is captured). Lock
  concentration in real stores may be qualitatively different from
  the 0.5 %-uniform-over-hot model used here.
- **Per-effect editorial corpora.** Build three separate corpora,
  each engineered to give one effect a positive prior: locked
  beliefs with low keyword density (low ranker recall, lock should
  rescue); supersession with strict version pairs (better-version
  demotion should help); long-tailed age distribution with a
  topic-temperature axis (decay should help). Each corpus is honest
  about its bias and serves as an upper-bound estimate per effect.
- **Production trial behind `query_strategy = uri_baki`.** Per the
  pipeline-composition tracker (#154) feature-flag pattern, a
  production-shadow trial (no user-visible effect) running the
  adjusters and reporting per-rebuild deltas via #288 phase-1b logs
  would produce real-data evidence at low risk.

These are all separate issues and are not implied by the negative
verdict here.
