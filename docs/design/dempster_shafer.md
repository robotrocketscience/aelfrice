# Dempster–Shafer / subjective logic for belief ranking — explored and closed

**Status: closed (negative). Do not re-propose without new evidence.** This memo records a
deep investigation into generalizing aelfrice's Beta-Bernoulli belief ranking with
Dempster–Shafer (DS) evidence theory / subjective logic. The conclusion, reached through
~a dozen cheap disconfirmations rather than a speculative build, is that **DS offers aelfrice
no net win on either ranking or contradiction handling** given the current substrate and the
#605 determinism constraint.

## Motivation

aelfrice ranks beliefs with `score = log(bm25_pos) + posterior_weight · log(α/(α+β))`
(`scoring.partial_bayesian_score`). Each belief carries a Beta posterior — α (evidence-for),
β (evidence-against). The stated hope: DS's mass on the *ignorance* set {T,F} separates "no
evidence yet" from "balanced conflicting evidence" — two states a single mean α/(α+β) can't
tell apart (both read 0.5).

## What DS / subjective logic actually is (the bridge)

A binary subjective-logic *opinion* (belief b, disbelief d, **uncertainty u**, base rate a),
with b+d+u=1, *is* a DS binary mass function: m({T})=b, m({F})=d, m({T,F})=u. It maps
bijectively to a Beta(α,β): with prior weight W=2, `u = W/(α+β)`, and the projected
probability **`P = b + a·u = α/(α+β)`** — it *recovers the current posterior mean exactly*.
So DS-via-subjective-logic is a strict superset of the Beta model, and ranking by its
projection is identical to today.

## Two arms, both closed by empirical R&D

### Arm 1 — DS as a ranking score. DEAD.

- **R1 (identity):** `BetP = m({T}) + ½·m({T,F}) = α/(α+β)` exactly (verified numerically,
  max error 1e-16). Ranking by the DS projection is a **provable no-op**. Only `Bel` /`Pl`
  can differ, and they are an affine repackaging of `(mean, α+β)` the store already carries
  (`scoring.uncertainty_score` even computes the Beta entropy — but it is never used in
  ranking).
- **R2 (real-store reorder, 40 queries):** DS-`Bel`/`Pl` reorder ~13–17% of the top-10 — but
  as a pure **evidence-count reweight** (`Bel = mean − 0.5/(α+β)`, a confidence lower bound),
  not a new relevance signal.
- **R2b (controlled regime ablation):** DS-`Bel`/`Pl` P@5 swings wildly with the
  evidence↔relevance correlation (0.45 when relevant beliefs are well-evidenced, 0.05 when
  they're cold); **Beta-mean is regime-robust (~0.235)**. In the null regime (evidence ⟂
  relevance — the realistic default) DS-`Bel` *hurts*. It's a conditional bet on a correlation,
  not a relevance signal.
- **Corroborated by prior campaigns:** #365 found posterior reweights don't flip retrieval;
  #1081 found evidence/exposure rewards float **junk** (the correlation is *backwards*). The
  real gap is recall, not calibration.

### Arm 2 — DS conflict measure K for CONTRADICTS handling. DEAD.

The one genuinely-novel idea: for a CONTRADICTS pair, `K = b_X · b_Y` (evidence-weighted
belief masses) is high only when *both* sides are strongly evidenced — a "real standoff" the
current authority-takes-all resolver (origin precedence, `valence=0.0`) cannot distinguish
from a stale loser. It died on the substrate, not the mechanism:

- **A2 (redundancy):** η²=0.724 — `b_mass` is 72% explained by origin tier, so K is largely
  redundant with the precedence resolver; its genuine value is confined to same-high-tier
  standoffs.
- **A5 (governance):** the CONTRADICTS substrate is **default-off, and ratified as such** — the
  detector docstring: *"flipping the default requires re-opening #897."* (Perf is fine —
  detection is incremental, #1000 — the gate is scope ratification.)
- **A6 (detector is crude):** the CONTRADICTS rule is literally *"lexically similar + one side
  has a negation token (no/not/never/n't) that the other lacks."* No semantics, no
  assertion-worthiness gate.
- **R3-0 + ceiling test:** over a real 45k-belief store, the deterministic detector surfaces
  114 CONTRADICTS pairs in a 5k sample. An LLM classification of all 114 found
  **0 genuine contradictions (0% precision)** — every flag a false positive (LoCoMo chat
  pleasantries, "No information available" prediction records, near-duplicate task strings).
  The "9 high-K standoffs" were all false positives. Even with *perfect* (LLM) detection, the
  true-contradiction surface was **empty** in the sample — and LLM detection breaks #605.

## Verdict

DS / subjective logic gives aelfrice nothing the Beta posterior + origin precedence don't
already provide. The rerank is a no-op or a harmful evidence-count bet; the conflict-K arm
has no reliable substrate (crude detector = noise; reliable detector = determinism-breaking +
near-empty surface). **Closed.** The salvageable *insight* — ignorance is a separate axis from
the point estimate — is, if ever wanted, better implemented inside the existing Beta machinery
(a variance-aware score, or a Walley Imprecise-Dirichlet interval) than by importing a second
uncertainty calculus with its BPA-construction and conflict-normalization burdens.

## Reproduce

The R&D rounds were run against a read-only copy of the dev store using
`scoring.partial_bayesian_score`, `relationship_detector.relationships_audit`, and
`calibration_metrics`. Key seams: `scoring.py` (`posterior_mean`, `partial_bayesian_score`,
`uncertainty_score`), `retrieval.py::_l1_hits` (the rerank seam), `contradiction.py::_pick_winner`
(authority-takes-all resolver), `relationship_detector.py` (the negation-XOR CONTRADICTS detector,
default-off per #897).
