"""Axiomatic retrieval constraints as a merge gate on the scorer (#1174).

Fang, Tao & Zhai (2004), *A Formal Study of Information Retrieval
Heuristics* (SIGIR '04), give a set of constraints any sane term-weighting
function must satisfy. They are **model-free** — they hold for BM25,
BM25+, LM-Dirichlet, PL2 and DPH alike — so they gate the *property* the
ranker is supposed to have rather than the constants it currently uses.
That is what makes them safe to keep across a scorer rewrite, and it is
the reason this file exists: `eval-calibration` is the only byte-exact
ranking baseline in CI, and a byte-exact baseline cannot say whether a
*new* ranking is sane, only whether it changed.

Every constraint is asserted **in both scoring modes** — the shipped
single-field lane and the per-field BM25F lane (#1180) — so a change to
one cannot quietly diverge from the other.

**Known violations are asserted, not skipped.** Three constraints do not
hold at shipped defaults. Each is pinned as an explicit assertion of the
*current* behaviour with a pointer to the issue that owns it, rather than
`xfail`-ed. That has two consequences, both wanted:

- the gate states the defect out loud instead of hiding it behind a
  permanently-green skip, and
- whoever fixes one of them has to come here and flip the assertion,
  which is the moment to decide whether the fix was intended.

A test named ``test_*_is_currently_violated`` is therefore a defect
record, not a passing property.
"""
from __future__ import annotations

import pytest

from aelfrice.bm25 import BM25Index
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.scoring import partial_bayesian_score
from aelfrice.store import MemoryStore

ANCHOR_W = 3

# Filler that is never a query term, long enough that `dl` differences
# between documents come from the terms under test rather than from noise.
PAD = " ".join(f"z{i}" for i in range(30))

# Both scoring modes. Every constraint below runs under each.
MODES = [
    pytest.param({}, id="single-field"),
    pytest.param({"per_field": True}, id="per-field"),
]


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _index(
    docs: list[tuple[str, str]],
    *,
    anchors: list[tuple[str, str]] | None = None,
    **kw: object,
) -> BM25Index:
    """Build an in-memory index from `(belief_id, content)` pairs.

    `anchors` is a list of `(anchor_text, dst_belief_id)`; each mints its
    own citer belief. Callers that compare two documents must put both in
    the *same* index — building two indexes and comparing across them
    varies `n_docs`, which moves `idf` for every term and silently
    confounds the comparison.
    """
    store = MemoryStore(":memory:")
    for bid, text in docs:
        store.insert_belief(_mk(bid, text))
    for i, (anchor_text, dst) in enumerate(anchors or []):
        store.insert_belief(_mk(f"__citer{i}", f"citing belief {i}"))
        store.insert_edge(Edge(
            src=f"__citer{i}", dst=dst, type="cites", weight=1.0,
            anchor_text=anchor_text,
        ))
    return BM25Index.build(store, anchor_weight=ANCHOR_W, **kw)  # type: ignore[arg-type]


def _score(index: BM25Index, query: str, belief_id: str) -> float:
    """Score for one belief, 0.0 when it does not match at all."""
    return dict(index.score(query, top_k=500)).get(belief_id, 0.0)


# --- constraints that hold ------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_tfc1_more_occurrences_scores_higher(mode: dict) -> None:
    """TFC1. Two documents of equal length; the one containing the query
    term more often must score strictly higher."""
    index = _index(
        [("more", f"alpha alpha {PAD}"), ("fewer", f"alpha beta {PAD}")],
        **mode,
    )
    assert _score(index, "alpha", "more") > _score(index, "alpha", "fewer")


@pytest.mark.parametrize("mode", MODES)
def test_tfc2_term_frequency_has_diminishing_returns(mode: dict) -> None:
    """TFC2. The gain from one more occurrence must strictly shrink as
    occurrences accumulate — the saturation property. A scorer that lost
    it would let one keyword-stuffed belief dominate every pack.

    Document length is held **constant** across the sweep: each extra
    `alpha` displaces one filler token rather than being appended. Under
    a growing document the length penalty produces diminishing returns
    on its own, so a sweep that let `dl` vary would pass even with
    saturation removed entirely — verified by mutation, which is why the
    substitution matters.
    """
    width = 24
    scores = []
    for tf in range(1, 6):
        body = ["alpha"] * tf + [f"f{i}" for i in range(width - tf)]
        index = _index(
            [("d", " ".join(body) + f" {PAD}"), ("other", f"beta {PAD}")],
            **mode,
        )
        scores.append(_score(index, "alpha", "d"))
    deltas = [b - a for a, b in zip(scores, scores[1:])]
    assert all(x > 0 for x in deltas), f"not monotone increasing: {deltas}"
    assert all(a > b for a, b in zip(deltas, deltas[1:])), (
        f"gains not diminishing: {deltas}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_lnc1_padding_with_non_query_terms_cannot_help(mode: dict) -> None:
    """LNC1. Appending terms the query never mentions must not raise the
    score. Without it, verbosity is a ranking strategy."""
    short = _index([("d", f"alpha {PAD}"), ("o", f"beta {PAD}")], **mode)
    padded = _index(
        [("d", f"alpha {PAD} qqq www eee rrr"), ("o", f"beta {PAD}")], **mode,
    )
    assert _score(padded, "alpha", "d") <= _score(short, "alpha", "d")


@pytest.mark.parametrize("mode", MODES)
def test_lnc2_self_concatenation_cannot_hurt(mode: dict) -> None:
    """LNC2 / TF-LNC. A document concatenated with itself says the same
    thing more emphatically; it must not score *below* the original.
    Over-aggressive length normalisation is what this catches."""
    once = _index([("d", f"alpha {PAD}"), ("o", f"beta {PAD}")], **mode)
    thrice = _index(
        [("d", f"alpha {PAD} " * 3), ("o", f"beta {PAD}")], **mode,
    )
    assert _score(thrice, "alpha", "d") >= _score(once, "alpha", "d") - 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_tdc_mass_on_the_rarer_term_wins(mode: dict) -> None:
    """TDC. For equal total query-term mass, the document concentrating
    it on the higher-idf term must score higher — the ranker has to
    prefer the discriminating term over the common one."""
    docs = [
        ("rare_heavy", f"rare rare common {PAD}"),
        ("common_heavy", f"rare common common {PAD}"),
    ]
    docs += [(f"bg{i}", f"common {PAD}") for i in range(12)]
    index = _index(docs, **mode)
    assert (
        _score(index, "rare common", "rare_heavy")
        > _score(index, "rare common", "common_heavy")
    )


@pytest.mark.parametrize("mode", MODES)
def test_on_topic_anchor_text_cannot_demote(mode: dict) -> None:
    """Stream monotonicity, on-topic half. Against an identical uncited
    twin *in the same index*, a belief whose citers used the query term
    must not score lower. This is the property the anchor stream exists
    to provide."""
    index = _index(
        [("cited", f"alpha {PAD}"), ("uncited", f"alpha {PAD} x")],
        anchors=[("alpha topic", "cited")],
        **mode,
    )
    assert _score(index, "alpha", "cited") >= _score(index, "alpha", "uncited")


# --- posterior blend ------------------------------------------------------


def test_posterior_blend_is_monotone_in_earned_evidence() -> None:
    """A belief with corroborating evidence must outrank an identical
    one without it, at equal lexical relevance."""
    assert (
        partial_bayesian_score(-2.0, 5.0, 1.0)
        > partial_bayesian_score(-2.0, 0.5, 0.5)
    )


def test_posterior_blend_ignores_bm25_sign_convention() -> None:
    """The blend consumes a positive relevance magnitude. Stronger
    lexical match must score higher at equal posterior — a regression
    here inverts the entire L1 ordering."""
    assert (
        partial_bayesian_score(-4.0, 1.0, 1.0)
        > partial_bayesian_score(-1.0, 1.0, 1.0)
    )


# --- known violations, pinned rather than skipped -------------------------


@pytest.mark.parametrize("mode", MODES)
def test_qtfc_is_currently_violated_at_the_default_k3(mode: dict) -> None:
    """QTFC. Repeating a query term should change the score.

    **Violated at the shipped default**, deliberately: `DEFAULT_K3 = 0.0`
    collapses the query-saturation factor to 1.0 for every count >= 1, so
    `score("budget token") == score("budget budget budget token")`. #1179
    fixed the underlying assignment-vs-accumulation bug but kept 0.0 as
    the default, because three shipped components express a boost as a
    duplicated token and their multipliers were tuned against the FTS5
    lane. Flipping `k3` is bench-gated separately.

    Pinned both ways so the default's cost is visible and the mechanism
    is proven live rather than assumed.
    """
    kw = {"docs": [("d", f"budget token {PAD}"), ("o", f"other {PAD}")]}
    once, thrice = "budget token", "budget budget budget token"

    off = _index(kw["docs"], **mode)  # type: ignore[arg-type]
    assert _score(off, once, "d") == _score(off, thrice, "d"), (
        "k3 defaulted away from 0.0 without updating this gate — "
        "repetition now moves the score. See #1179."
    )

    on = _index(kw["docs"], k3=8.0, **mode)  # type: ignore[arg-type]
    assert _score(on, thrice, "d") > _score(on, once, "d"), (
        "k3 > 0 no longer weights repeated query terms — the qf "
        "mechanism has gone inert again. See #1179."
    )


def test_off_topic_anchor_text_demotes_on_the_single_field_lane() -> None:
    """Stream monotonicity, off-topic half — **violated on the shipped
    lane** (#1180).

    A belief whose citers wrote about something else is demoted below an
    otherwise identical uncited belief, because the anchor replicas land
    in the same `dl` and length-penalise the belief's own content terms.
    Nothing about the belief itself changed; it is punished for what was
    written *about* it.

    Pinned as the defect it is. The per-field lane satisfies the
    constraint — see the companion test below — so this assertion is what
    records that the default lane still does not.
    """
    index = _index(
        [("cited", f"alpha {PAD}"), ("uncited", f"alpha {PAD}")],
        anchors=[(" ".join(f"unrelated{i}" for i in range(40)), "cited")],
    )
    assert _score(index, "alpha", "cited") < _score(index, "alpha", "uncited")


def test_off_topic_anchor_text_is_neutral_on_the_per_field_lane() -> None:
    """The same corpus under per-field BM25F (#1180) satisfies stream
    monotonicity: anchor text that never mentions the query term
    contributes nothing, so the cited belief scores exactly what its
    uncited twin does.

    This pair is the whole argument for the per-field lane reduced to one
    model-free axiom, and it is why the constraint is worth gating on.
    """
    index = _index(
        [("cited", f"alpha {PAD}"), ("uncited", f"alpha {PAD}")],
        anchors=[(" ".join(f"unrelated{i}" for i in range(40)), "cited")],
        per_field=True,
    )
    assert _score(index, "alpha", "cited") == pytest.approx(
        _score(index, "alpha", "uncited"), rel=1e-6,
    )


def test_ingest_prior_is_currently_penalised_against_no_evidence() -> None:
    """Posterior neutrality — **violated** (#1174).

    A belief carrying zero earned evidence should rank as if the
    posterior term said nothing about it. Instead the agent-inferred
    factual ingest prior (alpha=0.6, beta=1.0) gives mu=0.375, which is
    *below* the mu=0.5 an unobserved belief reads, so simply being
    ingested applies a penalty relative to knowing nothing.

    That is not a rounding detail: the #1174 measurement found 67.4% of
    beliefs on a real store sitting exactly on this prior, all carrying
    the penalty this test pins.
    """
    on_prior = partial_bayesian_score(-2.0, 0.6, 1.0)
    no_evidence = partial_bayesian_score(-2.0, 0.5, 0.5)
    assert on_prior < no_evidence
    # Pin the size too — a fix that shrinks it should have to say so.
    assert no_evidence - on_prior == pytest.approx(0.1438, abs=1e-3)
