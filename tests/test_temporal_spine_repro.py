"""#1064 G5 flip-gate evidence — determinism / reproducibility.

The temporal-spine flip gate's final criterion (G5, see
``docs/design/feature-temporal-spine.md`` § Flip gate) has two halves:

  1. **Two-build byte-identity of the spine table on a fixed corpus.**
     The backfill writer is stdlib-only and sampling-free (#605), so the
     same corpus fed in the same insertion order must produce a
     byte-identical ``TEMPORAL_NEXT`` table on every build. This pins
     that: two independent stores fed the identical corpus yield the
     identical serialized spine table (compared in stored/rowid order,
     not sorted — so any ordering non-determinism would fail), and a
     re-backfill on an already-built store changes nothing.

  2. **Ablation bench green in CI.** The full ablation
     (``benchmarks/temporal_spine_ablation.py``) scores gold-set coverage
     on LoCoMo and is run-on-demand — a labelled corpus and minutes of
     retrieval don't belong in the pytest matrix. This pins the
     *mechanism the bench measures* on a controlled fixture, driving the
     real ``retrieve_v2`` lane through the bench's own scoring
     accumulators (``CoverageAccumulator`` + ``RankInvarianceAccumulator``):
     a gold belief with zero lexical overlap with the query is unreachable
     lane-off and reachable lane-on via a single ``TEMPORAL_NEXT`` hop, so
     coverage strictly improves while the ``[locked, l25, l1, hrr]`` core
     prefix stays invariant.

Deterministic and stdlib-only — no LLM, no embedding, no sampling, and no
wall-clock assertion (latency gates flake on shared runners, #739/#754) —
so both halves run green in the ordinary pytest CI matrix.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    Belief,
)
from aelfrice.retrieval import last_lane_telemetry, retrieve_v2
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import backfill_temporal_spine

# Import the bench's own scoring accumulators so "the ablation bench is
# green" is exercised through the exact code the bench uses, not a
# re-implementation. Mirror test_temporal_spine_ablation.py's sys.path shim.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.temporal_spine_ablation import (  # noqa: E402
    CoverageAccumulator,
    RankInvarianceAccumulator,
)

# Production operating point (the #1064 G2 gate budget); small fixture, so
# the budget is generous relative to the corpus — the point is reachability
# lane-on vs lane-off, not the trim (the trim is G2's job).
_BUDGET = 1500
_L1_LIMIT = 50


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    session_id: str | None,
    created_at: str,
) -> None:
    store.insert_belief(Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        session_id=session_id,
    ))


# ---------------------------------------------------------------------------
# G5 half 1 — two-build byte-identity of the spine table
# ---------------------------------------------------------------------------

# A fixed corpus in a fixed insertion order. Exercises the ordering paths
# that could introduce non-determinism: a created_at tie ("a2"/"a3" share a
# timestamp, so the tie-break falls to insertion order / rowid), two
# interleaved sessions, and a null-session belief that gets no edge.
_CORPUS: tuple[tuple[str, str, str | None, str], ...] = (
    ("a1", "alpha one",   "s1", "2026-01-01T00:00:00Z"),
    ("a2", "alpha two",   "s1", "2026-01-01T00:01:00Z"),
    ("a3", "alpha three", "s1", "2026-01-01T00:01:00Z"),  # tie with a2
    ("b1", "beta one",    "s2", "2026-01-01T00:00:30Z"),
    ("b2", "beta two",    "s2", "2026-01-01T00:02:00Z"),
    ("n1", "no session",  None, "2026-01-01T00:00:00Z"),
)


def _build_corpus_store() -> MemoryStore:
    store = MemoryStore(":memory:")
    for belief_id, content, session_id, created_at in _CORPUS:
        _make_belief(
            store,
            belief_id=belief_id,
            content=content,
            session_id=session_id,
            created_at=created_at,
        )
    return store


def _serialize_spine_table(store: MemoryStore) -> bytes:
    """Canonical byte-serialization of the TEMPORAL_NEXT table.

    Rows are read in stored (rowid) order — NOT sorted — so a build that
    inserted the same edges in a different sequence would serialize
    differently and fail the byte-identity assertion. The blob is the
    determinism artifact the G5 gate names.
    """
    rows = store._conn.execute(  # type: ignore[attr-defined]  # noqa: SLF001
        "SELECT src, dst, type, weight FROM edges "
        "WHERE type = ? ORDER BY rowid",
        (EDGE_TEMPORAL_NEXT,),
    ).fetchall()
    lines = [f"{r[0]}|{r[1]}|{r[2]}|{r[3]!r}" for r in rows]
    return "\n".join(lines).encode("utf-8")


def test_two_build_byte_identity() -> None:
    """The same corpus built twice into independent stores yields a
    byte-identical spine table."""
    store_a = _build_corpus_store()
    backfill_temporal_spine(store_a)
    blob_a = _serialize_spine_table(store_a)

    store_b = _build_corpus_store()
    backfill_temporal_spine(store_b)
    blob_b = _serialize_spine_table(store_b)

    # Non-vacuous: the corpus actually produces edges (else identity is trivial).
    assert blob_a, "fixed corpus produced an empty spine table — check the fixture"
    assert blob_a == blob_b
    # Hash equality is implied by byte equality; assert it too so a failure
    # message surfaces the digests rather than a multi-line blob diff.
    assert hashlib.sha256(blob_a).hexdigest() == hashlib.sha256(blob_b).hexdigest()


def test_rebuild_on_built_store_is_byte_identical() -> None:
    """Re-running the backfill on an already-built store is a no-op at the
    byte level — idempotency guarantees the table neither grows nor
    reorders (the #605 determinism property under a partial-build retry)."""
    store = _build_corpus_store()
    first = backfill_temporal_spine(store)
    blob_before = _serialize_spine_table(store)
    second = backfill_temporal_spine(store)
    blob_after = _serialize_spine_table(store)
    assert first.n_edges_written > 0
    assert second.n_edges_written == 0
    assert second.n_edges_existing == first.n_edges_written
    assert blob_before == blob_after


# ---------------------------------------------------------------------------
# G5 half 2 — ablation bench green (mechanism smoke on a fixture corpus)
# ---------------------------------------------------------------------------

# The query matches "b_match" lexically and shares NO salient term with the
# gold belief "b_gold"; the two are chronologically consecutive in one
# session, so the spine links them. Lane-off can only reach b_match; lane-on
# reaches b_gold through the single TEMPORAL_NEXT hop.
_ABLATION_QUERY = "docker kubernetes deployment"
_ABLATION_GOLD = "b_gold"


@pytest.fixture
def ablation_store() -> MemoryStore:
    store = MemoryStore(":memory:")
    _make_belief(
        store, belief_id="b_match",
        content="the deployment pipeline uses docker containers and kubernetes",
        session_id="sess", created_at="2026-01-01T00:00:00Z",
    )
    _make_belief(
        store, belief_id="b_gold",
        content="the amber ledger reconciliation threshold equals forty two",
        session_id="sess", created_at="2026-01-01T00:01:00Z",
    )
    # A distractor in a different session so the store isn't a single chain.
    _make_belief(
        store, belief_id="b_other",
        content="unrelated grocery list milk eggs bread",
        session_id="other", created_at="2026-01-01T00:00:30Z",
    )
    report = backfill_temporal_spine(store)
    assert report.n_edges_written == 1, "fixture must build exactly the b_match→b_gold edge"
    return store


def _retrieved_ids(store: MemoryStore, *, lane_on: bool) -> list[str]:
    result = retrieve_v2(
        store, _ABLATION_QUERY,
        budget=_BUDGET, l1_limit=_L1_LIMIT,
        include_locked=False, use_temporal_spine=lane_on,
    )
    return [b.id for b in result.beliefs]


def test_ablation_smoke_coverage_gain(ablation_store: MemoryStore) -> None:
    """Gold coverage (the bench's metric) strictly improves lane-off → on.

    Scored through the bench's own ``CoverageAccumulator`` with the same
    ``|gold ∩ retrieved| / |gold|`` formula ``run_arm_on_store`` uses, so a
    green result here is the ablation bench being green on a controlled
    corpus."""
    gold = {_ABLATION_GOLD}
    off_ids = _retrieved_ids(ablation_store, lane_on=False)
    on_ids = _retrieved_ids(ablation_store, lane_on=True)

    off_acc = CoverageAccumulator()
    off_acc.add(0, len(gold & set(off_ids)) / len(gold))
    on_acc = CoverageAccumulator()
    on_acc.add(0, len(gold & set(on_ids)) / len(gold))

    # Zero-overlap gold: unreachable lane-off, fully covered lane-on.
    assert off_acc.overall() == 0.0
    assert on_acc.overall() == 1.0
    assert on_acc.overall() > off_acc.overall()  # the ablation's "+coverage" claim
    # The lane actually fired (not a vacuous null — #981).
    tel = last_lane_telemetry()
    assert tel.temporal_spine_candidates >= 1
    assert tel.temporal_spine >= 1


def test_ablation_smoke_top_rank_invariant(ablation_store: MemoryStore) -> None:
    """The lane appends the gold below the core; it never displaces or
    reorders the ``[locked, l25, l1, hrr]`` core prefix — G2's "no
    top-rank regression", pinned on a fixture through the bench's own
    ``RankInvarianceAccumulator``."""
    off_result = retrieve_v2(
        ablation_store, _ABLATION_QUERY,
        budget=_BUDGET, l1_limit=_L1_LIMIT,
        include_locked=False, use_temporal_spine=False,
    )
    off_ids = [b.id for b in off_result.beliefs]
    tel_off = last_lane_telemetry()
    # No locks in the fixture, so the core is l25 + l1 + hrr (include_locked
    # False drops the locked count — see the shadow-eval methodology note).
    core_off = tel_off.l25 + tel_off.l1 + tel_off.hrr_expand

    on_result = retrieve_v2(
        ablation_store, _ABLATION_QUERY,
        budget=_BUDGET, l1_limit=_L1_LIMIT,
        include_locked=False, use_temporal_spine=True,
    )
    on_ids = [b.id for b in on_result.beliefs]
    tel_on = last_lane_telemetry()
    core_on = tel_on.l25 + tel_on.l1 + tel_on.hrr_expand

    acc = RankInvarianceAccumulator()
    acc.add(off_ids, on_ids, core_off, core_on, tel_on.temporal_spine)

    assert acc.passed()
    assert acc.top_rank_displacements == 0
    assert acc.core_mismatch == 0
    assert acc.spine_added_sum >= 1  # the lane contributed the gold belief
