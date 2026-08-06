"""#1366 record-at-site lane firing on `LaneTelemetry`.

Ten retrieval lanes shipped with a resolver and no way to observe them
firing. "The flag resolves True" and "the lane did work" are different
claims, and the repo has repeatedly discovered — one lane at a time, by
reading call sites — that the second was false while the first stayed
true for releases (the heat-kernel lane, `edge_rerank`, the R3 IDF-clip
boost arm). These fields close that gap by recording *at the leaf where
the work happens*.

Two properties are load-bearing and each has its own test below:

* **The record happens at the leaf, not at flag resolution.** A field
  derived from the flag, or from a tier count, keeps reporting the old
  answer after the lane is re-wired — which is the whole defect. Every
  field here has a test that goes red if its `_record_lane_fired` call
  is deleted, and the "lane on but did no work" cases pin that the
  counter is not just the flag with extra steps.

  That claim was false for `compression_renders` as first shipped: both
  of its cases asserted `== 0`, because the shared store fixture's
  beliefs are all `RETENTION_UNKNOWN` and `compression.py:139` routes
  those to `STRATEGY_VERBATIM`. Suppressing the record left the whole
  suite green. The pair now uses a `RETENTION_SNAPSHOT` fixture, so the
  lane-on case is non-zero and the two halves actually differ.
* **The counters are per call.** A stale value carried in from the
  previous call would attribute one call's work to the next.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from aelfrice.hrr_index import HRRStructIndexCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
    RETENTION_SNAPSHOT,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    ENV_FAN_EFFECT,
    ENV_MAX_COVERAGE_PACK,
    ENV_ORIGIN_TIEBREAK,
    ENV_SUPERSESSION_DEMOTE,
    ENV_USE_GAMMA_POSTERIOR_TEMPERATURE,
    ENV_USE_ZETA_POSTERIOR_RERANK,
    LANE_FIRING_FIELDS,
    SUPERSESSION_TREATMENT_DEMOTE,
    LaneTelemetry,
    _entity_persist_penalty,
    _lane_fired,
    _origin_tiebreak_decisions,
    _record_lane_fired,
    _reset_lane_firings,
    _supersession_penalty,
    last_lane_telemetry,
    retrieve,
    retrieve_v2,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    *,
    origin: str = ORIGIN_AGENT_INFERRED,
    created: str = "2026-06-01T00:00:00Z",
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created,
        last_retrieved_at=None,
        origin=origin,
    )


def _add_entity(store: MemoryStore, bid: str, lower: str, kind: str) -> None:
    store._conn.execute(
        "INSERT INTO belief_entities(belief_id, entity_lower, entity_raw, "
        "kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
        (bid, lower, lower, kind),
    )
    store._conn.commit()


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    for i in range(6):
        s.insert_belief(_mk(f"b{i}", f"alpha beta shared token variant {i}"))
    yield s
    s.close()


# --- The carrier ---------------------------------------------------------


def test_every_recorded_field_is_a_real_telemetry_field() -> None:
    """A typo'd record key would silently read as a dead lane.

    `_record_lane_fired` takes a string, so nothing but this stops a
    site from writing under a name `LaneTelemetry` never reads — and the
    symptom would be a lane reporting "never fired", which is exactly
    the fabricated finding #1366 exists to avoid.
    """
    known = {f.name for f in LaneTelemetry.__dataclass_fields__.values()}
    assert set(LANE_FIRING_FIELDS) <= known
    assert len(set(LANE_FIRING_FIELDS)) == len(LANE_FIRING_FIELDS)


def test_a_real_call_records_nothing_outside_the_declared_fields(
    store: MemoryStore,
) -> None:
    """No live site writes under a key the telemetry does not read."""
    from aelfrice.retrieval import _LANE_FIRINGS

    retrieve(store, "alpha beta")
    assert set(_LANE_FIRINGS) <= set(LANE_FIRING_FIELDS)


def test_counters_are_per_call_not_cumulative(store: MemoryStore) -> None:
    """Two identical calls report the same count, not double.

    Without the per-call reset the second call inherits the first
    call's work and every rate above 1.0 is an artifact.
    """
    retrieve(store, "alpha beta")
    first = last_lane_telemetry().cluster_packed
    retrieve(store, "alpha beta")
    second = last_lane_telemetry().cluster_packed
    # `cluster_packed` rather than `compression_renders`: the latter is 0
    # on this fixture by construction (every belief compresses verbatim),
    # and a counter that is 0 in both calls cannot distinguish a working
    # reset from a missing one.
    assert first > 0
    assert second == first


def test_pre_1366_construction_still_works() -> None:
    """Additive fields with defaults: no caller has to change."""
    tel = LaneTelemetry(locked=1, l1=2)
    for name in LANE_FIRING_FIELDS:
        assert getattr(tel, name) in (0, False)


# --- type_aware_compression ---------------------------------------------


@pytest.fixture()
def snapshot_store() -> MemoryStore:
    """Beliefs the compressor genuinely shortens.

    The shared `store` fixture is all `RETENTION_UNKNOWN`, which
    `compression.py:143` routes to `STRATEGY_VERBATIM` — so on that
    fixture the counter is 0 whether the record fires or not, and the
    lane-on / lane-off pair below could not distinguish a working record
    from a deleted one. `RETENTION_SNAPSHOT` with multi-sentence content
    takes the `_headline` arm and really does shrink.
    """
    s = MemoryStore(":memory:")
    for i in range(6):
        b = _mk(
            f"s{i}",
            f"alpha beta shared token variant {i} was recorded on Tuesday. "
            f"It lists twelve distinct items counted by hand, including "
            f"bananas, apples and pears, none of which fit in one line.",
        )
        s.insert_belief(replace(b, retention_class=RETENTION_SNAPSHOT))
    yield s
    s.close()


def test_compression_renders_is_nonzero_when_the_compressor_shortens(
    snapshot_store: MemoryStore,
) -> None:
    """The distinguishing half — this is what deletion of the record kills.

    As first shipped both cases asserted `== 0` on the all-UNKNOWN
    fixture, so suppressing `_record_lane_fired("compression_renders")`
    left the entire suite green. That is the exact defect this module
    exists to catch, in this module's own instrument.

    Asserted as `> 0` rather than as an exact count deliberately: `_cost`
    is unmemoised and the packer costs a belief several times per call,
    so the magnitude is an arm-dependent multiple of the belief count.
    `> 0` is the only part of this field any consumer reads.
    """
    retrieve(snapshot_store, "alpha beta", use_type_aware_compression=True)
    assert last_lane_telemetry().compression_renders > 0


def test_compression_renders_is_zero_when_the_lane_is_off(
    snapshot_store: MemoryStore,
) -> None:
    """The counter is the compressor's work, not the flag's value.

    Runs on the SAME fixture as the case above, so the pair differs only
    in the flag. On the old all-verbatim fixture both halves read 0 and
    the comparison was vacuous.
    """
    retrieve(snapshot_store, "alpha beta", use_type_aware_compression=False)
    assert last_lane_telemetry().compression_renders == 0


def test_compression_renders_stays_zero_when_nothing_shortens(
    store: MemoryStore,
) -> None:
    """Lane on, work absent: the third case, and the reason for the rate.

    A `STRATEGY_VERBATIM` return costs exactly `_estimate_tokens(content)`,
    which is what the uncompressed path already charges — counting it
    would report a fire for a call that changed no cost, and the rate
    would restate "the flag is on". This is what took the live figure
    from 500/500 to 212/500.
    """
    retrieve(store, "alpha beta", use_type_aware_compression=True)
    assert last_lane_telemetry().compression_renders == 0


# --- intentional_clustering ---------------------------------------------


def test_cluster_packed_counts_what_the_cluster_arm_packed(
    store: MemoryStore,
) -> None:
    retrieve_v2(store, "alpha beta", use_intentional_clustering=True)
    assert last_lane_telemetry().cluster_packed > 0


def test_cluster_packed_is_zero_when_the_arm_loses_precedence(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max-coverage takes the L1 pack when both resolve on.

    The clustering flag is still True on this call. A counter read off
    the flag would report the cluster lane firing on a call where it
    packed nothing at all.
    """
    monkeypatch.setenv(ENV_MAX_COVERAGE_PACK, "1")
    retrieve_v2(store, "alpha beta", use_intentional_clustering=True)
    tel = last_lane_telemetry()
    assert tel.cluster_packed == 0
    assert tel.max_coverage_packed > 0


# --- max_coverage_pack ---------------------------------------------------


def test_max_coverage_packed_counts_the_selector_output(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_MAX_COVERAGE_PACK, "1")
    retrieve_v2(store, "alpha beta")
    assert last_lane_telemetry().max_coverage_packed > 0


def test_max_coverage_packed_is_zero_by_default(store: MemoryStore) -> None:
    retrieve_v2(store, "alpha beta")
    assert last_lane_telemetry().max_coverage_packed == 0


# --- entity_persist_demote ----------------------------------------------


def test_entity_persist_demoted_records_only_a_real_demotion() -> None:
    """The leaf records the beliefs the penalty actually moved.

    Both halves matter: a belief absent from the S1 map, and one whose
    S1 clamps the penalty to 0.0, keep the lane-off ordering and are not
    firings. Removing the record call makes the first assertion fail.
    """
    _reset_lane_firings()
    assert _entity_persist_penalty({"b1": 0.05}, "b1") < 0.0
    assert _lane_fired("entity_persist_demoted") == 1
    # Not in the map -> no entities -> never penalised.
    assert _entity_persist_penalty({"b1": 0.05}, "b2") == 0.0
    # Lane off entirely.
    assert _entity_persist_penalty(None, "b1") == 0.0
    assert _lane_fired("entity_persist_demoted") == 1


def test_entity_persist_demoted_reaches_the_telemetry() -> None:
    """End to end: the leaf counter is wired into `LaneTelemetry`."""
    s = MemoryStore(":memory:")
    try:
        for i in range(3):
            s.insert_belief(_mk(f"b{i}", f"alpha beta ephemeral note {i}"))
            # A transient-only entity drives S1 well below 1.
            _add_entity(s, f"b{i}", f"#{900 + i}", "identifier")
        retrieve_v2(s, "alpha beta", use_entity_persist_demote=True)
        assert last_lane_telemetry().entity_persist_demoted > 0
        retrieve_v2(s, "alpha beta", use_entity_persist_demote=False)
        assert last_lane_telemetry().entity_persist_demoted == 0
    finally:
        s.close()


# --- supersession_demote -------------------------------------------------


def test_supersession_demoted_records_only_a_real_demotion() -> None:
    """`factor = 1.0` resolves the flag on and demotes nothing.

    Recording that as a firing is the same conflation as reading the
    resolver, so the leaf records the penalty, not the branch.
    """
    _reset_lane_firings()
    assert _supersession_penalty(frozenset({"b1"}), "b1", 0.5) < 0.0
    assert _lane_fired("supersession_demoted") == 1
    assert _supersession_penalty(frozenset({"b1"}), "b1", 1.0) == 0.0
    assert _supersession_penalty(frozenset({"b1"}), "b2", 0.5) == 0.0
    assert _supersession_penalty(None, "b1", 0.5) == 0.0
    assert _lane_fired("supersession_demoted") == 1


def test_supersession_demoted_reaches_the_telemetry(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        s.insert_belief(_mk("bold", "alpha beta deploy target is heroku"))
        s.insert_belief(_mk("bnew", "alpha beta deploy target is fly.io"))
        s.insert_edge(
            Edge(src="bnew", dst="bold", type=EDGE_SUPERSEDES, weight=1.0)
        )
        retrieve_v2(
            s, "alpha beta deploy",
            use_supersession_demote=True,
            supersession_treatment=SUPERSESSION_TREATMENT_DEMOTE,
            supersession_factor=0.5,
        )
        assert last_lane_telemetry().supersession_demoted > 0
        retrieve_v2(s, "alpha beta deploy", use_supersession_demote=False)
        assert last_lane_telemetry().supersession_demoted == 0
    finally:
        s.close()


# --- origin_tiebreak -----------------------------------------------------


def test_origin_tiebreak_decisions_counts_only_decided_pairs() -> None:
    """The count is pairs the origin term settled, not pairs sorted.

    Equal score + different origin priority is the only shape where the
    secondary key, rather than the id break, picked the order. A field
    set from `use_origin_tiebreak` would report a firing on every call
    with the flag on, including ones where no two candidates tied.
    """
    hi = _mk("b1", "x", origin=ORIGIN_USER_VALIDATED)
    lo = _mk("b2", "x", origin=ORIGIN_AGENT_INFERRED)
    other = _mk("b3", "x", origin=ORIGIN_AGENT_INFERRED)
    # Tied scores, differing priority -> decided.
    assert _origin_tiebreak_decisions([(1.0, "b1", hi), (1.0, "b2", lo)]) == 1
    # Tied scores, same priority -> the id break decided it.
    assert _origin_tiebreak_decisions(
        [(1.0, "b2", lo), (1.0, "b3", other)]
    ) == 0
    # Different scores -> the primary term decided it.
    assert _origin_tiebreak_decisions([(2.0, "b1", hi), (1.0, "b2", lo)]) == 0
    assert _origin_tiebreak_decisions([]) == 0


def test_origin_tiebreak_decided_reaches_the_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical content across two origins forces the tie.

    Removing the `_record_lane_fired` after the sort makes this read 0.
    """
    monkeypatch.setenv(ENV_ORIGIN_TIEBREAK, "1")
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "alpha beta", origin=ORIGIN_USER_VALIDATED))
        s.insert_belief(_mk("b2", "alpha beta", origin=ORIGIN_AGENT_INFERRED))
        retrieve_v2(s, "alpha beta", use_origin_tiebreak=True)
        assert last_lane_telemetry().origin_tiebreak_decided > 0
    finally:
        s.close()


def test_origin_tiebreak_decided_is_zero_when_the_lane_is_off() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "alpha beta", origin=ORIGIN_USER_VALIDATED))
        s.insert_belief(_mk("b2", "alpha beta", origin=ORIGIN_AGENT_INFERRED))
        retrieve_v2(s, "alpha beta", use_origin_tiebreak=False)
        assert last_lane_telemetry().origin_tiebreak_decided == 0
    finally:
        s.close()


# --- gamma / zeta rerank -------------------------------------------------


def test_gamma_rerank_scored_counts_beliefs_through_the_gamma_branch(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_USE_GAMMA_POSTERIOR_TEMPERATURE, "1")
    retrieve_v2(store, "alpha beta")
    tel = last_lane_telemetry()
    assert tel.gamma_rerank_scored > 0
    assert tel.zeta_rerank_scored == 0


def test_zeta_rerank_scored_counts_beliefs_through_the_zeta_branch(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_USE_ZETA_POSTERIOR_RERANK, "1")
    retrieve_v2(store, "alpha beta")
    tel = last_lane_telemetry()
    assert tel.zeta_rerank_scored > 0
    assert tel.gamma_rerank_scored == 0


def test_neither_rerank_records_on_the_default_path(
    store: MemoryStore,
) -> None:
    retrieve_v2(store, "alpha beta")
    tel = last_lane_telemetry()
    assert tel.gamma_rerank_scored == 0
    assert tel.zeta_rerank_scored == 0


# --- Both L1 arms --------------------------------------------------------
#
# `_l1_hits` carries two independent rerank loops — the BM25F arm and the
# FTS5 arm — and the record calls are duplicated across them. The default
# resolver sends every call down the BM25F arm, so a test suite that only
# runs the default leaves each FTS5-arm record call untested and a
# deletion there invisible. `use_bm25f=False` is the opt-out.


@pytest.mark.parametrize("use_bm25f", [True, False])
def test_gamma_records_on_both_l1_arms(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, use_bm25f: bool,
) -> None:
    monkeypatch.setenv(ENV_USE_GAMMA_POSTERIOR_TEMPERATURE, "1")
    retrieve_v2(store, "alpha beta", use_bm25f=use_bm25f)
    tel = last_lane_telemetry()
    assert tel.bm25f_used is use_bm25f
    assert tel.gamma_rerank_scored > 0


@pytest.mark.parametrize("use_bm25f", [True, False])
def test_zeta_records_on_both_l1_arms(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, use_bm25f: bool,
) -> None:
    monkeypatch.setenv(ENV_USE_ZETA_POSTERIOR_RERANK, "1")
    retrieve_v2(store, "alpha beta", use_bm25f=use_bm25f)
    tel = last_lane_telemetry()
    assert tel.bm25f_used is use_bm25f
    assert tel.zeta_rerank_scored > 0


@pytest.mark.parametrize("use_bm25f", [True, False])
def test_origin_tiebreak_records_on_both_l1_arms(
    monkeypatch: pytest.MonkeyPatch, use_bm25f: bool,
) -> None:
    monkeypatch.setenv(ENV_ORIGIN_TIEBREAK, "1")
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "alpha beta", origin=ORIGIN_USER_VALIDATED))
        s.insert_belief(_mk("b2", "alpha beta", origin=ORIGIN_AGENT_INFERRED))
        retrieve_v2(
            s, "alpha beta", use_origin_tiebreak=True, use_bm25f=use_bm25f,
        )
        assert last_lane_telemetry().origin_tiebreak_decided > 0
    finally:
        s.close()


@pytest.mark.parametrize("use_bm25f", [True, False])
def test_entity_persist_records_on_both_l1_arms(use_bm25f: bool) -> None:
    s = MemoryStore(":memory:")
    try:
        for i in range(3):
            s.insert_belief(_mk(f"b{i}", f"alpha beta ephemeral note {i}"))
            _add_entity(s, f"b{i}", f"#{900 + i}", "identifier")
        retrieve_v2(
            s, "alpha beta",
            use_entity_persist_demote=True, use_bm25f=use_bm25f,
        )
        assert last_lane_telemetry().entity_persist_demoted > 0
    finally:
        s.close()


@pytest.mark.parametrize("use_bm25f", [True, False])
def test_supersession_records_on_both_l1_arms(
    tmp_path: Path, use_bm25f: bool,
) -> None:
    s = MemoryStore(str(tmp_path / f"memory_{use_bm25f}.db"))
    try:
        s.insert_belief(_mk("bold", "alpha beta deploy target is heroku"))
        s.insert_belief(_mk("bnew", "alpha beta deploy target is fly.io"))
        s.insert_edge(
            Edge(src="bnew", dst="bold", type=EDGE_SUPERSEDES, weight=1.0)
        )
        retrieve_v2(
            s, "alpha beta deploy",
            use_supersession_demote=True,
            supersession_treatment=SUPERSESSION_TREATMENT_DEMOTE,
            supersession_factor=0.5,
            use_bm25f=use_bm25f,
        )
        assert last_lane_telemetry().supersession_demoted > 0
    finally:
        s.close()


# --- fan_effect ----------------------------------------------------------


def test_fan_effect_ranked_records_where_the_ordering_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2.5 has to return hits before anything is recorded."""
    monkeypatch.setenv(ENV_FAN_EFFECT, "1")
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "the deploy target is fly.io"))
        _add_entity(s, "b1", "fly.io", "identifier")
        retrieve_v2(s, "fly.io deploy", use_fan_effect=True)
        assert last_lane_telemetry().fan_effect_ranked > 0
    finally:
        s.close()


def test_fan_effect_ranked_is_zero_when_the_lane_is_off(
    store: MemoryStore,
) -> None:
    retrieve_v2(store, "alpha beta", use_fan_effect=False)
    assert last_lane_telemetry().fan_effect_ranked == 0


# --- hrr_structural ------------------------------------------------------


def test_hrr_structural_hit_is_recorded_on_the_early_return(
    tmp_path: Path,
) -> None:
    """The structural lane returns before `retrieve_with_tiers` runs.

    Before #1366 that early return left `last_lane_telemetry()` holding
    the *previous* call's counters, so a reader saw a stale snapshot
    presented as the current one. Now the path publishes its own.
    """
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        s.insert_belief(_mk("b1", "alpha beta source"))
        s.insert_belief(_mk("b2", "alpha beta target"))
        s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_CONTRADICTS,
                           weight=1.0))
        cache = HRRStructIndexCache(store=s, dim=512, seed=42)
        # A non-structural call first, so a stale snapshot would be
        # distinguishable from a fresh one.
        retrieve_v2(s, "alpha beta")
        stale = last_lane_telemetry()
        assert stale.hrr_structural_hit is False
        assert stale.l1 > 0
        result = retrieve_v2(
            s, "CONTRADICTS:b2",
            use_hrr_structural=True,
            hrr_struct_index_cache=cache,
            budget=10_000,
        )
        assert "b1" in [b.id for b in result.beliefs]
        tel = last_lane_telemetry()
        assert tel.hrr_structural_hit is True
        # ...and it is not the previous call's snapshot.
        assert tel.l1 == 0
    finally:
        s.close()


def test_hrr_structural_hit_is_false_on_a_fall_through(
    store: MemoryStore,
) -> None:
    """A non-marker prompt falls through; the lane did not take it."""
    retrieve_v2(store, "alpha beta", use_hrr_structural=True)
    assert last_lane_telemetry().hrr_structural_hit is False


# --- The recorder itself -------------------------------------------------


def test_record_lane_fired_ignores_a_zero_sized_record() -> None:
    """`_record_lane_fired(name, 0)` is "the lane packed nothing".

    Creating the key with a 0 would make `fired > 0` still False, but it
    would also make the counter dict claim the site ran. Keeping it
    absent means the dict itself is a list of what fired.
    """
    _reset_lane_firings()
    _record_lane_fired("cluster_packed", 0)
    assert _lane_fired("cluster_packed") == 0
    _record_lane_fired("cluster_packed", 3)
    _record_lane_fired("cluster_packed", 2)
    assert _lane_fired("cluster_packed") == 5


def test_supersession_penalty_is_still_log_additive() -> None:
    """The record must not have changed the returned value (#1187).

    The penalty is log-additive precisely because the composite score is
    log-domain and routinely negative; a refactor that turned the return
    into the recorded count would invert the lane.
    """
    assert _supersession_penalty(
        frozenset({"b1"}), "b1", 0.5,
    ) == pytest.approx(math.log(0.5))


# --- the two reset sites, pinned INDEPENDENTLY ---------------------------
#
# `_reset_lane_firings()` is called from two places: `retrieve_with_tiers`
# and `retrieve_v2`'s structural early-return branch. Review found they
# mask each other — deleting either one alone left the whole suite green,
# because every existing test that would notice passes through the other.
# Jointly pinned is not pinned: a refactor removing one site would ship.


def test_the_retrieve_with_tiers_reset_is_pinned_on_its_own(
    store: MemoryStore,
) -> None:
    """Exercises the tiers path only, so the structural reset cannot cover.

    `use_hrr_structural=False` keeps `retrieve_v2` out of the branch that
    holds the other reset, so these counters can only have been zeroed by
    the one inside `retrieve_with_tiers`. Deleting that call alone turns
    this red; with the structural branch left enabled it does not, because
    its reset runs first and masks the deletion.
    """
    retrieve_v2(store, "alpha beta", use_hrr_structural=False)
    first = last_lane_telemetry().cluster_packed
    assert first > 0
    retrieve_v2(store, "alpha beta", use_hrr_structural=False)
    assert last_lane_telemetry().cluster_packed == first


# The sibling reset inside `retrieve_v2`'s structural branch
# (retrieval.py:5067) is NOT independently pinned, and that is a statement
# about the code rather than a gap in this file.
#
# That branch publishes `LaneTelemetry(locked=..., hrr_structural_hit=...)`
# at retrieval.py:5081 — every other field takes its dataclass default, so
# no accumulator-backed counter it exposes can carry a stale value. The one
# field it does read, `hrr_structural_hit`, is recorded by
# `_route_structural_query` on exactly the path that early-returns, so it
# is True with or without the reset.
#
# So the reset there is defensive, not load-bearing, and no assertion can
# distinguish its presence. Writing one that appeared to would be the
# defect this whole issue is about. If the publish ever widens to pass
# through more accumulator fields, that reset becomes load-bearing and
# should get a pin at the same time.
#
# The property that IS observable on that path — that it publishes its own
# snapshot rather than leaving the previous call's — is covered by
# `test_hrr_structural_hit_is_recorded_on_the_early_return` above, which
# asserts `tel.l1 == 0` after a textual call that set `l1 > 0`.
