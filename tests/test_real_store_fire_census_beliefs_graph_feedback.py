"""Planted-store tests for the beliefs-graph-feedback census probes (#1887).

Each test plants a store under `tmp_path` with aelfrice's own code (CLI
handlers, hook entry points, ingest, retrieval), then checks the probe on
a read-only census context. Every traceable probe is also checked silent
on an empty store, and every untraceable probe reports `untraceable`.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import promotion
from aelfrice import sentiment_feedback as sf
from aelfrice.deferred_feedback import enqueue_retrieval_exposures
from aelfrice.derivation_worker import run_worker
from aelfrice.feedback import apply_feedback
from aelfrice.hook_search import record_retrieval
from aelfrice.retrieval import retrieve
from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_beliefs_graph_feedback as pb
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

P = {p.item_id: p for p in pb.PROBES}
OLD_TS = "2020-01-01T00:00:00Z"
LATER_TS = "2030-01-01T00:00:00+00:00"
SENT_BELIEF = "The census tool reads sqlite copies in read-only mode."
SENT_QUERY = "how does the census tool read sqlite copies"

_FEATURE_FLAGS = (
    "AELFRICE_AUTO_RELATIONSHIPS", "AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE",
    "AELFRICE_FEEDBACK_SENTIMENT_NEGATIVE", "AELFRICE_LOG_REWORDED_SUPPORT",
    "AELFRICE_EXPOSURE_UPDATES_POSTERIOR", "AELFRICE_VALENCE_PROPAGATION",
    "AELFRICE_HOOK_AUDIT", "AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE",
)


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    store = plant(tmp_path, monkeypatch)
    monkeypatch.setenv("AELFRICE_FEED_LOG", "1")
    for var in _FEATURE_FLAGS:
        monkeypatch.delenv(var, raising=False)
    yield store
    store.close()


def check(ps: PlantedStore, item_id: str, status: str, count: int | None = None,
          **detail: int) -> cc.Result:
    r = ps.probe(P[item_id])
    assert r.status == status, (item_id, r.to_json())
    if count is not None:
        assert r.count == count, (item_id, r.to_json())
    for k, v in detail.items():
        assert r.detail.get(k, 0) == v, (item_id, k, r.to_json())
    return r


def only_lock_id(ps: PlantedStore) -> str:
    c = ps.ctx()
    try:
        rows = c.require_db().execute(
            "SELECT id FROM beliefs WHERE lock_level = 'user'").fetchall()
    finally:
        c.close()
    assert len(rows) == 1
    return str(rows[0][0])


def ups(ps: PlantedStore, prompt: str, session: str = "s1") -> int:
    return ps.hook("main", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                            "session_id": session, "cwd": str(ps.repo)})


def raw_update(ps: PlantedStore, sql: str) -> None:
    """A direct write for a column no current aelfrice code writes."""
    ps.close()
    conn = sqlite3.connect(str(ps.db_file))
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


# --- module shape and empty store -------------------------------------------

EXPECTED_IDS = frozenset(f"A{i:03d}" for i in range(1, 38))


def test_module_ids_match_the_subsystem() -> None:
    assert {p.item_id for p in pb.PROBES} == EXPECTED_IDS
    assert len(pb.PROBES) == len(EXPECTED_IDS)
    assert pb.TRACEABLE | pb.UNTRACEABLE == EXPECTED_IDS
    assert not pb.TRACEABLE & pb.UNTRACEABLE


@pytest.mark.parametrize("item_id", sorted(pb.TRACEABLE))
def test_empty_store_is_silent(ps: PlantedStore, item_id: str) -> None:
    check(ps, item_id, "silent", 0)


@pytest.mark.parametrize("item_id", sorted(pb.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(ps: PlantedStore, item_id: str) -> None:
    r = ps.probe(P[item_id])
    assert r.status == "untraceable" and r.count is None and r.note


# --- pre-aelfrice schema ------------------------------------------------------


def _legacy_ctx(tmp_path: Path) -> cc.Ctx:
    d = tmp_path / "legacy"
    d.mkdir()
    db = d / "memory.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            "CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, belief_type TEXT, "
            "beta_param REAL, locked INTEGER, created_at TEXT);"
            "CREATE TABLE edges (from_id TEXT, to_id TEXT, edge_type TEXT, created_at TEXT);"
            "CREATE TABLE feedback_history (id INTEGER PRIMARY KEY, belief_id TEXT, "
            "valence REAL, source TEXT, created_at TEXT);"
            "INSERT INTO beliefs VALUES ('b1', 'x', 'correction', 1.0, 0, '2026-05-01');"
            "INSERT INTO beliefs VALUES ('b2', 'y', 'factual', 1.0, 1, '2026-05-01');"
            "INSERT INTO edges VALUES ('b1', 'b2', 'SUPPORTS', '2026-05-01');"
            "INSERT INTO edges VALUES ('b1', 'b2', 'RESOLVES', '2026-05-01');")
        conn.commit()
    finally:
        conn.close()
    c = cc.Ctx("legacy01", d, None, False, db, "legacy")
    c.open()
    return c


@pytest.mark.parametrize(("item_id", "status", "count"), [
    ("A002", "silent", 0), ("A005", "untraceable", None), ("A012", "untraceable", None),
    ("A018", "untraceable", None), ("A025", "silent", None), ("A026", "silent", 0),
    ("A029", "untraceable", None), ("A030", "untraceable", None), ("A037", "silent", 0),
])
def test_pre_aelfrice_schema(tmp_path: Path, item_id: str, status: str,
                             count: int | None) -> None:
    c = _legacy_ctx(tmp_path)
    try:
        r = cc.run_one(P[item_id], c)
    finally:
        c.close()
    assert (r.status, r.count) == (status, count), (item_id, r.to_json())


def test_pre_aelfrice_schema_keeps_legacy_detail(tmp_path: Path) -> None:
    c = _legacy_ctx(tmp_path)
    try:
        a002 = cc.run_one(P["A002"], c)
        a005 = cc.run_one(P["A005"], c)
        a012 = cc.run_one(P["A012"], c)
        a029 = cc.run_one(P["A029"], c)
        a037 = cc.run_one(P["A037"], c)
    finally:
        c.close()
    assert a002.detail["aelfrice_lock_schema"] == 0
    assert a005.detail["legacy_schema"] == 1 and a005.detail["legacy_resolves_edges"] == 1
    assert a012.detail["predecessor_belief_type_correction"] == 1
    assert a029.detail["legacy_edge_type_resolves"] == 1
    assert a037.detail["edges_supports"] == 1


# --- feedback and promotion CLI ---------------------------------------------


def test_a002_confirm_on_lock(ps: PlantedStore) -> None:
    assert ps.lock("Never run the census against a real store.") == 0
    lid = only_lock_id(ps)
    assert ps.cli("confirm", lid) == 0
    check(ps, "A002", "fired", 2, fb_user_confirmed=1, on_locked_after_lock=1,
          feed_confirm_rows=1, aelfrice_lock_schema=1)
    # confirm is exempt from the floor, so A018 must not count it
    check(ps, "A018", "silent", 0, user_locks=1)


def test_a003_demote_both_reverts(ps: PlantedStore) -> None:
    x = ps.belief("An agent inferred belief about the test runner.")
    s = ps.belief("A phantom about the test runner cache.", type="speculative",
                  origin="speculative")
    assert ps.cli("validate", x) == 0
    promotion.promote_on_evidence(ps.open(), s)
    assert ps.cli("demote", x) == 0
    assert ps.cli("demote", s) == 0
    check(ps, "A003", "fired", 2, revert_to_agent_inferred=1, revert_evidence=1)


def test_a008_unlock(ps: PlantedStore) -> None:
    assert ps.lock("A lock that will be unlocked.") == 0
    assert ps.cli("unlock", only_lock_id(ps)) == 0
    check(ps, "A008", "ambiguous", 1)


def test_a009_validate_and_promote_with_label_and_scope(ps: PlantedStore) -> None:
    x = ps.belief("An agent inferred belief about formatting.")
    y = ps.belief("Another agent inferred belief about formatting.")
    assert ps.cli("validate", x) == 0
    assert ps.cli("promote", y, "--source", "slash", "--to-scope", "global") == 0
    check(ps, "A009", "fired", 2, user_validated=1, custom_label=1, scope_rows=1,
          feed_promoted_rows=2)


def test_a009_ignores_phantom_and_evidence_promotions(ps: PlantedStore) -> None:
    s = ps.belief("Run the census only against byte copies of each store.",
                  type="speculative", origin="speculative")
    assert ps.lock("Run the census only against byte copies of each store.") == 0
    check(ps, "A023", "fired", 1)
    t = ps.belief("A second phantom about byte copies.", type="speculative",
                  origin="speculative")
    promotion.promote_on_evidence(ps.open(), t)
    check(ps, "A009", "silent", 0)
    assert s != t


def test_a016_feedback_used_and_harmful(ps: PlantedStore) -> None:
    a = ps.belief("Alpha belief about the build cache.")
    b = ps.belief("Beta belief about the build cache.")
    assert ps.cli("feedback", a, "used") == 0
    assert ps.cli("feedback", b, "harmful") == 0
    check(ps, "A016", "fired", 2, used=1, harmful=1, feed_applied=2)


def test_a018_floor_on_locked_feedback(ps: PlantedStore) -> None:
    assert ps.lock("Locked rule about the census sandbox.") == 0
    lid = only_lock_id(ps)
    assert ps.cli("feedback", lid, "harmful") == 0
    check(ps, "A018", "fired", 1, locked_posterior_off_prior=0, user_locks=1,
          floor_rows_any_time=1, floor_rows_post_4_3_0=1)
    check(ps, "A016", "fired", 1, feed_locked_not_applied=1)


def test_a018_ignores_feedback_dated_before_the_lock(ps: PlantedStore) -> None:
    text = "A belief that is criticised first and locked later."
    bid = ps.belief(text)
    apply_feedback(ps.open(), bid, -1.0, "user", now=OLD_TS)
    assert ps.lock(text) == 0
    assert only_lock_id(ps) == bid
    check(ps, "A018", "silent", 0, floor_rows_any_time=1)
    check(ps, "A016", "fired", 1)


def test_a026_a037_propagation(ps: PlantedStore) -> None:
    a = ps.belief("Alpha belief about the linker.")
    b = ps.belief("Beta belief about the linker.")
    ps.edge(a, b, "SUPPORTS")
    assert ps.cli("feedback", a, "used") == 0
    check(ps, "A037", "fired", 1, src_user=1, edges_supports=1, trig_post_ship=1,
          trig_post_ship_with_out_edge=1)
    check(ps, "A026", "fired", 1, abs_ge_0_5=1, abs_lt_0_05=0,
          propagation_rows_after_ship=1, eligible_triggers_with_edges=1)


def test_a037_silent_when_propagation_disabled(ps: PlantedStore,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_VALENCE_PROPAGATION", "0")
    a = ps.belief("Alpha belief about the linker.")
    b = ps.belief("Beta belief about the linker.")
    ps.edge(a, b, "SUPPORTS")
    assert ps.cli("feedback", a, "used") == 0
    # the trigger is visible beside the silent result
    check(ps, "A037", "silent", 0, trig_post_ship=1)
    check(ps, "A026", "silent", 0, eligible_triggers_with_edges=1)
    check(ps, "A016", "fired", 1)


def test_a023_phantom_lock_match(ps: PlantedStore) -> None:
    ps.belief("Run the census only against byte copies of each store.",
              type="speculative", origin="speculative")
    assert ps.lock("Run the census only against byte copies of each store.") == 0
    check(ps, "A023", "fired", 1)


# --- contradiction graph -----------------------------------------------------


def test_a005_a011_a029_resolve(ps: PlantedStore) -> None:
    a = ps.belief("Builds use the fast linker.", origin="agent_inferred")
    b = ps.belief("Builds use the slow linker.", origin="user_validated")
    ps.edge(a, b, "CONTRADICTS")
    c = ps.belief("Tests run in parallel.", origin="agent_inferred")
    d = ps.belief("Tests run serially.", origin="agent_inferred")
    ps.edge(c, d, "CONTRADICTS")
    check(ps, "A005", "silent", 0, unresolved_pairs=2)
    check(ps, "A029", "silent", 0, contradicts=2, tiebreak_rows_total=0)
    assert ps.cli("resolve") == 0
    check(ps, "A005", "fired", 2, resolves_edges=2, supersedes_edges=2, unresolved_pairs=0)
    check(ps, "A011", "fired", 1, by_class=1, by_tiebreak=1,
          user_validated_beats_agent_inferred=1,
          agent_inferred_beats_agent_inferred_by_recency=1)
    check(ps, "A029", "fired", 2, weight_0=2, supersedes=2, tiebreak_rows_total=2,
          tiebreak_rows_after_5_2_0=2)


def test_a017_a027_ingest_detector(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_AUTO_RELATIONSHIPS", "1")
    ps.ingest("The deploy script always runs the database migrations first.", session_id="s1")
    ps.ingest("The deploy script never runs the database migrations first.", session_id="s2")
    check(ps, "A017", "fired", 1, contradicts_total=1, contradicts_with_anchor=0)
    check(ps, "A027", "fired", 1, detector_contradicts=1, potentially_stale=0,
          max_contradicts_per_belief=1)


def test_a017_silent_with_detector_off(ps: PlantedStore) -> None:
    ps.ingest("The deploy script always runs the database migrations first.", session_id="s1")
    ps.ingest("The deploy script never runs the database migrations first.", session_id="s2")
    check(ps, "A017", "silent", 0)
    check(ps, "A027", "silent", 0)


def test_a017_ignores_triple_edges(ps: PlantedStore) -> None:
    a = ps.belief("Option alpha for the cache.")
    b = ps.belief("Option beta for the cache.")
    lo, hi = sorted((a, b))
    ps.edge(lo, hi, "CONTRADICTS", anchor_text="alpha contradicts beta")
    check(ps, "A017", "silent", 0, contradicts_total=1, contradicts_with_anchor=1)


def test_a025_a027_detect_stale(ps: PlantedStore) -> None:
    ps.ingest("The deploy script usually runs the database migrations first.", session_id="s1")
    ps.ingest("The deploy script sometimes runs the database migrations first.", session_id="s2")
    assert ps.cli("doctor", "--detect-stale") == 0
    check(ps, "A025", "fired", 1, weight_1=1)
    check(ps, "A027", "fired", 1, detector_contradicts=0, potentially_stale=1)


# --- classification ---------------------------------------------------------


def test_a012_correction_detector(ps: PlantedStore) -> None:
    assert ps.ingest("Don't use the old linker, we discussed this already.",
                     session_id="s1") == 1
    # host-LLM override path: type=correction without the detector
    store = ps.open()
    store.record_ingest(source_kind="filesystem", raw_text="The cache sits in the tmp folder.",
                        source_path="doc:notes.md",
                        raw_meta={"override_belief_type": "correction"})
    run_worker(store)
    c = ps.ctx()
    try:
        total = c.count("SELECT COUNT(*) FROM beliefs WHERE type = 'correction'")
    finally:
        c.close()
    assert total == 2
    check(ps, "A012", "fired", 1, active=1, via_llm_override=1)


def test_a012_silent_on_factual(ps: PlantedStore) -> None:
    ps.ingest("The cache is fine.", session_id="s1")
    check(ps, "A012", "silent", 0)


# --- deferred feedback queue --------------------------------------------------


def test_a014_enqueue_on_retrieve(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ps.belief("The census tool reads sqlite copies in read-only mode.")
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "1")
    hits = retrieve(ps.open(), "census sqlite copies")
    assert len(hits) == 1
    check(ps, "A014", "fired", 1, total=1, pre_flip=0, post_flip=1)


def test_a014_silent_with_enqueue_off(ps: PlantedStore) -> None:
    ps.belief("The census tool reads sqlite copies in read-only mode.")
    hits = retrieve(ps.open(), "census sqlite copies")
    assert len(hits) == 1
    check(ps, "A014", "silent", 0, total=0)


def test_a014_ignores_rows_from_before_the_flip(ps: PlantedStore) -> None:
    bid = ps.belief("A belief exposed while enqueue was still the default.")
    enqueue_retrieval_exposures(ps.open(), [bid], now=OLD_TS)
    check(ps, "A014", "silent", 0, total=1, pre_flip=1, post_flip=0)


def test_a014_ignores_other_event_types(ps: PlantedStore) -> None:
    # no production writer uses another event_type today; the store API allows one
    bid = ps.belief("A belief queued under a different event type.")
    ps.open().enqueue_deferred_feedback(bid, event_type="search_exposure", enqueued_at=LATER_TS)
    check(ps, "A014", "silent", 0, total=0)


def test_a006_sweep_gc(ps: PlantedStore) -> None:
    bid = ps.belief("Old exposure target belief.")
    enqueue_retrieval_exposures(ps.open(), [bid, bid], now=OLD_TS)
    check(ps, "A006", "silent", 0, queue_rows=2)
    assert ps.cli("sweep-feedback") == 0
    check(ps, "A006", "silent", 0, queue_rows=2)
    assert ps.cli("sweep-feedback", "--gc") == 0
    check(ps, "A006", "ambiguous", 2, queue_rows=0)


# --- exposure ------------------------------------------------------------------


def test_a030_exposure_audit_only(ps: PlantedStore) -> None:
    a = ps.belief("Alpha belief surfaced by the hook.")
    b = ps.belief("Beta belief linked to alpha.")
    ps.belief("Gamma belief that nothing touches.")
    ps.edge(a, b, "SUPPORTS")
    recorded = record_retrieval(ps.open(), [ps.open().get_belief(a)])
    assert recorded == 1
    # the default mode records exposure but moves no posterior
    check(ps, "A030", "silent", 0, hook_rows_post_4_0_0=1, prop_hook_post=0,
          hook_only_beliefs=1, at_prior=1)
    check(ps, "A037", "silent", 0)


def test_a030_legacy_mode_bumps_and_propagates(ps: PlantedStore,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_EXPOSURE_UPDATES_POSTERIOR", "1")
    a = ps.belief("Alpha belief surfaced by the hook.")
    b = ps.belief("Beta belief linked to alpha.")
    ps.belief("Gamma belief that nothing touches.")
    ps.edge(a, b, "SUPPORTS")
    recorded = record_retrieval(ps.open(), [ps.open().get_belief(a)])
    assert recorded == 1
    check(ps, "A030", "fired", 2, prop_hook_post=1, bumped_incl_post_rows=1)


def test_a030_rows_before_the_release_are_unobservable(ps: PlantedStore) -> None:
    bid = ps.belief("A belief exposed before the 4.0.0 release.")
    ps.open().insert_feedback_event(bid, 0.1, "hook", OLD_TS)
    check(ps, "A030", "silent", 0, hook_rows_pre_4_0_0=1, hook_rows_post_4_0_0=0)


def test_a028_silent_with_injection_rows(ps: PlantedStore) -> None:
    bid = ps.belief("A belief that was injected.")
    ps.injection(bid)
    check(ps, "A028", "silent", 0, rows=1, scored=0, injected_post_removal=1)


def test_a028_counts_rows_scored_after_the_removal(ps: PlantedStore) -> None:
    # No current code writes `referenced`; the removed sweeper did. A raw
    # write stands in for it, once dated before the removal and once after.
    a = ps.belief("A belief scored by the old sweeper.")
    b = ps.belief("A belief scored after the sweeper was removed.")
    ps.injection(a, turn_id="t1")
    ps.injection(b, turn_id="t2")
    raw_update(ps, "UPDATE injection_events SET referenced = 1, "
               f"referenced_at = '2026-09-01T00:00:00Z' WHERE belief_id = '{a}'")
    check(ps, "A028", "silent", 0, scored=1, scored_pre_removal=1)
    raw_update(ps, "UPDATE injection_events SET referenced = 0, "
               f"referenced_at = '2026-10-07T00:00:00Z' WHERE belief_id = '{b}'")
    check(ps, "A028", "fired", 1, scored=2, scored_post_removal=1, referenced_1=1)


# --- support evidence --------------------------------------------------------------


def test_a031_reworded_support(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_LOG_REWORDED_SUPPORT", "1")
    ps.belief("The nightly backup job copies the database files to the archive bucket.",
              session_id="s0")
    # supporter must postdate the target; a same-second ts can sort before it
    ps.ingest_turns([{"role": "user", "session_id": "s9", "ts": LATER_TS,
                      "text": "The backup job copies database files to the archive bucket."}])
    check(ps, "A031", "fired", 1, applied=0, distinct_supporters=1)


def test_a031_silent_with_log_off(ps: PlantedStore) -> None:
    ps.belief("The nightly backup job copies the database files to the archive bucket.",
              session_id="s0")
    ps.ingest_turns([{"role": "user", "session_id": "s9", "ts": LATER_TS,
                      "text": "The backup job copies database files to the archive bucket."}])
    check(ps, "A031", "silent", 0)


# --- sentiment lane (UserPromptSubmit hook) ---------------------------------------


def test_a024_a034_positive_sentiment(ps: PlantedStore) -> None:
    ps.belief(SENT_BELIEF)
    assert ups(ps, SENT_QUERY) == 0
    assert ups(ps, "perfect") == 0
    check(ps, "A024", "fired", 1, audit_positive_applied=1)
    check(ps, "A034", "fired", 1, strong_1_5=1, base_1_0=0, store_rows_abs_1_5=1)
    check(ps, "A033", "fired", 1, propagation_rows=0, groups_summing_one_unit=1)
    assert ups(ps, SENT_QUERY) == 0
    assert ups(ps, "thanks") == 0
    check(ps, "A034", "fired", 2, strong_1_5=1, base_1_0=1)
    check(ps, "A024", "fired", 2)


def test_a024_abstains_without_prior_injection(ps: PlantedStore) -> None:
    ps.belief(SENT_BELIEF)
    assert ups(ps, "perfect", session="fresh") == 0
    check(ps, "A024", "silent", 0, audit_no_prior_injection=1)
    check(ps, "A034", "fired", 1)


def test_a032_long_prompt_writes_no_sentiment_row(ps: PlantedStore) -> None:
    ps.belief(SENT_BELIEF)
    assert ups(ps, SENT_QUERY) == 0
    assert ups(ps, "perfect " + "x" * 250) == 0
    check(ps, "A034", "silent", 0)
    check(ps, "A024", "silent", 0)


def test_a022_negative_disabled_then_enabled(ps: PlantedStore,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    ps.belief(SENT_BELIEF)
    assert ups(ps, SENT_QUERY) == 0
    assert ups(ps, "that's wrong") == 0
    check(ps, "A022", "fired", 1, audit_negative_disabled=1, store_negative_rows=0)
    monkeypatch.setenv("AELFRICE_FEEDBACK_SENTIMENT_NEGATIVE", "1")
    assert ups(ps, SENT_QUERY) == 0
    assert ups(ps, "that's wrong") == 0
    check(ps, "A022", "fired", 2, audit_negative_disabled=1, store_negative_rows=1,
          audit_negative_other=1)
    check(ps, "A024", "silent", 0)


def test_a033_equal_split_and_lock_aware_divisor(ps: PlantedStore) -> None:
    a = ps.belief("Alpha belief on the prior turn.")
    b = ps.belief("Beta belief on the prior turn.")
    sig = sf.detect_sentiment("perfect")
    assert sig is not None and sig.valence == 1.5
    sf.apply_sentiment_to_pending(ps.open(), sig, [a, b], now="2026-01-01T00:00:00Z")
    check(ps, "A033", "fired", 2, fires_by_timestamp=1, groups_summing_one_unit=1,
          rows_fractional_share=2, propagation_rows=0)
    assert ps.lock("A locked belief on the prior turn.") == 0
    lid = only_lock_id(ps)
    # dated after locked_at so the floor row is visible to A018
    sf.apply_sentiment_to_pending(ps.open(), sig, [a, lid], now="2030-01-02T00:00:00Z")
    # receivers = 1: the unlocked id takes the whole 1.5, the lock an audit-only 1.5
    check(ps, "A033", "fired", 4, fires_by_timestamp=2, rows_fractional_share=2)
    check(ps, "A018", "fired", 1)
