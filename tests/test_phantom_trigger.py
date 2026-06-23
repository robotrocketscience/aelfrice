"""Deterministic phantom-opportunity trigger (#980 item 3).

Covers the pure detector (gap conditions, gate-skip exclusion, thin-query
floor, reason classification), the normalize/dedup-key stability, and the
per-session bound (dedup + rate cap + fail-soft on no state dir).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.phantom_trigger import (
    DEFAULT_MAX_PER_SESSION,
    PhantomOpportunity,
    dedup_key_for,
    detect_phantom_opportunity,
    normalize_query,
    register_opportunity,
)


# ---------------------------------------------------------------------------
# normalize_query / dedup_key_for — deterministic, order-independent
# ---------------------------------------------------------------------------


def test_normalize_drops_short_tokens_and_sorts() -> None:
    # "in", "a", "db" are < 3 chars and dropped; result is sorted-unique.
    assert normalize_query("About Session IDs in a DB") == (
        "about",
        "ids",
        "session",
    )


def test_normalize_is_order_independent() -> None:
    assert normalize_query("session id propagation") == normalize_query(
        "propagation id session"
    )


def test_dedup_key_stable_across_word_order() -> None:
    assert dedup_key_for("session id propagation") == dedup_key_for(
        "propagation session id"
    )


def test_dedup_key_differs_for_different_topics() -> None:
    assert dedup_key_for("session id propagation") != dedup_key_for(
        "retrieval ranking lanes"
    )


# ---------------------------------------------------------------------------
# detect_phantom_opportunity — gap conditions
# ---------------------------------------------------------------------------


def test_no_hits_real_query_is_a_gap() -> None:
    opp = detect_phantom_opportunity(
        "how does session id propagation work", 0, gate_skipped=False
    )
    assert opp is not None
    assert opp.reason == "no_hits"
    assert opp.n_hits == 0


def test_hits_present_is_not_a_gap() -> None:
    assert (
        detect_phantom_opportunity(
            "how does session id propagation work", 3, gate_skipped=False
        )
        is None
    )


def test_gate_skipped_is_never_a_gap() -> None:
    # Trivial acks / system envelopes the prompt-shape gate skips need no
    # belief — they must never trigger generation even with zero hits.
    assert (
        detect_phantom_opportunity("ok thanks", 0, gate_skipped=True) is None
    )


def test_thin_query_is_not_a_gap() -> None:
    # Fewer than min_query_tokens meaningful tokens → can't anchor a phantom.
    assert detect_phantom_opportunity("hi", 0, gate_skipped=False) is None


def test_below_floor_reason_when_min_hits_raised() -> None:
    opp = detect_phantom_opportunity(
        "how does session id propagation work",
        2,
        gate_skipped=False,
        min_hits=3,
    )
    assert opp is not None
    assert opp.reason == "below_floor"
    assert opp.n_hits == 2


def test_two_token_query_rejected_by_default_floor() -> None:
    assert (
        detect_phantom_opportunity("session propagation", 0, gate_skipped=False)
        is None
    )


def test_two_token_query_accepted_with_lower_floor() -> None:
    opp = detect_phantom_opportunity(
        "session propagation", 0, gate_skipped=False, min_query_tokens=2
    )
    assert opp is not None


# ---------------------------------------------------------------------------
# register_opportunity — dedup + rate cap + fail-soft
# ---------------------------------------------------------------------------


def _opp(query: str) -> PhantomOpportunity:
    got = detect_phantom_opportunity(query, 0, gate_skipped=False)
    assert got is not None
    return got


def test_register_first_time_emits(tmp_path: Path) -> None:
    assert register_opportunity(
        tmp_path, "sess", _opp("session identifier propagation")
    )


def test_register_dedups_same_topic(tmp_path: Path) -> None:
    opp = _opp("session identifier propagation")
    assert register_opportunity(tmp_path, "sess", opp) is True
    # Same normalized topic (different word order) → deduped.
    again = _opp("propagation identifier session")
    assert register_opportunity(tmp_path, "sess", again) is False


def test_register_distinct_topics_both_emit(tmp_path: Path) -> None:
    assert register_opportunity(
        tmp_path, "sess", _opp("session identifier propagation")
    )
    assert register_opportunity(tmp_path, "sess", _opp("retrieval ranking lanes"))


def test_register_rate_cap(tmp_path: Path) -> None:
    queries = [
        "session identifier propagation",
        "retrieval ranking lanes",
        "belief posterior temperature",
        "phantom lifecycle audit",  # 4th — exceeds default cap of 3
    ]
    results = [
        register_opportunity(tmp_path, "sess", _opp(q)) for q in queries
    ]
    assert results == [True, True, True, False]
    assert DEFAULT_MAX_PER_SESSION == 3


def test_register_per_session_isolation(tmp_path: Path) -> None:
    opp = _opp("session identifier propagation")
    assert register_opportunity(tmp_path, "sess-a", opp) is True
    # Same topic, different session → not deduped (separate sidecar).
    assert register_opportunity(tmp_path, "sess-b", opp) is True


def test_register_failsoft_when_no_state_dir() -> None:
    # In-memory store (None state dir): can't bound → errs quiet (no emit).
    assert (
        register_opportunity(None, "sess", _opp("session identifier propagation"))
        is False
    )


# ---------------------------------------------------------------------------
# Full user_prompt_submit() integration
# ---------------------------------------------------------------------------

import io  # noqa: E402
import json  # noqa: E402

from aelfrice.hook import user_prompt_submit  # noqa: E402

_HINT_OPEN = "<aelfrice-phantom-opportunity>"


def _ups_payload(prompt: str, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _feed_events(db_dir: Path) -> list[dict[str, object]]:
    feed = db_dir / "feed.jsonl"
    if not feed.exists():
        return []
    return [json.loads(x) for x in feed.read_text().splitlines() if x]


def test_ups_emits_hint_on_gap_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"  # empty store → real query gets 0 hits
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_PHANTOM_TRIGGER", "1")
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")

    sout, serr = io.StringIO(), io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(
            _ups_payload("how does the submarine deployment pipeline work")
        ),
        stdout=sout,
        stderr=serr,
    )
    assert rc == 0
    out = sout.getvalue()
    assert _HINT_OPEN in out
    assert "/aelf:wonder" in out
    gc = [e for e in _feed_events(tmp_path) if e.get("event") == "phantom.opportunity"]
    assert len(gc) == 1
    assert gc[0]["reason"] == "no_hits"


def test_ups_no_hint_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_PHANTOM_TRIGGER", raising=False)
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")

    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(
            _ups_payload("how does the submarine deployment pipeline work")
        ),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert _HINT_OPEN not in sout.getvalue()
    assert _feed_events(tmp_path) == []


def test_ups_hint_deduped_within_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_PHANTOM_TRIGGER", "1")
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")

    def _fire(prompt: str) -> str:
        sout = io.StringIO()
        user_prompt_submit(
            stdin=io.StringIO(_ups_payload(prompt)),
            stdout=sout,
            stderr=io.StringIO(),
        )
        return sout.getvalue()

    first = _fire("how does the submarine deployment pipeline work")
    # Same normalized topic, reworded → deduped, no second hint.
    second = _fire("the submarine deployment pipeline — how does work")
    assert _HINT_OPEN in first
    assert _HINT_OPEN not in second
