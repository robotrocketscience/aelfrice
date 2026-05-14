"""Sentiment-feedback hook lane wired into UserPromptSubmit (#606)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    AUDIT_FILENAME,
    AUDIT_HOOK_SENTIMENT_FEEDBACK,
    AUDIT_HOOK_USER_PROMPT_SUBMIT,
    _audit_path_for_db,
    _load_aelfrice_toml,
    _load_prior_ups_belief_ids,
    apply_sentiment_feedback,
    read_hook_audit,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.sentiment_feedback import (
    ENV_SENTIMENT,
    SENTIMENT_INFERRED_SOURCE,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _read_belief(db_path: Path, belief_id: str) -> Belief:
    store = MemoryStore(str(db_path))
    try:
        b = store.get_belief(belief_id)
        assert b is not None, f"belief {belief_id} missing"
        return b
    finally:
        store.close()


def _payload(prompt: str, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _enable_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SENTIMENT, "1")


def _disable_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_SENTIMENT, raising=False)


# ---------------------------------------------------------------------------
# _load_aelfrice_toml
# ---------------------------------------------------------------------------


def test_load_toml_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert _load_aelfrice_toml(start=tmp_path) == {}


def test_load_toml_returns_parsed_dict(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[feedback]\nsentiment_from_prose = true\n",
        encoding="utf-8",
    )
    parsed = _load_aelfrice_toml(start=tmp_path)
    assert parsed.get("feedback") == {"sentiment_from_prose": True}


def test_load_toml_returns_empty_on_malformed_toml(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text("not = toml = at all", encoding="utf-8")
    serr = io.StringIO()
    assert _load_aelfrice_toml(start=tmp_path, stderr=serr) == {}
    assert "malformed TOML" in serr.getvalue()


# ---------------------------------------------------------------------------
# _load_prior_ups_belief_ids
# ---------------------------------------------------------------------------


def test_prior_ups_returns_empty_when_audit_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_db(monkeypatch, tmp_path / "memory.db")
    assert _load_prior_ups_belief_ids("s1") == []


def test_prior_ups_returns_empty_when_session_id_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_db(monkeypatch, tmp_path / "memory.db")
    assert _load_prior_ups_belief_ids("") == []


def test_prior_ups_filters_by_session_and_takes_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
            "session_id": "s_other",
            "beliefs": [{"id": "X1"}],
        },
        {
            "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
            "session_id": "s1",
            "beliefs": [{"id": "A1"}, {"id": "A2"}],
        },
        {
            "hook": "session_start",
            "session_id": "s1",
            "beliefs": [{"id": "Z9"}],
        },
        {
            "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
            "session_id": "s1",
            "beliefs": [{"id": "B1"}, {"id": "B2"}, {"id": "B3"}],
        },
    ]
    audit_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    assert _load_prior_ups_belief_ids("s1") == ["B1", "B2", "B3"]
    assert _load_prior_ups_belief_ids("s_other") == ["X1"]
    assert _load_prior_ups_belief_ids("s_missing") == []


def test_prior_ups_skips_non_dict_belief_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
                "session_id": "s1",
                "beliefs": ["junk", {"no_id": True}, {"id": "OK"}, {"id": ""}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _load_prior_ups_belief_ids("s1") == ["OK"]


# ---------------------------------------------------------------------------
# apply_sentiment_feedback
# ---------------------------------------------------------------------------


def test_apply_returns_zero_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    # Pre-write a UPS audit row so the lane has prior beliefs to bump
    # if it were enabled — the disabled gate must short-circuit before
    # this is read.
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
                "session_id": "s1",
                "beliefs": [{"id": "F1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n = apply_sentiment_feedback("no, wrong", "s1")

    assert n == 0
    # No sentiment_feedback row should have been written.
    rows = read_hook_audit(audit_path)
    assert not any(r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK for r in rows)
    # Belief posterior unchanged.
    b = _read_belief(db, "F1")
    assert b.alpha == 1.0 and b.beta == 1.0


def test_apply_returns_zero_when_no_signal_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
                "session_id": "s1",
                "beliefs": [{"id": "F1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n = apply_sentiment_feedback("show me the bananas", "s1")

    assert n == 0
    rows = read_hook_audit(audit_path)
    assert not any(r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK for r in rows)


def test_apply_returns_zero_when_no_prior_ups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    # No audit file at all.
    n = apply_sentiment_feedback("no, wrong", "s1")
    assert n == 0


def test_apply_demotes_prior_turn_beliefs_and_writes_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk("F1", "the answer is purple"),
            _mk("F2", "the answer is orange"),
        ],
    )
    _set_db(monkeypatch, db)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
                "session_id": "s1",
                "beliefs": [{"id": "F1"}, {"id": "F2"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n = apply_sentiment_feedback("no, that's wrong", "s1")

    assert n == 2
    # Both beliefs received negative feedback -> beta increased.
    for bid in ("F1", "F2"):
        b = _read_belief(db, bid)
        assert b.beta > 1.0, f"{bid} beta should have grown from negative feedback"
        assert b.alpha == 1.0
    # Sentiment audit row landed with all required fields.
    rows = read_hook_audit(audit_path)
    sf_rows = [r for r in rows if r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK]
    assert len(sf_rows) == 1
    row = sf_rows[0]
    assert row["session_id"] == "s1"
    assert row["sentiment"] == "negative"
    assert row["pattern"] in {"wrong", "no"}
    assert row["valence"] < 0
    assert sorted(row["belief_ids"]) == ["F1", "F2"]
    assert row["n_beliefs"] == 2


def test_apply_skips_belief_ids_that_no_longer_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the answer is purple")])
    _set_db(monkeypatch, db)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # Prior turn surfaced F1 (still present) AND F_GONE (since deleted).
    audit_path.write_text(
        json.dumps(
            {
                "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
                "session_id": "s1",
                "beliefs": [{"id": "F1"}, {"id": "F_GONE"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    n = apply_sentiment_feedback("no, wrong", "s1")

    assert n == 1  # Only F1 was updatable.
    rows = read_hook_audit(audit_path)
    sf_rows = [r for r in rows if r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK]
    assert sf_rows[0]["belief_ids"] == ["F1"]


# ---------------------------------------------------------------------------
# integration: UPS hook end-to-end + two-session correction ranking (AC4)
# ---------------------------------------------------------------------------


def test_ups_hook_skips_sentiment_lane_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the answer is purple")])
    _set_db(monkeypatch, db)

    # Turn 1: prime the audit log with a UPS row.
    user_prompt_submit(
        stdin=io.StringIO(_payload("answer purple")),
        stdout=io.StringIO(),
    )
    # Turn 2: correction prompt.
    user_prompt_submit(
        stdin=io.StringIO(_payload("no, that's wrong")),
        stdout=io.StringIO(),
    )

    b = _read_belief(db, "F1")
    assert b.beta == 1.0, "lane was disabled, posterior must not move"
    rows = read_hook_audit(_audit_path_for_db(db))
    assert not any(r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK for r in rows)


def test_ups_hook_applies_sentiment_to_prior_turn_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the answer is purple")])
    _set_db(monkeypatch, db)

    # Turn 1: query that should retrieve F1; UPS writes its audit row.
    user_prompt_submit(
        stdin=io.StringIO(_payload("what is the answer in purple")),
        stdout=io.StringIO(),
    )
    pre = _read_belief(db, "F1")
    # Turn 2: user says "no, that's wrong" — sentiment lane fires before
    # this turn's retrieval and bumps F1's beta.
    user_prompt_submit(
        stdin=io.StringIO(_payload("no, that's wrong")),
        stdout=io.StringIO(),
    )
    post = _read_belief(db, "F1")

    assert post.beta > pre.beta, "negative sentiment must increase beta"
    rows = read_hook_audit(_audit_path_for_db(db))
    sf_rows = [r for r in rows if r.get("hook") == AUDIT_HOOK_SENTIMENT_FEEDBACK]
    assert len(sf_rows) == 1


def test_correction_lowers_subsequent_ranking_across_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: session A applies a correction; session B replays the same
    query; the corrected belief now ranks below an uncorrected sibling.
    """
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    # Both beliefs start at uniform prior (alpha=beta=1).
    _seed_db(
        db,
        [
            _mk("WRONG", "the recommended strategy is to apply patches directly"),
            _mk("OK", "use a separate approach for handling edge cases"),
        ],
    )
    _set_db(monkeypatch, db)

    # Session A: turn 1 surfaces only WRONG (prompt is narrower).
    user_prompt_submit(
        stdin=io.StringIO(_payload("apply patches strategy", session_id="sA")),
        stdout=io.StringIO(),
    )
    # Session A: turn 2 — user says "no, wrong". Lane bumps WRONG only,
    # because only WRONG was in sA's prior UPS retrieval set.
    user_prompt_submit(
        stdin=io.StringIO(_payload("no, that's wrong", session_id="sA")),
        stdout=io.StringIO(),
    )

    # Session B: same DB; replay a query that surfaces both beliefs.
    sout_b = io.StringIO()
    user_prompt_submit(
        stdin=io.StringIO(
            _payload("patches approach separate handling", session_id="sB")
        ),
        stdout=sout_b,
    )
    block = sout_b.getvalue()

    wrong_pos = block.find('id="WRONG"')
    ok_pos = block.find('id="OK"')
    assert wrong_pos != -1, "WRONG should still be retrieved"
    assert ok_pos != -1, "OK should still be retrieved"
    assert ok_pos < wrong_pos, (
        "the uncorrected belief must rank above the corrected one in session B"
    )

    # WRONG's posterior moved; OK's did not.
    wrong = _read_belief(db, "WRONG")
    ok = _read_belief(db, "OK")
    assert wrong.beta > 1.0
    assert ok.beta == 1.0


def test_feedback_history_row_uses_sentiment_inferred_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_sentiment(monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "purple bananas in the kitchen")])
    _set_db(monkeypatch, db)
    user_prompt_submit(
        stdin=io.StringIO(_payload("purple bananas in kitchen")), stdout=io.StringIO()
    )
    user_prompt_submit(
        stdin=io.StringIO(_payload("no, wrong")), stdout=io.StringIO()
    )

    store = MemoryStore(str(db))
    try:
        events = store.list_feedback_events(belief_id="F1")
    finally:
        store.close()
    assert any(
        e.source == SENTIMENT_INFERRED_SOURCE for e in events
    ), "feedback_history row should carry the sentiment_inferred source tag"
