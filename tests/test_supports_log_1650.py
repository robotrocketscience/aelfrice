"""#1650 part 5: the log-only reworded-restatement pass.

`supports_log.log_reworded_support` records each reworded restatement a
user sentence makes of a non-user belief from another session, with
applied=0, and applies nothing. These tests build every store from
synthetic text.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import aelfrice.supports_log as supports_log
from aelfrice.ingest import _ingest_turn
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    SUPPORT_EVIDENCE_KIND_REWORDED,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.supports_log import (
    ENV_LOG_REWORDED_SUPPORT,
    REWORDED_CANDIDATE_TOP_K,
    USER_ORIGINS,
    is_reworded_support_log_enabled,
    log_reworded_support,
)

_TARGET = (
    "Our release checklist covers the changelog. "
    "The scanner reliably detects leaked secrets in committed source code."
)
_USER = "The scanner detects leaked secrets in source code."
_SPAN = "The scanner reliably detects leaked secrets in committed source code."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))
    monkeypatch.delenv(ENV_LOG_REWORDED_SUPPORT, raising=False)
    # The process cwd holds no `.aelfrice.toml`, so only a file a test
    # writes can decide a result.
    work = tmp_path / "proc-cwd"
    work.mkdir()
    monkeypatch.chdir(work)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "sl.db"))
    yield s
    s.close()


def _target(
    s: MemoryStore, bid: str = "p1", content: str = _TARGET, *,
    session: str | None = "s1", origin: str = "agent_inferred",
) -> None:
    s.insert_belief(Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0,
        beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
        session_id=session, origin=origin,
    ))


def _typed(
    s: MemoryStore, text: str = _USER, *, session: str | None = "s2",
    role: str | None = "user",
) -> list[tuple[str, str]]:
    """Ingest one turn; return its (belief id, ingest_log id) pairs."""
    turn = _ingest_turn(
        s, text, "transcript", session_id=session,
        created_at="2026-10-02T00:00:00Z", role=role,
    )
    pairs: list[tuple[str, str]] = []
    for bid in turn.inserted:
        row = s._conn.execute(  # noqa: SLF001
            "SELECT id FROM ingest_log WHERE derived_belief_ids LIKE ?",
            (f'%"{bid}"%',),
        ).fetchone()
        pairs.append((bid, str(row[0])))
    assert pairs, "the synthetic turn inserted no belief"
    return pairs


def _rows(s: MemoryStore) -> list[tuple[object, ...]]:
    return [tuple(r) for r in s._conn.execute(  # noqa: SLF001
        "SELECT supporter_id, target_id, kind, target_sentence_index, "
        "supporter_session_id, supporter_ingest_log_id, matched_span, applied "
        "FROM belief_support_evidence ORDER BY supporter_id, target_id")]


# --- The flag --------------------------------------------------------------


def test_the_flag_is_off_by_default() -> None:
    assert is_reworded_support_log_enabled() is False


def test_the_toml_key_turns_the_flag_on() -> None:
    Path(".aelfrice.toml").write_text("[ingest]\nlog_reworded_support = true\n")
    assert is_reworded_support_log_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_a_falsy_env_value_wins_over_a_toml_yes(value: str) -> None:
    Path(".aelfrice.toml").write_text("[ingest]\nlog_reworded_support = true\n")
    assert is_reworded_support_log_enabled(
        env={ENV_LOG_REWORDED_SUPPORT: value}) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " Yes "])
def test_a_truthy_env_value_wins_over_a_toml_no(value: str) -> None:
    Path(".aelfrice.toml").write_text("[ingest]\nlog_reworded_support = false\n")
    assert is_reworded_support_log_enabled(
        env={ENV_LOG_REWORDED_SUPPORT: value}) is True


def test_the_ambient_env_is_read_when_no_mapping_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    assert is_reworded_support_log_enabled() is True


def test_an_unrecognized_env_value_defers_to_the_toml_key() -> None:
    Path(".aelfrice.toml").write_text("[ingest]\nlog_reworded_support = true\n")
    assert is_reworded_support_log_enabled(
        env={ENV_LOG_REWORDED_SUPPORT: "maybe"}) is True
    assert is_reworded_support_log_enabled(
        env={ENV_LOG_REWORDED_SUPPORT: "maybe"},
        start=Path("/nonexistent-1650")) is False


@pytest.mark.parametrize("toml", [
    "[ingest]\nlog_reworded_support = \"yes\"\n",
    "[ingest]\nlog_reworded_support = 1\n",
    "ingest = 3\n",
    "[other]\nlog_reworded_support = true\n",
])
def test_an_unusable_toml_value_keeps_the_default(toml: str) -> None:
    Path(".aelfrice.toml").write_text(toml)
    assert is_reworded_support_log_enabled() is False


def test_a_malformed_toml_keeps_the_default_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    Path(".aelfrice.toml").write_text("[ingest\n")
    assert is_reworded_support_log_enabled() is False
    assert "cannot read log_reworded_support" in capsys.readouterr().err


# --- The pass -------------------------------------------------------------


def test_a_reworded_restatement_is_logged_unapplied(store: MemoryStore) -> None:
    _target(store)
    pairs = _typed(store)
    assert log_reworded_support(store, pairs) == 1
    (bid, log_id), = pairs
    assert _rows(store) == [(
        bid, "p1", SUPPORT_EVIDENCE_KIND_REWORDED, 1, "s2", log_id, _SPAN, 0,
    )]


def test_the_pass_writes_no_edge_and_moves_no_posterior(
    store: MemoryStore,
) -> None:
    _target(store)
    pairs = _typed(store)
    edges = store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM edges").fetchone()[0]
    before = store.get_belief("p1")
    assert log_reworded_support(store, pairs) == 1
    after = store.get_belief("p1")
    assert before is not None and after is not None
    assert (after.alpha, after.beta) == (before.alpha, before.beta)
    assert store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM edges").fetchone()[0] == edges
    assert store.count_feedback_events("p1") == 0
    assert store.count_support_evidence(applied=True) == 0


def test_a_repeat_pass_adds_nothing(store: MemoryStore) -> None:
    _target(store)
    pairs = _typed(store)
    assert log_reworded_support(store, pairs) == 1
    assert log_reworded_support(store, pairs) == 0
    assert len(_rows(store)) == 1


def test_a_target_from_the_supporters_session_is_skipped(
    store: MemoryStore,
) -> None:
    _target(store, session="s2")
    assert log_reworded_support(store, _typed(store, session="s2")) == 0
    assert _rows(store) == []


def test_a_target_with_no_session_is_kept(store: MemoryStore) -> None:
    _target(store, session=None)
    assert log_reworded_support(store, _typed(store)) == 1


def test_a_supporter_with_no_session_is_skipped(store: MemoryStore) -> None:
    # A target with no session passes the session rule, so only the
    # supporter's missing session keeps this pair out.
    _target(store, session=None)
    assert log_reworded_support(store, _typed(store, session=None)) == 0


_USER_ORIGIN_NAMES = (
    "user_corrected", "user_stated", "user_transcript", "user_validated",
)


def test_user_origins_are_the_four_user_tiers() -> None:
    assert USER_ORIGINS == frozenset(_USER_ORIGIN_NAMES)


@pytest.mark.parametrize("origin", _USER_ORIGIN_NAMES)
def test_a_user_belief_is_never_a_target(
    store: MemoryStore, origin: str,
) -> None:
    _target(store, origin=origin)
    assert log_reworded_support(store, _typed(store)) == 0


def test_only_non_user_targets_are_logged(store: MemoryStore) -> None:
    _target(store, "p1", origin="agent_inferred")
    _target(store, "p2", _TARGET + " It runs nightly.", origin="user_transcript")
    log_reworded_support(store, _typed(store))
    assert [r[1] for r in _rows(store)] == ["p1"]


def test_the_supporter_never_supports_itself(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Count the supporter as a non-user belief, so only the self rule
    # keeps it out of its own candidates.
    monkeypatch.setattr(supports_log, "USER_ORIGINS", frozenset[str]())
    pairs = _typed(store)
    # Move the supporter belief out of its session, so only the self rule
    # keeps it out of its own candidates.
    store._conn.execute(  # noqa: SLF001
        "UPDATE beliefs SET session_id = 's9' WHERE id = ?", (pairs[0][0],))
    store._conn.commit()  # noqa: SLF001
    assert log_reworded_support(store, pairs) == 0
    assert _rows(store) == []
    # Never a candidate, so the store's own refusal never fires.
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("role", ["assistant", "system", None])
def test_a_supporter_not_typed_by_the_user_is_skipped(
    store: MemoryStore, role: str | None,
) -> None:
    _target(store)
    assert log_reworded_support(store, _typed(store, role=role)) == 0
    assert _rows(store) == []


def test_a_log_row_for_another_belief_is_not_proof(store: MemoryStore) -> None:
    _target(store)
    (bid, _), = _typed(store)
    (_, other_log), = _typed(store, "The parser handles each request quickly.")
    assert log_reworded_support(store, [(bid, other_log)]) == 0


def test_a_log_row_from_another_source_kind_is_not_proof(
    store: MemoryStore,
) -> None:
    _target(store)
    pairs = _typed(store)
    store._conn.execute(  # noqa: SLF001
        "UPDATE ingest_log SET source_kind = 'filesystem' WHERE id = ?",
        (pairs[0][1],))
    store._conn.commit()  # noqa: SLF001
    assert log_reworded_support(store, pairs) == 0


def test_an_unknown_log_row_is_not_proof(store: MemoryStore) -> None:
    _target(store)
    (bid, _), = _typed(store)
    assert log_reworded_support(store, [(bid, "no-such-log")]) == 0


def test_candidates_per_sentence_are_bounded_by_k(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert REWORDED_CANDIDATE_TOP_K == 20
    for i in range(5):
        _target(store, f"p{i}", _TARGET + f" Build {i} passed.")
    monkeypatch.setattr(supports_log, "REWORDED_CANDIDATE_TOP_K", 2)
    calls: list[str] = []
    real = supports_log.match_reworded

    def spy(user: str, target: str) -> object:
        calls.append(target)
        return real(user, target)

    monkeypatch.setattr(supports_log, "match_reworded", spy)
    assert log_reworded_support(store, _typed(store)) == 2
    assert len(calls) == 2


def test_a_failure_inside_the_pass_is_reported_and_swallowed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _target(store)
    _target(store, "p2", _TARGET + " It runs nightly.")
    pairs = _typed(store)
    real = supports_log.match_reworded
    seen: list[str] = []

    def flaky(user: str, target: str) -> object:
        seen.append(target)
        if len(seen) == 2:
            raise RuntimeError("synthetic matcher failure")
        return real(user, target)

    monkeypatch.setattr(supports_log, "match_reworded", flaky)
    assert log_reworded_support(store, pairs) == 0
    assert "RuntimeError: synthetic matcher failure" in capsys.readouterr().err
    # The first match was rolled back with the pass, and the turn stays.
    assert _rows(store) == []
    assert store.get_belief(pairs[0][0]) is not None
    assert store.transaction_open is False


def test_a_read_only_store_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "ro.db"
    s = MemoryStore(str(db))
    _target(s)
    pairs = _typed(s)
    s.close()
    ro = MemoryStore(str(db), read_only=True)
    try:
        assert log_reworded_support(ro, pairs) == 0
    finally:
        ro.close()
    # Skipped before any write is tried, so nothing failed to report.
    assert capsys.readouterr().err == ""


def test_a_store_without_the_table_is_skipped(
    store: MemoryStore, capsys: pytest.CaptureFixture[str],
) -> None:
    _target(store)
    pairs = _typed(store)
    store._conn.execute("DROP TABLE belief_support_evidence")  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    assert log_reworded_support(store, pairs) == 0
    assert "no such table" in capsys.readouterr().err
    assert store.count_support_evidence(applied=False) == 0
    with pytest.raises(sqlite3.OperationalError):
        store._conn.execute(  # noqa: SLF001
            "SELECT 1 FROM belief_support_evidence")
