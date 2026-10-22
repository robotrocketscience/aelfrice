"""#1650 part 5: the log-only reworded-restatement pass.

`supports_log.log_reworded_support` records each reworded restatement a
user sentence makes of a non-user belief from another session, with
applied=0, and applies nothing. These tests build every store from
synthetic text.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import aelfrice.reworded_restatement as reworded_restatement
from aelfrice import supports_log
from aelfrice.ingest import _ingest_turn, ingest_jsonl
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
    created_at: str = "2026-10-01T00:00:00Z",
) -> None:
    s.insert_belief(Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0,
        beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at=created_at, last_retrieved_at=None,
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


def test_the_iterator_yields_rejected_candidates_and_the_pass_logs_only_matches(
    store: MemoryStore,
) -> None:
    # p2 holds every candidate term but negates the claim, so the
    # matcher rejects it.
    _target(store, "p1")
    _target(store, "p2", "The scanner never detects leaked secrets in source code.")
    pairs = _typed(store)
    found = {c.target_id: c for c in supports_log.iter_reworded_candidates(store, pairs)}
    assert sorted(found) == ["p1", "p2"]
    assert found["p2"].match is None
    match = found["p1"].match
    assert match is not None and match.target_sentence == _SPAN
    assert (found["p1"].supporter_text, found["p1"].session_id) == (_USER, "s2")
    assert log_reworded_support(store, pairs) == 1
    assert [r[1] for r in _rows(store)] == ["p1"]


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


def test_a_target_created_after_the_sentence_was_typed_is_skipped(
    store: MemoryStore,
) -> None:
    # The user typed the sentence on 2026-10-02; this target came later.
    _target(store, created_at="2026-10-20T00:00:00Z")
    assert log_reworded_support(store, _typed(store)) == 0
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
    real = reworded_restatement.match_reworded

    def spy(user: str, target: str) -> object:
        calls.append(target)
        return real(user, target)

    monkeypatch.setattr(reworded_restatement, "match_reworded", spy)
    assert log_reworded_support(store, _typed(store)) == 2
    assert len(calls) == 2


@pytest.mark.parametrize("target", [
    "The ｓｃａｎｎｅｒ reliably detects leaked secrets in committed source code.",
    "The scanner reliably detects leaked se\u00adcrets in committed source code.",
    "The scanner reliably detects leaked se\u200bcrets in committed source code.",
], ids=["fullwidth", "soft-hyphen", "zero-width-space"])
def test_a_target_the_search_index_splits_or_keeps_unfolded_is_not_found(
    store: MemoryStore, target: str,
) -> None:
    # The documented limit (supports_log, candidate_terms, CONFIG.md):
    # the matcher reads the target normalized, but the search index that
    # picks candidates holds it as stored. If the index learns to fold
    # these forms, this test fails and the docs must change with it.
    assert reworded_restatement.match_reworded(_USER, target) is not None
    _target(store, content=target)
    assert log_reworded_support(store, _typed(store)) == 0


def test_a_failure_inside_the_pass_is_reported_and_swallowed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _target(store)
    _target(store, "p2", _TARGET + " It runs nightly.")
    pairs = _typed(store)
    real = reworded_restatement.match_reworded
    seen: list[str] = []

    def flaky(user: str, target: str) -> object:
        seen.append(target)
        if len(seen) == 2:
            raise RuntimeError("synthetic matcher failure")
        return real(user, target)

    monkeypatch.setattr(reworded_restatement, "match_reworded", flaky)
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


@pytest.mark.timeout(130)
def test_reading_the_flag_does_not_load_the_matcher() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import aelfrice.supports_log as s; "
        "s.is_reworded_support_log_enabled(env={}); "
        "print('aelfrice.reworded_restatement' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True, timeout=120,
    )
    assert out.stdout.strip() == "False"


# --- Ingest wiring ---------------------------------------------------------


def _jsonl(path: Path, *records: tuple[str, str, str]) -> Path:
    """A transcript-logger turns.jsonl of (role, session, text) records."""
    lines = [
        json.dumps({"role": role, "text": text, "session_id": session,
                    "ts": f"2026-10-02T00:00:{i:02d}Z"})
        for i, (role, session, text) in enumerate(records)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _log_id_for(s: MemoryStore, bid: str) -> str:
    row = s._conn.execute(  # noqa: SLF001
        "SELECT id FROM ingest_log WHERE derived_belief_ids LIKE ?",
        (f'%"{bid}"%',)).fetchone()
    return str(row[0])


def _spy_matcher(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = reworded_restatement.match_reworded

    def spy(user: str, target: str) -> object:
        calls.append(target)
        return real(user, target)

    monkeypatch.setattr(reworded_restatement, "match_reworded", spy)
    return calls


_OTHER = "The parser handles each request within one second."


def test_ingest_with_the_flag_off_logs_nothing_and_runs_no_matcher(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target(store)
    calls = _spy_matcher(monkeypatch)
    passes: list[object] = []

    def no_pass(*args: object) -> int:
        passes.append(args)
        return 0

    monkeypatch.setattr(supports_log, "log_reworded_support", no_pass)
    reads: list[bool] = []
    real_flag = supports_log.is_reworded_support_log_enabled

    def flag(**kwargs: object) -> bool:
        value = real_flag()
        reads.append(value)
        return value

    monkeypatch.setattr(supports_log, "is_reworded_support_log_enabled", flag)
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", _USER), ("user", "s2", _OTHER)))
    assert result.beliefs_inserted == 2
    assert passes == [] and calls == [] and _rows(store) == []
    assert reads == [False]


def test_ingest_with_the_flag_on_logs_a_later_session_restatement(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store)
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", _USER)))
    assert result.beliefs_inserted == 1
    (row,) = _rows(store)
    bid = str(row[0])
    assert row == (bid, "p1", SUPPORT_EVIDENCE_KIND_REWORDED, 1, "s2",
                   _log_id_for(store, bid), _SPAN, 0)


def test_ingest_logs_the_restating_sentence_of_a_longer_turn(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store)
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", f"{_OTHER} {_USER}")))
    assert result.beliefs_inserted == 2
    (row,) = _rows(store)
    bid = str(row[0])
    supporter = store.get_belief(bid)
    assert supporter is not None and supporter.content == _USER
    assert row[5] == _log_id_for(store, bid)


def test_ingest_reads_the_toml_flag(store: MemoryStore, tmp_path: Path) -> None:
    Path(".aelfrice.toml").write_text("[ingest]\nlog_reworded_support = true\n")
    _target(store)
    ingest_jsonl(store, _jsonl(tmp_path / "t.jsonl", ("user", "s2", _USER)))
    assert len(_rows(store)) == 1


def test_ingest_skips_a_target_from_the_same_session(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store, session="s2")
    ingest_jsonl(store, _jsonl(tmp_path / "t.jsonl", ("user", "s2", _USER)))
    assert _rows(store) == []


def test_a_backfill_of_an_older_session_logs_nothing_for_a_newer_target(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The transcript turn is stamped 2026-10-02; the target was created
    # on 2026-10-20 and the transcript is ingested after that.
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store, session="s9", created_at="2026-10-20T00:00:00Z")
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", _USER)))
    assert result.beliefs_inserted == 1
    assert _rows(store) == []


def test_a_corroborating_turn_offers_no_supporter(store: MemoryStore) -> None:
    first = _ingest_turn(store, _USER, "transcript", session_id="s2",
                         created_at="2026-10-02T00:00:00Z", role="user")
    again = _ingest_turn(store, _USER, "transcript", session_id="s3",
                         created_at="2026-10-03T00:00:00Z", role="user")
    assert len(first.sources) == 1
    assert again.inserted == [] and again.sources == ()
    assert again.resolved == [first.sources[0][0]]


def test_ingest_logs_nothing_for_a_sentence_that_corroborates(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Typed first with the log off, then again in a later session with
    # it on. The second turn only corroborates the first belief, so it
    # inserts nothing and no supporter is offered.
    _target(store)
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "0")
    first = ingest_jsonl(store, _jsonl(
        tmp_path / "a.jsonl", ("user", "s2", _USER)))
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    again = ingest_jsonl(store, _jsonl(
        tmp_path / "b.jsonl", ("user", "s3", _USER)))
    assert (first.beliefs_inserted, again.beliefs_inserted) == (1, 0)
    assert _rows(store) == []


def test_ingest_never_logs_an_assistant_or_system_turn_as_a_supporter(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store)
    calls = _spy_matcher(monkeypatch)
    passes: list[object] = []
    real_pass = supports_log.log_reworded_support

    def spy_pass(s: MemoryStore, pairs: tuple[tuple[str, str], ...]) -> int:
        passes.append(pairs)
        return real_pass(s, pairs)

    monkeypatch.setattr(supports_log, "log_reworded_support", spy_pass)
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("assistant", "s2", _USER), ("system", "s2", _USER)))
    # The system line is ingested as a belief, but no pass runs for it.
    assert result.beliefs_inserted == 1
    assert passes == [] and calls == [] and _rows(store) == []


def test_ingest_never_logs_support_for_the_users_own_belief(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s1", _SPAN), ("user", "s2", _USER)))
    assert _rows(store) == []


def test_a_failing_pass_leaves_the_ingest_committed(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store)

    def boom(user: str, target: str) -> object:
        raise RuntimeError("synthetic matcher failure")

    monkeypatch.setattr(reworded_restatement, "match_reworded", boom)
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", _USER), ("user", "s2", _OTHER)))
    assert result.beliefs_inserted == 2 and result.turns_ingested == 2
    assert "synthetic matcher failure" in capsys.readouterr().err
    assert store.transaction_open is False
    other = sqlite3.connect(str(tmp_path / "sl.db"))
    try:
        assert other.execute(
            "SELECT COUNT(*) FROM beliefs WHERE session_id = 's2'"
        ).fetchone()[0] == 2
    finally:
        other.close()
    assert _rows(store) == []


def test_ingest_on_a_store_without_the_table_still_commits(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    _target(store)
    store._conn.execute("DROP TABLE belief_support_evidence")  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    result = ingest_jsonl(store, _jsonl(
        tmp_path / "t.jsonl", ("user", "s2", _USER)))
    assert result.beliefs_inserted == 1
    assert store.count_support_evidence(applied=False) == 0
