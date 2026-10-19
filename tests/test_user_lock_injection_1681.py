"""User-scope locks in the L0 lock set (#1681, PR2 of 3).

A lock written with `aelf lock --user` joins the repository's locks in
every injection path. These tests pin that it is injected, that the user
store is opened read-only and never created, that a missing store changes
nothing, that an unreadable one warns once, that a near-duplicate of a
repository lock loses to the repository copy, that the order is fixed,
and that user locks spend the same budget as repository locks.

Every path is a temp path. The conftest pins `AELFRICE_DB` and
`AELFRICE_USER_DB` at a sandbox; each test repoints both.
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import user_locks as ul
from aelfrice.context_rebuilder import RecentTurn, rebuild_v14
from aelfrice.hook import (
    _record_injection_events,  # pyright: ignore[reportPrivateUsage]
    _retrieve_baseline_with_block,  # pyright: ignore[reportPrivateUsage]
    session_start,
    user_prompt_submit,
)
from aelfrice.hook_search import record_retrieval
from aelfrice.hrr_index import HRRStructIndexCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve, retrieve_v2
from aelfrice.store import MemoryStore

REPO_RULE = "This repository builds with make, not cmake."
USER_RULE = "Route all questions and decisions through the question tool."
# Near-duplicate of REPO_RULE: same tokens, one character apart.
USER_NEAR_DUP = "This repository builds with make, not cmake!"


def _lock(
    bid: str,
    content: str,
    *,
    locked_at: str = "2026-10-01T00:00:00+00:00",
    expires: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at=locked_at,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
        lock_expires_at=expires,
    )


def _plain(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _seed(path: Path, beliefs: list[Belief]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(str(path))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


@pytest.fixture()
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(repo_db, user_db), both temp paths, neither created yet."""
    repo = tmp_path / "repo" / "memory.db"
    user = tmp_path / "home" / ".aelfrice" / "user" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    return repo, user


@pytest.fixture(autouse=True)
def _reset_warned() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]


def _retrieve_ids(repo: Path, query: str = "", **kw: object) -> list[str]:
    s = MemoryStore(str(repo))
    try:
        return [b.id for b in retrieve(s, query, **kw)]  # type: ignore[arg-type]
    finally:
        s.close()


def _lock_level_on_disk(path: Path, bid: str) -> str:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT lock_level FROM beliefs WHERE id = ?", (bid,)
        ).fetchone()
    finally:
        conn.close()
    return str(row[0])


# --- injection --------------------------------------------------------------


@pytest.mark.timeout(60)
def test_retrieve_injects_a_user_lock_after_the_repo_locks(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    assert _retrieve_ids(repo) == ["r1", "u1"]


@pytest.mark.timeout(60)
def test_session_start_baseline_carries_the_user_lock(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    out, err = io.StringIO(), io.StringIO()
    assert session_start(
        stdin=io.StringIO(json.dumps({"session_id": "ss1681"})),
        stdout=out, stderr=err,
    ) == 0
    text = out.getvalue()
    assert f'<belief id="u1" lock="user">{USER_RULE}</belief>' in text
    assert text.index('id="r1"') < text.index('id="u1"')


@pytest.mark.timeout(60)
def test_prompt_hook_injects_the_user_lock_and_writes_no_traceback(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE), _plain("p1", "questions go to a tool")])
    _seed(user, [_lock("u1", USER_RULE)])
    out, err = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "prompt": "which tool do questions go through",
        "session_id": "ups1681", "cwd": str(tmp_path),
    })
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=out, stderr=err,
    ) == 0
    text = out.getvalue()
    # The first prompt's `<locked>` section is built apart from retrieval,
    # so check the user lock is inside it, not only among the hits.
    locked_section = text[text.index("<locked>"):text.index("</locked>")]
    assert f'<belief id="u1" lock="user">{USER_RULE}</belief>' in locked_section
    assert locked_section.index('id="r1"') < locked_section.index('id="u1"')
    # The exposure and injection-event writes skip the user lock; before
    # they did, each one printed a traceback for an id this store lacks.
    assert "Traceback" not in err.getvalue()


@pytest.mark.timeout(60)
def test_search_tool_hook_tags_the_user_lock_l0(
    stores: tuple[Path, Path],
) -> None:
    from aelfrice import hook_search_tool as hst

    repo, user = stores
    _seed(repo, [_plain("p1", "the question tool routes decisions")])
    _seed(user, [_lock("u1", USER_RULE)])
    out = io.StringIO()
    payload = {
        "hook_event_name": "PreToolUse", "tool_name": "Grep",
        "tool_input": {"pattern": "question tool"}, "cwd": "/tmp",
        "session_id": "st1681",
    }
    assert hst.main(
        stdin=io.StringIO(json.dumps(payload)), stdout=out,
        stderr=io.StringIO(),
    ) == 0
    assert "[L0] u1" in out.getvalue()


@pytest.mark.timeout(60)
def test_rebuild_puts_the_user_lock_in_l0(stores: tuple[Path, Path]) -> None:
    """L0 is never trimmed; a budget too small for the lock proves the tier."""
    repo, user = stores
    big_rule = "Always explain the widget plan before writing code. " * 10
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", big_rule)])
    s = MemoryStore(str(repo))
    try:
        block = rebuild_v14(
            [RecentTurn(role="user", text="widget plan code")], s,
            token_budget=20,
        )
    finally:
        s.close()
    # The rebuild block shortens long content, so match its start.
    assert "Always explain the widget plan" in block
    assert block.index(REPO_RULE) < block.index("Always explain the widget")


@pytest.mark.timeout(60)
def test_structural_lane_pins_the_user_lock_too(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    _seed(repo, [_plain("b1", "one"), _plain("b2", "two")])
    s = MemoryStore(str(repo))
    try:
        s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_CONTRADICTS, weight=1.0))
        result = retrieve_v2(
            s, "CONTRADICTS:b2",
            use_hrr_structural=True,
            hrr_struct_index_cache=HRRStructIndexCache(store=s, dim=512, seed=42),
            user_locks=[_lock("u1", USER_RULE)],
        )
    finally:
        s.close()
    assert result.locked_ids == ["u1"]
    assert result.beliefs[0].id == "u1"


# --- read-only, never create ------------------------------------------------


@pytest.mark.timeout(60)
def test_missing_user_store_is_never_created(stores: tuple[Path, Path]) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    assert _retrieve_ids(repo) == ["r1"]
    _retrieve_baseline_with_block()
    assert not user.exists()
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_user_store_is_read_without_the_expiry_sweep_writing(
    stores: tuple[Path, Path],
) -> None:
    """A writable open would sweep the expired lock to 'none' on disk."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [
        _lock("u1", USER_RULE),
        _lock("u2", "Expired rule about tabs.", expires="2026-01-01T00:00:00+00:00"),
    ])
    assert _lock_level_on_disk(user, "u2") == LOCK_USER
    assert _retrieve_ids(repo) == ["r1", "u1"]
    assert _lock_level_on_disk(user, "u2") == LOCK_USER


@pytest.mark.timeout(60)
def test_an_expired_user_lock_is_not_injected(stores: tuple[Path, Path]) -> None:
    _, user = stores
    _seed(user, [
        _lock("u1", USER_RULE, expires="2026-10-02T00:00:00+00:00"),
    ])
    assert [b.id for b in ul.read_user_locks(now="2026-10-01T23:59:59+00:00")] == ["u1"]
    assert ul.read_user_locks(now="2026-10-02T00:00:00+00:00") == []


# --- missing store: unchanged ------------------------------------------------


@pytest.mark.timeout(60)
def test_missing_user_store_leaves_the_baseline_block_unchanged(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE), _lock("r2", "Second repo rule here.")])
    hits, block = _retrieve_baseline_with_block()
    s = MemoryStore(str(repo))
    try:
        repo_only = retrieve(s, "", manifest_reference_locks=True, user_locks=())
    finally:
        s.close()
    assert [b.id for b in hits] == [b.id for b in repo_only]
    assert "r1" in block and "r2" in block
    assert not user.exists()


@pytest.mark.timeout(60)
def test_missing_user_store_prints_nothing(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    assert ul.read_user_locks() == []
    assert capsys.readouterr().err == ""


# --- unreadable store --------------------------------------------------------


@pytest.mark.timeout(60)
def test_unreadable_user_store_warns_once_and_injects_repo_locks(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    user.parent.mkdir(parents=True)
    user.write_bytes(b"not a database " * 20)
    assert _retrieve_ids(repo) == ["r1"]
    assert _retrieve_ids(repo, "make")[0] == "r1"
    err = capsys.readouterr().err.splitlines()
    warnings = [line for line in err if "user lock store" in line]
    assert len(warnings) == 1
    assert str(user) in warnings[0]
    assert "injecting repository locks only" in warnings[0]


@pytest.mark.timeout(60)
def test_user_store_that_is_the_repo_store_is_not_read_twice(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    assert ul.read_user_locks(str(repo)) == []
    assert [b.id for b in ul.read_user_locks()] == ["r1"]


# --- dedupe ------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_near_duplicate_user_lock_loses_to_the_repo_copy(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_NEAR_DUP), _lock("u2", USER_RULE)])
    assert _retrieve_ids(repo) == ["r1", "u2"]


@pytest.mark.timeout(60)
def test_same_statement_in_both_scopes_is_injected_once(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("same", REPO_RULE)])
    _seed(user, [_lock("same", REPO_RULE)])
    s = MemoryStore(str(repo))
    try:
        hits = retrieve(s, "")
    finally:
        s.close()
    assert [b.id for b in hits] == ["same"]
    assert not ul.is_user_scope_lock(hits[0])


def test_merge_keeps_a_user_lock_that_is_only_similar() -> None:
    repo = [_lock("r1", REPO_RULE)]
    near = _lock("u1", USER_NEAR_DUP)
    far = _lock("u2", "This repository builds with bazel and never with cmake.")
    assert [b.id for b in ul.merge_lock_sets(repo, [near, far])] == ["r1", "u2"]


# --- order -------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_order_is_repo_then_user_each_newest_first_then_id(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [
        _lock("r_old", "Old repo rule about linting.", locked_at="2026-09-01T00:00:00+00:00"),
        _lock("r_new", "New repo rule about formatting.", locked_at="2026-10-01T00:00:00+00:00"),
    ])
    # Inserted out of order; two share a timestamp so the id breaks the tie.
    _seed(user, [
        _lock("u_b", "User rule bravo on commits.", locked_at="2026-10-05T00:00:00+00:00"),
        _lock("u_old", "User rule on reviews.", locked_at="2026-08-01T00:00:00+00:00"),
        _lock("u_a", "User rule alpha on branches.", locked_at="2026-10-05T00:00:00+00:00"),
    ])
    expected = ["r_new", "r_old", "u_a", "u_b", "u_old"]
    assert _retrieve_ids(repo) == expected
    assert _retrieve_ids(repo) == expected


# --- budget ------------------------------------------------------------------


def _fill_repo(repo: Path) -> None:
    beliefs: list[Belief] = [_lock("r1", REPO_RULE)]
    for i in range(40):
        beliefs.append(_plain(
            f"p{i:02d}",
            f"widget calibration note {i}: the widget calibration step "
            f"runs before the widget assembly step on line {i}.",
        ))
    _seed(repo, beliefs)


@pytest.mark.timeout(60)
def test_user_locks_spend_the_shared_budget(stores: tuple[Path, Path]) -> None:
    repo, user = stores
    _fill_repo(repo)
    budget = 600
    without = _retrieve_ids(repo, "widget calibration", token_budget=budget)
    _seed(user, [_lock("u1", "Always explain the widget plan first. " * 12)])
    with_user = _retrieve_ids(repo, "widget calibration", token_budget=budget)
    assert "u1" in with_user
    n_without = len([i for i in without if i.startswith("p")])
    n_with = len([i for i in with_user if i.startswith("p")])
    assert 0 < n_with < n_without


@pytest.mark.timeout(60)
def test_user_locks_share_the_relevance_floor(stores: tuple[Path, Path]) -> None:
    """A user lock larger than the budget still leaves the floor for hits."""
    repo, user = stores
    _fill_repo(repo)
    _seed(user, [_lock("u1", "Always explain the widget plan first. " * 200)])
    ids = _retrieve_ids(repo, "widget calibration", token_budget=600)
    assert ids[:2] == ["r1", "u1"]
    assert any(i.startswith("p") for i in ids)


# --- repository-store writes skip user locks ---------------------------------


@pytest.mark.timeout(60)
def test_exposure_rows_skip_user_locks(stores: tuple[Path, Path]) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    u = _lock("u1", USER_RULE)
    setattr(u, ul.USER_SCOPE_ATTR, ul.USER_SCOPE_VALUE)
    err = io.StringIO()
    s = MemoryStore(str(repo))
    try:
        r1 = s.get_belief("r1")
        assert r1 is not None
        assert record_retrieval(s, [r1, u], stderr=err) == 1
    finally:
        s.close()
    assert err.getvalue() == ""


@pytest.mark.timeout(60)
def test_injection_events_keep_the_repo_rows_beside_a_user_lock(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    u = _lock("u1", USER_RULE)
    setattr(u, ul.USER_SCOPE_ATTR, ul.USER_SCOPE_VALUE)
    err = io.StringIO()
    s = MemoryStore(str(repo))
    try:
        r1 = s.get_belief("r1")
        assert r1 is not None
        _record_injection_events(
            session_id="ie1681", turn_id="t1", hits=[r1, u], source="ups",
            active_consumers=[], stderr=err, store=s,
        )
        rows = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT belief_id FROM injection_events"
        ).fetchall()
    finally:
        s.close()
    assert [r[0] for r in rows] == ["r1"]


@pytest.mark.timeout(60)
def test_deferred_feedback_queue_skips_user_locks(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aelfrice.deferred_feedback import ENV_ENQUEUE

    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    monkeypatch.setenv(ENV_ENQUEUE, "1")
    s = MemoryStore(str(repo))
    try:
        assert [b.id for b in retrieve(s, "")] == ["r1", "u1"]
        rows = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT belief_id FROM deferred_feedback_queue"
        ).fetchall()
    finally:
        s.close()
    # The foreign key would fail the whole batch, losing r1's row too.
    assert [r[0] for r in rows] == ["r1"]
    assert "enqueue failed" not in capsys.readouterr().err


# --- corpus measurements read repository locks only --------------------------


@pytest.mark.timeout(120)
def test_benchmark_scores_ignore_the_user_store(stores: tuple[Path, Path]) -> None:
    from aelfrice.benchmark import run_benchmark, seed_corpus

    _, user = stores

    def _score() -> tuple[float, float, float]:
        s = MemoryStore(":memory:")
        try:
            seed_corpus(s)
            r = run_benchmark(s, aelfrice_version="test")
        finally:
            s.close()
        return (r.hit_at_1, r.hit_at_3, r.mrr)

    before = _score()
    _seed(user, [
        _lock(f"u{i}", f"User rule number {i} about store retrieval and locks.")
        for i in range(5)
    ])
    assert _score() == before


@pytest.mark.timeout(120)
def test_wonder_gap_analysis_ignores_the_user_store(
    stores: tuple[Path, Path],
) -> None:
    from aelfrice.wonder.dispatch import analyze_gaps

    repo, user = stores
    _seed(repo, [_plain("p1", "the question tool routes decisions")])
    _seed(user, [_lock("u1", USER_RULE)])
    s = MemoryStore(str(repo))
    try:
        gaps = analyze_gaps(s, "question tool decisions")
    finally:
        s.close()
    assert [b.id for b in gaps.known_beliefs] == ["p1"]
