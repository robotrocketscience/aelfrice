"""#1513 — the SessionStart BM25 sidecar warm.

The measured cost is a session-first tail: bucketed by position within the
session, `benchmarks/sidecar_rebuild_rate.py` reported 6/26 = 23.1%
full rebuilds on session-FIRST fires against 1/97 = 1.0% on every later one
(re-derived 2026-08-26 over 19 logs / 123 scored fires; the issue's original
n=39 sample read 5/13 = 38.5%). The fix warms the sidecar from a detached
child spawned at `SessionStart`.

Two things have to be true and both are pinned here:

* the warm actually converts the next fire from `full_rebuild` to `fresh` —
  otherwise the rate cannot move, whatever the plumbing looks like;
* it is silent-safe (AC4) — a warm that cannot start or that raises leaves
  `SessionStart` byte-identical to what it emits today.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(120)


def _mk(bid: str, content: str, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0 if lock == LOCK_USER else 1.0,
        beta=0.5 if lock == LOCK_USER else 1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-08-26T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-08-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L1", "the sidecar warm runs off the hook path"))
        store.insert_belief(_mk("L2", "bm25 indexes belief content and anchors"))
        store.insert_belief(
            _mk("L3", "locked baseline belief", LOCK_USER)
        )
    finally:
        store.close()


def _sidecar(db: Path) -> Path:
    return Path(str(db) + ".bm25f")


def _outcome_of_a_retrieval_fire(db: Path) -> str | None:
    """Run the L1 lane exactly as a fresh hook process would, and report the
    sidecar outcome it paid."""
    from aelfrice.retrieval import bm25f_cache_for_lane
    from aelfrice.sidecar_outcome import (
        last_sidecar_outcome,
        reset_sidecar_outcome,
    )

    reset_sidecar_outcome()
    store = MemoryStore(str(db))
    try:
        cache = bm25f_cache_for_lane(store, now_ts=1_756_000_000)
        cache.get()
    finally:
        store.close()
    return last_sidecar_outcome()


# ---- the mechanism: a warm turns the next fire's rebuild into a load ----


def test_a_cold_store_costs_the_next_fire_a_full_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline this fix exists to move. Without it the assertion below
    proves nothing: it would pass on a store that was never cold."""
    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    assert not _sidecar(db).exists()

    assert _outcome_of_a_retrieval_fire(db) == "full_rebuild"


def test_the_warm_makes_the_next_fire_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1's mechanism. `full_rebuild` on the cold store above becomes
    `fresh` once the warm has run — which is the only way a session-first
    rate can fall, since the warm runs before the first prompt exists."""
    from aelfrice.sidecar_warm import warm_sidecar

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    assert warm_sidecar() == "full_rebuild"
    assert _sidecar(db).exists()

    assert _outcome_of_a_retrieval_fire(db) == "fresh"


def test_the_warm_also_covers_the_stale_sidecar_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production shape, which the missing-sidecar case is not.

    A session-first fire usually finds a sidecar that *exists* and is stale:
    the previous session's Stop hook ingested beliefs, so the generation
    stamp moved. Whether the warm then pays an `incremental` or a
    `full_rebuild` is not the contract — the contract is that the fire after
    it is `fresh`. Asserted here without pinning which of the two the warm
    paid, because #1199's incremental path decides that and is out of scope.
    """
    from aelfrice.sidecar_warm import warm_sidecar

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    assert warm_sidecar() == "full_rebuild"

    # A sibling write between sessions: the stamp moves, the blob stays.
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L4", "a belief written between sessions"))
    finally:
        store.close()
    assert _sidecar(db).exists()
    assert _outcome_of_a_retrieval_fire(db) != "fresh", (
        "the store mutation did not stale the sidecar, so this test is "
        "measuring the same thing as the missing-sidecar case"
    )

    # Re-stale it the same way, then warm rather than fire.
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L5", "and another one between sessions"))
    finally:
        store.close()

    assert warm_sidecar() in {"incremental", "full_rebuild"}
    assert _outcome_of_a_retrieval_fire(db) == "fresh"


# ---- the shared-parameter invariant ------------------------------------


def test_the_lane_helper_passes_exactly_the_four_resolved_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bm25f_cache_for_lane` must not resolve anything of its own.

    The four values decide what documents the index describes. A sidecar
    built under different ones is rejected as describing different
    documents, so the warm's whole benefit rests on this call matching the
    lane's. Pinned against the resolvers rather than against literals: a
    literal would pass while both sides drifted together to a wrong value.

    This closes a real gap. Flipping `per_field` here survives the entire
    533-test retrieval/bm25 selection — nothing else in the suite reads
    what the lane passes.
    """
    import aelfrice.retrieval as r

    db = tmp_path / "memory.db"
    _seed(db)
    seen: dict[str, object] = {}

    def _record(store: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(r, "_store_scoped_bm25f_cache", _record)

    store = MemoryStore(str(db))
    try:
        r.bm25f_cache_for_lane(store, now_ts=1_756_000_000)
        expected = {
            "anchor_weight": r.resolve_bm25f_anchor_weight_with_meta(
                store, now_ts=1_756_000_000
            ),
            "k3": r.resolve_bm25_k3(),
            "per_field": r.resolve_bm25f_per_field(),
            "b_anchor": r.resolve_bm25_b_anchor(),
        }
    finally:
        store.close()

    assert seen == expected, (
        "the lane helper is not passing the resolvers' values; a sidecar "
        "built through it would describe different documents than the lane "
        f"reads.\npassed={seen!r}\nresolvers={expected!r}"
    )


def test_the_l1_lane_routes_through_the_shared_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the lane must actually use it, or the invariant guards nothing.

    A retrieval call with the BM25F lane on has to reach
    `bm25f_cache_for_lane`. Without this the test above pins a helper the
    shipped path could stop calling.
    """
    import aelfrice.retrieval as r

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_BM25F", "1")

    calls: list[int] = []
    real = r.bm25f_cache_for_lane

    def _spy(store: object, *, now_ts: int) -> object:
        calls.append(now_ts)
        return real(store, now_ts=now_ts)  # type: ignore[arg-type]

    monkeypatch.setattr(r, "bm25f_cache_for_lane", _spy)

    store = MemoryStore(str(db))
    try:
        r.retrieve(store, "sidecar warm bm25 anchors", token_budget=2000)
    finally:
        store.close()

    assert calls, "the L1 lane did not go through bm25f_cache_for_lane"


# ---- AC4: failure is silent-safe ---------------------------------------


def test_a_warm_that_raises_is_swallowed_and_reports_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that cannot be opened must not propagate out of the child."""
    import aelfrice.store as store_mod
    from aelfrice.sidecar_warm import warm_sidecar

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    def _boom(*_a: object, **_k: object) -> MemoryStore:
        raise RuntimeError("store is on fire")

    monkeypatch.setattr(store_mod, "MemoryStore", _boom)

    assert warm_sidecar() is None


def test_a_spawn_failure_is_swallowed_and_reports_false(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Popen` itself failing (fork limit, missing interpreter) must be
    invisible to the caller."""
    import subprocess

    from aelfrice.sidecar_warm import spawn_sidecar_warm

    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("cannot fork")

    monkeypatch.setattr(subprocess, "Popen", _boom)

    assert spawn_sidecar_warm() is False


def test_session_start_output_is_identical_when_the_warm_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4, at the surface that matters.

    The hook's stdout, its return code, and its stderr silence must be the
    same whether the warm spawns or blows up. Compared against a run with
    the warm suppressed outright, so a difference in either direction fails.
    """
    import aelfrice.sidecar_warm as warm_mod
    from aelfrice.hook import _spawn_sidecar_warm, session_start

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    payload = '{"session_id": "s-1", "source": "startup"}'

    def _run() -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        rc = session_start(
            stdin=io.StringIO(payload), stdout=out, stderr=err
        )
        return rc, out.getvalue(), err.getvalue()

    # Warm suppressed by the conftest default: the behaviour of record.
    baseline = _run()
    assert baseline[0] == 0
    assert "locked baseline belief" in baseline[1]

    # Now let the spawn be attempted, and make it explode. The hook's own
    # helper is the thing under test: it must convert the raise into False.
    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)

    def _boom(env: object = None) -> bool:
        raise RuntimeError("spawn path is broken")

    monkeypatch.setattr(warm_mod, "spawn_sidecar_warm", _boom)

    assert _spawn_sidecar_warm() is False, (
        "the hook's helper let the warm's exception escape"
    )
    broken = _run()

    assert broken == baseline, (
        "SessionStart behaved differently when the warm failed; AC4 requires "
        f"it to be exactly as it is today.\nbaseline={baseline!r}\n"
        f"broken={broken!r}"
    )


# ---- AC2: SessionStart does not absorb the rebuild ----------------------


def test_session_start_does_not_build_the_index_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2. The warm must be spawned, not run.

    With the spawn replaced by a recorder, `SessionStart` returns having
    called it exactly once and having written no sidecar of its own. If the
    hook ever absorbed the build synchronously, the sidecar file would exist
    when it returns — the latency moved rather than removed.
    """
    from aelfrice import hook as hook_mod
    from aelfrice.hook import session_start

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    calls: list[bool] = []
    monkeypatch.setattr(
        hook_mod, "_spawn_sidecar_warm", lambda: calls.append(True) or True
    )

    rc = session_start(
        stdin=io.StringIO('{"session_id": "s-1"}'),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert rc == 0
    assert calls == [True], "SessionStart did not dispatch the warm"
    assert not _sidecar(db).exists(), (
        "SessionStart built the index on its own thread of control; AC2 "
        "forbids it absorbing the rebuild synchronously"
    )


def test_the_env_var_suppresses_the_spawn(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-out has to actually opt out, or the conftest default that
    keeps the suite from forking ~100 children is a lie."""
    import subprocess

    from aelfrice.sidecar_warm import spawn_sidecar_warm

    spawned: list[object] = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: spawned.append(a) or object()
    )

    monkeypatch.setenv("AELF_NO_SIDECAR_WARM", "1")
    assert spawn_sidecar_warm() is False
    assert spawned == []

    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)
    assert spawn_sidecar_warm() is True
    assert len(spawned) == 1
