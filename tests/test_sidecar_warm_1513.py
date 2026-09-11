"""#1513 — the SessionStart BM25 sidecar warm.

The measured cost is a session-first tail: bucketed by position within the
session, `benchmarks/sidecar_rebuild_rate.py` reports a materially higher
`full_rebuild` rate on the first scored fire of a session than on every
later one. No magnitude is quoted, here or anywhere else on this branch —
the audit log grows and rotates, successive re-derivations moved that rate
substantially, and the earlier populations no longer exist.
Run the script for the current split. The fix warms the sidecar from a
detached child spawned at `SessionStart`.

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


# ---- the decaying parameter: sharing a resolver is not sharing an answer ----
#
# Three of the four index parameters resolve from env and TOML, so two
# processes agree on them by construction. `anchor_weight` does not: under
# the #757 meta-belief it decodes a decaying posterior, so the warm reading
# its clock and the fire reading its own seconds later can land on different
# integers. `_load_sidecar` then rejects the blob and the fire rebuilds --
# the warm becomes pure cost, with nothing in the outcome vocabulary saying
# the feature stopped working.
#
# The two timestamps below are a real crossing of the 30-day-half-life decay
# under twenty `bm25_l0_ratio` evidence events, found by bisection rather
# than assumed; the test asserts the divergence before it relies on it.

_T_WARM: int = 1_704_450_164
_T_FIRE: int = _T_WARM + 30


def _install_a_moving_anchor_weight(db: Path) -> None:
    """Give the store a #757 meta-belief still moving through the band."""
    import aelfrice.retrieval as r
    from aelfrice.meta_beliefs import SIGNAL_BM25_L0_RATIO

    store = MemoryStore(str(db))
    try:
        r.install_bm25f_anchor_weight_meta_belief(store, now_ts=1_700_000_000)
        for _ in range(20):
            store.update_meta_belief(
                r.META_BM25F_ANCHOR_WEIGHT_KEY,
                SIGNAL_BM25_L0_RATIO,
                1.0,
                now_ts=1_700_000_000,
            )
    finally:
        store.close()


def _resolved_at(db: Path, ts: int) -> int:
    import aelfrice.retrieval as r

    store = MemoryStore(str(db))
    try:
        return r.resolve_bm25f_anchor_weight_with_meta(store, now_ts=ts)
    finally:
        store.close()


def _warm_at(ts: int, monkeypatch: pytest.MonkeyPatch) -> str | None:
    """Run the warm child with its wall clock pinned to `ts`."""
    import time

    from aelfrice.sidecar_warm import warm_sidecar

    real = time.time
    time.time = lambda: float(ts)  # type: ignore[assignment]
    try:
        return warm_sidecar()
    finally:
        time.time = real  # type: ignore[assignment]


def test_a_thirty_second_gap_does_not_make_the_fire_reject_the_warm_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm's blob must survive the clock gap it was built across.

    Without the pin the fire resolves its own `anchor_weight`, differs from
    the warm's by one, and `_load_sidecar` throws the blob away: the warm
    pays a full rebuild, the fire pays a second one, and the outcome the
    audit log records is the same `full_rebuild` it recorded before the
    feature existed.
    """
    import aelfrice.retrieval as r
    from aelfrice.sidecar_outcome import (
        last_sidecar_outcome,
        reset_sidecar_outcome,
    )

    db = tmp_path / "memory.db"
    _seed(db)
    _install_a_moving_anchor_weight(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT", "1")

    # The premise, asserted rather than assumed. If the two clocks happened
    # to decode the same integer the assertion below would pass on the
    # unfixed code and prove nothing.
    at_warm = _resolved_at(db, _T_WARM)
    at_fire = _resolved_at(db, _T_FIRE)
    assert at_warm != at_fire, (
        "the two timestamps resolve the same anchor_weight, so this test "
        f"cannot see the divergence it exists to pin ({at_warm} == {at_fire})"
    )

    assert _warm_at(_T_WARM, monkeypatch) == "full_rebuild"
    assert _sidecar(db).exists()

    reset_sidecar_outcome()
    store = MemoryStore(str(db))
    try:
        cache = r.bm25f_cache_for_lane(store, now_ts=_T_FIRE)
        cache.get()
    finally:
        store.close()

    assert cache.anchor_weight == at_warm, (
        "the fire resolved its own anchor_weight instead of the one the "
        f"fresh sidecar carries ({cache.anchor_weight} != {at_warm})"
    )
    assert last_sidecar_outcome() == "fresh", (
        "the fire rejected the warm's blob over a one-integer anchor_weight "
        "difference that a 30-second clock gap produced; the warm is pure "
        "cost in this configuration"
    )


def test_a_store_write_lifts_the_pin_so_the_meta_belief_still_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinguishing half. Pinning to the blob must not freeze the knob.

    A fix that always took the sidecar's `anchor_weight` would satisfy the
    test above and silently stop the #757 meta-belief from ever taking
    effect. The pin is scoped to a *fresh* blob, so the first write to the
    store lifts it -- and a rebuild is due at that point anyway.
    """
    import aelfrice.retrieval as r

    db = tmp_path / "memory.db"
    _seed(db)
    _install_a_moving_anchor_weight(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT", "1")

    at_warm = _resolved_at(db, _T_WARM)
    at_fire = _resolved_at(db, _T_FIRE)
    assert at_warm != at_fire

    assert _warm_at(_T_WARM, monkeypatch) == "full_rebuild"

    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L9", "a belief written after the warm"))
    finally:
        store.close()

    store = MemoryStore(str(db))
    try:
        cache = r.bm25f_cache_for_lane(store, now_ts=_T_FIRE)
    finally:
        store.close()

    assert cache.anchor_weight == at_fire, (
        "a stale sidecar is still pinning the anchor_weight, so the #757 "
        f"meta-belief can never move it ({cache.anchor_weight} != {at_fire})"
    )


def test_with_the_meta_belief_flag_off_the_sidecar_pins_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blast radius. The flag ships off, and off must read nothing.

    A sidecar built under `anchor_weight` 9 sits next to the store. With the
    flag off the lane must still resolve `DEFAULT_ANCHOR_WEIGHT`, and must
    not open the blob to find that out.
    """
    import aelfrice.bm25 as b
    import aelfrice.retrieval as r
    from aelfrice.bm25 import DEFAULT_ANCHOR_WEIGHT

    db = tmp_path / "memory.db"
    _seed(db)
    _install_a_moving_anchor_weight(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    monkeypatch.setenv("AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT", "1")
    assert _warm_at(_T_WARM, monkeypatch) == "full_rebuild"
    store = MemoryStore(str(db))
    try:
        assert b.sidecar_anchor_weight(store) == _resolved_at(db, _T_WARM)
    finally:
        store.close()

    monkeypatch.delenv("AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT")
    peeked: list[object] = []
    monkeypatch.setattr(r, "_store_scoped_bm25f_cache", lambda s, **kw: kw)
    monkeypatch.setattr(
        b, "sidecar_anchor_weight", lambda s: peeked.append(s) or 9
    )

    store = MemoryStore(str(db))
    try:
        passed = r.bm25f_cache_for_lane(store, now_ts=_T_FIRE)
    finally:
        store.close()

    assert passed["anchor_weight"] == DEFAULT_ANCHOR_WEIGHT, passed
    assert peeked == [], "the flag-off path read the sidecar header"


# ---- the load-bearing edge: the child the parent never waits on ---------
#
# `_CHILD_SOURCE` is the only line connecting the spawned process to the
# work it exists to do, and every test above this point either calls
# `warm_sidecar()` in-process or replaces `Popen` with a recorder. Neither
# executes that string. Round 1 shipped with it uncovered: mutating it to
# `from aelfrice.NOSUCHMODULE import main; raise SystemExit(main())` left
# the whole file green while, in production, `Popen` still succeeded, the
# child died rc=1 with all three streams on /dev/null, and nothing
# downstream could tell the difference. The two tests below run the real
# child.


def _warm_audit_rows(db: Path) -> list[dict[str, object]]:
    """Every `sidecar_warm` row the child left next to `db`."""
    import json

    from aelfrice.sidecar_warm import WARM_AUDIT_HOOK

    log = db.parent / "hook_audit.jsonl"
    if not log.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("hook") == WARM_AUDIT_HOOK:
            rows.append(rec)
    return rows


def _await_warm_row(db: Path, deadline_s: float = 60.0) -> dict[str, object]:
    """Block until the detached child records its outcome, or fail.

    Polling is the only option available: the child is deliberately
    detached, `spawn_sidecar_warm` returns a bool rather than the `Popen`,
    and there is nothing to `wait()` on. The deadline is generous against a
    loaded machine because the assertion is on the row's CONTENT, never on
    how long it took — this file publishes no timing figure.
    """
    import time

    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        rows = _warm_audit_rows(db)
        if rows:
            return rows[-1]
        time.sleep(0.05)
    raise AssertionError(
        "the detached warm child wrote no sidecar_warm audit row within "
        f"{deadline_s}s. The child never reached `main()` — check "
        "`_CHILD_SOURCE`, which is the only line that names it."
    )


@pytest.mark.timeout(180)
def test_the_real_detached_child_warms_the_store_it_was_spawned_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end edge, run through a real `Popen`.

    `spawn_sidecar_warm()` returning True proves only that `Popen` did not
    raise; it is true of a child that dies on its first import. What is
    asserted here is the child's observable effect on the store it was
    spawned for: it records its outcome, it leaves a sidecar on disk, and
    the fire that follows reads that sidecar instead of rebuilding.

    The child recomputes `db_path()` for itself, so `AELFRICE_DB` has to be
    in the real environment (`monkeypatch.setenv` puts it there) — no
    `setattr` in this process can steer a separate interpreter.
    """
    from aelfrice.sidecar_warm import spawn_sidecar_warm

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)
    assert not _sidecar(db).exists()
    assert _warm_audit_rows(db) == []

    assert spawn_sidecar_warm() is True

    row = _await_warm_row(db)
    assert row.get("sidecar_outcome") == "full_rebuild", (
        "the child ran but did not build the index it was spawned to "
        f"build: {row!r}"
    )
    assert _sidecar(db).exists(), (
        "the child reported an outcome but left no sidecar on disk"
    )

    assert _outcome_of_a_retrieval_fire(db) == "fresh", (
        "the fire after a real spawned warm still rebuilt the index; the "
        "session-first rate cannot move"
    )


def test_the_spawn_detaches_the_child_from_the_parents_session(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """`start_new_session=True` is passed, so the host cannot reap the warm.

    This pins the ARGUMENT, not the kernel's behaviour: the spawn returns a
    bool rather than the `Popen`, so no caller can read the child's process
    group back out. Stated plainly because an argument-level assertion is
    weaker than the end-to-end test above and should not be read as more.

    It is still worth pinning. Without the flag the child stays in the
    hook's process group, and a host that signals that group on hook exit
    kills the build mid-flight — the failure mode the module docstring
    rejects the daemon-thread design for.
    """
    import subprocess

    from aelfrice.sidecar_warm import spawn_sidecar_warm

    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: seen.append(k) or object()
    )
    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)

    assert spawn_sidecar_warm() is True
    assert len(seen) == 1
    assert seen[0].get("start_new_session") is True, (
        "the warm child shares the hook's process group; a host that "
        f"signals it on hook exit kills the build: {seen[0]!r}"
    )


# ---- the new writer, and the two switches that must reach it ----------
#
# `_record_warm_outcome` is a writer this branch adds to a process the user
# cannot see: detached, all three streams on /dev/null, return code
# discarded. Two properties of it were shipped unpinned, and both were
# demonstrated unpinned by mutation against the whole suite.
#
# The audit opt-out. Deleting `if not cfg.enabled: return` from
# `_record_warm_outcome` left the full suite green at 8624 passed. In
# production a user who set `AELFRICE_HOOK_AUDIT=0`, or `enabled = false`
# under `[hook_audit]`, would still get one row per session from a process
# they cannot see. That check is the only thing between the writer and a
# user who asked for no audit log.
#
# The `hook` label. Changing `WARM_AUDIT_HOOK` to `user_prompt_submit` also
# left the suite green at 8624 — `_warm_audit_rows` above filters on the
# constant, so it follows any mutation of it. The tests below assert the
# LITERAL and the scored consequence instead.


def _all_audit_rows(db: Path) -> list[dict[str, object]]:
    """Every row in the audit log next to `db`, whatever its `hook`.

    Deliberately unfiltered, unlike `_warm_audit_rows`: a filter keyed on
    `WARM_AUDIT_HOOK` follows a mutation of that constant, so counting
    through one cannot distinguish "no row was written" from "the row was
    written under another name".
    """
    import json

    log = db.parent / "hook_audit.jsonl"
    if not log.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _audit_toml(tmp_path: Path, *, enabled: bool) -> None:
    """Write a `.aelfrice.toml` that decides the audit for `tmp_path`.

    Written in every case, enabled arm included, so the config the test
    resolves is this file rather than whatever happens to sit above the
    pytest temp directory on the machine running it.
    """
    (tmp_path / ".aelfrice.toml").write_text(
        f"[hook_audit]\nenabled = {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )


def test_the_env_opt_out_keeps_the_warm_row_out_of_the_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AELFRICE_HOOK_AUDIT=0` has to reach the detached child's writer.

    The enabled arm runs first and is what makes this distinguishing: an
    assertion that the log is empty passes just as well on a misdirected
    `db_path()` or on a writer whose every exception is swallowed, neither
    of which is the opt-out working.
    """
    from aelfrice.sidecar_warm import _record_warm_outcome

    db = tmp_path / "memory.db"
    monkeypatch.chdir(tmp_path)
    _audit_toml(tmp_path, enabled=True)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _record_warm_outcome("full_rebuild")
    assert len(_all_audit_rows(db)) == 1, (
        "the warm wrote no row with the audit enabled, so the opt-out arm "
        "below would pass for the wrong reason"
    )

    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "0")
    _record_warm_outcome("full_rebuild")

    assert len(_all_audit_rows(db)) == 1, (
        "AELFRICE_HOOK_AUDIT=0 did not reach `_record_warm_outcome`: the "
        "detached child appended a row for a user who opted out of the "
        f"audit log. Rows: {_all_audit_rows(db)!r}"
    )


def test_the_toml_opt_out_keeps_the_warm_row_out_of_the_audit_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[hook_audit] enabled = false` has to reach it too.

    Separate from the env arm because they are separate tiers of
    `load_hook_audit_config`: the env var short-circuits before the TOML
    walk, so a writer that consulted only the env var would pass the test
    above and still write through a user's `.aelfrice.toml`. Only the
    `enabled` bit changes between the two halves.
    """
    from aelfrice.sidecar_warm import _record_warm_outcome

    db = tmp_path / "memory.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(db))

    _audit_toml(tmp_path, enabled=True)
    _record_warm_outcome("full_rebuild")
    assert len(_all_audit_rows(db)) == 1, (
        "the warm wrote no row with `enabled = true`, so the opt-out arm "
        "below would pass for the wrong reason"
    )

    _audit_toml(tmp_path, enabled=False)
    _record_warm_outcome("full_rebuild")

    assert len(_all_audit_rows(db)) == 1, (
        "`[hook_audit] enabled = false` did not reach "
        "`_record_warm_outcome`: the detached child appended a row for a "
        f"user who opted out of the audit log. Rows: {_all_audit_rows(db)!r}"
    )


def test_the_warm_row_carries_the_literal_hook_name_sidecar_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`WARM_AUDIT_HOOK` is `sidecar_warm`, asserted as a literal.

    The value is asserted here, not the mechanism. Every other assertion
    in this file that touches the warm row reaches it through
    `_warm_audit_rows`, which filters on the constant and therefore
    follows any change to it: renaming the constant to
    `user_prompt_submit` — the exact corruption `_record_warm_outcome`'s
    own docstring argues against — left the whole suite green at 8624
    passed.

    The written row is checked against the same literal rather than
    against the constant, so this also fails on a module that imports a
    correct constant and then stamps something else.
    """
    from aelfrice.sidecar_warm import WARM_AUDIT_HOOK, _record_warm_outcome

    assert WARM_AUDIT_HOOK == "sidecar_warm"

    db = tmp_path / "memory.db"
    monkeypatch.chdir(tmp_path)
    _audit_toml(tmp_path, enabled=True)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _record_warm_outcome("full_rebuild")

    rows = _all_audit_rows(db)
    assert len(rows) == 1, rows
    assert rows[0].get("hook") == "sidecar_warm", rows[0]


def test_a_warm_row_is_not_scored_as_a_fire_by_the_rebuild_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production consequence of that literal, read off the script.

    `benchmarks/sidecar_rebuild_rate.py` selects rows by
    `hook == "user_prompt_submit"`. A warm row stamped with that name is
    indistinguishable from a user-visible fire: it carries a
    `sidecar_outcome`, so it enters the scored denominator and the
    full-rebuild rate the warm exists to move, and — carrying no
    `session_id` — it reaches the position split as an unattributed fire.
    One real fire and one warm row show all three.
    """
    import json
    import subprocess
    import sys

    from aelfrice.sidecar_warm import _record_warm_outcome

    db = tmp_path / "memory.db"
    monkeypatch.chdir(tmp_path)
    _audit_toml(tmp_path, enabled=True)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    log = db.parent / "hook_audit.jsonl"
    log.write_text(
        json.dumps(
            {
                "hook": "user_prompt_submit",
                "ts": "2026-09-01T00:00:00Z",
                "session_id": "s",
                "sidecar_outcome": "fresh",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _record_warm_outcome("full_rebuild")
    assert len(_all_audit_rows(db)) == 2, _all_audit_rows(db)

    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "sidecar_rebuild_rate.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(log)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    assert "non-UPS rows (ignored)         1" in out, out
    assert "fires with an outcome (scored) 1" in out, out
    assert "of scored fires             0/1 = 0.00%" in out, (
        "the warm's own full_rebuild entered the rate it exists to "
        f"move\n{out}"
    )
    first = [ln for ln in out.splitlines() if "session-FIRST fires" in ln]
    assert len(first) == 1, out
    assert "0/1 = 0.0%" in first[0], first[0]
    assert "no session_id" not in out, (
        "the warm row reached the session-position split as an "
        f"unattributed fire\n{out}"
    )
