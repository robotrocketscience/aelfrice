"""Unit tests for the v1.7 (#289 / #364) relevance floor.

Covers:
  - composite-score formula determinism + clamp behavior
  - L0 always-pack guarantee (no floor on locked beliefs)
  - L1 hard-floor behavior (above-floor packs, below-floor drops)
  - all-floored-out + no-locks => empty rebuild block ("")
  - `[rebuild_floor]` config override resolves and validates
  - rebuild_log `n_dropped_by_floor` accounting matches drops
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.context_rebuilder import (
    DEFAULT_FLOOR_L1,
    DEFAULT_FLOOR_SESSION,
    RecentTurn,
    floor_composite_score,
    load_rebuilder_config,
    rebuild_v14,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db_path: Path, beliefs: list[Belief]) -> MemoryStore:
    store = MemoryStore(str(db_path))
    for b in beliefs:
        store.insert_belief(b)
    return store


# --- composite-score formula --------------------------------------------


def test_composite_zero_bm25_yields_zero() -> None:
    """`bm25_raw=0` (no FTS5 signal) zeroes the composite. Posterior
    cannot rescue a belief that did not match any query token."""
    assert floor_composite_score(0.0, 1.0, 1.0) == 0.0


def test_composite_strong_bm25_neutral_posterior_is_three_quarters() -> None:
    """bm25_raw <= -1 saturates `bm25_normalized = 1.0`; with
    `posterior_mean = 0.5` the composite is `1.0 * (0.5 + 0.25) = 0.75`."""
    assert abs(floor_composite_score(-1.5, 1.0, 1.0) - 0.75) < 1e-9


def test_composite_high_posterior_lifts_score() -> None:
    """`posterior_mean = 1.0` (alpha >> beta) drives the composite to
    `bm25_normalized * 1.0`, the upper bound."""
    composite = floor_composite_score(-1.0, 100.0, 0.5)
    assert composite > 0.99


def test_composite_off_fts5_candidate_uses_unit_bm25() -> None:
    """Entity-only / BFS / session-only candidates pass `bm25_raw=None`
    and get `bm25_normalized = 1.0`. Floor decision rests on
    posterior alone for those candidates."""
    none_composite = floor_composite_score(None, 1.0, 1.0)
    saturated_composite = floor_composite_score(-1.5, 1.0, 1.0)
    assert none_composite == saturated_composite


def test_composite_is_deterministic() -> None:
    """Same inputs -> same composite, every call. The floor decision
    must not jitter run-to-run for the same store snapshot."""
    args = (-0.42, 3.0, 2.0)
    a = floor_composite_score(*args)
    b = floor_composite_score(*args)
    c = floor_composite_score(*args)
    assert a == b == c


# --- L0 always packs ----------------------------------------------------


def test_locked_belief_packs_under_aggressive_floor(tmp_path: Path) -> None:
    """An L0 lock with no query-overlap survives even an absurdly
    high floor. L0 bypasses the floor by contract."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1",
            "completely unrelated locked truth",
            lock_level=LOCK_USER,
            locked_at="2026-04-26T00:00:00Z",
        )],
    )
    try:
        block = rebuild_v14(
            [RecentTurn(role="user", text="totally different topic")],
            store,
            floor_session=10.0,
            floor_l1=10.0,
        )
    finally:
        store.close()
    assert "completely unrelated locked truth" in block


# --- empty path ---------------------------------------------------------


def test_empty_when_all_hits_floored_and_no_locks(tmp_path: Path) -> None:
    """No locks + every candidate below the floor -> rebuild_v14
    returns the empty string."""
    store = _seed(
        tmp_path / "m.db",
        [_mk("F1", "kitchen has bananas")],
    )
    try:
        block = rebuild_v14(
            [RecentTurn(role="user", text="kitchen contents")],
            store,
            floor_session=10.0,
            floor_l1=10.0,
        )
    finally:
        store.close()
    assert block == ""


def test_empty_path_default_zero_floor_packs_normally(tmp_path: Path) -> None:
    """Backwards-compatible default: `floor_session=0.0, floor_l1=0.0`
    gates nothing — direct callers see the v1.6 packing behavior."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1",
            "kitchen has bananas",
            lock_level=LOCK_USER,
            locked_at="2026-04-26T00:00:00Z",
        )],
    )
    try:
        block = rebuild_v14(
            [RecentTurn(role="user", text="kitchen contents")],
            store,
        )
    finally:
        store.close()
    assert block != ""
    assert "kitchen has bananas" in block


# --- config loader ------------------------------------------------------


def test_default_floor_constants_match_spec() -> None:
    """Pin the placeholder values to `docs/relevance_floor.md` §4.
    A change to either trips this test so a follow-up calibration
    PR cannot slip past review unnoticed."""
    assert DEFAULT_FLOOR_SESSION == 0.10
    assert DEFAULT_FLOOR_L1 == 0.40


def test_config_override_resolves_floor_values(tmp_path: Path) -> None:
    """`[rebuild_floor]` block overrides defaults; resolved cfg has
    operator-tuned values."""
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuild_floor]\nsession = 0.25\nl1 = 0.55\n',
        encoding="utf-8",
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.floor_session == 0.25
    assert cfg.floor_l1 == 0.55


def test_config_negative_floor_falls_back_to_default(tmp_path: Path) -> None:
    """Out-of-range values degrade silently to defaults; never raise."""
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuild_floor]\nsession = -1.0\nl1 = -0.5\n',
        encoding="utf-8",
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.floor_session == DEFAULT_FLOOR_SESSION
    assert cfg.floor_l1 == DEFAULT_FLOOR_L1


def test_config_zero_floor_is_valid_explicit_opt_out(tmp_path: Path) -> None:
    """`0.0` is a valid operator-tunable value (full opt-out of floor),
    distinct from the placeholder defaults."""
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuild_floor]\nsession = 0.0\nl1 = 0.0\n',
        encoding="utf-8",
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.floor_session == 0.0
    assert cfg.floor_l1 == 0.0


# --- rebuild_log accounting --------------------------------------------


def test_rebuild_log_records_floor_drops(tmp_path: Path) -> None:
    """`pack_summary.n_dropped_by_floor` matches the count of dropped
    candidates whose `reason` starts with `below_floor_`."""
    store = _seed(
        tmp_path / "m.db",
        [
            _mk(
                "L1",
                "anchor lock for the rebuild path",
                lock_level=LOCK_USER,
                locked_at="2026-04-26T00:00:00Z",
            ),
            _mk("F1", "weak match content"),
        ],
    )
    log_path = tmp_path / "rebuild.jsonl"
    try:
        block = rebuild_v14(
            [RecentTurn(
                role="user",
                text="weak match content query",
                session_id="sess-x",
            )],
            store,
            rebuild_log_path=log_path,
            rebuild_log_enabled=True,
            session_id_for_log="sess-x",
            floor_session=10.0,
            floor_l1=10.0,
        )
    finally:
        store.close()
    # Block is non-empty — the L0 lock packed under any floor.
    assert block
    raw = log_path.read_text(encoding="utf-8")
    record = json.loads(raw.strip().splitlines()[-1])
    summary = record["pack_summary"]
    n_floor = summary["n_dropped_by_floor"]
    candidate_floor = sum(
        1
        for c in record["candidates"]
        if isinstance(c.get("reason"), str)
        and c["reason"].startswith("below_floor_")
    )
    # Invariant: n_dropped_by_floor matches the count of candidates
    # whose reason starts with `below_floor_`. (We don't pin a specific
    # `>= 1` here because retrieve()'s candidate set in this in-process
    # path can vary; the invariant is the load-bearing claim.)
    assert n_floor == candidate_floor


def test_session_scoped_belief_floored_increments_count(tmp_path: Path) -> None:
    """Pin a guaranteed floor drop: a session-scoped belief that the
    rebuilder always considers gets dropped under an aggressive floor,
    and the rebuild_log records that drop in `n_dropped_by_floor`.
    """
    sid = "sess-floor-pin"
    # Inject a belief tagged with the session_id so `_session_scoped_hits`
    # picks it up unconditionally — independent of FTS5 / retrieve().
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        store.insert_belief(_mk(
            "L1",
            "anchor lock to keep block non-empty",
            lock_level=LOCK_USER,
            locked_at="2026-04-26T00:00:00Z",
        ))
        # Insert via raw SQL so we can pin session_id (insert_belief
        # respects it but the helper doesn't take it).
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE beliefs SET session_id = ? WHERE id = ?",
            (None, "L1"),  # locked belief stays unsessioned
        )
        store.insert_belief(_mk("S1", "session scoped low posterior"))
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE beliefs SET session_id = ?, alpha = ?, beta = ? "
            "WHERE id = ?",
            (sid, 0.01, 100.0, "S1"),  # posterior_mean ~ 0.0001
        )
        store._conn.commit()  # pyright: ignore[reportPrivateUsage]

        log_path = tmp_path / "rebuild.jsonl"
        block = rebuild_v14(
            [RecentTurn(role="user", text="any topic", session_id=sid)],
            store,
            rebuild_log_path=log_path,
            rebuild_log_enabled=True,
            session_id_for_log=sid,
            # Off-FTS5 candidate with bm25=None gets bm25_normalized=1.0;
            # composite = 1.0 * (0.5 + 0.5 * pm). With pm ~ 1e-4, composite
            # ~ 0.5. Floor 0.75 drops it; floor_l1=0.0 doesn't gate L1.
            floor_session=0.75,
            floor_l1=0.0,
        )
    finally:
        store.close()
    assert block  # L0 lock kept it non-empty
    raw = log_path.read_text(encoding="utf-8")
    record = json.loads(raw.strip().splitlines()[-1])
    summary = record["pack_summary"]
    assert summary["n_dropped_by_floor"] >= 1
    s1_candidate = next(
        c for c in record["candidates"] if c["belief_id"] == "S1"
    )
    assert s1_candidate["decision"] == "dropped"
    assert isinstance(s1_candidate["reason"], str)
    assert s1_candidate["reason"].startswith("below_floor_session")
