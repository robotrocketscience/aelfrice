"""Acceptance tests for #373 / #199 H3 — selective L0 injection.

Each test maps to one of the issue's acceptance bullets:

1. N=50 locks + a focused git-commit prompt -> <= max_k git-relevant
   locks reach the model (not all 50).
2. Token-budget regression: <aelfrice-memory> payload size scales with
   max_k, not with locked-set cardinality.
3. Locked semantics preserved: a locked belief still cannot be demoted
   by retrieval; only its per-turn injection is conditional.
4. Determinism: bytes-identical injection ordering for the same
   (locked_set_snapshot, prompt) pair.
5. inject_all_locked = true restores legacy unbounded L0 (no data loss
   path).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.retrieval import top_k_locks
from aelfrice.store import MemoryStore


# Reused fixture vocabulary: 50 locks, exactly five mention `git commit`
# (the focused-prompt target). The other 45 are off-topic. The focused
# prompt is "what's the right git commit message format?".
_GIT_LOCK_TEMPLATES = [
    "git commit messages should follow conventional-commits prefix",
    "git commit body explains the why not the what",
    "git commit subject under 70 characters is house style",
    "git commit signing uses ssh key id_rrs",
    "git commit body uses heredoc for multi-line messages",
]
_NON_GIT_TOPICS = [
    "react useState dependency arrays must be exhaustive",
    "kubernetes pod resource limits prevent oom-kill cascades",
    "tailwind class ordering uses headwind plugin",
    "postgres autovacuum tuning depends on table churn rate",
    "sqlite wal mode enables concurrent readers",
    "docker compose v2 ditches the docker-compose binary",
    "rust borrow checker rejects aliased mutable references",
    "go context cancellation propagates through goroutines",
    "grpc deadline propagation needs explicit context.WithDeadline",
    "redis persistence has rdb snapshots and aof appendonly",
]


def _mk_lock(bid: str, content: str, locked_at: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0,  # saturated posterior — locks
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_50_locks(db_path: Path) -> list[str]:
    """Seed 50 locked beliefs: 5 git-commit-relevant + 45 off-topic."""
    store = MemoryStore(str(db_path))
    ids: list[str] = []
    try:
        for i, c in enumerate(_GIT_LOCK_TEMPLATES):
            bid = f"L_git_{i:02d}"
            store.insert_belief(_mk_lock(
                bid, c, f"2026-04-26T05:{i:02d}:00Z"
            ))
            ids.append(bid)
        for i in range(45):
            template = _NON_GIT_TOPICS[i % len(_NON_GIT_TOPICS)]
            bid = f"L_off_{i:02d}"
            store.insert_belief(_mk_lock(
                bid, f"{template} (variant {i})",
                f"2026-04-26T04:{i:02d}:00Z"
            ))
            ids.append(bid)
    finally:
        store.close()
    return ids


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _write_toml(tmp_path: Path, body: str) -> None:
    (tmp_path / ".aelfrice.toml").write_text(body, encoding="utf-8")


def _run_ups(prompt: str) -> str:
    sin = io.StringIO(json.dumps({"prompt": prompt}))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    return sout.getvalue()


# ---- AC1 + AC2: focused prompt + payload-size scaling -------------------


def test_focused_prompt_caps_injected_locks_at_max_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """50 locks, max_k=5, git-commit prompt -> exactly the 5 git locks land."""
    db = tmp_path / "memory.db"
    _seed_50_locks(db)
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)  # default config: selective on, max_k=5
    out = _run_ups("what's the right git commit message format?")
    # Every L_git_* should appear; no L_off_* should.
    for i in range(5):
        assert f'id="L_git_{i:02d}"' in out, f"missing L_git_{i:02d}"
    assert 'id="L_off_' not in out, (
        "off-topic lock leaked into top-K injection"
    )


def test_payload_scales_with_max_k_not_locked_cardinality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance bullet 2: payload size is bounded by max_k.

    Compares <aelfrice-memory> output sizes across max_k=2 vs max_k=5
    against the same 50-lock store. The smaller k must produce a
    strictly smaller payload — otherwise the cap isn't load-bearing.
    """
    db = tmp_path / "memory.db"
    _seed_50_locks(db)
    _set_db(monkeypatch, db)

    _write_toml(tmp_path, (
        "[user_prompt_submit_hook]\nlocked_max_k = 2\n"
    ))
    monkeypatch.chdir(tmp_path)
    small = _run_ups("git commit message format")

    _write_toml(tmp_path, (
        "[user_prompt_submit_hook]\nlocked_max_k = 5\n"
    ))
    large = _run_ups("git commit message format")

    assert len(small) < len(large), (
        f"max_k=2 ({len(small)} bytes) should be smaller than "
        f"max_k=5 ({len(large)} bytes)"
    )
    # And both must be much smaller than what 50 locks would produce
    # under legacy. Use legacy mode as the upper-bound reference.
    _write_toml(tmp_path, (
        "[user_prompt_submit_hook]\ninject_all_locked = true\n"
    ))
    legacy = _run_ups("git commit message format")
    assert len(large) < len(legacy)


# ---- AC4: determinism ---------------------------------------------------


def test_topk_locks_is_deterministic_across_repeat_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same (locked_set_snapshot, prompt) -> bytes-identical injection."""
    db = tmp_path / "memory.db"
    _seed_50_locks(db)
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)
    prompt = "git commit message format and signing"
    out_a = _run_ups(prompt)
    out_b = _run_ups(prompt)
    out_c = _run_ups(prompt)
    assert out_a == out_b == out_c


def test_topk_locks_helper_is_pure_and_deterministic() -> None:
    """Unit-level guard on the scoring helper.

    Ensures sort stability (input-order tiebreak when scores tie) and
    that repeated calls return identical lists.
    """
    locks = [
        _mk_lock("a", "git commit message", "2026-04-26T03:00:00Z"),
        _mk_lock("b", "git push origin main", "2026-04-26T02:00:00Z"),
        _mk_lock("c", "no overlap whatever", "2026-04-26T01:00:00Z"),
        _mk_lock("d", "git commit signing key", "2026-04-26T00:00:00Z"),
    ]
    out1 = top_k_locks(locks, "git commit", k=2)
    out2 = top_k_locks(locks, "git commit", k=2)
    assert [b.id for b in out1] == [b.id for b in out2]
    # 'a' has overlap=2 (git+commit), 'd' has overlap=2 (git+commit),
    # 'b' has overlap=1 (git), 'c' has 0. With score tie between a/d,
    # input-order wins -> a before d.
    assert [b.id for b in out1] == ["a", "d"]


# ---- AC5: inject_all_locked fallback (no data loss) ---------------------


def test_inject_all_locked_restores_legacy_unbounded_l0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_50_locks(db)
    _set_db(monkeypatch, db)
    _write_toml(tmp_path, (
        "[user_prompt_submit_hook]\ninject_all_locked = true\n"
    ))
    monkeypatch.chdir(tmp_path)
    out = _run_ups("git commit message format")
    # All 50 locks must be present (this is the no-data-loss path).
    git_count = sum(1 for i in range(5) if f'id="L_git_{i:02d}"' in out)
    off_count = sum(1 for i in range(45) if f'id="L_off_{i:02d}"' in out)
    assert git_count == 5
    assert off_count == 45


# ---- AC3: locked semantics preserved ------------------------------------


def test_unselected_locks_are_not_demoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locks excluded from the per-turn top-K must keep their lock_level.

    Selective injection trims display, not authority — `lock_level`
    must remain LOCK_USER and demotion_pressure must not advance for
    locks the prompt failed to score.
    """
    db = tmp_path / "memory.db"
    _seed_50_locks(db)
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)
    _run_ups("git commit message format")

    # Re-open and verify every off-topic lock still has lock_level=user
    # and zero demotion_pressure.
    store = MemoryStore(str(db))
    try:
        all_locked = store.list_locked_beliefs()
        assert len(all_locked) == 50, (
            "selective injection must not delete or unlock beliefs"
        )
        for b in all_locked:
            assert b.lock_level == LOCK_USER, (
                f"{b.id}: lock_level changed to {b.lock_level}"
            )
            assert b.demotion_pressure == 0, (
                f"{b.id}: demotion_pressure advanced to "
                f"{b.demotion_pressure}"
            )
    finally:
        store.close()
