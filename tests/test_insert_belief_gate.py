"""Tests for the v2.x view-flip gate on `MemoryStore.insert_belief`
(#265 PR-B commit 4).

When `AELFRICE_WRITE_LOG_AUTHORITATIVE` is on, `insert_belief()`
walks the call stack to find the first frame outside `aelfrice.store`
and raises `WriteLogAuthorityViolation` unless that module appears in
`INSERT_BELIEF_ALLOWLIST` = {derivation_worker, wonder.simulator,
benchmark, migrate}. With the flag off (production default at this
commit), the check is a no-op.

Each test states a falsifiable hypothesis about the gate contract.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation_worker import run_worker
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    RETENTION_UNKNOWN,
    Belief,
)
from aelfrice.store import (
    INSERT_BELIEF_ALLOWLIST,
    MemoryStore,
    WriteLogAuthorityViolation,
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "gate.db"))
    yield s
    s.close()


def _belief(bid: str = "b" * 16, content: str = "alpha gate test") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-08T00:00:00+00:00",
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_AGENT_INFERRED,
        retention_class=RETENTION_UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Default-off behavior
# ---------------------------------------------------------------------------


def test_gate_off_by_default(store: MemoryStore) -> None:
    """Hypothesis: with no env var set, insert_belief from a test module
    (not on the allowlist) succeeds. The gate is a no-op when the flag
    is off — production default at this commit. Falsifiable by a raise
    on insert_belief without setting the env var."""
    store.insert_belief(_belief())
    assert store.count_beliefs() == 1


def test_gate_off_explicit_zero_value(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: an explicit non-truthy value (`0`, `false`, `off`,
    `no`, …) keeps the gate off. Falsifiable by a raise on any of
    these values."""
    for falsy in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", falsy)
        # Each iteration uses a distinct id so the row insert succeeds.
        bid = f"{ord(falsy[0:1] or 'x'):x}".rjust(16, "0")
        store.insert_belief(_belief(bid=bid, content=f"falsy {falsy!r}"))


# ---------------------------------------------------------------------------
# Gate-on, non-allowlisted callers raise
# ---------------------------------------------------------------------------


def test_gate_on_test_module_raises(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: with the gate on, calling insert_belief from this
    test module (not on the allowlist) raises
    `WriteLogAuthorityViolation`. The exception message names the
    offending module. Falsifiable by silent acceptance."""
    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    with pytest.raises(WriteLogAuthorityViolation) as exc_info:
        store.insert_belief(_belief())
    msg = str(exc_info.value)
    # Module name appears verbatim so the operator can grep it.
    assert "test_insert_belief_gate" in msg


def test_gate_on_truthy_variants(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: every truthy spelling (`1`, `true`, `yes`, `on`,
    case-insensitive, surrounded by whitespace) trips the gate.
    Falsifiable by silent acceptance for any variant."""
    for truthy in ("1", "true", "TRUE", "yes", "on", " on ", "True"):
        monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", truthy)
        with pytest.raises(WriteLogAuthorityViolation):
            store.insert_belief(_belief())


# ---------------------------------------------------------------------------
# Gate-on, allowlisted callers pass
# ---------------------------------------------------------------------------


def test_gate_on_worker_path_passes(
    store: MemoryStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: the worker is on the allowlist, so a
    `record_ingest` + `run_worker` round-trip with the gate on still
    writes a canonical belief. The worker reaches `insert_belief` via
    `insert_or_corroborate`; the stack-walk skips the aelfrice.store
    frames and finds `aelfrice.derivation_worker` as the true caller.
    Falsifiable by a raise inside the worker."""
    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:gate-on.md",
        raw_text="Aelfrice stores beliefs in a SQLite database.",
        raw_meta={"call_site": "filesystem_ingest"},
        ts="2026-05-08T01:00:00+00:00",
    )
    result = run_worker(store)
    assert result.beliefs_inserted == 1
    assert store.count_beliefs() == 1


def _simulator_caller(store: MemoryStore, b: Belief) -> None:
    """Helper invoked from inside `aelfrice.wonder.simulator` to exercise
    the allowlist path. Lives in this test module via dynamic
    `__name__` patching below."""
    store.insert_belief(b)


def test_gate_on_allowlisted_module_passes(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: when the calling frame's `__name__` is in
    `INSERT_BELIEF_ALLOWLIST`, the gate lets the write through.
    Simulated by patching the helper's globals to claim the allowlisted
    module name. Falsifiable by a raise despite the allowlisted
    identity. The simulator/benchmark/migrate paths are tested this
    way because their real call paths have shape that's awkward to
    drive from a unit test (synthetic corpus seeders, version
    migration tooling)."""
    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    for allowlisted in INSERT_BELIEF_ALLOWLIST:
        # Patch the helper's frame to look like it's in the allowlisted
        # module. The stack walk reads `frame.f_globals["__name__"]`.
        original = _simulator_caller.__globals__.get("__name__")
        _simulator_caller.__globals__["__name__"] = allowlisted
        try:
            bid = ("a" * 8 + str(hash(allowlisted) & 0xFFFF).zfill(8))[:16]
            _simulator_caller(store, _belief(bid=bid, content=allowlisted))
        finally:
            if original is not None:
                _simulator_caller.__globals__["__name__"] = original


# ---------------------------------------------------------------------------
# Allowlist surface
# ---------------------------------------------------------------------------


def test_allowlist_contents_are_ratified_set() -> None:
    """Hypothesis: the allowlist is exactly the ratified set of modules.

    PR #478 ratified the original four. #548 added wonder.lifecycle for
    speculative phantom ingest. Adding or removing entries is a design
    decision that requires re-ratification. Falsifiable by drift on either
    side.
    """
    assert INSERT_BELIEF_ALLOWLIST == frozenset({
        "aelfrice.derivation_worker",
        "aelfrice.wonder.simulator",
        "aelfrice.wonder.lifecycle",
        "aelfrice.benchmark",
        "aelfrice.migrate",
    })


def test_violation_message_lists_allowlist(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: the violation message includes the allowlist so the
    operator can debug a misrouted caller without reading source.
    Falsifiable by an empty / generic message."""
    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    try:
        store.insert_belief(_belief())
    except WriteLogAuthorityViolation as exc:
        msg = str(exc)
        for module in INSERT_BELIEF_ALLOWLIST:
            assert module in msg, (
                f"violation message missing allowlisted module {module!r}"
            )
    else:
        pytest.fail("insert_belief did not raise under the gate")


def test_insert_or_corroborate_walks_past_store_frames(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: when `insert_or_corroborate` calls `insert_belief`
    internally, the stack walk skips the aelfrice.store frames and
    checks the *outer* caller against the allowlist. With the test as
    outer caller, the gate raises. Falsifiable by silent acceptance
    (the gate would have to mistake `insert_or_corroborate`'s own
    frame for the true caller — that's the bug the skip-loop guards
    against)."""
    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    with pytest.raises(WriteLogAuthorityViolation):
        store.insert_or_corroborate(
            _belief(),
            source_type="filesystem_ingest",
        )


# ---------------------------------------------------------------------------
# Real-module smoke tests (production allowlist code paths)
# ---------------------------------------------------------------------------


def test_gate_on_benchmark_seed_corpus_passes(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: `aelfrice.benchmark.seed_corpus` writes via
    `insert_belief` and is on the allowlist; the production seeder
    works with the gate on. Falsifiable by a violation raised inside
    seed_corpus."""
    from aelfrice.benchmark import seed_corpus

    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    inserted = seed_corpus(store)
    assert inserted >= 1
    assert store.count_beliefs() == inserted


def test_gate_on_simulator_populate_store_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: `aelfrice.wonder.simulator.populate_store` writes
    via `insert_belief` and is on the allowlist; the synthetic-corpus
    seeder works with the gate on. Falsifiable by a violation raised
    inside populate_store."""
    import random

    from aelfrice.wonder.simulator import build_corpus, populate_store

    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    rng = random.Random(42)
    corpus = build_corpus(rng=rng, n_topics=4, n_atoms_per_topic=5)
    s = MemoryStore(str(tmp_path / "sim.db"))
    try:
        populate_store(s, corpus, rng=rng)
        assert s.count_beliefs() == 4 * 5
    finally:
        s.close()


def test_gate_on_migrate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: `aelfrice.migrate.migrate` writes via
    `insert_belief` and is on the allowlist; the v1→v2 migration
    works with the gate on. The legacy_unknown rows are emitted by
    the migration tool itself (per ratification: no ingest_log
    injection). Falsifiable by a violation inside migrate()."""
    from aelfrice.migrate import migrate

    # Seed a v1-shape source store with the gate OFF so we can write
    # the source state, then turn the gate on for the migration step.
    src_path = tmp_path / "v1.db"
    dst_path = tmp_path / "v2.db"
    src = MemoryStore(str(src_path))
    try:
        src.insert_belief(_belief(bid="0123456789abcdef", content="src belief"))
    finally:
        src.close()

    monkeypatch.setenv("AELFRICE_WRITE_LOG_AUTHORITATIVE", "1")
    report = migrate(
        legacy_path=src_path,
        target_path=dst_path,
        project_root=tmp_path,
        apply=True,
        copy_all=True,
    )
    assert report.counts.inserted_beliefs == 1
