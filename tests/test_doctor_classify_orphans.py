"""Acceptance tests for aelf doctor --classify-orphans (issue #206).

All tests mock the Anthropic client via the _call_anthropic seam used
by the existing onboard test suite.  No real network calls.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import aelfrice.llm_classifier as llm
import aelfrice.cli as cli_module
from aelfrice.doctor import (
    classify_orphans,
    format_orphan_report,
    OrphanRunReport,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    belief_type: str = "unknown",
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
) -> Belief:
    """Insert a belief and return it."""
    import hashlib

    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=alpha,
        beta=beta,
        type=belief_type,
        lock_level=lock_level,
        locked_at=None,
        created_at="2025-01-01T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_UNKNOWN,
    )
    store.insert_belief(b)
    return b


def _all_beliefs(store: MemoryStore) -> list[Belief]:
    cur = store._conn.execute("SELECT * FROM beliefs")  # type: ignore[attr-defined]
    out: list[Belief] = []
    for row in cur.fetchall():
        out.append(
            Belief(
                id=row["id"],
                content=row["content"],
                content_hash=row["content_hash"],
                alpha=row["alpha"],
                beta=row["beta"],
                type=row["type"],
                lock_level=row["lock_level"],
                locked_at=row["locked_at"],
                created_at=row["created_at"],
                last_retrieved_at=row["last_retrieved_at"],
                origin=row["origin"] if "origin" in row.keys() else "unknown",
            )
        )
    return out


def _fake_call_anthropic_factory(
    belief_types: list[str],
) -> Any:
    """Return a _call_anthropic stub that classifies in the given order.

    belief_types[i] is the type returned for candidate i. Every candidate
    gets persist=True and origin='agent_inferred'.
    """

    def _fake(**kwargs: Any) -> llm.ClientResponse:
        user_msg = kwargs.get("user_message", "[]")
        candidates: list[Any] = json.loads(user_msg)
        results = []
        for item in candidates:
            idx = item["index"]
            btype = belief_types[idx] if idx < len(belief_types) else BELIEF_FACTUAL
            results.append(
                {
                    "index": idx,
                    "belief_type": btype,
                    "origin": "agent_inferred",
                    "persist": True,
                }
            )
        return llm.ClientResponse(
            text=json.dumps(results),
            input_tokens=10 * len(candidates),
            output_tokens=5 * len(candidates),
        )

    return _fake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memdb() -> Iterator[MemoryStore]:
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def stub_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make _anthropic_importable return True (no real SDK needed)."""
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _check: True)


# ---------------------------------------------------------------------------
# Unit tests: classify_orphans() core function
# ---------------------------------------------------------------------------


def test_orphan_finder_picks_unknown_low_prior(memdb: MemoryStore) -> None:
    """find_orphan_beliefs selects type='unknown' AND alpha+beta < 2."""
    _make_belief(memdb, belief_id="o1", content="orphan one", belief_type="unknown",
                 alpha=1.0, beta=1.0)
    # Already typed — not an orphan
    _make_belief(memdb, belief_id="t1", content="typed one", belief_type="factual",
                 alpha=1.0, beta=1.0)
    # Has feedback signal (alpha+beta > 2) — not an orphan
    _make_belief(memdb, belief_id="o2", content="orphan but feedback",
                 belief_type="unknown", alpha=1.5, beta=1.0)

    orphans = memdb.find_orphan_beliefs()
    assert len(orphans) == 1
    assert orphans[0].id == "o1"


def test_orphan_finder_respects_max_n(memdb: MemoryStore) -> None:
    for i in range(5):
        _make_belief(memdb, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown", alpha=1.0, beta=1.0)
    assert len(memdb.find_orphan_beliefs(max_n=3)) == 3
    assert len(memdb.find_orphan_beliefs()) == 5


def test_classify_orphans_dry_run_no_writes(
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run=True must not make network calls or change the store."""
    for i in range(4):
        _make_belief(memdb, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")

    call_count = {"n": 0}

    def tripwire(**kwargs: Any) -> llm.ClientResponse:
        call_count["n"] += 1
        raise AssertionError("classify_orphans dry-run reached _call_anthropic")

    monkeypatch.setattr(llm, "_call_anthropic", tripwire)

    report = classify_orphans(
        memdb,
        api_key="dummy",
        model=llm.DEFAULT_MODEL,
        max_tokens=0,
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.orphans_found == 4
    assert report.classified == 0
    assert call_count["n"] == 0
    # Store unchanged
    remaining_orphans = memdb.find_orphan_beliefs()
    assert len(remaining_orphans) == 4


def test_classify_orphans_updates_types_in_store(
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """classify_orphans writes the new type back to the store."""
    contents = [
        ("o0", "CI must always be green", BELIEF_REQUIREMENT),
        ("o1", "prefer composition over inheritance", BELIEF_PREFERENCE),
        ("o2", "the scheduler runs at midnight", BELIEF_FACTUAL),
    ]
    for bid, content, _ in contents:
        _make_belief(memdb, belief_id=bid, content=content, belief_type="unknown")

    expected_types = [btype for _, _, btype in contents]
    monkeypatch.setattr(
        llm, "_call_anthropic", _fake_call_anthropic_factory(expected_types)
    )

    report = classify_orphans(
        memdb,
        api_key="key",
        model=llm.DEFAULT_MODEL,
        max_tokens=0,
    )

    assert report.orphans_found == 3
    assert report.classified == 3
    assert report.skipped == 0

    beliefs = {b.id: b for b in _all_beliefs(memdb)}
    for bid, _, expected_type in contents:
        assert beliefs[bid].type == expected_type
        assert beliefs[bid].origin == ORIGIN_AGENT_INFERRED


def test_classify_orphans_95_percent_recovery() -> None:
    """Synthetic corpus: N=100 orphans; mock classifier returns valid type
    for all; assert >= 95 classified (acceptance criterion from issue #206)."""
    store = MemoryStore(":memory:")
    try:
        N = 100
        for i in range(N):
            _make_belief(
                store,
                belief_id=f"orphan{i:03d}",
                content=f"belief content {i}",
                belief_type="unknown",
            )
        # Also add 20 non-orphans that must not be touched.
        for i in range(20):
            _make_belief(
                store,
                belief_id=f"typed{i:03d}",
                content=f"typed content {i}",
                belief_type=BELIEF_FACTUAL,
                alpha=2.0,
                beta=1.0,
            )

        # Mock: classify all as 'factual'.
        all_factual = [BELIEF_FACTUAL] * N

        import unittest.mock as mock

        with mock.patch.object(
            llm, "_call_anthropic", side_effect=_fake_call_anthropic_factory(all_factual)
        ):
            report = classify_orphans(
                store,
                api_key="key",
                model=llm.DEFAULT_MODEL,
                max_tokens=0,
            )

        assert report.orphans_found == N
        recovery_rate = report.classified / N
        assert recovery_rate >= 0.95, (
            f"Expected >= 95% recovery, got {recovery_rate:.1%} "
            f"({report.classified}/{N} orphans classified)"
        )

        # Non-orphans must be untouched.
        beliefs = {b.id: b for b in _all_beliefs(store)}
        for i in range(20):
            bid = f"typed{i:03d}"
            assert beliefs[bid].type == BELIEF_FACTUAL
            assert beliefs[bid].alpha == 2.0
    finally:
        store.close()


def test_classify_orphans_no_orphans_is_noop(
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no orphans exist the function exits cleanly with zero classified."""
    _make_belief(memdb, belief_id="t1", content="typed", belief_type="factual",
                 alpha=2.0, beta=1.0)

    call_count = {"n": 0}

    def tripwire(**kw: Any) -> Any:
        call_count["n"] += 1
        raise AssertionError("should not reach LLM with no orphans")

    monkeypatch.setattr(llm, "_call_anthropic", tripwire)

    report = classify_orphans(
        memdb, api_key="key", model=llm.DEFAULT_MODEL, max_tokens=0,
    )
    assert report.orphans_found == 0
    assert report.classified == 0
    assert call_count["n"] == 0


def test_classify_orphans_max_n_caps_llm_input(
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--max N caps how many orphans are sent to the LLM."""
    for i in range(10):
        _make_belief(memdb, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")

    sent_count: list[int] = []

    def counting_fake(**kw: Any) -> llm.ClientResponse:
        candidates = json.loads(kw["user_message"])
        sent_count.append(len(candidates))
        results = [
            {"index": c["index"], "belief_type": "factual",
             "origin": "agent_inferred", "persist": True}
            for c in candidates
        ]
        return llm.ClientResponse(text=json.dumps(results),
                                  input_tokens=10, output_tokens=5)

    monkeypatch.setattr(llm, "_call_anthropic", counting_fake)

    report = classify_orphans(
        memdb, api_key="key", model=llm.DEFAULT_MODEL, max_tokens=0, max_n=4,
    )
    assert report.orphans_found == 10
    assert report.classified == 4
    assert sum(sent_count) == 4


def test_format_orphan_report_dry_run(memdb: MemoryStore) -> None:
    """format_orphan_report includes dry-run marker and no 'after' section."""
    for i in range(3):
        _make_belief(memdb, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")
    report = classify_orphans(
        memdb, api_key="", model=llm.DEFAULT_MODEL, max_tokens=0, dry_run=True,
    )
    text = format_orphan_report(report)
    assert "dry-run" in text
    assert "3 orphan(s) found" in text
    assert "after" not in text


def test_format_orphan_report_live_run_includes_cost(
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """format_orphan_report includes token counts and cost estimate."""
    _make_belief(memdb, belief_id="o1", content="orphan one", belief_type="unknown")

    monkeypatch.setattr(
        llm, "_call_anthropic", _fake_call_anthropic_factory([BELIEF_FACTUAL])
    )

    report = classify_orphans(
        memdb, api_key="key", model=llm.DEFAULT_MODEL, max_tokens=0,
    )
    text = format_orphan_report(report)
    assert "tokens:" in text
    assert "estimated cost:" in text
    assert "after" in text


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def memdb_cli(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    """In-memory store wired into the CLI _open_store."""
    store = MemoryStore(":memory:")
    real_close = store.close
    store.close = lambda: None  # type: ignore[method-assign]

    def _open() -> MemoryStore:
        return store

    monkeypatch.setattr(cli_module, "_open_store", _open)
    yield store
    real_close()


def test_cli_classify_orphans_dry_run(
    memdb_cli: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_anthropic: None,
) -> None:
    """aelf doctor --classify-orphans --dry-run prints report, exits 0."""
    for i in range(5):
        _make_belief(memdb_cli, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm, "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(  # type: ignore[arg-type]
            AssertionError("dry-run must not reach LLM")
        ),
    )

    out = io.StringIO()
    rc = cli_module.main(
        ["doctor", "--classify-orphans", "--dry-run"], out=out
    )
    assert rc == 0
    text = out.getvalue()
    assert "5 orphan(s) found" in text
    assert "dry-run" in text


def test_cli_classify_orphans_live(
    memdb_cli: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_anthropic: None,
) -> None:
    """aelf doctor --classify-orphans classifies and updates the store."""
    for i in range(3):
        _make_belief(memdb_cli, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm, "_call_anthropic",
        _fake_call_anthropic_factory([BELIEF_FACTUAL, BELIEF_PREFERENCE, BELIEF_REQUIREMENT]),
    )

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)
    assert rc == 0
    text = out.getvalue()
    assert "classified: 3" in text

    # Verify store was updated
    beliefs = {b.id: b for b in _all_beliefs(memdb_cli)}
    assert beliefs["o0"].type == BELIEF_FACTUAL
    assert beliefs["o1"].type == BELIEF_PREFERENCE
    assert beliefs["o2"].type == BELIEF_REQUIREMENT


def test_cli_classify_orphans_no_api_key(
    memdb_cli: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_anthropic: None,
) -> None:
    """Without ANTHROPIC_API_KEY and no --dry-run, exits 1 with hint."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _make_belief(memdb_cli, belief_id="o1", content="orphan",
                 belief_type="unknown")

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)
    assert rc == 1


def test_cli_classify_orphans_max_n(
    memdb_cli: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_anthropic: None,
) -> None:
    """--max N limits how many beliefs are sent to the LLM."""
    for i in range(10):
        _make_belief(memdb_cli, belief_id=f"o{i}", content=f"orphan {i}",
                     belief_type="unknown")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # All factual for simplicity
    monkeypatch.setattr(
        llm, "_call_anthropic",
        _fake_call_anthropic_factory([BELIEF_FACTUAL] * 10),
    )

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans", "--max", "3"], out=out)
    assert rc == 0
    text = out.getvalue()
    assert "10 orphan(s) found" in text
    assert "classified: 3" in text


def test_count_beliefs_by_type(memdb: MemoryStore) -> None:
    """count_beliefs_by_type returns correct distribution dict."""
    _make_belief(memdb, belief_id="f1", content="factual 1", belief_type="factual")
    _make_belief(memdb, belief_id="f2", content="factual 2", belief_type="factual")
    _make_belief(memdb, belief_id="p1", content="pref 1", belief_type="preference")
    _make_belief(memdb, belief_id="u1", content="unk 1", belief_type="unknown")

    dist = memdb.count_beliefs_by_type()
    assert dist["factual"] == 2
    assert dist["preference"] == 1
    assert dist["unknown"] == 1
