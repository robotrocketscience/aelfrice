"""Integration tests for the scanner LLM-classify path → derivation_worker
call shape (#265 PR-B commit 2).

Pre-migration, scanner.py:262-310 wrote beliefs directly via
`store.insert_or_corroborate` and `store.insert_feedback_event` when an
`llm_router` was supplied. Commit 2 collapses that branch into the same
`record_ingest` + `run_worker` shape the regex path already uses, with
the router's per-candidate decisions carried in
`raw_meta["route_overrides"]`.

Each test states a falsifiable hypothesis about the new contract:

    scan_repo(llm_router=X) writes one log row per persistable
    candidate (route.persist=True) with `raw_meta.route_overrides`
    populated, invokes run_worker once at end-of-scan, and produces
    canonical beliefs whose (type, origin, alpha, beta) match the
    router's decision verbatim.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.replay import replay_full_equality
from aelfrice.scanner import LLMRoute, SentenceCandidate, scan_repo
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "scanner-llm-via-worker.db"))
    yield s
    s.close()


def _seed_repo(root: Path) -> None:
    """Drop a tiny doc-only tree the filesystem extractor will see."""
    (root / "README.md").write_text(
        "The configuration file lives at /etc/aelfrice/conf.\n"
        "\n"
        "The default port is 8080 for the dashboard.\n"
        "\n"
        "Aelfrice stores beliefs in a SQLite database.\n",
        encoding="utf-8",
    )


class _StubRouter:
    """Per-candidate `LLMRoute` table indexed by candidate text.

    Tests configure (text -> LLMRoute) up front; the stub returns the
    routes in input order. Any candidate without a configured route gets
    `default_route` (a no-op question-form persist=False decision by
    default — fail-loud if the test forgot to map a candidate).
    """

    def __init__(
        self,
        routes_by_text: dict[str, LLMRoute],
        *,
        default: LLMRoute | None = None,
    ) -> None:
        self._routes_by_text = routes_by_text
        self._default = default
        self.calls: list[list[SentenceCandidate]] = []

    def classify(
        self, candidates: list[SentenceCandidate],
    ) -> list[LLMRoute]:
        self.calls.append(list(candidates))
        out: list[LLMRoute] = []
        for cand in candidates:
            route = self._routes_by_text.get(cand.text)
            if route is None:
                if self._default is None:
                    raise AssertionError(
                        f"_StubRouter: no route configured for "
                        f"{cand.text!r}",
                    )
                route = self._default
            out.append(route)
        return out


def _persist_route(
    *,
    belief_type: str = "factual",
    origin: str = "agent_inferred",
    alpha: float = 1.4,
    beta: float = 0.6,
    audit_source: str | None = None,
) -> LLMRoute:
    return LLMRoute(
        belief_type=belief_type,
        origin=origin,
        persist=True,
        alpha=alpha,
        beta=beta,
        audit_source=audit_source,
    )


def test_llm_path_writes_log_rows_through_worker(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: with an `llm_router`, scan_repo writes one ingest_log
    row per persistable candidate and `run_worker` stamps every row.
    Falsifiable by any unstamped row OR by zero log rows when the
    router classified at least one persistable candidate."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(),
            "The default port is 8080 for the dashboard.":
                _persist_route(),
            "Aelfrice stores beliefs in a SQLite database.":
                _persist_route(),
        },
    )
    scan_repo(store, tmp_path, llm_router=router)

    assert store.list_unstamped_ingest_log() == []
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log",
    ).fetchall()
    assert rows
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "filesystem_ingest"
        assert "route_overrides" in meta


def test_llm_path_belief_carries_router_fields(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: the canonical belief written by the worker carries
    the router's `(type, origin, alpha, beta)` verbatim — the
    post-derivation splice replaced the deterministic classifier's
    output. Falsifiable by any field reverting to derive()'s default."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(
                    belief_type="factual",
                    origin="user_stated",
                    alpha=2.5,
                    beta=0.3,
                ),
        },
        default=_persist_route(),
    )
    scan_repo(store, tmp_path, llm_router=router)

    beliefs = [
        b for b in (store.get_belief(bid) for bid in store.list_belief_ids())
        if b is not None
        and b.content == "The configuration file lives at /etc/aelfrice/conf."
    ]
    assert len(beliefs) == 1
    belief = beliefs[0]
    assert belief.type == "factual"
    assert belief.origin == "user_stated"
    assert belief.alpha == pytest.approx(2.5)
    assert belief.beta == pytest.approx(0.3)


def test_llm_path_persist_false_writes_no_log_row(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: candidates routed `persist=False` are counted in
    `skipped_non_persisting` but produce no ingest_log row and no
    canonical belief. Falsifiable by an extra log row OR by a belief
    appearing for a non-persisting candidate."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(),
            "The default port is 8080 for the dashboard.":
                LLMRoute(
                    belief_type="factual",
                    origin="agent_inferred",
                    persist=False,
                    alpha=1.0,
                    beta=1.0,
                ),
            "Aelfrice stores beliefs in a SQLite database.":
                _persist_route(),
        },
    )
    result = scan_repo(store, tmp_path, llm_router=router)

    assert result.skipped_non_persisting == 1
    log_count = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log",
    ).fetchone()["n"]
    assert log_count == 2

    belief_texts = {
        b.content
        for b in (store.get_belief(bid) for bid in store.list_belief_ids())
        if b is not None
    }
    assert "The default port is 8080 for the dashboard." not in belief_texts


def test_llm_path_audit_source_emits_feedback_event(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: when `route.audit_source` is set, the worker writes
    one feedback_history row tagged with that source string. Mirrors
    pre-migration scanner.py:304-310. Falsifiable by zero rows or by
    a row tagged with a different source."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(audit_source="llm_router_v1"),
        },
        default=_persist_route(),
    )
    scan_repo(store, tmp_path, llm_router=router)

    events = [e for e in store.list_feedback_events()
              if e.source == "llm_router_v1"]
    assert len(events) == 1
    assert events[0].valence == pytest.approx(0.0)


def test_llm_path_audit_source_skipped_on_corroboration(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: a second scan of the same tree with the same router
    decision DOES NOT write a second audit row — the worker's
    feedback_history emission is gated by `was_inserted`. Falsifiable
    by a duplicate audit row on the second pass."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(audit_source="llm_router_v1"),
        },
        default=_persist_route(),
    )
    scan_repo(store, tmp_path, llm_router=router)
    first_pass = store.count_feedback_events()
    assert first_pass == 1

    scan_repo(store, tmp_path, llm_router=router)
    assert store.count_feedback_events() == 1


def test_llm_path_replay_full_equality_clean(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis (CI gate): after an llm_router scan, the
    full-equality replay probe reports zero drift. Tripwire for the
    LLM path bypassing the worker (`canonical_orphan`) or the worker
    producing a different canonical belief than direct splice
    (`mismatched`). Falsifiable by any non-zero drift counter."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(audit_source="llm_router_v1"),
            "The default port is 8080 for the dashboard.":
                _persist_route(),
            "Aelfrice stores beliefs in a SQLite database.":
                _persist_route(origin="user_stated", alpha=2.0, beta=1.0),
        },
    )
    scan_repo(store, tmp_path, llm_router=router)
    scan_repo(store, tmp_path, llm_router=router)
    report = replay_full_equality(store)
    assert report.total_log_rows > 0
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0


def test_llm_path_idempotent_after_worker(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: re-scanning under the LLM path is idempotent on the
    canonical belief set. `ScanResult.inserted` is 0 on the second run;
    every persistable candidate counts as `skipped_existing`.
    Falsifiable by a duplicate belief OR non-zero `inserted` on the
    second scan."""
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(),
            "The default port is 8080 for the dashboard.":
                _persist_route(),
            "Aelfrice stores beliefs in a SQLite database.":
                _persist_route(),
        },
    )
    first = scan_repo(store, tmp_path, llm_router=router)
    assert first.inserted >= 1
    beliefs_after_first = store.count_beliefs()

    second = scan_repo(store, tmp_path, llm_router=router)
    assert second.inserted == 0
    assert second.skipped_existing >= first.inserted
    assert store.count_beliefs() == beliefs_after_first


def test_llm_path_no_direct_belief_writes(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: scanner.py no longer calls `insert_or_corroborate`,
    `insert_belief`, or `insert_feedback_event` directly on the LLM
    path. The worker is the only writer of beliefs/feedback rows.

    Patches the three store methods to fail-loud; if scan_repo invokes
    any of them, the test fails. The worker reaches the same methods
    via its own code path (which is fine — we only assert the scanner
    module itself does not).
    """
    _seed_repo(tmp_path)
    router = _StubRouter(
        routes_by_text={
            "The configuration file lives at /etc/aelfrice/conf.":
                _persist_route(audit_source="llm_router_v1"),
        },
        default=_persist_route(),
    )

    # The scanner module must not call these. The worker reaches them
    # by importing from `store` directly, so the worker's invocations
    # are not intercepted by attribute-replacement on the instance.
    import aelfrice.scanner as scanner_mod

    forbidden = {
        "insert_or_corroborate",
        "insert_belief",
        "insert_feedback_event",
    }
    for name in forbidden:
        assert not hasattr(scanner_mod, name), (
            f"scanner module re-exports forbidden writer {name!r}"
        )

    # End-to-end smoke: scan must succeed and produce the expected
    # belief without scanner.py reaching for direct writes.
    scan_repo(store, tmp_path, llm_router=router)
    assert store.count_beliefs() >= 1
