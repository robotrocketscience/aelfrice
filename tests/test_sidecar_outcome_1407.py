"""#1407: record which BM25 sidecar outcome each hook fire took.

#1380's cost case is `cold_cost x cold_rate`. `cold_cost` is measured; the only
prior estimate of `cold_rate` was a latency proxy that cannot tell a rebuild
from lock contention or a cold page cache. This records the branch directly.
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path

import pytest

from aelfrice import bm25
from aelfrice.bm25 import (
    SIDECAR_FRESH,
    SIDECAR_FULL_REBUILD,
    SIDECAR_INCREMENTAL,
    SIDECAR_OUTCOMES,
    BM25IndexCache,
    last_sidecar_outcome,
    reset_sidecar_outcome,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk(i: int) -> Belief:
    return Belief(
        id=f"b{i}",
        content=f"belief number {i} about retrieval and indexing",
        content_hash=f"h{i}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture
def file_store(tmp_path: Path):
    """A file-backed store. `:memory:` has no sidecar path at all
    (`sidecar_path_for` returns None), so the incremental branch is
    unreachable on one and a test using it would silently only ever see
    `full_rebuild`."""
    s = MemoryStore(os.path.join(str(tmp_path), "m.db"))
    for i in range(20):
        s.insert_belief(_mk(i))
    yield s
    s.close()


# --- the three outcomes, end to end --------------------------------------


def test_absent_before_any_get_and_distinguishable_from_fresh() -> None:
    """A fire that never builds an index records nothing. `None` must not
    read as `fresh`, or a no-op fire counts as a cache hit and inflates the
    very rate this exists to measure."""
    reset_sidecar_outcome()
    assert last_sidecar_outcome() is None
    assert None not in SIDECAR_OUTCOMES


def test_first_build_is_a_full_rebuild(file_store: MemoryStore) -> None:
    cache = BM25IndexCache(store=file_store)
    reset_sidecar_outcome()
    cache.get()
    assert last_sidecar_outcome() == SIDECAR_FULL_REBUILD


def test_second_get_is_fresh(file_store: MemoryStore) -> None:
    cache = BM25IndexCache(store=file_store)
    cache.get()
    reset_sidecar_outcome()
    cache.get()
    assert last_sidecar_outcome() == SIDECAR_FRESH


def test_a_mutation_takes_the_incremental_path(file_store: MemoryStore) -> None:
    """The state that makes this three-valued rather than a boolean.

    Since #1199 a stale sidecar no longer implies a full rebuild, and
    collapsing this into `full_rebuild` is what made #1199's 86.2% and the
    8.5% latency proxy look contradictory when they measured different
    events.
    """
    cache = BM25IndexCache(store=file_store)
    cache.get()
    file_store.insert_belief(_mk(99))
    reset_sidecar_outcome()
    cache.get()
    assert last_sidecar_outcome() == SIDECAR_INCREMENTAL


def test_a_fresh_cache_over_a_current_sidecar_is_fresh(
    file_store: MemoryStore,
) -> None:
    """The cross-process case: the sidecar on disk still matches the
    generation, so a new cache loads it without building."""
    BM25IndexCache(store=file_store).get()
    reset_sidecar_outcome()
    BM25IndexCache(store=file_store).get()
    assert last_sidecar_outcome() == SIDECAR_FRESH


# --- the branch guard ----------------------------------------------------


def test_every_index_constructing_branch_records_an_outcome() -> None:
    """AC2: a test must fail if a branch is added without an outcome.

    Parses `BM25IndexCache.get` and requires that every statement
    constructing an index (`BM25Index.build`, `BM25Index.update_from`) is
    accompanied by a `_record_sidecar_outcome(...)` call in the same
    enclosing block. A behavioural test cannot cover this: a new branch
    nobody wrote a scenario for would simply never run.
    """
    src = textwrap.dedent(inspect.getsource(BM25IndexCache.get))
    tree = ast.parse(src)

    def _calls(node: ast.AST) -> list[str]:
        out: list[str] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute):
                    out.append(f.attr)
                elif isinstance(f, ast.Name):
                    out.append(f.id)
        return out

    constructing = {"build", "update_from"}
    found_any = False
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:
            names = _calls(stmt)
            if constructing & set(names):
                found_any = True
                # the recorder must appear in this same block
                block_names: list[str] = []
                for sibling in body:
                    block_names.extend(_calls(sibling))
                assert "_record_sidecar_outcome" in block_names, (
                    "an index-constructing branch in BM25IndexCache.get has no "
                    f"_record_sidecar_outcome call in its block: {ast.dump(stmt)[:200]}"
                )
    assert found_any, "found no index-constructing call — did get() move?"


def test_the_outcome_vocabulary_is_closed() -> None:
    """The rate script and the audit reader both key on these exact three
    strings; adding a fourth silently makes older rows unclassifiable."""
    assert SIDECAR_OUTCOMES == {"fresh", "incremental", "full_rebuild"}


def test_recorded_values_are_always_in_the_vocabulary(
    file_store: MemoryStore,
) -> None:
    cache = BM25IndexCache(store=file_store)
    seen = set()
    for _ in range(2):
        reset_sidecar_outcome()
        cache.get()
        seen.add(last_sidecar_outcome())
        file_store.insert_belief(_mk(len(seen) + 100))
    assert seen <= SIDECAR_OUTCOMES
    assert seen


# --- the audit row -------------------------------------------------------


def _audit_rows(tmp_path: Path) -> list[dict[str, object]]:
    import json

    out: list[dict[str, object]] = []
    for p in tmp_path.rglob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_row(
    tmp_path: Path, outcome: str | None, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    """Drive the live audit writer so the record schema stays in sync with
    production, rather than asserting against a hand-built dict."""
    from aelfrice import hook

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    hook._write_hook_audit_record(
        hook="user_prompt_submit",
        prompt="p",
        rendered_block="b",
        n_beliefs=0,
        n_locked=0,
        sidecar_outcome=outcome,
    )
    return _audit_rows(tmp_path)


def test_audit_row_omits_the_key_when_no_index_work_happened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: absence is a distinct state and must be an *absent key*, not a
    default value — a row missing the key must not be counted as `fresh`."""
    rows = _write_row(tmp_path, None, monkeypatch)
    assert len(rows) == 1
    assert "sidecar_outcome" not in rows[0]


def test_audit_row_carries_the_outcome_when_there_was_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _write_row(tmp_path, SIDECAR_FULL_REBUILD, monkeypatch)
    assert len(rows) == 1
    assert rows[0]["sidecar_outcome"] == "full_rebuild"


def test_the_hook_helper_reads_the_live_snapshot(
    file_store: MemoryStore,
) -> None:
    from aelfrice import hook

    cache = BM25IndexCache(store=file_store)
    reset_sidecar_outcome()
    cache.get()
    assert hook._last_sidecar_outcome() == last_sidecar_outcome()
    assert hook._last_sidecar_outcome() in SIDECAR_OUTCOMES


def test_the_hook_helper_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit row must never be the reason a hook breaks."""
    from aelfrice import hook

    def boom() -> str | None:
        raise RuntimeError("bm25 unavailable")

    monkeypatch.setattr(bm25, "last_sidecar_outcome", boom)
    assert hook._last_sidecar_outcome() is None
