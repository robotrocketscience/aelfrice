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


def _assert_constructing_branches_record(src: str) -> None:
    """Raise AssertionError if any index-constructing block lacks the recorder.

    Factored out of the AC2 guard so the guard itself can be tested against a
    known-bad source. A guard nothing proves is a guard that silently stops
    guarding — this one already had that failure: it scanned only `.body`, so
    a rebuild in an `else:` arm passed.
    """
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
    # Every attribute that can hold a statement list. `.body` alone is not
    # enough: an `else:` arm lives in `.orelse`, and a `finally:` arm in
    # `.finalbody`, so a rebuild added there would be invisible to the guard
    # while the CHANGELOG claims the guard covers it. (`except:` arms are
    # reached anyway — ast.walk yields each ExceptHandler, whose block is its
    # own `.body`.)
    block_attrs = ("body", "orelse", "finalbody")
    for node in ast.walk(tree):
        for attr in block_attrs:
            block = getattr(node, attr, None)
            if not isinstance(block, list):
                continue
            if not all(isinstance(s, ast.stmt) for s in block):
                continue  # e.g. IfExp.orelse is an expression, not a block
            for stmt in block:
                names = _calls(stmt)
                if constructing & set(names):
                    found_any = True
                    # the recorder must appear in this same block
                    block_names: list[str] = []
                    for sibling in block:
                        block_names.extend(_calls(sibling))
                    assert "_record_sidecar_outcome" in block_names, (
                        "an index-constructing branch in BM25IndexCache.get "
                        "has no _record_sidecar_outcome call in its "
                        f"{attr} block: {ast.dump(stmt)[:200]}"
                    )
    assert found_any, "found no index-constructing call — did get() move?"


def test_every_index_constructing_branch_records_an_outcome() -> None:
    """AC2: a test must fail if a branch is added without an outcome.

    Parses `BM25IndexCache.get` and requires that every statement
    constructing an index (`BM25Index.build`, `BM25Index.update_from`) is
    accompanied by a `_record_sidecar_outcome(...)` call in the same
    enclosing block. A behavioural test cannot cover this: a new branch
    nobody wrote a scenario for would simply never run.
    """
    _assert_constructing_branches_record(
        textwrap.dedent(inspect.getsource(BM25IndexCache.get))
    )


@pytest.mark.parametrize(
    ("arm", "expected_block"),
    [
        pytest.param(
            "    if cond:\n"
            "        self._index = BM25Index.build(rows)\n"
            "        _record_sidecar_outcome(SIDECAR_FULL_REBUILD)\n"
            "    else:\n"
            "        self._index = BM25Index.build(rows)\n",
            "orelse",
            id="if-else-arm",
        ),
        pytest.param(
            "    for _ in rows:\n"
            "        self._index = BM25Index.build(rows)\n"
            "        _record_sidecar_outcome(SIDECAR_FULL_REBUILD)\n"
            "    else:\n"
            "        self._index = BM25Index.build(rows)\n",
            "orelse",
            id="for-else-arm",
        ),
        pytest.param(
            "    try:\n"
            "        self._index = BM25Index.build(rows)\n"
            "        _record_sidecar_outcome(SIDECAR_FULL_REBUILD)\n"
            "    finally:\n"
            "        self._index = BM25Index.build(rows)\n",
            "finalbody",
            id="finally-arm",
        ),
    ],
)
def test_the_branch_guard_catches_a_rebuild_outside_dot_body(
    arm: str, expected_block: str
) -> None:
    """The guard must scan `orelse` and `finalbody`, not just `body`.

    Before #1407's review this was live: the walk used
    `getattr(node, "body", None)` only, so a bare `BM25Index.build(...)` in an
    `else:` arm left every test in this file green.

    Each source below puts a *compliant* rebuild in the `body` arm and a bare
    one in the arm under test. That shape matters: `_calls` recurses the whole
    subtree, so a function with no recorder anywhere fails at the outermost
    block whether or not `orelse` is scanned — a naive version of this test
    passes under the mutation it is meant to catch. Asserting on the block
    *name* is what makes it distinguishing: narrowing the walk back to
    `("body",)` turns these three green-by-accident into failures to raise.
    """
    src = "def get(self):\n" + arm
    with pytest.raises(AssertionError, match=f"{expected_block} block"):
        _assert_constructing_branches_record(src)


def test_the_outcome_vocabulary_is_closed() -> None:
    """The rate script and the audit reader both key on these exact three
    strings; adding a fourth silently makes older rows unclassifiable."""
    assert SIDECAR_OUTCOMES == {"fresh", "incremental", "full_rebuild"}


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        # The case that exists today: a cadence-driven rebuild, then the main
        # retrieval hits the now-warm cache. Under last-write-wins this fire
        # recorded `fresh` despite having just paid a full rebuild — and at
        # 74 ms the latency proxy missed it too.
        ([SIDECAR_FULL_REBUILD, SIDECAR_FRESH], SIDECAR_FULL_REBUILD),
        ([SIDECAR_INCREMENTAL, SIDECAR_FRESH], SIDECAR_INCREMENTAL),
        ([SIDECAR_FRESH, SIDECAR_FULL_REBUILD], SIDECAR_FULL_REBUILD),
        ([SIDECAR_INCREMENTAL, SIDECAR_FULL_REBUILD], SIDECAR_FULL_REBUILD),
        ([SIDECAR_FULL_REBUILD, SIDECAR_INCREMENTAL], SIDECAR_FULL_REBUILD),
        ([SIDECAR_FRESH, SIDECAR_FRESH], SIDECAR_FRESH),
    ],
)
def test_a_fire_is_classified_by_its_most_expensive_get(
    sequence: list[str], expected: str
) -> None:
    """Multiple `get()` calls in one fire must not let a later cheap one erase
    an earlier expensive one — the field answers "did this fire pay for a
    rebuild", and #1380 is priced on that answer."""
    reset_sidecar_outcome()
    for outcome in sequence:
        bm25._record_sidecar_outcome(outcome)
    assert last_sidecar_outcome() == expected


def test_the_recorder_rejects_a_value_outside_the_vocabulary() -> None:
    """`SIDECAR_OUTCOMES` must be enforced in production, not just asserted
    about in tests.

    Without this the frozenset is referenced only from this file — which is
    what CodeQL flagged as an unused global, and it was right in substance: an
    unrecognised state would have reached `hook_audit.jsonl` and been counted
    as a category no aggregator knows, skewing the rate the field exists to
    measure. No behavioural test can cover it, because no production branch
    passes a bad value; that is exactly why it needs a direct one.
    """
    with pytest.raises(ValueError, match="unknown sidecar outcome"):
        bm25._record_sidecar_outcome("stale")
    # and the snapshot is not corrupted by the rejected write
    reset_sidecar_outcome()
    with pytest.raises(ValueError):
        bm25._record_sidecar_outcome("")
    assert last_sidecar_outcome() is None


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
    """The audit row must never be the reason a hook breaks.

    Patched on `aelfrice.sidecar_outcome`, which is where the hook reads it
    from: `bm25` re-exports the name, so patching the re-export would leave
    the hook's own lookup untouched and the test would pass vacuously.
    """
    from aelfrice import hook, sidecar_outcome

    def boom() -> str | None:
        raise RuntimeError("sidecar outcome unavailable")

    monkeypatch.setattr(sidecar_outcome, "last_sidecar_outcome", boom)
    assert hook._last_sidecar_outcome() is None


# --- the edge between the two ends ----------------------------------------
#
# Everything above tests one end or the other: the recorder inside
# `BM25IndexCache.get`, and `_write_hook_audit_record` called directly with an
# explicit `sidecar_outcome=` kwarg. Neither observes that
# `user_prompt_submit` actually joins them, and that join is the whole
# product — the field exists to appear in production audit rows.
#
# Measured before adding these: deleting `sidecar_outcome=_last_sidecar_outcome(),`
# from the production call site left the full suite green at 7,611 passed, and
# so did replacing `reset_sidecar_outcome()` with a no-op.


def _drive_ups(tmp_path: Path, prompt: str, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fire the real UserPromptSubmit hook and return its audit row."""
    import io
    import json

    from aelfrice.hook import user_prompt_submit
    from aelfrice.store import MemoryStore

    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        for i in range(20):
            s.insert_belief(_mk(i))
    finally:
        s.close()

    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    user_prompt_submit(
        stdin=io.StringIO(
            json.dumps({"prompt": prompt, "session_id": "s1", "cwd": str(tmp_path)})
        ),
        stdout=io.StringIO(),
    )
    rows = [
        r
        for r in _audit_rows(tmp_path)
        if r.get("hook") == "user_prompt_submit"
    ]
    assert rows, "the hook wrote no user_prompt_submit audit row"
    return rows[-1]


def test_a_real_fire_carries_the_outcome_into_its_audit_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The edge: production must actually pass the recorder's value through.

    Asserted against the vocabulary rather than a literal, because which of
    the three a first fire lands on is an implementation detail of the cache;
    what must hold is that a fire which did index work records *something
    real* rather than the key going missing.

    Falsifiable by deleting `sidecar_outcome=_last_sidecar_outcome(),` from
    the `_write_hook_audit_record` call in `user_prompt_submit` — which the
    rest of this file does not catch.
    """
    row = _drive_ups(tmp_path, "belief number 3 about retrieval and indexing", monkeypatch)
    assert "sidecar_outcome" in row, "the production call site stopped passing the field"
    assert row["sidecar_outcome"] in SIDECAR_OUTCOMES


def test_the_reset_stops_a_fire_inheriting_the_previous_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-fire reset, observed through the hook rather than the helper.

    A stale process global is the defect class this repo has shipped twice
    (#1366 lane firings, #1444's `_LAST_LOCK_CONFLICTS`). Here it would make a
    fire that did no index work inherit the previous fire's outcome and count
    as a rebuild.

    Set the global to a value no real fire could have left, then drive a
    fire that runs retrieval but never reaches `BM25IndexCache.get()`. With
    the reset in place the key is absent; without it the planted value
    survives into the row and a no-work fire is counted as a rebuild.

    `AELFRICE_BM25F=0` is what makes the case reachable: `get()` is called
    from `_l1_hits` under `if use_bm25f_anchors:`, so with the lane off
    retrieval still runs and still writes an audit row, but no index work
    happens. A *gate-skipped* prompt does not exercise this — that path
    never reaches the reset either, and its audit write never passes the
    field at all, so it passes with or without the reset.

    Falsifiable by replacing `reset_sidecar_outcome()` with a no-op.
    """
    from aelfrice import bm25 as _bm25

    monkeypatch.setenv("AELFRICE_BM25F", "0")
    _bm25._record_sidecar_outcome(SIDECAR_FULL_REBUILD)
    assert last_sidecar_outcome() == SIDECAR_FULL_REBUILD

    row = _drive_ups(tmp_path, "belief number 3 about retrieval and indexing", monkeypatch)
    assert not row.get("prompt_shape_gate_skip"), (
        "fixture must run retrieval, or it does not exercise the reset"
    )
    assert "sidecar_outcome" not in row, (
        "a fire that did no index work inherited the previous fire's outcome"
    )


def test_the_cadence_pass_rebuild_survives_the_per_fire_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reset must run ABOVE the cadence dispatch, not below it.

    `_maybe_run_ups_cadence_checkpoint` reaches BM25 --
    `_run_cadence_rebuild` -> `_rebuild_and_format` -> `rebuild_v14` ->
    `retrieve()` -> the L1 lane -> `BM25IndexCache.get()`. So a fire landing
    on a cadence boundary with a stale sidecar pays a full rebuild inside
    that pass. If the per-fire reset sits *after* the dispatch, that outcome
    is erased and the main retrieval then records `fresh` against a sidecar
    the cadence pass just warmed -- undercounting exactly the expensive fires
    #1380 is priced on, and making max-wins inert for its one justifying
    interleaving.

    **The warm-up fire is what gives this test its power, and the first
    version of it was worthless without one.** `_drive_ups` builds a fresh
    store, so an unwarmed fire records `full_rebuild` from its own main
    retrieval -- and the assertion passed identically whether the reset ran
    above or below the dispatch. Warming first makes the main retrieval
    `fresh`, so `full_rebuild` can only have come from the cadence pass.

    Stubbing the cadence pass is what makes this deterministic: cadence is
    default-off, and reaching a real boundary with a stale sidecar is not
    something a unit test can arrange. The stub stands in for "the cadence
    pass did index work", which is the only property under test.

    Falsifiable by moving `reset_sidecar_outcome()` back below the
    `_maybe_run_ups_cadence_checkpoint` call, which is where it shipped.
    """
    from aelfrice import hook as hook_mod
    from aelfrice.bm25 import _record_sidecar_outcome

    import io
    import json

    from aelfrice.hook import user_prompt_submit

    prompt = "belief number 3 about retrieval and indexing"

    def _fire_again() -> dict:
        """Fire the hook again against the store `_drive_ups` already seeded.

        `_drive_ups` re-inserts its beliefs on every call, which trips the
        `beliefs.content_hash` UNIQUE constraint the second time, so repeat
        fires cannot go through it.
        """
        user_prompt_submit(
            stdin=io.StringIO(
                json.dumps({
                    "prompt": prompt,
                    "session_id": "s1",
                    "cwd": str(tmp_path),
                })
            ),
            stdout=io.StringIO(),
        )
        rows = [
            r for r in _audit_rows(tmp_path)
            if r.get("hook") == "user_prompt_submit"
        ]
        assert rows, "the hook wrote no user_prompt_submit audit row"
        return rows[-1]

    # Warm-up fire: builds the index, so the NEXT fire's main retrieval is
    # not itself a full rebuild. Without this the assertion below cannot
    # tell the cadence pass's outcome from the main retrieval's.
    warm = _drive_ups(tmp_path, prompt, monkeypatch)
    assert warm["sidecar_outcome"] == "full_rebuild", (
        "expected the first fire on a fresh store to build the index"
    )

    # Control: with the index warm and no cadence pass, the fire is `fresh`.
    control = _fire_again()
    assert control["sidecar_outcome"] == "fresh", (
        f"expected a warm fire to record 'fresh', got "
        f"{control['sidecar_outcome']!r}; the control for this test does not "
        "hold and the assertion below would prove nothing"
    )

    def _cadence_that_rebuilt(*_a: object, **_k: object) -> None:
        # Stand in for the cadence rebuild's get() taking the build branch.
        _record_sidecar_outcome("full_rebuild")
        return None

    monkeypatch.setattr(
        hook_mod, "_maybe_run_ups_cadence_checkpoint", _cadence_that_rebuilt,
    )

    row = _fire_again()

    assert row.get("sidecar_outcome") == "full_rebuild", (
        f"the fire recorded {row.get('sidecar_outcome')!r}, not 'full_rebuild'. "
        "The cadence pass paid a full rebuild and the fire must be classified "
        "by its costliest get(); a reset below the cadence dispatch erases it. "
        "The control above proves this fire would otherwise be 'fresh'."
    )
