"""The rebuild log records the query `retrieve()` was handed (#1405).

`input.extracted_query` is `_query_for_recent_turns(...)`, and **neither**
production path scores that string:

* `rebuild_v14` scores `transform_query(raw_query, ...)`
  (`context_rebuilder.py:388-395`), while the log recomputed the raw form at
  `:1184`;
* `user_prompt_submit` scores a conversation-aware composition built in
  `hook.py` — the prompt repeated `conversation_aware_prompt_weight` times plus
  the recent turn window, default **on** — which `context_rebuilder` never sees.

So every replay of the recorded query measured a population production does not
issue. 97.7% of rows on the development store came from the UPS path, so the
dominant case was the one the log described least.

The load-bearing property is that the recorded value is **captured, not
re-derived**. A test that recomputes the expected query the same way the code
under test does would pass while the two drifted — which is exactly how this
defect survived. So the record-level assertions use a sentinel string
`_query_for_recent_turns` could not produce, and the call-site assertions read
which *expression* each caller hands over rather than re-deriving its value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aelfrice.context_rebuilder import (
    _build_rebuild_log_record,
    record_user_prompt_submit_log,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-08-06T00:00:00Z",
        last_retrieved_at=None,
    )


def _call_block(source: str, opener: str) -> str:
    """The full text of the first `opener(...)` call, parens balanced.

    Slicing to the next `)` is not enough: these call sites carry comments
    that themselves contain parentheses, and a naive cut lands inside one —
    which silently shortens the region and makes the assertion vacuous.
    """
    start = source.index(opener)
    depth = 0
    for i in range(start + len(opener) - 1, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced call block for {opener!r}")


def _row(path: Path) -> dict[str, Any]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected one row, got {len(lines)}"
    return json.loads(lines[0])


def test_the_recorded_query_is_the_one_passed_in_not_recomputed() -> None:
    """The record carries the caller's string verbatim.

    Uses a value `_query_for_recent_turns` could not produce from the turns,
    so a recomputing implementation cannot accidentally pass.
    """
    record = _build_rebuild_log_record(
        recent_turns=[],
        session_id="s1",
        candidates=[],
        pack_summary={},
        scored_query="ZZZ sentinel not derivable from the turns ZZZ",
    )
    payload = record["input"]
    assert isinstance(payload, dict)
    assert payload["scored_query"] == "ZZZ sentinel not derivable from the turns ZZZ"


def test_extracted_query_is_retained_alongside_it() -> None:
    """Both, not one replacing the other.

    `extracted_query` is what makes the difference between the two strings
    auditable; dropping it would hide the very gap this issue documents.
    """
    record = _build_rebuild_log_record(
        recent_turns=[],
        session_id="s1",
        candidates=[],
        pack_summary={},
        scored_query="scored",
    )
    payload = record["input"]
    assert isinstance(payload, dict)
    assert "extracted_query" in payload
    assert "scored_query" in payload


def test_absent_scored_query_is_null_and_distinguishable_from_empty() -> None:
    """Forward-only. `None` means "unknown", `""` means "scored nothing".

    A consumer must be able to tell a pre-#1405 row from a fire whose query
    genuinely transformed to the empty string — 8.5% of recorded queries do
    exactly that, and conflating the two would silently reclassify them.
    """
    unknown = _build_rebuild_log_record(
        recent_turns=[], session_id="s", candidates=[], pack_summary={},
    )
    empty = _build_rebuild_log_record(
        recent_turns=[], session_id="s", candidates=[], pack_summary={},
        scored_query="",
    )
    unknown_payload = unknown["input"]
    empty_payload = empty["input"]
    assert isinstance(unknown_payload, dict)
    assert isinstance(empty_payload, dict)
    assert unknown_payload["scored_query"] is None
    assert empty_payload["scored_query"] == ""


def test_ups_path_records_the_string_retrieve_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end on the path that writes 97.7% of rows.

    The conversation-aware query is *not* the prompt and not the extraction of
    it, so this asserts the row carries the composed string and that it differs
    from `extracted_query` — the second half being the whole point.
    """
    log_path = tmp_path / "s1.jsonl"
    composed = "the prompt the prompt the prompt earlier turn text"
    record_user_prompt_submit_log(
        prompt="the prompt",
        session_id="s1",
        hits_pre_dedup=[_belief("b1")],
        hits_post_dedup=[_belief("b1")],
        log_path=log_path,
        scored_query=composed,
        enabled=True,
    )
    payload = _row(log_path)["input"]
    assert payload["scored_query"] == composed
    assert payload["scored_query"] != payload["extracted_query"], (
        "if these were equal the fixture would not exercise the defect"
    )


def test_hook_passes_the_retrieval_query_and_none_on_the_gate_skip_branch() -> None:
    """Both hook call sites, read from source rather than executed.

    Driving `user_prompt_submit` end-to-end here would need a store, a payload
    and a settings file; the property under test is which *expression* each
    call site passes, which the source states directly.

    The gate-skip branch must pass `None`: retrieval never ran there, and
    `retrieval_query` is assigned only in the sibling branch — naming it would
    raise `NameError` inside the hook.

    That is **not** a crash, which is what makes it worth pinning here. The
    emit at `hook.py:1285` sits inside the `try:` opened at `hook.py:927`,
    whose handler at `hook.py:1343` catches bare `Exception` and only prints a
    traceback (`# non-blocking: surface but do not fail`). `NameError` is an
    `Exception`, so the hook still returns 0 and the turn still proceeds; what
    is lost is the `rebuild_log` row for that turn, silently, plus stderr
    noise the user never sees. An observability hole is exactly the kind of
    defect a source-level assertion has to carry, because no behavioural test
    would go red on it.
    """
    source = (
        Path(__file__).resolve().parents[1] / "src" / "aelfrice" / "hook.py"
    ).read_text()
    assert "scored_query=retrieval_query," in source
    assert "scored_query=None," in source

    skip_branch = source[source.index("        elif gate_skip:"):]
    skip_call = _call_block(
        skip_branch, "_emit_user_prompt_submit_rebuild_log("
    )
    assert "scored_query=None," in skip_call
    assert "scored_query=retrieval_query," not in skip_call


def test_rebuild_v14_passes_the_post_transform_query() -> None:
    """The call site must hand over `query`, not `raw_query`.

    They differ on 99.4% of real queries, and `raw_query` is precisely the
    string the old row already carried — passing it would leave the defect in
    place under a new key.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "aelfrice" / "context_rebuilder.py"
    ).read_text()
    call = _call_block(source, "record = _build_rebuild_log_record(")
    assert "scored_query=query," in call
    assert "scored_query=raw_query" not in call


def _rebuild_store(db_path: Path):
    """A store whose L0 lock makes the candidate set deterministic."""
    from aelfrice.models import LOCK_USER
    from aelfrice.store import MemoryStore

    store = MemoryStore(str(db_path))
    store.insert_belief(
        Belief(
            id="L1",
            content="user prefers uv over pip",
            content_hash="h_L1",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at="2026-04-28T00:00:00Z",
            created_at="2026-04-28T00:00:00Z",
            last_retrieved_at=None,
        )
    )
    return store

# ---- same-fire equality, executed rather than grepped -------------------


def test_rebuild_v14_logs_the_same_string_it_handed_retrieve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1405's load-bearing AC, on the path a source grep cannot cover.

    `test_rebuild_v14_passes_the_post_transform_query` asserts the literal
    `scored_query=query,` appears in the call block. That is a claim about
    identifier spelling, not about a value: rebinding `query` between
    `retrieve(...)` at :395 and the log block at :588 — a later lane doing
    `query = expand(query)` — leaves the grep green while every row again
    records a string `retrieve()` never saw.

    This spies on the real `retrieve` and compares what it received with
    what landed on disk, in the same fire. A marker on `transform_query`
    keeps the two distinguishable: without it the pre- and post-transform
    strings can coincide and the assertion would hold for the wrong
    reason.
    """
    from aelfrice import context_rebuilder as cr

    seen: list[str] = []
    real_retrieve = cr.retrieve

    def spy(store: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        seen.append(query)
        return real_retrieve(store, query, *args, **kwargs)

    monkeypatch.setattr(cr, "retrieve", spy)
    monkeypatch.setattr(
        cr, "transform_query", lambda raw, *a, **k: f"TRANSFORMED {raw}"
    )

    store = _rebuild_store(tmp_path / "m.db")
    log_path = tmp_path / "rebuild_logs" / "sess1.jsonl"
    try:
        cr.rebuild_v14(
            [cr.RecentTurn(role="user", text="uv over pip for environments")],
            store,
            rebuild_log_path=log_path,
            session_id_for_log="sess1",
        )
    finally:
        store.close()

    assert len(seen) == 1, f"expected one retrieve() call, got {len(seen)}"
    rows = [
        json.loads(ln)
        for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1
    logged = rows[0]["input"]["scored_query"]

    assert logged == seen[0], (
        "the row must carry the string retrieve() was handed in this same "
        f"fire; retrieve() got {seen[0]!r}, the row recorded {logged!r}"
    )
    # Distinguishing half: proves the marker actually rode through, so a
    # regression to the pre-transform string cannot satisfy the equality
    # above by the two happening to coincide.
    assert logged.startswith("TRANSFORMED "), logged
    assert rows[0]["input"]["extracted_query"] == logged.removeprefix(
        "TRANSFORMED "
    ), "extracted_query must remain the pre-transform value, not be overwritten"
