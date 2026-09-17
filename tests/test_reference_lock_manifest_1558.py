"""A reference-tier lock renders as a manifest entry on every path (#1558).

`aelf lock --reference` exists so long-form locked material costs one line
and is read on demand (#1016-B). Two lanes render a lock, and only one of
them honoured the tier: `_split_belief_lines` diverts a reference lock to
`retrieval.lock_manifest_line`, while the `<locked>` loop of
`_build_session_start_subblock` rendered every row verbatim. The sub-block is
what a session's *first* prompt carries, so the tier was a measured no-op
exactly where a lock-only store overruns the block ceiling, and on the
retrieval branch the envelope carried the full text **and** a `ref` pointer
to that same text a few lines below.

`<core>` is the third `<belief>` renderer and is not a third case of this
defect. `_core_belief_line` has no `is_reference_lock` branch and would emit
a full element for a reference lock if it were handed one; it is not handed
one, because `_build_session_start_subblock` filters every id in
`store.list_locked_beliefs()` out of `core_candidates`. That is exclusion,
not diversion, and it is a different mechanism with a different failure
mode, so it is pinned separately below rather than counted as a renderer
that "already honoured" the tier.

Three things these tests are careful about:

**A test per emit path.** The defect is that one of the two lock-rendering
call sites diverted and the other was never asked, so each of
`user_prompt_submit`'s gate-skip branch, its retrieval branch and the
`session_start` baseline is fired separately here. A renderer that picks up
the old behaviour by omission has to red one of the three.

**The frozen arm is a byte comparison, not a shape assertion.** The golden
below was produced by running the pre-#1558 `_build_session_start_subblock`
over the same fixture, so "a frozen lock is unchanged" means the bytes, not
the tags.

**The diverted id must stay out of `_ids_rendered_verbatim_in`.** That set
means "the full text is in this window", which is the one thing a manifest
entry says is not true. #1547's dedupe and the #1382 ledger both read it, so
a fix that swapped the element's content instead of diverting the row would
have made both of them lie.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import pytest

from aelfrice.hook import (
    CORE_CLOSE_TAG,
    CORE_OPEN_TAG,
    LOCKS_MANIFEST_CLOSE_TAG,
    LOCKS_MANIFEST_OPEN_TAG,
    SESSION_START_SUBBLOCK_CLOSE,
    _build_session_start_subblock,
    _core_belief_line,
    _drop_duplicate_ref_lines,
    _format_hits_with_session_start,
    _ids_rendered_verbatim_in,
    _lift_manifest_block,
    session_start,
    user_prompt_submit,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

_REPO = Path(__file__).resolve().parents[1]
_MBC_SCRIPT = _REPO / "scripts" / "measure_block_ceiling.py"
_mbc_spec = importlib.util.spec_from_file_location("_mbc_1558", _MBC_SCRIPT)
assert _mbc_spec and _mbc_spec.loader
# `Any` on purpose: pyright runs `tests/` strict, and an implicitly-typed
# module object turns every attribute read into an `Unknown`.
_mbc: Any = importlib.util.module_from_spec(_mbc_spec)
_mbc_spec.loader.exec_module(_mbc)

_FROZEN_ID = "F" * 16
_REFERENCE_ID = "R" * 16

_FROZEN = "the deploy runs through <merge-train>, never a direct push"
# A sentinel no framing tag, id or topic can contain, so "the text is in the
# block" is decided by one substring and not by a shape.
_SENTINEL = "zqxjrelockedbodyzqxj"
_REFERENCE = (
    "Reference material for the retrieval lane. " + _SENTINEL + " " + "x" * 400
)

# Over `hook._MIN_PROMPT_LEN` (12), so `_should_skip_bm25` admits BM25 and the
# fire takes the retrieval branch.
_PROMPT = "tell me everything about the reference material please"
# Under it, so the fire takes the `elif gate_skip:` branch instead.
_GATED_PROMPT = "ok"

# The exact bytes the `<locked>` loop emitted for `_FROZEN` before #1558,
# captured by running the pre-change `_build_session_start_subblock` over the
# `_frozen_store` fixture. This is the regression arm: the diversion must not
# move a single byte of a frozen lock's render.
_FROZEN_GOLDEN = (
    "<session-start>\n"
    "<locked>\n"
    f'<belief id="{_FROZEN_ID}" lock="user">the deploy runs through '
    "&lt;merge-train&gt;, never a direct push</belief>\n"
    "</locked>\n"
    "<core>\n"
    "</core>\n"
    "</session-start>"
)


@pytest.fixture(autouse=True)
def _pin_ceiling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is not what is under test; an exported value must not act."""
    monkeypatch.delenv("AELFRICE_HOOK_BLOCK_CEILING", raising=False)


def _lock(bid: str, content: str, tier: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        lock_tier=tier,
        locked_at="2026-04-26T00:00:00Z",
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _store(tmp_path: Path, *locks: Belief) -> Path:
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        for b in locks:
            store.insert_belief(b)
    finally:
        store.close()
    return db


def _sub_block(db: Path, cwd: Path) -> str:
    store = MemoryStore(str(db))
    try:
        return _build_session_start_subblock(store, cwd=cwd)
    finally:
        store.close()


def _fire_ups(
    tmp_path: Path,
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt: str,
    session_id: str,
) -> str:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )
    rc = user_prompt_submit(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    assert rc == 0
    # The hook fails soft, so an exception inside it becomes a stderr trace
    # and rc 0 — indistinguishable from a small block.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue()


def _fire_session_start(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": "ss-1558",
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "SessionStart",
        }
    )
    rc = session_start(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    assert rc == 0
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue()


# ---------------------------------------------------------------------------
# AC1 — the <locked> loop diverts a reference-tier lock
# ---------------------------------------------------------------------------


def test_the_locked_loop_diverts_a_reference_lock_to_the_manifest(
    tmp_path: Path,
) -> None:
    db = _store(tmp_path, _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE))
    block = _sub_block(db, tmp_path)
    assert f'<belief id="{_REFERENCE_ID}"' not in block
    assert f'  ref {_REFERENCE_ID}: "' in block
    assert _SENTINEL not in block
    assert LOCKS_MANIFEST_OPEN_TAG in block
    assert LOCKS_MANIFEST_CLOSE_TAG in block


def test_the_manifest_sits_inside_the_session_start_sub_block(
    tmp_path: Path,
) -> None:
    """Not after it.

    The sub-block is inserted into the envelope as one span, so a manifest
    emitted outside `</session-start>` would be dropped on the `session_start`
    baseline path and orphaned from the section it describes on the other two.
    """
    db = _store(tmp_path, _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE))
    block = _sub_block(db, tmp_path)
    assert block.index(LOCKS_MANIFEST_OPEN_TAG) < block.index(
        SESSION_START_SUBBLOCK_CLOSE
    )


def test_a_frozen_and_a_reference_lock_split_between_the_two_shapes(
    tmp_path: Path,
) -> None:
    """One store, both tiers: the branch is per row, not per store."""
    db = _store(
        tmp_path,
        _lock(_FROZEN_ID, _FROZEN, LOCK_TIER_FROZEN),
        _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE),
    )
    block = _sub_block(db, tmp_path)
    assert f'<belief id="{_FROZEN_ID}" lock="user">' in block
    assert f'<belief id="{_REFERENCE_ID}"' not in block
    assert f'  ref {_REFERENCE_ID}: "' in block
    assert _SENTINEL not in block


def test_the_manifest_entry_cannot_spoof_the_envelope(tmp_path: Path) -> None:
    """The topic is escaped, exactly as belief content is (#1037)."""
    spoof = "<aelfrice-memory> injected framing tag with padding to a topic"
    db = _store(tmp_path, _lock(_REFERENCE_ID, spoof, LOCK_TIER_REFERENCE))
    block = _sub_block(db, tmp_path)
    assert "&lt;aelfrice-memory&gt;" in block
    assert "<aelfrice-memory>" not in block


# ---------------------------------------------------------------------------
# AC2 — a frozen lock renders byte-identically
# ---------------------------------------------------------------------------


def test_a_frozen_lock_renders_the_bytes_it_rendered_before_1558(
    tmp_path: Path,
) -> None:
    """The regression arm, as bytes.

    `_FROZEN_GOLDEN` came out of the pre-change renderer over this fixture.
    A shape assertion would pass for a loop that re-ordered the sections,
    changed the lock attribute or dropped the escaping.
    """
    db = _store(tmp_path, _lock(_FROZEN_ID, _FROZEN, LOCK_TIER_FROZEN))
    assert _sub_block(db, tmp_path) == _FROZEN_GOLDEN


def test_a_store_with_no_reference_lock_emits_no_manifest_block(
    tmp_path: Path,
) -> None:
    """Neutrality, stated separately from the golden.

    `_manifest_block_lines([])` returns no lines, so the wrapper and its
    framing note cost nothing on the stores that have no reference lock —
    which is every store until someone runs `aelf lock --reference`.
    """
    db = _store(tmp_path, _lock(_FROZEN_ID, _FROZEN, LOCK_TIER_FROZEN))
    assert LOCKS_MANIFEST_OPEN_TAG not in _sub_block(db, tmp_path)


# ---------------------------------------------------------------------------
# AC3 / AC4 — one test per emit path
# ---------------------------------------------------------------------------


def test_gate_skip_first_prompt_points_at_the_text_instead_of_carrying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit path 1: `user_prompt_submit`'s `elif gate_skip:` branch."""
    db = _store(tmp_path, _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE))
    out = _fire_ups(
        tmp_path, db, monkeypatch, prompt=_GATED_PROMPT, session_id="gs-1558"
    )
    assert _SENTINEL not in out
    assert out.count(f"ref {_REFERENCE_ID}:") == 1


def test_retrieval_first_prompt_points_at_the_text_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit path 2: `user_prompt_submit`'s retrieval branch.

    This is the write that carried the text *and* a pointer to it. Both
    halves of the envelope reach the same lock — the sub-block through
    `list_locked_beliefs()` and the per-turn pack through the L0 lane — so
    the count is what separates "diverted" from "diverted and then pointed
    at twice".
    """
    db = _store(tmp_path, _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE))
    out = _fire_ups(
        tmp_path, db, monkeypatch, prompt=_PROMPT, session_id="r1-1558"
    )
    assert _SENTINEL not in out
    assert out.count(f"ref {_REFERENCE_ID}:") == 1


def test_session_start_baseline_points_at_the_text_instead_of_carrying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit path 3: the `session_start` baseline.

    This path already diverted before #1558 — it renders through
    `_format_baseline_hits`, not the sub-block — and it is here so the third
    call site is asked rather than assumed.
    """
    db = _store(tmp_path, _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE))
    out = _fire_session_start(tmp_path, db, monkeypatch)
    assert _SENTINEL not in out
    assert out.count(f"ref {_REFERENCE_ID}:") == 1


def test_a_frozen_lock_still_reaches_all_three_paths_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of every assertion above.

    Without this, a renderer that dropped locked content altogether would
    satisfy every "the text is not in the block" test in this file.
    """
    db = _store(tmp_path, _lock(_FROZEN_ID, _FROZEN, LOCK_TIER_FROZEN))
    escaped = "the deploy runs through &lt;merge-train&gt;, never a direct push"
    assert escaped in _fire_ups(
        tmp_path, db, monkeypatch, prompt=_GATED_PROMPT, session_id="gs-frozen"
    )
    assert escaped in _fire_ups(
        tmp_path, db, monkeypatch, prompt=_PROMPT, session_id="r1-frozen"
    )
    assert escaped in _fire_session_start(tmp_path, db, monkeypatch)


# ---------------------------------------------------------------------------
# The invariant the fix is shaped around
# ---------------------------------------------------------------------------


def test_the_diverted_id_is_not_counted_as_rendered_verbatim(
    tmp_path: Path,
) -> None:
    """A `ref` entry is not an exposure.

    `_ids_rendered_verbatim_in` feeds #1547's dedupe and the #1382 ledger,
    both of which mean "the full text is in this window". The text is not,
    so the id must be absent — which is why the row is diverted rather than
    the element's content swapped for a manifest line.
    """
    db = _store(
        tmp_path,
        _lock(_FROZEN_ID, _FROZEN, LOCK_TIER_FROZEN),
        _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE),
    )
    rendered = _ids_rendered_verbatim_in(_sub_block(db, tmp_path))
    assert _FROZEN_ID in rendered
    assert _REFERENCE_ID not in rendered


def test_the_envelope_drops_a_ref_line_the_sub_block_already_carries() -> None:
    """`_drop_duplicate_ref_lines`, on the shape the envelope hands it.

    Both arguments are manifest entries: the second is what
    `_lift_manifest_block` cut out of the sub-block, not the sub-block.
    """
    lines = [
        f'  ref {_REFERENCE_ID}: "topic"',
        '  ref OTHER0000000000: "another topic"',
    ]
    already = [f'  ref {_REFERENCE_ID}: "topic"']
    assert _drop_duplicate_ref_lines(lines, already) == [lines[1]]


def test_a_seen_pointer_is_not_dropped_by_the_ref_dedupe() -> None:
    """`seen` and `ref` are different claims and have different accounting.

    A `seen` line names an element rendered verbatim in this window;
    `enforce_block_ceiling` removes one only when it removes that element.
    Filtering it here on an id match would delete a live pointer.
    """
    lines = [f'  seen {_FROZEN_ID}: "topic"']
    assert _drop_duplicate_ref_lines(
        lines, [f'  ref {_FROZEN_ID}: "topic"']
    ) == lines


def test_an_envelope_without_a_sub_block_keeps_its_ref_line() -> None:
    """Turn two, where the dedupe has nothing to compare against.

    From the second prompt of a session there is no sub-block, so the
    per-turn `ref` line is the envelope's only pointer and dropping it would
    lose the reference lock entirely.
    """
    hit = _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE)
    out = _format_hits_with_session_start([hit], "")
    assert out.count(f"ref {_REFERENCE_ID}:") == 1
    assert _SENTINEL not in out
    assert _drop_duplicate_ref_lines(
        [f'  ref {_REFERENCE_ID}: "topic"'], []
    ) == [f'  ref {_REFERENCE_ID}: "topic"']


# ---------------------------------------------------------------------------
# One manifest wrapper per envelope
# ---------------------------------------------------------------------------


def test_a_first_prompt_envelope_carries_one_manifest_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture that produced two of them.

    A frozen lock renders verbatim in `<locked>` and is reached again by the
    per-turn pack, which points at it with a `seen` line; a reference lock
    is diverted into the sub-block's own `ref` line. Before the wrappers
    were merged that envelope opened `<aelfrice-locks-manifest>` twice and
    repeated its framing note, on the one write this branch exists to
    shrink. Both entries must still be present — a single wrapper reached by
    dropping one of them is not the fix.
    """
    db = _store(
        tmp_path,
        _lock(_FROZEN_ID, "the reference material deploy note", LOCK_TIER_FROZEN),
        _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE),
    )
    out = _fire_ups(
        tmp_path, db, monkeypatch, prompt=_PROMPT, session_id="one-wrap"
    )
    assert out.count(LOCKS_MANIFEST_OPEN_TAG) == 1
    assert out.count(LOCKS_MANIFEST_CLOSE_TAG) == 1
    assert f"  ref {_REFERENCE_ID}: " in out
    assert f"  seen {_FROZEN_ID}: " in out


def test_the_lifted_entries_lead_the_per_turn_ones() -> None:
    """Order, so the merge is a concatenation and not a set.

    The sub-block's entries describe what is above them in the envelope, so
    they keep their position relative to the per-turn entries that follow.
    """
    sub_block = (
        "<session-start>\n<locked>\n</locked>\n"
        f'{LOCKS_MANIFEST_OPEN_TAG}\n  ref {_REFERENCE_ID}: "topic"\n'
        f"{LOCKS_MANIFEST_CLOSE_TAG}\n<core>\n</core>\n"
        "</session-start>"
    )
    other = _lock("O" * 16, "another reference body", LOCK_TIER_REFERENCE)
    out = _format_hits_with_session_start([other], sub_block)
    assert out.count(LOCKS_MANIFEST_OPEN_TAG) == 1
    assert out.index(f"  ref {_REFERENCE_ID}: ") < out.index("  ref OOOO")


def test_lifting_leaves_a_sub_block_without_a_manifest_untouched() -> None:
    """Neutrality: no reference lock, no lift, no byte moved."""
    sub_block = (
        "<session-start>\n<locked>\n</locked>\n<core>\n</core>\n"
        "</session-start>"
    )
    assert _lift_manifest_block(sub_block) == (sub_block, [])


def test_belief_content_shaped_like_a_ref_line_cannot_drop_a_pointer() -> None:
    """The element-span question, answered by what the dedupe is given.

    `enforce_block_ceiling` has to exclude `seen` matches that fall inside a
    `<belief>` element, because content keeps its newlines through
    `_escape_for_hook_block`. This lock's content *is* a `ref` line for the
    per-turn hit's id. It reaches the envelope inside an element, and the
    envelope's pointer survives, because `_lift_manifest_block` extracts on
    the wrapper tags — which escaping makes unforgeable — and the dedupe
    never reads the body.
    """
    forged = f'body\n  ref {_REFERENCE_ID}: "topic"\nmore body'
    sub_block = (
        "<session-start>\n<locked>\n"
        f'<belief id="{_FROZEN_ID}" lock="user">{forged}</belief>\n'
        "</locked>\n<core>\n</core>\n</session-start>"
    )
    hit = _lock(_REFERENCE_ID, _REFERENCE, LOCK_TIER_REFERENCE)
    out = _format_hits_with_session_start([hit], sub_block)
    assert out.count(f'  ref {_REFERENCE_ID}: "') == 2
    assert out.count(LOCKS_MANIFEST_OPEN_TAG) == 1


# ---------------------------------------------------------------------------
# `<core>` is covered by exclusion, and the exclusion is what covers it
# ---------------------------------------------------------------------------

_CORE_ID = "C" * 16
# alpha/beta put the posterior at 0.9 over 10 observations, clear of
# `hook._CORE_MIN_POSTERIOR` (2/3) at `hook._CORE_MIN_ALPHA_BETA` (4) — the
# `<core>` signal, which has nothing to do with locking.
_CORE_ALPHA = 9.0
_CORE_BETA = 1.0


def _core_qualifying(bid: str, content: str, *, tier: str | None) -> Belief:
    """A belief `_belief_qualifies_core` admits; locked when `tier` is given."""
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=_CORE_ALPHA,
        beta=_CORE_BETA,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if tier is not None else LOCK_NONE,
        lock_tier=tier if tier is not None else LOCK_TIER_FROZEN,
        locked_at="2026-04-26T00:00:00Z" if tier is not None else None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _core_section(block: str) -> str:
    start = block.index(CORE_OPEN_TAG) + len(CORE_OPEN_TAG)
    return block[start : block.index(CORE_CLOSE_TAG)]


def test_the_core_renderer_does_not_divert_a_reference_lock() -> None:
    """The half of the claim that is about the renderer.

    `_core_belief_line` has no `is_reference_lock` branch. Handed a
    reference-tier lock it returns the belief's full text inside a
    `<belief>` element, exactly as it would for any other belief — so
    whatever keeps a reference lock's body out of `<core>`, it is not this
    function. Pinned so the framing cannot drift back to calling `<core>` a
    renderer that already honoured the tier.
    """
    ref = _core_qualifying(_REFERENCE_ID, _REFERENCE, tier=LOCK_TIER_REFERENCE)
    line = _core_belief_line(ref)
    assert line.startswith(f'<belief id="{_REFERENCE_ID}"')
    assert _SENTINEL in line
    assert " ref " not in line


def test_core_excludes_every_lock_rather_than_diverting_one(
    tmp_path: Path,
) -> None:
    """The half that is about the mechanism, with its own control.

    Both beliefs carry the same `<core>` signal and differ only in whether
    they are locked. The unlocked one reaches `<core>`, which is what makes
    the lock's absence a statement about `core_candidates` rather than about
    a fixture that failed to qualify: `_build_session_start_subblock` skips
    every id `store.list_locked_beliefs()` returns, and that query selects
    `lock_level != 'none'`, so it takes both lock tiers alike.
    """
    unlocked = _core_qualifying(_CORE_ID, "unlocked core belief", tier=None)
    ref = _core_qualifying(_REFERENCE_ID, _REFERENCE, tier=LOCK_TIER_REFERENCE)
    block = _sub_block(_store(tmp_path, unlocked, ref), tmp_path)
    core = _core_section(block)
    assert f'<belief id="{_CORE_ID}"' in core
    assert _REFERENCE_ID not in core
    assert _SENTINEL not in block


def test_a_frozen_lock_is_excluded_from_core_too(tmp_path: Path) -> None:
    """Exclusion is per lock, not per tier.

    A diversion branch would have to be written twice to get this right;
    the filter gets it right once. This is the assertion that separates
    "`<core>` skips locks" from "`<core>` skips reference locks".
    """
    frozen = _core_qualifying(_FROZEN_ID, _FROZEN, tier=LOCK_TIER_FROZEN)
    block = _sub_block(_store(tmp_path, frozen), tmp_path)
    assert _FROZEN_ID not in _core_section(block)
    assert f'<belief id="{_FROZEN_ID}" lock="user">' in block


# ---------------------------------------------------------------------------
# The producer's vacuity guard is per write
# ---------------------------------------------------------------------------


def _tier_rows(equal_on: str) -> dict[str, int]:
    """Reference-arm figures that match the frozen arm on `equal_on` alone."""
    return {w: (100 if w == equal_on else 10) for w in _mbc.REF_WRITES}


@pytest.mark.parametrize("write", list(_mbc.REF_WRITES))
def test_the_reference_tier_guard_trips_on_one_equal_write(
    write: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One equal column is enough, and the message names which one.

    The producer's own docstring publishes this, so it is pinned rather than
    described. An "equal on every write" guard — which is what this was
    before #1558 — passes as long as one write differs, and until #1558 one
    did: turn two and `session_start` bounded while both first-prompt writes
    read the same figure at either tier. Parametrised over every write so
    the guard cannot come to hold on three of the four.
    """
    frozen: dict[str, int] = {w: 100 for w in _mbc.REF_WRITES}

    def _rows(tier: str) -> dict[str, int]:
        return frozen if tier == LOCK_TIER_FROZEN else _tier_rows(write)

    monkeypatch.setattr(_mbc, "reference_tier", _rows)
    with pytest.raises(SystemExit) as exc:
        _mbc.reference_tier_table()
    message = str(exc.value)
    assert write in message
    for other in _mbc.REF_WRITES:
        if other != write:
            assert other not in message


def test_the_reference_tier_guard_passes_when_every_write_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control.

    Without it the test above would also pass for a guard that raised on
    every input, which proves nothing about the condition.
    """
    def _rows(tier: str) -> dict[str, int]:
        value = 100 if tier == LOCK_TIER_FROZEN else 10
        return {w: value for w in _mbc.REF_WRITES}

    monkeypatch.setattr(_mbc, "reference_tier", _rows)
    rows = _mbc.reference_tier_table()
    assert set(rows) == {LOCK_TIER_FROZEN, LOCK_TIER_REFERENCE}
