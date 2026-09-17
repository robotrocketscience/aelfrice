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

import io
import json
from pathlib import Path

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
    """`_drop_duplicate_ref_lines`, on the shape the envelope hands it."""
    sub_block = (
        "<session-start>\n<locked>\n</locked>\n"
        f'{LOCKS_MANIFEST_OPEN_TAG}\n  ref {_REFERENCE_ID}: "topic"\n'
        f"{LOCKS_MANIFEST_CLOSE_TAG}\n</session-start>"
    )
    lines = [
        f'  ref {_REFERENCE_ID}: "topic"',
        '  ref OTHER0000000000: "another topic"',
    ]
    assert _drop_duplicate_ref_lines(lines, sub_block) == [lines[1]]


def test_a_seen_pointer_is_not_dropped_by_the_ref_dedupe() -> None:
    """`seen` and `ref` are different claims and have different accounting.

    A `seen` line names an element rendered verbatim in this window;
    `enforce_block_ceiling` removes one only when it removes that element.
    Filtering it here on an id match would delete a live pointer.
    """
    sub_block = (
        f'{LOCKS_MANIFEST_OPEN_TAG}\n  ref {_FROZEN_ID}: "topic"\n'
        f"{LOCKS_MANIFEST_CLOSE_TAG}"
    )
    lines = [f'  seen {_FROZEN_ID}: "topic"']
    assert _drop_duplicate_ref_lines(lines, sub_block) == lines


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
        [f'  ref {_REFERENCE_ID}: "topic"'], ""
    ) == [f'  ref {_REFERENCE_ID}: "topic"']


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
