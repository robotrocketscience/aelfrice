"""The turn-differential is actually wired into the live hook path (#1382).

`tests/test_turn_differential_1382.py` drives the formatters directly with a
hand-built `already_rendered` set. That is the right test for the render rule,
and it is **not** evidence that the feature runs: for two commits the ledger
was imported by no production module, `begin_epoch` and `record_rendered` had
zero production callers, and every production call site of
`_split_belief_lines` passed the default empty set. The suite was green the
whole time.

So this file asserts the edges, not the rule. Each test fires the real hook
entry point and looks at what reached stdout and what reached disk. Delete any
one of the three wiring edges and a named test here goes red:

  * UserPromptSubmit reads the ledger    -> `test_a_second_fire_references_...`
  * UserPromptSubmit writes the ledger   -> `test_a_fire_records_what_it_...`
  * SessionStart opens the epoch         -> `test_session_start_replaces_...`

The remaining tests pin the switch and the safety properties: the feature is
off by default (2026-08-19, reversing the 2026-08-11 default-on ruling), a
suppressed block records nothing, and every epoch boundary actually resets —
including the ones a store-read failure or an empty baseline used to skip.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook as hook_mod
from aelfrice import injection_ledger as ledger_mod
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

# MUST exceed retrieval._LOCK_TOPIC_MAX (80). Below the cap,
# `seen_manifest_line` embeds the WHOLE content, so `assert _CONTENT in out`
# is satisfied by the reference line and every verbatim-vs-reference assertion
# built on it is vacuous. Three tests in this file were vacuous for exactly
# that reason until the adversarial review of PR #1515 caught it.
_CONTENT = (
    "the cellar door is full of oak barrels and copper casks, and the north "
    "aisle inventory is recounted every quarter by the night warden"
)

# `retrieval._LOCK_TOPIC_MAX` (80) plus the `seen <id>: ""` wrapper and the
# two-space manifest indent. Stated as a bound, not the exact width, so the
# test pins "bounded" rather than a formatting detail.
_MANIFEST_LINE_CAP = 120
_PROMPT = "how many barrels are in the cellar door storage"


def _mk(bid: str, content: str, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-08-19T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-08-19T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _fire(prompt: str = _PROMPT, session_id: str = "sess-A") -> str:
    sout = io.StringIO()
    rc = hook_mod.user_prompt_submit(
        stdin=io.StringIO(
            json.dumps(
                {
                    "session_id": session_id,
                    "transcript_path": "/dev/null",
                    "cwd": "/tmp",
                    "hook_event_name": "UserPromptSubmit",
                    "prompt": prompt,
                }
            )
        ),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return sout.getvalue()


def _fire_session_start(session_id: str = "sess-A") -> str:
    sout = io.StringIO()
    rc = hook_mod.session_start(
        stdin=io.StringIO(
            json.dumps(
                {
                    "session_id": session_id,
                    "cwd": "/tmp",
                    "hook_event_name": "SessionStart",
                    "source": "startup",
                }
            )
        ),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return sout.getvalue()


def _ledger(tmp_path: Path) -> dict[str, object] | None:
    p = tmp_path / ledger_mod.LEDGER_FILENAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A seeded store with the differential on and the block emitting."""
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    # Default-OFF since 2026-08-19, so the feature must be switched ON here.
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    return tmp_path


# --------------------------------------------------------------------------
# The three wiring edges
# --------------------------------------------------------------------------


def test_a_fire_records_what_it_rendered_verbatim(wired: Path) -> None:
    """Edge 2: UserPromptSubmit writes the ledger after rendering."""
    out = _fire()
    assert "HIT01" in out, "belief did not reach the block; test is not set up"
    data = _ledger(wired)
    assert data is not None, (
        "no ledger was written — record_rendered has no production caller"
    )
    assert data["session_id"] == "sess-A"
    assert "HIT01" in data["rendered"]


def test_a_second_fire_references_instead_of_repeating(wired: Path) -> None:
    """Edge 1: UserPromptSubmit reads the ledger before rendering.

    This is the feature. Asserted structurally rather than on the absence of
    the content string, because `seen_manifest_line` embeds the topic — and
    for a belief shorter than `_LOCK_TOPIC_MAX` (80) the topic *is* the whole
    content. A content-absence assertion would therefore pass only for long
    beliefs and read as a broken feature for short ones.
    """
    first = _fire()
    assert '<belief id="HIT01"' in first, (
        "first fire did not render the belief verbatim"
    )
    assert "seen HIT01" not in first, "first fire referenced a belief it never showed"

    second = _fire()
    assert '<belief id="HIT01"' not in second, (
        "the second fire re-rendered the identical <belief> element — the "
        "ledger is written but never read"
    )
    assert "seen HIT01" in second, (
        "the belief vanished entirely; it must still appear as a reference"
    )


def test_the_reference_is_bounded_where_the_saving_comes_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prize is the cap, not the tag overhead.

    `seen_manifest_line` truncates at `_LOCK_TOPIC_MAX` (80 chars), so a long
    belief collapses to a bounded line while a short one saves only the ~24
    characters of `<belief …>` wrapper. This pins the regime that actually
    pays, so a change to the cap cannot quietly delete the benefit.
    """
    long_content = "the cellar door inventory " + "barrels and casks " * 40
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", long_content)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)

    q = "how many barrels and casks are in the cellar door inventory"
    first, second = _fire(q), _fire(q)
    assert len(second) < len(first), "the second fire was not smaller"

    # The invariant that pays, asserted directly rather than through the
    # block total: the reference is bounded, whatever the belief's length.
    # The block total also carries a fixed framing header and a one-time
    # manifest note, so asserting on it alone would pin unrelated prose.
    seen = [ln for ln in second.splitlines() if ln.strip().startswith("seen ")]
    assert len(seen) == 1, f"expected one seen line, got {seen}"
    assert len(seen[0]) < _MANIFEST_LINE_CAP, (
        f"the reference line is {len(seen[0])} chars for a "
        f"{len(long_content)}-char belief; it must be bounded by the topic cap"
    )
    assert len(long_content) > 4 * len(seen[0]), (
        "fixture too short to demonstrate the bound; lengthen it"
    )


def test_session_start_replaces_the_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge 3: SessionStart opens an epoch, and it replaces rather than unions.

    A stale ledger naming a belief the new window has never seen must not
    survive the boundary, or that belief is suppressed forever.

    Seeds a LOCKED belief: the SessionStart baseline retrieves on an empty
    query, so an unlocked hit never reaches it and the test would assert on an
    empty block.
    """
    db = tmp_path / "memory.db"
    _seed(db, [_mk("LOCK01", "never push directly to main", LOCK_USER)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)

    (tmp_path / ledger_mod.LEDGER_FILENAME).write_text(
        json.dumps({"session_id": "sess-A", "rendered": ["GHOST"]}),
        encoding="utf-8",
    )
    out = _fire_session_start()
    assert out, "session_start emitted no baseline; test is not set up"
    data = _ledger(tmp_path)
    assert data is not None, "SessionStart wrote no ledger — begin_epoch unwired"
    assert "GHOST" not in data["rendered"], (
        "the previous epoch's ids survived the boundary; begin_epoch must "
        "replace, not union"
    )
    assert "LOCK01" in data["rendered"], (
        "the baseline rendered LOCK01 verbatim but did not record it"
    )


# --------------------------------------------------------------------------
# The switch, and the safety properties
# --------------------------------------------------------------------------


def test_a_suppressed_block_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1359's off-switch must not manufacture an under-injection path.

    A suppressed fire puts no text in the context window. Recording it would
    make the next turn emit a reference to content the model was never shown —
    one of the ways this feature can under-inject.
    """
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")

    out = _fire()
    assert _CONTENT not in out, "the block was emitted despite the off-switch"
    data = _ledger(tmp_path)
    assert data is None or "HIT01" not in data.get("rendered", []), (
        "a suppressed fire recorded an injection that never happened"
    )


def test_the_feature_is_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratified default-OFF, 2026-08-19.

    With the variable unset, two identical fires must both render verbatim.
    This is the assertion that catches a default flip, so it must not be
    weakened to "the second fire is smaller".
    """
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_TURN_DIFFERENTIAL", raising=False)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)

    assert '<belief id="HIT01"' in _fire()
    assert '<belief id="HIT01"' in _fire(), (
        "the turn-differential is ON by default; the 2026-08-19 ruling is OFF"
    )


def test_the_off_switch_restores_verbatim_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit 0 renders verbatim even if the default later flips on.

    Asserted on the `<belief>` element, not on the content string: `_CONTENT`
    used to be shorter than `_LOCK_TOPIC_MAX`, which made `_CONTENT in out`
    true of the `seen` line as well and left this test unable to detect a
    deleted off-switch at all.
    """
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "0")
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)

    assert '<belief id="HIT01"' in _fire()
    assert '<belief id="HIT01"' in _fire(), (
        "the off switch did not restore pre-#1382 verbatim rendering"
    )


def test_a_corrupt_ledger_renders_verbatim(wired: Path) -> None:
    """Fail-soft direction: unreadable state costs tokens, never content."""
    _fire()
    (wired / ledger_mod.LEDGER_FILENAME).write_text("{ not json", encoding="utf-8")
    assert '<belief id="HIT01"' in _fire(), (
        "a corrupt ledger suppressed content; every failure path must render "
        "verbatim"
    )


def test_a_foreign_session_renders_verbatim(wired: Path) -> None:
    """A ledger written under another session is not ours to trust."""
    _fire(session_id="sess-A")
    assert '<belief id="HIT01"' in _fire(session_id="sess-B"), (
        "a ledger from a different session suppressed content"
    )


# --------------------------------------------------------------------------
# The under-injection holes found by adversarial review of PR #1515
# --------------------------------------------------------------------------


def test_compaction_resets_the_epoch_even_with_an_empty_baseline(
    wired: Path,
) -> None:
    """The hole that falsified the "can only ever over-inject" claim.

    `begin_epoch` used to sit inside `if body:` in `session_start`. A store
    with no LOCKED beliefs renders an empty baseline, so the epoch never
    reset, the pre-compaction ledger survived under the same `session_id`, and
    every later turn emitted `seen <id>` for text the window no longer held —
    permanently reducing the belief to an 80-character stub the model was
    never shown.

    The fixture store holds one unlocked belief, so the SessionStart baseline
    is empty by construction. That is the case, not an accident of it.
    """
    assert '<belief id="HIT01"' in _fire(), "turn 1 did not render verbatim"
    assert "HIT01" in (_ledger(wired) or {}).get("rendered", [])

    body = _fire_session_start()
    assert body == "", (
        "the baseline rendered something; this fixture must have no locked "
        "beliefs for the test to exercise the empty-baseline path"
    )

    assert '<belief id="HIT01"' in _fire(), (
        "after compaction the belief was referenced but never shown — the "
        "epoch did not reset on an empty baseline"
    )


def test_a_boundary_without_a_session_id_invalidates_the_ledger(
    wired: Path,
) -> None:
    """`begin_epoch` returns early on a falsy id; that must not leave state.

    A SessionStart whose payload carries no `session_id` cannot scope an
    epoch. Leaving the previous epoch's file in place is an under-injection
    path, because the next fire may well match that id.
    """
    _fire(session_id="sess-A")
    assert (wired / ledger_mod.LEDGER_FILENAME).exists()

    sout = io.StringIO()
    hook_mod.session_start(
        stdin=io.StringIO(
            json.dumps({"cwd": "/tmp", "hook_event_name": "SessionStart"})
        ),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert not (wired / ledger_mod.LEDGER_FILENAME).exists(), (
        "an un-scopeable epoch boundary left the previous epoch's ledger live"
    )
    assert '<belief id="HIT01"' in _fire(session_id="sess-A")


def test_pre_compact_resets_the_epoch(wired: Path) -> None:
    """PreCompact is the event actually guaranteed to precede compaction."""
    _fire(session_id="sess-A")
    assert "HIT01" in (_ledger(wired) or {}).get("rendered", [])

    hook_mod.pre_compact(
        stdin=io.StringIO(
            json.dumps(
                {
                    "session_id": "sess-A",
                    "cwd": "/tmp",
                    "hook_event_name": "PreCompact",
                }
            )
        ),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert "HIT01" not in (_ledger(wired) or {}).get("rendered", []), (
        "PreCompact did not reset the epoch"
    )
    assert '<belief id="HIT01"' in _fire(session_id="sess-A")


def test_a_failed_baseline_read_still_resets_the_epoch(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary must not depend on the store read succeeding.

    `_retrieve_baseline_with_block` opens the store, which is a write (DDL
    plus migrations), so it can raise — and `session_start` wraps its whole
    body in one `except`. Reset after the read and any failure there skips the
    boundary, leaving the previous epoch live under an unchanged `session_id`.
    That is the same under-injection the empty-baseline case produced, by a
    different route, and it is likelier in practice: on a large store the hook
    is more often killed at its timeout mid-retrieve than raised out of.
    """
    _fire(session_id="sess-A")
    assert "HIT01" in (_ledger(wired) or {}).get("rendered", [])

    def boom(_budget: int) -> tuple[list[Belief], str]:
        raise RuntimeError("store open failed")

    original = hook_mod._retrieve_baseline_with_block
    monkeypatch.setattr(hook_mod, "_retrieve_baseline_with_block", boom)
    _fire_session_start(session_id="sess-A")

    assert "HIT01" not in (_ledger(wired) or {}).get("rendered", []), (
        "a failed baseline read skipped the epoch reset; the previous epoch's "
        "ids are still live"
    )

    # Restore the one attribute, NOT monkeypatch.undo(): `wired` sets
    # AELFRICE_DB and the feature flag through this same monkeypatch instance,
    # so undo() would point the next fire at a different store entirely and
    # the assertion below would report on the wrong tree.
    monkeypatch.setattr(hook_mod, "_retrieve_baseline_with_block", original)
    assert '<belief id="HIT01"' in _fire(session_id="sess-A"), (
        "after the reset the belief must render verbatim again"
    )


def test_a_failed_invalidate_is_reported_not_swallowed(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale ledger that cannot be removed goes on suppressing content.

    Every other write here fails benignly — the cost is a redundant verbatim
    injection. This one does not, so it must not pass silently as a completed
    reset. The hook contract forbids raising, so `invalidate` returns whether
    the ledger is gone and the caller surfaces a failure on stderr.
    """
    monkeypatch.setattr(ledger_mod, "invalidate", lambda **_kw: False)
    serr = io.StringIO()
    hook_mod._begin_injection_epoch(None, stderr=serr)

    assert "injection ledger" in serr.getvalue(), (
        "a failed invalidate was swallowed; an un-removable stale ledger must "
        f"be reported. stderr was: {serr.getvalue()!r}"
    )


def test_a_successful_invalidate_stays_quiet(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warning must not fire on the normal path, or it is noise."""
    serr = io.StringIO()
    hook_mod._begin_injection_epoch(None, stderr=serr)
    assert serr.getvalue() == "", (
        f"a successful reset wrote to stderr: {serr.getvalue()!r}"
    )
