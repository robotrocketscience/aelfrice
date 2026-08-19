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

The remaining tests pin the two properties the default-ON ruling of
2026-08-11 rests on: the failure direction is one-way, and a suppressed block
records nothing.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import session_start, user_prompt_submit
from aelfrice.injection_ledger import LEDGER_FILENAME
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

_CONTENT = "the cellar door is full of barrels and casks"

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
    rc = user_prompt_submit(
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
    rc = session_start(
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
    p = tmp_path / LEDGER_FILENAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A seeded store with the differential on and the block emitting."""
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_TURN_DIFFERENTIAL", raising=False)
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
    monkeypatch.delenv("AELFRICE_TURN_DIFFERENTIAL", raising=False)
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
    monkeypatch.delenv("AELFRICE_TURN_DIFFERENTIAL", raising=False)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)

    (tmp_path / LEDGER_FILENAME).write_text(
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
# The properties the default-ON ruling rests on
# --------------------------------------------------------------------------


def test_a_suppressed_block_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1359's off-switch must not manufacture an under-injection path.

    A suppressed fire puts no text in the context window. Recording it would
    make the next turn emit a reference to content the model was never shown —
    the one way this feature can under-inject.
    """
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_TURN_DIFFERENTIAL", raising=False)
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")

    out = _fire()
    assert _CONTENT not in out, "the block was emitted despite the off-switch"
    data = _ledger(tmp_path)
    assert data is None or "HIT01" not in data.get("rendered", []), (
        "a suppressed fire recorded an injection that never happened"
    )


def test_the_off_switch_restores_verbatim_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", _CONTENT)])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "0")

    assert _CONTENT in _fire()
    assert _CONTENT in _fire(), (
        "the off switch did not restore pre-#1382 verbatim rendering"
    )


def test_a_corrupt_ledger_renders_verbatim(wired: Path) -> None:
    """Fail-soft direction: unreadable state costs tokens, never content."""
    _fire()
    (wired / LEDGER_FILENAME).write_text("{ not json", encoding="utf-8")
    assert _CONTENT in _fire(), (
        "a corrupt ledger suppressed content; every failure path must render "
        "verbatim"
    )


def test_a_foreign_session_renders_verbatim(wired: Path) -> None:
    """A ledger written under another session is not ours to trust."""
    _fire(session_id="sess-A")
    assert _CONTENT in _fire(session_id="sess-B"), (
        "a ledger from a different session suppressed content"
    )
