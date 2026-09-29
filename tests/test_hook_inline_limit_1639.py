"""#1639: every hook fire's whole stdout fits the host's inline limit.

The host inlines at most 10,000 characters of a hook's output and replaces
anything longer with a 2,000-character preview plus a path to the saved file
(https://code.claude.com/docs/en/hooks.md). A block over that limit reaches
the model mostly unread, and user locks outside the preview are not
delivered at all. Measured on transcripts before this change: 41% of
aelfrice hook outputs were saved rather than inlined, and 46% of the saved
ones that carried locks lost at least one lock from the preview.

Operator rulings 2026-09-29, which reopen #1560 for one change:

* **One payload bound,** `HOOK_PAYLOAD_CHAR_LIMIT` = 9,500 characters over
  everything a fire writes to stdout. The per-block bounds stay.
* **The existing shed order does the trimming.** No new order.
* **Locks come first, and a lock that does not fit is named, not dropped
  silently.** When the locks alone overflow, the block ends with a line
  naming every omitted id and `aelf locked`, and stderr says how many.
* **The cadence checkpoint packs to the room left,** with a hard cut at an
  element boundary as the backstop for its soft pack loop.

Each fixture here was measured over the bound on github/main before the fix
(`FIX_1639_HYPOTHESES.md`, F1/F2/F4), so a pass is not vacuous.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

from aelfrice.hook import (
    HOOK_PAYLOAD_CHAR_LIMIT,
    session_start,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

# The shipped bound, as a literal: a fixture derived from the constant
# would follow it if someone raised it.
_LIMIT = 9_500
_WORD = "banana"
_PROMPT = f"tell me everything about the {_WORD} please"
_GATED_PROMPT = "ok"
_LOCK_RE = re.compile(r'<belief id="([^"]+)" lock="user"')
_ANY_BELIEF_RE = re.compile(r"<belief ")
_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AELFRICE_HOOK_BLOCK_CEILING", raising=False)
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))


def _mk(bid: str, content: str, *, locked: bool = False,
        alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=alpha,
        beta=beta, type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-26T00:00:00Z" if locked else None,
        created_at="2026-04-26T00:00:00Z", last_retrieved_at=None,
    )


def _seed(db: Path, *, n_locks: int = 0, lock_chars: int = 150,
          n_core: int = 0, core_chars: int = 200, n_hits: int = 0,
          hit_chars: int = 400) -> list[str]:
    """Seed a store; return the lock ids in insertion order."""
    locks: list[str] = []
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            bid = f"L{i:031d}"
            store.insert_belief(
                _mk(bid, "lockword " + "q" * lock_chars, locked=True))
            locks.append(bid)
        for i in range(n_core):
            store.insert_belief(_mk(
                f"C{i:031d}", "coreword " + "w" * core_chars,
                alpha=4.0, beta=1.0))
        for i in range(n_hits):
            store.insert_belief(
                _mk(f"H{i:031d}", f"{_WORD} fact " + "z" * hit_chars))
    finally:
        store.close()
    return locks


def _fire_ups(tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
              prompt: str = _PROMPT, session_id: str = "s1") -> tuple[str, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": session_id, "transcript_path": "/dev/null",
        "cwd": str(tmp_path), "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
    })
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr) == 0
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def _fire_session_start(tmp_path: Path, db: Path,
                        monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": "s-start", "transcript_path": "/dev/null",
        "cwd": str(tmp_path), "hook_event_name": "SessionStart",
    })
    assert session_start(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr) == 0
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def test_the_bound_is_pinned() -> None:
    assert HOOK_PAYLOAD_CHAR_LIMIT == _LIMIT


# --- F1 / F2: the whole payload fits ------------------------------------

@pytest.mark.timeout(60)
@pytest.mark.parametrize("prompt", [_PROMPT, _GATED_PROMPT],
                         ids=["retrieval", "gate-skip"])
def test_ups_fits_when_locks_alone_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str,
) -> None:
    db = tmp_path / "a.db"
    _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_ups(tmp_path, db, monkeypatch, prompt=prompt)
    assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(60)
def test_ups_fits_on_a_mixed_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "c.db"
    _seed(db, n_locks=20, n_core=20, core_chars=2000, n_hits=20)
    for session in ("s1", "s1"):  # first prompt, then turn two
        out, _ = _fire_ups(tmp_path, db, monkeypatch, session_id=session)
        assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("fixture", ["locks", "mixed"])
def test_session_start_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture: str,
) -> None:
    db = tmp_path / "s.db"
    if fixture == "locks":
        _seed(db, n_locks=300, lock_chars=150)
    else:
        _seed(db, n_locks=20, n_core=20, core_chars=2000, n_hits=20)
    out, _ = _fire_session_start(tmp_path, db, monkeypatch)
    assert len(out) <= _LIMIT, len(out)


def _load_producer():
    spec = importlib.util.spec_from_file_location(
        "measure_block_ceiling", _ROOT / "scripts" / "measure_block_ceiling.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_block_ceiling"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(120)
def test_a_cadence_fire_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every default-off writer live at once: cadence plus both phantom notes.

    The same store and config the producer's `--cadence` arm fires, which
    measured the whole payload at about four times the host limit before
    #1639.
    """
    mbc = _load_producer()
    work = tmp_path / "cadence"
    work.mkdir()
    db = mbc._cadence_store(work)
    transcript = mbc._cadence_transcript(work, mbc.CADENCE_SESSION)
    (work / ".aelfrice.toml").write_text(
        "[cadence]\nenabled = true\npolicy = \"p1_every_k_turns\"\n"
        f"k = {mbc.CADENCE_K}\n[phantom_generation]\nenabled = true\n"
        "[phantom_promotion]\nenabled = true\n", encoding="utf-8")
    (db.parent / "session_injected_ids.json").write_text(json.dumps({
        "session_id": mbc.CADENCE_SESSION, "ring": [], "ring_max": 200,
        "next_fire_idx": mbc.CADENCE_K, "evicted_total": 0,
    }), encoding="utf-8")
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": mbc.CADENCE_SESSION, "transcript_path": str(transcript),
        "cwd": str(work), "hook_event_name": "UserPromptSubmit",
        "prompt": mbc.CADENCE_PROMPT,
    })
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr) == 0
    out = sout.getvalue()
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    # The checkpoint still ships -- packed smaller, not skipped.
    assert "<cadence-checkpoint" in out and "</cadence-checkpoint>" in out
    assert len(out) <= _LIMIT, len(out)


# --- F3: locks render first ---------------------------------------------

@pytest.mark.timeout(60)
@pytest.mark.parametrize("hook", ["ups-first", "ups-gated", "session-start"])
def test_the_first_belief_is_a_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str,
) -> None:
    db = tmp_path / "f3.db"
    _seed(db, n_locks=5, n_core=20, core_chars=2000, n_hits=20)
    if hook == "session-start":
        out, _ = _fire_session_start(tmp_path, db, monkeypatch)
    else:
        out, _ = _fire_ups(tmp_path, db, monkeypatch,
                           prompt=_PROMPT if hook == "ups-first" else _GATED_PROMPT)
    first = _ANY_BELIEF_RE.search(out)
    assert first is not None
    assert out[first.start():].startswith("<belief ") and \
        'lock="user"' in out[first.start():out.index(">", first.start())]


# --- F4: an overflow of locks is visible, not silent --------------------

@pytest.mark.timeout(60)
@pytest.mark.parametrize("hook", ["ups", "session-start"])
def test_locks_that_do_not_fit_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str,
) -> None:
    db = tmp_path / "f4.db"
    locks = _seed(db, n_locks=300, lock_chars=150)
    if hook == "ups":
        out, err = _fire_ups(tmp_path, db, monkeypatch)
    else:
        out, err = _fire_session_start(tmp_path, db, monkeypatch)
    shown = _LOCK_RE.findall(out)
    omitted = [b for b in locks if b not in shown]
    assert shown and omitted, (len(shown), len(omitted))
    # The shown locks are a prefix of the render order, cut at an element.
    assert shown == [b for b in locks if b in shown][: len(shown)]
    pointer = [ln for ln in out.splitlines()
               if "user lock(s) did not fit" in ln]
    assert len(pointer) == 1, pointer
    assert "aelf locked" in pointer[0]
    assert pointer[0].startswith(f"aelfrice: {len(omitted)} user lock(s)")
    # Amended from the pre-registered "every omitted id is named" after
    # the fix was drafted: on 300 locks the full list alone fills the
    # limit. The first LOCK_POINTER_ID_CAP are named and the rest counted.
    named = omitted[:20]
    assert all(b in pointer[0] for b in named)
    if len(omitted) > 20:
        assert f"and {len(omitted) - 20} more" in pointer[0]
    # The pointer is the block's last line: nothing the model reads after
    # it can push it out of view.
    assert out.rstrip("\n").splitlines()[-1] == pointer[0]
    assert f"{len(omitted)} user lock(s)" in err


# --- the two blocks charged against the room, pinned individually -------

@pytest.mark.timeout(60)
def test_the_phantom_notes_are_charged_against_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phantom note written after the envelope still shrinks its room.

    Stubbed at 3,000 characters so the note alone is a large share of the
    bound: an envelope that ignored it would fill the whole room and push
    the fire over.
    """
    from aelfrice import hook

    note = ("<aelfrice-phantom-opportunity>\n" + "p" * 3_000
            + "\n</aelfrice-phantom-opportunity>\n")
    monkeypatch.setattr(hook, "_maybe_phantom_opportunity_block",
                        lambda **_: note)
    db = tmp_path / "ph.db"
    _seed(db, n_locks=5, n_core=20, core_chars=2000, n_hits=20)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    assert note in out
    assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(60)
def test_no_room_sheds_the_envelope_rather_than_disabling_the_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other blocks that fill the bound leave the envelope nothing.

    `enforce_block_ceiling` reads a limit of 0 as "disabled". Handing it
    the zero room literally would emit the envelope untrimmed on exactly
    the fire with the least space for it. The fire still overruns here --
    the note alone is over the bound, and nothing trims a note -- so what
    is asserted is that the envelope gave up every belief it could.
    """
    from aelfrice import hook

    note = ("<aelfrice-phantom-opportunity>\n" + "p" * _LIMIT
            + "\n</aelfrice-phantom-opportunity>\n")
    monkeypatch.setattr(hook, "_maybe_phantom_opportunity_block",
                        lambda **_: note)
    db = tmp_path / "zero.db"
    locks = _seed(db, n_locks=5, n_core=20, core_chars=2000, n_hits=20)
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    envelope = out.replace(note, "")
    assert _ANY_BELIEF_RE.search(envelope) is None, envelope[:500]
    assert f"{len(locks)} user lock(s)" in err


@pytest.mark.timeout(60)
def test_the_compact_rebuild_block_gets_the_room_the_baseline_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SessionStart(compact) writes the baseline, then the rebuild block."""
    from aelfrice import hook

    elements = "".join(
        f'<belief id="R{i:031d}" lock="none">{"r" * 400}</belief>\n'
        for i in range(40)
    )
    rebuild = f"<aelfrice-rebuild>\n{elements}</aelfrice-rebuild>"
    assert len(rebuild) > _LIMIT
    monkeypatch.setattr(hook, "_build_rebuild_block_from_payload",
                        lambda _payload: rebuild)
    db = tmp_path / "rb.db"
    _seed(db, n_locks=5)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": "s-compact", "transcript_path": "/dev/null",
        "cwd": str(tmp_path), "hook_event_name": "SessionStart",
        "source": "compact",
    })
    assert session_start(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr) == 0
    out = sout.getvalue()
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    assert "<aelfrice-rebuild>" in out
    # Cut at element boundaries: no half-element survives.
    assert out.count("<belief ") == out.count("</belief>")
    assert len(out) <= _LIMIT, len(out)
