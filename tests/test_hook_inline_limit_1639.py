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


# --- review round 1 ------------------------------------------------------

def _seed_reference_locks(db: Path, n: int) -> list[str]:
    """Reference-tier locks: each renders as a one-line `ref` entry."""
    from dataclasses import replace

    from aelfrice.models import LOCK_TIER_REFERENCE

    ids: list[str] = []
    store = MemoryStore(str(db))
    try:
        for i in range(n):
            bid = f"R{i:031d}"
            b = _mk(bid, f"reference rule {i} " + "t" * 400, locked=True)
            store.insert_belief(replace(b, lock_tier=LOCK_TIER_REFERENCE))
            ids.append(bid)
    finally:
        store.close()
    return ids


@pytest.mark.timeout(60)
@pytest.mark.parametrize("hook", ["ups", "ups-gated", "session-start"])
def test_reference_locks_are_cut_and_named_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str,
) -> None:
    """A `ref` manifest line is a lock render; the bound cuts it like one.

    Found by review: 200 reference-tier locks wrote 25,765 characters,
    because only `<belief>` elements were cut and `ref` lines never were.
    """
    db = tmp_path / "ref.db"
    ids = _seed_reference_locks(db, 200)
    if hook == "session-start":
        out, err = _fire_session_start(tmp_path, db, monkeypatch)
    else:
        out, err = _fire_ups(tmp_path, db, monkeypatch,
                             prompt=_PROMPT if hook == "ups" else _GATED_PROMPT)
    assert len(out) <= _LIMIT, len(out)
    shown = [b for b in ids if f"ref {b}:" in out]
    assert shown and len(shown) < len(ids)
    assert "user lock(s) did not fit" in out and "user lock(s)" in err


@pytest.mark.timeout(60)
def test_a_lower_block_ceiling_still_names_the_locks_it_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator ceiling under the room must not switch omission off.

    Found by review: `AELFRICE_HOOK_BLOCK_CEILING=2000` emitted 106,565
    characters, because omission was only enabled when the room was the
    tighter of the two. Locks are cut against the payload room, never to
    meet the token ceiling alone, so the output lands under the bound and
    not necessarily under 2,000 tokens.
    """
    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "2000")
    db = tmp_path / "low.db"
    _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    assert len(out) <= _LIMIT, len(out)
    assert "user lock(s) did not fit" in out


@pytest.mark.timeout(60)
def test_the_ups_audit_row_counts_only_the_locks_it_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "aud.db"
    locks = _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    from aelfrice.hook import (
        AUDIT_HOOK_USER_PROMPT_SUBMIT,
        _audit_path_for_db,
        read_hook_audit,
    )

    row = [r for r in read_hook_audit(_audit_path_for_db(db))
           if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT][-1]
    shown = sum(1 for b in locks if f'<belief id="{b}"' in out)
    assert 0 < shown < len(locks)
    assert row["n_locked"] == shown, (row["n_locked"], shown)


@pytest.mark.timeout(60)
def test_session_start_records_only_the_locks_it_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The epoch ledger and audit row skip a lock the bound cut."""
    db = tmp_path / "ssx.db"
    locks = _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_session_start(tmp_path, db, monkeypatch)
    from aelfrice.hook import (
        AUDIT_HOOK_SESSION_START,
        _audit_path_for_db,
        read_hook_audit,
    )

    shown = {b for b in locks if f'<belief id="{b}"' in out}
    assert 0 < len(shown) < len(locks)
    row = [r for r in read_hook_audit(_audit_path_for_db(db))
           if r.get("hook") == AUDIT_HOOK_SESSION_START][-1]
    assert row["n_beliefs"] == len(shown)
    assert row["n_locked"] == len(shown)
    audited = {b["id"] for b in row["beliefs"]}
    assert audited == shown


@pytest.mark.timeout(60)
def test_the_session_start_recap_line_is_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recap line is printed after both blocks and still fits.

    Found by review: 42 recap-eligible feed rows took a SessionStart fire
    to 9,511 characters, because the line was printed uncounted.
    """
    from aelfrice import hook

    # Longer than the real line on purpose: locks are cut in ~210-character
    # steps, and a slack under one lock would absorb an 85-character line
    # whether it was charged or not.
    line = "aelfrice: recap " + "x" * 1_000
    monkeypatch.setattr(hook, "_recap_enabled", lambda *a, **k: True)
    monkeypatch.setattr(hook, "build_session_start_recap_line",
                        lambda **_: line)
    db = tmp_path / "rc.db"
    _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_session_start(tmp_path, db, monkeypatch)
    assert out.rstrip("\n").endswith(line)
    assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(30)
def test_the_search_hook_context_fits_and_sheds_l1_before_locks() -> None:
    """The PreToolUse search hook is an aelfrice hook too (AC1).

    Found by review: locks bypass this lane's 600-token budget, and 300 of
    them wrote an `additionalContext` of 55,108 characters.
    """
    from types import SimpleNamespace

    from aelfrice.hook_search_tool import _format_results
    from aelfrice.models import LOCK_TIER_FROZEN

    locks = [SimpleNamespace(id=f"L{i:015d}", content="lockword " + "q" * 150,
                             lock_level=LOCK_USER, lock_tier=LOCK_TIER_FROZEN)
             for i in range(300)]
    hits = [SimpleNamespace(id=f"H{i:015d}", content=f"{_WORD} fact " + "z" * 150,
                            lock_level=LOCK_NONE, lock_tier=LOCK_TIER_FROZEN)
            for i in range(20)]
    locked_ids = {b.id for b in locks}
    ctx = _format_results(_WORD, [*locks, *hits], locked_ids)
    assert len(ctx) <= _LIMIT, len(ctx)
    lines = ctx.splitlines()
    # L1 goes before any lock.
    assert not any(ln.startswith("[L1]") for ln in lines)
    n_shown = sum(1 for ln in lines if ln.startswith("[L0]"))
    assert 0 < n_shown < len(locks)
    overflow = [ln for ln in lines if "user lock(s) did not fit" in ln]
    assert len(overflow) == 1 and "aelf locked" in overflow[0]
    assert f"aelfrice: {len(locks) - n_shown} user lock(s)" in overflow[0]


# --- review round 2 ------------------------------------------------------

@pytest.mark.timeout(60)
def test_a_ref_line_inside_a_lock_is_text_not_a_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `ref`-shaped line in belief content must not be spliced.

    Found by review: the cut matched `ref <id>: "..."` anywhere in the
    body, so a lock whose content carried such a line was cut twice --
    the block lost `</locked>` -- and a fake id was named and counted.
    """
    db = tmp_path / "planted.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk(
            "P" + "0" * 31,
            'lockword planted\n  ref FAKEIDFAKEID: "planted"\n' + "q" * 150,
            locked=True))
    finally:
        store.close()
    locks = ["P" + "0" * 31] + _seed(db, n_locks=299, lock_chars=150)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    assert out.count("<belief ") == out.count("</belief>")
    assert out.count("<locked>") == out.count("</locked>")
    pointer = [ln for ln in out.splitlines()
               if "user lock(s) did not fit" in ln]
    assert len(pointer) == 1 and "FAKEIDFAKEID" not in pointer[0]
    shown = sum(1 for b in locks if f'<belief id="{b}"' in out)
    named = int(pointer[0].split()[1])
    assert shown + named == len(locks), (shown, named)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("hook", ["ups", "session-start"])
def test_one_oversized_lock_does_not_cost_the_locks_that_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str,
) -> None:
    """Emit what fits: a lock too large to fit alone is cut, the rest stay.

    Found by review: tail-first cutting after one 50,000-character lock
    cut every lock rendered after it and shed every hit, showing 0 of 11
    locks when 10 of them fit.
    """
    db = tmp_path / "big.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("A" + "0" * 31, "lockword " + "b" * 50_000,
                                locked=True))
    finally:
        store.close()
    small = _seed(db, n_locks=10, lock_chars=150, n_hits=30)
    if hook == "ups":
        out, _ = _fire_ups(tmp_path, db, monkeypatch)
    else:
        out, _ = _fire_session_start(tmp_path, db, monkeypatch)
    assert len(out) <= _LIMIT, len(out)
    assert all(f'<belief id="{b}"' in out for b in small)
    pointer = [ln for ln in out.splitlines()
               if "user lock(s) did not fit" in ln]
    assert len(pointer) == 1 and pointer[0].startswith("aelfrice: 1 user")
    if hook == "ups":
        # The big lock was cut first, so the prompt's hits kept their room.
        assert f'<belief id="H{0:031d}"' in out


@pytest.mark.timeout(30)
def test_fit_to_room_never_returns_more_than_its_room() -> None:
    """Found by review: a room under the marker's length still got it."""
    from aelfrice.hook import _fit_to_room

    body = "<aelfrice-rebuild>\n" + ("line of rebuild text\n" * 500)
    for room in range(0, 120):
        assert len(_fit_to_room(body, room)) <= room, room


@pytest.mark.timeout(60)
def test_the_search_hook_records_only_the_ids_it_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result line cut to fit is not injected, and is not recorded.

    Found by review: the session ring recorded every retrieved id, so the
    next fire said cut results were "already in prompt context".
    """
    from aelfrice.hook_search_tool import _do_search
    from aelfrice.session_ring import read_ring_state

    db = tmp_path / "ring.db"
    _seed(db, n_locks=300, lock_chars=150, n_hits=10, hit_chars=100)
    hit_ids = [f"H{i:031d}" for i in range(10)]
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    _do_search({
        "hook_event_name": "PreToolUse", "tool_name": "Grep",
        "tool_input": {"pattern": _WORD}, "cwd": str(tmp_path),
        "session_id": "s-ring",
    }, stdout=sout, stderr=io.StringIO())
    ctx = json.loads(sout.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert len(ctx) <= _LIMIT and "user lock(s) did not fit" in ctx
    # The L1 hits sit behind 300 locks and are cut first. The ring records
    # unlocked ids (locks always pass through), so this is where a cut
    # line recorded as shown would appear.
    ring = json.dumps(read_ring_state("s-ring"))
    # Every hit line is cut here (no L1 line survives the 300 locks), so
    # every hit id is a cut id; checked, not assumed.
    assert not any(ln.startswith("[L1]") for ln in ctx.splitlines())
    assert not [h for h in hit_ids if h in ring], "a cut hit was recorded"


@pytest.mark.timeout(30)
def test_the_search_hook_bounds_a_huge_query() -> None:
    """Found by review: a 20,000-character Grep pattern wrote 21,016."""
    from aelfrice.hook_search_tool import _format_results

    assert len(_format_results("a" * 20_000, [], set())) <= _LIMIT
    assert len(_format_results(
        "a" * 20_000, [], set(), bash_source=("grep", "b" * 20_000),
    )) <= _LIMIT


@pytest.mark.timeout(60)
def test_the_command_note_is_charged_against_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: removing the charge passed the whole suite.

    A note at `COMMAND_NOTE_CAP` beside 300 locks measured 11,341 without
    it.
    """
    from aelfrice import hook

    monkeypatch.setattr(
        hook, "execute_aelf_command",
        lambda *a, **k: hook.CommandOutcome("c" * hook.COMMAND_NOTE_CAP, True),
    )
    db = tmp_path / "cmd.db"
    _seed(db, n_locks=300, lock_chars=150)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    assert "c" * hook.COMMAND_NOTE_CAP in out
    assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(60)
def test_a_line_cut_rebuild_block_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker is the only sign a rebuild block was cut mid-structure."""
    from aelfrice import hook

    rebuild = "<aelfrice-rebuild>\n" + ("rebuild line\n" * 2_000) + "</aelfrice-rebuild>"
    monkeypatch.setattr(hook, "_build_rebuild_block_from_payload",
                        lambda _payload: rebuild)
    db = tmp_path / "mk.db"
    _seed(db, n_locks=5)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    session_start(stdin=io.StringIO(json.dumps({
        "session_id": "s-mk", "transcript_path": "/dev/null",
        "cwd": str(tmp_path), "hook_event_name": "SessionStart",
        "source": "compact",
    })), stdout=sout, stderr=serr)
    out = sout.getvalue()
    assert len(out) <= _LIMIT, len(out)
    assert out.rstrip("\n").endswith(
        "[block cut to fit the hook output limit]")


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


# --- review round 4 --------------------------------------------------------

def _add_promotable_phantoms(db: Path, n: int, content: str) -> None:
    """`n` speculative beliefs, each corroborated in three sessions."""
    from aelfrice.models import ORIGIN_SPECULATIVE

    store = MemoryStore(str(db))
    try:
        for i in range(n):
            bid = f"P{i:015d}"
            b = _mk(bid, f"{content} {i}")
            b.origin = ORIGIN_SPECULATIVE
            store.insert_belief(b)
            for j in range(3):
                store.record_corroboration(
                    bid, source_type="transcript_ingest",
                    session_id=f"sess{j}", ts=f"2026-04-2{j + 1}T00:00:00Z",
                )
    finally:
        store.close()


@pytest.mark.timeout(120)
@pytest.mark.parametrize(("n", "max_fires", "content"), [
    (32, 32, "phantom speculative claim " + "x" * 150),
    (3, 3, "&" * 170),   # escaping grows each line after truncation
], ids=["many-notes", "escaped-notes"])
def test_the_phantom_notes_cannot_eat_the_envelope_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    n: int, max_fires: int, content: str,
) -> None:
    """Found by review: 32 promotion lines took a cadence fire to 13,824.

    The notes are fitted to what the bound leaves after the envelope's
    reserve, and an opportunity that did not fit is not recorded as fired,
    so it is still owed to a later turn.
    """
    from aelfrice.session_ring import read_promotion_state

    mbc = _load_producer()
    work = tmp_path / "cadence"
    work.mkdir()
    db = mbc._cadence_store(work)
    _add_promotable_phantoms(db, n, content)
    transcript = mbc._cadence_transcript(work, mbc.CADENCE_SESSION)
    (work / ".aelfrice.toml").write_text(
        "[cadence]\nenabled = true\npolicy = \"p1_every_k_turns\"\n"
        f"k = {mbc.CADENCE_K}\n[phantom_generation]\nenabled = true\n"
        "[phantom_promotion]\nenabled = true\n"
        f"max_fires_per_session = {max_fires}\n", encoding="utf-8")
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
    assert len(out) <= _LIMIT, len(out)
    shown = re.findall(r"^- (P\d{15}):", out, flags=re.MULTILINE)
    state = read_promotion_state(mbc.CADENCE_SESSION)
    assert sorted(state["promotion_dedup"]) == sorted(shown)
    assert int(state["promotion_fires"]) == len(shown)


@pytest.mark.parametrize("room", [0, 200, 400, 800, 100_000])
def test_a_note_is_cut_to_whole_lines_and_records_only_those(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, room: int,
) -> None:
    """The fit is by whole opportunity, and only a shown one is spent."""
    from aelfrice.phantom_promotion_opportunity import (
        PhantomPromotionConfig,
        detect_promotable_phantoms,
        evaluate_promotion_opportunities,
        format_promotion_note,
    )
    from aelfrice.session_ring import read_promotion_state

    db = tmp_path / "p.db"
    # The session ring lives beside the store: pin it to this test's.
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _add_promotable_phantoms(db, 5, "claim " + "x" * 100)
    store = MemoryStore(str(db))
    try:
        every = detect_promotable_phantoms(
            store, min_corroborations=2, min_sessions=2)
        fired = evaluate_promotion_opportunities(
            store=store, session_id="s-room",
            config=PhantomPromotionConfig(
                enabled=True, min_corroborations=2, min_sessions=2,
                max_fires_per_session=5),
            room_chars=room,
        )
    finally:
        store.close()
    assert len(every) == 5
    assert len(format_promotion_note(fired)) <= room
    # The largest prefix that fits, not merely one that fits.
    fits = [k for k in range(6) if len(format_promotion_note(every[:k])) <= room]
    assert [o.belief_id for o in fired] == [o.belief_id for o in every[:max(fits)]]
    assert int(read_promotion_state("s-room")["promotion_fires"]) == len(fired)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("size", [9_000, 9_300])
def test_a_lock_that_fits_only_without_its_frame_is_cut_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int,
) -> None:
    """Found by review: the precut priced the lock alone, not beside the
    envelope's frame, so a 9,000-character lock shed every hit and was cut
    anyway. The 50,000-character case above never reached that edge.
    """
    db = tmp_path / "edge.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("A" + "0" * 31, "lockword " + "b" * size,
                                locked=True))
    finally:
        store.close()
    _seed(db, n_locks=5, lock_chars=150, n_hits=30)
    _fire_ups(tmp_path, db, monkeypatch)
    out, _ = _fire_ups(tmp_path, db, monkeypatch,
                       prompt="what else about the banana fact")
    assert len(out) <= _LIMIT, len(out)
    assert "aelfrice: 1 user lock(s) did not fit" in out
    assert f'<belief id="H{0:031d}"' in out


@pytest.mark.timeout(60)
def test_the_ceiling_note_survives_the_room_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: reporting the room pass's own flag would drop the
    note that the operator's ceiling was not met."""
    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "2000")
    db = tmp_path / "low.db"
    _seed(db, n_locks=300, lock_chars=150)
    _, err = _fire_ups(tmp_path, db, monkeypatch)
    assert "still over the 2000-token ceiling" in err


@pytest.mark.timeout(60)
def test_a_lock_cut_in_both_passes_is_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock the first pass cut is named once, and counted once on
    stderr, when the second pass cuts more."""
    db = tmp_path / "both.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("A" + "0" * 31, "lockword " + "b" * 50_000,
                                locked=True))
    finally:
        store.close()
    _seed(db, n_locks=300, lock_chars=150)
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    line = re.search(r"aelfrice: (\d+) user lock\(s\) did not fit", out)
    note = re.search(r"aelfrice hook: (\d+) user lock\(s\) did not fit", err)
    assert line is not None and note is not None
    assert line.group(1) == note.group(1)
    assert out.count("A" + "0" * 31) == 1


def _raw_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """An untrimmed UPS envelope: 40 locks and 30 hits, ceiling off."""
    from aelfrice import hook

    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "0")
    db = tmp_path / "raw.db"
    _seed(db, n_locks=40, lock_chars=150, n_hits=30)
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    start = out.index(hook.OPEN_TAG)
    end = out.index(hook.CLOSE_TAG) + len(hook.CLOSE_TAG)
    return out[start:end] + "\n"


@pytest.mark.timeout(120)
def test_every_room_is_met_with_its_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line that names cut locks is paid for inside the room, both
    while hits are shed and when a block that already fits still owes it.
    """
    from aelfrice.hook import _tokens_from_chars, enforce_block_ceiling

    body = _raw_envelope(tmp_path, monkeypatch)
    prior = ("Z" * 32,)
    # Below the frame plus the line, nothing can fit; that is the floor.
    floor = _tokens_from_chars(len(enforce_block_ceiling(
        body, 1, omit_locks=True, prior_omitted=prior).body))
    assert 300 < floor < 600, floor
    for limit in range(floor, 2_600, 23):
        for owed in ((), prior):
            got = enforce_block_ceiling(
                body, limit, omit_locks=True, prior_omitted=owed,
            ).body
            assert _tokens_from_chars(len(got)) <= limit, (limit, owed)
            assert got.count("<belief ") == got.count("</belief>")


def test_a_recap_lock_is_never_cut_as_a_lock_render() -> None:
    """A lock inside a recap goes with the recap's own rule, not the lock
    cut, or the two would splice the same span twice."""
    from aelfrice import hook
    from aelfrice.hook import CORE_CLOSE_TAG, CORE_OPEN_TAG

    pad = "y" * 900
    recap = (
        "<cadence-resume from='prev' policy='p1' ts='t'>\n"
        f'<belief id="R1" lock="none">{pad}</belief>\n'
        f'<belief id="R2" lock="user">{pad * 3}</belief>\n'
        "</cadence-resume>"
    )
    core = (f"{CORE_OPEN_TAG}\n"
            f'<belief id="C1" lock="none">{pad}</belief>\n{CORE_CLOSE_TAG}')
    body = f"{hook.OPEN_TAG}\n{recap}\n{core}\n{hook.CLOSE_TAG}\n"
    out = hook.enforce_block_ceiling(body, 300, omit_locks=True)
    assert out.omitted_lock_ids == ()
    assert out.body.count("<belief ") == out.body.count("</belief>")
    assert '<belief id="R2" lock="user">' in out.body


def test_fit_to_room_drops_whole_trailing_elements_first() -> None:
    from aelfrice.hook import _fit_to_room

    els = [f'<belief id="E{i}" lock="none">{"e" * 100}</belief>\n'
           for i in range(4)]
    body = "<aelfrice-rebuild>\n" + "".join(els) + "</aelfrice-rebuild>"
    room = len(body) - len(els[3]) - len(els[2])
    got = _fit_to_room(body, room)
    assert got == "<aelfrice-rebuild>\n" + els[0] + els[1] + "</aelfrice-rebuild>"


def test_fit_to_room_cuts_an_element_free_body_at_a_line() -> None:
    from aelfrice.hook import _fit_to_room

    body = "<aelfrice-rebuild>\n" + "".join(
        f"line number {i:03d} of rebuild text\n" for i in range(50))
    got = _fit_to_room(body, 300)
    kept, marker = got.rsplit("\n", 1)
    assert marker == "[block cut to fit the hook output limit]"
    assert body.startswith(kept + "\n")


def _cadence_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml_extra: str = "",
) -> tuple[str, str]:
    """One UPS fire on the producer's cadence store, cadence on."""
    mbc = _load_producer()
    work = tmp_path / "cadence"
    work.mkdir()
    db = mbc._cadence_store(work)
    transcript = mbc._cadence_transcript(work, mbc.CADENCE_SESSION)
    (work / ".aelfrice.toml").write_text(
        "[cadence]\nenabled = true\npolicy = \"p1_every_k_turns\"\n"
        f"k = {mbc.CADENCE_K}\n{toml_extra}", encoding="utf-8")
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
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


@pytest.mark.timeout(120)
def test_each_block_is_charged_before_the_envelope_is_trimmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every character the fire writes outside the envelope is charged
    against the envelope's room, and the notes share one allowance.

    Exact accounting rather than a total under the bound: a block charged
    a few characters short fits here and overruns on a fuller store.
    """
    from aelfrice import hook

    seen: dict[str, int] = {}
    phantom = "<aelfrice-phantom-opportunity>\n" + "p" * 300 + "\n"

    def _phantom(**kw: object) -> str:
        seen["phantom_room"] = int(kw["room_chars"])  # type: ignore[arg-type]
        return phantom

    real_promo = hook._maybe_phantom_promotion_block  # pyright: ignore[reportPrivateUsage]

    def _promo(**kw: object) -> str:
        seen["promo_room"] = int(kw["room_chars"])  # type: ignore[arg-type]
        return real_promo(**kw)  # type: ignore[arg-type]

    real_write = hook._write_memory_block  # pyright: ignore[reportPrivateUsage]

    def _write(body: str, **kw: object) -> object:
        seen["envelope_room"] = int(kw["room_chars"])  # type: ignore[arg-type]
        return real_write(body, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook, "_maybe_phantom_opportunity_block", _phantom)
    monkeypatch.setattr(hook, "_maybe_phantom_promotion_block", _promo)
    monkeypatch.setattr(hook, "_write_memory_block", _write)
    # A plain fire with a full envelope: a cadence checkpoint takes all but
    # the envelope's reserve, which leaves the notes almost no room.
    db = tmp_path / "notes.db"
    _seed(db, n_locks=300, lock_chars=150, n_hits=10)
    _add_promotable_phantoms(db, 3, "claim " + "x" * 60)
    (tmp_path / ".aelfrice.toml").write_text(
        "[phantom_promotion]\nenabled = true\n", encoding="utf-8")
    out, _ = _fire_ups(tmp_path, db, monkeypatch)
    before = out.index(hook.OPEN_TAG)
    promotion = out[out.index(hook.CLOSE_TAG) + len(hook.CLOSE_TAG):]
    promotion = promotion.split(phantom, 1)[1]
    assert "<aelfrice-phantom-promotion-opportunity>" in promotion
    assert seen["promo_room"] == seen["phantom_room"] - len(phantom)
    assert seen["phantom_room"] == (
        _LIMIT - before - hook.CADENCE_ENVELOPE_RESERVE_CHARS)
    assert seen["envelope_room"] == (
        _LIMIT - before - len(phantom) - len(promotion))
    assert len(out) <= _LIMIT, len(out)


@pytest.mark.timeout(120)
def test_the_checkpoint_is_packed_to_its_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rebuilder packs to the checkpoint's room before the cut, so the
    cut is a backstop and not the whole bound (#1546 makes it soft)."""
    from aelfrice import hook

    budgets: list[int] = []
    real = hook._rebuild_and_format  # pyright: ignore[reportPrivateUsage]

    def _spy(recent: object, token_budget: int, **kw: object) -> object:
        budgets.append(token_budget)
        return real(recent, token_budget, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook, "_rebuild_and_format", _spy)
    _cadence_fire(tmp_path, monkeypatch)
    assert budgets, "the cadence fire did not rebuild"
    assert all(4 * b <= _LIMIT - hook.CADENCE_ENVELOPE_RESERVE_CHARS
               for b in budgets), budgets


def _add_phantoms_with(db: Path, contents: list[str]) -> None:
    from aelfrice.models import ORIGIN_SPECULATIVE

    store = MemoryStore(str(db))
    try:
        for i, content in enumerate(contents):
            bid = f"P{i:015d}"
            b = _mk(bid, content)
            b.origin = ORIGIN_SPECULATIVE
            store.insert_belief(b)
            for j in range(3):
                store.record_corroboration(
                    bid, source_type="transcript_ingest",
                    session_id=f"sess{j}", ts=f"2026-04-2{j + 1}T00:00:00Z",
                )
    finally:
        store.close()


def test_a_note_stops_at_the_first_opportunity_that_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In order, not first-fit: a later, shorter opportunity does not jump
    ahead of one that did not fit, so the oldest stays owed first."""
    from aelfrice.phantom_promotion_opportunity import (
        PhantomPromotionConfig,
        detect_promotable_phantoms,
        evaluate_promotion_opportunities,
        format_promotion_note,
    )

    db = tmp_path / "order.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _add_phantoms_with(db, ["short one", "long " + "x" * 150, "short two"])
    store = MemoryStore(str(db))
    try:
        every = detect_promotable_phantoms(
            store, min_corroborations=2, min_sessions=2)
        room = len(format_promotion_note([every[0], every[2]]))
        assert len(format_promotion_note(every[:2])) > room
        fired = evaluate_promotion_opportunities(
            store=store, session_id="s-order",
            config=PhantomPromotionConfig(
                enabled=True, min_corroborations=2, min_sessions=2,
                max_fires_per_session=5),
            room_chars=room,
        )
    finally:
        store.close()
    assert [o.belief_id for o in fired] == [every[0].belief_id]


@pytest.mark.parametrize("room", [0, 250, 400, 10**6])
def test_the_phantom_note_fits_its_room_and_records_only_what_it_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, room: int,
) -> None:
    from aelfrice.phantom_trigger import (
        PhantomGenerationConfig,
        evaluate_opportunities,
        format_opportunity_note,
    )
    from aelfrice.session_ring import read_phantom_state

    db = tmp_path / "gen.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    MemoryStore(str(db)).close()
    prompt = ("Why does QuetzalRouter fail in src/zanzibar/xylo.py and "
              "MarimbaCache on v9.8.7?")
    store = MemoryStore(str(db))
    try:
        fired = evaluate_opportunities(
            prompt=prompt, store=store, session_id=f"s-gen-{room}",
            hit_count=0,
            config=PhantomGenerationConfig(enabled=True,
                                           max_fires_per_session=10),
            room_chars=room,
        )
    finally:
        store.close()
    note = format_opportunity_note(fired)
    assert len(note) <= room
    if room == 10**6:
        assert len(fired) >= 3
    else:
        assert len(fired) < 3
    state = read_phantom_state(f"s-gen-{room}")
    assert int(state["phantom_fires"]) == len(fired)


@pytest.mark.timeout(60)
def test_a_block_that_fits_without_its_owed_line_still_pays_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.hook import _tokens_from_chars, enforce_block_ceiling

    body = _raw_envelope(tmp_path, monkeypatch)
    limit = _tokens_from_chars(len(body))
    got = enforce_block_ceiling(
        body, limit, omit_locks=True, prior_omitted=("Z" * 32,)).body
    assert "Z" * 32 in got
    assert _tokens_from_chars(len(got)) <= limit


def _envelope_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   name: str, n_locks: int, n_hits: int) -> str:
    from aelfrice import hook

    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "0")
    db = tmp_path / f"{name}.db"
    _seed(db, n_locks=n_locks, lock_chars=150, n_hits=n_hits)
    out, _ = _fire_ups(tmp_path, db, monkeypatch, session_id=name)
    start = out.index(hook.OPEN_TAG)
    end = out.index(hook.CLOSE_TAG) + len(hook.CLOSE_TAG)
    return out[start:end] + "\n"


@pytest.mark.timeout(60)
def test_a_lone_lock_that_fits_beside_its_frame_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No other lock is cut, so no line is owed, and a lock that fits
    beside the frame stays while the hits make room for it. One character
    more and it cannot fit at all, so it is cut first and the hits stay.
    """
    from aelfrice.hook import enforce_block_ceiling

    body = _envelope_with(tmp_path, monkeypatch, "lone", 1, 30)
    hit_re = re.compile(r'<belief id="H[^"]*"[^>]*>.*?</belief>\n?', re.S)
    lock_id = f"L{0:031d}"

    def sized(text: str, s: int) -> str:
        return text.replace("lockword " + "q" * 150, "lockword " + "q" * s)

    limit = 1_500
    alone = hit_re.sub("", body)
    assert lock_id in alone and "<belief id=\"H" not in alone
    fits = 4 * limit - len(alone) + 150   # the largest lock that fits alone
    assert fits > 150
    kept = enforce_block_ceiling(sized(body, fits), limit, omit_locks=True)
    assert kept.omitted_lock_ids == ()
    assert f'<belief id="{lock_id}"' in kept.body
    cut = enforce_block_ceiling(sized(body, fits + 1), limit, omit_locks=True)
    assert cut.omitted_lock_ids == (lock_id,)
    assert f'<belief id="H{0:031d}"' in cut.body


@pytest.mark.timeout(60)
def test_the_search_hook_telemetry_counts_what_it_showed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: the counts were taken before lines were cut."""
    from aelfrice.hook_search_tool import _do_search

    db = tmp_path / "tel.db"
    _seed(db, n_locks=300, lock_chars=150, n_hits=10)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    _do_search({
        "hook_event_name": "PreToolUse", "tool_name": "Bash",
        "tool_input": {"command": f"grep -rn {_WORD} ."},
        "cwd": str(tmp_path), "session_id": "s-tel",
    }, stdout=sout, stderr=io.StringIO())
    ctx = json.loads(sout.getvalue())["hookSpecificOutput"]["additionalContext"]
    shown_l0 = sum(1 for ln in ctx.splitlines() if ln.startswith("[L0]"))
    shown_l1 = sum(1 for ln in ctx.splitlines() if ln.startswith("[L1]"))
    row = json.loads((tmp_path / "telemetry" / "search_tool_hook.jsonl")
                     .read_text().strip().splitlines()[-1])
    assert 0 < shown_l0 < 300
    assert (row["injected_l0"], row["injected_l1"]) == (shown_l0, shown_l1)


@pytest.mark.timeout(60)
def test_a_lock_that_fits_its_frame_but_not_the_owed_line_is_cut_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the locks together overflow, some lock is cut whatever happens,
    so the line naming it is owed. A lock that fits beside the frame but
    not beside the frame and that line cannot stay: it is cut first, and
    neither the hits nor the other locks pay for it.
    """
    from aelfrice.hook import enforce_block_ceiling

    body = _envelope_with(tmp_path, monkeypatch, "owed", 5, 30)
    hit_re = re.compile(r'<belief id="H[^"]*"[^>]*>.*?</belief>\n?', re.S)
    lock_re = re.compile(r'<belief id="L[^"]*" lock="user"[^>]*>.*?</belief>\n?',
                         re.S)
    big = f"L{0:031d}"
    seen_re = re.compile(r"^  seen L\d+: .*\n", re.M)
    alone = hit_re.sub("", body)
    # A lock's cost is its element plus the `seen` line that names it.
    costs = [len(m.group(0)) for m in lock_re.finditer(alone)]
    pointers = [len(m.group(0)) for m in seen_re.finditer(alone)]
    assert len(costs) == len(pointers) == 5
    frame = len(alone) - sum(costs) - sum(pointers)
    limit = 1_500
    # The big lock fills the room beside the frame exactly.
    s = 4 * limit - frame - (costs[0] + pointers[0]) + 150
    first = re.search(
        rf'(<belief id="{big}"[^>]*>)lockword q{{150}}(</belief>)', body)
    assert first is not None
    sized = body.replace(first.group(0),
                         first.group(1) + "lockword " + "q" * s + first.group(2))
    got = enforce_block_ceiling(sized, limit, omit_locks=True)
    assert got.omitted_lock_ids == (big,)
    assert all(f'<belief id="L{i:031d}"' in got.body for i in range(1, 5))
    assert f'<belief id="H{0:031d}"' in got.body


@pytest.mark.timeout(120)
def test_the_checkpoint_is_charged_whole_before_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint and the blank line after it are both charged: the
    envelope's room is exactly what the bound leaves after them."""
    from aelfrice import hook

    rooms: list[int] = []
    real_write = hook._write_memory_block  # pyright: ignore[reportPrivateUsage]

    def _write(body: str, **kw: object) -> object:
        rooms.append(int(kw["room_chars"]))  # type: ignore[arg-type]
        return real_write(body, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook, "_write_memory_block", _write)
    out, _ = _cadence_fire(tmp_path, monkeypatch)
    assert "</cadence-checkpoint>\n\n" in out
    # No phantom note is on: everything from the envelope on is the
    # envelope's, including the lines it writes after its close tag.
    assert "phantom" not in out[out.index(hook.OPEN_TAG):]
    assert rooms == [_LIMIT - out.index(hook.OPEN_TAG)]
