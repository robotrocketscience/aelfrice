"""#1560: the payload is bounded per block, and nothing bounds their sum.

The ruling these tests pin is the one `HOOK_BLOCK_TOKEN_CEILING`'s
docstring states: every block `user_prompt_submit` writes names its own
bound, and no bound spans them. The rejected alternative was a single
payload ceiling with the `<cadence-checkpoint>` block shedding first.

**Both options pass a test that checks each block separately**, which is
why neither test below does that. What separates them is a payload whose
blocks are each *inside* their own bound while their sum is *outside*
`HOOK_BLOCK_TOKEN_CEILING`: under the shipped contract that payload is
correct and is emitted whole, and under a payload ceiling something in it
would have been shed. So the fixture constructs exactly that payload, and
the assertions are on captured stdout from the real hook entrypoint
rather than on either block in isolation.

The second test is the shed-order half. A cadence fire and a
cadence-disabled fire are run against identically seeded stores, on a
fixture large enough that the memory block's own ceiling *does* trim, and
the emitted envelope must be byte-identical across the pair: the bytes
the ceiling deletes cannot depend on whether a sibling block shares the
payload.

**Reachability, so nothing here is read as a defect everyone is exposed
to.** `[cadence] enabled` is unset by default and
`_maybe_run_ups_cadence_checkpoint` returns None without it, so a stock
install never writes the second block. Both tests enable cadence
explicitly through `.aelfrice.toml`.

The cadence body is stubbed rather than rebuilt, because these tests are
about what the emit boundary does with a block of a known size, not about
what the rebuilder packs into one. The measured sizes of the real blocks
are `scripts/measure_block_ceiling.py --cadence`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.context_rebuilder import RecentTurn
from aelfrice.hook import (
    HOOK_BLOCK_TOKEN_CEILING,
    _audit_tokens_from_block,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.rebuild_log import DEFAULT_REBUILDER_TOKEN_BUDGET
from aelfrice.store import MemoryStore

_CEILING_ENV = "AELFRICE_HOOK_BLOCK_CEILING"
_CADENCE_OPEN = "<cadence-checkpoint>"
_CADENCE_CLOSE = "</cadence-checkpoint>"
_MEMORY_OPEN = "<aelfrice-memory>"

_WORD = "banana"
_PROMPT = f"tell me everything about the {_WORD} please"
_K = 5

# Sized so the checkpoint block lands inside
# `DEFAULT_REBUILDER_TOKEN_BUDGET`, wrapper tags included, at the shipped
# 4-chars-per-token estimator. The tests assert that containment rather
# than trusting the arithmetic, but the literal is chosen for it: a body
# over the budget would make the "each block is inside its own bound"
# premise false, and the payload under test would no longer be the one
# the ruling is about.
_CADENCE_BODY_CHARS = 15_600
_CADENCE_BODY = (
    "CADENCE-BODY-" + "c" * (_CADENCE_BODY_CHARS - len("CADENCE-BODY-"))
)


@pytest.fixture(autouse=True)
def _pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exported value may decide a result here."""
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
    ):
        monkeypatch.delenv(var, raising=False)


def _mk(bid: str, content: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-26T00:00:00Z" if locked else None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path, *, n_locks: int, n_hits: int) -> None:
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            store.insert_belief(
                _mk(f"L{i:031d}", "lockword " + "q" * 150, locked=True)
            )
        for i in range(n_hits):
            store.insert_belief(_mk(f"H{i:031d}", f"{_WORD} fact " + "z" * 400))
    finally:
        store.close()


def _stub_rebuilder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the cadence dispatch a window and a body of a known size."""
    monkeypatch.setattr(
        hook,
        "_read_recent_for_pre_compact",
        lambda _payload, _n: [RecentTurn(role="user", text="turn")],
    )
    monkeypatch.setattr(
        hook,
        "_rebuild_and_format",
        lambda recent, token_budget, **kwargs: _CADENCE_BODY,
    )


def _fire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cadence: bool,
    n_locks: int,
    n_hits: int,
    name: str,
) -> tuple[str, str]:
    """One real `user_prompt_submit` fire; return its stdout and stderr."""
    work = tmp_path / name
    work.mkdir()
    db = work / "memory.db"
    _seed(db, n_locks=n_locks, n_hits=n_hits)
    (work / ".aelfrice.toml").write_text(
        "[cadence]\n"
        f"enabled = {'true' if cadence else 'false'}\n"
        'policy = "p1_every_k_turns"\n'
        f"k = {_K}\n",
        encoding="utf-8",
    )
    # The ring `_maybe_run_ups_cadence_checkpoint` reads its fire index
    # from. `k` divides it, so the P1 policy says fire.
    (db.parent / "session_injected_ids.json").write_text(
        json.dumps({
            "session_id": "sess",
            "ring": [],
            "ring_max": 200,
            "next_fire_idx": _K,
            "evicted_total": 0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": "sess",
        "transcript_path": "/dev/null",
        "cwd": str(work),
        "hook_event_name": "UserPromptSubmit",
        "prompt": _PROMPT,
    })
    rc = user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr
    )
    assert rc == 0
    # The hook fails soft, so an exception inside it becomes a stderr
    # trace and rc 0 — indistinguishable from a small payload.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def _split(out: str) -> tuple[str, str]:
    """The `<cadence-checkpoint>` block and everything written after it.

    The memory half is taken to the end of the payload rather than to
    `</aelfrice-memory>`, because the ceiling is applied to the body
    `_write_memory_block` receives and that body carries
    `MEMORY_BLOCK_HINT` past the closing tag.
    """
    assert out.startswith(_CADENCE_OPEN), out[:200]
    end = out.index(_CADENCE_CLOSE) + len(_CADENCE_CLOSE)
    rest = out[end:]
    assert rest.startswith("\n\n"), repr(rest[:40])
    return out[:end], rest[2:]


def test_payload_over_the_ceiling_is_emitted_whole_when_each_block_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that tells the two contracts apart.

    Each block is inside its own bound and their sum is outside
    `HOOK_BLOCK_TOKEN_CEILING`. Under the shipped per-block contract that
    payload is correct: both blocks reach stdout whole, and nothing was
    shed to bring the total under a bound that does not exist. Under the
    rejected single-payload ceiling one of them would have been trimmed.
    """
    _stub_rebuilder(monkeypatch)
    out, err = _fire(
        tmp_path, monkeypatch,
        cadence=True, n_locks=20, n_hits=6, name="fits",
    )
    cadence_block, memory_block = _split(out)

    # Premise 1: each block is inside its own bound.
    assert _audit_tokens_from_block(cadence_block) <= (
        DEFAULT_REBUILDER_TOKEN_BUDGET
    )
    assert _audit_tokens_from_block(memory_block) <= HOOK_BLOCK_TOKEN_CEILING

    # Premise 2: their sum is outside the block ceiling. Without this the
    # test is satisfied by a payload neither contract disagrees about.
    assert _audit_tokens_from_block(out) > HOOK_BLOCK_TOKEN_CEILING

    # The claim: the payload is emitted whole anyway.
    assert _CADENCE_BODY in out
    assert memory_block.startswith(_MEMORY_OPEN)
    assert f'<belief id="H{0:031d}"' in memory_block
    # Nothing was trimmed and nothing overran: a payload ceiling that
    # sheds would have had to say so here, on the stream the ceiling
    # reports to.
    assert "ceiling" not in err, err


def test_the_ceiling_sheds_the_same_bytes_with_and_without_a_cadence_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shed order does not reach across blocks.

    Same store, same prompt, cadence on and off, on a fixture whose
    memory block is over its ceiling so the trim is live rather than the
    identity. The emitted envelope must be byte-identical: which beliefs
    `enforce_block_ceiling` deletes is a function of that block alone.
    Were the bound extended over the payload — the rejected option — the
    cadence block's presence would take further beliefs out of the
    envelope, or the recap would be shed to keep them.
    """
    _stub_rebuilder(monkeypatch)
    on, err_on = _fire(
        tmp_path, monkeypatch,
        cadence=True, n_locks=60, n_hits=20, name="on",
    )
    off, err_off = _fire(
        tmp_path, monkeypatch,
        cadence=False, n_locks=60, n_hits=20, name="off",
    )
    assert _CADENCE_OPEN not in off
    _, memory_on = _split(on)

    # The trim is live on this fixture, in both arms. A fixture the
    # ceiling never acts on cannot tell a per-block shed from a
    # cross-block one.
    assert "dropped" in err_on, err_on
    assert "dropped" in err_off, err_off

    assert memory_on == off
    assert _CADENCE_BODY in on
    assert _audit_tokens_from_block(on) > HOOK_BLOCK_TOKEN_CEILING


def test_the_cadence_fire_is_off_on_a_stock_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exposure is cadence-enabled-only, and this says so.

    Without `[cadence] enabled` the second block is never written, so the
    payload the two tests above construct is unreachable on a default
    install. The `_stub_rebuilder` patch is applied so the absence is the
    flag's doing rather than an empty rebuild window's.
    """
    _stub_rebuilder(monkeypatch)
    out, err = _fire(
        tmp_path, monkeypatch,
        cadence=False, n_locks=20, n_hits=6, name="stock",
    )
    assert _CADENCE_OPEN not in out
    assert _CADENCE_BODY not in out
    assert "ups cadence checkpoint" not in err
    assert out.startswith(_MEMORY_OPEN)
