"""#1560: the payload is bounded per block, and nothing bounds their sum.

The ruling these tests pin is the one `HOOK_BLOCK_TOKEN_CEILING`'s
docstring states: every block `user_prompt_submit` writes is bounded on
its own, and no bound spans two of them. The rejected alternative was a
single payload ceiling with the `<cadence-checkpoint>` block shedding
first.

Which four blocks those are is itself pinned here rather than left to a
reader: `test_the_stdout_writer_enumeration_is_re_derived_from_the_source`
parses `user_prompt_submit` and compares its stdout writers against
`_STDOUT_WRITERS`, so a fifth one reds instead of quietly falsifying the
docstring's "those four are the whole of it".

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
install never writes the second block. The two contract tests enable
cadence explicitly through `.aelfrice.toml`; the reachability tests
*omit* the `[cadence]` section, and also fire with no config file at
all, because writing `enabled = false` pins an explicit off rather than
the shipped default — and their fire index is a P1 boundary under the
shipped `k` as well as this module's, so the silence is the flags' doing
rather than a missed boundary.

**Reachability has a Stop half, and it is fired rather than inferred.**
The resume cache the #871 recap reads is written by
`_maybe_fire_cadence_checkpoint`, which only the Stop hook calls, so a
UPS fire finding no `<cadence-resume>` in its envelope cannot say why:
it never ran the code that would have created the file. The stock-install
Stop test calls `stop()` on both stock spellings and asserts the path
`_cadence_resume_cache_path` resolves does not exist afterwards, and a
cadence-enabled control fires the same helper and finds the file
written — otherwise an absence would be evidence of a fixture no policy
fires on rather than of the default being off.

A third test guards the fixture itself: the two literals below sit
between three bounds, and the distances are asserted rather than
intended.

The cadence body is stubbed rather than rebuilt, because these tests are
about what the emit boundary does with a block of a known size, not about
what the rebuilder packs into one. The measured sizes of the real blocks
are `scripts/measure_block_ceiling.py --cadence`.

What this module does *not* cover is the `<cadence-resume>` recap (#871),
which rides inside the `<aelfrice-memory>` envelope rather than beside
it: `test_hook_recap_shed_order_1564.py` covers that, and the
reachability tests here assert a stock install gets neither the recap nor
the cache it would have been read from.
"""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import DEFAULT_K
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
# A fire index both this module's `k` and the shipped default divide, so
# the P1 boundary is reached whichever `k` is in force. The product is
# the cheap way to stay divisible by both if either constant moves.
_FIRE_IDX = _K * DEFAULT_K

# The store the first test fires: small enough that the memory block
# stays under its ceiling untrimmed, so "emitted whole" is a claim about
# the payload rather than about what survived a trim.
_FITS_LOCKS = 40
_FITS_HITS = 6

# Sized so the checkpoint block lands inside
# `DEFAULT_REBUILDER_TOKEN_BUDGET`, wrapper tags included, at the shipped
# 4-chars-per-token estimator. The tests assert that containment rather
# than trusting the arithmetic, but the literal is chosen for it: a body
# over the budget would make the "each block is inside its own bound"
# premise false, and the payload under test would no longer be the one
# the ruling is about.
#
# **The margins are asserted, not just intended.** This constant used to
# sit close enough to the rebuilder budget that a render change of a few
# hundred characters would have crossed it — a contract test one edit
# away from being a flake, and nothing said so. `_FITS_LOCKS` grew when
# this shrank: the first test needs the two blocks' *sum* over the block
# ceiling while each stays inside its own bound, so buying headroom in
# the cadence body is paid for by a larger envelope.
# `test_the_emit_boundary_fixture_keeps_its_margins` holds all three
# distances at `_MIN_MARGIN_TOKENS`, so the next person to move either
# literal is told rather than trusted.
_CADENCE_BODY_CHARS = 10_000
_CADENCE_BODY = (
    "CADENCE-BODY-" + "c" * (_CADENCE_BODY_CHARS - len("CADENCE-BODY-"))
)

# How much room each of the fixture's three distances must keep: the
# cadence block under the rebuilder budget, the memory block under the
# block ceiling, and their sum over the block ceiling. Chosen as a round
# number well clear of any plausible single-render drift, not derived —
# the point is that a margin exists and is checked, not its exact size.
_MIN_MARGIN_TOKENS = 500


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


def _cadence_toml(*, enabled: bool) -> str:
    """The `[cadence]` section, switched on or explicitly off."""
    return (
        "[cadence]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        'policy = "p1_every_k_turns"\n'
        f"k = {_K}\n"
    )


# A config that omits the `[cadence]` section entirely, and the absence of
# a config file at all. Both are what a stock install looks like; neither
# is `enabled = false`, which pins a setting rather than the default.
_NO_CADENCE_SECTION = "[retrieval]\ntoken_budget = 1500\n"
_NO_CONFIG_FILE = None


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: str | None,
    n_locks: int,
    n_hits: int,
    name: str,
) -> Path:
    """Seed one work directory and point `AELFRICE_DB` at its store.

    `config` is the `.aelfrice.toml` to write, or None to write no config
    file at all. The returned directory is the store's parent, which is
    also where `_cadence_resume_cache_path` resolves the resume cache.
    """
    work = tmp_path / name
    work.mkdir()
    db = work / "memory.db"
    _seed(db, n_locks=n_locks, n_hits=n_hits)
    if config is not None:
        (work / ".aelfrice.toml").write_text(config, encoding="utf-8")
    # The ring both cadence dispatchers read their fire index from —
    # `_maybe_run_ups_cadence_checkpoint` on the UPS side and
    # `_maybe_fire_cadence_checkpoint` on the Stop side. Both this
    # module's `k` and the shipped `DEFAULT_K` divide it, so the P1
    # policy says fire under either — which is what makes the
    # stock-install arms' silence attributable to the default flags
    # rather than to a fire index that happened to miss the boundary.
    (db.parent / "session_injected_ids.json").write_text(
        json.dumps({
            "session_id": "sess",
            "ring": [],
            "ring_max": 200,
            "next_fire_idx": _FIRE_IDX,
            "evicted_total": 0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return work


def _fire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: str | None,
    n_locks: int,
    n_hits: int,
    name: str,
) -> tuple[str, str]:
    """One real `user_prompt_submit` fire; return its stdout and stderr."""
    work = _prepare(
        tmp_path, monkeypatch,
        config=config, n_locks=n_locks, n_hits=n_hits, name=name,
    )
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


def _stop_fire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: str | None,
    name: str,
) -> tuple[Path, str]:
    """One real `stop()` fire; return the resume-cache path and stderr.

    The path is resolved by the shipped `_cadence_resume_cache_path`
    rather than rebuilt here, so a move of the cache file cannot leave
    this asserting the absence of something at an address nothing writes.
    """
    work = _prepare(
        tmp_path, monkeypatch,
        config=config, n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name=name,
    )
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": "sess",
        "transcript_path": "/dev/null",
        "cwd": str(work),
        "hook_event_name": "Stop",
    })
    rc = hook.stop(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    assert rc == 0
    # Stop fails soft too, so an exception inside it becomes a stderr
    # trace and rc 0 — indistinguishable from a policy that declined.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    # Stop has no `additionalContext` channel; nothing it does belongs on
    # stdout, and a cache write that leaked there would not be one.
    assert sout.getvalue() == "", sout.getvalue()[:200]
    cache = hook._cadence_resume_cache_path()
    assert cache is not None
    return cache, serr.getvalue()


# ---------------------------------------------------------------------------
# The writer enumeration, re-derived rather than trusted
# ---------------------------------------------------------------------------

# Every stdout writer in `user_prompt_submit`, keyed by the site shape
# `_stdout_writer_sites` reports, and valued by the block it emits. The
# ceiling docstring's "those four are the whole of what
# `user_prompt_submit` sends to stdout" is this table, and a fifth writer
# is a key this table does not carry.
_STDOUT_WRITERS = {
    "call:_write_memory_block": "<aelfrice-memory>",
    "write:cadence_checkpoint_block": "<cadence-checkpoint>",
    "write:phantom_block": "<aelfrice-phantom-opportunity>",
    "write:promotion_block": "<aelfrice-phantom-promotion-opportunity>",
}

# The stream expressions `user_prompt_submit` starts with: its own
# `stdout` parameter, and the process stream it falls back to.
_STREAM_ROOTS = frozenset({"stdout", "sys.stdout"})
_WRITE_METHODS = frozenset({"write", "writelines"})


def _ups_function() -> ast.FunctionDef:
    """`user_prompt_submit` as parsed from the shipped source file."""
    tree = ast.parse(Path(hook.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "user_prompt_submit"
        ):
            return node
    raise AssertionError("user_prompt_submit not found in hook.py")


def _stream_names(fn: ast.FunctionDef) -> set[str]:
    """The names inside `fn` that hold the stdout stream.

    Seeded with `_STREAM_ROOTS` and grown to a fixed point over plain
    aliasing assignments only — `x = <stream>` and
    `x = <stream> if ... else <stream>`, which is the shape
    `sout = stdout if stdout is not None else sys.stdout` has. A value
    that merely *mentions* the stream (`outcome = f(stdout=sout)`) is not
    an alias, and admitting it would make every result downstream of a
    write look like another stream.
    """
    names = set(_STREAM_ROOTS)

    def is_stream(node: ast.expr) -> bool:
        return (
            isinstance(node, (ast.Name, ast.Attribute))
            and ast.unparse(node) in names
        )

    for _ in range(len(list(ast.walk(fn)))):
        grown = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if isinstance(value, ast.IfExp):
                ok = is_stream(value.body) and is_stream(value.orelse)
            else:
                ok = is_stream(value)
            if not ok:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    grown = True
        if not grown:
            break
    return names


def _stdout_writer_sites() -> dict[str, list[int]]:
    """Every site in `user_prompt_submit` that can reach stdout.

    Parsed rather than grepped: a substring scan cannot tell a write from
    the same identifier in one of the comments around it, and the
    enumeration this pins is a claim about the function body.

    Three shapes reach the stream:

    * a write method called on it, `sout.write(...)`, keyed by the names
      in the written expression;
    * a call handed it as an argument, `_write_memory_block(...,
      stdout=sout, ...)`, keyed by the callee;
    * `print(...)` with no `file=`, which goes to stdout by default, or
      with a `file=` naming the stream.

    Returns site key -> the line numbers carrying it, so a failure names
    where to look.
    """
    fn = _ups_function()
    names = _stream_names(fn)
    sites: dict[str, list[int]] = {}

    def add(key: str, lineno: int) -> None:
        sites.setdefault(key, []).append(lineno)

    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _WRITE_METHODS
            and ast.unparse(func.value) in names
        ):
            written = sorted({
                sub.id for arg in node.args for sub in ast.walk(arg)
                if isinstance(sub, ast.Name)
            })
            add("write:" + "+".join(written or ["<literal>"]), node.lineno)
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if isinstance(func, ast.Name) and func.id == "print":
            target = keywords.get("file")
            if target is None or ast.unparse(target) in names:
                add("print:" + ast.unparse(node)[:60], node.lineno)
            continue
        passed = [
            arg for arg in list(node.args) + list(keywords.values())
            if isinstance(arg, (ast.Name, ast.Attribute))
            and ast.unparse(arg) in names
        ]
        if passed:
            add("call:" + ast.unparse(func), node.lineno)
    return sites


def test_the_stdout_writer_enumeration_is_re_derived_from_the_source() -> None:
    """A fifth stdout writer reds here rather than rotting a docstring.

    `HOOK_BLOCK_TOKEN_CEILING`'s docstring names four blocks and says they
    are the whole of what `user_prompt_submit` writes to stdout. That was
    true when written and nothing held it there: the only way to check it
    was to read the function. This re-derives the set from the function's
    own AST, so adding a writer without adding it to `_STDOUT_WRITERS` —
    and to the docstring `_STDOUT_WRITERS` mirrors — fails.

    The stream is resolved by aliasing rather than by the name `sout`, so
    renaming the local does not silently empty this.
    """
    sites = _stdout_writer_sites()
    assert sites, (
        "no stdout writer sites found in user_prompt_submit — either the "
        "function stopped writing to stdout, or this scan stopped being "
        "able to see it, and in both cases the comparison below is vacuous"
    )
    unexpected = {k: v for k, v in sites.items() if k not in _STDOUT_WRITERS}
    assert not unexpected, (
        f"user_prompt_submit writes to stdout at {unexpected}, which the "
        "four-writer enumeration does not carry. Add the block to "
        "`_STDOUT_WRITERS` here and to `HOOK_BLOCK_TOKEN_CEILING`'s "
        "docstring, which says these four are the whole of it."
    )
    missing = sorted(set(_STDOUT_WRITERS) - set(sites))
    assert not missing, (
        f"{missing} is enumerated here and in the ceiling docstring but no "
        "longer writes to stdout in user_prompt_submit"
    )


def test_the_writer_scan_follows_the_stream_alias() -> None:
    """The scan's own premise, asserted rather than assumed.

    `_stdout_writer_sites` finds three of the four writers only because
    `_stream_names` resolves the local the parameter is assigned to. If
    that resolution silently returned the seed set, the scan would still
    find `_write_memory_block` — it takes `stdout=` by keyword — and
    would report three writers missing rather than a hole, which is a
    confusing failure for the wrong reason.
    """
    names = _stream_names(_ups_function())
    aliases = names - _STREAM_ROOTS
    assert aliases, (
        "no local alias of the stdout parameter was resolved; the writer "
        "scan can only see calls that name `stdout` or `sys.stdout`"
    )
    # An alias-only assignment is followed; a call result is not.
    assert "outcome" not in names, sorted(names)


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


def test_the_emit_boundary_fixture_keeps_its_margins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture's three distances are not near their boundaries.

    The test below asserts three inequalities; each of them is a premise
    the ruling's comparison rests on, and a fixture sitting a few dozen
    tokens from any of them is a flake waiting for a render change rather
    than a contract. This one asserts the *distance* instead of the
    inequality, so the failure a drift produces names the fixture rather
    than looking like the contract breaking.
    """
    _stub_rebuilder(monkeypatch)
    out, _ = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=True),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="margins",
    )
    cadence_block, memory_block = _split(out)
    margins = {
        "cadence block under the rebuilder budget": (
            DEFAULT_REBUILDER_TOKEN_BUDGET
            - _audit_tokens_from_block(cadence_block)
        ),
        "memory block under the block ceiling": (
            HOOK_BLOCK_TOKEN_CEILING - _audit_tokens_from_block(memory_block)
        ),
        "payload over the block ceiling": (
            _audit_tokens_from_block(out) - HOOK_BLOCK_TOKEN_CEILING
        ),
    }
    tight = {
        name: value for name, value in margins.items()
        if value < _MIN_MARGIN_TOKENS
    }
    assert not tight, (
        f"fixture margins under {_MIN_MARGIN_TOKENS} tokens: {tight}; "
        f"all three are {margins}. Re-size `_CADENCE_BODY_CHARS` and "
        "`_FITS_LOCKS` together — they trade against each other."
    )


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
        config=_cadence_toml(enabled=True),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="fits",
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
    # Every seeded hit is still in the envelope. An `in` on one id would
    # pass a payload that had shed most of the lane to fit a bound.
    missing = [
        i for i in range(_FITS_HITS)
        if f'<belief id="H{i:031d}"' not in memory_block
    ]
    assert not missing, missing
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
        config=_cadence_toml(enabled=True),
        n_locks=60, n_hits=20, name="on",
    )
    off, err_off = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=False),
        n_locks=60, n_hits=20, name="off",
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


@pytest.mark.parametrize(
    ("label", "config"),
    [
        ("no-section", _NO_CADENCE_SECTION),
        ("no-file", _NO_CONFIG_FILE),
    ],
)
def test_the_cadence_fire_is_off_on_a_stock_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, config: str | None,
) -> None:
    """The exposure is cadence-enabled-only, and this says so.

    Without `[cadence] enabled` the second block is never written, so the
    payload the two tests above construct is unreachable on a default
    install. The `_stub_rebuilder` patch is applied so the absence is the
    flag's doing rather than an empty rebuild window's.

    **The section is omitted, not set to `false`.** This test used to
    write `enabled = false`, which pins what an explicit off does and
    says nothing about the shipped default — a default flipped to on
    would have left it green. Both stock spellings are fired: a config
    file with no `[cadence]` section, and no config file at all. The
    first is what separates "the section is absent" from "the file is
    absent"; the second is what a fresh checkout actually looks like.
    """
    _stub_rebuilder(monkeypatch)
    out, err = _fire(
        tmp_path, monkeypatch, config=config,
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name=f"stock-{label}",
    )
    assert _CADENCE_OPEN not in out
    assert _CADENCE_BODY not in out
    assert "ups cadence checkpoint" not in err
    assert out.startswith(_MEMORY_OPEN)
    # No #871 recap rides the envelope either. That is all this arm
    # shows: it never runs the Stop side, so it cannot say why the cache
    # the recap would have come from is missing.
    # `test_a_stock_install_writes_no_resume_cache_on_stop` runs it.
    assert "<cadence-resume" not in out


def test_the_stop_side_writes_a_resume_cache_when_cadence_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the two arms below.

    A test that fires `stop()` and finds no cache proves nothing on its
    own — the fixture might simply be one no cadence policy would fire
    on, whatever the flags said. So this arm enables cadence at a P1
    boundary the shipped `k` also divides and shows the write happening,
    with the stubbed body round-tripped through the file: the harness
    reaches the writer, and what the stock arms are missing is the flag.
    """
    _stub_rebuilder(monkeypatch)
    cache, err = _stop_fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=True), name="stop-on",
    )
    assert cache.exists(), err
    record = json.loads(cache.read_text(encoding="utf-8"))
    assert record["body"] == _CADENCE_BODY
    assert record["session_id"] == "sess"
    assert "cadence checkpoint fired" in err, err


@pytest.mark.parametrize(
    ("label", "config"),
    [
        ("no-section", _NO_CADENCE_SECTION),
        ("no-file", _NO_CONFIG_FILE),
    ],
)
def test_a_stock_install_writes_no_resume_cache_on_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, config: str | None,
) -> None:
    """The Stop half of the reachability claim, run rather than inferred.

    `HOOK_BLOCK_TOKEN_CEILING`'s docstring says a stock install gets no
    #871 recap *because* no Stop-side fire writes a resume cache. The UPS
    arm above can only show the recap's absence from one envelope; the
    cache is written by `_maybe_fire_cadence_checkpoint`, which only the
    Stop hook calls. This fires that hook on the same two stock spellings
    and asserts the file the recap reads was never created.
    """
    _stub_rebuilder(monkeypatch)
    cache, err = _stop_fire(
        tmp_path, monkeypatch, config=config, name=f"stop-stock-{label}",
    )
    assert not cache.exists(), cache.read_text(encoding="utf-8")[:400]
    assert "cadence checkpoint fired" not in err, err
    assert "cadence resume cache write failed" not in err, err
