"""#1560 and #1639: every block is bounded, and so is their sum.

**This module now pins the #1639 contract, which reverses #1560 option A.**
#1560 ruled (2026-09-17) that each block `user_prompt_submit` writes is
bounded on its own and that no bound spans two of them. #1639 reopened that
ruling for one change (operator ruling 2026-09-29): the host inlines at
most 10,000 characters of a hook's output and replaces anything longer with
a 2,000-character preview (https://code.claude.com/docs/en/hooks.md). That
cap applies to the fire's whole stdout, so per-block bounds cannot
guarantee delivery: a payload whose blocks each fit could still reach the
model as its first 2,000 characters. `HOOK_PAYLOAD_CHAR_LIMIT` (9,500) now
bounds the sum. The per-block bounds stay. The `<cadence-checkpoint>` block
packs to the room the bound leaves after the envelope's reserve, and the
memory envelope gets what is left after every other block, trimmed in the
shed order it already had.

Two contract tests carry that. One fires every writer and asserts the whole
payload fits. The other fires one store twice, cadence on and off, and
asserts the cadence block's presence takes further beliefs out of the
envelope: the exact outcome #1560's version of this module ruled out.

Which four blocks those are is itself pinned here rather than left to a
reader: `test_the_stdout_writer_enumeration_is_re_derived_from_the_source`
parses `user_prompt_submit` and compares its stdout writers against
`_STDOUT_WRITERS`, so a sixth one reds instead of quietly falsifying the
docstring's "those four are the whole of it".

**A test that checks each block separately passes under both contracts**,
which is why neither test below does that. The fixture's envelope fits the
payload bound on its own and does not fit beside the cadence block, so the
two arms differ exactly where the contracts do. The assertions are on
captured stdout from the real hook entrypoint.

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
import re
import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import DEFAULT_K
from aelfrice.context_rebuilder import RecentTurn
from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
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

# The store both contract tests fire. Sized so the envelope fits the
# #1639 payload bound on its own (6,661 characters measured, cadence off)
# and does not fit the room a cadence fire leaves it (about 4,000). Every
# lock still fits that room, so the cadence arm's cut falls on the hits,
# not on the locks. Was 40 locks under #1560, when the fixture needed the
# envelope large enough that the two blocks' sum crossed the token ceiling.
_FITS_LOCKS = 8
_FITS_HITS = 6

# The stubbed checkpoint body. Over the whole payload bound on its own, so
# the #1639 backstop in `_fit_to_room` must cut it: the stub ignores the
# token budget it is handed, which makes it stand in for the rebuilder's
# soft pack loop (#1546) at its worst. It carries no `<belief>` element, so
# the cut is the last-resort one, at a line boundary with a marker.
#
# **The margins are asserted, not just intended.**
# `test_the_emit_boundary_fixture_keeps_its_margins` holds all three
# distances at `_MIN_MARGIN_CHARS`, so the next person to move either
# literal is told rather than trusted.
_CADENCE_BODY_CHARS = 10_000
_CADENCE_BODY = (
    "CADENCE-BODY-" + "c" * (_CADENCE_BODY_CHARS - len("CADENCE-BODY-"))
)

# How much room each of the fixture's three distances must keep, in
# characters (the unit the #1639 bound counts). Chosen as a round number
# well clear of any plausible single-render drift, not derived: the point
# is that a margin exists and is checked, not its exact size.
_MIN_MARGIN_CHARS = 500
_PAYLOAD_LIMIT = 9_500


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
    # This Stop hook never writes stdout: `additionalContext` there would
    # continue the conversation (#1651), and a cache write that leaked
    # there would not be one.
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
    # #1626: the executed-command note. Bounded by
    # COMMAND_NOTE_CAP, a character length like the two phantom
    # notes above, not a token budget.
    # Two spellings of one writer, and they must be two entries: the
    # frame differs by whether the command took effect, and a table
    # carrying only one of them cannot notice the other going missing.
    # #1639 binds each frame to `command_block` so its length is charged
    # against the payload bound; the literal tag stays in the call.
    "write:command_block+command_note@<aelfrice-command-executed>": (
        "<aelfrice-command-executed>"
    ),
    "write:command_block+command_note@<aelfrice-command-failed>": (
        "<aelfrice-command-failed>"
    ),
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


_BLOCK_TAG_RE = re.compile(r"^\s*<(/?[a-z][a-z0-9-]*)>")


def _leading_block_tag(node: ast.Call) -> str | None:
    """The `<tag>` a write call opens with, when it writes a literal one.

    Returns None for a call that writes only variables, which keeps the
    pre-existing keys for those writers unchanged -- this qualifier adds
    resolution where a literal tag exists and changes nothing where one
    does not.
    """
    found: list[str] = []
    for arg in node.args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                m = _BLOCK_TAG_RE.match(sub.value)
                if m:
                    found.append(m.group(1))
    # Prefer the OPENING tag. `ast.walk` is breadth-first, so for a
    # write built by concatenation the closing tag can be reached
    # first; keying on it still separates the blocks but names the
    # wrong half in the failure message, which is the part a reader
    # acts on.
    for tag in found:
        if not tag.startswith("/"):
            return f"<{tag}>"
    return f"<{found[0]}>" if found else None


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
            key = "write:" + "+".join(written or ["<literal>"])
            # Qualified by the block's own opening tag when the call
            # carries one. Keying on the written NAMES alone collapsed
            # two distinct blocks that reuse one local into a single
            # entry -- `<aelfrice-command-executed>` and
            # `<aelfrice-command-failed>` both wrote `command_note` --
            # so the table recorded only the first tag, and a third
            # block reusing the same local was invisible to this scan.
            # That is the exact regression this gate exists to catch,
            # and it was reachable while the ceiling docstring claimed
            # otherwise.
            tag = _leading_block_tag(node)
            if tag:
                key += f"@{tag}"
            add(key, node.lineno)
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

    Each is a premise the two contract tests below rest on: the stubbed
    checkpoint body is over the payload bound on its own, so the backstop
    cut is live; the envelope fits the bound on its own, so the cadence-off
    arm is untrimmed; and the envelope does not fit the room a cadence fire
    leaves it, so the cadence-on arm must shed. The distances are asserted
    rather than the inequalities, so a drift names the fixture instead of
    looking like the contract breaking.
    """
    _stub_rebuilder(monkeypatch)
    on, _ = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=True),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="margins-on",
    )
    off, _ = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=False),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="margins-off",
    )
    cadence_block, _ = _split(on)
    room_beside_cadence = _PAYLOAD_LIMIT - len(cadence_block) - 2
    margins = {
        "cadence body over the payload bound": (
            _CADENCE_BODY_CHARS - _PAYLOAD_LIMIT
        ),
        "envelope alone under the payload bound": _PAYLOAD_LIMIT - len(off),
        "envelope alone over the room beside the cadence block": (
            len(off) - room_beside_cadence
        ),
    }
    tight = {
        name: value for name, value in margins.items()
        if value < _MIN_MARGIN_CHARS
    }
    assert not tight, (
        f"fixture margins under {_MIN_MARGIN_CHARS} characters: {tight}; "
        f"all three are {margins}. Re-size `_CADENCE_BODY_CHARS` and "
        "`_FITS_LOCKS` together — they trade against each other."
    )


def test_a_cadence_fire_fits_the_payload_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1639: every writer live, and the whole payload still fits.

    Replaces `test_payload_over_the_ceiling_is_emitted_whole_when_each_block_fits`,
    which pinned #1560 option A: a payload over the token ceiling was
    correct as long as each block fit its own bound. Under the host's
    10,000-character cap that payload reached the model as a
    2,000-character preview, so it is now the defect. The checkpoint still
    ships, cut to its room and marked; the envelope still ships, with every
    lock whole; and the sum fits.
    """
    _stub_rebuilder(monkeypatch)
    out, err = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=True),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="fits",
    )
    cadence_block, memory_block = _split(out)

    assert len(out) <= _PAYLOAD_LIMIT, len(out)
    # The checkpoint was cut, not dropped, and says so.
    assert cadence_block.endswith(_CADENCE_CLOSE)
    assert "CADENCE-BODY-" in cadence_block
    assert _CADENCE_BODY not in out
    assert "cut to fit the hook output limit" in cadence_block
    # The envelope shipped, and every lock is in it whole.
    assert memory_block.startswith(_MEMORY_OPEN)
    missing = [
        i for i in range(_FITS_LOCKS)
        if f'<belief id="L{i:031d}"' not in memory_block
    ]
    assert not missing, missing
    assert "did not fit" not in err, err
    assert "still over the" not in err, err


def test_a_cadence_block_takes_further_beliefs_from_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1639: the shed now reaches across blocks, in the existing order.

    Replaces `test_the_ceiling_sheds_the_same_bytes_with_and_without_a_cadence_block`,
    which required the envelope byte-identical across the two arms: the
    #1560 option A property "the bytes the ceiling deletes cannot depend on
    whether a sibling block shares the payload". #1639 reverses exactly
    that. The same store fired with cadence off emits every hit untrimmed;
    fired with cadence on, the envelope sheds hits to make room, keeps every
    lock, and the hits it keeps are a subset of the ones the off arm kept.
    No new shed order: this fixture has no `<core>` or recap, so the hits
    are the lane the existing order reaches.
    """
    _stub_rebuilder(monkeypatch)
    on, err_on = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=True),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="on",
    )
    off, err_off = _fire(
        tmp_path, monkeypatch,
        config=_cadence_toml(enabled=False),
        n_locks=_FITS_LOCKS, n_hits=_FITS_HITS, name="off",
    )
    assert _CADENCE_OPEN not in off
    _, memory_on = _split(on)

    def ids(block: str, prefix: str, n: int) -> set[int]:
        return {i for i in range(n) if f'<belief id="{prefix}{i:031d}"' in block}

    hits_off = ids(off, "H", _FITS_HITS)
    hits_on = ids(memory_on, "H", _FITS_HITS)
    # The off arm is untrimmed: every hit, and nothing shed.
    assert hits_off == set(range(_FITS_HITS)), hits_off
    assert "dropped" not in err_off, err_off
    # The on arm shed hits because the cadence block shares the bound.
    assert hits_on < hits_off, (hits_on, hits_off)
    assert "dropped" in err_on, err_on
    # Locks are not what went.
    assert ids(memory_on, "L", _FITS_LOCKS) == set(range(_FITS_LOCKS))
    assert len(on) <= _PAYLOAD_LIMIT, len(on)


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
