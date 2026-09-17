"""`aelf show <belief-id>` — read one belief back by id (#1553).

Every injected `<belief>` element carries an id, and before this command
nothing turned that id back into the belief's full text. The tests below
are organised by the issue's acceptance criteria, one section each, and
every one carries an assertion that distinguishes the behaviour from its
nearest wrong implementation: a field block that omits a field, a prefix
resolver that guesses, a not-found path that reads like an empty belief,
a tombstone that is unreachable or unlabelled, a writable store open, an
unregistered subcommand, and an output shape that drifts.
"""
from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own throwaway DB at <tmp>/aelf.db."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _belief(bid: str, content: str, **overrides: object) -> Belief:
    fields: dict[str, object] = {
        "id": bid,
        "content": content,
        "content_hash": hashlib.sha256(
            f"{bid}:{content}".encode()
        ).hexdigest(),
        "alpha": 3.0,
        "beta": 1.0,
        "type": BELIEF_FACTUAL,
        "lock_level": LOCK_NONE,
        "locked_at": None,
        "created_at": "2026-05-05T00:00:00Z",
        "last_retrieved_at": None,
        "origin": ORIGIN_AGENT_INFERRED,
        "retention_class": RETENTION_FACT,
    }
    fields.update(overrides)
    return Belief(**fields)  # type: ignore[arg-type]


def _seed(db: Path, *beliefs: Belief) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


# --- AC1: one belief, every field ------------------------------------------


def test_show_prints_every_field_the_issue_names(isolated_db: Path) -> None:
    """AC1. All nine fields, each with the seeded value, not a default.

    Distinguishing: every value is seeded away from what a Belief would
    default to, so a handler that printed the dataclass defaults, or
    dropped a field, fails on that field's line rather than passing on a
    coincidence.
    """
    _seed(isolated_db, _belief(
        "a1b2c3d4e5f60000",
        "uv is the package manager for every Python environment here",
        alpha=7.0,
        beta=1.0,
        lock_level=LOCK_USER,
        locked_at="2026-05-06T00:00:00Z",
        origin=ORIGIN_USER_STATED,
        retention_class=RETENTION_SNAPSHOT,
        created_at="2026-05-05T12:34:56Z",
        valid_to="2026-06-01T00:00:00Z",
        scope=BELIEF_SCOPE_GLOBAL,
    ))
    code, out = _run("show", "a1b2c3d4e5f60000")
    assert code == 0
    assert "belief: a1b2c3d4e5f60000" in out
    assert (
        "uv is the package manager for every Python environment here" in out
    )
    assert "origin: user_stated" in out
    assert "lock: user" in out
    assert "retention: snapshot" in out
    assert "posterior: 0.875 (alpha=7, beta=1)" in out
    assert "created: 2026-05-05T12:34:56Z" in out
    assert "valid-to: 2026-06-01T00:00:00Z" in out
    assert "scope: global" in out


def test_show_prints_long_content_in_full(isolated_db: Path) -> None:
    """AC1, the part `show` exists for: content is never truncated.

    Distinguishing: the assertion is on the *last* characters of a
    multi-line, 2 KB body. A handler that truncated, elided, or printed
    a preview passes the "content appears" check and fails this one.
    """
    body = "\n".join(
        f"line {i:03d}: a sentence long enough to survive any preview cap"
        for i in range(48)
    )
    assert len(body) > 2000
    _seed(isolated_db, _belief("bb00000000000001", body))
    code, out = _run("show", "bb00000000000001")
    assert code == 0
    assert body in out
    assert out.rstrip("\n").endswith(
        "line 047: a sentence long enough to survive any preview cap"
    )


# --- AC2: exact-id lookup, never a guess ------------------------------------


def test_an_unambiguous_prefix_resolves(isolated_db: Path) -> None:
    """AC2. One belief starts with the prefix, so it resolves."""
    _seed(
        isolated_db,
        _belief("ffee000000000001", "the prefixed belief"),
        _belief("00aa000000000002", "an unrelated belief"),
    )
    code, out = _run("show", "ffee")
    assert code == 0
    assert "belief: ffee000000000001" in out
    assert "the prefixed belief" in out


def test_an_ambiguous_prefix_names_the_ambiguity_and_prints_no_belief(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC2. Two candidates: it reports both and resolves neither.

    Distinguishing on *both* halves. A handler that took the first row
    would exit 0 with a belief block, so stdout is asserted empty and
    the exit code non-zero; a handler that refused without saying why
    would pass an exit-code-only check, so both candidate ids must be
    named in the message.
    """
    _seed(
        isolated_db,
        _belief("dead000000000001", "the first candidate"),
        _belief("dead000000000002", "the second candidate"),
    )
    capsys.readouterr()
    code, out = _run("show", "dead")
    err = capsys.readouterr().err
    assert code != 0
    assert out == ""
    assert "ambiguous" in err
    assert "dead000000000001" in err
    assert "dead000000000002" in err
    assert "the first candidate" not in err + out
    assert "the second candidate" not in err + out


def test_an_exact_id_wins_over_being_a_prefix_of_another(
    isolated_db: Path
) -> None:
    """AC2. `abcd` addresses `abcd`, even though `abcd1234` starts with it.

    Distinguishing: a pure prefix resolver reports an ambiguity here and
    exits non-zero. The exact row must come back, alone.
    """
    _seed(
        isolated_db,
        _belief("abcd", "the short id, addressing itself"),
        _belief("abcd1234", "the long id that merely starts the same"),
    )
    code, out = _run("show", "abcd")
    assert code == 0
    assert "belief: abcd\n" in out
    assert "the short id, addressing itself" in out
    assert "abcd1234" not in out


def test_show_does_not_match_on_content(isolated_db: Path) -> None:
    """AC2. Lookup is by id; content words are not a query.

    Distinguishing: the word is in the belief's content and `aelf
    search` finds it. A handler that fell back to FTS5 would resolve it;
    `show` must not.
    """
    _seed(isolated_db, _belief("cc00000000000003", "quokkas are marsupials"))
    assert _run("search", "quokkas")[0] == 0
    assert "quokkas" in _run("search", "quokkas")[1]
    code, out = _run("show", "quokkas")
    assert code != 0
    assert out == ""


def test_a_percent_sign_is_not_a_wildcard(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC2. The prefix is compared literally, not as a LIKE pattern.

    Distinguishing, and the reason it asserts on *which* failure: under
    `LIKE prefix || '%'` a bare `%` matches every id in the store, so a
    LIKE-based resolver also exits non-zero — it reports an ambiguity
    over the whole store. Exit code alone therefore proves nothing here.
    The assertion is that the failure is "no belief with id", and that
    no candidate was ever in the running.
    """
    _seed(
        isolated_db,
        _belief("11aa000000000001", "the first"),
        _belief("22bb000000000002", "the second"),
    )
    capsys.readouterr()
    code, out = _run("show", "%")
    err = capsys.readouterr().err
    assert code != 0
    assert out == ""
    assert "no belief with id" in err
    assert "ambiguous" not in err
    assert "11aa000000000001" not in err


# --- AC3: unknown id, distinguishable from an empty belief -----------------


def test_an_unknown_id_exits_non_zero_and_says_not_found(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC3, first half."""
    _seed(isolated_db, _belief("aa00000000000001", "a belief that exists"))
    capsys.readouterr()
    code, out = _run("show", "0000000000000000")
    err = capsys.readouterr().err
    assert code != 0
    assert out == ""
    assert "no belief with id" in err
    assert "0000000000000000" in err


def test_a_not_found_id_is_distinguishable_from_empty_content(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC3, the half that is easy to get wrong.

    A belief whose content is the empty string is a belief: it resolves,
    exits 0, and prints its field block with an empty content body. The
    missing id does none of those. Distinguishing: the two runs must
    differ in exit code *and* in whether a field block was printed — a
    handler that printed nothing for empty content, or exited 0 for a
    missing id, collapses the two cases this asserts apart.
    """
    _seed(isolated_db, _belief("ee00000000000001", ""))
    capsys.readouterr()
    empty_code, empty_out = _run("show", "ee00000000000001")
    capsys.readouterr()
    missing_code, missing_out = _run("show", "ee00000000000009")
    missing_err = capsys.readouterr().err

    assert empty_code == 0
    assert empty_out.endswith("content:\n\n")
    assert "belief: ee00000000000001" in empty_out
    assert missing_code == 1
    assert missing_out == ""
    assert "no belief with id" in missing_err
    assert empty_code != missing_code


# --- AC4: a retired belief is reachable, and labelled ----------------------


def test_a_retired_belief_is_reachable_and_labelled_retired(
    isolated_db: Path
) -> None:
    """AC4. The id in a week-old injected block may name a tombstone.

    Distinguishing: `get_belief` hides retired rows by default (#1210),
    so a handler built on it exits non-zero here. And a handler that
    reached the row but printed the active header would miss `retired`.
    """
    _seed(isolated_db, _belief("7e7e000000000001", "a retired truth"))
    assert _run("retire", "7e7e000000000001")[0] == 0
    code, out = _run("show", "7e7e000000000001")
    assert code == 0
    assert "status: retired" in out
    assert "a retired truth" in out
    assert "status: active" not in out


def test_an_active_belief_is_labelled_active(isolated_db: Path) -> None:
    """AC4's other arm: the label is read off the row, not hard-coded.

    Without this, a handler that printed `status: retired` for every
    belief would pass the retired test above.
    """
    _seed(isolated_db, _belief("7e7e000000000002", "a live truth"))
    code, out = _run("show", "7e7e000000000002")
    assert code == 0
    assert "status: active" in out
    assert "valid-to: (none)" in out


# --- AC5: it reads a store it cannot write (see the note below) ------------

_POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission bits; root ignores them",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def frozen_store(
    isolated_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, str]]:
    """A seeded store whose directory is read-only, permissions restored.

    Mirrors `tests/test_readonly_store_1416.py`: the directory is frozen
    only after seeding, a live connection holds the `-wal`/`-shm`
    sidecars open (SQLite deletes both when the last connection closes,
    and cannot create them in an unwritable directory), and one table
    the open-time DDL battery recreates is dropped so the *writable*
    open has something it must write and therefore fails. Without that
    drop the writable open succeeds against a complete schema and the
    read-only path is never exercised.
    """
    d = tmp_path / "frozen"
    d.mkdir()
    db = d / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed(db, _belief("f0f0000000000001", "readable but not writable"))
    holder = sqlite3.connect(str(db))
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("SELECT count(*) FROM beliefs").fetchone()
    sqlite3.connect(str(db)).executescript(
        "DROP TABLE IF EXISTS exploration_events;"
    )
    for child in d.iterdir():
        child.chmod(0o444)
    d.chmod(0o555)
    try:
        yield db, "f0f0000000000001"
    finally:
        holder.close()
        d.chmod(0o755)
        for child in d.iterdir():
            child.chmod(0o644)


@_POSIX_ONLY
def test_show_reads_a_store_it_cannot_write(
    frozen_store: tuple[Path, str]
) -> None:
    """The half of AC5 this branch does deliver, behaviourally.

    Distinguishing, and not a call-site grep: against a store whose
    directory is read-only and whose schema is one table short, a
    writable open raises `sqlite3.OperationalError: attempt to write a
    readonly database` from `MemoryStore.__init__` before the belief is
    ever read. `show` must succeed, and the database file must be
    byte-identical afterwards.
    """
    db, bid = frozen_store
    before = _digest(db)
    code, out = _run("show", bid)
    assert code == 0
    assert "readable but not writable" in out
    assert _digest(db) == before


# There is deliberately no "and it writes nothing when the store *is*
# writable" test here, and it is not an omission: there is nothing to
# assert. `open_store_for_read` attempts the writable open first and
# only a permission failure falls back, so on a writable store the two
# handles are the same object and the open pays the full DDL battery,
# the migrations, the scope-id mint and the expired-lock sweep.
#
# AC5 as written asks for that cost to be gone, and this branch does not
# remove it. The operator's ruling was to ship `show` on the existing
# #1416 routing and re-file the cost as its own issue over every
# read-only verb, so the missing half is tracked there rather than
# papered over with a test that would pass identically against
# `_open_store()`. The frozen-store arm above is the one arm that can
# tell the two routings apart at all.


# --- AC6: the subcommand is registered -------------------------------------


def test_show_is_registered_in_the_enumerated_command_list() -> None:
    """AC6. `show` is in the list the CLI tests enumerate, and it's visible.

    `tests/test_slash_commands.py` asserts the CLI's subparser set equals
    EXPECTED_COMMANDS ∪ HIDDEN_SUBCOMMANDS; this states the #1553 half of
    that directly, including the operator's ruling that the verb ships
    visible rather than hidden.
    """
    from aelfrice.cli import build_parser
    from tests.test_slash_commands import (
        EXPECTED_COMMANDS,
        HIDDEN_SUBCOMMANDS,
    )

    assert "show" in EXPECTED_COMMANDS
    assert "show" not in HIDDEN_SUBCOMMANDS

    parser = build_parser()
    sub_actions = [
        a for a in parser._subparsers._actions  # type: ignore[union-attr]
        if a.__class__.__name__ == "_SubParsersAction"
    ]
    assert "show" in sub_actions[0].choices  # type: ignore[attr-defined]


def test_show_appears_in_the_default_help_output() -> None:
    """AC6. Visible, not a verb only `--advanced` reveals.

    Distinguishing against a `help=argparse.SUPPRESS` registration: a
    suppressed subparser still appears inside the `{a,b,c}` choices
    metavar — `context` and the other hidden verbs are all in there — but
    it gets no description line of its own. This asserts on that line.
    """
    from aelfrice.cli import build_parser

    described = [
        line for line in build_parser().format_help().splitlines()
        if line.strip().startswith("show ")
    ]
    assert described, "no `show` entry in the default --help body"
    assert "belief" in described[0]


# --- AC7: deterministic output shape ---------------------------------------


_EXPECTED_BLOCK = """\
belief: 5150000000000001
status: active
origin: agent_inferred
lock: none
retention: fact
posterior: 0.750 (alpha=3, beta=1)
created: 2026-05-05T00:00:00Z
valid-to: (none)
scope: project
content:
the pinned belief body
"""


def test_the_output_shape_is_pinned(isolated_db: Path) -> None:
    """AC7. The whole block, byte for byte, in order.

    Distinguishing at the strongest available level: any reordering,
    relabelling, added line, dropped line, or changed number formatting
    fails this, including changes an "is the value present" test would
    not see. Every value is deterministic — no clock, no id hashing, no
    float that depends on the machine.
    """
    _seed(isolated_db, _belief("5150000000000001", "the pinned belief body"))
    code, out = _run("show", "5150000000000001")
    assert code == 0
    assert out == _EXPECTED_BLOCK


def test_the_output_is_identical_across_repeated_runs(
    isolated_db: Path
) -> None:
    """AC7. Determinism, stated as a property rather than a literal.

    Catches a field that varies run to run — a `last_retrieved_at` bump,
    a clock read, a set iteration — which the pinned literal above would
    only catch if the variation happened to differ from the recorded run.
    """
    _seed(isolated_db, _belief("5150000000000002", "a repeatable body"))
    runs = [_run("show", "5150000000000002") for _ in range(3)]
    assert [rc for rc, _ in runs] == [0, 0, 0]
    assert len({out for _, out in runs}) == 1


# --- store-level: the prefix lookup itself ---------------------------------


def test_the_prefix_lookup_includes_tombstones(isolated_db: Path) -> None:
    """The new store method is the AC4 enabler; pin it at its own level.

    Distinguishing against `get_belief`'s default, which excludes
    retired rows (#1210): with `valid_to` set, this must still return it.
    """
    _seed(isolated_db, _belief(
        "d0d0000000000001", "a tombstone", valid_to="2026-06-01T00:00:00Z",
    ))
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief("d0d0000000000001") is None
        found = s.find_beliefs_by_id_prefix("d0d0000000000001")
        assert [b.id for b in found] == ["d0d0000000000001"]
        assert found[0].valid_to == "2026-06-01T00:00:00Z"
    finally:
        s.close()


def test_an_empty_prefix_matches_nothing(isolated_db: Path) -> None:
    """"Every belief in the store" is not an answer to an id lookup."""
    _seed(
        isolated_db,
        _belief("aa11000000000001", "one"),
        _belief("bb22000000000002", "two"),
    )
    s = MemoryStore(str(isolated_db))
    try:
        assert s.find_beliefs_by_id_prefix("") == []
        assert len(s.find_beliefs_by_id_prefix("")) != 2
    finally:
        s.close()


def test_the_prefix_lookup_is_id_ordered_and_capped(
    isolated_db: Path
) -> None:
    """Deterministic order, and `limit` binds.

    Distinguishing on both: the rows are inserted in descending id order,
    so an insertion-ordered result fails the ordering assertion, and a
    handler that ignored `limit` fails the count.
    """
    _seed(isolated_db, *[
        _belief(f"c0c00000000000{i}", f"body {i}")
        for i in (5, 4, 3, 2, 1)
    ])
    s = MemoryStore(str(isolated_db))
    try:
        found = s.find_beliefs_by_id_prefix("c0c0", limit=3)
        assert [b.id for b in found] == [
            "c0c000000000001", "c0c000000000002", "c0c000000000003",
        ]
    finally:
        s.close()


def test_a_non_positive_limit_is_refused(isolated_db: Path) -> None:
    """A silent empty result would read as "no such belief"."""
    s = MemoryStore(str(isolated_db))
    try:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            s.find_beliefs_by_id_prefix("aa", limit=0)
    finally:
        s.close()
