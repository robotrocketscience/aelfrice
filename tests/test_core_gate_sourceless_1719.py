"""#1719: `core_gate` survives an install that ships no source.

`CLASSIFIER_VERSION` hashes the source of `build_prompt` and
`parse_labels`, so on an install with only bytecode `inspect.getsource`
raises `OSError`, and a `.pyc` that doesn't match its `.py` can raise
`SyntaxError` or `tokenize.TokenError`. The import used to fail with it.
The UserPromptSubmit hook imports `core_gate` while it builds the first
prompt's `<session-start>` block whenever there are core candidates, so
its fail-soft wrapper returned that block empty, `<locked>` section
included.

Now the version is None when the source is missing, and the failure stays
inside the core lane: the gate admits no unlocked belief, no label applies,
and nothing writes a batch or a label. A label made under one classifier
never applies under another, and None is not a classifier.

The fixture simulates the missing source by making `inspect.getsource`
raise for `core_gate`'s functions and importing the module fresh. Each
test names the mutation that kills it. Every store lives under `tmp_path`.
"""
from __future__ import annotations

import importlib
import inspect
import io
import re
import sqlite3
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

import aelfrice.core_gate as real_core_gate
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, ORIGIN_AGENT_INFERRED, Belief
from aelfrice.store import MemoryStore

LOCKED = "locked0000001719"
# In core through the posterior arm on a normal install.
CORE = "core000000001719"


def _hash(bid: str) -> str:
    return f"h_{bid}"


def _mk(store: MemoryStore, bid: str, *, locked: bool) -> None:
    store.insert_belief(Belief(
        id=bid, content=f"the widget {bid} has a stated property",
        content_hash=_hash(bid), alpha=1.0 if locked else 9.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-08-01T09:00:00+00:00" if locked else None,
        created_at="2026-08-01T09:00:00+00:00", last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    ))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(path))
    s = MemoryStore(str(path))
    _mk(s, LOCKED, locked=True)
    _mk(s, CORE, locked=False)
    # Labeled A under the real classifier: on a normal install it is in
    # core, so its absence below is the gate's doing, not the fixture's.
    s.put_core_gate_labels(
        {_hash(CORE): "A", _hash(LOCKED): "A"},
        classifier_version=real_core_gate.CLASSIFIER_VERSION, batch_id=None,
        labeled_at="2026-10-05T00:00:00+00:00",
    )
    try:
        yield s
    finally:
        s.close()


#: What `inspect` raises for a function whose source it can't read: no
#: `.py` at all, or a `.py` that doesn't match the `.pyc` it was compiled to.
_SOURCE_ERRORS: list[Exception] = [
    OSError("could not get source code"),
    TypeError("module, class, method, function, traceback, frame, or code object was expected"),
    SyntaxError("invalid syntax"),
    tokenize.TokenError("EOF in multi-line statement", (1, 0)),
]


@pytest.fixture(params=_SOURCE_ERRORS, ids=lambda e: type(e).__name__)
def sourceless(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModuleType]:
    """Import `core_gate` fresh with its source unavailable, and yield it.
    Every lazy `from aelfrice.core_gate import ...` and
    `from aelfrice import core_gate` then resolves to it. The real module
    comes back in `sys.modules` and on the package afterwards."""
    real_getsource = inspect.getsource
    error: Exception = request.param

    def getsource(obj: object) -> str:
        if getattr(obj, "__module__", None) == "aelfrice.core_gate":
            raise error
        return real_getsource(obj)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(inspect, "getsource", getsource)
    monkeypatch.setattr(sys.modules["aelfrice"], "core_gate", real_core_gate)
    monkeypatch.delitem(sys.modules, "aelfrice.core_gate")
    yield importlib.import_module("aelfrice.core_gate")


def test_the_real_install_has_a_version() -> None:
    """The fallback must not fire when the source is there."""
    assert isinstance(real_core_gate.CLASSIFIER_VERSION, str)
    assert real_core_gate.CLASSIFIER_VERSION.startswith("core-gate-")


def test_import_survives_without_source(sourceless: ModuleType) -> None:
    """Mutation: drop the `OSError` fallback around the digest."""
    assert sourceless.CLASSIFIER_VERSION is None


def test_session_start_keeps_locked_and_drops_core(
    sourceless: ModuleType, store: MemoryStore, tmp_path: Path,
) -> None:
    """Mutations: drop the fallback (the block is empty); pass candidates
    through when the version is None (CORE reappears in `<core>`)."""
    from aelfrice.hook import _retrieve_session_start_block  # pyright: ignore[reportPrivateUsage]

    err = io.StringIO()
    block = _retrieve_session_start_block(err, cwd=tmp_path, store=store)
    assert "build failed" not in err.getvalue(), err.getvalue()
    locked = re.search(r"<locked>(.*?)</locked>", block, re.S)
    assert locked is not None, block
    assert f'<belief id="{LOCKED}"' in locked.group(1)
    assert f'<belief id="{CORE}"' not in block


def test_no_label_applies_without_a_version(
    sourceless: ModuleType, store: MemoryStore,
) -> None:
    """Mutation: pass candidates through when the version is None."""
    candidate = store.get_belief(CORE)
    assert candidate is not None
    admitted, labels = store.gate_core_candidates([candidate], {})
    assert admitted == []
    assert labels == {}


def test_a_label_always_has_a_version(store: MemoryStore) -> None:
    """AC2 rests on the schema: `classifier_version` is `NOT NULL`, so no
    stored label has a NULL version for a None lookup to match.
    Mutation: drop `NOT NULL` from the column."""
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        store.put_core_gate_labels(
            {_hash(CORE): "A"}, classifier_version=None,  # pyright: ignore[reportArgumentType]
            batch_id=None, labeled_at="2026-10-05T00:00:00+00:00",
        )


def test_a_none_version_matches_no_stored_label(store: MemoryStore) -> None:
    """The lookup compares with `=`, never true against NULL. The fixture
    stores labels for both beliefs, so the lookup has rows to find.
    Mutation: compare with `IS NOT` instead of `=`."""
    assert store.core_gate_labels_for([_hash(CORE), _hash(LOCKED)], None) == {}


# --- nothing writes a batch or a label without a version -----------------


def test_writers_get_no_version(sourceless: ModuleType) -> None:
    """Mutation: return None from `current_classifier_version`."""
    with pytest.raises(sourceless.ClassifierUnavailable, match="#1719"):
        sourceless.current_classifier_version()


class _UnreadableStdin(io.StringIO):
    """A terminal nobody types into: reading it is the failure under test."""

    def read(self, size: int | None = -1) -> str:
        raise AssertionError("stdin was read before the refusal")


def test_core_gate_accept_refuses(
    sourceless: ModuleType, store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """It refuses before reading stdin, so an interactive run doesn't wait
    for EOF. Mutations: drop the version check; check after the read."""
    from aelfrice.cli import main

    monkeypatch.setattr(sys, "stdin", _UnreadableStdin())
    assert main(argv=["core-gate", "accept", "nobatch"], out=io.StringIO()) == 1
    err = capsys.readouterr().err
    assert "#1719" in err
    assert "no core-gate batch" not in err


def test_doctor_emit_refuses_and_writes_nothing(
    sourceless: ModuleType, store: MemoryStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mutation: drop the version check in `aelf doctor core-gate --emit`."""
    from aelfrice.cli import main

    assert main(argv=["doctor", "core-gate", "--emit"], out=io.StringIO()) == 1
    assert "#1719" in capsys.readouterr().err
    assert store.list_open_core_gate_batches(
        origin="doctor", classifier_version=str(real_core_gate.CLASSIFIER_VERSION),
    ) == []


def test_doctor_emit_library_call_refuses(
    sourceless: ModuleType, store: MemoryStore,
) -> None:
    """The library entry point refuses too, for a caller other than the CLI.
    Mutation: read the version without `current_classifier_version`."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import emit_core_gate_batches

    with pytest.raises(sourceless.ClassifierUnavailable):
        emit_core_gate_batches(
            store, default_core_rule, limit=None,
            created_at="2026-10-06T00:00:00+00:00",
        )


def test_doctor_rerun_refuses(
    sourceless: ModuleType, store: MemoryStore,
) -> None:
    """Mutation: drop the version check in `rerun_core_gate_batch`."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import CoreGateRerunRefused, rerun_core_gate_batch

    with pytest.raises(CoreGateRerunRefused, match="#1719"):
        rerun_core_gate_batch(store, default_core_rule, "nobatch")


def test_doctor_coverage_says_none_are_admitted(
    sourceless: ModuleType, store: MemoryStore,
) -> None:
    """Mutation: report the usual coverage block for a None version."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import core_gate_coverage, format_core_gate_coverage

    report = core_gate_coverage(store, default_core_rule)
    assert report.classifier_version is None
    assert report.unlabeled == report.candidates == 1
    text = "\n".join(format_core_gate_coverage(report))
    assert "none admitted" in text
    assert "follow today's rule" not in text


SESSION = "session-1719"


def _session_end(store: MemoryStore, tmp_path: Path) -> bool:
    from aelfrice.hook import (
        CORE_GATE_SESSION_END_ENV,
        _maybe_core_gate_session_end,  # pyright: ignore[reportPrivateUsage]
    )

    return _maybe_core_gate_session_end(
        store, {"cwd": str(tmp_path)}, SESSION,
        env={CORE_GATE_SESSION_END_ENV: "1"},
        stdout=io.StringIO(), stderr=io.StringIO(),
    )


def _mk_session_candidate(store: MemoryStore) -> None:
    store.insert_belief(Belief(
        id="sess000000001719", content="the widget in this session is stable",
        content_hash="h_sess000000001719", alpha=9.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-08-01T09:00:00+00:00", last_retrieved_at=None,
        session_id=SESSION, origin=ORIGIN_AGENT_INFERRED,
    ))


def test_session_end_batches_on_a_normal_install(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """The control: the same store does batch when the version exists."""
    _mk_session_candidate(store)
    assert _session_end(store, tmp_path) is True


def test_session_end_writes_no_batch_without_a_version(
    sourceless: ModuleType, store: MemoryStore, tmp_path: Path,
) -> None:
    """Mutation: drop the None check in `_collect_core_gate_session_candidates`."""
    _mk_session_candidate(store)
    assert _session_end(store, tmp_path) is False
    assert store.list_open_core_gate_batches(
        origin="session_end", classifier_version=str(real_core_gate.CLASSIFIER_VERSION),
    ) == []


def test_doctor_rerun_cli_refuses_before_creating_out(
    sourceless: ModuleType, store: MemoryStore, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mutation: check the version after `--out` is created."""
    from aelfrice.cli import main

    out_dir = tmp_path / "prompts"
    argv = ["doctor", "core-gate", "--rerun", "nobatch", "--out", str(out_dir)]
    assert main(argv=argv, out=io.StringIO()) == 1
    assert "#1719" in capsys.readouterr().err
    assert not out_dir.exists()


def _fs_rows(store: MemoryStore) -> int:
    from aelfrice.models import CORROBORATION_SOURCES_NON_ASSERTING

    return sum(store.count_corroborations_by_source(
        CORROBORATION_SOURCES_NON_ASSERTING,
    ).values())


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "apply"])
def test_gc_filesystem_corroboration_refuses(
    sourceless: ModuleType, store: MemoryStore,
    capsys: pytest.CaptureFixture[str], apply: bool,
) -> None:
    """With no classifier version, core membership reads empty, so the
    pass would report nothing leaving core and delete the rows anyway.
    It refuses instead, dry run and apply alike, and writes nothing.
    Mutation: drop the version check in `gc_filesystem_corroboration`."""
    from aelfrice.cli import main
    from aelfrice.models import CORROBORATION_SOURCE_FILESYSTEM_INGEST

    for i, day in enumerate((2, 5)):
        store.record_corroboration(
            CORE, source_type=CORROBORATION_SOURCE_FILESYSTEM_INGEST,
            session_id=f"scan-{i}", ts=f"2026-08-{day:02d}T09:00:00+00:00",
        )
    before = _fs_rows(store)
    assert before == 2
    argv = ["doctor", "--gc-filesystem-corroboration"] + (["--apply"] if apply else [])
    out = io.StringIO()
    assert main(argv=argv, out=out) == 1
    assert "#1719" in capsys.readouterr().err
    assert out.getvalue() == ""
    assert _fs_rows(store) == before


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "apply"])
def test_gc_filesystem_corroboration_refuses_before_opening_a_store(
    sourceless: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], apply: bool,
) -> None:
    """Opening a store creates and migrates it, so a refusal must come
    first. Mutation: check the version after `_open_store()`."""
    from aelfrice.cli import main

    db = tmp_path / "never-opened.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    argv = ["doctor", "--gc-filesystem-corroboration"] + (["--apply"] if apply else [])
    assert main(argv=argv, out=io.StringIO()) == 1
    assert "#1719" in capsys.readouterr().err
    assert not db.exists()


def test_doctor_emit_refuses_before_creating_out(
    sourceless: ModuleType, store: MemoryStore, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mutation: check the version after `--out` is created."""
    from aelfrice.cli import main

    out_dir = tmp_path / "prompts"
    argv = ["doctor", "core-gate", "--emit", "--out", str(out_dir)]
    assert main(argv=argv, out=io.StringIO()) == 1
    assert "#1719" in capsys.readouterr().err
    assert not out_dir.exists()


class _UntouchableStore:
    """Any attribute read fails: the pass must refuse before it looks at
    the store at all."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"store.{name} was read before the refusal")


def test_gc_filesystem_corroboration_library_refuses_before_reading(
    sourceless: ModuleType,
) -> None:
    """The library check runs before any read, write, or transaction, not
    inside the transaction where a rollback would hide a late check.
    Mutation: move the check after `delete_corroborations_by_source`."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import gc_filesystem_corroboration

    with pytest.raises(sourceless.ClassifierUnavailable):
        gc_filesystem_corroboration(
            _UntouchableStore(),  # pyright: ignore[reportArgumentType]
            qualifies=default_core_rule, dry_run=False,
        )
