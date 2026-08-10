"""`hooks.json` updates must not overwrite foreign changes (#1428).

`install_codex_hooks` / `remove_codex_hooks` did `Path.read_text()` ...
`Path.write_text()` with nothing in between. Two losses followed, both
silent because the result stays valid JSON:

* a **stale snapshot** — anything written to the file between our read
  and our write was replaced by what we had read;
* a **shape coercion** — a parseable but structurally unexpected foreign
  value (`{"hooks":{"UserPromptSubmit":{"foreign":"keep"}}}`) was
  substituted with an aelfrice event list during an ordinary setup run,
  with no error and no `--force`.

The interleaving tests inject the competing write at the read boundary
in one process. That is the mechanism control; the multi-writer test
below runs the real thing through the lock.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from aelfrice.file_lock import HAVE_ADVISORY_LOCKS, try_lock_exclusive
from aelfrice.host_codex import install_codex_hooks, remove_codex_hooks

#: Ceiling on every wait below. The lock timeout is 10s, so six writers
#: contending for it cannot legitimately need more than this.
_JOIN_TIMEOUT_S = 60.0

_FOREIGN_A = {
    "hooks": {
        "SessionEnd": [
            {"hooks": [{"type": "command", "command": "foreign-a"}]}
        ]
    }
}
_FOREIGN_AB = {
    "hooks": {
        "SessionEnd": [
            {"hooks": [{"type": "command", "command": "foreign-a"}]}
        ],
        "Notification": [
            {"hooks": [{"type": "command", "command": "foreign-b"}]}
        ],
    }
}


def _write(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc), encoding="utf-8")


def _arm_interleaved_write(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    newer: object,
    *,
    times: int = 1,
    distinct: bool = False,
) -> None:
    """Rewrite `target` from inside its own read, `times` times.

    Reproduces a competing writer landing after our read and before our
    commit, deterministically and without a second process. `distinct`
    makes every injected document different, modelling a writer that
    keeps changing the file rather than one that converges.
    """
    real = Path.read_bytes
    remaining = {"n": times}

    def interleaved(self: Path, *args: object, **kwargs: object) -> bytes:
        data = real(self, *args, **kwargs)  # type: ignore[arg-type]
        if self == target and remaining["n"] > 0:
            remaining["n"] -= 1
            doc = json.loads(json.dumps(newer))
            if distinct:
                doc["foreign-seq"] = remaining["n"]
            _write(target, doc)
        return data

    monkeypatch.setattr(Path, "read_bytes", interleaved)


# --- stale snapshot --------------------------------------------------------


def test_interleaved_write_is_not_clobbered_by_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    _arm_interleaved_write(monkeypatch, p, _FOREIGN_AB)

    result = install_codex_hooks(p)

    final = json.loads(p.read_text(encoding="utf-8"))
    assert result.error is None
    assert "Notification" in final["hooks"], "concurrent update was lost"
    assert "SessionEnd" in final["hooks"]
    assert final["hooks"]["UserPromptSubmit"], "our own set must land too"


def test_interleaved_write_is_not_clobbered_by_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "hooks.json"
    install_codex_hooks(p)
    installed = json.loads(p.read_text(encoding="utf-8"))
    newer = json.loads(json.dumps(installed))
    newer["hooks"]["Notification"] = [
        {"hooks": [{"type": "command", "command": "foreign-b"}]}
    ]
    _arm_interleaved_write(monkeypatch, p, newer)

    result = remove_codex_hooks(p)

    final = json.loads(p.read_text(encoding="utf-8"))
    assert result.error is None
    assert "Notification" in final["hooks"], "concurrent update was lost"
    assert "UserPromptSubmit" not in final["hooks"]


def test_relentless_competing_writer_aborts_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded retry, then a clear refusal — never a stale overwrite."""
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    _arm_interleaved_write(
        monkeypatch, p, _FOREIGN_AB, times=99, distinct=True,
    )

    result = install_codex_hooks(p)

    assert result.error is not None
    assert "modified by another process" in result.error
    assert result.changed is False
    final = json.loads(p.read_text(encoding="utf-8"))
    assert final["hooks"] == _FOREIGN_AB["hooks"], (
        "the competing writer's document stands"
    )
    assert "UserPromptSubmit" not in final["hooks"]


# --- foreign shapes --------------------------------------------------------


def test_unexpected_event_shape_is_refused_not_normalised(
    tmp_path: Path,
) -> None:
    p = tmp_path / "hooks.json"
    p.write_text(
        '{"hooks":{"UserPromptSubmit":{"foreign":"keep"}}}', encoding="utf-8"
    )
    before = p.read_bytes()

    result = install_codex_hooks(p)

    assert result.error is not None
    assert "UserPromptSubmit" in result.error
    assert result.changed is False
    assert p.read_bytes() == before


def test_non_object_hooks_value_is_refused_not_normalised(
    tmp_path: Path,
) -> None:
    p = tmp_path / "hooks.json"
    p.write_text('{"hooks": 42}', encoding="utf-8")
    before = p.read_bytes()

    result = install_codex_hooks(p)

    assert result.error is not None
    assert "non-object `hooks`" in result.error
    assert p.read_bytes() == before


def test_force_is_the_only_way_past_a_foreign_shape(tmp_path: Path) -> None:
    p = tmp_path / "hooks.json"
    p.write_text(
        '{"hooks":{"UserPromptSubmit":{"foreign":"keep"}}}', encoding="utf-8"
    )

    result = install_codex_hooks(p, force=True)

    assert result.error is None
    assert result.changed is True
    final = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(final["hooks"]["UserPromptSubmit"], list)


def test_a_json_null_document_is_refused_not_overwritten(
    tmp_path: Path,
) -> None:
    """`null` is a legal JSON document, and not a JSON object.

    It must land on the same refusal as `42` or `[1,2,3]`. It did not:
    the refusal guard was written as `parsed is not None`, reusing None
    as the "could not parse" sentinel, so a file holding `null` slipped
    past it and was replaced with a fresh aelfrice document — no error,
    no `--force`, `changed=True`.
    """
    p = tmp_path / "hooks.json"
    p.write_text("null", encoding="utf-8")
    before = p.read_bytes()

    result = install_codex_hooks(p)

    assert result.error is not None
    assert "not a JSON object" in result.error
    assert result.changed is False
    assert p.read_bytes() == before


@pytest.mark.parametrize(
    ("document", "needle"),
    [
        ('{"hooks": null}', "non-object `hooks`"),
        ('{"hooks": {"UserPromptSubmit": null}}', "UserPromptSubmit"),
    ],
)
def test_a_null_below_the_top_level_is_refused_not_reshaped(
    tmp_path: Path, document: str, needle: str
) -> None:
    """`null` is a legal value at every level, not just the document.

    Fixing only the document level left the identical `None`-as-sentinel
    hole one and two levels down: `dict.get(k)` answers None for a
    missing key *and* for a key set to `null`, so `{"hooks": null}` took
    the "no `hooks` key yet, fill it in" branch and
    `{"hooks":{"UserPromptSubmit":null}}` took the "no value for this
    event yet" branch — both reshaped with `error=None`, `changed=True`
    and no `--force`, while `{"hooks": 42}` and
    `{"hooks":{"UserPromptSubmit":42}}` were refused. The shipped promise
    ("a non-object `hooks` value, or a non-list value on an event
    aelfrice installs into, is left byte-for-byte alone and reported") is
    an absolute, so `null` must land on it too.
    """
    p = tmp_path / "hooks.json"
    p.write_text(document, encoding="utf-8")
    before = p.read_bytes()

    result = install_codex_hooks(p)

    assert result.error is not None
    assert needle in result.error
    assert "null" in result.error, "the refusal must name the JSON type"
    assert result.changed is False
    assert p.read_bytes() == before


@pytest.mark.parametrize(
    "document",
    ['{"hooks": null}', '{"hooks": {"UserPromptSubmit": null}}'],
)
def test_force_is_still_the_way_past_a_null_below_the_top_level(
    tmp_path: Path, document: str
) -> None:
    """The new refusals must be escapable, like every sibling refusal."""
    p = tmp_path / "hooks.json"
    p.write_text(document, encoding="utf-8")

    result = install_codex_hooks(p, force=True)

    assert result.error is None
    assert result.changed is True
    final = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(final["hooks"]["UserPromptSubmit"], list)
    assert final["hooks"]["UserPromptSubmit"]


def test_a_top_level_list_is_refused_not_overwritten(tmp_path: Path) -> None:
    """The sibling case that never regressed — pins both sentinels."""
    p = tmp_path / "hooks.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    before = p.read_bytes()

    result = install_codex_hooks(p)

    assert result.error is not None
    assert "not a JSON object" in result.error
    assert p.read_bytes() == before


def test_force_still_replaces_a_json_null_document(tmp_path: Path) -> None:
    """The refusal must be escapable, exactly as for the other shapes."""
    p = tmp_path / "hooks.json"
    p.write_text("null", encoding="utf-8")

    result = install_codex_hooks(p, force=True)

    assert result.error is None
    assert result.changed is True
    assert json.loads(p.read_text(encoding="utf-8"))["hooks"]


def test_unparseable_bytes_are_still_replaced_under_force(
    tmp_path: Path,
) -> None:
    """A syntax error and a `null` document must stay distinguishable.

    Collapsing the two sentinels the other way — treating an unparseable
    file as a parsed value — would make `--force` refuse the truncated
    `hooks.json` it exists for.
    """
    p = tmp_path / "hooks.json"
    p.write_text('{"hooks": {"UserPrompt', encoding="utf-8")

    assert install_codex_hooks(p).error is not None
    result = install_codex_hooks(p, force=True)

    assert result.error is None
    assert json.loads(p.read_text(encoding="utf-8"))["hooks"]


def test_unknown_top_level_keys_and_foreign_events_survive(
    tmp_path: Path,
) -> None:
    p = tmp_path / "hooks.json"
    _write(
        p,
        {
            "unrelated": {"deep": [1, 2, {"x": None}]},
            "hooks": _FOREIGN_A["hooks"],
        },
    )

    assert install_codex_hooks(p).error is None

    final = json.loads(p.read_text(encoding="utf-8"))
    assert final["unrelated"] == {"deep": [1, 2, {"x": None}]}
    assert final["hooks"]["SessionEnd"] == _FOREIGN_A["hooks"]["SessionEnd"]


# --- an uninstall on a host that never had Codex --------------------------


def test_remove_creates_nothing_when_there_is_no_hooks_json(
    tmp_path: Path,
) -> None:
    """An uninstall verb must not create the directory it removes from.

    Taking the transaction's lock is itself a write: `_open_lock` mkdirs
    the parent and creates `hooks.json.lock`. Routing the "no file,
    nothing to remove" case through the transaction therefore made
    `aelf unsetup --host codex` (and `aelf uninstall --host codex`, which
    reaches the same handler) bring the Codex home into being on a host
    that never had Codex, leaving behind a lock file this code
    deliberately never sweeps — the #1173 uninstall-artifact class, and
    the opposite of what INSTALL.md promises.

    Asserted over the whole subtree rather than over `hooks.json`,
    because nothing the defect created was named `hooks.json`.
    """
    root = tmp_path / "host-without-codex"
    root.mkdir()
    hooks_path = root / ".codex" / "hooks.json"

    result = remove_codex_hooks(hooks_path)

    assert result.error is None
    assert result.changed is False
    assert sorted(p.name for p in root.rglob("*")) == []


def test_remove_still_transacts_when_the_file_exists(tmp_path: Path) -> None:
    """The early return is an absence check, not a bypass.

    Pins that skipping the transaction is conditional on the file being
    missing: with a real `hooks.json` present the lock is taken, so the
    sibling lock file appears and the aelfrice entries are stripped.
    """
    p = tmp_path / "hooks.json"
    install_codex_hooks(p)
    assert json.loads(p.read_text(encoding="utf-8"))["hooks"]

    result = remove_codex_hooks(p)

    assert result.error is None
    assert result.changed is True
    assert (tmp_path / "hooks.json.lock").exists(), "the lock was not taken"
    assert "UserPromptSubmit" not in json.loads(
        p.read_text(encoding="utf-8")
    ).get("hooks", {})


# --- atomic commit ---------------------------------------------------------


def test_a_failed_commit_leaves_the_previous_document_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_text` truncates first; `os.replace` cannot half-apply."""
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    before = p.read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError("injected: crash between fsync and rename")

    monkeypatch.setattr("aelfrice.host_codex.os.replace", boom)
    result = install_codex_hooks(p)

    assert result.error is not None
    assert "unchanged" in result.error
    assert result.changed is False
    assert p.read_bytes() == before
    leftovers = sorted(
        c.name for c in tmp_path.iterdir()
        if c.name not in ("hooks.json", "hooks.json.lock")
    )
    assert leftovers == [], f"temp file left behind: {leftovers}"


def test_existing_permission_bits_are_preserved(tmp_path: Path) -> None:
    """`mkstemp` creates 0600; a shared config must not narrow to that."""
    if os.name == "nt":  # pragma: no cover - POSIX mode bits
        pytest.skip("POSIX mode bits")
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    p.chmod(0o644)

    assert install_codex_hooks(p).error is None

    assert stat.S_IMODE(p.stat().st_mode) == 0o644


# --- the lock itself -------------------------------------------------------
#
# The tests below are the ones that fail when the lock is deleted. The
# multi-writer test further down does NOT: its six writers are carried
# entirely by the fingerprint-and-retry loop, so replacing
# `exclusive_file_lock(...)` with `contextlib.nullcontext()` leaves it
# green. A claimed mechanism with no distinguishing assert is not a
# tested mechanism, so the lock is asserted directly — held across the
# read-modify-write, and reported rather than raised when contended.


@pytest.mark.skipif(
    not HAVE_ADVISORY_LOCKS, reason="no advisory locking on this filesystem"
)
def test_the_lock_is_held_across_the_read_modify_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second opener cannot take `hooks.json.lock` while we plan.

    Probed from inside the snapshot read — the exact instant the
    pre-#1428 code was exposed — with a fresh file descriptor, because
    `flock` is held per open file description and a re-open in the same
    process contends exactly as another process would.
    """
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    lock_path = tmp_path / "hooks.json.lock"
    observed: list[bool] = []
    real = Path.read_bytes

    def probing(self: Path, *args: object, **kwargs: object) -> bytes:
        if self == p:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try_lock_exclusive(fd)
                observed.append(False)  # acquired => we were NOT holding it
            except OSError:
                observed.append(True)
            finally:
                os.close(fd)
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", probing)
    assert install_codex_hooks(p).error is None

    assert observed, "the snapshot read never ran"
    assert all(observed), (
        "hooks.json.lock was acquirable during the read-modify-write — "
        "the transaction is running unlocked"
    )


@pytest.mark.skipif(
    not HAVE_ADVISORY_LOCKS, reason="no advisory locking on this filesystem"
)
def test_a_contended_lock_is_reported_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held lock must produce `result.error`, not a `FileLockTimeout`.

    The timeout is shortened here because the wait *duration* is not the
    subject; that it is bounded, and that the refusal names this host's
    file, is. The shipped default is pinned separately below.
    """
    monkeypatch.setattr("aelfrice.host_codex._HOOKS_LOCK_TIMEOUT", 0.05)
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    before = p.read_bytes()

    fd = os.open(str(tmp_path / "hooks.json.lock"), os.O_CREAT | os.O_RDWR)
    try:
        try_lock_exclusive(fd)
        result = install_codex_hooks(p)
    finally:
        os.close(fd)

    assert result.error is not None
    assert "another aelfrice process is writing" in result.error
    assert "hooks.json" in result.error
    assert "settings.json" not in result.error, (
        "the #1161 wrapper's message names the other host's file"
    )
    assert result.changed is False
    assert p.read_bytes() == before


def test_the_shipped_lock_timeout_is_bounded_and_positive() -> None:
    """Pins the value the previous test monkeypatches away."""
    from aelfrice.host_codex import _HOOKS_LOCK_TIMEOUT

    assert _HOOKS_LOCK_TIMEOUT == 10.0


# --- concurrent aelfrice writers ------------------------------------------


def test_concurrent_writers_preserve_the_foreign_entry(
    tmp_path: Path,
) -> None:
    """Six real writers, no injected interleave: the union survives.

    Threads rather than processes: `exclusive_file_lock` takes `flock`
    on a per-open file description, so two threads each opening their own
    fd serialise exactly as two processes do — and the test stays
    deterministic and fast enough to run on every CI leg.

    This is an end-to-end outcome assertion and deliberately does *not*
    claim to pin the lock: it stays green with the lock removed, because
    the fingerprint-and-retry loop alone carries six writers on this
    machine. The two tests above are the ones that go red.
    """
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    barrier = threading.Barrier(6)
    errors: list[str] = []

    def worker(remove: bool) -> None:
        barrier.wait(timeout=_JOIN_TIMEOUT_S)
        result = remove_codex_hooks(p) if remove else install_codex_hooks(p)
        if result.error:
            errors.append(result.error)

    threads = [
        threading.Thread(target=worker, args=(i % 2 == 1,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    assert not [t for t in threads if t.is_alive()], "a writer never finished"

    assert errors == []
    final = json.loads(p.read_text(encoding="utf-8"))
    assert final["hooks"]["SessionEnd"] == _FOREIGN_A["hooks"]["SessionEnd"]
