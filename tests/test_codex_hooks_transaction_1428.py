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

from aelfrice.host_codex import install_codex_hooks, remove_codex_hooks

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


# --- concurrent aelfrice writers ------------------------------------------


def test_concurrent_writers_preserve_the_foreign_entry(
    tmp_path: Path,
) -> None:
    """Real contention through the lock, not an injected interleave.

    Threads rather than processes: `exclusive_file_lock` takes `flock`
    on a per-open file description, so two threads each opening their own
    fd serialise exactly as two processes do — and the test stays
    deterministic and fast enough to run on every CI leg.
    """
    p = tmp_path / "hooks.json"
    _write(p, _FOREIGN_A)
    barrier = threading.Barrier(6)
    errors: list[str] = []

    def worker(remove: bool) -> None:
        barrier.wait()
        result = remove_codex_hooks(p) if remove else install_codex_hooks(p)
        if result.error:
            errors.append(result.error)

    threads = [
        threading.Thread(target=worker, args=(i % 2 == 1,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    final = json.loads(p.read_text(encoding="utf-8"))
    assert final["hooks"]["SessionEnd"] == _FOREIGN_A["hooks"]["SessionEnd"]
