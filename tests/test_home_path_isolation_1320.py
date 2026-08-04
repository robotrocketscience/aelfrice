"""#1320 — home-derived paths must be resolvable at call time, not import time.

Two independent guards live here.

1. A structural one: no function in the package may carry a parameter
   default that is a `Path` under `Path.home()`. Such a default is
   evaluated once, at import — which for pytest is collection time, always
   before any test body runs. `monkeypatch.setenv("HOME", …)` is therefore
   too late, and `monkeypatch.setattr(module, "CONST", …)` never reaches
   the already-bound default. Both of the suite's isolation mechanisms were
   silently inert, and running `pytest` rewrote the contributor's real
   `~/.aelfrice/` and `~/.claude/commands/aelf/`.

2. Behavioural ones: for each function that was actually observed writing
   real user state, patch the module constant and assert the write landed
   on the patched path **and that the real path is untouched**. Asserting
   only that the patched path was written proves nothing — that already
   worked when an explicit argument was passed; the defect was exclusively
   on the omitted-argument path, so every call below omits the argument.

Nothing here writes outside `tmp_path`. Every assertion about "the real
path" is a read (existence + bytes), never a write.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path, PurePath

import pytest

import aelfrice
from aelfrice import auto_install, lifecycle, temporal_spine
from tests.conftest import REAL_HOME


# ---------------------------------------------------------------------------
# 1. Structural guard — pure introspection, no I/O, terminates by construction
# ---------------------------------------------------------------------------


def _iter_package_modules() -> list[str]:
    """Every importable module under the `aelfrice` package."""
    names = [aelfrice.__name__]
    for info in pkgutil.walk_packages(
        aelfrice.__path__, prefix=aelfrice.__name__ + "."
    ):
        names.append(info.name)
    return sorted(names)


def _home_rooted_defaults(module_name: str) -> list[str]:
    """Return `"func(param)"` for each default that is a real-home Path."""
    # Both roots, and REAL_HOME is the load-bearing one. The autouse
    # `_sandbox_real_home` fixture repoints HOME at a sandbox before this
    # runs, so a default bound at *import* from the contributor's actual
    # home is not relative to `Path.home()` any more — checking only the
    # live value makes this guard pass on precisely the defect it hunts.
    # Verified: reverting `maybe_backfill_temporal_spine`'s default to the
    # pre-bound `SPINE_BACKFILLED_SENTINEL` left this test green until
    # REAL_HOME was added.
    roots = {REAL_HOME, Path.home()}
    mod = importlib.import_module(module_name)
    offenders: list[str] = []
    for obj_name, obj in vars(mod).items():
        if not inspect.isroutine(obj):
            continue
        # Skip re-exports: attribute the finding to the defining module.
        if getattr(obj, "__module__", None) != module_name:
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        for param in sig.parameters.values():
            default = param.default
            if not isinstance(default, PurePath):
                continue
            try:
                rooted = any(
                    Path(default).is_relative_to(root) for root in roots
                )
            except (OSError, ValueError):  # pragma: no cover — defensive
                continue
            if rooted:
                offenders.append(
                    f"{module_name}.{obj_name}({param.name}={default})"
                )
    return offenders


@pytest.mark.timeout(60)
def test_no_parameter_default_is_a_home_rooted_path() -> None:
    """No function in the package pre-binds a path under the user's home.

    This is the guard that makes the next regression fail immediately
    instead of quietly writing a contributor's machine. The fix shape is
    always the same: default to `None`, resolve from the module-level
    constant inside the body (`auto_install.read_stamp`, #839/#1320).
    """
    offenders: list[str] = []
    for name in _iter_package_modules():
        try:
            importlib.import_module(name)
        except Exception:  # pragma: no cover — optional-extra modules
            continue
        offenders.extend(_home_rooted_defaults(name))
    assert offenders == [], (
        "parameter defaults bound to a real-home path at import time — "
        "make the default None and resolve from the module constant in "
        "the body (#1320):\n  " + "\n  ".join(offenders)
    )


def test_the_structural_guard_can_fail() -> None:
    """The guard above is vacuous unless it detects a planted offender.

    `_home_rooted_defaults` is the whole detector; exercise it against a
    module-shaped namespace carrying exactly the defect it hunts.
    """
    import types

    planted = types.ModuleType("planted_offender_1320")

    def leaks(p: Path = Path.home() / ".aelfrice" / "x") -> Path:
        return p

    leaks.__module__ = "planted_offender_1320"
    planted.leaks = leaks  # pyright: ignore[reportAttributeAccessIssue]

    import sys

    sys.modules["planted_offender_1320"] = planted
    try:
        found = _home_rooted_defaults("planted_offender_1320")
    finally:
        del sys.modules["planted_offender_1320"]
    assert len(found) == 1 and "leaks(p=" in found[0]


# ---------------------------------------------------------------------------
# 2. Behavioural guards — the patched path is used, the real one is not
# ---------------------------------------------------------------------------


def _real(*parts: str) -> Path:
    """A real home-rooted path. Only ever read, never written, by this file."""
    return Path.home().joinpath(*parts)


def _snapshot(path: Path) -> tuple[bool, bytes]:
    """Existence + bytes, so an assertion can prove *unchanged*, not absent."""
    if not path.exists() or not path.is_file():
        return (False, b"")
    return (True, path.read_bytes())


def test_add_opt_out_honours_a_patched_opt_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`add_opt_out()` with no path argument writes the patched location."""
    real = _real(".aelfrice", "opt-out-hooks.json")
    before = _snapshot(real)

    pinned = tmp_path / "dotdir" / "opt-out-hooks.json"
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", pinned)

    auto_install.add_opt_out("transcript_ingest")

    assert json.loads(pinned.read_text())["opt_out"] == ["transcript_ingest"]
    assert auto_install.read_opt_outs() == frozenset({"transcript_ingest"})
    assert _snapshot(real) == before, (
        f"add_opt_out() wrote the real {real} despite OPT_OUT_PATH being "
        "patched — the parameter default is bound at import (#1320)"
    )


def test_remove_opt_out_honours_a_patched_opt_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unlink branch must not reach the real ledger either."""
    real = _real(".aelfrice", "opt-out-hooks.json")
    before = _snapshot(real)

    pinned = tmp_path / "dotdir" / "opt-out-hooks.json"
    pinned.parent.mkdir(parents=True)
    pinned.write_text(json.dumps({"opt_out": ["session_start"]}))
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", pinned)

    auto_install.remove_opt_out("session_start")

    assert not pinned.exists()
    assert _snapshot(real) == before, (
        f"remove_opt_out() unlinked or rewrote the real {real} (#1320)"
    )


def test_read_stamp_honours_a_patched_stamp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_stamp()` reads the patched stamp, not the host's."""
    pinned = tmp_path / "installed-manifest-version"
    pinned.write_text("9.9.9\n")
    monkeypatch.setattr(auto_install, "STAMP_PATH", pinned)
    assert auto_install.read_stamp() == "9.9.9"


def test_host_opt_out_writers_honour_a_patched_opt_out_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`add_host_opt_out` / `remove_host_opt_out` — the #1053 ledger keys."""
    real = _real(".aelfrice", "opt-out-hooks.json")
    before = _snapshot(real)

    pinned = tmp_path / "dotdir" / "opt-out-hooks.json"
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", pinned)

    auto_install.add_host_opt_out("codex")
    assert auto_install.read_host_opt_outs() == frozenset({"codex"})
    assert auto_install.remove_host_opt_out("codex") is True

    assert _snapshot(real) == before, (
        f"host opt-out writers reached the real {real} (#1320)"
    )


def test_clear_cache_honours_a_patched_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`clear_cache()` must delete the patched cache, never `~/.cache`."""
    real = _real(".cache", "aelfrice", "update_check.json")
    before = _snapshot(real)

    pinned = tmp_path / "update_check.json"
    pinned.write_text("{}")
    monkeypatch.setattr(lifecycle, "CACHE_FILE", pinned)

    lifecycle.clear_cache()

    assert not pinned.exists()
    assert _snapshot(real) == before, (
        f"clear_cache() deleted or rewrote the real {real} (#1320)"
    )


def test_check_for_update_writes_only_the_patched_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync check writes the patched cache. No network: `fetch` is a stub."""
    real = _real(".cache", "aelfrice", "update_check.json")
    before = _snapshot(real)

    pinned = tmp_path / "cache" / "update_check.json"
    monkeypatch.setattr(lifecycle, "CACHE_FILE", pinned)
    monkeypatch.delenv(lifecycle.ENV_DISABLE, raising=False)

    payload = {"info": {"version": "0.0.1"}, "releases": {}}
    lifecycle.check_for_update(fetch=lambda _url: payload)

    assert json.loads(pinned.read_text())["latest"] == "0.0.1"
    assert _snapshot(real) == before, (
        f"check_for_update() wrote the real {real} (#1320)"
    )


def test_maybe_migrate_to_uv_honours_a_patched_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentinel short-circuit must key off the patched path.

    Polarity matters: the guard is `sentinel.exists()`, so a bound default
    meant the real `~/.aelfrice/migrated-to-uv` decided whether a live
    `uv tool install --force aelfrice` ran. This test pins an *existing*
    sentinel precisely so no subprocess is reachable from here.
    """
    pinned = tmp_path / "migrated-to-uv"
    pinned.touch()
    monkeypatch.setattr(lifecycle, "MIGRATED_TO_UV_SENTINEL", pinned)

    result = lifecycle.maybe_migrate_to_uv()

    assert result.attempted is False
    assert "sentinel exists" in result.reason


def test_maybe_backfill_spine_honours_a_patched_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-shot spine sentinel must never be read from the real home.

    A bound default let the suite create `~/.aelfrice/spine-backfilled` on
    contributor machines, permanently suppressing the #1064-G4 backfill —
    silently, because the documented reversal is `aelf spine clear`, not
    deleting the file.
    """
    pinned = tmp_path / "spine-backfilled"
    pinned.touch()
    monkeypatch.setattr(temporal_spine, "SPINE_BACKFILLED_SENTINEL", pinned)

    result = temporal_spine.maybe_backfill_temporal_spine(
        store=None,  # pyright: ignore[reportArgumentType] — never dereferenced
    )

    assert result.ran is False
    assert "sentinel exists" in result.reason


# ---------------------------------------------------------------------------
# 3. The sibling-key regression the same audit turned up
# ---------------------------------------------------------------------------


def test_add_opt_out_preserves_opt_out_hosts(tmp_path: Path) -> None:
    """A per-hook opt-out must not delete the #1053 host-level key."""
    ledger = tmp_path / "opt-out-hooks.json"
    auto_install.add_host_opt_out("codex", ledger)
    auto_install.add_opt_out("commit_ingest", ledger)

    doc = json.loads(ledger.read_text())
    assert doc["opt_out"] == ["commit_ingest"]
    assert doc["opt_out_hosts"] == ["codex"]


def test_remove_last_opt_out_preserves_opt_out_hosts(tmp_path: Path) -> None:
    """Rescinding the final per-hook opt-out must not unlink the ledger."""
    ledger = tmp_path / "opt-out-hooks.json"
    auto_install.add_host_opt_out("codex", ledger)
    auto_install.add_opt_out("commit_ingest", ledger)

    auto_install.remove_opt_out("commit_ingest", ledger)

    assert ledger.exists(), "unlinked the ledger, destroying opt_out_hosts"
    doc = json.loads(ledger.read_text())
    assert doc["opt_out"] == []
    assert doc["opt_out_hosts"] == ["codex"]
    assert auto_install.read_host_opt_outs(ledger) == frozenset({"codex"})


def test_remove_last_opt_out_unlinks_when_no_hosts_remain(
    tmp_path: Path,
) -> None:
    """With both keys empty the ledger is still cleaned up (no behaviour loss)."""
    ledger = tmp_path / "opt-out-hooks.json"
    auto_install.add_opt_out("commit_ingest", ledger)
    auto_install.remove_opt_out("commit_ingest", ledger)
    assert not ledger.exists()
