"""Shared pytest fixtures (#319 v2.0 bench-gate harness).

The bench-gate harness consumes the v2.0 evaluation corpus from #307. Corpus
content lives in the private lab repo only; the public repo carries the
schema contract and harness scaffold. The `AELFRICE_CORPUS_ROOT` env var
points the harness at a mounted corpus. When the var is unset, or when a
specific module directory is empty, bench-gate tests skip cleanly so public
CI passes without corpus access.

See `tests/corpus/v2_0/README.md` for the schema contract.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

CORPUS_ENV_VAR = "AELFRICE_CORPUS_ROOT"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--run-perf`, the opt-in for the latency benchmarks.

    Four modules gate a latency assertion on a `_has_run_perf` helper
    that reads this option (`test_bm25_index.py`, `test_heat_kernel.py`,
    `test_graph_spectral.py`, `test_hrr_struct_index.py`). Until #1160
    nothing registered it: `test_bm25_index.py` carried a
    `pytest_addoption` stub, but pytest only calls that hook from
    `conftest.py` or an installed plugin, never from a plain test
    module. So `getoption` always raised, every helper fell back to
    False, and `pytest --run-perf` failed outright with
    `unrecognized arguments` (exit 4) — the tests could not be opted
    into at all, only reached by editing the guard.

    Registering here rather than re-stubbing per module keeps one
    authority for the flag. Default False, so the suite's behaviour is
    unchanged unless the flag is passed.
    """
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help=(
            "run the latency benchmarks (large synthetic stores; wall-clock "
            "assertions, so results are load-sensitive)"
        ),
    )


def _corpus_root() -> Path | None:
    raw = os.environ.get(CORPUS_ENV_VAR)
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


@pytest.fixture(scope="session")
def aelfrice_corpus_root() -> Path:
    """Resolve `AELFRICE_CORPUS_ROOT` to a directory; skip the test otherwise.

    Tests that depend on labeled corpus rows should request this fixture and
    will skip on public CI where the env var is unset.
    """
    root = _corpus_root()
    if root is None:
        pytest.skip(
            f"{CORPUS_ENV_VAR} not set or not a directory; "
            "skipping bench-gate test (lab corpus absent)"
        )
    return root


@pytest.fixture(autouse=True)
def _skip_bench_gated_without_corpus(request: pytest.FixtureRequest) -> None:
    """Autouse guard: any test marked `bench_gated` skips when corpus is absent.

    Backstops the `aelfrice_corpus_root` fixture for marker-only tests that
    forget to request it explicitly.
    """
    if "bench_gated" not in request.keywords:
        return
    if _corpus_root() is None:
        pytest.skip(
            f"{CORPUS_ENV_VAR} not set or not a directory; "
            "skipping bench-gate test (lab corpus absent)"
        )


def load_corpus_module(root: Path, module: str) -> list[dict]:
    """Load every `*.jsonl` row under `root/<module>/`. Skip if empty."""
    mod_dir = root / module
    if not mod_dir.is_dir():
        pytest.skip(f"corpus module {module!r} missing under {root}")
    rows: list[dict] = []
    for p in sorted(mod_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        pytest.skip(f"corpus module {module!r} empty under {root}")
    return rows


# ---------------------------------------------------------------------------
# #1320 — keep the suite out of the contributor's real home directory.
# ---------------------------------------------------------------------------

# Every module-level constant below is derived from `Path.home()` at import
# time and is *written* during a plain `pytest tests` run. Pinning them here
# is the enforcement half of #1320; the source half (late-bound `= None`
# parameter defaults instead of pre-bound `Path` defaults) is what makes the
# pin reach the call sites at all.
#
# Auditing note: a pin is only load-bearing if the consumer reads the module
# attribute at call time. `tests/test_home_path_isolation_1320.py` asserts
# that property directly, so a future function that re-binds a home path as
# a parameter default fails there rather than silently escaping this fixture.
REAL_HOME: Path = Path.home()
"""The contributor's actual home, captured before `_sandbox_real_home` runs.

conftest is imported during collection; the session fixture below does not
apply until the first test sets up. So this is the last point at which the
real home is observable from inside the suite.

`tests/test_home_path_isolation_1320.py` needs it: its structural guard asks
"is this parameter default rooted under the user's home", and once the
fixture has repointed `HOME` at a sandbox, a default bound at import from
the *real* home is no longer relative to `Path.home()` — so the guard would
pass on exactly the defect it exists to catch.
"""

_HOME_PINS: list[tuple[str, str, str]] = [
    # (module, attribute, relative path under the sandbox home)
    ("aelfrice.auto_install", "AELFRICE_DOTDIR", ".aelfrice"),
    ("aelfrice.auto_install", "STAMP_PATH", ".aelfrice/installed-manifest-version"),
    ("aelfrice.auto_install", "OPT_OUT_PATH", ".aelfrice/opt-out-hooks.json"),
    ("aelfrice.setup", "USER_SETTINGS_PATH", ".claude/settings.json"),
    ("aelfrice.setup", "SLASH_COMMANDS_DIR_DEFAULT", ".claude/commands/aelf"),
    ("aelfrice.lifecycle", "CACHE_DIR", ".cache/aelfrice"),
    ("aelfrice.lifecycle", "CACHE_FILE", ".cache/aelfrice/update_check.json"),
    ("aelfrice.lifecycle", "MIGRATED_TO_UV_SENTINEL", ".aelfrice/migrated-to-uv"),
    ("aelfrice.temporal_spine", "SPINE_BACKFILLED_SENTINEL", ".aelfrice/spine-backfilled"),
    # Read-only today, but pinned so the dotdir stays internally
    # consistent: `test_uninstall_dotdir` asserts HOOK_FAILURES_LOG is
    # addressed relative to AELFRICE_DOTDIR, and that invariant has to
    # survive the sandbox as well as the real host.
    ("aelfrice.doctor", "HOOK_FAILURES_LOG", ".aelfrice/logs/hook-failures.log"),
    ("aelfrice.doctor", "_AELFRICE_PROJECTS_DIR", ".aelfrice/projects"),
    ("aelfrice.telemetry", "DEFAULT_TELEMETRY_PATH", ".aelfrice/telemetry.jsonl"),
    ("aelfrice.transcript_logger", "LEGACY_TRANSCRIPTS_DIR", ".aelfrice/transcripts"),
]

# Sentinels whose guard is `path.exists() -> no-op`. Pinning one of these at
# a *fresh* tmp path is worse than not pinning it: the short-circuit stops
# firing and the guarded side effect (a real `uv tool install --force
# aelfrice`, a real spine backfill) runs on every call. Pre-create them.
_PRECREATED_SENTINELS: frozenset[str] = frozenset({
    "MIGRATED_TO_UV_SENTINEL",
    "SPINE_BACKFILLED_SENTINEL",
})


@pytest.fixture(scope="session", autouse=True)
def _sandbox_real_home(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """Redirect every written home-derived path at a session sandbox (#1320).

    Without this the suite rewrote `~/.aelfrice/opt-out-hooks.json` (losing
    the `opt_out_hosts` key), `~/.aelfrice/installed-manifest-version`,
    `~/.claude/commands/aelf/` (pruning user-authored files),
    `~/.cache/aelfrice/update_check.json`, and created
    `~/.aelfrice/spine-backfilled`, which permanently suppresses the
    one-shot spine backfill on the contributor's machine.

    `AELF_NO_UPDATE_CHECK` is set because no `setattr` can reach the update
    check: `maybe_check_for_update_async` spawns a *detached interpreter*
    that re-imports `aelfrice.lifecycle` and recomputes `CACHE_FILE` from
    the real home. The env var is the only lever that crosses that boundary.

    Session-scoped, so a test wanting per-test control still overrides it
    with its own function-scoped `monkeypatch` (which is restored after).
    """
    import importlib

    home = tmp_path_factory.mktemp("sandbox_home")
    mp = pytest.MonkeyPatch()
    try:
        # HOME is the only lever that crosses a process boundary, and the
        # suite is full of subprocess-driven CLI tests: a child re-imports
        # aelfrice in a fresh interpreter and recomputes every constant
        # below from the real home, where no `setattr` can follow it.
        # Setting it here (children inherit) is what actually stops the
        # `aelf setup` integration tests rewriting ~/.aelfrice.
        mp.setenv("HOME", str(home))
        mp.setenv("AELF_NO_UPDATE_CHECK", "1")
        for mod_name, attr, relpath in _HOME_PINS:
            target = home / relpath
            if attr in _PRECREATED_SENTINELS:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            mp.setattr(importlib.import_module(mod_name), attr, target)
        yield home
    finally:
        mp.undo()
