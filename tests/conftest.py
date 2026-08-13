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
import math
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

CORPUS_ENV_VAR = "AELFRICE_CORPUS_ROOT"

BENCH_MEASUREMENT_PROPERTY = "bench_measurement"
"""`record_property` key a bench-gate test uses to report its numbers.

A gate that only prints on failure records nothing on the runs that
matter most — the green ones at a release cut, which are the only
evidence that the measurement was taken at all and the only place a
drift between cuts would show. Inverted checks make this acute: a
tripwire is *expected* to be green, so a message built solely inside its
assertion is dead code in the shipped path.

The summary prints whatever tests attach under this key, so the numbers
land in the same block that already says which modules ran.
"""

TIMEOUT_SCALE_ENV_VAR = "AELF_TEST_TIMEOUT_SCALE"
"""Multiplier applied to every resolved pytest-timeout budget (#1472)."""

MAX_TIMEOUT_SCALE = 1000.0
"""Upper bound on the scale, above which it is treated as malformed.

Not a policy on how patient anyone may be. `pytest.mark.timeout` is
handed the product, and a large enough one reaches `inf` or a value
`signal.setitimer` refuses, which aborts the whole session with
INTERNALERROR rather than failing one test. A typo in an environment
variable must not be able to do that.
"""

DEFAULT_TIMEOUT_SCALE = 4.0
"""Headroom for a loaded developer machine. CI pins this to 1.

Sized from the #1472 census rather than picked: the three tests that lost
the PR gate ran in 1.48 s, 0.43 s and 0.17 s unloaded against a 5 s budget,
and failed it at load ~30. A gate that a 3x slowdown defeats is measuring
the machine. 4x is one binding step past the observed spread and still
leaves a 30 s unit test terminating inside two minutes.
"""


def _timeout_scale() -> float:
    """Resolve the scale from the environment, falling back to the default.

    Never raises and never returns a non-positive value. A malformed
    setting must not decide how long the suite is allowed to run, and a
    scale of zero would disable pytest-timeout outright — which is the
    one outcome worse than a timeout that is too tight.
    """
    raw = os.environ.get(TIMEOUT_SCALE_ENV_VAR)
    if raw is None:
        return DEFAULT_TIMEOUT_SCALE
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SCALE
    if not math.isfinite(scale) or scale <= 0 or scale > MAX_TIMEOUT_SCALE:
        return DEFAULT_TIMEOUT_SCALE
    return scale


def _resolved_timeout(item: pytest.Item, env_timeout: object) -> float | None:
    """The budget pytest-timeout would apply to `item`, before scaling.

    Mirrors `pytest_timeout._get_item_settings`: the closest `timeout`
    marker wins, and only in its absence does the ini/CLI value apply.
    That order is the whole reason this hook exists — raising the ini
    `timeout`, or passing `--timeout`, provably cannot reach the 179
    per-test markers, because the marker is consulted first.

    Returns `None` when there is nothing to scale: no marker and no ini
    value, a non-numeric marker argument, or a value of zero, which
    pytest-timeout reads as "disabled".
    """
    marker = item.get_closest_marker("timeout")
    value: object = None
    if marker is not None:
        if marker.args:
            value = marker.args[0]
        elif "timeout" in marker.kwargs:
            value = marker.kwargs["timeout"]
    if value is None:
        value = env_timeout
    try:
        resolved = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Scale every test's wall-clock timeout by `AELF_TEST_TIMEOUT_SCALE`.

    #1472. The global budget is sized for an idle machine, so a developer
    running the suite under real load loses tests that have no defect —
    twelve PR-gate attempts went that way, and `aelf-pr-open.sh` runs
    pytest with `-x`, so the first such loss ends the run.

    Scaling at collection is what reaches the whole population. The ini
    value governs only unmarked tests; 179 tests carry their own marker
    and a further handful carry one built from a module constant, which
    a static census misses entirely. Re-adding the marker with
    `append=False` puts it where `get_closest_marker` looks first, so it
    wins over the test's own — that is the load-bearing detail, and
    flipping it to `append=True` makes the scale silently inert for
    every marked test while still passing for unmarked ones.

    CI pins the scale to 1, per workflow. The suite there runs on a
    dedicated runner, and a 4x-looser budget would weaken hang detection
    in `ci.yml` and `publish.yml`, neither of which carries a
    `timeout-minutes` backstop.

    One deliberate side effect: an item that carried no `timeout` marker
    gains one, so it starts matching `-m timeout` and stops matching
    `-m "not timeout"`. Nothing in the suite selects on that marker, and
    the alternative — leaving the ini population unscaled — would give a
    loaded machine headroom on 179 tests and none on the rest.
    """
    scale = _timeout_scale()
    if scale == 1.0:
        return
    env_timeout = getattr(config, "_env_timeout", None)
    for item in items:
        resolved = _resolved_timeout(item, env_timeout)
        if resolved is None:
            continue
        # Carry the marker's other keywords through. `pytest-timeout`
        # takes `method`, `func_only` and `disable_debugger_detection`
        # alongside the budget, and re-adding a bare
        # `pytest.mark.timeout(N)` in front of one that set them would
        # silently drop them — turning a test that needs the thread
        # method into one that does not have it. No marker in the suite
        # uses them today; that is exactly why this would go unnoticed.
        marker = item.get_closest_marker("timeout")
        kwargs = dict(marker.kwargs) if marker is not None else {}
        kwargs.pop("timeout", None)
        item.add_marker(
            pytest.mark.timeout(resolved * scale, **kwargs), append=False,
        )


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


# The one skip reason for the whole tier, so the terminal summary below can
# recognise these skips by prefix rather than by re-deriving which tests are
# bench-gated. It names the issue that decided the tier runs lab-side, because
# the failure this guards against is a reader concluding the quality tier ran
# (#1456 AC1).
BENCH_GATE_SKIP_REASON: str = (
    f"{CORPUS_ENV_VAR} not set or not a directory; skipping bench-gate test "
    "(lab corpus absent by design — see #1420 §3; the corpus is private and "
    "this repository is public)"
)


def _corpus_root() -> Path | None:
    raw = os.environ.get(CORPUS_ENV_VAR)
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


# A corpus module that exists but holds nothing skips with its own reason,
# distinct from the whole-tier one above. `load_corpus_module` writes both.
_MODULE_SKIP_RE = re.compile(
    r"corpus module '(?P<module>[^']+)' (?P<why>missing|empty) under "
)


def _skip_reason(rep: object) -> str:
    """The reason text of a skip report, or '' if it has none."""
    longrepr = getattr(rep, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return ""


def pytest_terminal_summary(terminalreporter) -> None:  # type: ignore[no-untyped-def]
    """Report the bench-gate tier per module, executed against skipped.

    #1456 AC2 got the aggregate count: without it the tier's skips are
    indistinguishable from every other skip in a `N passed, M skipped`
    tail, and a reader checking whether the quality gates ran sees a
    green run and concludes they did.

    An aggregate is not enough once a corpus exists, which is #1477 AC3.
    The corpus covers 3 of the modules scaffolded in `tests/corpus/`, so
    a lab-side run with `AELFRICE_CORPUS_ROOT` set reports a healthy
    "N passed" while most of the tier skipped for want of rows — the
    same misreading one level in. The three states are therefore
    reported separately and by name:

    * the whole tier skipped, because no corpus root is set at all;
    * a named module skipped, because it is missing or holds no rows;
    * a bench-gated test actually executed.

    Classification is by skip *reason*, not by re-deriving which tests
    carry the marker, so an unrelated skip inside a bench-gated module
    is not folded in. Executed tests are counted off the marker, which
    is the only place that signal survives to summary time.
    """
    stats = terminalreporter.stats
    tier_skips = 0
    by_module: dict[tuple[str, str], int] = {}
    for rep in stats.get("skipped", []):
        reason = _skip_reason(rep)
        if CORPUS_ENV_VAR in reason:
            tier_skips += 1
            continue
        m = _MODULE_SKIP_RE.search(reason)
        if m:
            key = (m.group("module"), m.group("why"))
            by_module[key] = by_module.get(key, 0) + 1

    executed = 0
    measurements: list[str] = []
    for outcome in ("passed", "failed"):
        for rep in stats.get(outcome, []):
            if "bench_gated" not in getattr(rep, "keywords", {}):
                continue
            executed += 1
            for key, value in getattr(rep, "user_properties", ()):
                if key == BENCH_MEASUREMENT_PROPERTY:
                    measurements.append(str(value))

    if not (tier_skips or by_module or executed):
        return

    terminalreporter.write_sep("-", "bench-gate tier")
    if tier_skips:
        terminalreporter.write_line(
            f"{tier_skips} bench-gate tests skipped: lab corpus absent by "
            f"design (#1420 §3). These are the retrieval / compression / "
            f"clustering quality gates; they did NOT run. Set "
            f"{CORPUS_ENV_VAR} to a corpus root to execute them."
        )
    if executed:
        terminalreporter.write_line(
            f"{executed} bench-gate tests executed against the corpus."
        )
    for (module, why), n in sorted(by_module.items()):
        terminalreporter.write_line(
            f"  module {module!r}: {n} test(s) skipped — the corpus module "
            f"is {why}. This module's gate has no verdict."
        )
    for line in sorted(measurements):
        terminalreporter.write_line(f"  {line}")


@pytest.fixture(scope="session")
def aelfrice_corpus_root() -> Path:
    """Resolve `AELFRICE_CORPUS_ROOT` to a directory; skip the test otherwise.

    Tests that depend on labeled corpus rows should request this fixture and
    will skip on public CI where the env var is unset.
    """
    root = _corpus_root()
    if root is None:
        pytest.skip(BENCH_GATE_SKIP_REASON)
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
        pytest.skip(BENCH_GATE_SKIP_REASON)


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
# #1431 — keep the suite out of the contributor's ambient uv/XDG config.
# ---------------------------------------------------------------------------

# uv-tool detection is env-first: `UV_TOOL_DIR` wins outright, then
# `XDG_DATA_HOME` / `APPDATA` select the platform default
# (`lifecycle._uv_tool_dir`). Every test that builds a fake tools tree under
# a sandbox home is therefore only correct on a machine where none of these
# are exported — green on CI, red for any contributor who sets one, and
# `UV_TOOL_DIR` is the very override #1431 added. Measured on the #1431
# branch before this fixture: `XDG_DATA_HOME=/tmp/xdgfake pytest
# tests/test_uv_tool_layout_1431.py tests/test_upgrade_ux.py
# tests/test_auto_install.py` gave 10 failed / 65 passed against 75 passed
# with the variable unset.
#
# Cleared session-wide rather than per-module: any future consumer of uv's
# layout inherits the isolation, and a test that *wants* one of these sets
# it explicitly with its own function-scoped monkeypatch.
_AMBIENT_LAYOUT_ENV_VARS: tuple[str, ...] = (
    "UV_TOOL_DIR",
    "UV_TOOL_BIN_DIR",
    "XDG_DATA_HOME",
    "XDG_BIN_HOME",
    "APPDATA",
)


@pytest.fixture(scope="session", autouse=True)
def _clear_ambient_layout_env() -> Iterator[None]:
    """Unset uv/XDG layout variables for the whole suite (#1431)."""
    mp = pytest.MonkeyPatch()
    try:
        for name in _AMBIENT_LAYOUT_ENV_VARS:
            mp.delenv(name, raising=False)
        yield
    finally:
        mp.undo()


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
    ("aelfrice.mcp_cleanup", "MCP_CLEANUP_SENTINEL", ".aelfrice/mcp-surface-removed"),
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
    "MCP_CLEANUP_SENTINEL",
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
