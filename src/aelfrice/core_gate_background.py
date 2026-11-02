"""Label core-gate batches in the background, with no conversation turn (#1741).

The Stop hook records a session-end batch of this session's unlabeled core
candidates, then starts this module as a detached child. The child rebuilds
the batch's classifier prompt from the store, runs the host's `claude` CLI on
its smallest model, and caches the labels the way `aelf core-gate accept`
does. Nothing is written to the hook's stdout, so the user's conversation
gets no turn and sees nothing (#1740 turned the turn-taking path off).

The runner's flags were measured on #1741 before it was built:

- `--tools ""`, `--strict-mcp-config`, `--disable-slash-commands`,
  `--system-prompt`, and the two `CLAUDE_CODE_DISABLE_*` variables keep
  the CLI's own context out of the run. Without them each run carried
  about 51,000-64,000 input tokens; with them, about 2,300.
- `--setting-sources ""` loads none of the user's, project's, or local
  settings files, so no hook, `apiKeyHelper`, or settings `env` block
  reaches the run (verified: the login still authenticates). Managed
  settings an organization deploys still apply.
- `--settings <file>` turns hooks off as well, so the run can't fire
  aelfrice's hooks and recurse. The file holds `{"disableAllHooks":
  true}` and nothing else; it is written per run (`settings_file`) and
  removed when the run returns. The JSON was passed inline before #1761;
  a file keeps it out of the quoting that an npm `claude.cmd` shim on
  Windows passes its arguments through.
- `--no-session-persistence` keeps the run out of the session history.
- Thinking stays on: with it off, agreement with the reference labels fell
  by about 19 points.
- `--max-budget-usd 0.10` caps each run (operator ruling, 2026-10-08).

The labels are the classifier's own: the prompt is `core_gate.build_prompt`
and the reply goes through `core_gate.parse_labels`, the same parser
`aelf core-gate accept` uses. The child's environment drops any API key,
so the run always uses the `claude` login (`runner_env`).

On Windows (#1761) the child starts with `DETACHED_PROCESS`,
`CREATE_NO_WINDOW`, and `CREATE_NEW_PROCESS_GROUP` in place of
`start_new_session`, which Popen ignores there, and runs the CLI with
`CREATE_NO_WINDOW`, so neither opens a console window. `find_claude`
resolves `claude` through the `PATHEXT` extensions, so an npm
`claude.cmd` shim is found, and only in the folders named on `PATH`, so
a `claude.cmd` in the project the hook runs in is not. The reply is
decoded as UTF-8 rather than through the ANSI code page. A run that
passes its timeout ends with its whole process tree (`run_bounded`), so
the node process behind a `claude.cmd` doesn't outlive it.

Never raises into the hook. A failed run leaves the batch open; an open
batch claims its beliefs, so no later Stop asks about them, and
`aelf doctor core-gate --emit` batches them again. Unlabeled beliefs keep
today's core rule, so a failure costs nothing but the label.
"""
from __future__ import annotations

import json
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Final, Generator, cast

from aelfrice import launcher
from aelfrice.print_mode import ENV_HOST_ENTRYPOINT

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

BACKGROUND_ENV: Final[str] = "AELFRICE_CORE_GATE_BACKGROUND"
"""`0`/`false`/`no`/`off` turns background labeling off and
`1`/`true`/`yes`/`on` turns it on, overriding the TOML key in both
directions; any other value falls through to the key."""

BACKGROUND_KEY: Final[str] = "background"
"""`[core_gate] background` in `.aelfrice.toml`. Default `true`."""

CLAUDE_BIN: Final[str] = "claude"
RUNNER_MODEL: Final[str] = "haiku"
RUNNER_BUDGET_USD: Final[str] = "0.10"
RUNNER_TIMEOUT_SECONDS: Final[int] = 300
#: After a timeout, the most `run_bounded` waits for `taskkill`, and then
#: for the run's pipes to close (see `run_bounded`).
RUNNER_KILL_WAIT_SECONDS: Final[int] = 10
RUNNER_SYSTEM_PROMPT: Final[str] = (
    "You label short text snippets. Follow the user's instructions exactly."
)
LOG_FILENAME: Final[str] = "core_gate_background.jsonl"

#: The whole content of the per-run `--settings` file (see `settings_file`).
RUNNER_SETTINGS_JSON: Final[str] = '{"disableAllHooks": true}'
RUNNER_SETTINGS_PREFIX: Final[str] = "aelfrice-core-gate-"

#: Removed from the child's environment (see `runner_env`).
RUNNER_DROPPED_ENV: Final[tuple[str, ...]] = (
    "MAX_THINKING_TOKENS", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
)

#: The parent host session's identity, also removed (see `runner_env`):
#: the variables the host sets per session, and those it removes itself
#: before it starts a background session.
RUNNER_SESSION_ENV: Final[tuple[str, ...]] = (
    "CLAUDECODE", "CLAUDE_PID", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_HOST_WORKTREE",
    "CLAUDE_CODE_HOST_WORKTREE_FENCE", "CLAUDE_EFFORT", "AI_AGENT",
    "TRACEPARENT",
)

#: The headless entrypoint the child runs under (see `runner_env`).
RUNNER_ENTRYPOINT: Final[str] = "sdk-cli"

#: Inherited entrypoints `runner_env` replaces with `RUNNER_ENTRYPOINT`.
#: Others, such as a desktop label, also select how the host
#: authenticates, so they are kept.
RUNNER_REPLACED_ENTRYPOINTS: Final[frozenset[str | None]] = frozenset({None, "", "cli"})

#: Outcomes written to the log, one per run.
OUTCOME_ACCEPTED: Final[str] = "accepted"
OUTCOME_NO_BATCH: Final[str] = "no_batch"
OUTCOME_NOT_OPEN: Final[str] = "not_open"
OUTCOME_STALE_VERSION: Final[str] = "stale_version"
OUTCOME_STALE_BELIEF: Final[str] = "stale_belief"
OUTCOME_RUN_FAILED: Final[str] = "run_failed"
OUTCOME_BAD_REPLY: Final[str] = "bad_reply"
OUTCOME_SPAWN_FAILED: Final[str] = "spawn_failed"
OUTCOME_REFUSED: Final[str] = "refused"
OUTCOME_STORE_ERROR: Final[str] = "store_error"

_ON: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_OFF: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

#: Windows `CreateProcess` flags by value (processthreadsapi.h), used when
#: `subprocess` lacks the name, which it does on every other platform.
_WIN32_CREATION_FLAGS: Final[dict[str, int]] = {
    "DETACHED_PROCESS": 0x00000008,
    "CREATE_NEW_PROCESS_GROUP": 0x00000200,
    "CREATE_NO_WINDOW": 0x08000000,
}

#: The executable extensions `find_claude` tries when `PATHEXT` is unset.
_WIN32_DEFAULT_PATHEXT: Final[str] = ".COM;.EXE;.BAT;.CMD"


def _is_win32() -> bool:
    return sys.platform == "win32"


def _creationflags(*names: str) -> int:
    """The named Windows flags, combined, resolved through `subprocess`."""
    flags = 0
    for name in names:
        flags |= int(getattr(subprocess, name, _WIN32_CREATION_FLAGS[name]))
    return flags


def _env_override(env: dict[str, str] | None) -> bool | None:
    raw = (env if env is not None else os.environ).get(BACKGROUND_ENV)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ON:
        return True
    if norm in _OFF:
        return False
    return None


CORE_GATE_CONFIG_SECTION: Final[str] = "core_gate"


def _load_toml(start: Path | None, stderr: IO[str]) -> dict[str, Any]:
    """The nearest `.aelfrice.toml`, parsed, or `{}`. Never raises.

    The same discovery and failure handling as the hook's
    `_load_aelfrice_toml`, kept here because this module must not import
    `aelfrice.hook`, which imports it (#1631 keeps the hook out of import
    cycles).
    """
    import tomllib  # noqa: PLC0415

    from aelfrice.config_discovery import discover_config  # noqa: PLC0415

    try:
        candidate = discover_config(start)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"aelfrice hook: cannot look for .aelfrice.toml: {exc}", file=stderr)
        return {}
    if candidate is None:
        return {}
    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        print(f"aelfrice hook: cannot read {candidate}: {exc}", file=stderr)
        return {}
    try:
        return tomllib.loads(raw.decode("utf-8", errors="replace"))
    except tomllib.TOMLDecodeError as exc:
        print(f"aelfrice hook: malformed TOML in {candidate}: {exc}", file=stderr)
        return {}


def background_enabled(
    env: dict[str, str] | None, start: Path | None, stderr: IO[str],
) -> bool:
    """Whether the Stop hook labels in the background. Default True.

    The env var decides when it holds an on or off value; otherwise
    `[core_gate] background` in the nearest `.aelfrice.toml`. A missing
    file, section, or key reads as the default; a non-bool value reads as
    the default with a stderr trace. Never raises.
    """
    override = _env_override(env)
    if override is not None:
        return override
    section: Any = _load_toml(start, stderr).get(CORE_GATE_CONFIG_SECTION, {})
    if not isinstance(section, dict):
        return True
    val: Any = cast(dict[str, Any], section).get(BACKGROUND_KEY, True)
    if not isinstance(val, bool):
        print(
            f"aelfrice hook: ignoring [{CORE_GATE_CONFIG_SECTION}] "
            f"{BACKGROUND_KEY} (expected bool)",
            file=stderr,
        )
        return True
    return val


def find_claude() -> str | None:
    """Path of the host's `claude` CLI, or None when it isn't on PATH.

    On Windows (#1761) it asks for `claude` with each `PATHEXT` extension
    in turn, so the native installer's `claude.exe` and the npm shim
    `claude.cmd` both resolve, and the first extension found anywhere on
    PATH wins. It never asks for the bare name: npm installs an
    extensionless `sh` script beside the shim, which `CreateProcess` can't
    start, and Python 3.12.0's `which` tries the bare name first.

    Each lookup goes through `launcher.which_on_path` (#1412), because
    `shutil.which` on Windows searches the working directory before PATH,
    and the Stop hook's working directory is the user's project. A hit
    outside the PATH folders is dropped, so a `claude.cmd` planted in the
    project never runs. When it has the extension a PATH `claude` has, it
    hides that one too, and this returns None: no run, rather than the
    planted one.
    """
    if not _is_win32():
        return shutil.which(CLAUDE_BIN)
    pathext = os.environ.get("PATHEXT") or _WIN32_DEFAULT_PATHEXT
    for ext in pathext.split(";"):
        if not ext:
            continue
        found = launcher.which_on_path(CLAUDE_BIN + ext)
        if found is not None:
            return str(found)
    return None


@contextmanager
def settings_file() -> Generator[str, None, None]:
    """A file holding `RUNNER_SETTINGS_JSON`, removed on exit.

    One file per run, in the temp directory, created by `mkstemp`, so on
    POSIX it is readable by its owner only. It holds no belief text. The
    `finally` removes it on every exit the child's completion path can
    take: a normal return, a timeout, or a failed start. `run_bounded`
    makes the timeout exit come at most `RUNNER_TIMEOUT_SECONDS` plus
    twice `RUNNER_KILL_WAIT_SECONDS` after the start. Only a child killed
    outright leaves the file behind, as a 25-byte file whose name starts
    with `RUNNER_SETTINGS_PREFIX`.
    """
    fd, name = tempfile.mkstemp(prefix=RUNNER_SETTINGS_PREFIX, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(RUNNER_SETTINGS_JSON)
        yield name
    finally:
        try:
            os.unlink(name)
        except OSError:
            # Already gone, or the temp directory refused: nothing to do,
            # and the run's outcome must not depend on it.
            pass


def runner_argv(claude: str, settings: str) -> list[str]:
    """The measured classifier invocation (see the module docstring).

    `settings` is the path `settings_file` yields.
    """
    return [
        claude, "-p",
        "--model", RUNNER_MODEL,
        "--tools", "",
        "--setting-sources", "",
        "--settings", settings,
        "--no-session-persistence",
        "--output-format", "json",
        "--max-budget-usd", RUNNER_BUDGET_USD,
        "--system-prompt", RUNNER_SYSTEM_PROMPT,
        "--strict-mcp-config",
        "--disable-slash-commands",
    ]


def runner_env(base: dict[str, str]) -> dict[str, str]:
    """`base` for the child run.

    Drops a thinking cap, turns CLAUDE.md and auto memory off, and drops
    the API credentials. In `-p` mode an exported `ANTHROPIC_API_KEY` is
    always used when present (https://code.claude.com/docs/en/authentication),
    so leaving it would bill that key's account instead of the login
    the user chose (operator ruling on #1741, 2026-10-08). A user with
    key-only auth then gets no background labels: the run fails and the
    batch stays open.

    It also drops the identity of the host session the hook ran in (its
    session ID, effort, trace, and messaging socket and token), so the
    child is its own session, as in the measured runs. A missing or `cli`
    entrypoint becomes the headless `sdk-cli`, so if a hook fired in the
    child anyway, aelfrice's headless check would skip it. Any other
    entrypoint is kept: the host also uses a desktop label to choose how
    it authenticates, so replacing it could break the run. For those, the
    disabled hooks and settings are the only guard.
    """
    env = dict(base)
    for name in (*RUNNER_DROPPED_ENV, *RUNNER_SESSION_ENV):
        env.pop(name, None)
    if env.get(ENV_HOST_ENTRYPOINT) in RUNNER_REPLACED_ENTRYPOINTS:
        env[ENV_HOST_ENTRYPOINT] = RUNNER_ENTRYPOINT
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _taskkill_path() -> str:
    """System32's `taskkill.exe`, by full path.

    `CreateProcess` searches the working directory, here the user's
    project, before System32 for a bare name, so a bare `taskkill` would
    run a `taskkill.exe` planted there.
    """
    root = os.environ.get("SystemRoot") or "C:\\Windows"
    return ntpath.join(root, "System32", "taskkill.exe")


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """End `proc` and, on Windows, every process it started. Never raises.

    An npm `claude.cmd` runs as cmd.exe, which starts node.exe and waits
    on it, and node.exe holds the run's pipes. `Popen.kill` ends cmd.exe
    only, so on Windows `taskkill /F /T` ends the tree first, while
    cmd.exe is alive to link it. `Popen.kill` follows either way, so a
    `taskkill` that fails or is missing still ends the direct child.
    """
    if _is_win32():
        try:
            subprocess.run(  # noqa: S603 - a fixed System32 path and a pid
                [_taskkill_path(), "/F", "/T", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=RUNNER_KILL_WAIT_SECONDS,
                check=False, creationflags=_creationflags("CREATE_NO_WINDOW"),
            )
        except (OSError, subprocess.SubprocessError):
            # No taskkill, or it hung: the kill below still ends cmd.exe.
            pass
    try:
        proc.kill()
    except OSError:
        # Already exited.
        pass


def run_bounded(
    argv: list[str], *, input: str, timeout: float, capture_output: bool = True,
    **kwargs: Any,
) -> "subprocess.CompletedProcess[str]":
    """`subprocess.run` for the CLI, with a timeout that holds through a shim.

    On a timeout, `subprocess.run` kills only its direct child, and on
    Windows it then waits with no limit for the pipes to close (`kill()`,
    then `communicate()` with no timeout, in CPython 3.12-3.14). Through
    an npm `claude.cmd` the direct child is cmd.exe and node.exe holds the
    pipes, so that wait lasts as long as the hung run. Here a timeout ends
    the process tree (`_kill_tree`), waits at most
    `RUNNER_KILL_WAIT_SECONDS` for the pipes, and raises `TimeoutExpired`
    whether or not they closed. The call returns or raises within
    `timeout` plus twice `RUNNER_KILL_WAIT_SECONDS`.
    """
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
        argv, stdin=subprocess.PIPE, **kwargs,
    )
    try:
        out, err = proc.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=RUNNER_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            # A process the kill didn't reach still holds the pipes. The
            # bound matters more than the output of a run that failed.
            pass
        raise
    except BaseException:
        _kill_tree(proc)
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def label_batch(
    store: "MemoryStore",
    batch_id: str,
    *,
    claude: str,
    run: Runner | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run the classifier on one open batch and cache its labels.

    Returns a log record whose `outcome` is one of the `OUTCOME_*` values.
    Writes labels only on `accepted`; every other outcome writes nothing.
    Never raises: an exception from the store, such as a lock held past
    its timeout, is `store_error`, and the record keeps whatever it had
    gathered, including the cost of a run that already happened.
    """
    record: dict[str, object] = {"batch_id": batch_id}
    try:
        _label_batch(store, batch_id, record, claude=claude, run=run, env=env)
    except Exception as exc:
        record["outcome"] = OUTCOME_STORE_ERROR
        record["error"] = type(exc).__name__
    return record


def _label_batch(
    store: "MemoryStore",
    batch_id: str,
    record: dict[str, object],
    *,
    claude: str,
    run: Runner | None,
    env: dict[str, str] | None,
) -> None:
    """`label_batch`'s body. Fills `record` in place, so a raise keeps it."""
    from aelfrice import core_gate  # noqa: PLC0415
    from aelfrice.store import CoreGateAcceptRefused  # noqa: PLC0415

    try:
        version = core_gate.current_classifier_version()
    except core_gate.ClassifierUnavailable:
        record["outcome"] = OUTCOME_STALE_VERSION
        return
    batch = store.get_core_gate_batch(batch_id)
    if batch is None:
        record["outcome"] = OUTCOME_NO_BATCH
        return
    if batch.accepted_at is not None:
        record["outcome"] = OUTCOME_NOT_OPEN
        return
    if batch.classifier_version != version:
        record["outcome"] = OUTCOME_STALE_VERSION
        return
    snippets: list[tuple[int, str]] = []
    for item in sorted(batch.items, key=lambda i: i.index):
        belief = store.get_belief(item.belief_id)
        if belief is None or belief.content_hash != item.content_hash:
            # The label is cached under the hash, so the model must see
            # the content that hash names.
            record["outcome"] = OUTCOME_STALE_BELIEF
            return
        snippets.append((item.index, belief.content))
    prompt = core_gate.build_prompt(snippets)
    runner = run if run is not None else run_bounded
    platform_kwargs: dict[str, Any] = {}
    if _is_win32():
        # This process has no console (`spawn`), so a console program it
        # starts would get a new, visible one without this flag.
        platform_kwargs["creationflags"] = _creationflags("CREATE_NO_WINDOW")
    started = time.monotonic()
    try:
        with settings_file() as settings:
            # `encoding`: the CLI writes UTF-8; `text=True` alone would
            # decode it through the ANSI code page on Windows (#1441).
            proc = runner(
                runner_argv(claude, settings), input=prompt, capture_output=True,
                text=True, encoding="utf-8", timeout=RUNNER_TIMEOUT_SECONDS,
                env=runner_env(dict(env if env is not None else os.environ)),
                **platform_kwargs,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        record["outcome"] = OUTCOME_RUN_FAILED
        record["error"] = type(exc).__name__
        return
    record["seconds"] = round(time.monotonic() - started, 1)
    try:
        out = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        out = None
    if isinstance(out, dict):
        cost = cast(dict[str, Any], out).get("total_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            record["cost_usd"] = cost
    if proc.returncode != 0 or not isinstance(out, dict):
        record["outcome"] = OUTCOME_RUN_FAILED
        record["returncode"] = proc.returncode
        return
    result = cast(dict[str, Any], out).get("result")
    if not isinstance(result, str):
        record["outcome"] = OUTCOME_BAD_REPLY
        return
    try:
        labels = core_gate.parse_labels(
            result, {item.index for item in batch.items},
        )
    except ValueError:
        record["outcome"] = OUTCOME_BAD_REPLY
        return
    try:
        store.accept_core_gate_batch(
            batch_id, labels, classifier_version=version,
            accepted_at=datetime.now(timezone.utc).isoformat(),
        )
    except CoreGateAcceptRefused:
        record["outcome"] = OUTCOME_REFUSED
        return
    record["outcome"] = OUTCOME_ACCEPTED
    record["labels"] = len(labels)
    return


def _append_log(store_path: str, record: dict[str, object]) -> None:
    """One JSON line per run beside the store. Holds no snippet text."""
    if store_path == ":memory:":
        return
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    try:
        with open(Path(store_path).parent / LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        # A diagnostic only: an unwritable log must not fail the run or the
        # hook. The batch stays open either way.
        pass


def spawn(batch_id: str, *, db_path: str, claude: str) -> bool:
    """Start the detached child for `batch_id`. True iff it was launched.

    The child gets the hook's store path and `claude` path explicitly, so
    it doesn't depend on resolving them again. Never raises.
    """
    env = dict(os.environ)
    env["AELFRICE_DB"] = db_path
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL, "close_fds": True, "env": env,
    }
    if _is_win32():
        # Popen ignores `start_new_session` on Windows. Detached: no
        # console, so no window; its own process group, so the host's
        # Ctrl+C doesn't reach it (#1761).
        kwargs["creationflags"] = _creationflags(
            "DETACHED_PROCESS", "CREATE_NO_WINDOW", "CREATE_NEW_PROCESS_GROUP",
        )
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(  # noqa: S603 - args are package-internal
            # `-P`: `python -m` would otherwise put the working directory,
            # the project the hook ran in, first on the import path, so an
            # `aelfrice` package there would run instead of this one.
            [sys.executable, "-P", "-m", "aelfrice.core_gate_background", batch_id, claude],
            **kwargs,
        )
    except Exception:
        return False
    return True


def report_spawn_failure(store_path: str, batch_id: str, stderr: IO[str]) -> None:
    """One line to the hook's stderr, and one log record, when `spawn` fails.

    The batch stays open, as it does after a failed run. The record is
    written first, and a stderr that can't be written to is ignored, so
    neither stops the other or raises into the hook.
    """
    _append_log(store_path, {"batch_id": batch_id, "outcome": OUTCOME_SPAWN_FAILED})
    try:
        print(
            f"aelfrice hook: could not start background core-gate labeling for "
            f"batch {batch_id}; `aelf doctor core-gate --emit` batches its beliefs again",
            file=stderr,
        )
    except (OSError, ValueError):
        # A closed or broken stderr: the log record above is already written.
        pass


def main(argv: list[str] | None = None) -> int:
    """The detached child: `python -m aelfrice.core_gate_background BATCH CLAUDE`."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        return 2
    batch_id, claude = args
    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    path = str(db_path())
    try:
        store = MemoryStore(path)
    except Exception as exc:
        _append_log(path, {"batch_id": batch_id, "outcome": OUTCOME_RUN_FAILED,
                           "error": type(exc).__name__})
        return 1
    record: dict[str, object]
    try:
        record = label_batch(store, batch_id, claude=claude)
    except Exception as exc:  # never let the child die silently without a trace
        record = {"batch_id": batch_id, "outcome": OUTCOME_RUN_FAILED,
                  "error": type(exc).__name__}
    finally:
        store.close()
    _append_log(path, record)
    return 0 if record.get("outcome") == OUTCOME_ACCEPTED else 1


if __name__ == "__main__":
    sys.exit(main())
