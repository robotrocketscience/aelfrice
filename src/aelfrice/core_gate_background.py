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
- `--settings '{"disableAllHooks": true}'` turns hooks off as well, so
  the run can't fire aelfrice's hooks and recurse.
- `--no-session-persistence` keeps the run out of the session history.
- Thinking stays on: with it off, agreement with the reference labels fell
  by about 19 points.
- `--max-budget-usd 0.10` caps each run (operator ruling, 2026-10-08).

The labels are the classifier's own: the prompt is `core_gate.build_prompt`
and the reply goes through `core_gate.parse_labels`, the same parser
`aelf core-gate accept` uses. The child's environment drops any API key,
so the run always uses the `claude` login (`runner_env`). It doesn't run
on Windows yet (`find_claude`).

Never raises into the hook. A failed run leaves the batch open; an open
batch claims its beliefs, so no later Stop asks about them, and
`aelf doctor core-gate --emit` batches them again. Unlabeled beliefs keep
today's core rule, so a failure costs nothing but the label.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Final, cast

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
RUNNER_SYSTEM_PROMPT: Final[str] = (
    "You label short text snippets. Follow the user's instructions exactly."
)
LOG_FILENAME: Final[str] = "core_gate_background.jsonl"

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

    Always None on Windows until the path is verified there (operator
    ruling on #1741, 2026-10-08): a detached child may open a console
    window, and an npm `claude.cmd` shim passes the JSON `--settings`
    argument through cmd.exe quoting. Windows then behaves as 5.1.1 did.
    """
    if sys.platform == "win32":
        return None
    return shutil.which(CLAUDE_BIN)


def runner_argv(claude: str) -> list[str]:
    """The measured classifier invocation (see the module docstring)."""
    return [
        claude, "-p",
        "--model", RUNNER_MODEL,
        "--tools", "",
        "--setting-sources", "",
        "--settings", '{"disableAllHooks": true}',
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
    runner = run if run is not None else subprocess.run
    started = time.monotonic()
    try:
        proc = runner(
            runner_argv(claude), input=prompt, capture_output=True, text=True,
            timeout=RUNNER_TIMEOUT_SECONDS,
            env=runner_env(dict(env if env is not None else os.environ)),
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
    # POSIX only: `find_claude` reports no CLI on Windows, so the hook
    # never gets here there.
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
