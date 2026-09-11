"""#1513 — warm the BM25 sidecar off the user-visible path.

`benchmarks/sidecar_rebuild_rate.py`, bucketed by position within the
session, shows that the full-rebuild cost is a **session-first tail**, not an
average: the `full_rebuild` rate on the first scored fire of a session runs
materially above the rate on every later fire. That is the prompt a user is
least willing to wait on.

No rate or latency figure is quoted here, deliberately. The population is a
live, growing, single-slot-rotating audit log; successive re-derivations
weeks apart moved the session-first magnitude substantially while the sign
never flipped, and the rows the earlier runs scored are gone to rotation, so
the numbers this docstring used to carry were not reproducible by anyone.
Re-derive the split instead — the script's own docstring gives the command,
and says which population each form of it covers.

## Why a detached process, and not the two alternatives

**A detached child spawned from `SessionStart`.** `SessionStart` fires before
any user message, so the child has the whole of the user's first typing pause
to build in. The parent pays one `Popen`, and every decision that costs
anything — is the L1 lane even on, is the sidecar already fresh, what
tokenisation parameters apply — is made *inside the child*, so the
user-visible path never opens the store or imports numpy on this account.

*A thread the hook does not join* was rejected on the failure mode. The hook
is a short-lived process: a non-daemon thread keeps the interpreter alive
until the build finishes, so the hook process does not exit until the
rebuild does, and the host waits on the hook process — that MOVES the
latency into `SessionStart` rather than removing it, which AC2 names
explicitly. Making it a daemon thread inverts the failure: interpreter exit
kills it mid-build, the sidecar is never written, and the warm silently does
nothing. Neither arm survives.

*Warming lazily on a later fire* was rejected on the measurement. The later
bucket already runs far below the session-first one. A warm that runs on the
second fire arrives after the fire that pays.

## Fail-soft

Both halves are fail-soft, and neither can change what the hook emits. The
parent swallows every spawn error and returns False. The child swallows every
exception too — once in `warm_sidecar`, again in `_record_warm_outcome`. A
warm that fails leaves behaviour exactly as it is today: the next fire
rebuilds, as it does now.

## What the child writes

Fail-soft is not the same as read-only, and this child is not read-only. It
writes four things, on a machine the user is not watching:

- **The database's parent directory.** `warm_sidecar` calls
  `mkdir(parents=True, exist_ok=True)` on it before opening the store.
- **The store itself.** `MemoryStore(str(p))` is opened read-write, and
  `MemoryStore.__init__` documents a bare open as a write: the DDL battery,
  any pending migrations, the `schema_meta` seed, `_resolve_local_scope_id`
  (which generates and persists an id on a store that has none) and, since
  #1314, `sweep_expired_locks`, which flips a user's expired locks to
  unlocked. `hook.py` calls the same open "a write (DDL plus migrations)".
  A session whose first prompt never arrives still pays that write here.
- **The sidecar blob**, through `BM25IndexCache._write_sidecar`: a temp file
  in the sidecar's own directory, then `os.replace`, so a concurrent reader
  sees the old blob or the new one, never a torn one.
- **One `sidecar_warm` row in `hook_audit.jsonl`**, appended by
  `_record_warm_outcome`. It is suppressed when `load_hook_audit_config()`
  reports the audit off — `AELFRICE_HOOK_AUDIT=0`, or `enabled = false`
  under `[hook_audit]` in `.aelfrice.toml`. That check is the only thing
  between this writer and a user who asked for no audit log, because the
  child runs detached and the user cannot see it refuse. The append can
  also rotate the log: `_append_audit` renames the live file to `.1` once
  the size cap is passed, overwriting any previous `.1`, and starts a fresh
  file with an `audit_rotation` marker.
"""
from __future__ import annotations

import os
from typing import Final

WARM_DISABLE_ENV: Final[str] = "AELF_NO_SIDECAR_WARM"
"""Set truthy to suppress the spawn. The escape hatch for anyone who does
not want a second process touching the store behind their session."""

WARM_AUDIT_HOOK: Final[str] = "sidecar_warm"
"""`hook` value on the row the child writes. Deliberately not
`user_prompt_submit`: see `_record_warm_outcome`."""

_CHILD_SOURCE: Final[str] = (
    "from aelfrice.sidecar_warm import main; raise SystemExit(main())"
)


def warm_disabled(env: dict[str, str] | None = None) -> bool:
    """True when `AELF_NO_SIDECAR_WARM` is set to a truthy value."""
    raw = (env if env is not None else os.environ).get(WARM_DISABLE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def spawn_sidecar_warm(env: dict[str, str] | None = None) -> bool:
    """Launch the detached warm child. Return True iff one was launched.

    Deliberately does no work of its own beyond the env check: it must not
    open the store, resolve a retrieval flag, or import numpy, because every
    one of those costs the caller the milliseconds this issue exists to
    remove. The child re-decides all of that for itself.

    Never raises. A spawn failure returns False and leaves the caller's
    behaviour unchanged — the next fire rebuilds exactly as it does today.
    """
    if warm_disabled(env):
        return False
    import subprocess  # noqa: PLC0415 — kept off the hook's import graph
    import sys  # noqa: PLC0415

    try:
        subprocess.Popen(  # noqa: S603 — args are package-internal
            [sys.executable, "-c", _CHILD_SOURCE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except Exception:
        # Fail-soft (AC4): a warm that cannot start must be invisible.
        return False


def warm_sidecar() -> str | None:
    """Build or refresh the sidecar. Return the outcome, or None.

    Runs in the detached child. Returns one of `fresh` / `incremental` /
    `full_rebuild` — the same vocabulary the audit log records, so a warm can
    be read against the fires it is meant to spare — or None when there was
    nothing to do (the L1 BM25F lane is off) or when anything failed.

    The index is built through `retrieval.bm25f_cache_for_lane`, which is
    the *same* call the L1 lane makes. That is load-bearing rather than
    tidy: a sidecar written under different tokenisation parameters is
    rejected by `_load_sidecar` as describing different documents, so a warm
    that resolved its own parameters would pay the whole expensive build and
    still leave the next fire rebuilding.

    Sharing the call is not on its own sufficient, because it makes the two
    sides run the same resolvers rather than see the same answer: under the
    #757 meta-belief `anchor_weight` decodes a decaying posterior, and this
    child's clock is minutes behind the fire's by design. The helper closes
    that gap by taking the weight off a fresh sidecar when one exists; see
    `retrieval.bm25f_cache_for_lane`.

    `now_ts` is this child's own wall clock and is not the lane's. Nothing
    it decides may depend on the two agreeing.
    """
    try:
        import time  # noqa: PLC0415

        from aelfrice.db_paths import db_path  # noqa: PLC0415
        from aelfrice.retrieval import (  # noqa: PLC0415
            bm25f_cache_for_lane,
            resolve_use_bm25f_anchors,
        )
        from aelfrice.sidecar_outcome import (  # noqa: PLC0415
            last_sidecar_outcome,
            reset_sidecar_outcome,
        )
        from aelfrice.store import MemoryStore  # noqa: PLC0415

        if not resolve_use_bm25f_anchors(None):
            # The lane that reads the sidecar is off, so building one would
            # be pure cost. Not a failure — there is nothing to warm.
            return None
        p = db_path()
        if str(p) == ":memory:":
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        reset_sidecar_outcome()
        store = MemoryStore(str(p))
        try:
            cache = bm25f_cache_for_lane(store, now_ts=int(time.time()))
            cache.get()
        finally:
            store.close()
        return last_sidecar_outcome()
    except Exception:
        # Fail-soft (AC4). The child is detached and its streams go to
        # /dev/null, so a traceback would reach nobody; the observable
        # contract is that the next fire behaves exactly as it does today.
        return None


def _record_warm_outcome(outcome: str | None) -> None:
    """Write one observability row for this warm. Never raises.

    Without it a warm that worked and a warm that never ran leave the same
    trace, which is none: the parent does not wait on the child, all three
    of the child's streams go to `/dev/null`, and its return code is
    discarded. A feature whose failure is indistinguishable from its success
    cannot be operated, so the child says what it did.

    The row reuses the hook-audit sink, the way `transcript_logger` does for
    the same reason, so `aelf tail` shows the warm beside the fires it is
    meant to spare and it inherits the log's rotation.

    `hook` is `sidecar_warm`, never `user_prompt_submit`: the rate in
    `benchmarks/sidecar_rebuild_rate.py` is per user-visible fire, and a
    warm is not one. A row claiming to be a UPS fire would enter that
    denominator and move the number the warm exists to move.

    `outcome` is None when there was nothing to warm (the L1 lane is off, or
    the store is in-memory) and also when the warm raised. Those are not the
    same event, but the child cannot tell them apart without re-running the
    work, so the row records the value it has and does not invent a
    distinction.
    """
    try:
        from datetime import datetime, timezone  # noqa: PLC0415

        from aelfrice.db_paths import db_path  # noqa: PLC0415
        from aelfrice.hook_audit import (  # noqa: PLC0415
            _append_audit,  # pyright: ignore[reportPrivateUsage]
            _audit_path_for_db,  # pyright: ignore[reportPrivateUsage]
            load_hook_audit_config,
        )

        cfg = load_hook_audit_config()
        if not cfg.enabled:
            return
        p = db_path()
        if str(p) == ":memory:":
            return
        record: dict[str, object] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hook": WARM_AUDIT_HOOK,
            "sidecar_outcome": outcome,
        }
        _append_audit(_audit_path_for_db(p), record, cfg.max_bytes)
    except Exception:
        # Fail-soft (AC4), like every other line in this module: an audit
        # write that fails must not change what the warm did.
        return


def main() -> int:
    """Console entry for the detached child. Always returns 0.

    The return code is not the signal. Nothing waits on this process and its
    streams are `/dev/null`, so the audit row `_record_warm_outcome` writes
    is the only thing that survives the child.
    """
    _record_warm_outcome(warm_sidecar())
    return 0
