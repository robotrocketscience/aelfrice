"""#1513 — warm the BM25 sidecar off the user-visible path.

`benchmarks/sidecar_rebuild_rate.py`, bucketed by position within the
session, showed that the full-rebuild cost is a **session-first tail**, not
an average::

    session-FIRST fires:  full_rebuild 5, fresh  8            ->  5/13 = 38.5%
    LATER fires:          full_rebuild 0, fresh 23, incr 3    ->  0/26 =  0.0%

Every rebuild in that sample was the first scored fire of a session, and it
cost between 347 ms and 1825 ms — paid on the prompt where a user is least
willing to wait.

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
bucket is already 0/26. A warm that runs on the second fire arrives after
the only fire that ever pays.

## Fail-soft

Both halves are fail-soft, and neither can change what the hook emits. The
parent swallows every spawn error and returns False. The child writes nothing
but the sidecar blob (which `BM25IndexCache._write_sidecar` replaces
atomically, so a concurrent reader sees the old blob or the new one, never a
torn one) and swallows every exception. A warm that fails leaves behaviour
exactly as it is today: the next fire rebuilds, as it does now.
"""
from __future__ import annotations

import os
from typing import Final

WARM_DISABLE_ENV: Final[str] = "AELF_NO_SIDECAR_WARM"
"""Set truthy to suppress the spawn. The escape hatch for anyone who does
not want a second process touching the store behind their session."""

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

    The index is built through the *same* helper and the *same* resolvers
    that the L1 lane in `retrieval` uses. That is load-bearing rather than
    tidy: a sidecar written under different tokenisation parameters is
    rejected by `_load_sidecar` as describing different documents, so a warm
    that resolved its own parameters would pay the whole expensive build and
    still leave the next fire rebuilding.
    """
    try:
        import time  # noqa: PLC0415

        from aelfrice.db_paths import db_path  # noqa: PLC0415
        from aelfrice.retrieval import (  # noqa: PLC0415
            _store_scoped_bm25f_cache,
            resolve_bm25_b_anchor,
            resolve_bm25_k3,
            resolve_bm25f_anchor_weight_with_meta,
            resolve_bm25f_per_field,
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
            cache = _store_scoped_bm25f_cache(
                store,
                anchor_weight=resolve_bm25f_anchor_weight_with_meta(
                    store, now_ts=int(time.time()),
                ),
                k3=resolve_bm25_k3(),
                per_field=resolve_bm25f_per_field(),
                b_anchor=resolve_bm25_b_anchor(),
            )
            cache.get()
        finally:
            store.close()
        return last_sidecar_outcome()
    except Exception:
        # Fail-soft (AC4). The child is detached and its streams go to
        # /dev/null, so a traceback would reach nobody; the observable
        # contract is that the next fire behaves exactly as it does today.
        return None


def main() -> int:
    """Console entry for the detached child. Always returns 0."""
    warm_sidecar()
    return 0
