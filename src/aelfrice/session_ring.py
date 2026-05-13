"""Per-session belief-injection dedup ring (#740).

Tracks the belief IDs that have already been injected into the agent's
prompt context during the current session, so subsequent
``PreToolUse:Grep|Glob|Bash`` hook fires can suppress redundant
re-injection of beliefs the agent has already seen.

Persistence: a single JSON file at
``<git-common-dir>/aelfrice/session_injected_ids.json`` (sibling of
``memory.db`` and ``session_first_prompt.json``). Bounded rolling window
(default 200 IDs, FIFO eviction) keyed by ``session_id`` — a new session
wipes the ring entirely.

Locked beliefs are exempt: they are tagged on insertion and always count
as ``new`` when filtering so the session-start guarantee (locked beliefs
re-ship on every fire) is preserved.

Concurrency: read-modify-write of the JSON file is serialized by an
``fcntl.LOCK_EX`` advisory lock on a sibling ``.session-ring.lock``
file. Multiple hook processes (UPS + PreToolUse) can fire near
simultaneously; the lock keeps appends from clobbering each other. The
critical section is tiny (parse JSON, append, atomic write) so no
timeout is needed.

All public functions are fail-soft: filesystem, JSON, or lock errors
return a no-op result and never raise. The injection lane is not
allowed to break the hook.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from aelfrice.db_paths import db_path

SESSION_RING_FILENAME: Final[str] = "session_injected_ids.json"
"""Sentinel filename; sibling of ``memory.db``."""

SESSION_RING_LOCK_FILENAME: Final[str] = ".session-ring.lock"
"""Advisory-lock filename; sibling of the sentinel."""

DEFAULT_RING_MAX: Final[int] = 200
"""Default rolling-window cap. Override via ``AELFRICE_INJECTION_RING_MAX``.

Sized so a heavy turn (1 UPS + 5 Grep + 3 Bash-search) emitting up to
~12 beliefs each comfortably fits a multi-turn dedup window without
evicting beliefs from earlier turns the agent may still be reasoning
about.
"""

RING_MAX_ENV: Final[str] = "AELFRICE_INJECTION_RING_MAX"
"""Env var overriding the rolling-window cap. Must parse as a positive int."""


def _resolve_ring_max() -> int:
    raw = os.environ.get(RING_MAX_ENV, "").strip()
    if not raw:
        return DEFAULT_RING_MAX
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_RING_MAX
    if n < 1:
        return DEFAULT_RING_MAX
    return n


@dataclass(frozen=True)
class RingFilterResult:
    """Outcome of :func:`filter_against_ring`.

    Attributes:
        new_beliefs: Beliefs not present in the ring (or locked — locked
            beliefs always pass through as ``new`` to preserve the
            session-start guarantee).
        recent_ids: IDs of the input beliefs that are already in the
            ring and not locked. Surface for "N more matches already in
            context" callouts.
        latest_fire_idx: The largest ``fire_idx`` among ``recent_ids`` in
            the ring, or -1 when ``recent_ids`` is empty. Lets emitters
            point at "injected at turn N".
    """

    new_beliefs: list[Any]
    recent_ids: list[str]
    latest_fire_idx: int


def _session_ring_path() -> Path | None:
    """Return the ring sentinel path, or ``None`` for in-memory DBs."""
    try:
        p = db_path()
    except Exception:
        return None
    if str(p) == ":memory:":
        return None
    return p.parent / SESSION_RING_FILENAME


def _session_ring_lock_path(ring_path: Path) -> Path:
    return ring_path.parent / SESSION_RING_LOCK_FILENAME


def _read_ring_unlocked(ring_path: Path) -> dict[str, Any]:
    """Parse the ring file. Returns an empty dict on missing / malformed."""
    if not ring_path.exists():
        return {}
    try:
        data = json.loads(ring_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _atomic_write(ring_path: Path, payload: str) -> None:
    """Tempfile + ``os.replace`` write. Caller holds the advisory lock."""
    ring_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=ring_path.name + ".",
        suffix=".tmp",
        dir=str(ring_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ring_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _open_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)


def _normalize_for_session(
    data: dict[str, Any], session_id: str, ring_max: int
) -> dict[str, Any]:
    """Return ``data`` shaped for ``session_id``.

    If the stored ``session_id`` differs (or is missing), start a fresh
    ring. Otherwise normalize the existing ring shape (clamp ring_max,
    coerce missing fields, trim oversize).
    """
    stored_sid = data.get("session_id") if isinstance(data, dict) else None
    if not isinstance(stored_sid, str) or stored_sid != session_id:
        return {
            "session_id": session_id,
            "ring": [],
            "ring_max": ring_max,
            "next_fire_idx": 0,
            "evicted_total": 0,
        }
    ring = data.get("ring")
    if not isinstance(ring, list):
        ring = []
    next_idx = data.get("next_fire_idx")
    if not isinstance(next_idx, int) or next_idx < 0:
        next_idx = 0
    evicted = data.get("evicted_total")
    if not isinstance(evicted, int) or evicted < 0:
        evicted = 0
    return {
        "session_id": session_id,
        "ring": [e for e in ring if isinstance(e, dict) and isinstance(e.get("id"), str)],
        "ring_max": ring_max,
        "next_fire_idx": next_idx,
        "evicted_total": evicted,
    }


def filter_against_ring(
    session_id: str | None,
    beliefs: list[Any],
    *,
    locked_ids: set[str] | None = None,
    stderr: IO[str] | None = None,
) -> RingFilterResult:
    """Partition ``beliefs`` into ``new`` vs ``recent`` against the ring.

    A belief is ``recent`` iff its id is present in the session's ring AND
    not in ``locked_ids``. Locked beliefs are never deduped — the
    session-start guarantee requires them on every fire.

    Read-only: does not mutate the ring file. Use :func:`append_ids` after
    emitting the (possibly filtered) block to record what was injected.

    Returns an empty-filter result (everything ``new``) when:

    - ``session_id`` is empty / None,
    - the ring sentinel does not exist (first fire of session),
    - the ring file is malformed,
    - the stored ``session_id`` differs (cross-session — fresh ring),
    - any I/O error occurs.
    """
    if not session_id:
        return RingFilterResult(new_beliefs=list(beliefs), recent_ids=[], latest_fire_idx=-1)
    ring_path = _session_ring_path()
    if ring_path is None:
        return RingFilterResult(new_beliefs=list(beliefs), recent_ids=[], latest_fire_idx=-1)
    locked = locked_ids if locked_ids is not None else set()
    try:
        data = _read_ring_unlocked(ring_path)
    except Exception as exc:  # pragma: no cover — defensive
        _warn(stderr, f"session_ring: read failed (non-fatal): {exc}")
        return RingFilterResult(new_beliefs=list(beliefs), recent_ids=[], latest_fire_idx=-1)
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        return RingFilterResult(new_beliefs=list(beliefs), recent_ids=[], latest_fire_idx=-1)
    ring = data.get("ring") if isinstance(data.get("ring"), list) else []
    id_to_fire: dict[str, int] = {}
    for entry in ring:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("id")
        fire_idx = entry.get("fire_idx")
        if isinstance(bid, str) and bid:
            id_to_fire[bid] = fire_idx if isinstance(fire_idx, int) else -1
    new_beliefs: list[Any] = []
    recent_ids: list[str] = []
    latest_fire = -1
    for b in beliefs:
        bid = getattr(b, "id", None)
        if not isinstance(bid, str) or not bid:
            new_beliefs.append(b)
            continue
        if bid in locked:
            new_beliefs.append(b)
            continue
        if bid in id_to_fire:
            recent_ids.append(bid)
            fire_idx = id_to_fire[bid]
            if fire_idx > latest_fire:
                latest_fire = fire_idx
            continue
        new_beliefs.append(b)
    return RingFilterResult(
        new_beliefs=new_beliefs,
        recent_ids=recent_ids,
        latest_fire_idx=latest_fire,
    )


def append_ids(
    session_id: str | None,
    ids: list[str],
    *,
    locked_ids: set[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Append ``ids`` to the ring under ``session_id``. Returns next fire_idx.

    A new session_id (or missing ring file) starts a fresh ring. IDs
    already in the ring have their ``fire_idx`` refreshed to the new
    value (so the FIFO eviction order tracks most-recent-injection, not
    first-injection). Locked IDs are tagged ``locked: true`` in the ring
    so future filter calls treat them as exempt regardless of caller
    intent.

    Eviction: when the ring exceeds ``ring_max`` after the append,
    oldest entries (lowest ``fire_idx``) are evicted FIFO until the cap
    holds. The total evicted count is tracked in ``evicted_total`` for
    telemetry.

    Returns the ``next_fire_idx`` written to the ring (i.e. one past the
    fire_idx these IDs were tagged with). Returns -1 on any error,
    short-circuiting the caller without raising. Empty ``ids`` is a
    no-op and returns the current ``next_fire_idx`` (or 0 if the ring
    was just created).
    """
    if not session_id:
        return -1
    ring_path = _session_ring_path()
    if ring_path is None:
        return -1
    ring_max = _resolve_ring_max()
    lock_path = _session_ring_lock_path(ring_path)
    try:
        lock_fd = _open_lock(lock_path)
    except OSError as exc:
        _warn(stderr, f"session_ring: lock open failed (non-fatal): {exc}")
        return -1
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as exc:
            _warn(stderr, f"session_ring: flock failed (non-fatal): {exc}")
            return -1
        try:
            data = _read_ring_unlocked(ring_path)
            data = _normalize_for_session(data, session_id, ring_max)
            fire_idx = int(data["next_fire_idx"])
            ring_list: list[dict[str, Any]] = data["ring"]
            locked = locked_ids if locked_ids is not None else set()
            # Index existing entries by id so we can refresh in place.
            id_to_pos = {
                entry["id"]: i
                for i, entry in enumerate(ring_list)
                if isinstance(entry, dict) and isinstance(entry.get("id"), str)
            }
            for bid in ids:
                if not isinstance(bid, str) or not bid:
                    continue
                entry: dict[str, Any] = {
                    "id": bid,
                    "fire_idx": fire_idx,
                    "locked": bid in locked,
                }
                if bid in id_to_pos:
                    ring_list[id_to_pos[bid]] = entry
                else:
                    ring_list.append(entry)
                    id_to_pos[bid] = len(ring_list) - 1
            # FIFO evict oldest until cap holds.
            evicted = 0
            if len(ring_list) > ring_max:
                # Stable sort by fire_idx ascending; keep last ``ring_max``.
                ring_list.sort(key=lambda e: (e.get("fire_idx", -1), 0))
                evicted = len(ring_list) - ring_max
                ring_list[:] = ring_list[-ring_max:]
            next_fire_idx = fire_idx + 1
            data["ring"] = ring_list
            data["next_fire_idx"] = next_fire_idx
            data["evicted_total"] = int(data.get("evicted_total", 0)) + evicted
            data["ring_max"] = ring_max
            _atomic_write(ring_path, json.dumps(data))
            return next_fire_idx
        except Exception as exc:
            _warn(stderr, f"session_ring: append failed (non-fatal): {exc}")
            return -1
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                # Best-effort cleanup; lock_fd may already be closed.
                pass
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def read_ring_state(session_id: str | None) -> dict[str, Any]:
    """Return the ring shape for ``session_id``, or ``{}`` when absent.

    Read-only. Surface for ``aelf doctor`` telemetry — callers should
    treat the dict as opaque and read ``ring`` length, ``ring_max``, and
    ``evicted_total`` for display.
    """
    if not session_id:
        return {}
    ring_path = _session_ring_path()
    if ring_path is None:
        return {}
    data = _read_ring_unlocked(ring_path)
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        return {}
    return data


def read_ring_file() -> dict[str, Any]:
    """Return the raw ring shape, ignoring session_id matching.

    Surface for ``aelf doctor`` which has no session_id at hand but
    still wants to display whatever ring is currently persisted.
    Returns ``{}`` when the sentinel does not exist or is malformed.
    """
    ring_path = _session_ring_path()
    if ring_path is None:
        return {}
    if not ring_path.exists():
        return {}
    data = _read_ring_unlocked(ring_path)
    if not isinstance(data, dict):
        return {}
    return data


def _warn(stderr: IO[str] | None, msg: str) -> None:
    print(msg, file=stderr if stderr is not None else sys.stderr)
