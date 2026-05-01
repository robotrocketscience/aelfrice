"""Implicit retrieval-driven feedback sweeper (#191, T2 of phantom-prereqs).

`apply_feedback` is invoked rarely in normal use, but `retrieve()`
fires constantly. This module turns retrieval exposure into a small,
deferred posterior signal: the retrieval path enqueues one row per
surfaced belief; a periodic sweeper (CLI: `aelf sweep-feedback`)
applies `+epsilon` to each belief whose grace window has elapsed
without an explicit correction or contradiction landing on it.

Contracts (see issue #191 for full spec):

  * `T_grace`: enqueue_at + T_grace must be <= now before a row is
    eligible. Default 1800 s (30 min). Configurable via
    `[implicit_feedback] grace_window_seconds` in `.aelfrice.toml`
    or `AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS` env var.
  * `epsilon`: alpha increment per applied row. Default 0.05.
    Configurable via `[implicit_feedback] epsilon` /
    `AELFRICE_IMPLICIT_FEEDBACK_EPSILON`.
  * `RETRIEVAL_DRIVEN_FEEDBACK_SOURCE`: the source string written to
    feedback_history for every applied row. Distinct from any
    explicit-user-feedback source so the audit trail can split
    explicit vs implicit signals.
  * Cancellation: any feedback_history row for the same belief whose
    `source` is NOT `RETRIEVAL_DRIVEN_FEEDBACK_SOURCE` and whose
    `created_at` is in [enqueued_at, now] cancels the pending row
    (no alpha change). This implements the "explicit beats implicit"
    + "contradiction within grace" contracts in one query, since
    contradiction-tiebreaker resolutions also write to
    feedback_history with a distinct source prefix.
  * Idempotency: only `status='enqueued'` rows are processed.
    `applied`/`cancelled` rows are skipped on re-run.
  * Atomicity per row: belief alpha update + feedback_history insert
    + queue status update share a single explicit transaction. A
    crash mid-row leaves the row `enqueued` (the alpha update is
    rolled back with the rest); re-run applies it once and only once.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Final

import tomllib

from aelfrice.store import MemoryStore

# --- Public constants ---------------------------------------------------

DEFAULT_T_GRACE_SECONDS: Final[int] = 1800
DEFAULT_EPSILON: Final[float] = 0.05

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
IMPLICIT_FEEDBACK_SECTION: Final[str] = "implicit_feedback"
GRACE_KEY: Final[str] = "grace_window_seconds"
EPSILON_KEY: Final[str] = "epsilon"
ENQUEUE_KEY: Final[str] = "enqueue_on_retrieve"

ENV_GRACE: Final[str] = "AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS"
ENV_EPSILON: Final[str] = "AELFRICE_IMPLICIT_FEEDBACK_EPSILON"
ENV_ENQUEUE: Final[str] = "AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE"

EVENT_RETRIEVAL_EXPOSURE: Final[str] = "retrieval_exposure"
RETRIEVAL_DRIVEN_FEEDBACK_SOURCE: Final[str] = "retrieval_driven_feedback"

_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


# --- Result shape -------------------------------------------------------


@dataclass
class SweepResult:
    """Outcome of one `sweep_deferred_feedback` invocation.

    `applied` and `cancelled` count rows whose status transitioned
    during this call. `skipped_no_belief` counts rows whose belief_id
    no longer resolves (the belief was deleted between enqueue and
    sweep) — those rows are marked cancelled so the queue drains.
    """

    applied: int = 0
    cancelled: int = 0
    skipped_no_belief: int = 0
    pending_unmet_grace: int = 0
    epsilon_used: float = 0.0
    grace_seconds_used: int = 0
    applied_belief_ids: list[str] = field(default_factory=list)
    cancelled_belief_ids: list[str] = field(default_factory=list)


# --- Time helpers -------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp; tolerate both `Z` and `+00:00` suffixes."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


# --- Config resolution --------------------------------------------------


def _read_toml_value(
    key: str, *, start: Path | None = None
) -> Any:  # noqa: ANN401 - typed by callers
    """Walk up from `start` finding `[implicit_feedback] <key>`. Returns
    the raw TOML value, or None when no file / no key. Fail-soft."""
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            try:
                raw = candidate.read_bytes()
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice implicit_feedback: cannot read {candidate}: "
                    f"{exc}",
                    file=serr,
                )
                return None
            section = parsed.get(IMPLICIT_FEEDBACK_SECTION, {})
            if not isinstance(section, dict):
                return None
            return section.get(key)  # type: ignore[no-any-return]
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_grace_seconds(
    explicit: int | None = None, *, start: Path | None = None
) -> int:
    """Env > kwarg > TOML > DEFAULT_T_GRACE_SECONDS. Negative clamps to 0."""
    raw_env = os.environ.get(ENV_GRACE)
    if raw_env is not None and raw_env.strip():
        try:
            return max(0, int(raw_env.strip()))
        except ValueError:
            print(
                f"aelfrice implicit_feedback: ignoring {ENV_GRACE}={raw_env!r}"
                " (expected int)",
                file=sys.stderr,
            )
    if explicit is not None:
        return max(0, int(explicit))
    toml_v = _read_toml_value(GRACE_KEY, start=start)
    if isinstance(toml_v, int) and not isinstance(toml_v, bool):
        return max(0, toml_v)
    return DEFAULT_T_GRACE_SECONDS


def resolve_epsilon(
    explicit: float | None = None, *, start: Path | None = None
) -> float:
    """Env > kwarg > TOML > DEFAULT_EPSILON. Negative clamps to 0.0."""
    raw_env = os.environ.get(ENV_EPSILON)
    if raw_env is not None and raw_env.strip():
        try:
            return max(0.0, float(raw_env.strip()))
        except ValueError:
            print(
                f"aelfrice implicit_feedback: ignoring {ENV_EPSILON}={raw_env!r}"
                " (expected float)",
                file=sys.stderr,
            )
    if explicit is not None:
        return max(0.0, float(explicit))
    toml_v = _read_toml_value(EPSILON_KEY, start=start)
    if isinstance(toml_v, bool):
        return DEFAULT_EPSILON
    if isinstance(toml_v, (int, float)):
        return max(0.0, float(toml_v))
    return DEFAULT_EPSILON


def is_enqueue_on_retrieve_enabled(
    explicit: bool | None = None, *, start: Path | None = None
) -> bool:
    """Env > kwarg > TOML > default True. Default-on because the queue is
    additive (no consumer reads it until the sweeper runs); operators can
    flip it off without losing any other functionality."""
    raw_env = os.environ.get(ENV_ENQUEUE)
    if raw_env is not None:
        norm = raw_env.strip().lower()
        if norm in _ENV_FALSY:
            return False
        if norm in _ENV_TRUTHY:
            return True
    if explicit is not None:
        return explicit
    toml_v = _read_toml_value(ENQUEUE_KEY, start=start)
    if isinstance(toml_v, bool):
        return toml_v
    return True


# --- Enqueue path -------------------------------------------------------


def enqueue_retrieval_exposures(
    store: MemoryStore,
    belief_ids: list[str],
    *,
    now: str | None = None,
) -> int:
    """Enqueue one `retrieval_exposure` row per belief_id. Returns the
    count of rows inserted. A single shared `enqueued_at` keeps the
    grace window well-defined for a batch from one retrieve() call."""
    if not belief_ids:
        return 0
    ts = now if now is not None else _utc_now_iso()
    n = 0
    for bid in belief_ids:
        store.enqueue_deferred_feedback(
            bid,
            event_type=EVENT_RETRIEVAL_EXPOSURE,
            enqueued_at=ts,
        )
        n += 1
    return n


# --- Sweeper ------------------------------------------------------------


def sweep_deferred_feedback(
    store: MemoryStore,
    *,
    now: str | None = None,
    grace_seconds: int | None = None,
    epsilon: float | None = None,
    limit: int = 10_000,
    config_start: Path | None = None,
) -> SweepResult:
    """Process pending queue rows whose grace window has elapsed.

    For each pending row with `enqueued_at <= now - grace_seconds`:

      * If `feedback_history` has an entry for this belief whose source
        is not `RETRIEVAL_DRIVEN_FEEDBACK_SOURCE` and whose created_at
        is in `[enqueued_at, now]`, mark `cancelled` (no posterior
        change). This covers explicit user feedback AND contradiction
        tiebreaker events in one query.
      * Else apply `+epsilon` to belief.alpha, write a feedback_history
        row with source=RETRIEVAL_DRIVEN_FEEDBACK_SOURCE, and mark
        `applied`. The three writes share one transaction so a crash
        mid-row leaves the queue row `enqueued` and the alpha
        unchanged — re-run applies once.
      * If the belief no longer exists, mark `cancelled` and count
        toward `skipped_no_belief` so the queue drains.

    Idempotent: only `enqueued` rows are touched; re-running the
    sweeper over the resulting state is a no-op for the rows it
    already processed."""
    grace_eff = (
        grace_seconds
        if grace_seconds is not None
        else resolve_grace_seconds(start=config_start)
    )
    eps_eff = (
        epsilon
        if epsilon is not None
        else resolve_epsilon(start=config_start)
    )
    now_iso = now if now is not None else _utc_now_iso()
    cutoff_dt = _parse_iso(now_iso) - timedelta(seconds=grace_eff)
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    result = SweepResult(
        epsilon_used=eps_eff,
        grace_seconds_used=grace_eff,
    )

    # Pending = enqueued with grace elapsed.
    pending = store.list_pending_deferred_feedback(
        cutoff_iso=cutoff_iso, limit=limit
    )

    # Count rows still in their grace window separately.
    by_status = store.count_deferred_feedback_by_status()
    enqueued_total = by_status.get("enqueued", 0)
    result.pending_unmet_grace = max(0, enqueued_total - len(pending))

    conn = store._conn  # noqa: SLF001 - intentional, atomic per row

    for row_id, belief_id, enqueued_at, _event_type in pending:
        belief = store.get_belief(belief_id)
        if belief is None:
            # Drain; nothing to update.
            conn.execute(
                "UPDATE deferred_feedback_queue "
                "SET status='cancelled', applied_at=? WHERE id=?",
                (now_iso, row_id),
            )
            conn.commit()
            result.skipped_no_belief += 1
            result.cancelled += 1
            result.cancelled_belief_ids.append(belief_id)
            continue

        cancelled = store.has_explicit_feedback_in_window(
            belief_id,
            window_start_iso=enqueued_at,
            window_end_iso=now_iso,
            retrieval_source=RETRIEVAL_DRIVEN_FEEDBACK_SOURCE,
        )

        # Single explicit transaction wrapping the row's writes.
        # `BEGIN IMMEDIATE` acquires the write lock up-front so a
        # concurrent writer cannot interleave.
        conn.execute("BEGIN IMMEDIATE")
        try:
            if cancelled:
                conn.execute(
                    "UPDATE deferred_feedback_queue "
                    "SET status='cancelled', applied_at=? WHERE id=?",
                    (now_iso, row_id),
                )
                conn.execute("COMMIT")
                result.cancelled += 1
                result.cancelled_belief_ids.append(belief_id)
            else:
                conn.execute(
                    "UPDATE beliefs SET alpha = alpha + ? WHERE id = ?",
                    (eps_eff, belief_id),
                )
                conn.execute(
                    "INSERT INTO feedback_history "
                    "(belief_id, valence, source, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        belief_id,
                        eps_eff,
                        RETRIEVAL_DRIVEN_FEEDBACK_SOURCE,
                        now_iso,
                    ),
                )
                conn.execute(
                    "UPDATE deferred_feedback_queue "
                    "SET status='applied', applied_at=? WHERE id=?",
                    (now_iso, row_id),
                )
                conn.execute("COMMIT")
                result.applied += 1
                result.applied_belief_ids.append(belief_id)
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return result
