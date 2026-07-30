"""Implicit retrieval-driven feedback sweeper (#191, T2 of phantom-prereqs).

`apply_feedback` is invoked rarely in normal use, but `retrieve()`
fires constantly. This module turns retrieval exposure into a small,
deferred posterior signal: the retrieval path enqueues one row per
surfaced belief; a periodic sweeper (CLI: `aelf sweep-feedback`)
applies `+epsilon` to each belief whose grace window has elapsed
without an explicit correction or contradiction landing on it.

**Enqueuing is opt-in since #1162** (`[implicit_feedback]
enqueue_on_retrieve`, default False). It defaulted on, writing a row
per surfaced belief inside every `retrieve()`, on the reasoning that
nothing consumes a row until the sweeper runs — which was true only
because nothing schedules the sweeper. See
`is_enqueue_on_retrieve_enabled` for the rest of that argument.

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

from aelfrice.models import LOCK_USER
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
    """Outcome of one `sweep_deferred_feedback` audit (#1162).

    **Every field is a projection, not a record of work done.** The
    sweep writes nothing: no `alpha`, no `feedback_history` row, no
    queue-status transition. `would_apply` is the count of rows that
    *would* have received `+epsilon` under the pre-#1162 sweeper, and
    `alpha_withheld` is what that would have totalled.

    Naming them `would_*` is the point. The previous shape called them
    `applied` / `cancelled`, and an audit-only sweeper reporting an
    `applied` count is exactly the ambiguity that lets "the sweep ran
    and reported 12k applied" be read as a mutation that happened.

    `would_cancel` counts rows an explicit signal landed on inside the
    grace window — under the old sweeper those drained without a
    posterior change. `would_skip_no_belief`, `would_skip_locked` and
    `would_skip_foreign` (#1168) are rows whose belief no longer
    resolves, carries a user lock, or belongs to a federated peer; all
    three are also counted in `would_cancel`, matching how the mutating
    sweeper drained them.

    Because nothing is written, the audit is repeatable: running it
    twice reports the same numbers rather than draining to zero. That
    is what makes the count usable as a standing measurement of how
    much implicit signal the store is sitting on.
    """

    would_apply: int = 0
    would_cancel: int = 0
    would_skip_no_belief: int = 0
    would_skip_locked: int = 0
    would_skip_foreign: int = 0
    pending_unmet_grace: int = 0
    alpha_withheld: float = 0.0
    epsilon_used: float = 0.0
    grace_seconds_used: int = 0
    would_apply_belief_ids: list[str] = field(default_factory=list)
    would_cancel_belief_ids: list[str] = field(default_factory=list)
    # Permanently False, asserted rather than documented. A future
    # change that reintroduces writes has to flip this and face the
    # test that pins it.
    mutated: bool = False


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
    """Env > kwarg > TOML > default **False** (#1162).

    This defaulted True on the argument that the queue is additive —
    nothing reads a row until the sweeper runs. That held only because
    the sweeper is a manual command nothing schedules, which is an
    accident of deployment rather than a design property. Meanwhile the
    call sits inside every `retrieve()` and writes a row per surfaced
    belief, so a store banks rows without bound.

    It also ran against a decision already taken: #1086 set
    `_exposure_updates_posterior()` default False — retrieval exposure
    is deliberately not posterior evidence. This queue was a second,
    unflagged, default-on route to the same posterior bump.

    Enqueuing is still a one-line opt-in for anyone measuring exposure.
    What it can no longer do is feed `alpha`: the sweeper is audit-only
    since #1162, so the rows are a record, not a pending mutation.
    """
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
    return False


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
    # #1135: one commit for the batch instead of one per row — this
    # runs inside every retrieve() with N surfaced beliefs.
    with store.transaction():
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
    """Audit the deferred-feedback queue. **Writes nothing** (#1162).

    Classifies every pending row exactly as the mutating sweeper did —
    grace elapsed, explicit signal in window, belief missing, locked,
    or foreign — and reports what it *would* have applied. No `alpha`
    moves, no `feedback_history` row is written, no queue status
    changes.

    The sweeper used to apply `+epsilon` per eligible row. Two things
    made that unsafe rather than merely unused:

      * **No counterweight.** `scoring.decay` / `type_half_life` have
        no production caller, so a frequently-retrieved belief's alpha
        grows without bound and its posterior mean walks to 1.0,
        permanently outranking equal-BM25 peers.
      * **A banked backlog.** Enqueuing was default-on inside every
        `retrieve()`, so real stores carry six figures of pending rows.
        One invocation would have fired the entire backlog at once —
        which is why leaving a mutating sweeper in place while merely
        flipping the enqueue default would not have been enough.

    Making the audit read-only rather than "mutate but drain" is
    deliberate: a sweeper that consumed the rows would report a
    non-zero count once and zero forever after, which reads as "there
    is nothing here" rather than "this was already spent". Nothing is
    consumed, so the number stays honest and the audit is repeatable.

    Turning implicit exposure into real feedback again is a separate
    proposal — it reverses #1086, changes ranking for every user, and
    needs a bench in front of it. It is not a matter of re-enabling
    this function.
    """
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

    for _row_id, belief_id, enqueued_at, _event_type in pending:
        # No transaction, and no BEGIN IMMEDIATE. The mutating sweeper
        # took the write lock before its eligibility reads to close the
        # check-then-act window #1168 found; with nothing written there
        # is no window to close, and holding a write lock across an
        # audit of a six-figure queue would block ingest for no reason.
        belief = store.get_belief(belief_id)

        # Same ineligibility ladder the mutating sweeper applied, in the
        # same order, so the projection describes that sweeper rather
        # than a simplified model of it.
        skip_counter: str | None = None
        if belief is None:
            skip_counter = "would_skip_no_belief"
        elif belief.lock_level == LOCK_USER:
            skip_counter = "would_skip_locked"
        else:
            try:
                store.assert_local_ownership(belief_id)
            except ValueError:
                skip_counter = "would_skip_foreign"

        if skip_counter is not None:
            setattr(result, skip_counter, getattr(result, skip_counter) + 1)
            result.would_cancel += 1
            result.would_cancel_belief_ids.append(belief_id)
            continue

        if store.has_explicit_feedback_in_window(
            belief_id,
            window_start_iso=enqueued_at,
            window_end_iso=now_iso,
            retrieval_source=RETRIEVAL_DRIVEN_FEEDBACK_SOURCE,
        ):
            result.would_cancel += 1
            result.would_cancel_belief_ids.append(belief_id)
        else:
            result.would_apply += 1
            result.would_apply_belief_ids.append(belief_id)

    result.alpha_withheld = result.would_apply * eps_eff
    return result
