"""Temporal spine — per-session chronological TEMPORAL_NEXT chains (#1064).

The spine writer links each newly inserted belief to the previous belief
in the same ``session_id`` (ordered by ``created_at``, insertion order as
the tie-break) with a ``TEMPORAL_NEXT`` edge: src = the temporal
successor, dst = the predecessor (matching the ``models.py`` edge
semantics), weight ``TEMPORAL_SPINE_EDGE_WEIGHT``. One edge per belief,
O(1) per insert — the chain grows at the tail as the session grows.

The mechanism is complementary to lexical matching: gold beliefs that
share zero salient terms with a question are unreachable by any lexical
means, but become reachable through chronological adjacency to beliefs
that *do* match. A shuffled control (identical edge count, permuted
endpoints) isolates the cause as the chronology, not the connectivity.

Default-OFF (``is_temporal_spine_write_enabled``): a fresh install writes
no spine edges and ingest is byte-identical to today. The flag is the
landing posture, not the end state — the default-ON flip is gated on the
pre-registered criteria recorded in issue #1064 (G1–G5). Deterministic,
stdlib-only: no LLM, no embedding, no sampling (#605).

Soft-deleted beliefs (``valid_to`` set) are not excluded from predecessor
selection: chain integrity must survive GC, so a successor links to the
most recent session predecessor regardless of its lifecycle state
(skip-but-continue happens at read time, in the traversal lane).
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, cast

from aelfrice.models import EDGE_TEMPORAL_NEXT, Edge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aelfrice.store import MemoryStore

ENV_TEMPORAL_SPINE_WRITE: Final[str] = "AELFRICE_TEMPORAL_SPINE_WRITE"
CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
SECTION: Final[str] = "ingest"
WRITE_KEY: Final[str] = "write_temporal_spine"

# Chronological adjacency is a strong structural signal (stronger than
# the generic RELATES_TO tier) but carries no evidential content, so it
# sits below the evidential edges' 1.0. Distinct from the propagation
# valence in models.EDGE_VALENCE (0.2) — that family tunes feedback
# propagation, this is the graph-traversal edge weight.
TEMPORAL_SPINE_EDGE_WEIGHT: Final[float] = 0.8

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


# --- Flag resolver (mirrors the #988 auto-detect resolver) --------------


def _env_spine_write_override() -> bool | None:
    """Return True/False if ``AELFRICE_TEMPORAL_SPINE_WRITE`` is set to a
    recognised truthy / falsy value, else None (env not decisive)."""
    raw = os.environ.get(ENV_TEMPORAL_SPINE_WRITE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_spine_write_toml(start: Path | None = None) -> bool | None:
    """Read ``[ingest] write_temporal_spine`` from the nearest
    `.aelfrice.toml`. Returns None on missing file / section / key /
    malformed TOML / non-bool value; never raises."""
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    candidate.read_bytes().decode("utf-8", errors="replace"),
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice temporal_spine: cannot read "
                    f"{WRITE_KEY} in {candidate}: {exc}",
                    file=serr,
                )
                return None
            section_obj: Any = parsed.get(SECTION, {})
            if not isinstance(section_obj, dict):
                return None
            val: Any = cast("dict[str, Any]", section_obj).get(WRITE_KEY)
            if isinstance(val, bool):
                return val
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def is_temporal_spine_write_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the ingest-time temporal-spine writer flag (#1064).

    Precedence (first decisive wins):
      1. ``AELFRICE_TEMPORAL_SPINE_WRITE`` env var (truthy / falsy
         normalised).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[ingest] write_temporal_spine`` in `.aelfrice.toml`.
      4. Default: False (default-OFF).

    Default-off is the landing posture: a fresh install must not start
    writing spine edges at ingest until the #1064 flip-gate criteria
    (G2–G5) pass. Flipping the default is that issue's deliverable 5,
    not a config change.
    """
    env = _env_spine_write_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_spine_write_toml(start)
    if toml_value is not None:
        return toml_value
    return False


# --- Spine writer --------------------------------------------------------


@dataclass(frozen=True)
class SpineWriteReport:
    """Summary of one ``write_temporal_spine`` run.

    ``n_beliefs_seen``
        Distinct belief ids processed (input order, duplicates dropped).

    ``n_edges_written``
        Beliefs that produced a new TEMPORAL_NEXT edge this run.

    ``n_skipped_no_session``
        Beliefs skipped because they carry a NULL ``session_id`` (or the
        id resolved to no belief row at all) — no chain to join.

    ``n_skipped_no_predecessor``
        Beliefs that are the first of their session — nothing earlier to
        link to. The next insert in the session links back to them.

    ``n_skipped_existing``
        Beliefs whose spine edge already existed (idempotency guard).
    """

    n_beliefs_seen: int
    n_edges_written: int
    n_skipped_no_session: int
    n_skipped_no_predecessor: int
    n_skipped_existing: int


def write_temporal_spine(
    store: "MemoryStore",
    *,
    new_belief_ids: "Sequence[str]",
) -> SpineWriteReport:
    """Chain newly inserted beliefs into their session's temporal spine.

    For each belief in ``new_belief_ids`` (this turn's inserts), find the
    belief immediately before it in the same session — ordered by
    ``(created_at, rowid)``, i.e. creation time with insertion order as
    the tie-break — and insert one ``TEMPORAL_NEXT`` edge with
    src = successor, dst = predecessor, weight
    ``TEMPORAL_SPINE_EDGE_WEIGHT``.

    Per-belief work is independent (each new belief looks up its own
    predecessor, which is already in the store by the time this runs),
    so processing order does not affect the resulting edge set and the
    output is deterministic for a given store state.

    **Idempotent.** Each ``(src, dst, TEMPORAL_NEXT)`` triple is checked
    before insert and skipped if present; re-running over the same ids
    writes nothing new.

    Stdlib-only: no LLM, no embedding. O(1) per belief (one indexed
    predecessor lookup + one edge insert).
    """
    n_seen = 0
    n_written = 0
    n_no_session = 0
    n_no_predecessor = 0
    n_existing = 0

    processed: set[str] = set()
    for belief_id in new_belief_ids:
        if belief_id in processed:
            continue
        processed.add(belief_id)
        n_seen += 1

        belief = store.get_belief(belief_id)
        if belief is None or belief.session_id is None:
            n_no_session += 1
            continue

        predecessor_id = store.session_predecessor_id(belief_id)
        if predecessor_id is None:
            n_no_predecessor += 1
            continue

        if store.get_edge(belief_id, predecessor_id, EDGE_TEMPORAL_NEXT) is not None:
            n_existing += 1
            continue

        store.insert_edge(Edge(
            src=belief_id,
            dst=predecessor_id,
            type=EDGE_TEMPORAL_NEXT,
            weight=TEMPORAL_SPINE_EDGE_WEIGHT,
        ))
        n_written += 1

    return SpineWriteReport(
        n_beliefs_seen=n_seen,
        n_edges_written=n_written,
        n_skipped_no_session=n_no_session,
        n_skipped_no_predecessor=n_no_predecessor,
        n_skipped_existing=n_existing,
    )


# --- Backfill (existing stores predate the writer) ------------------------


@dataclass(frozen=True)
class SpineBackfillReport:
    """Summary of one ``backfill_temporal_spine`` run.

    ``n_sessions``
        Distinct non-NULL sessions visited.

    ``n_beliefs_in_sessions``
        Beliefs carrying a session_id (chain members, including chain
        heads that get no outgoing spine edge).

    ``n_edges_written``
        Consecutive pairs that produced a new TEMPORAL_NEXT edge — or,
        under ``dry_run``, would have.

    ``n_edges_existing``
        Consecutive pairs whose spine edge already existed (idempotency
        guard fired; re-running the backfill writes nothing new).
    """

    n_sessions: int
    n_beliefs_in_sessions: int
    n_edges_written: int
    n_edges_existing: int


def backfill_temporal_spine(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
) -> SpineBackfillReport:
    """Build per-session TEMPORAL_NEXT chains over an existing store.

    Existing stores predate the ingest-time writer; the migration story
    cannot be "re-ingest everything". This walks every session's beliefs
    in ``(created_at, rowid)`` order and links each consecutive pair
    with the same edge the writer would have produced (src = successor,
    dst = predecessor, weight ``TEMPORAL_SPINE_EDGE_WEIGHT``).

    **Idempotent** per ``(src, dst, TEMPORAL_NEXT)`` triple: pairs whose
    edge exists are counted and skipped, so re-running after a partial
    build (or on a store the writer is already chaining) is safe.

    ``dry_run`` counts what a real run would write without touching the
    store.
    """
    from aelfrice.models import EDGE_TEMPORAL_NEXT as _EDGE  # noqa: PLC0415

    sessions: set[str] = set()
    n_beliefs = 0
    n_written = 0
    n_existing = 0

    prev_session: str | None = None
    prev_belief: str | None = None
    for session_id, belief_id in store.session_belief_ids_ordered():
        sessions.add(session_id)
        n_beliefs += 1
        if session_id == prev_session and prev_belief is not None:
            if store.get_edge(belief_id, prev_belief, _EDGE) is not None:
                n_existing += 1
            else:
                if not dry_run:
                    store.insert_edge(Edge(
                        src=belief_id,
                        dst=prev_belief,
                        type=_EDGE,
                        weight=TEMPORAL_SPINE_EDGE_WEIGHT,
                    ))
                n_written += 1
        prev_session = session_id
        prev_belief = belief_id

    return SpineBackfillReport(
        n_sessions=len(sessions),
        n_beliefs_in_sessions=n_beliefs,
        n_edges_written=n_written,
        n_edges_existing=n_existing,
    )
