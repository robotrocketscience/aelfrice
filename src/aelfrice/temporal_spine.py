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

Default-ON (``is_temporal_spine_write_enabled``) since the #1064 flip:
every evidence gate (G1–G5) passed, so a fresh install chains new beliefs
by default. Opt out with ``AELFRICE_TEMPORAL_SPINE_WRITE=0`` or ``[ingest]
write_temporal_spine = false``, in which case ingest is byte-identical to
the pre-spine behaviour. Deterministic, stdlib-only: no LLM, no embedding,
no sampling (#605).

Soft-deleted beliefs (``valid_to`` set) are not excluded from predecessor
selection: chain integrity must survive GC, so a successor links to the
most recent session predecessor regardless of its lifecycle state
(skip-but-continue happens at read time, in the traversal lane).
"""
from __future__ import annotations

import os
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, cast

from aelfrice.models import EDGE_TEMPORAL_NEXT, Edge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aelfrice.models import Belief
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
      4. Default: True (default-ON since the #1064 flip).

    Default-ON since the #1064 flip: every evidence gate (G1–G5) passed,
    so the ingest writer chains new beliefs by default. Opt out with
    ``AELFRICE_TEMPORAL_SPINE_WRITE=0`` or ``[ingest] write_temporal_spine
    = false``. (Landed default-off; the flip is that issue's deliverable
    5, shipped as a release default change plus the auto-once backfill for
    existing stores.)
    """
    env = _env_spine_write_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_spine_write_toml(start)
    if toml_value is not None:
        return toml_value
    return True


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


def clear_temporal_spine(store: "MemoryStore") -> int:
    """Delete every ``TEMPORAL_NEXT`` edge; return the count removed.

    The reversibility counterpart to ``backfill_temporal_spine`` — backs
    ``aelf spine clear`` (#1064 G4). Beliefs are untouched; only the
    per-session chain edges are dropped, so a later backfill rebuilds
    them byte-identically (the G5 determinism property).
    """
    return store.delete_edges_by_type(EDGE_TEMPORAL_NEXT)


# --- Auto-backfill migration (G4) -----------------------------------------

# Sentinel marking that the one-shot auto-backfill has run on this host.
# Persists across versions (lives beside the uv-migration + auto_install
# stamps under ~/.aelfrice/); once written, `aelf setup` short-circuits
# the check. Reversal is `aelf spine clear`, not deleting this file.
SPINE_BACKFILLED_SENTINEL: Final[Path] = (
    Path.home() / ".aelfrice" / "spine-backfilled"
)


@dataclass(frozen=True)
class SpineAutoBackfillResult:
    """Outcome of a ``maybe_backfill_temporal_spine()`` call.

    ``ran`` is True iff the backfill actually executed (the sentinel was
    absent and the writer was enabled). ``n_edges`` is what it wrote (0 on
    a fresh/empty or already-chained store). ``reason`` is a one-line
    description suitable for a stderr notice, populated in the skipped
    paths too.
    """

    ran: bool
    n_edges: int
    reason: str


def maybe_backfill_temporal_spine(
    store: "MemoryStore",
    *,
    sentinel_path: Path = SPINE_BACKFILLED_SENTINEL,
    write_enabled: bool | None = None,
) -> SpineAutoBackfillResult:
    """One-shot auto-backfill of the spine on the flip release (#1064 G4).

    The default-ON flip starts the ingest writer chaining *new* beliefs,
    but existing stores have no spine over their history. This builds it
    once so the lane is effective on day one rather than only after enough
    new turns accrue.

    Sentinel-gated + idempotent, so it fires exactly once per host and is
    safe to call from every ``aelf setup``. Short-circuit order (cheapest
    first):

      1. sentinel exists -> no-op (already ran).
      2. the spine writer is disabled (a host that opted out via
         env/toml) -> no-op, and the sentinel is NOT written, so the
         check re-arms and fires on the first setup after the writer is
         re-enabled.
      3. otherwise run ``backfill_temporal_spine`` (idempotent), write the
         sentinel, and report the edge count.

    Reversible via ``aelf spine clear``. Never raises — any failure
    returns a result describing it and leaves the store untouched.
    """
    if sentinel_path.exists():
        return SpineAutoBackfillResult(
            False, 0, "already backfilled (sentinel exists)"
        )
    enabled = (
        write_enabled if write_enabled is not None
        else is_temporal_spine_write_enabled()
    )
    if not enabled:
        return SpineAutoBackfillResult(
            False, 0, "spine writer disabled — backfill deferred until the flip"
        )
    try:
        report = backfill_temporal_spine(store)
    except Exception as exc:  # noqa: BLE001 — never break `aelf setup`
        return SpineAutoBackfillResult(False, 0, f"backfill failed: {exc}")
    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"spine auto-backfilled {report.n_edges_written} edges "
            f"at {time.time():.0f}\n"
        )
    except OSError as exc:
        # Backfill is done + idempotent; a missing sentinel only costs a
        # harmless re-run next setup (which writes 0 new edges).
        return SpineAutoBackfillResult(
            True, report.n_edges_written,
            f"backfilled {report.n_edges_written} edges "
            f"(sentinel write failed: {exc})",
        )
    return SpineAutoBackfillResult(
        True, report.n_edges_written,
        f"backfilled {report.n_edges_written} TEMPORAL_NEXT edge(s) over "
        f"{report.n_sessions} session(s)",
    )


# --- Retrieval-lane traversal ---------------------------------------------

DEFAULT_SPINE_SEED_COUNT: Final[int] = 5
DEFAULT_SPINE_DEPTH: Final[int] = 1
DEFAULT_SPINE_NODE_BUDGET: Final[int] = 32


def spine_neighbors(
    store: "MemoryStore",
    seed_ids: "Sequence[str]",
    *,
    depth: int = DEFAULT_SPINE_DEPTH,
    node_budget: int = DEFAULT_SPINE_NODE_BUDGET,
) -> list["Belief"]:
    """Chronological neighbours of ``seed_ids`` over TEMPORAL_NEXT edges.

    Bidirectional: for each frontier belief, both its temporal
    successors (edges whose ``dst`` is the belief) and its predecessor
    (edges whose ``src`` is the belief) are visited. Traversal is
    breadth-first to ``depth`` hops, emitting at most ``node_budget``
    beliefs. The #1064 confirmatory evidence ran depth-1; the monotone
    budget curve (~+2.5pp per doubling at 32/64/128) says the budget
    knob is the one to revisit at flip time.

    Deterministic: seeds are processed in input order; within one
    frontier belief, successors come before the predecessor and each
    group is sorted by belief id. No sampling, no scores.

    Soft-deleted beliefs (``valid_to`` set) are skip-but-continue
    (#1064 open question 2): they are traversed *through* — kept on the
    frontier so a GC'd chain segment doesn't sever the spine — but are
    never emitted and never consume ``node_budget``.

    Seeds themselves are never emitted. Returns beliefs in discovery
    order (the caller packs them within its token budget).
    """
    if depth <= 0 or node_budget <= 0 or not seed_ids:
        return []

    visited: set[str] = set(seed_ids)
    emitted: list["Belief"] = []
    frontier: list[str] = list(dict.fromkeys(seed_ids))

    for _hop in range(depth):
        if not frontier:
            break
        edges = [
            e for e in store.edges_for_beliefs(list(frontier))
            if e.type == EDGE_TEMPORAL_NEXT
        ]
        successors: dict[str, list[str]] = {}
        predecessors: dict[str, list[str]] = {}
        for e in edges:
            # src = temporal successor of dst (models.py semantics).
            successors.setdefault(e.dst, []).append(e.src)
            predecessors.setdefault(e.src, []).append(e.dst)

        next_frontier: list[str] = []
        for node in frontier:
            neighbours = (
                sorted(successors.get(node, []))
                + sorted(predecessors.get(node, []))
            )
            for nid in neighbours:
                if nid in visited:
                    continue
                visited.add(nid)
                belief = store.get_belief(nid)
                if belief is None:
                    continue
                if belief.valid_to is not None:
                    # skip-but-continue: traverse through GC'd segments.
                    next_frontier.append(nid)
                    continue
                emitted.append(belief)
                next_frontier.append(nid)
                if len(emitted) >= node_budget:
                    return emitted
        frontier = next_frontier

    return emitted
