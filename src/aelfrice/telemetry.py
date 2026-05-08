"""Per-session telemetry writer for `~/.aelfrice/telemetry.jsonl`.

Public API:

    compute_session_delta(store, session_id) -> dict
        Pure function. Reads belief / feedback / correction data for the
        given session_id, computes per-session delta fields, and combines
        with a full snapshot of the current store state to produce one
        v=1 telemetry row as a plain dict. Returns the complete row dict
        — callers can inspect, test, or pass to `emit_session_delta`.

    emit_session_delta(session_id, *, store=None, path=None) -> None
        Wraps `compute_session_delta`, builds the full v=1 JSON row, and
        appends it (newline-terminated) to `path` (default
        ``~/.aelfrice/telemetry.jsonl``). Thread-safe append — opens in
        "a" mode, which is O_APPEND on POSIX. Never raises on an empty /
        missing session_id: logs a warning to stderr and returns.

Schema contract (v=1)
---------------------
The row matches the shape that the HOME-repo
`aelfrice-session-end-telemetry.sh` hook already produces:

  {
    "v": 1,
    "ts": "<iso8601>",
    "session": {
      "retrieval_tokens": int,
      "classification_tokens": int,
      "beliefs_created": int,
      "corrections_detected": int,
      "searches_performed": int,
      "feedback_given": int,
      "velocity_items_per_hour": float,
      "velocity_tier": "deep|...",
      "duration_seconds": float
    },
    "feedback": {
      "outcome_counts": {"used": int, "ignored": int, ...},
      "detection_layer_counts": {"implicit": int, ...},
      "feedback_rate": float
    },
    "beliefs": {
      "total_active": int,
      "total_superseded": int,
      "total_locked": int,
      "confidence_distribution": {...},
      "type_distribution": {...},
      "source_distribution": {...},
      "churn_rate": float,
      "orphan_count": int
    },
    "graph": {
      "total_edges": int,
      "edge_type_distribution": {...},
      "avg_edges_per_belief": float
    },
    "window_7": {
      "sessions_in_window": int,
      "totals": {...},
      "averages": {...},
      "feedback_rate": float,
      "correction_rate": float
    },
    "window_30": { ... }
  }

Notes on per-session sourcing
------------------------------
The current schema does not tag `feedback_history` rows with a
`session_id`. Feedback events can however be cross-referenced via the
beliefs created in a session (``beliefs WHERE session_id = ?``) — so
`feedback_given` is the count of `feedback_history` rows whose
`belief_id` is among the session's beliefs. This is a conservative
proxy (feedback on pre-existing beliefs during the session is not
counted), but it is correct and honest about what it measures.

`corrections_detected` counts session beliefs whose `type = 'correction'`.
`retrieval_tokens` and `classification_tokens` are not directly
measurable from the current schema; they are emitted as 0 for
compatibility (the field exists in the schema so future token-tracking
additions can populate it without a schema change).
`velocity_items_per_hour` is `beliefs_created / (duration_seconds / 3600)`.
`duration_seconds` is `completed_at - started_at` from the sessions
table when present, else 0.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from aelfrice.store import MemoryStore

DEFAULT_TELEMETRY_PATH: Path = Path.home() / ".aelfrice" / "telemetry.jsonl"

# Velocity tiers: items/hour thresholds.
_VELOCITY_TIERS: list[tuple[float, str]] = [
    (20.0, "deep"),      # < 20 items/h → deep (slow, thoughtful)
    (60.0, "active"),    # 20-60 → active
    (120.0, "rapid"),    # 60-120 → rapid
]
_VELOCITY_TIER_MAX: str = "burst"  # > 120 items/h


def _velocity_tier(items_per_hour: float) -> str:
    for threshold, tier in _VELOCITY_TIERS:
        if items_per_hour < threshold:
            return tier
    return _VELOCITY_TIER_MAX


def _duration_seconds(store: MemoryStore, session_id: str) -> float:
    """Seconds between session start and completion, or 0 when unknown."""
    session = store.get_session(session_id)
    if session is None:
        return 0.0
    started = session.started_at
    completed = session.completed_at
    if not started or not completed:
        return 0.0
    try:
        start_dt = datetime.fromisoformat(started)
        end_dt = datetime.fromisoformat(completed)
    except ValueError:
        return 0.0
    delta = (end_dt - start_dt).total_seconds()
    return max(0.0, delta)


def _session_beliefs(store: MemoryStore, session_id: str) -> list[str]:
    """Return belief ids tagged with ``session_id``."""
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT id FROM beliefs WHERE session_id = ?",
        (session_id,),
    )
    return [str(row["id"]) for row in cur.fetchall()]


def _session_belief_types(store: MemoryStore, session_id: str) -> dict[str, int]:
    """Count of each belief type created in this session."""
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT type, COUNT(*) AS n FROM beliefs WHERE session_id = ? GROUP BY type",
        (session_id,),
    )
    return {str(row["type"]): int(row["n"]) for row in cur.fetchall()}


def _count_feedback_for_beliefs(store: MemoryStore, belief_ids: list[str]) -> int:
    """Count feedback_history rows for the given belief ids."""
    if not belief_ids:
        return 0
    placeholders = ",".join("?" * len(belief_ids))
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        f"SELECT COUNT(*) AS n FROM feedback_history WHERE belief_id IN ({placeholders})",
        belief_ids,
    )
    row = cur.fetchone()
    return int(row["n"]) if row else 0


def _snapshot_beliefs(store: MemoryStore) -> dict[str, Any]:
    """Snapshot the full belief state for the beliefs block."""
    n_total = store.count_beliefs()
    n_locked = store.count_locked()

    # Count superseded (beliefs with an incoming SUPERSEDES edge)
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        """
        SELECT COUNT(DISTINCT dst) AS n FROM edges WHERE type = 'SUPERSEDES'
        """
    )
    row = cur.fetchone()
    n_superseded = int(row["n"]) if row else 0
    n_active = max(0, n_total - n_superseded)

    # Confidence distribution (bucket into 0.0-0.2, 0.2-0.4, ..., 0.8-1.0)
    pairs = store.alpha_beta_pairs()
    buckets: dict[str, int] = {
        "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0,
    }
    for alpha, beta in pairs:
        denom = alpha + beta
        conf = (alpha / denom) if denom > 0 else 0.5
        if conf < 0.2:
            buckets["0.0-0.2"] += 1
        elif conf < 0.4:
            buckets["0.2-0.4"] += 1
        elif conf < 0.6:
            buckets["0.4-0.6"] += 1
        elif conf < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1

    # Type distribution
    cur2 = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT type, COUNT(*) AS n FROM beliefs GROUP BY type"
    )
    type_dist: dict[str, int] = {str(r["type"]): int(r["n"]) for r in cur2.fetchall()}

    # Source/origin distribution
    cur3 = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT origin, COUNT(*) AS n FROM beliefs GROUP BY origin"
    )
    source_dist: dict[str, int] = {str(r["origin"]): int(r["n"]) for r in cur3.fetchall()}

    # Churn rate: superseded / total (0 if no beliefs)
    churn_rate = (n_superseded / n_total) if n_total > 0 else 0.0

    # Orphan count: beliefs with no edges (no src, no dst)
    cur4 = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        """
        SELECT COUNT(*) AS n FROM beliefs b
        WHERE NOT EXISTS (SELECT 1 FROM edges WHERE src = b.id OR dst = b.id)
        """
    )
    row4 = cur4.fetchone()
    orphan_count = int(row4["n"]) if row4 else 0

    return {
        "total_active": n_active,
        "total_superseded": n_superseded,
        "total_locked": n_locked,
        "confidence_distribution": buckets,
        "type_distribution": type_dist,
        "source_distribution": source_dist,
        "churn_rate": round(churn_rate, 4),
        "orphan_count": orphan_count,
    }


def _snapshot_graph(store: MemoryStore) -> dict[str, Any]:
    """Snapshot the full graph state for the graph block."""
    n_edges = store.count_edges()
    n_beliefs = store.count_beliefs()
    edge_type_dist = store.count_edges_by_type()
    avg_edges_per_belief = (n_edges / n_beliefs) if n_beliefs > 0 else 0.0
    return {
        "total_edges": n_edges,
        "edge_type_distribution": edge_type_dist,
        "avg_edges_per_belief": round(avg_edges_per_belief, 4),
    }


def _window_rollup(
    telemetry_path: Path,
    window_days: int,
    now: datetime,
) -> dict[str, Any]:
    """Compute a rolling-window rollup from the existing telemetry.jsonl.

    Walks the file tail and counts rows whose ``ts`` falls within
    ``window_days`` of ``now``. Returns the count plus aggregate totals
    and averages for the session block fields.

    When the file does not exist or has no valid rows in the window,
    returns a zero-row block. Never raises.
    """
    zero: dict[str, Any] = {
        "sessions_in_window": 0,
        "totals": {
            "beliefs_created": 0,
            "corrections_detected": 0,
            "feedback_given": 0,
        },
        "averages": {
            "beliefs_created": 0.0,
            "corrections_detected": 0.0,
            "feedback_given": 0.0,
            "velocity_items_per_hour": 0.0,
        },
        "feedback_rate": 0.0,
        "correction_rate": 0.0,
    }
    if not telemetry_path.is_file():
        return zero

    cutoff = now.timestamp() - window_days * 86400.0
    in_window: list[dict[str, Any]] = []

    try:
        with telemetry_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj: Any = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                obj_d: dict[str, Any] = cast(dict[str, Any], obj)
                ts_str = obj_d.get("ts", "")
                if not isinstance(ts_str, str) or not ts_str:
                    continue
                try:
                    ts_dt = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if ts_dt.timestamp() < cutoff:
                    continue
                in_window.append(obj_d)
    except OSError:
        return zero

    n = len(in_window)
    if n == 0:
        return zero

    total_beliefs = 0
    total_corrections = 0
    total_feedback = 0
    total_velocity = 0.0

    for row in in_window:
        raw_sess = row.get("session")
        sess: dict[str, Any] = cast(dict[str, Any], raw_sess) if isinstance(raw_sess, dict) else {}
        bc: Any = sess.get("beliefs_created", 0)
        cd: Any = sess.get("corrections_detected", 0)
        fg: Any = sess.get("feedback_given", 0)
        vph: Any = sess.get("velocity_items_per_hour", 0.0)
        total_beliefs += int(bc) if isinstance(bc, (int, float)) else 0
        total_corrections += int(cd) if isinstance(cd, (int, float)) else 0
        total_feedback += int(fg) if isinstance(fg, (int, float)) else 0
        total_velocity += float(vph) if isinstance(vph, (int, float)) else 0.0

    feedback_rate = (total_feedback / total_beliefs) if total_beliefs > 0 else 0.0
    correction_rate = (total_corrections / total_beliefs) if total_beliefs > 0 else 0.0

    return {
        "sessions_in_window": n,
        "totals": {
            "beliefs_created": total_beliefs,
            "corrections_detected": total_corrections,
            "feedback_given": total_feedback,
        },
        "averages": {
            "beliefs_created": round(total_beliefs / n, 4),
            "corrections_detected": round(total_corrections / n, 4),
            "feedback_given": round(total_feedback / n, 4),
            "velocity_items_per_hour": round(total_velocity / n, 4),
        },
        "feedback_rate": round(feedback_rate, 4),
        "correction_rate": round(correction_rate, 4),
    }


def compute_session_delta(
    store: MemoryStore,
    session_id: str,
    *,
    telemetry_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute one v=1 telemetry row for ``session_id``.

    Pure function (no file I/O except the optional telemetry.jsonl read
    for window rollups). The ``store`` must already be open; the caller
    is responsible for opening and closing it.

    ``now`` pins the timestamp for deterministic tests (default: UTC now).
    ``telemetry_path`` is used only for the window rollup reads; it
    defaults to ``DEFAULT_TELEMETRY_PATH``.

    If ``session_id`` is unknown (no beliefs tagged with it), all
    per-session delta counts are 0 and a zero-row is still returned so
    idle sessions are recorded (the issue spec says idle sessions MUST
    emit a row so ``len(telemetry.jsonl)`` equals session count).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if telemetry_path is None:
        telemetry_path = DEFAULT_TELEMETRY_PATH

    ts = now.isoformat()

    # --- Per-session deltas ------------------------------------------------
    belief_ids = _session_beliefs(store, session_id)
    beliefs_created = len(belief_ids)
    type_counts = _session_belief_types(store, session_id)
    corrections_detected = type_counts.get("correction", 0)
    feedback_given = _count_feedback_for_beliefs(store, belief_ids)

    duration_secs = _duration_seconds(store, session_id)
    velocity = (beliefs_created / (duration_secs / 3600.0)) if duration_secs > 0.0 else 0.0
    vtier = _velocity_tier(velocity)

    session_block: dict[str, Any] = {
        "retrieval_tokens": 0,       # not tracked in current schema
        "classification_tokens": 0,  # not tracked in current schema
        "beliefs_created": beliefs_created,
        "corrections_detected": corrections_detected,
        "searches_performed": 0,     # not tracked in current schema
        "feedback_given": feedback_given,
        "velocity_items_per_hour": round(velocity, 4),
        "velocity_tier": vtier,
        "duration_seconds": round(duration_secs, 3),
    }

    # feedback block (per-session)
    feedback_rate = (feedback_given / beliefs_created) if beliefs_created > 0 else 0.0
    feedback_block: dict[str, Any] = {
        "outcome_counts": {
            "used": feedback_given,   # positive valence = used
            "ignored": 0,             # not directly trackable per-session
        },
        "detection_layer_counts": {
            "implicit": 0,            # not tracked in current schema
            "explicit": feedback_given,
        },
        "feedback_rate": round(feedback_rate, 4),
    }

    # --- Snapshot blocks (read at write time) ------------------------------
    beliefs_block = _snapshot_beliefs(store)
    graph_block = _snapshot_graph(store)

    # --- Window rollups (from existing telemetry.jsonl) -------------------
    window_7 = _window_rollup(telemetry_path, 7, now)
    window_30 = _window_rollup(telemetry_path, 30, now)

    return {
        "v": 1,
        "ts": ts,
        "session": session_block,
        "feedback": feedback_block,
        "beliefs": beliefs_block,
        "graph": graph_block,
        "window_7": window_7,
        "window_30": window_30,
    }


def emit_session_delta(
    session_id: str,
    *,
    store: MemoryStore | None = None,
    path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Compute and append one v=1 row to the telemetry JSONL.

    Behavior when ``session_id`` is empty/missing: logs a warning to
    stderr and returns (silent no-op — never raises, always exits 0).
    This matches the hook contract where a missing id must not break the
    calling shell session.

    ``store``: open MemoryStore to read from. When None, the caller's
    environment variable / git-common-dir resolution is used (imports
    cli.db_path() lazily to avoid a circular import). Callers that
    already hold an open store should pass it explicitly.

    ``path``: telemetry JSONL path. Defaults to
    ``DEFAULT_TELEMETRY_PATH``. Parent directory is created on first
    write.

    ``now``: UTC datetime for the row timestamp + window cutoffs. Pins
    to a fixed value in tests for determinism.
    """
    if not session_id:
        print(
            "aelf session-delta: session_id is empty — skipping",
            file=sys.stderr,
        )
        return

    if path is None:
        path = DEFAULT_TELEMETRY_PATH

    if now is None:
        now = datetime.now(timezone.utc)

    own_store = store is None
    if own_store:
        from aelfrice.db_paths import _open_store
        store = _open_store()

    try:
        row = compute_session_delta(
            store, session_id, telemetry_path=path, now=now
        )
    finally:
        if own_store:
            store.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
