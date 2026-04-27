"""Eight-command CLI matching the MVP user surface.

Commands:
  onboard <path>                   scan a project and ingest beliefs
  search <query> [--budget N]      L0 locked + L1 FTS5 retrieval
  lock <statement>                 insert (or upgrade) a user-locked belief
  locked [--pressured]             list locked beliefs
  demote <id>                      manually demote a lock to none
  feedback <id> <used|harmful>     apply one Bayesian feedback event
  stats                            summary of belief / lock / history counts
  health                           regime classifier output

DB path resolves from AELFRICE_DB environment variable when set,
otherwise from ~/.aelfrice/memory.db. Callers can run `main(argv=...)`
in-process for tests; the `aelf` entry point in pyproject.toml maps to
`main()`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

from aelfrice.feedback import apply_feedback
from aelfrice.health import (
    REGIME_INSUFFICIENT_DATA,
    assess_health,
    regime_description,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.store import Store

DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"
_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}
_LOCK_ID_LEN: Final[int] = 16


def db_path() -> Path:
    """Resolve the DB path from $AELFRICE_DB or the default."""
    override = os.environ.get("AELFRICE_DB")
    if override:
        return Path(override)
    return DEFAULT_DB_DIR / DEFAULT_DB_FILENAME


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_store() -> Store:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return Store(str(p))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lock_id_for(content: str) -> str:
    return hashlib.sha256(f"lock\x00{content}".encode("utf-8")).hexdigest()[:_LOCK_ID_LEN]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# --- Command handlers ----------------------------------------------------


def _cmd_onboard(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        result = scan_repo(store, Path(args.path))
    finally:
        store.close()
    print(
        f"onboarded {args.path}: "
        f"{result.inserted} added, "
        f"{result.skipped_existing} skipped (already present), "
        f"{result.skipped_non_persisting} skipped (non-persisting), "
        f"{result.total_candidates} candidates seen",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_search(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        hits = retrieve(store, args.query, token_budget=args.budget)
    finally:
        store.close()
    if not hits:
        print("no results", file=out)  # type: ignore[arg-type]
        return 0
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        print(f"{prefix} {h.id}: {h.content}", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_lock(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        bid = _lock_id_for(args.statement)
        existing = store.get_belief(bid)
        now = _utc_now_iso()
        if existing is None:
            store.insert_belief(Belief(
                id=bid,
                content=args.statement,
                content_hash=_content_hash(args.statement),
                alpha=9.0,
                beta=0.5,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_USER,
                locked_at=now,
                demotion_pressure=0,
                created_at=now,
                last_retrieved_at=None,
            ))
            print(f"locked: {bid}", file=out)  # type: ignore[arg-type]
        else:
            existing.lock_level = LOCK_USER
            existing.locked_at = now
            existing.demotion_pressure = 0
            store.update_belief(existing)
            print(f"upgraded existing belief to lock: {bid}", file=out)  # type: ignore[arg-type]
    finally:
        store.close()
    return 0


def _cmd_locked(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        locked = store.list_locked_beliefs()
    finally:
        store.close()
    if args.pressured:
        locked = [b for b in locked if b.demotion_pressure > 0]
    if not locked:
        msg = "no pressured locks" if args.pressured else "no locked beliefs"
        print(msg, file=out)  # type: ignore[arg-type]
        return 0
    for b in locked:
        pressure_marker = (
            f" (pressure={b.demotion_pressure})" if b.demotion_pressure > 0 else ""
        )
        print(f"{b.id}{pressure_marker}: {b.content}", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_demote(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        belief = store.get_belief(args.belief_id)
        if belief is None:
            print(f"belief not found: {args.belief_id}", file=sys.stderr)
            return 1
        if belief.lock_level == LOCK_NONE:
            print(f"belief is not locked: {args.belief_id}", file=out)  # type: ignore[arg-type]
            return 0
        belief.lock_level = LOCK_NONE
        belief.locked_at = None
        belief.demotion_pressure = 0
        store.update_belief(belief)
        print(f"demoted: {args.belief_id}", file=out)  # type: ignore[arg-type]
    finally:
        store.close()
    return 0


def _cmd_feedback(args: argparse.Namespace, out: object) -> int:
    valence = _FEEDBACK_VALENCES.get(args.signal)
    if valence is None:
        print(f"signal must be 'used' or 'harmful', got: {args.signal}",
              file=sys.stderr)
        return 1
    store = _open_store()
    try:
        result = apply_feedback(
            store=store,
            belief_id=args.belief_id,
            valence=valence,
            source=args.source,
        )
    except ValueError as exc:
        print(f"feedback error: {exc}", file=sys.stderr)
        store.close()
        return 1
    finally:
        # store.close is fine to call twice; redundant in error path.
        try:
            store.close()
        except Exception:
            pass
    print(
        f"applied {args.signal} to {args.belief_id}: "
        f"alpha {result.prior_alpha:.3f}->{result.new_alpha:.3f}, "
        f"beta {result.prior_beta:.3f}->{result.new_beta:.3f}, "
        f"pressured={result.pressured_locks}, demoted={result.demoted_locks}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_stats(args: argparse.Namespace, out: object) -> int:
    _ = args
    store = _open_store()
    try:
        n_beliefs = store.count_beliefs()
        n_edges = store.count_edges()
        n_locked = store.count_locked()
        n_history = store.count_feedback_events()
    finally:
        store.close()
    print(f"beliefs:           {n_beliefs}", file=out)  # type: ignore[arg-type]
    print(f"edges:             {n_edges}", file=out)  # type: ignore[arg-type]
    print(f"locked:            {n_locked}", file=out)  # type: ignore[arg-type]
    print(f"feedback events:   {n_history}", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_health(args: argparse.Namespace, out: object) -> int:
    _ = args
    store = _open_store()
    try:
        report = assess_health(store)
    finally:
        store.close()
    print(f"brain mode:                  {report.regime}", file=out)  # type: ignore[arg-type]
    if report.regime != REGIME_INSUFFICIENT_DATA:
        print(
            f"classification confidence:   {report.classification_confidence:.2f}",
            file=out,  # type: ignore[arg-type]
        )
    print("", file=out)  # type: ignore[arg-type]
    print(regime_description(report.regime), file=out)  # type: ignore[arg-type]
    if report.regime != REGIME_INSUFFICIENT_DATA:
        f = report.features
        print("", file=out)  # type: ignore[arg-type]
        print(f"beliefs:                {f.n_beliefs}", file=out)  # type: ignore[arg-type]
        print(f"mean confidence:        {f.confidence_mean:.3f}", file=out)  # type: ignore[arg-type]
        print(f"mean evidence (a+b):    {f.mass_mean:.2f}", file=out)  # type: ignore[arg-type]
        print(f"locks per 1000 beliefs: {f.lock_per_1000:.2f}", file=out)  # type: ignore[arg-type]
        print(f"edges per belief:       {f.edge_per_belief:.2f}", file=out)  # type: ignore[arg-type]
    return 0


# --- Dispatcher ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aelf",
        description="Bayesian memory designed for feedback-driven learning.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_onboard = sub.add_parser("onboard", help="scan a project and ingest beliefs")
    p_onboard.add_argument("path", help="path to a project directory")
    p_onboard.set_defaults(func=_cmd_onboard)

    p_search = sub.add_parser("search", help="L0 locked + L1 FTS5 retrieval")
    p_search.add_argument("query", help="keyword query")
    p_search.add_argument(
        "--budget", type=int, default=DEFAULT_TOKEN_BUDGET,
        help="output token budget (default 2000)",
    )
    p_search.set_defaults(func=_cmd_search)

    p_lock = sub.add_parser("lock", help="insert (or upgrade) a user-locked belief")
    p_lock.add_argument("statement", help="belief text to lock as ground truth")
    p_lock.set_defaults(func=_cmd_lock)

    p_locked = sub.add_parser("locked", help="list locked beliefs")
    p_locked.add_argument(
        "--pressured", action="store_true",
        help="only show locks with nonzero demotion_pressure",
    )
    p_locked.set_defaults(func=_cmd_locked)

    p_demote = sub.add_parser("demote", help="manually demote a lock to none")
    p_demote.add_argument("belief_id", help="id of the belief to demote")
    p_demote.set_defaults(func=_cmd_demote)

    p_feedback = sub.add_parser("feedback", help="apply one feedback event")
    p_feedback.add_argument("belief_id", help="id of the belief")
    p_feedback.add_argument("signal", choices=["used", "harmful"],
                             help="feedback signal sign")
    p_feedback.add_argument("--source", default="user",
                             help="audit source label (default 'user')")
    p_feedback.set_defaults(func=_cmd_feedback)

    p_stats = sub.add_parser("stats", help="summary of belief / lock / history counts")
    p_stats.set_defaults(func=_cmd_stats)

    p_health = sub.add_parser("health", help="regime classifier output")
    p_health.set_defaults(func=_cmd_health)

    return parser


def main(argv: Sequence[str] | None = None, out: object = None) -> int:
    """CLI entry point. Returns process exit code.

    `argv` lets tests pass synthetic args; defaults to sys.argv[1:].
    `out` lets tests capture stdout; defaults to sys.stdout.
    """
    if out is None:
        out = sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args, out))
