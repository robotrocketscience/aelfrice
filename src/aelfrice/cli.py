"""Eleven-command CLI matching the MVP user surface.

Commands:
  onboard <path>                   scan a project and ingest beliefs
  search <query> [--budget N]      L0 locked + L1 FTS5 retrieval
  lock <statement>                 insert (or upgrade) a user-locked belief
  locked [--pressured]             list locked beliefs
  demote <id>                      manually demote a lock to none
  feedback <id> <used|harmful>     apply one Bayesian feedback event
  stats                            summary of belief / lock / history counts
  health                           regime classifier output
  setup                            install UserPromptSubmit hook in Claude Code
  unsetup                          remove UserPromptSubmit hook from Claude Code
  bench                            run the v0.9.0-rc benchmark harness

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
from aelfrice import __version__ as _AELFRICE_VERSION
from aelfrice.benchmark import run_benchmark, seed_corpus
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.setup import (
    SettingsScope,
    default_settings_path,
    install_user_prompt_submit_hook,
    uninstall_user_prompt_submit_hook,
)
from aelfrice.store import Store

DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"
DEFAULT_HOOK_COMMAND: Final[str] = "aelf-hook"
_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}
_LOCK_ID_LEN: Final[int] = 16
_VALID_SCOPES: Final[tuple[SettingsScope, ...]] = ("user", "project")


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


# Academic-suite targets that are scaffolded but not yet runnable at v1.0.0.
# Each entry maps target name -> phase that activates it.
# See benchmarks/README.md for status detail.
_BENCH_INERT_TARGETS: Final[dict[str, str]] = {
    "mab": "v1.2.0 (P2 — triple extraction port)",
    "locomo": "v1.2.0 (P2 — ingest pipeline port)",
    "longmemeval": "v1.2.0 (P2 — ingest pipeline port)",
    "structmemeval": "v1.2.0 (P2 — ingest pipeline port)",
    "amabench": "v1.2.0 (P2 — ingest pipeline port)",
    "all": "v2.0.0 (full reproducibility milestone)",
}


def _cmd_bench(args: argparse.Namespace, out: object) -> int:
    """Run a benchmark target.

    Default (no target / 'synthetic'): the v0.9.0-rc synthetic harness —
    seed an in-memory store with a 16-belief corpus and score retrieval.
    Fully reproducible across runs (latency varies).

    Academic-suite targets (mab, locomo, longmemeval, structmemeval,
    amabench, all) are scaffolded in benchmarks/ but inert at v1.0.0;
    they exit 2 with a pointer to benchmarks/README.md until their
    feature dependencies (aelfrice.ingest, MemoryStore) port from lab.

    Stdlib-only utilities exposed today:
      verify-clean PATH        contamination gate over a retrieval JSON
      longmemeval-score PREDS GT  scoring without aelfrice imports
    """
    import json
    target = (args.target or "synthetic").lower()

    if target == "synthetic":
        db = ":memory:" if args.db is None else str(Path(args.db))
        store = Store(db)
        try:
            seed_corpus(store)
            report = run_benchmark(
                store, aelfrice_version=_AELFRICE_VERSION, top_k=args.top_k
            )
        finally:
            store.close()
        print(json.dumps(report.to_dict(), indent=2), file=out)  # type: ignore[arg-type]
        return 0

    if target == "verify-clean":
        try:
            from benchmarks import verify_clean
        except ModuleNotFoundError:
            print(
                "aelf bench verify-clean requires the source tree "
                "(benchmarks/ is dev-only and not shipped in the wheel). "
                "Clone the repo and run from the repo root.",
                file=out,  # type: ignore[arg-type]
            )
            return 2
        if not args.rest:
            print("usage: aelf bench verify-clean PATH [PATH ...]", file=out)  # type: ignore[arg-type]
            return 2
        all_clean = True
        for path in args.rest:
            if not verify_clean.verify_file(path):
                all_clean = False
        return 0 if all_clean else 1

    if target == "longmemeval-score":
        try:
            from benchmarks import longmemeval_score
        except ModuleNotFoundError:
            print(
                "aelf bench longmemeval-score requires the source tree "
                "(benchmarks/ is dev-only and not shipped in the wheel). "
                "Clone the repo and run from the repo root.",
                file=out,  # type: ignore[arg-type]
            )
            return 2
        if len(args.rest) < 3:
            print("usage: aelf bench longmemeval-score PREDS GT JUDGE", file=out)  # type: ignore[arg-type]
            return 2
        return longmemeval_score.score(args.rest[0], args.rest[1], args.rest[2])

    if target in _BENCH_INERT_TARGETS:
        phase = _BENCH_INERT_TARGETS[target]
        print(
            f"aelf bench {target}: scaffolded but not runnable at "
            f"aelfrice {_AELFRICE_VERSION}.\n"
            f"Activates in {phase}.\n"
            f"See benchmarks/README.md for the full activation roadmap.",
            file=out,  # type: ignore[arg-type]
        )
        return 2

    print(
        f"aelf bench: unknown target {target!r}.\n"
        f"Known targets: synthetic (default), verify-clean, "
        f"longmemeval-score, "
        f"{', '.join(sorted(_BENCH_INERT_TARGETS))}.",
        file=out,  # type: ignore[arg-type]
    )
    return 2


def _resolve_settings_path(args: argparse.Namespace) -> Path:
    if args.settings_path is not None:
        return Path(args.settings_path)
    scope: SettingsScope = args.scope
    project_root = (
        Path(args.project_root) if args.project_root is not None else None
    )
    return default_settings_path(scope, project_root=project_root)


def _cmd_setup(args: argparse.Namespace, out: object) -> int:
    path = _resolve_settings_path(args)
    result = install_user_prompt_submit_hook(
        path,
        command=args.command,
        timeout=args.timeout,
        status_message=args.status_message,
    )
    if result.already_present:
        print(
            f"hook already installed in {result.path} "
            f"(command={args.command!r})",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            f"installed UserPromptSubmit hook in {result.path} "
            f"(command={args.command!r})",
            file=out,  # type: ignore[arg-type]
        )
    return 0


def _cmd_unsetup(args: argparse.Namespace, out: object) -> int:
    path = _resolve_settings_path(args)
    result = uninstall_user_prompt_submit_hook(path, command=args.command)
    if result.removed == 0:
        print(
            f"no matching hook in {result.path} "
            f"(command={args.command!r})",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            f"removed {result.removed} hook entr"
            f"{'y' if result.removed == 1 else 'ies'} from {result.path} "
            f"(command={args.command!r})",
            file=out,  # type: ignore[arg-type]
        )
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

    p_setup = sub.add_parser(
        "setup",
        help="install the UserPromptSubmit hook in Claude Code settings.json",
    )
    _add_hook_scope_args(p_setup)
    p_setup.add_argument(
        "--command", default=DEFAULT_HOOK_COMMAND,
        help=(
            f"hook command Claude Code will spawn "
            f"(default {DEFAULT_HOOK_COMMAND!r})"
        ),
    )
    p_setup.add_argument(
        "--timeout", type=int, default=None,
        help="hook execution timeout in seconds (default: not set)",
    )
    p_setup.add_argument(
        "--status-message", default=None,
        help="status message Claude Code shows while the hook runs",
    )
    p_setup.set_defaults(func=_cmd_setup)

    p_unsetup = sub.add_parser(
        "unsetup",
        help="remove the UserPromptSubmit hook from Claude Code settings.json",
    )
    _add_hook_scope_args(p_unsetup)
    p_unsetup.add_argument(
        "--command", default=DEFAULT_HOOK_COMMAND,
        help=(
            "hook command string to remove "
            f"(default {DEFAULT_HOOK_COMMAND!r})"
        ),
    )
    p_unsetup.set_defaults(func=_cmd_unsetup)

    p_bench = sub.add_parser(
        "bench",
        help=(
            "run a benchmark target. Default: v0.9.0-rc synthetic harness. "
            "Academic-suite targets (mab, locomo, ...) are scaffolded in "
            "benchmarks/ but inert until their dependencies port from lab."
        ),
    )
    p_bench.add_argument(
        "target", nargs="?", default=None,
        help=(
            "benchmark target: synthetic (default), verify-clean, "
            "longmemeval-score, mab, locomo, longmemeval, structmemeval, "
            "amabench, all. See benchmarks/README.md."
        ),
    )
    p_bench.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="target-specific arguments",
    )
    p_bench.add_argument(
        "--db", default=None,
        help=(
            "(synthetic only) path to an empty SQLite file to seed. "
            "Default: in-memory store (fully reproducible across runs)."
        ),
    )
    p_bench.add_argument(
        "--top-k", type=int, default=5,
        help="(synthetic only) retrieval depth for hit@k (default 5)",
    )
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def _add_hook_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope", choices=list(_VALID_SCOPES), default="user",
        help="settings.json scope (default 'user' = ~/.claude/settings.json)",
    )
    parser.add_argument(
        "--project-root", default=None,
        help="project root for --scope project (default: current directory)",
    )
    parser.add_argument(
        "--settings-path", default=None,
        help=(
            "explicit settings.json path; overrides --scope and "
            "--project-root when set"
        ),
    )


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
