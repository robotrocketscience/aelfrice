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
  doctor                           verify hook commands resolve in settings.json
  setup                            install UserPromptSubmit hook in Claude Code
  unsetup                          remove UserPromptSubmit hook from Claude Code
  upgrade                          print the right pip-upgrade command line
  uninstall                        tear down aelfrice locally + handle DB
  statusline                       emit Claude Code statusline snippet
  bench                            run the v0.9.0-rc benchmark harness

DB path resolves from AELFRICE_DB environment variable when set,
otherwise from <git-common-dir>/aelfrice/memory.db when cwd is inside
a git work-tree, otherwise from ~/.aelfrice/memory.db. The git-common-dir
branch makes worktrees of one repo share a single DB; .git/ is not
git-tracked, so the brain graph never crosses the git boundary. Callers
can run `main(argv=...)` in-process for tests; the `aelf` entry point in
pyproject.toml maps to `main()`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
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
from aelfrice.doctor import diagnose, format_report
from aelfrice.lifecycle import (
    PACKAGE_NAME as _PKG,
    UpdateStatus,
    check_for_update,
    clear_cache as _clear_update_cache,
    is_disabled as _update_check_disabled,
    is_newer,
    maybe_check_for_update_async,
    read_cache as _read_update_cache,
    uninstall as _lifecycle_uninstall,
    upgrade_advice,
)
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.setup import (
    SettingsScope,
    clean_dangling_shims,
    default_settings_path,
    detect_default_scope,
    install_statusline,
    install_user_prompt_submit_hook,
    resolve_hook_command,
    uninstall_statusline,
    uninstall_user_prompt_submit_hook,
)
from aelfrice.store import MemoryStore

DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"
DEFAULT_HOOK_COMMAND: Final[str] = "aelf-hook"
_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}
_LOCK_ID_LEN: Final[int] = 16
_VALID_SCOPES: Final[tuple[SettingsScope, ...]] = ("user", "project")


def _git_common_dir() -> Path | None:
    """Absolute path of cwd's git-common-dir, or None when not in a repo.

    Two worktrees of one repo share a --git-common-dir, so resolving
    against this gives them a single shared DB. Returns None when cwd
    is outside any git work-tree, when the `git` binary is missing, or
    when the rev-parse call fails for any reason — callers fall back to
    the home-dir path.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    return Path(raw).resolve()


def db_path() -> Path:
    """Resolve the DB path.

    Resolution order:
    1. $AELFRICE_DB (explicit override; honoured even inside a git repo).
    2. <git-common-dir>/aelfrice/memory.db when cwd is in a git work-tree.
    3. ~/.aelfrice/memory.db (legacy global fallback for non-git dirs).

    The DB stays under .git/, which git does not track — the brain
    graph never crosses the git boundary.
    """
    override = os.environ.get("AELFRICE_DB")
    if override:
        return Path(override)
    git_dir = _git_common_dir()
    if git_dir is not None:
        return git_dir / "aelfrice" / DEFAULT_DB_FILENAME
    return DEFAULT_DB_DIR / DEFAULT_DB_FILENAME


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(str(p))


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


def _cmd_resolve(args: argparse.Namespace, out: object) -> int:
    """Resolve all unresolved CONTRADICTS edges via the v1.0.1 tie-breaker."""
    _ = args
    from aelfrice.contradiction import (
        auto_resolve_all_contradictions,
        find_unresolved_contradictions,
    )

    store = _open_store()
    try:
        unresolved = find_unresolved_contradictions(store)
        if not unresolved:
            print("no unresolved contradictions", file=out)  # type: ignore[arg-type]
            return 0
        results = auto_resolve_all_contradictions(store)
    finally:
        store.close()
    for r in results:
        print(
            f"resolved: {r.winner_id} supersedes {r.loser_id} "
            f"({r.rule_fired})",
            file=out,  # type: ignore[arg-type]
        )
    skipped = len(unresolved) - len(results)
    if skipped > 0:
        print(
            f"skipped {skipped} pair(s) with missing endpoints",
            file=out,  # type: ignore[arg-type]
        )
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
        store = MemoryStore(db)
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


def _effective_scope(args: argparse.Namespace) -> SettingsScope:
    """Return the explicit `--scope` if given, else auto-detect."""
    scope = getattr(args, "scope", None)
    if scope == "user" or scope == "project":
        return scope
    project_root = (
        Path(args.project_root) if args.project_root is not None else None
    )
    return detect_default_scope(cwd=project_root)


def _resolve_settings_path(args: argparse.Namespace) -> Path:
    if args.settings_path is not None:
        return Path(args.settings_path)
    project_root = (
        Path(args.project_root) if args.project_root is not None else None
    )
    return default_settings_path(_effective_scope(args), project_root=project_root)


def _cmd_setup(args: argparse.Namespace, out: object) -> int:
    scope = _effective_scope(args)
    path = _resolve_settings_path(args)
    command = args.command if args.command is not None else resolve_hook_command(scope)
    cleanup = clean_dangling_shims()
    for removed_path in cleanup.removed:
        print(
            f"cleaned dangling shim: {removed_path}",
            file=out,  # type: ignore[arg-type]
        )
    result = install_user_prompt_submit_hook(
        path,
        command=command,
        timeout=args.timeout,
        status_message=args.status_message,
    )
    if result.already_present:
        print(
            f"hook already installed in {result.path} "
            f"(command={command!r})",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            f"installed UserPromptSubmit hook in {result.path} "
            f"(command={command!r})",
            file=out,  # type: ignore[arg-type]
        )
    if not args.no_statusline:
        sl = install_statusline(path)
        if sl.mode == "installed":
            print(
                f"installed statusline in {sl.path} "
                f"(command='aelf statusline')",
                file=out,  # type: ignore[arg-type]
            )
        elif sl.mode == "composed":
            print(
                f"composed statusline into existing command in {sl.path}",
                file=out,  # type: ignore[arg-type]
            )
        elif sl.mode == "already":
            pass  # silent: already wired
        elif sl.mode == "skipped":
            print(
                f"statusline NOT installed in {sl.path}: existing "
                f"statusLine looks complex (shell metacharacters). "
                f"To enable update notifications append "
                f"' ; aelf statusline 2>/dev/null' to your existing "
                f"statusLine command manually.",
                file=out,  # type: ignore[arg-type]
            )
    return 0


def _cmd_unsetup(args: argparse.Namespace, out: object) -> int:
    path = _resolve_settings_path(args)
    if args.command is None:
        result = uninstall_user_prompt_submit_hook(
            path, command_basename=DEFAULT_HOOK_COMMAND
        )
        match_label = f"basename={DEFAULT_HOOK_COMMAND!r}"
    else:
        result = uninstall_user_prompt_submit_hook(path, command=args.command)
        match_label = f"command={args.command!r}"
    if result.removed == 0:
        print(
            f"no matching hook in {result.path} ({match_label})",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            f"removed {result.removed} hook entr"
            f"{'y' if result.removed == 1 else 'ies'} from {result.path} "
            f"({match_label})",
            file=out,  # type: ignore[arg-type]
        )
    sl = uninstall_statusline(path)
    if sl.mode == "removed":
        print(
            f"removed statusline from {sl.path}",
            file=out,  # type: ignore[arg-type]
        )
    elif sl.mode == "unwrapped":
        print(
            f"restored prior statusline command in {sl.path}",
            file=out,  # type: ignore[arg-type]
        )
    return 0


def _read_password(args: argparse.Namespace) -> str | None:
    """Read the archive password.

    Order of precedence:
      1. --password-stdin: read first line of stdin (no trailing newline,
         no shell history exposure).
      2. Interactive: getpass twice, must match.
    Never accepts password on argv (would leak via ps/proc/cmdline).
    """
    if args.password_stdin:
        line = sys.stdin.readline()
        return line.rstrip("\n\r")
    import getpass

    pw1 = getpass.getpass("archive password: ")
    pw2 = getpass.getpass("confirm password: ")
    if pw1 != pw2:
        return None
    return pw1


def _cmd_uninstall(args: argparse.Namespace, out: object) -> int:
    """Tear down aelfrice's local footprint with redundant data gates.

    Mutually exclusive --keep-db / --purge / --archive PATH.
    --purge gates:
      1. Must be passed explicitly (default is none -> error w/ help).
      2. Print the affected DB path + size.
      3. Require user to type 'PURGE' verbatim (or --yes to skip).
      4. Final [y/N] confirmation (or --yes to skip).
    Default: also runs `aelf unsetup` for the user-scope settings.json
    so the hook + statusline are removed in one go (--keep-hook opts
    out).
    """
    chosen = sum(
        [bool(args.keep_db), bool(args.purge), args.archive is not None]
    )
    if chosen == 0:
        print(
            "error: pick one of --keep-db, --purge, --archive PATH "
            "(see 'aelf uninstall --help')",
            file=sys.stderr,
        )
        return 2
    if chosen > 1:
        print(
            "error: --keep-db, --purge, and --archive are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    target_db = db_path()

    # --- Gate 1+2+3: redundant prompts before destroying data ---------
    if args.purge:
        if target_db.exists():
            try:
                size = target_db.stat().st_size
                size_str = f"{size:,} bytes"
            except OSError:
                size_str = "unknown size"
            print(
                f"--purge will permanently delete {target_db} ({size_str}).",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"--purge target {target_db} does not exist; nothing to delete.",
                file=out,  # type: ignore[arg-type]
            )
        if not args.yes:
            try:
                ack = input("type 'PURGE' to confirm: ")
            except EOFError:
                ack = ""
            if ack != "PURGE":
                print("aborted (no 'PURGE' confirmation).", file=out)  # type: ignore[arg-type]
                return 1
            try:
                final = input("Last chance. Cannot be undone. Continue? [y/N]: ")
            except EOFError:
                final = ""
            if final.strip().lower() not in {"y", "yes"}:
                print("aborted.", file=out)  # type: ignore[arg-type]
                return 1

    # --- Archive path: collect password BEFORE touching the DB --------
    password: str | None = None
    archive_path: Path | None = None
    if args.archive is not None:
        archive_path = Path(args.archive)
        password = _read_password(args)
        if not password:
            print(
                "aborted: empty or non-matching password.",
                file=sys.stderr,
            )
            return 1

    # --- Apply data disposition --------------------------------------
    try:
        result = _lifecycle_uninstall(
            target_db,
            keep_db=args.keep_db,
            purge=args.purge,
            archive_path=archive_path,
            archive_password=password,
        )
    except RuntimeError as exc:
        # cryptography missing -- surface the install hint, exit non-zero.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.mode == "kept":
        print(
            f"DB preserved at {target_db}.",
            file=out,  # type: ignore[arg-type]
        )
    elif result.mode == "purged":
        print(
            f"DB deleted: {target_db}.",
            file=out,  # type: ignore[arg-type]
        )
    elif result.mode == "archived":
        print(
            f"DB encrypted to {result.archive_path}, original deleted.",
            file=out,  # type: ignore[arg-type]
        )
        print(
            "(decrypt later via aelfrice.lifecycle.decrypt_archive(path, pw))",
            file=out,  # type: ignore[arg-type]
        )

    # --- Clear update-check cache (no point keeping it post-uninstall)
    _clear_update_cache()

    # --- Hook + statusline removal (default on, --keep-hook opts out)-
    if not args.keep_hook:
        # Delegate by re-using the existing _cmd_unsetup path: pretend
        # the user invoked `aelf unsetup --scope user`. command=None
        # triggers basename-match cleanup, which catches both bare and
        # absolute-path installs.
        unsetup_args = argparse.Namespace(
            scope="user",
            project_root=None,
            settings_path=args.settings_path,
            command=None,
            cmd="unsetup",
            func=_cmd_unsetup,
        )
        _cmd_unsetup(unsetup_args, out)

    print(
        f"\nFinish removing aelfrice with: pip uninstall {_PKG}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_statusline(args: argparse.Namespace, out: object) -> int:
    """Print a statusline prefix snippet ('' when no update pending).

    Composes with any existing statusline command via shell:
    'original-cmd ; aelf statusline 2>/dev/null'. Reads the cache only,
    never networks. End-of-line is intentionally absent so the snippet
    sits inline with whatever else is on the bar.
    """
    _ = args
    from aelfrice.statusline import render

    snippet = render()
    if snippet:
        # Use write rather than print so we don't append a newline that
        # would force a line break in the host's statusline.
        out.write(snippet)  # type: ignore[attr-defined]
    return 0


def _cmd_upgrade(args: argparse.Namespace, out: object) -> int:
    """Print the right upgrade command for this install context.

    Does NOT shell out to pip itself: replacing the running package
    mid-process is unreliable on Windows and can leave the user with a
    broken interpreter. We tell the user the exact line; they run it.

    --check: only print "up to date" / "update available" status,
    suppress the upgrade command line. Useful for scripts that want
    a yes/no answer without copy-paste material.
    """
    advice = upgrade_advice()
    # Force a fresh sync check unless explicitly disabled. This is the
    # one CLI surface where the user has explicitly asked about updates,
    # so we ignore the 24h TTL.
    if _update_check_disabled():
        status = _read_update_cache()
    else:
        status = check_for_update()
        if status.installed == "" and status.latest == "":
            # Network failed: fall back to whatever is cached.
            status = _read_update_cache()
    if status.update_available:
        print(
            f"aelfrice {status.latest} available "
            f"(installed: {status.installed or _AELFRICE_VERSION})",
            file=out,  # type: ignore[arg-type]
        )
        if status.sha256:
            print(
                f"verify: sha256:{status.sha256}",
                file=out,  # type: ignore[arg-type]
            )
            print(
                f"        https://pypi.org/project/{_PKG}/{status.latest}/",
                file=out,  # type: ignore[arg-type]
            )
        if not args.check:
            print(
                f"run: {advice.command}",
                file=out,  # type: ignore[arg-type]
            )
        return 0
    # No update or unknown -- be explicit.
    if status.latest and not status.update_available:
        print(
            f"aelfrice is up to date "
            f"(installed: {status.installed or _AELFRICE_VERSION}, "
            f"latest: {status.latest})",
            file=out,  # type: ignore[arg-type]
        )
        # The cache says we're current; clear any stale "available"
        # marker that might still be sitting around from before this
        # check.
        _clear_update_cache()
    else:
        print(
            f"aelfrice {_AELFRICE_VERSION} (no update info available)",
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


def _cmd_doctor(args: argparse.Namespace, out: object) -> int:
    """Diagnose hook + statusline command resolution in settings.json files.

    Exit 0 when nothing is broken, exit 1 when at least one broken
    command is found. CI-friendly.
    """
    project_root = Path(args.project_root) if args.project_root else None
    user_settings = (
        Path(args.user_settings) if args.user_settings else None
    )
    report = diagnose(
        user_settings=user_settings, project_root=project_root
    )
    print(format_report(report), file=out)  # type: ignore[arg-type]
    return 1 if report.broken else 0


# --- Dispatcher ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aelf",
        description="Bayesian memory designed for feedback-driven learning.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aelf {_AELFRICE_VERSION}",
        help="print the installed aelfrice version and exit",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_onboard = sub.add_parser(
        "onboard",
        help="scan a project and ingest beliefs",
        epilog=(
            "Power users: tune the onboard noise filter via "
            ".aelfrice.toml at the project root. Disable categories, "
            "override the fragment threshold, or add project-specific "
            "exclude_words / exclude_phrases. See docs/CONFIG.md for "
            "the schema and worked examples."
        ),
    )
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

    p_resolve = sub.add_parser(
        "resolve",
        help="resolve unresolved CONTRADICTS edges via the v1.0.1 tie-breaker",
        epilog=(
            "Picks a winner per precedence (user_stated > user_corrected "
            "> document_recent; ties broken by recency, then id) and "
            "creates a SUPERSEDES edge from winner to loser. Each "
            "resolution writes an audit row to feedback_history with "
            "source='contradiction_tiebreaker:<rule>'. Idempotent — "
            "already-resolved pairs are skipped."
        ),
    )
    p_resolve.set_defaults(func=_cmd_resolve)

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

    p_doctor = sub.add_parser(
        "doctor",
        help=(
            "diagnose Claude Code hook + statusline commands in user "
            "and project settings.json. Exits 1 if any are broken."
        ),
    )
    p_doctor.add_argument(
        "--user-settings", default=None,
        help="override user settings.json path (default: ~/.claude/settings.json)",
    )
    p_doctor.add_argument(
        "--project-root", default=None,
        help="override project root for project-scope check (default: cwd)",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    p_setup = sub.add_parser(
        "setup",
        help="install the UserPromptSubmit hook in Claude Code settings.json",
    )
    _add_hook_scope_args(p_setup)
    p_setup.add_argument(
        "--command", default=None,
        help=(
            "hook command Claude Code will spawn. Default: auto-resolved "
            "absolute path to aelf-hook (project venv for project scope, "
            "$PATH for user scope)."
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
    p_setup.add_argument(
        "--no-statusline", action="store_true",
        help="skip the auto-install of the update-notifier statusline snippet",
    )
    p_setup.set_defaults(func=_cmd_setup)

    p_uninstall = sub.add_parser(
        "uninstall",
        help=(
            "tear down aelfrice locally: pick exactly one of --keep-db / "
            "--purge / --archive PATH for the brain-graph DB. Also runs "
            "unsetup unless --keep-hook is given."
        ),
    )
    p_uninstall.add_argument(
        "--keep-db", action="store_true",
        help="leave ~/.aelfrice/memory.db untouched (safe default for review)",
    )
    p_uninstall.add_argument(
        "--purge", action="store_true",
        help=(
            "PERMANENTLY DELETE the brain-graph DB. Requires typing "
            "'PURGE' followed by [y] unless --yes is passed."
        ),
    )
    p_uninstall.add_argument(
        "--archive", default=None,
        help=(
            "encrypt the DB to this path with a password, then delete "
            "the original (requires: pip install 'aelfrice[archive]')"
        ),
    )
    p_uninstall.add_argument(
        "--password-stdin", action="store_true",
        help="read archive password from stdin (no interactive prompt)",
    )
    p_uninstall.add_argument(
        "--yes", action="store_true",
        help=(
            "skip confirmation prompts (still requires --purge to be "
            "passed explicitly -- never auto-purges)"
        ),
    )
    p_uninstall.add_argument(
        "--keep-hook", action="store_true",
        help="do not run unsetup; leave the Claude Code hook in place",
    )
    p_uninstall.add_argument(
        "--settings-path", default=None,
        help="explicit settings.json for the unsetup half (defaults to user-scope)",
    )
    p_uninstall.set_defaults(func=_cmd_uninstall)

    p_statusline = sub.add_parser(
        "statusline",
        help=(
            "emit a statusline prefix snippet (orange update banner "
            "or empty). Compose with: 'your-cmd ; aelf statusline 2>/dev/null'"
        ),
    )
    p_statusline.set_defaults(func=_cmd_statusline)

    p_upgrade = sub.add_parser(
        "upgrade",
        help="print the right pip-upgrade command for this install context",
    )
    p_upgrade.add_argument(
        "--check", action="store_true",
        help="only report status, do not print the upgrade command line",
    )
    p_upgrade.set_defaults(func=_cmd_upgrade)

    p_unsetup = sub.add_parser(
        "unsetup",
        help="remove the UserPromptSubmit hook from Claude Code settings.json",
    )
    _add_hook_scope_args(p_unsetup)
    p_unsetup.add_argument(
        "--command", default=None,
        help=(
            "exact hook command string to remove. Default: remove every "
            f"entry whose command basename is {DEFAULT_HOOK_COMMAND!r} "
            "(matches both bare-name and absolute-path installs)."
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
        "--scope", choices=list(_VALID_SCOPES), default=None,
        help=(
            "settings.json scope. Default: auto -- 'project' if cwd has a "
            ".venv matching the active interpreter, else 'user'."
        ),
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


_UPDATE_CHECK_SKIP_CMDS: Final[frozenset[str]] = frozenset(
    {"upgrade", "uninstall", "statusline"}
)


def _maybe_emit_update_banner(cmd: str | None) -> None:
    """Print a one-line orange notice on stderr if an update is pending.

    Skipped for commands that already speak about updates themselves
    (`aelf upgrade`, `aelf uninstall`, `aelf statusline`) so we do not
    double-print or stomp on machine-readable output.
    """
    if cmd in _UPDATE_CHECK_SKIP_CMDS:
        return
    if _update_check_disabled():
        return
    status = _read_update_cache()
    if not status.update_available:
        return
    print(
        f"\x1b[38;5;208m⬆ aelfrice {status.latest} available, "
        f"run: aelf upgrade\x1b[0m",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None, out: object = None) -> int:
    """CLI entry point. Returns process exit code.

    `argv` lets tests pass synthetic args; defaults to sys.argv[1:].
    `out` lets tests capture stdout; defaults to sys.stdout.

    Two things happen for free on every invocation:
      1. A TTL-gated background update check is fired (detached
         subprocess, never blocks).
      2. After the command runs, if the cache says an update is
         available, an orange banner is printed to stderr.

    Both are skipped if AELF_NO_UPDATE_CHECK is set, and the banner
    is skipped for commands that already handle update messaging
    themselves (upgrade / uninstall / statusline).
    """
    if out is None:
        out = sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = getattr(args, "cmd", None)
    if not _update_check_disabled() and cmd not in _UPDATE_CHECK_SKIP_CMDS:
        # Fire-and-forget: cache TTL gates duplicate work, never blocks.
        maybe_check_for_update_async()
    code = int(args.func(args, out))
    _maybe_emit_update_banner(cmd)
    return code
