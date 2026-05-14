"""Eleven-command CLI matching the MVP user surface.

Commands:
  onboard <path>                   scan a project and ingest beliefs
  search <query> [--budget N]      L0 locked + L1 FTS5 retrieval
  lock <statement>                 insert (or upgrade) a user-locked belief
  locked [--pressured]             list locked beliefs
  demote <id>                      manually demote a lock to none
  confirm <id> [--source S]        explicitly affirm a belief (bumps Beta-Bernoulli alpha)
  feedback <id> <used|harmful>     apply one Bayesian feedback event
  stats                            summary of belief / lock / history counts
  health                           structural auditor (orphan threads, FTS5 sync, locked contradictions)
  status                           alias for health
  regime                           v1.0 regime classifier (supersede / ignore / mixed)
  migrate                          copy beliefs from legacy ~/.aelfrice/memory.db
  doctor                           verify hook commands resolve in settings.json
  setup                            install UserPromptSubmit hook in Claude Code
  unsetup                          remove UserPromptSubmit hook from Claude Code
  upgrade-cmd                      print the right pip-upgrade command line (renamed from `upgrade` at #427)
  uninstall                        tear down aelfrice locally + handle DB
  statusline                       emit Claude Code statusline snippet
  bench                            run the v0.9.0-rc benchmark harness
  project-warm <path>              CwdChanged hook entry — pre-load the project's belief cache

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
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from aelfrice.doc_linker import ANCHOR_MANUAL, link_belief_to_document
from aelfrice.auditor import (
    CORPUS_MIN_DEFAULT as AUDIT_CORPUS_MIN_DEFAULT,
    SEVERITY_FAIL as AUDIT_SEVERITY_FAIL,
    SEVERITY_WARN as AUDIT_SEVERITY_WARN,
    audit,
)
from aelfrice.feedback import apply_feedback
from aelfrice.migrate import (
    MigrateReport,
    default_legacy_db_path,
    migrate as _migrate_action,
)
from aelfrice.health import (
    REGIME_INSUFFICIENT_DATA,
    assess_health,
    compute_features,
    regime_description,
)
from aelfrice.models import (
    CORROBORATION_SOURCE_CLI_REMEMBER,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    INGEST_SOURCE_CLI_REMEMBER,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_STATED,
    ORIGIN_USER_VALIDATED,
    Phantom,
)
from aelfrice.bfs_multihop import expand_bfs
from aelfrice.reason import (
    ConsequencePath,
    Impasse,
    Verdict,
    classify as _reason_classify,
    derive_paths as _reason_derive_paths,
    dispatch_policy as _reason_dispatch_policy,
    suggested_updates as _reason_suggested_updates,
)
from aelfrice import wonder_consolidation
from aelfrice import __version__ as _AELFRICE_VERSION
from aelfrice import auto_install as _auto_install
from aelfrice.auto_install import auto_install_at_cli_entry
from aelfrice.benchmark import run_benchmark, seed_corpus
from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    check_onboard_candidates,
    start_onboard_session,
)
from aelfrice.derivation import DerivationInput, derive
from aelfrice.derivation_worker import run_worker
from aelfrice.doctor import (
    DORMANT_IDLE_DAYS_DEFAULT,
    DormantDB,
    HookPruneResult,
    _check_dormant_dbs,
    classify_orphans as _classify_orphans,
    diagnose,
    format_orphan_feedback_report as _format_orphan_feedback_report,
    format_orphan_report as _format_orphan_report,
    format_promotion_report as _format_promotion_report,
    format_report,
    gc_orphan_feedback as _gc_orphan_feedback,
    promote_retention as _promote_retention,
    prune_broken_aelf_hooks,
)
from aelfrice.llm_classifier import (
    ENV_API_KEY as _LLM_ENV_API_KEY,
    LLMAuthError as _LLMAuthError,
    LLMConfig as _LLMConfig,
    LLMTokenCapExceeded as _LLMTokenCapExceeded,
    ScannerRouter as _LLMScannerRouter,
    check_gates as _llm_check_gates,
    format_telemetry_line as _llm_format_telemetry_line,
    is_sentinel_valid as _llm_is_sentinel_valid,
    prompt_for_consent as _llm_prompt_for_consent,
    read_sentinel as _llm_read_sentinel,
    resolve_enabled as _llm_resolve_enabled,
    revoke_sentinel as _llm_revoke_sentinel,
    sentinel_path as _llm_sentinel_path,
    write_sentinel as _llm_write_sentinel,
)
from aelfrice.lifecycle import (
    PACKAGE_NAME as _PKG,
    check_for_update,
    clear_cache as _clear_update_cache,
    format_update_banner as _format_update_banner,
    is_disabled as _update_check_disabled,
    detect_reachable_installs,
    maybe_check_for_update_async,
    maybe_migrate_to_uv,
    read_cache as _read_update_cache,
    uninstall as _lifecycle_uninstall,
    upgrade_advice,
)
from aelfrice.project_warm import (
    DEFAULT_DEBOUNCE_SECONDS as _PROJECT_WARM_DEBOUNCE,
    warm_path as _warm_path,
)
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.session_resolution import resolve_session_id
from aelfrice.setup import (
    COMMIT_INGEST_SCRIPT_NAME,
    SEARCH_TOOL_BASH_SCRIPT_NAME,
    SEARCH_TOOL_SCRIPT_NAME,
    SESSION_START_HOOK_SCRIPT_NAME,
    SLASH_COMMANDS_DIR_DEFAULT,  # noqa: F401 — re-exported for monkeypatch in tests
    STOP_HOOK_SCRIPT_NAME,
    SettingsScope,
    TRANSCRIPT_LOGGER_SCRIPT_NAME,
    clean_dangling_shims,
    default_settings_path,
    detect_default_scope,
    install_commit_ingest_hook,
    install_search_tool_bash_hook,
    install_search_tool_hook,
    install_pre_compact_hook,
    install_session_start_hook,
    install_slash_commands,
    install_statusline,
    install_stop_hook,
    install_transcript_ingest_hooks,
    install_user_prompt_submit_hook,
    resolve_commit_ingest_command,
    resolve_search_tool_bash_command,
    resolve_search_tool_command,
    resolve_hook_command,
    resolve_pre_compact_hook_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
    uninstall_commit_ingest_hook,
    uninstall_search_tool_bash_hook,
    uninstall_search_tool_hook,
    uninstall_pre_compact_hook,
    uninstall_session_start_hook,
    uninstall_slash_commands,
    uninstall_statusline,
    uninstall_stop_hook,
    uninstall_transcript_ingest_hooks,
    uninstall_user_prompt_submit_hook,
)
from aelfrice.db_paths import (
    DEFAULT_DB_DIR,
    DEFAULT_DB_FILENAME,
    _ensure_parent_dir,
    _git_common_dir,
    _open_store,
    db_path,
)
from aelfrice.store import MemoryStore

DEFAULT_HOOK_COMMAND: Final[str] = "aelf-hook"
DEFAULT_PRE_COMPACT_HOOK_COMMAND: Final[str] = "aelf-pre-compact-hook"
_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}
_VALID_SCOPES: Final[tuple[SettingsScope, ...]] = ("user", "project")


def _git_first_commit_age_days() -> int | None:
    """Days since cwd's first git commit, or None when unknown.

    Used by the corpus-volume warning in `aelf health` (issue #116) so
    a brand-new project does not get nagged for an empty store. Falls
    back to None when not in a repo, when git is missing, or when the
    repo has zero commits yet.
    """
    try:
        result = subprocess.run(
            [
                "git", "log", "--reverse", "--format=%aI",
                "--max-parents=0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.splitlines()[0] if result.stdout else ""
    line = line.strip()
    if not line:
        return None
    try:
        first = datetime.fromisoformat(line)
    except ValueError:
        return None
    now = datetime.now(first.tzinfo) if first.tzinfo else datetime.now()
    delta = now - first
    return max(0, delta.days)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_corpus_min() -> int:
    """Read AELFRICE_CORPUS_MIN env var, fall back to auditor default.

    Negative or non-integer values silently fall through to the default
    so a typo can never make health crash; wrong values just mean the
    warning fires at the wrong threshold.
    """
    raw = os.environ.get("AELFRICE_CORPUS_MIN")
    if raw is None:
        return AUDIT_CORPUS_MIN_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        return AUDIT_CORPUS_MIN_DEFAULT
    if n < 0:
        return AUDIT_CORPUS_MIN_DEFAULT
    return n


# --- Command handlers ----------------------------------------------------


def _load_llm_config(root: Path) -> _LLMConfig:
    """Walk up from `root` looking for `.aelfrice.toml`; parse
    `[onboard.llm]`. Returns the default config if no file is found,
    the file is malformed, or the table is missing.

    Mirrors `NoiseConfig.discover` resilience: failures degrade to
    defaults with a stderr trace; never raises.
    """
    import tomllib

    current = root.resolve() if root.exists() else root
    seen: set[Path] = set()
    candidate = current if current.is_dir() else current.parent
    while candidate not in seen:
        seen.add(candidate)
        cfg_path = candidate / ".aelfrice.toml"
        if cfg_path.is_file():
            try:
                raw = cfg_path.read_bytes()
                parsed: Any = tomllib.loads(
                    raw.decode("utf-8", errors="replace")
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice llm_classifier: cannot read {cfg_path}: {exc}",
                    file=sys.stderr,
                )
                return _LLMConfig.default()
            if not isinstance(parsed, dict):
                return _LLMConfig.default()
            from typing import cast
            parsed_dict = cast(dict[str, Any], parsed)
            section = parsed_dict.get("onboard", {})
            if isinstance(section, dict):
                section_dict = cast(dict[str, Any], section)
                llm_any = section_dict.get("llm", {})
                if isinstance(llm_any, dict):
                    return _LLMConfig.from_mapping(
                        cast(dict[str, Any], llm_any)
                    )
            return _LLMConfig.default()
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return _LLMConfig.default()


def _resolve_llm_flag(args: argparse.Namespace) -> bool | None:
    """Resolve the --llm-classify CLI flag to True/False/None.

    None: flag not present.
    True: --llm-classify or --llm-classify=true.
    False: --llm-classify=false.
    """
    raw = getattr(args, "llm_classify", None)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "y", ""):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return True  # any other value: opt-in


def _cmd_onboard_emit_candidates(args: argparse.Namespace, out: object) -> int:
    """Run the scanner+filter pipeline, persist a PENDING onboard session,
    and print a JSON payload the host can hand to a classifier subagent.
    No network call. No LLM gates. Pure local IO.
    """
    if not args.path:
        print(
            "aelf onboard --emit-candidates: <path> is required.",
            file=sys.stderr,
        )
        return 2
    repo_path = Path(args.path)
    store = _open_store()
    try:
        result = start_onboard_session(store, repo_path)
    finally:
        store.close()
    payload = {
        "session_id": result.session_id,
        "n_already_present": result.n_already_present,
        "sentences": [
            {"index": s.index, "text": s.text, "source": s.source}
            for s in result.sentences
        ],
    }
    print(json.dumps(payload), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_onboard_accept_classifications(
    args: argparse.Namespace, out: object
) -> int:
    """Apply host-supplied classifications to a pending onboard session.
    Reads `[{index, belief_type, persist}, ...]` JSON from a file or
    stdin (`-`), calls accept_classifications, prints a JSON summary.
    No network call.
    """
    session_id = getattr(args, "session_id", None)
    if not session_id:
        print(
            "aelf onboard --accept-classifications: --session-id is required.",
            file=sys.stderr,
        )
        return 2
    src = getattr(args, "classifications_file", None)
    if not src:
        print(
            "aelf onboard --accept-classifications: "
            "--classifications-file is required (use '-' for stdin).",
            file=sys.stderr,
        )
        return 2
    try:
        if src == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(src).read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"aelf onboard --accept-classifications: cannot read {src}: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"aelf onboard --accept-classifications: invalid JSON in {src}: {exc}",
            file=sys.stderr,
        )
        return 1
    if not isinstance(data, list):
        print(
            "aelf onboard --accept-classifications: expected a JSON array "
            "of {index, belief_type, persist} objects.",
            file=sys.stderr,
        )
        return 1
    classifications: list[HostClassification] = []
    for d in data:
        if not isinstance(d, dict):
            print(
                "aelf onboard --accept-classifications: array entries must "
                "be objects with index/belief_type/persist keys.",
                file=sys.stderr,
            )
            return 1
        try:
            classifications.append(
                HostClassification(
                    index=int(d["index"]),
                    belief_type=str(d["belief_type"]),
                    persist=bool(d["persist"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(
                f"aelf onboard --accept-classifications: bad entry {d}: {exc}",
                file=sys.stderr,
            )
            return 1
    store = _open_store()
    try:
        try:
            result = accept_classifications(store, session_id, classifications)
        except ValueError as exc:
            print(
                f"aelf onboard --accept-classifications: {exc}",
                file=sys.stderr,
            )
            return 1
    finally:
        store.close()
    print(
        json.dumps(
            {
                "session_id": result.session_id,
                "inserted": result.inserted,
                "skipped_non_persisting": result.skipped_non_persisting,
                "skipped_existing": result.skipped_existing,
                "skipped_unclassified": result.skipped_unclassified,
            }
        ),
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_onboard_check(args: argparse.Namespace, out: object) -> int:
    """Read-only pre-scan: print n_already_present / n_new and exit.

    Runs the extractor + id-dedup pipeline against the store without
    writing an onboard_sessions row or inserting beliefs. Lets the user
    see what a re-onboard would do before paying the classification
    cost (#761). No LLM gates, no network, no consent prompt.
    """
    if not args.path:
        print(
            "aelf onboard --check: <path> is required.",
            file=sys.stderr,
        )
        return 2
    repo_path = Path(args.path)
    store = _open_store()
    try:
        result = check_onboard_candidates(store, repo_path)
    finally:
        store.close()
    total = result.n_already_present + result.n_new
    pct_present = (
        (result.n_already_present * 100) // total if total > 0 else 0
    )
    print(
        f"path: {result.repo_path}\n"
        f"already present: {result.n_already_present} candidates "
        f"({pct_present}% of {total})\n"
        f"new since last onboard: {result.n_new} candidates\n"
        f"(read-only pre-scan; no beliefs inserted, no session persisted)",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_onboard(args: argparse.Namespace, out: object) -> int:
    # --check: read-only pre-scan, bypasses every other onboard path.
    # Runs before --emit-candidates / --accept-classifications because
    # it makes no store writes and short-circuits cleanly (#761).
    if getattr(args, "check", False):
        return _cmd_onboard_check(args, out)
    # --emit-candidates / --accept-classifications: low-level entry
    # points used by the /aelf:onboard slash command to drive the
    # polymorphic onboard handshake from a Claude Code session via
    # Haiku Task subagents (no API key, no network from this CLI).
    if getattr(args, "emit_candidates", False):
        return _cmd_onboard_emit_candidates(args, out)
    if getattr(args, "accept_classifications", False):
        return _cmd_onboard_accept_classifications(args, out)

    if not args.path:
        print(
            "aelf onboard: <path> is required (unless using "
            "--accept-classifications). See `aelf onboard --help`.",
            file=sys.stderr,
        )
        return 2
    path = Path(args.path)

    # --revoke-consent: remove sentinel and exit; no scan, no network.
    if getattr(args, "revoke_consent", False):
        sentinel = _llm_sentinel_path()
        removed = _llm_revoke_sentinel(sentinel)
        if removed:
            print(f"aelf: revoked LLM-classify consent ({sentinel})", file=out)  # type: ignore[arg-type]
        else:
            print(
                f"aelf: no LLM-classify consent sentinel at {sentinel}",
                file=out,  # type: ignore[arg-type]
            )
        return 0

    cfg = _load_llm_config(path)
    flag = _resolve_llm_flag(args)
    enabled = _llm_resolve_enabled(flag=flag, config_enabled=cfg.enabled)
    # Explicit opt-in is the CLI flag. When `enabled=True` came only
    # from the post-v1.5 default (no flag, no explicit config-disable),
    # we treat missing SDK / missing API key / non-TTY consent as soft
    # failures and fall back to the regex classifier silently.
    dry_run = bool(getattr(args, "dry_run", False))
    # Explicit opt-in is the CLI flag, OR --dry-run (which only makes
    # sense as a preview of the LLM path; soft-falling to regex would
    # silently ignore the user's intent).
    is_explicit_opt_in = (flag is True) or dry_run

    if not enabled:
        # Explicit opt-out: regex classifier, no LLM imports, zero
        # network. Reachable via --llm-classify=false or
        # [onboard.llm].enabled = false in .aelfrice.toml.
        if dry_run:
            print(
                "aelf: --dry-run requires --llm-classify; "
                "the regex path has nothing to dry-run.",
                file=sys.stderr,
            )
            return 1
        return _run_regex_onboard(args, out)

    # Gates 1-3 (extra installed, env var, opt-in resolved).
    gate = _llm_check_gates(
        enabled=enabled,
        model=cfg.model,
        treat_missing_as_soft=not is_explicit_opt_in,
    )
    if not gate.pass_all:
        if gate.exit_code is not None:
            if gate.message:
                print(gate.message, file=sys.stderr)
            return gate.exit_code
        # Soft-fall (default-derived path with missing SDK / key, or
        # an unrecognised configuration). Run the regex onboard.
        return _run_regex_onboard(args, out)

    # Gate 4: consent. --dry-run skips the prompt AND the network.
    sentinel = _llm_sentinel_path()
    sentinel_record = _llm_read_sentinel(sentinel)
    if not _llm_is_sentinel_valid(sentinel_record, model=cfg.model):
        if dry_run:
            # Dry-run must NOT prompt and MUST NOT write the sentinel.
            print(
                "aelf: dry-run skipping consent prompt "
                f"(no valid sentinel at {sentinel}). No network call.",
                file=sys.stderr,
            )
        else:
            prompt = _llm_prompt_for_consent()
            if not prompt.accepted:
                if is_explicit_opt_in:
                    print(
                        f"aelf: --llm-classify aborted ({prompt.reason}); "
                        "no network call.",
                        file=sys.stderr,
                    )
                    return 1
                # Default-derived consent rejection (or non-TTY) →
                # silent fall-through to regex onboard. The user did
                # not actively opt in; failing here would be hostile.
                if prompt.reason != "non-tty":
                    print(
                        "aelf: LLM classification declined; "
                        "falling back to regex classifier.",
                        file=sys.stderr,
                    )
                return _run_regex_onboard(args, out)
            _llm_write_sentinel(sentinel, model=cfg.model)

    # Dry-run: print candidates without contacting the network.
    if dry_run:
        return _run_llm_dry_run(args, out)

    # Real LLM path.
    return _run_llm_onboard(args, out, cfg)


def _run_regex_onboard(args: argparse.Namespace, out: object) -> int:
    """Default v1.0/v1.2 regex onboard. Unchanged behaviour."""
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


def _run_llm_dry_run(args: argparse.Namespace, out: object) -> int:
    """Print the candidates that would be sent, without network IO.

    Spec § 4.4: full extractor + noise filter + dedup pipeline,
    print one line per candidate prefixed with the source string.
    No sentinel side-effect (the caller already gated this path).
    """
    from aelfrice.scanner import (
        extract_ast,
        extract_filesystem,
        extract_git_log,
    )
    from aelfrice.noise_filter import NoiseConfig, is_noise

    root = Path(args.path)
    cfg = NoiseConfig.discover(root)
    candidates: list[Any] = []
    candidates.extend(extract_filesystem(root))
    candidates.extend(extract_git_log(root))
    candidates.extend(extract_ast(root))

    n_total = len(candidates)
    n_kept = 0
    print(f"# dry-run: extracting candidates from {root}", file=out)  # type: ignore[arg-type]
    for c in candidates:
        if is_noise(c.text, cfg):
            continue
        n_kept += 1
        # Rough token estimate: ~30 tokens per candidate (source + text)
        # — matches the spec § 6.1 budget.
        single_line = c.text.replace("\n", " ").replace("\r", " ")
        print(f"[{c.source}] {single_line}", file=out)  # type: ignore[arg-type]
    print(
        f"# dry-run: {n_kept} candidates after noise filter "
        f"(of {n_total} extracted). No network call. No sentinel written.",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _run_llm_onboard(
    args: argparse.Namespace,
    out: object,
    cfg: _LLMConfig,
) -> int:
    """Run scan_repo with an LLMScannerRouter installed.

    Maps router exceptions (auth, cap) to exit-1 codes per spec § 7.
    Telemetry is printed to stdout after the scan completes.
    """
    api_key = os.environ.get(_LLM_ENV_API_KEY, "")
    if not api_key:
        # Defensive: gates already checked this; double-check before
        # wiring the router.
        print(
            f"aelf: {_LLM_ENV_API_KEY} not set; cannot --llm-classify.",
            file=sys.stderr,
        )
        return 1
    router = _LLMScannerRouter(
        api_key=api_key,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
    )
    store = _open_store()
    try:
        result = scan_repo(store, Path(args.path), llm_router=router)
    except _LLMAuthError as exc:
        print(
            f"aelf: Anthropic auth failure ({exc}). No beliefs inserted. "
            f"Verify {_LLM_ENV_API_KEY}.",
            file=sys.stderr,
        )
        return 1
    except _LLMTokenCapExceeded as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
    print(
        _llm_format_telemetry_line(
            model=router.model, telemetry=router.telemetry,
        ),
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_search(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        hits = retrieve(store, args.query, token_budget=args.budget)
        # #655 read-only federation overlay: surface peer FTS5 hits
        # alongside local results. Peer hits are display-only and tagged
        # with their origin scope name; mutations on these ids are
        # rejected by the API surface (ForeignBeliefError). Empty when
        # no knowledge_deps.json or no reachable peers.
        peer_hits = store.search_peer_beliefs(args.query, limit=args.budget)
        if not hits and not peer_hits:
            n_beliefs = store.count_beliefs()
        else:
            n_beliefs = -1  # not consulted on the success path
    finally:
        store.close()
    if not hits and not peer_hits:
        if n_beliefs == 0:
            print(
                "no results — store is empty. Run `aelf onboard <path>` "
                "to populate the belief graph.",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"no results (store has {n_beliefs} belief(s); your query "
                f"is not in the indexed graph)",
                file=out,  # type: ignore[arg-type]
            )
        return 0
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        print(f"{prefix} {h.id}: {h.content}", file=out)  # type: ignore[arg-type]
    for belief, peer_name, _score in peer_hits:
        print(
            f"[scope:{peer_name}] {belief.id}: {belief.content}",
            file=out,  # type: ignore[arg-type]
        )
    return 0


def _seeds_with_scopes(
    store: MemoryStore,
    query: str,
    k: int,
) -> tuple[list, dict[str, str | None]]:
    """Return (seeds, scopes) for the reason walk's query path.

    Unions local search_beliefs + peer search_peer_beliefs results.
    Local seeds get scope=None; peer seeds get scope=<peer name>.

    Deduplication: first-seen wins. Local hits seed the dedup set, so a
    belief id present locally suppresses any peer copy (scope=None
    preserved). For ids that appear only on peers, the first peer in
    `search_peer_beliefs` iteration order wins; subsequent peer rows for
    the same id are dropped. This guarantees deterministic output even
    when multiple peers expose the same `scope='global'` belief id, or
    when a peer is wired more than once in the deps file.
    """
    local_hits = store.search_beliefs(query, limit=k)
    seen_ids: set[str] = {b.id for b in local_hits}

    peer_hits = store.search_peer_beliefs(query, limit=k)

    seeds: list = list(local_hits)
    scopes: dict[str, str | None] = {b.id: None for b in local_hits}

    for belief, peer_name, _score in peer_hits:
        if belief.id not in seen_ids:
            seeds.append(belief)
            scopes[belief.id] = peer_name
            seen_ids.add(belief.id)

    return seeds, scopes


def _cmd_reason(args: argparse.Namespace, out: object) -> int:
    """Surface a reasoning chain over the belief graph for a query.

    Seeds: explicit `--seed-id` (repeatable) wins; otherwise top-k
    `search_beliefs` BM25 hits over `args.query` unioned with peer
    FTS5 hits via `_seeds_with_scopes`. Walks `expand_bfs` from those
    seeds with terminal-tight defaults and prints either an indented
    hop tree (default) or JSON when `--json`.

    Read-only: never writes to the store.
    """
    store = _open_store()
    try:
        seeds: list = []
        seed_scopes: dict[str, str | None] = {}
        if args.seed_id:
            for sid in args.seed_id:
                b = store.get_belief(sid)
                if b is None:
                    # Fall through to peer stores before giving up (#713).
                    owner = store.find_foreign_owner(sid)
                    if owner is not None:
                        b = store.get_belief_in_scope(sid, owner)
                    if b is None:
                        print(
                            f"aelf reason: seed-id not found: {sid}",
                            file=out,  # type: ignore[arg-type]
                        )
                        return 2
                    seed_scopes[sid] = owner
                seeds.append(b)
        else:
            seeds, seed_scopes = _seeds_with_scopes(store, args.query, args.k)
        if not seeds:
            print(
                "aelf reason: no seeds (empty store, or query didn't "
                "match any indexed belief). Try --seed-id <id> to "
                "force a starting point.",
                file=out,  # type: ignore[arg-type]
            )
            return 0
        # Pass seed_scopes only when at least one seed has a non-None scope;
        # otherwise preserve byte-identical pre-#690 output for local-only
        # stores (seed_scopes=None, not an empty dict with all-None values).
        scopes_arg: dict[str, str | None] | None = (
            seed_scopes if any(v is not None for v in seed_scopes.values()) else None
        )
        hops = expand_bfs(
            seeds,
            store,
            max_depth=args.depth,
            nodes_per_hop=args.fanout,
            total_budget=args.budget,
            seed_scopes=scopes_arg,
        )
        paths = _reason_derive_paths(seeds, hops)
        verdict, impasses = _reason_classify(seeds, hops, store, paths=paths)
    finally:
        store.close()

    if args.json:
        import json
        dispatch_items = _reason_dispatch_policy(verdict, impasses)
        update_rows = _reason_suggested_updates(verdict, impasses, hops)
        payload = {
            "query": args.query,
            "seeds": [
                {
                    "id": b.id,
                    "content": b.content,
                    "owning_scope": seed_scopes.get(b.id),
                }
                for b in seeds
            ],
            "hops": [
                {
                    "id": h.belief.id,
                    "content": h.belief.content,
                    "score": h.score,
                    "depth": h.depth,
                    "path": h.path,
                    "owning_scope": h.owning_scope,
                }
                for h in hops
            ],
            "paths": [
                {
                    "belief_ids": list(p.belief_ids),
                    "edge_kinds": list(p.edge_kinds),
                    "compound_confidence": p.compound_confidence,
                    "weakest_link_belief_id": p.weakest_link_belief_id,
                    "fork_from": p.fork_from,
                }
                for p in paths
            ],
            "verdict": verdict.value,
            "impasses": [
                {
                    "kind": imp.kind.value,
                    "belief_ids": list(imp.belief_ids),
                    "note": imp.note,
                }
                for imp in impasses
            ],
            "dispatch": [
                {
                    "role": item.role.value,
                    "belief_ids": list(item.belief_ids),
                    "note": item.note,
                }
                for item in dispatch_items
            ],
            "suggested_updates": [
                {
                    "belief_id": row.belief_id,
                    "direction": row.direction,
                    "note": row.note,
                    "owning_scope": row.owning_scope,
                }
                for row in update_rows
            ],
        }
        print(json.dumps(payload, indent=2), file=out)  # type: ignore[arg-type]
        return 0

    print(f"query: {args.query}", file=out)  # type: ignore[arg-type]
    print("seeds:", file=out)  # type: ignore[arg-type]
    for b in seeds:
        scope = seed_scopes.get(b.id)
        scope_tag = f"[scope:{scope}] " if scope else ""
        print(
            f"  {scope_tag}{b.id}: {b.content}",
            file=out,  # type: ignore[arg-type]
        )
    if not hops:
        print("(no expansions — seeds have no outbound edges within budget)", file=out)  # type: ignore[arg-type]
        _emit_reason_footer(verdict, impasses, paths, out)
        return 0
    print("chain:", file=out)  # type: ignore[arg-type]
    for h in hops:
        indent = "  " * h.depth
        path_str = " -> ".join(h.path) if h.path else ""
        scope_tag = f"[scope:{h.owning_scope}] " if h.owning_scope else ""
        print(
            f"{indent}[{h.score:.3f}] {scope_tag}{h.belief.id}: {h.belief.content}",
            file=out,  # type: ignore[arg-type]
        )
        if path_str:
            print(f"{indent}  via {path_str}", file=out)  # type: ignore[arg-type]
    _emit_reason_footer(verdict, impasses, paths, out)
    return 0


def _emit_reason_footer(
    verdict: Verdict,
    impasses: list[Impasse],
    paths: list[ConsequencePath],
    out: object,
) -> None:
    """Print the verdict + impasses + forks block at the tail of `aelf reason`.

    Three sections: a ``verdict:`` line, the impasses block (either
    ``(none)`` or one indented row per impasse), and a ``forks:``
    summary that calls out CONTRADICTS-forked paths with their
    parent-path terminal id and the forked path's compound confidence.
    Format is grep-friendly so downstream tooling (e.g. R3 dispatch
    policy) can pick the verdict out of stdout.
    """
    print(f"verdict: {verdict.value}", file=out)  # type: ignore[arg-type]
    if not impasses:
        print("impasses: (none)", file=out)  # type: ignore[arg-type]
    else:
        print("impasses:", file=out)  # type: ignore[arg-type]
        for imp in impasses:
            ids_str = ",".join(imp.belief_ids)
            print(
                f"  {imp.kind.value} [{ids_str}]: {imp.note}",
                file=out,  # type: ignore[arg-type]
            )
    forks = [p for p in paths if p.fork_from is not None]
    if not forks:
        print("forks: (none)", file=out)  # type: ignore[arg-type]
        return
    print("forks:", file=out)  # type: ignore[arg-type]
    for p in forks:
        print(
            (
                f"  {p.fork_from} -> {p.belief_ids[-1]} "
                f"[compound={p.compound_confidence:.3f}, "
                f"weakest={p.weakest_link_belief_id}]"
            ),
            file=out,  # type: ignore[arg-type]
        )


# Edge-type → suggested-action heuristic for `aelf wonder`. The path
# the candidate was reached by hints at what relationship the user
# might want to materialize. This is a *suggestion* surface only; no
# edges are written by `aelf wonder` itself (#389 amendment 9 — store-
# write integration deferred to v2.x #229 lane).
_WONDER_ACTION_BY_EDGE: dict[str, str] = {
    EDGE_SUPERSEDES: "supersede",
    EDGE_CONTRADICTS: "contradict",
    EDGE_SUPPORTS: "merge",
}


def _suggested_action_for(path: list[str]) -> str:
    """Map a BFS edge-type path to a one-word suggested action.

    Picks the highest-priority edge type seen on the path, with
    fall-through to "relate" when none match. Priority order matches
    `_WONDER_ACTION_BY_EDGE` insertion order.
    """
    for edge_type in path:
        if edge_type in _WONDER_ACTION_BY_EDGE:
            return _WONDER_ACTION_BY_EDGE[edge_type]
    return "relate"


def _wonder_pick_seed(store: MemoryStore) -> object | None:
    """Deterministic seed picker: highest-degree non-locked belief.

    Ties broken by `belief.id` ascending. Returns None when the store
    has no non-locked beliefs (an empty store, or one where every
    belief is locked).

    Determinism is required so two runs against the same store
    produce the same wonder output. We count outbound edges only
    (matches BFS expansion direction).
    """
    best_id: str | None = None
    best_degree = -1
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None or b.lock_level == LOCK_USER:
            continue
        degree = len(store.edges_from(bid))
        if degree > best_degree or (
            degree == best_degree
            and best_id is not None
            and bid < best_id
        ):
            best_degree = degree
            best_id = bid
    if best_id is None:
        return None
    return store.get_belief(best_id)


_WONDER_AGENT_SHORTHAND_RE = re.compile(
    r"\b(?:(quick|deep)\s+)?(\d+)-agent(?:\s+wonder)?\b",
    re.IGNORECASE,
)


def _parse_wonder_query_shorthand(query: str) -> tuple[str, int | None]:
    """Strip agent-count shorthand from a wonder query (#645).

    Recognises ``N-agent`` with an optional ``quick``/``deep`` qualifier
    in front (and an optional trailing literal "wonder"). Returns the
    cleaned query and the parsed agent count, or ``(query, None)`` if
    no shorthand is present.

    Examples — input → (cleaned, count):
      "quick 2-agent wonder about X"   → ("about X",              2)
      "deep 6-agent wonder on Y"       → ("on Y",                 6)
      "3-agent gap on Z"               → ("gap on Z",             3)
      "no shorthand here"              → ("no shorthand here", None)

    The agent count is not clamped here; callers may apply their own
    bounds (e.g. ``wonder.max_agents`` from settings).
    """
    m = _WONDER_AGENT_SHORTHAND_RE.search(query)
    if not m:
        return query, None
    count = int(m.group(2))
    cleaned = (query[: m.start()] + query[m.end():]).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, count


def _cmd_export_canvas(args: argparse.Namespace, out: object) -> int:
    """Emit a JSON Canvas 1.0 document from seeds + BFS expansion (#763).

    Same seed-selection rules as ``_cmd_reason``: ``--seed-id``
    (repeatable) bypasses BM25 seed selection; otherwise top-k BM25
    hits via ``_seeds_with_scopes``. Expansion uses ``expand_bfs``
    with the user-supplied depth/budget/fanout. Result is serialised
    to ``--out`` (default stdout) as deterministic JSON.

    Read-only: never writes to the store.
    """
    from aelfrice.canvas_export import export_canvas  # local import
    import json as _json

    store = _open_store()
    try:
        seeds: list = []
        seed_scopes: dict[str, str | None] = {}
        if args.seed_id:
            for sid in args.seed_id:
                b = store.get_belief(sid)
                if b is None:
                    owner = store.find_foreign_owner(sid)
                    if owner is not None:
                        b = store.get_belief_in_scope(sid, owner)
                    if b is None:
                        print(
                            f"aelf export-canvas: seed-id not found: {sid}",
                            file=out,  # type: ignore[arg-type]
                        )
                        return 2
                    seed_scopes[sid] = owner
                seeds.append(b)
        else:
            seeds, seed_scopes = _seeds_with_scopes(store, args.query, args.k)
        if not seeds:
            print(
                "aelf export-canvas: no seeds (empty store, or query "
                "didn't match any indexed belief). Try --seed-id <id> "
                "to force a starting point.",
                file=out,  # type: ignore[arg-type]
            )
            return 0
        scopes_arg: dict[str, str | None] | None = (
            seed_scopes if any(v is not None for v in seed_scopes.values()) else None
        )
        hops = expand_bfs(
            seeds,
            store,
            max_depth=args.depth,
            nodes_per_hop=args.fanout,
            total_budget=args.budget,
            seed_scopes=scopes_arg,
        )
        payload = export_canvas(seeds, hops, store)
    finally:
        store.close()

    text = _json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.out and args.out != "-":
        from pathlib import Path as _Path
        _Path(args.out).write_text(text)
        print(
            f"aelf export-canvas: wrote {args.out} "
            f"({len(payload['nodes'])} nodes, {len(payload['edges'])} edges)",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(text, end="", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_wonder_axes(args: argparse.Namespace, out: object) -> int:
    """Emit dispatch-payload JSON for `aelf wonder QUERY` (#645) or
    the legacy alias `aelf wonder --axes QUERY` (#551).

    Mirrors the MCP `aelf_wonder` tool. Always JSON on stdout; exit 0
    on success.

    The query may carry agent-count shorthand ("quick 2-agent",
    "deep 6-agent", or a bare "N-agent"). When present and the caller
    did not pass an explicit ``--axes-agents``, the parsed count
    overrides the default. The shorthand is stripped from the query
    before dispatch so it does not pollute the gap-analysis query
    text.

    The computed values are also packed into a
    :class:`aelfrice.wonder.result.WonderResult` (#656); the
    dataclass is exposed via the ``_wonder_result`` attribute of
    the namespace when the caller sets ``_capture_result=True``
    (tests only; normal CLI path emits the dispatch-payload JSON).
    """
    import json as _json
    from aelfrice.wonder.dispatch import build_dispatch_payload
    from aelfrice.wonder.result import WonderResult, axes_coverage

    query, shorthand_count = _parse_wonder_query_shorthand(args.axes)
    # #645: shorthand wins only when the caller didn't override the
    # default explicitly. argparse's default for --axes-agents is 4;
    # when getattr(args, "axes_agents", None) != 4 we treat the value
    # as caller-supplied. This heuristic is good enough for the
    # ergonomic case; explicit overrides remain available.
    explicit_agents = getattr(args, "axes_agents", 4) != 4
    if shorthand_count is not None and not explicit_agents:
        agent_count = shorthand_count
    else:
        agent_count = getattr(args, "axes_agents", 4)

    store = _open_store()
    try:
        payload = build_dispatch_payload(
            store, query,
            budget=args.axes_budget,
            depth=args.axes_depth,
            agent_count=agent_count,
        )
    finally:
        store.close()

    ga = payload.gap_analysis
    axes_dicts = [a.to_dict() for a in payload.research_axes]
    phantoms_created: int = getattr(args, "_phantoms_created", 0)
    result = WonderResult(
        mode="axes",
        coverage=axes_coverage(phantoms_created, len(axes_dicts)),
        known_beliefs=[b.id for b in ga.known_beliefs],
        gaps=list(ga.gaps),
        research_axes=axes_dicts,
        anchor_speculative_ids=list(payload.speculative_anchor_ids),
        phantoms_created=phantoms_created,
        candidates=[],
    )
    if getattr(args, "_capture_result", False):
        args._wonder_result = result  # type: ignore[union-attr]

    print(_json.dumps(payload.to_dict(), indent=2), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_wonder_gc(args: argparse.Namespace, out: object) -> int:
    """Run wonder GC: soft-delete stale speculative beliefs (#549).

    Wraps :func:`aelfrice.wonder.lifecycle.wonder_gc`. Candidate beliefs
    must be ``type='speculative'``, still active (``valid_to IS NULL``),
    older than ``--ttl-days`` days, and have unchanged Bayesian priors +
    no feedback / RESOLVES edges.

    ``--dry-run`` reports candidates without mutating the store.
    Prints ``scanned=N deleted=N surviving=N`` on success.
    """
    from aelfrice.wonder.lifecycle import wonder_gc

    store = _open_store()
    try:
        result = wonder_gc(
            store,
            ttl_days=args.gc_ttl_days,
            dry_run=args.gc_dry_run,
        )
    finally:
        store.close()

    action = "would delete" if args.gc_dry_run else "deleted"
    print(
        f"wonder gc: scanned={result.scanned} {action}={result.deleted} "
        f"surviving={result.surviving}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_wonder_persist_docs(args: argparse.Namespace, out: object) -> int:
    """Ingest subagent research documents from a JSONL file (#552).

    Skill-layer integration entry point. After the host agent has
    collected one document per axis from a `/aelf:wonder --axes`
    dispatch, it writes them to a JSONL file with rows
    ``{axis_name, content, anchor_ids}`` and invokes this subcommand
    to convert + persist.

    Prints ``inserted=N skipped=N edges_created=N`` on success;
    mirrors the ``--persist`` summary so log-scraping is uniform.
    """
    from pathlib import Path

    from aelfrice.wonder.lifecycle import wonder_ingest
    from aelfrice.wonder.skill_integration import (
        documents_to_phantoms,
        load_documents_jsonl,
    )

    path = Path(args.persist_docs)
    if not path.exists():
        print(
            f"aelf wonder: --persist-docs file not found: {path}",
            file=out,  # type: ignore[arg-type]
        )
        return 2

    try:
        documents, anchor_ids = load_documents_jsonl(path)
    except ValueError as e:
        print(f"aelf wonder: {e}", file=out)  # type: ignore[arg-type]
        return 2

    if not documents:
        print(
            "aelf wonder: --persist-docs file contains no documents; nothing to ingest",
            file=out,  # type: ignore[arg-type]
        )
        return 0

    phantoms = documents_to_phantoms(documents, anchor_ids)

    store = _open_store()
    try:
        result = wonder_ingest(store, phantoms)
    finally:
        store.close()

    print(
        f"wonder persist-docs: inserted={result.inserted} "
        f"skipped={result.skipped} edges_created={result.edges_created}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_wonder(args: argparse.Namespace, out: object) -> int:
    """Surface consolidation candidates and (optionally) emit phantoms.

    Default behavior (#389 amendment 9): produces phantom-belief
    *candidates* in-memory and prints the top-N as ranked rows. When
    ``--persist`` is set, candidates are also written to the store via
    ``wonder_ingest`` and the insert summary is printed.

    ``--emit-phantoms`` prints candidates as Phantom JSON for offline
    review; mutually exclusive with ``--persist``.

    `aelf wonder QUERY` (#645) and `--axes QUERY` (#551, retained as an
    alias) bypass the seed/BFS path and emit the gap-analysis + research-
    axes JSON for research-agent dispatch. Mutually exclusive with
    ``--persist``. If both forms are passed, ``--axes`` wins.

    ``--persist-docs FILE`` (#552) reads subagent research documents
    from a JSONL file and persists them via ``wonder_ingest``. Used by
    the ``/aelf:wonder --axes`` skill-layer orchestration; mutually
    exclusive with ``--persist``, ``--axes``, and ``--emit-phantoms``.

    The graph-walk path packs its computed values into a
    :class:`aelfrice.wonder.result.WonderResult` (#656); ``--json``
    emits ``dataclasses.asdict(result)``.  Human-readable stdout is
    unchanged.
    """
    if getattr(args, "gc", False):
        return _cmd_wonder_gc(args, out)

    # #645: a positional QUERY routes to axes mode by default. The
    # legacy --axes flag is retained as an explicit alias and takes
    # precedence if both forms are passed.
    query = getattr(args, "query", None)
    if query is not None and getattr(args, "axes", None) is None:
        args.axes = query

    persist = getattr(args, "persist", False)
    persist_docs = getattr(args, "persist_docs", None)

    if persist_docs and (
        persist
        or getattr(args, "axes", None)
        or getattr(args, "emit_phantoms", False)
    ):
        print(
            "aelf wonder: --persist-docs cannot be combined with "
            "--persist / --axes / --emit-phantoms",
            file=out,  # type: ignore[arg-type]
        )
        return 2

    if persist_docs:
        return _cmd_wonder_persist_docs(args, out)

    if persist and getattr(args, "axes", None):
        print(
            "aelf wonder: --persist cannot be combined with --axes / QUERY",
            file=out,  # type: ignore[arg-type]
        )
        return 2

    if persist and getattr(args, "emit_phantoms", False):
        print(
            "aelf wonder: --persist cannot be combined with --emit-phantoms",
            file=out,  # type: ignore[arg-type]
        )
        return 2

    if getattr(args, "axes", None):
        return _cmd_wonder_axes(args, out)

    from aelfrice.wonder.result import WonderResult

    store = _open_store()
    try:
        seed_b: object | None
        if args.seed:
            seed_b = store.get_belief(args.seed)
            if seed_b is None:
                print(
                    f"aelf wonder: seed not found: {args.seed}",
                    file=out,  # type: ignore[arg-type]
                )
                return 2
        else:
            seed_b = _wonder_pick_seed(store)
        if seed_b is None:
            print(
                "aelf wonder: no eligible seeds (empty store or all "
                "beliefs are locked). Use --seed <id> to force one.",
                file=out,  # type: ignore[arg-type]
            )
            return 0
        hops = expand_bfs([seed_b], store, max_depth=2, total_budget=args.top * 2)
    finally:
        store.close()

    candidates: list[tuple] = []
    for h in hops:
        relatedness = wonder_consolidation.score(seed_b, h.belief)
        # Combine BFS path-score with token-overlap relatedness.
        # Multiplicative so both signals must be non-trivial for a
        # candidate to rank high.
        combined = h.score * (0.5 + 0.5 * relatedness)
        action = _suggested_action_for(h.path)
        candidates.append((combined, h, action, relatedness))
    candidates.sort(key=lambda r: (-r[0], r[1].belief.id))
    candidates = candidates[: args.top]

    phantoms = [
        Phantom(
            constituent_belief_ids=(seed_b.id, h.belief.id),  # type: ignore[union-attr]
            generator="bfs+wonder_consolidation",
            content=f"{seed_b.content} ⟷ {h.belief.content}",  # type: ignore[union-attr]
            score=combined,
        )
        for combined, h, _, _ in candidates
    ]

    if persist:
        from aelfrice.wonder.lifecycle import wonder_ingest
        ingest_store = _open_store()
        try:
            ingest_result = wonder_ingest(ingest_store, phantoms)
        finally:
            ingest_store.close()
        print(
            f"wonder persist: inserted={ingest_result.inserted} "
            f"skipped={ingest_result.skipped} "
            f"edges_created={ingest_result.edges_created}",
            file=out,  # type: ignore[arg-type]
        )
        return 0

    # Pack all computed values into the structured WonderResult (#656).
    candidate_rows = [
        {
            "candidate_id": h.belief.id,
            "score": combined,
            "relatedness": relatedness,
            "suggested_action": action,
            "path": list(h.path),
        }
        for combined, h, action, relatedness in candidates
    ]
    result = WonderResult(
        mode="graph_walk",
        coverage=0.0,
        known_beliefs=[seed_b.id],  # type: ignore[union-attr]
        gaps=[],
        research_axes=[],
        anchor_speculative_ids=[],
        phantoms_created=0,
        candidates=candidate_rows,
    )

    if args.emit_phantoms:
        import json as _json
        payload = [
            {
                "constituent_belief_ids": list(p.constituent_belief_ids),
                "generator": p.generator,
                "content": p.content,
                "score": p.score,
            }
            for p in phantoms
        ]
        print(_json.dumps(payload, indent=2), file=out)  # type: ignore[arg-type]
        return 0

    if args.json:
        import dataclasses as _dataclasses
        import json as _json
        print(
            _json.dumps(_dataclasses.asdict(result), indent=2),
            file=out,  # type: ignore[arg-type]
        )
        return 0

    print(f"seed: {result.known_beliefs[0]}: {seed_b.content}", file=out)  # type: ignore[arg-type,union-attr]
    if not candidates:
        print("(no candidates — seed has no outbound edges)", file=out)  # type: ignore[arg-type]
        return 0
    print(f"top {len(candidates)} consolidation candidate(s):", file=out)  # type: ignore[arg-type]
    for combined, h, action, relatedness in candidates:
        print(
            f"  [{combined:.3f}] ({action}) {h.belief.id}: {h.belief.content}",
            file=out,  # type: ignore[arg-type]
        )
    return 0


def _cmd_rebuild(args: argparse.Namespace, out: object) -> int:
    """Manual rebuild — same code path as the PreCompact hook.

    Reads recent turns from the canonical aelfrice transcript log if
    present, otherwise from a Claude Code internal transcript path
    given by --transcript. Prints the rebuild block to stdout. Useful
    for inspecting what the PreCompact hook would emit without
    triggering the actual hook.

    v1.4: now drives the same `rebuild_v14()` codepath the
    PreCompact hook uses (L0 + session-scoped + L2.5/L1 via
    `retrieve()`). Defaults follow `[rebuilder]` in `.aelfrice.toml`
    when present; CLI flags override.
    """
    from aelfrice.context_rebuilder import (
        find_aelfrice_log,
        load_rebuilder_config,
        read_recent_turns_aelfrice,
        read_recent_turns_claude_transcript,
        rebuild_v14,
    )

    config = load_rebuilder_config()
    n = args.n if args.n is not None else config.turn_window_n
    budget = (
        args.budget if args.budget is not None else config.token_budget
    )

    transcript_arg: str | None = args.transcript
    if transcript_arg:
        recent = read_recent_turns_claude_transcript(
            Path(transcript_arg), n=n
        )
    else:
        log_path = find_aelfrice_log(Path.cwd())
        if log_path is not None and log_path.exists():
            recent = read_recent_turns_aelfrice(log_path, n=n)
        else:
            recent = []

    store = _open_store()
    try:
        block = rebuild_v14(recent, store, token_budget=budget)
    finally:
        store.close()
    print(block, file=out, end="")  # type: ignore[arg-type]
    return 0


def _cmd_lock(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        now = _utc_now_iso()
        sid = resolve_session_id(
            getattr(args, "session_id", None),
            surface_name="aelf lock",
        )
        # #264 slice 2: route through the derivation worker. The worker
        # owns record_ingest -> derive -> insert_or_corroborate; the
        # entry point applies the re-lock semantic on an existing
        # lock-id belief (worker would otherwise just record a
        # corroboration without touching lock_level / locked_at).
        derived = derive(DerivationInput(
            raw_text=args.statement,
            source_kind=INGEST_SOURCE_CLI_REMEMBER,
            ts=now,
            session_id=sid,
        ))
        # cli_remember always produces a belief.
        assert derived.belief is not None
        lock_bid = derived.belief.id
        pre_existing_at_lock_id = store.get_belief(lock_bid) is not None
        ids_before: set[str] = set(store.list_belief_ids())
        log_id = store.record_ingest(
            source_kind=INGEST_SOURCE_CLI_REMEMBER,
            raw_text=args.statement,
            session_id=sid,
            ts=now,
            raw_meta={"call_site": CORROBORATION_SOURCE_CLI_REMEMBER},
        )
        run_worker(store)
        entry = store.get_ingest_log_entry(log_id)
        derived_ids = (
            entry.get("derived_belief_ids") if entry is not None else None
        )
        if not isinstance(derived_ids, list) or not derived_ids:
            # cli_remember always derives a belief; an empty list
            # indicates a downstream bug. Surface and bail.
            print("aelf lock: derivation produced no belief", file=out)  # type: ignore[arg-type]
            return 1
        actual_id = str(derived_ids[0])
        if pre_existing_at_lock_id and actual_id == lock_bid:
            # Re-lock of an existing lock-id belief: apply lock-upgrade.
            existing = store.get_belief(actual_id)
            if existing is not None:
                existing.lock_level = LOCK_USER
                existing.locked_at = now
                existing.demotion_pressure = 0
                existing.origin = ORIGIN_USER_STATED
                store.update_belief(existing)
            print(f"upgraded existing belief to lock: {actual_id}", file=out)  # type: ignore[arg-type]
        elif actual_id in ids_before:
            # content_hash collision with a different-source belief —
            # worker corroborated; preserve the prior surface message.
            print(f"locked: {actual_id} (corroborated existing)", file=out)  # type: ignore[arg-type]
        else:
            print(f"locked: {actual_id}", file=out)  # type: ignore[arg-type]

        # #435 doc-linker manual anchor. Idempotent on (belief_id,
        # doc_uri); subsequent lock --doc with the same URI is a no-op
        # write. The lock entry-point passes source_path=None to the
        # worker (cli_remember has no canonical document), so the
        # ingest-time hook does NOT fire — manual is the only path that
        # writes anchors here.
        doc_uri = getattr(args, "doc_uri", None)
        if doc_uri:
            link_belief_to_document(
                store,
                actual_id,
                doc_uri,
                anchor_type=ANCHOR_MANUAL,
            )

        # #550 Surface B: promote any speculative phantom whose
        # content_hash or normalized text (Jaccard ≥ 0.9) matches the
        # lock statement. Runs in the same DB transaction as the lock
        # write — promote() commits on each call, which is safe here
        # because each promote is idempotent.
        from aelfrice.promotion import (
            SOURCE_PROMOTE_PHANTOM_LOCK_MATCH,
            find_phantom_lock_matches,
            promote,
        )
        phantom_ids = find_phantom_lock_matches(store, args.statement)
        for phantom_id in phantom_ids:
            promote(
                store,
                phantom_id,
                source_label=SOURCE_PROMOTE_PHANTOM_LOCK_MATCH,
                now=now,
            )
            print(  # type: ignore[arg-type]
                f"promoted phantom: {phantom_id}",
                file=out,
            )
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


def _apply_scope_change(
    store: object,
    belief_id: str,
    to_scope: str,
    out: object,
) -> int:
    """Flip a belief's scope field and write a zero-valence audit row.

    Called by _cmd_promote and _cmd_demote when --to-scope is given.
    Validates the target scope, rejects foreign ids via
    assert_local_ownership, and writes a 'scope:<old>-><new>' audit row
    to feedback_history so the change is reconstructible.

    Returns 0 on success, 1 on error.  Does NOT touch origin or lock.
    """
    from datetime import datetime, timezone

    from aelfrice.federation import ForeignBeliefError
    from aelfrice.models import validate_belief_scope

    try:
        validate_belief_scope(to_scope)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    _store = store  # type: ignore[assignment]
    try:
        _store.assert_local_ownership(belief_id)  # type: ignore[union-attr]
    except ForeignBeliefError as e:
        print(f"scope change rejected: {e}", file=sys.stderr)
        return 1

    belief = _store.get_belief(belief_id)  # type: ignore[union-attr]
    if belief is None:
        print(f"belief not found: {belief_id}", file=sys.stderr)
        return 1

    old_scope = belief.scope
    if old_scope == to_scope:
        print(
            f"scope unchanged: {belief_id} (already {to_scope})",
            file=out,  # type: ignore[arg-type]
        )
        return 0

    belief.scope = to_scope
    _store.update_belief(belief)  # type: ignore[union-attr]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _store.insert_feedback_event(  # type: ignore[union-attr]
        belief_id=belief_id,
        valence=0.0,
        source=f"scope:{old_scope}->{to_scope}",
        created_at=ts,
    )
    print(
        f"scope updated: {belief_id} ({old_scope} -> {to_scope})",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_demote(args: argparse.Namespace, out: object) -> int:
    store = _open_store()
    try:
        # --to-scope: flip scope (orthogonal to tier demotion)
        to_scope: str | None = getattr(args, "to_scope", None)
        if to_scope is not None:
            rc = _apply_scope_change(store, args.belief_id, to_scope, out)
            return rc

        belief = store.get_belief(args.belief_id)
        if belief is None:
            print(f"belief not found: {args.belief_id}", file=sys.stderr)
            return 1
        # One tier per call: lock first, then user_validated.
        if belief.lock_level == LOCK_USER:
            from aelfrice.promotion import unlock
            unlock(store, args.belief_id)
            print(f"demoted: {args.belief_id}", file=out)  # type: ignore[arg-type]
            return 0
        if belief.origin == ORIGIN_USER_VALIDATED:
            from aelfrice.promotion import devalidate
            devalidate(store, args.belief_id)
            print(f"devalidated: {args.belief_id}", file=out)  # type: ignore[arg-type]
            return 0
        print(f"belief is not locked: {args.belief_id}", file=out)  # type: ignore[arg-type]
    finally:
        store.close()
    return 0


def _cmd_validate(args: argparse.Namespace, out: object) -> int:
    """Promote agent_inferred -> user_validated. v1.2.0."""
    from aelfrice.promotion import promote

    to_scope: str | None = getattr(args, "to_scope", None)

    store = _open_store()
    try:
        # --to-scope: orthogonal scope flip. Runs independently of origin flip.
        if to_scope is not None:
            rc = _apply_scope_change(store, args.belief_id, to_scope, out)
            if rc != 0:
                return rc

        # Origin flip: promote agent_inferred -> user_validated.
        # When --to-scope is the only action and the belief is already
        # user_validated, the idempotent already-validated path returns 0
        # without writing a redundant audit row — that is correct behaviour.
        try:
            result = promote(
                store, args.belief_id,
                source_label=f"promotion:{args.source}"
                if args.source != "user_validated"
                else "promotion:user_validated",
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        if result.already_validated:
            # Don't print "already validated" when --to-scope drove the call
            # and origin was already at the validated tier.
            if to_scope is None:
                print(f"already validated: {args.belief_id}", file=out)  # type: ignore[arg-type]
            return 0
        print(
            f"validated: {args.belief_id} "
            f"(origin: {result.prior_origin} -> {result.new_origin})",
            file=out,  # type: ignore[arg-type]
        )
    finally:
        store.close()
    return 0


def _cmd_unlock(args: argparse.Namespace, out: object) -> int:
    """Drop a user-lock without touching origin. Inverse of `aelf lock`."""
    from aelfrice.promotion import unlock

    store = _open_store()
    try:
        try:
            result = unlock(store, args.belief_id)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        if result.already_unlocked:
            print(f"already unlocked: {args.belief_id}", file=out)  # type: ignore[arg-type]
            return 0
        print(f"unlocked: {args.belief_id}", file=out)  # type: ignore[arg-type]
    finally:
        store.close()
    return 0


def _cmd_delete(args: argparse.Namespace, out: object) -> int:
    """Hard-delete one belief from the store. Writes an audit row first (#440)."""
    from aelfrice.federation import ForeignBeliefError
    from aelfrice.models import LOCK_USER

    store = _open_store()
    try:
        # #655 read-only federation: foreign ids must not be deleted
        # through the local DB. Check before the local `get_belief`
        # so the surfaced error names the owning scope rather than
        # the generic "not found".
        try:
            store.assert_local_ownership(args.belief_id)
        except ForeignBeliefError as e:
            print(f"delete error: {e}", file=sys.stderr)
            return 1
        belief = store.get_belief(args.belief_id)
        if belief is None:
            print(f"belief not found: {args.belief_id}", file=sys.stderr)
            return 1

        if belief.lock_level == LOCK_USER and not args.force:
            print(
                "belief is locked (lock_level=user); use --force to delete anyway",
                file=sys.stderr,
            )
            return 1

        if not args.yes:
            print(
                f"about to delete belief {args.belief_id}:",
                file=sys.stderr,
            )
            print(f'  "{belief.content}"', file=sys.stderr)
            try:
                answer = input(
                    "type the first 8 characters of the id to confirm: "
                )
            except EOFError:
                answer = ""
            if answer != args.belief_id[:8]:
                print("aborted: confirmation did not match", file=sys.stderr)
                return 1

        source = "user_deleted_force" if args.force else "user_deleted"
        store.insert_feedback_event(
            belief_id=args.belief_id,
            valence=-1.0,
            source=source,
            created_at=_utc_now_iso(),
        )
        store.delete_belief(args.belief_id)
        print(f"deleted: {args.belief_id}", file=out)  # type: ignore[arg-type]
    finally:
        store.close()
    return 0


def _cmd_confirm(args: argparse.Namespace, out: object) -> int:
    """Explicit user affirmation of a belief. Bumps Beta-Bernoulli alpha by 1.0."""
    from aelfrice.mcp_server import tool_confirm

    store = _open_store()
    try:
        result = tool_confirm(
            store,
            belief_id=args.belief_id,
            source=args.source,
            note=getattr(args, "note", "") or "",
        )
    finally:
        store.close()

    if result.get("kind") == "confirm.unknown_belief":
        print(f"confirm error: {result['error']}", file=sys.stderr)
        return 1

    prior_alpha: float = result["prior_alpha"]
    new_alpha: float = result["new_alpha"]
    new_beta: float = result["new_beta"]
    posterior_mean = new_alpha / (new_alpha + new_beta)
    msg = (
        f"confirmed {args.belief_id}: "
        f"alpha {prior_alpha:.3f}->{new_alpha:.3f}, "
        f"mean {posterior_mean:.3f}"
    )
    if result.get("note"):
        msg += f" [{result['note']}]"
    print(msg, file=out)  # type: ignore[arg-type]
    return 0


_CORE_MIN_CORROBORATION: int = 2
_CORE_MIN_POSTERIOR: float = 2.0 / 3.0
_CORE_MIN_ALPHA_BETA: int = 4


def _qualifies_core(b: object, args: argparse.Namespace) -> bool:
    """Return True if belief b meets any non-lock core signal."""
    alpha: float = b.alpha  # type: ignore[attr-defined]
    beta: float = b.beta  # type: ignore[attr-defined]
    corr: int = b.corroboration_count  # type: ignore[attr-defined]
    if corr >= args.min_corroboration:
        return True
    ab = alpha + beta
    if ab > 0 and ab >= args.min_alpha_beta and (alpha / ab) >= args.min_posterior:
        return True
    return False


def _emit_core(
    locked: list[object],
    candidates: list[object],
    args: argparse.Namespace,
    out: object,
) -> None:
    seen: set[str] = {b.id for b in locked}  # type: ignore[attr-defined]
    unlocked = [b for b in candidates if b.id not in seen]  # type: ignore[attr-defined]

    def _posterior(b: object) -> float:
        a: float = b.alpha  # type: ignore[attr-defined]
        bb: float = b.beta  # type: ignore[attr-defined]
        ab = a + bb
        return a / ab if ab > 0 else 0.0

    unlocked.sort(key=lambda b: (-_posterior(b), b.id))  # type: ignore[attr-defined]
    results = list(locked) + unlocked

    if args.limit is not None:
        results = results[: args.limit]

    if not results:
        print("no core beliefs", file=out)  # type: ignore[arg-type]
        return

    if args.json:
        rows = []
        for b in results:
            signals: list[str] = []
            if b.lock_level != LOCK_NONE:  # type: ignore[attr-defined]
                signals.append("lock")
            if b.corroboration_count >= args.min_corroboration:  # type: ignore[attr-defined]
                signals.append("corroboration")
            alpha: float = b.alpha  # type: ignore[attr-defined]
            beta: float = b.beta  # type: ignore[attr-defined]
            ab = alpha + beta
            if ab > 0 and ab >= args.min_alpha_beta and (alpha / ab) >= args.min_posterior:
                signals.append("posterior")
            rows.append({
                "id": b.id,  # type: ignore[attr-defined]
                "content": b.content,  # type: ignore[attr-defined]
                "lock_level": b.lock_level,  # type: ignore[attr-defined]
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": round(alpha / ab, 3) if ab else 0.0,
                "corroboration_count": b.corroboration_count,  # type: ignore[attr-defined]
                "signals": sorted(set(signals)),
            })
        print(json.dumps(rows, indent=2), file=out)  # type: ignore[arg-type]
        return

    for b in results:
        alpha = b.alpha  # type: ignore[attr-defined]
        beta = b.beta  # type: ignore[attr-defined]
        corr = b.corroboration_count  # type: ignore[attr-defined]
        ab = alpha + beta
        parts: list[str] = []
        if b.lock_level != LOCK_NONE:  # type: ignore[attr-defined]
            parts.append("LOCK")
        if corr >= args.min_corroboration:
            parts.append(f"CORR={corr}")
        if ab > 0 and ab >= args.min_alpha_beta and (alpha / ab) >= args.min_posterior:
            parts.append(f"α={alpha:.1f}")
            parts.append(f"β={beta:.1f}")
            parts.append(f"μ={alpha / ab:.3f}")
        tag = f" [{','.join(parts)}]" if parts else ""
        print(f"{b.id}{tag}: {b.content}", file=out)  # type: ignore[arg-type, attr-defined]


def _cmd_core(args: argparse.Namespace, out: object) -> int:
    """Surface load-bearing beliefs: locked ∪ corroborated ∪ high-posterior.

    Spec: docs/feature-aelf-core.md. No new store method — composition over
    list_locked_beliefs(), list_belief_ids(), and get_belief().
    """
    store = _open_store()
    try:
        locked: list[object] = [] if args.no_locked else store.list_locked_beliefs()
        candidates: list[object] = []
        if not args.locked_only:
            for bid in store.list_belief_ids():
                b = store.get_belief(bid)
                if b is None or b.lock_level != LOCK_NONE:
                    continue
                if _qualifies_core(b, args):
                    candidates.append(b)
    finally:
        store.close()
    _emit_core(locked, candidates, args, out)
    return 0


def _cmd_promote(args: argparse.Namespace, out: object) -> int:
    """Promote agent_inferred -> user_validated. Alias of `aelf validate`."""
    return _cmd_validate(args, out)


def _cmd_resolve(args: argparse.Namespace, out: object) -> int:
    """Resolve all unresolved CONTRADICTS threads via the v1.0.1 tie-breaker."""
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
    from aelfrice import __version__ as _aelf_version  # noqa: PLC0415

    store = _open_store()
    try:
        n_beliefs = store.count_beliefs()
        n_threads = store.count_edges()
        n_locked = store.count_locked()
        n_history = store.count_feedback_events()
    finally:
        store.close()
    print(f"version:           {_aelf_version}", file=out)  # type: ignore[arg-type]
    print(f"beliefs:           {n_beliefs}", file=out)  # type: ignore[arg-type]
    # v1.1.0 user-facing rename: "edges" -> "threads". Internal schema
    # keeps `edges`. MCP `aelf:stats` emits both keys for one minor.
    print(f"threads:           {n_threads}", file=out)  # type: ignore[arg-type]
    print(f"locked:            {n_locked}", file=out)  # type: ignore[arg-type]
    print(f"feedback events:   {n_history}", file=out)  # type: ignore[arg-type]
    print(_format_hrr_persist_status_line(), file=out)  # type: ignore[arg-type]
    return 0


def _format_hrr_persist_status_line() -> str:
    """Return the ``hrr.persist_state`` summary line for ``aelf status`` (#696)."""
    from aelfrice.hrr_index import (  # noqa: PLC0415
        HRRStructIndexCache,
        last_build_seconds as _last_build_seconds,
    )
    from aelfrice.store import MemoryStore as _MemoryStore  # noqa: PLC0415
    try:
        _store = _MemoryStore(":memory:")
        try:
            cache = HRRStructIndexCache(
                store=_store, dim=512, store_path=str(db_path()),
            )
            state = cache.resolve_persist_state()
        finally:
            _store.close()
    except Exception:  # noqa: BLE001
        return "hrr.persist_state: unknown (probe error)"
    enabled = bool(state.get("enabled", False))
    if enabled:
        on_disk = int(state.get("on_disk_bytes", 0))  # type: ignore[arg-type]
        lbs = _last_build_seconds()
        suffix = f", last build {lbs:.3f}s" if lbs is not None else ""
        return f"hrr.persist_state:  on {on_disk} bytes{suffix}"
    else:
        reason = state.get("reason") or "disabled"
        return f"hrr.persist_state:  off ({reason})"


# Academic-suite targets that are scaffolded but not yet runnable at v1.0.0.
# Each entry maps target name -> phase that activates it.
# See benchmarks/README.md for status detail.
_BENCH_INERT_TARGETS: Final[dict[str, str]] = {
    "mab": "v1.2.0 (P2 — triple extraction port)",
    "locomo": "v1.2.0 (P2 — ingest pipeline port)",
    "longmemeval": "v1.2.0 (P2 — ingest pipeline port)",
    "structmemeval": "v1.2.0 (P2 — ingest pipeline port)",
    "amabench": "v1.2.0 (P2 — ingest pipeline port)",
    # `all` is no longer inert — it dispatches via benchmarks.run.main_all().
}


def _cmd_gate(args: argparse.Namespace, out: object) -> int:
    """Aggregate open `gate:*` issues into a single screenful (#475).

    `aelf gate list` (default verb) prints sections for `gate:ratify`,
    `gate:prereq`, `gate:bench` (label `bench-gated`), `gate:license` —
    one line per issue with ask-count + age for ratify, plain `#N title`
    for the rest. `--json` emits the same structure as machine-readable
    JSON for scripting.

    Repo is auto-detected from cwd's git remote, the same model the rest
    of the gh-driven tooling uses (aelf-claim.sh, aelf-scan.sh).
    """
    from aelfrice import gate_list as _gl
    verb = (getattr(args, "gate_verb", None) or "list").lower()
    if verb != "list":
        print(
            f"aelf gate: unknown verb {verb!r}. Known: list.",
            file=out,  # type: ignore[arg-type]
        )
        return 2
    try:
        report = _gl.collect()
    except _gl.GhError as e:
        print(f"aelf gate list: {e}", file=out)  # type: ignore[arg-type]
        return 1
    if getattr(args, "gate_json", False):
        print(_gl.format_json(report), end="", file=out)  # type: ignore[arg-type]
    else:
        print(_gl.format_text(report), end="", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_bench(args: argparse.Namespace, out: object) -> int:
    """Run a benchmark target.

    Default (no target / 'synthetic'): the v0.9.0-rc synthetic harness —
    seed an in-memory store with a 16-belief corpus and score retrieval.
    Fully reproducible across runs (latency varies).

    `all`: dispatches to benchmarks.run.main_all() — the v2.0
    reproducibility harness ratified 2026-05-06 (#437). Subprocess each
    adapter at the canonical headline cut (full per the override),
    merge into one schema-v2 JSON. Requires --out PATH.

    Academic-suite single-name targets (mab, locomo, longmemeval,
    structmemeval, amabench) are still scaffolded but inert at this
    cut; invoke them directly with `python -m benchmarks.<name>_adapter`
    until they get a sub-subcommand wrapper.

    Stdlib-only utilities exposed today:
      verify-clean PATH        contamination gate over a retrieval JSON
      longmemeval-score PREDS GT  scoring without aelfrice imports
    """
    import json
    target = (args.target or "synthetic").lower()

    if target == "all":
        import sys as _sys
        from pathlib import Path as _Path
        if not getattr(args, "bench_out", None):
            print("aelf bench all: --out PATH is required.", file=out)  # type: ignore[arg-type]
            return 2
        # `benchmarks/` is the top-level academic-suite directory; per
        # pyproject.toml it is dev-only and not packaged in the wheel.
        # The harness is therefore reachable only from a source
        # checkout. Detect cwd, push onto sys.path, then import.
        cwd = _Path.cwd()
        if not (cwd / "benchmarks" / "run.py").exists():
            print(
                "aelf bench all: must be run from an aelfrice source "
                "checkout (cwd has no benchmarks/run.py).\n"
                "  git clone https://github.com/robotrocketscience/aelfrice && "
                "cd aelfrice && uv sync && aelf bench all --out PATH",
                file=out,  # type: ignore[arg-type]
            )
            return 2
        if str(cwd) not in _sys.path:
            _sys.path.insert(0, str(cwd))
        from benchmarks import run as _bench_run
        adapters_filter: tuple[str, ...] | None = None
        if getattr(args, "bench_adapters", None):
            adapters_filter = tuple(
                a.strip() for a in args.bench_adapters.split(",") if a.strip()
            )
        return _bench_run.main_all(
            out_path=_Path(args.bench_out),
            canonical=bool(getattr(args, "bench_canonical", False)),
            adapters=adapters_filter,
            smoke=bool(getattr(args, "bench_smoke", False)),
        )

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

    _DEV_TARGETS_MOVED: dict[str, str] = {
        "verify-clean": "python -m benchmarks.verify_clean",
        "longmemeval-score": "python -m benchmarks.longmemeval_score",
        "posterior-residual": "python -m benchmarks.posterior_ranking",
    }
    if target in _DEV_TARGETS_MOVED:
        print(
            f"aelf bench {target} has moved. Run "
            f"`{_DEV_TARGETS_MOVED[target]} ...` from a source checkout.",
            file=out,  # type: ignore[arg-type]
        )
        return 2

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
        f"Known targets: synthetic (default), "
        f"{', '.join(sorted(_BENCH_INERT_TARGETS))}.\n"
        f"Dev-only benchmarks moved to "
        f"`python -m benchmarks.<name>`: "
        f"{', '.join(sorted(_DEV_TARGETS_MOVED))}.",
        file=out,  # type: ignore[arg-type]
    )
    return 2


def _cmd_eval(args: argparse.Namespace, out: object) -> int:
    """Run the relevance-calibration harness (#365 R4 Phase B).

    Reuses ``aelfrice.eval_harness`` to score a synthetic corpus of
    ``(query, known_belief, noise_beliefs)`` fixtures and report
    P@K / ROC-AUC / Spearman ρ. Default corpus is the public synthetic
    fixture bundled with the wheel; ``--corpus PATH`` points at any
    JSONL with the same row schema.

    Determinism contract (#365 ship gate): same ``(corpus, --k, --seed)``
    -> bytes-identical output across reruns.

    Output formats:
      default   human-readable text block (same shape as
                ``audit_rebuild_log.py --calibrate-corpus``)
      --json    one JSON object per run with the report fields
                (machine-readable; intended for the R5 CI surface).

    Exit codes:
      0  report printed
      1  corpus missing or empty
      2  usage error (--k <= 0)
    """
    import json as _json

    from aelfrice import eval_harness  # noqa: PLC0415

    if args.eval_k <= 0:
        print("aelf eval: --k must be positive", file=out)  # type: ignore[arg-type]
        return 2

    corpus_path: Path = args.eval_corpus or eval_harness.DEFAULT_CALIBRATION_CORPUS
    if not corpus_path.is_file():
        print(
            f"aelf eval: calibration corpus not found: {corpus_path}",
            file=out,  # type: ignore[arg-type]
        )
        return 1

    fixtures = eval_harness.load_calibration_fixtures(corpus_path)
    if not fixtures:
        print(
            f"aelf eval: corpus is empty: {corpus_path}",
            file=out,  # type: ignore[arg-type]
        )
        return 1

    report = eval_harness.run_calibration_on_fixtures(
        fixtures, k=args.eval_k, seed=args.eval_seed,
    )

    if args.eval_json:
        payload = {
            "corpus": str(corpus_path),
            "seed": args.eval_seed,
            "k": report.k,
            "n_queries": report.n_queries,
            "n_truncated_queries": report.n_truncated_queries,
            "n_observations": report.n_observations,
            "p_at_k": report.p_at_k,
            "roc_auc": report.roc_auc,
            "spearman_rho": report.spearman_rho,
        }
        print(
            _json.dumps(payload, sort_keys=True, separators=(",", ":")),
            file=out,  # type: ignore[arg-type]
        )
    else:
        text = eval_harness.format_calibration_report(
            report, corpus_path=corpus_path, seed=args.eval_seed,
        )
        # ``format_calibration_report`` already terminates with "\n";
        # use ``end=""`` to avoid a double-newline trailing the block.
        print(text, file=out, end="")  # type: ignore[arg-type]
    return 0


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


def _cmd_project_warm(args: argparse.Namespace, out: object) -> int:
    """Hook entry-point: pre-load the project's SQLite + OS page cache.

    Called by the CwdChanged hook (HOME repo) on every cd. Silent
    no-op for unknown paths, denied paths, and the debounce window —
    we never block, never error, and never write to stdout. The hook
    fires fan-out across many cd events and any noise on stdout would
    leak into Claude Code's session log.
    """
    del out  # silent path
    debounce = args.debounce if args.debounce is not None else _PROJECT_WARM_DEBOUNCE
    try:
        _warm_path(Path(args.path), debounce_seconds=debounce)
    except Exception:  # noqa: BLE001
        # The hook fan-out semantics demand "never propagate". A
        # broken warm should never block the user's prompt.
        return 0
    return 0


def _cmd_setup(args: argparse.Namespace, out: object) -> int:
    # #733: auto-migrate non-uv aelfrice installs to uv tool on first
    # post-upgrade setup. Gated on a one-shot sentinel inside
    # maybe_migrate_to_uv so re-running setup for unrelated reasons
    # (config change, scope flip) does not re-trigger the migration.
    # Runs BEFORE hook reconciliation so `resolve_hook_command` resolves
    # the `aelf-hook` shim through the freshly-installed uv-tool path
    # rather than the now-orphaned pipx shim.
    migration = maybe_migrate_to_uv()
    if migration.attempted or migration.reason != "already migrated (sentinel exists)":
        # Surface a single stderr line in every case except the
        # post-migration steady state, so the operator knows why
        # `~/.local/bin/aelf` may have moved underneath them.
        if migration.reason != "already on uv tool":
            verb = "migrated" if migration.succeeded else "migration skipped"
            print(
                f"aelfrice: {verb} — {migration.reason}",
                file=sys.stderr,
            )
    scope = _effective_scope(args)
    path = _resolve_settings_path(args)
    command = args.command if args.command is not None else resolve_hook_command(scope)
    cleanup = clean_dangling_shims()
    for removed_path in cleanup.removed:
        print(
            f"cleaned dangling shim: {removed_path}",
            file=out,  # type: ignore[arg-type]
        )
    # #781: prune any `aelf-*` hook entries pointing at vanished venv
    # paths before reconciling installs. The dedupe-on-install path
    # (Layer A) only catches collisions on the basename we are about
    # to write; old stale entries for events we are NOT installing
    # this run still need to be swept out.
    prune = prune_broken_aelf_hooks(path)
    if prune.total_removed:
        events = ", ".join(
            f"{event}={count}"
            for event, count in sorted(prune.removed_per_event.items())
        )
        print(
            f"pruned {prune.total_removed} stale aelf-* hook entr"
            f"{'y' if prune.total_removed == 1 else 'ies'} "
            f"({events}) from {prune.settings_path}",
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
    if getattr(args, "transcript_ingest", True):
        ti_command = resolve_transcript_logger_command(scope)
        ti_result = install_transcript_ingest_hooks(
            path, command=ti_command, timeout=args.timeout,
        )
        if ti_result.installed:
            print(
                f"installed transcript-ingest hooks "
                f"({', '.join(ti_result.installed)}) in {ti_result.path} "
                f"(command={ti_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        if ti_result.already:
            print(
                f"transcript-ingest hooks already installed for "
                f"({', '.join(ti_result.already)}) in {ti_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "session_start", True):
        ss_command = resolve_session_start_hook_command(scope)
        ss_result = install_session_start_hook(
            path, command=ss_command, timeout=args.timeout,
            status_message=args.status_message,
        )
        if ss_result.already_present:
            print(
                f"SessionStart hook already installed in {ss_result.path} "
                f"(command={ss_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed SessionStart hook in {ss_result.path} "
                f"(command={ss_command!r})",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "stop_hook", True):
        st_command = resolve_stop_hook_command(scope)
        st_result = install_stop_hook(
            path, command=st_command, timeout=args.timeout,
            status_message=args.status_message,
        )
        if st_result.already_present:
            print(
                f"Stop hook already installed in {st_result.path} "
                f"(command={st_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed Stop hook in {st_result.path} "
                f"(command={st_command!r})",
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
    if getattr(args, "rebuilder", False):
        pc_command = resolve_pre_compact_hook_command(scope)
        pc_result = install_pre_compact_hook(
            path,
            command=pc_command,
            timeout=args.timeout,
            status_message=args.status_message,
        )
        if pc_result.already_present:
            print(
                f"PreCompact hook already installed in {pc_result.path} "
                f"(command={pc_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed PreCompact hook in {pc_result.path} "
                f"(command={pc_command!r})",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "commit_ingest", True):
        ci_command = resolve_commit_ingest_command(scope)
        ci_result = install_commit_ingest_hook(
            path, command=ci_command, timeout=args.timeout,
        )
        if ci_result.already_present:
            print(
                f"commit-ingest hook already installed in {ci_result.path} "
                f"(command={ci_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed commit-ingest PostToolUse hook in {ci_result.path} "
                f"(command={ci_command!r})",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "search_tool", True):
        st_command = resolve_search_tool_command(scope)
        st_result = install_search_tool_hook(
            path, command=st_command, timeout=args.timeout,
        )
        if st_result.already_present:
            print(
                f"search-tool hook already installed in {st_result.path} "
                f"(command={st_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed search-tool PreToolUse hook in {st_result.path} "
                f"(command={st_command!r})",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "search_tool_bash", True):
        stb_command = resolve_search_tool_bash_command(scope)
        stb_result = install_search_tool_bash_hook(
            path, command=stb_command, timeout=args.timeout,
        )
        if stb_result.already_present:
            print(
                f"search-tool-bash hook already installed in {stb_result.path} "
                f"(command={stb_command!r})",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"installed search-tool-bash PreToolUse:Bash hook in "
                f"{stb_result.path} (command={stb_command!r})",
                file=out,  # type: ignore[arg-type]
            )
    slash_dest = getattr(args, "slash_commands_dir", None)
    slash_dest_path = Path(slash_dest) if slash_dest else None
    sc_result = install_slash_commands(slash_dest_path)
    if sc_result.written:
        print(
            f"installed {len(sc_result.written)} slash command(s) in "
            f"{sc_result.dest_dir}: {', '.join(sc_result.written)}",
            file=out,  # type: ignore[arg-type]
        )
    if sc_result.pruned:
        print(
            f"removed {len(sc_result.pruned)} stale slash command(s) from "
            f"{sc_result.dest_dir}: {', '.join(sc_result.pruned)}",
            file=out,  # type: ignore[arg-type]
        )
    if not sc_result.written and not sc_result.pruned:
        print(
            f"slash commands already up to date in {sc_result.dest_dir}",
            file=out,  # type: ignore[arg-type]
        )
    _sync_setup_opt_outs_and_stamp(args)
    _print_setup_next_step(out)
    _print_setup_jsonl_history_hint(out)
    return 0


# Map argparse dest -> manifest hook `name`. A `False` value on the
# arg means the user passed --no-X, so the corresponding manifest
# hook is opted out (and stays opted out across upgrades, per #623).
_SETUP_FLAG_TO_HOOK_NAME: Final[dict[str, str]] = {
    "transcript_ingest": "transcript_ingest",
    "commit_ingest": "commit_ingest",
    "session_start": "session_start",
    "stop_hook": "stop_lock_prompt",
    "search_tool": "search_tool",
    "search_tool_bash": "search_tool_bash",
}


def _sync_setup_opt_outs_and_stamp(args: argparse.Namespace) -> None:
    """Reflect --no-X flags into the opt-out file; bump auto-install stamp.

    Called at the end of `_cmd_setup`. For each --no-* default-on flag
    the user passed, record the opt-out so a later bare `pipx upgrade
    aelfrice` + first `aelf <cmd>` does NOT silently re-add the hook.
    Conversely, when the user re-runs `aelf setup` without --no-X for
    a hook they previously disabled, the opt-out is rescinded.

    The version stamp is bumped to the installed version unconditionally
    here — `aelf setup` is the explicit-opt-in entry point, so after it
    runs successfully the next `aelf <cmd>` should fast-path with no
    further work.
    """
    for flag, hook_name in _SETUP_FLAG_TO_HOOK_NAME.items():
        wants = bool(getattr(args, flag, True))
        if wants:
            _auto_install.remove_opt_out(hook_name, _auto_install.OPT_OUT_PATH)
        else:
            _auto_install.add_opt_out(hook_name, _auto_install.OPT_OUT_PATH)
    try:
        _auto_install.write_stamp(_auto_install.STAMP_PATH, _AELFRICE_VERSION)
    except OSError:
        # Stamp write failure is non-fatal — auto-install will retry on
        # the next `aelf <cmd>` invocation.
        pass


def _print_setup_next_step(out: object) -> None:
    """One-line next-step hint when the active project's store is empty.

    Closes the loop on issue #116: after `aelf setup` succeeds in a
    fresh project the user otherwise gets no signal that the belief
    graph is empty and that hooks have nothing to retrieve. Skips the
    hint when the store already has beliefs, when opening the store
    fails, or when running in CI (anything that breaks should never
    derail setup itself).
    """
    try:
        store = _open_store()
    except Exception:  # pragma: no cover - defensive
        return
    try:
        n = store.count_beliefs()
    except Exception:  # pragma: no cover - defensive
        store.close()
        return
    finally:
        try:
            store.close()
        except Exception:  # pragma: no cover - defensive
            pass
    if n == 0:
        print(
            "next step: this project's belief store is empty. Run "
            "`aelf onboard .` to populate it from your repo.",
            file=out,  # type: ignore[arg-type]
        )


_CLAUDE_PROJECTS_DIR: Final[Path] = Path.home() / ".claude" / "projects"


def _print_setup_jsonl_history_hint(out: object) -> None:
    """One-line hint when historical Claude Code JSONLs exist on disk.

    Issue #115: a fresh aelf setup gives no signal that the user
    already has hundreds of session logs sitting unindexed at
    `~/.claude/projects/`. Surface the count + the batch-ingest
    invocation so they can choose to backfill.

    Quietly does nothing when the directory is missing or empty.
    Capped at counting up to 1000 files to keep setup fast.
    """
    if not _CLAUDE_PROJECTS_DIR.is_dir():
        return
    count = 0
    for _ in _CLAUDE_PROJECTS_DIR.glob("**/*.jsonl"):
        count += 1
        if count >= 1000:
            break
    if count == 0:
        return
    suffix = "+" if count >= 1000 else ""
    print(
        f"hint: {_CLAUDE_PROJECTS_DIR} has {count}{suffix} historical "
        f"session JSONL(s). To ingest them retroactively: "
        f"`aelf ingest-transcript --batch {_CLAUDE_PROJECTS_DIR}` "
        f"(see docs/INSTALL.md § Batch ingest of historical sessions "
        f"for the privacy trade-off).",
        file=out,  # type: ignore[arg-type]
    )


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
    if getattr(args, "session_start", True):
        ss_result = uninstall_session_start_hook(
            path, command_basename=SESSION_START_HOOK_SCRIPT_NAME,
        )
        if ss_result.removed == 0:
            print(
                f"no SessionStart hook in {ss_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {ss_result.removed} SessionStart entr"
                f"{'y' if ss_result.removed == 1 else 'ies'} from {ss_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "stop_hook", True):
        st_result = uninstall_stop_hook(
            path, command_basename=STOP_HOOK_SCRIPT_NAME,
        )
        if st_result.removed == 0:
            print(
                f"no Stop hook in {st_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {st_result.removed} Stop entr"
                f"{'y' if st_result.removed == 1 else 'ies'} from {st_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "transcript_ingest", True):
        ti_result = uninstall_transcript_ingest_hooks(
            path, command_basename=TRANSCRIPT_LOGGER_SCRIPT_NAME,
        )
        if not ti_result.removed:
            print(
                f"no transcript-ingest hooks in {ti_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            for event, n in ti_result.removed.items():
                print(
                    f"removed {n} {event} entr"
                    f"{'y' if n == 1 else 'ies'} from {ti_result.path}",
                    file=out,  # type: ignore[arg-type]
                )
    if getattr(args, "rebuilder", False):
        pc_result = uninstall_pre_compact_hook(
            path, command_basename="aelf-pre-compact-hook",
        )
        if pc_result.removed == 0:
            print(
                f"no rebuilder PreCompact hook in {pc_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {pc_result.removed} rebuilder PreCompact entr"
                f"{'y' if pc_result.removed == 1 else 'ies'} from {pc_result.path}",
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
    if getattr(args, "commit_ingest", True):
        ci_result = uninstall_commit_ingest_hook(
            path, command_basename=COMMIT_INGEST_SCRIPT_NAME,
        )
        if ci_result.removed == 0:
            print(
                f"no commit-ingest hook in {ci_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {ci_result.removed} commit-ingest entr"
                f"{'y' if ci_result.removed == 1 else 'ies'} from {ci_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "search_tool", True):
        st_result = uninstall_search_tool_hook(
            path, command_basename=SEARCH_TOOL_SCRIPT_NAME,
        )
        if st_result.removed == 0:
            print(
                f"no search-tool hook in {st_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {st_result.removed} search-tool entr"
                f"{'y' if st_result.removed == 1 else 'ies'} from {st_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    if getattr(args, "search_tool_bash", True):
        stb_result = uninstall_search_tool_bash_hook(
            path, command_basename=SEARCH_TOOL_BASH_SCRIPT_NAME,
        )
        if stb_result.removed == 0:
            print(
                f"no search-tool-bash hook in {stb_result.path}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"removed {stb_result.removed} search-tool-bash entr"
                f"{'y' if stb_result.removed == 1 else 'ies'} from {stb_result.path}",
                file=out,  # type: ignore[arg-type]
            )
    slash_dest = getattr(args, "slash_commands_dir", None)
    slash_dest_path = Path(slash_dest) if slash_dest else None
    usc_result = uninstall_slash_commands(slash_dest_path)
    if usc_result.pruned:
        print(
            f"removed {len(usc_result.pruned)} slash command(s) from "
            f"{usc_result.dest_dir}: {', '.join(usc_result.pruned)}",
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


def _cmd_mcp(args: argparse.Namespace, out: object) -> int:
    """Start the FastMCP stdio server exposing the aelfrice tool surface.

    Requires the `[mcp]` extra: `pip install 'aelfrice[mcp]'` (or
    `uv tool install --with fastmcp aelfrice`). Blocks until the host
    closes the stdio pipes; SIGINT exits cleanly with status 0.

    stdio MCP servers must never write to stdout — that channel carries
    the JSON-RPC protocol. The aelfrice tool handlers return dicts and
    never print; fastmcp itself respects the boundary.
    """
    _ = (args, out)
    try:
        from aelfrice.mcp_server import serve
    except ImportError as exc:  # pragma: no cover — defensive
        print(
            f"error: aelfrice.mcp_server import failed: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        serve()
    except RuntimeError as exc:
        # serve() raises RuntimeError when fastmcp is not installed —
        # the message includes the install hint.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Clean stop on Ctrl-C; hosts may signal shutdown via SIGINT
        # and a traceback would clutter their logs.
        return 0
    return 0


_UPGRADE_CONTEXT_NOTE: dict[str, str] = {
    "uv_tool": "installed via uv tool — use uv to upgrade",
    # #730: every non-uv install (pipx/venv/system) is routed to a
    # migration command. The note tells the user this isn't an in-place
    # upgrade — they're switching install channels.
    "non_uv": (
        "aelfrice is supported via `uv tool` only (#730); the `run:` "
        "line below migrates this install to uv"
    ),
}


_INSTALL_SITE_LABEL: dict[str, str] = {
    "uv_tool": "uv tool",
    "pipx": "pipx",
    "user_local_bin": "user-local",
}


def _format_multi_install_warning(
    sites: list, active_context: str
) -> list[str]:
    """Format a multi-install warning block, or return [] if not warranted.

    `sites` is the output of `lifecycle.detect_reachable_installs()`.
    Warning fires only when more than one distinct site is detected.
    """
    if len(sites) < 2:
        return []
    lines = ["warning: multiple aelfrice installs detected:"]
    for site in sites:
        marker = " (on PATH)" if site.on_path else ""
        label = _INSTALL_SITE_LABEL.get(site.kind, site.kind)
        lines.append(f"  - {label}: {site.path}{marker}")
    lines.append(
        f"upgrading the active install ({active_context}) will not change "
        "`which aelf` if a different install is on PATH."
    )
    lines.append("remove the stale install before upgrading.")
    return lines


def _cmd_upgrade_deprecated_alias(
    args: argparse.Namespace, out: object
) -> int:
    """Deprecated `aelf upgrade` alias — emit warning then delegate.

    The subcommand was renamed to `aelf upgrade-cmd` at #427 to read
    advisory rather than imperative. The bare `upgrade` form keeps
    working for one minor so existing scripts don't break; this
    handler prepends a one-line stderr deprecation notice before
    invoking the canonical handler.
    """
    print(
        "warning: `aelf upgrade` was renamed to `aelf upgrade-cmd` (#427); "
        "the bare alias will be removed in a future release.",
        file=sys.stderr,
    )
    return _cmd_upgrade(args, out)


def _cmd_upgrade(args: argparse.Namespace, out: object) -> int:
    """Print the right upgrade command for this install context.

    Does NOT shell out to pip itself: replacing the running package
    mid-process is unreliable on Windows and can leave the user with a
    broken interpreter. We tell the user the exact line; they run it.

    --check: previously suppressed the install-context note and the
    `run:` line. As of #522 both forms emit identical output when an
    update is available — the `/aelf:upgrade` slash file (#513)
    parses the `run:` line, so the two forms must agree. Both the
    no-flags form and `--check` print the version banner, the verify
    block, the install-context note, and `run: <command>`. The exit
    code is unchanged: 0 either way.
    """
    advice = upgrade_advice()
    multi_warning = _format_multi_install_warning(
        detect_reachable_installs(), advice.context
    )
    for line in multi_warning:
        print(line, file=out)  # type: ignore[arg-type]
    # Force a fresh sync check unless explicitly disabled. This is the
    # one CLI surface where the user has explicitly asked about updates,
    # so we ignore the cache TTL.
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
        # #522: emit the install-context note + `run:` line under both
        # forms so the /aelf:upgrade slash file (#513) can parse `run:`
        # whichever form it called. The pre-#522 `--check` short-circuit
        # broke the slash on pre-2.0.1 CLIs that had the new bundle.
        note = _UPGRADE_CONTEXT_NOTE.get(advice.context, "")
        if note:
            print(
                f"({note})",
                file=out,  # type: ignore[arg-type]
            )
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
    """Run the v1.1.0 structural auditor and print findings + metrics.

    Exits 1 if any audit check fails (orphan threads, FTS5 sync, locked
    contradictions). Informational metrics never affect exit status.
    The v1.0 regime classifier moved to `aelf regime`.

    The corpus-volume check (added in #116) warns when an established
    project has too few beliefs but is informational — it does not
    affect the exit code.

    --json: emit a single JSON object with keys "audit" and "features".
    "features.edges_by_type" carries the per-edge-type count dict.
    Exit code unchanged: 1 on auditor failure, 0 otherwise.
    """
    use_json = getattr(args, "json", False)
    corpus_min = _resolve_corpus_min()
    project_age = _git_first_commit_age_days()
    # #655 surface knowledge_deps.json parse failures as a finding-style
    # warning rather than letting them crash health. Done outside the
    # store-open block so a malformed peer config doesn't block the
    # rest of the audit.
    peer_config_error: str | None = None
    try:
        from aelfrice.federation import load_peer_deps as _load_peer_deps
        _load_peer_deps()
    except ValueError as e:
        peer_config_error = str(e)
    store = _open_store()
    try:
        report = audit(
            store,
            corpus_min=corpus_min,
            project_age_days=project_age,
        )
        features = compute_features(store)
        peer_snapshot = store.peer_health()
        scope_counts = store.count_beliefs_by_scope()
    finally:
        store.close()

    if use_json:
        # Serialise auditor findings + metrics as plain dicts; features via
        # edges_by_type only (regime fields are not part of this contract).
        findings_list = [
            {
                "check": f.check,
                "severity": f.severity,
                "count": f.count,
                "detail": f.detail,
            }
            for f in report.findings
        ]
        payload = {
            "audit": {
                "findings": findings_list,
                "metrics": dict(report.metrics),
                "failed": report.failed,
            },
            "features": {
                "edges_by_type": features.edges_by_type,
            },
            "federation": {
                "peers": peer_snapshot,
                "config_error": peer_config_error,
                "beliefs_by_scope": scope_counts,
            },
        }
        print(json.dumps(payload), file=out)  # type: ignore[arg-type]
        return 1 if report.failed else 0

    print("audit:", file=out)  # type: ignore[arg-type]
    for f in report.findings:
        if f.severity == AUDIT_SEVERITY_FAIL:
            marker = "FAIL"
        elif f.severity == AUDIT_SEVERITY_WARN:
            marker = "warn"
        else:
            marker = "ok"
        print(
            f"  [{marker:4s}] {f.check:24s} {f.detail}",
            file=out,  # type: ignore[arg-type]
        )
    print("", file=out)  # type: ignore[arg-type]
    print("metrics:", file=out)  # type: ignore[arg-type]
    for key, value in report.metrics.items():
        if isinstance(value, float):
            display = f"{value:.3f}"
        else:
            display = str(value)
        print(f"  {key:24s} {display}", file=out)  # type: ignore[arg-type]
    print("", file=out)  # type: ignore[arg-type]
    # Per-edge-type breakdown (sorted by count desc, then alphabetically).
    print("edges by type:", file=out)  # type: ignore[arg-type]
    if not any(features.edges_by_type.values()):
        print("  no edges yet", file=out)  # type: ignore[arg-type]
    else:
        sorted_entries = sorted(
            features.edges_by_type.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for edge_type, count in sorted_entries:
            print(f"  {edge_type:24s} {count}", file=out)  # type: ignore[arg-type]
    print("", file=out)  # type: ignore[arg-type]
    sentiment_state, sentiment_count = _sentiment_from_prose_state()
    if sentiment_state == "enabled":
        print(
            f"sentiment-from-prose feedback: enabled ({sentiment_count} matches)",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            "sentiment-from-prose feedback: disabled",
            file=out,  # type: ignore[arg-type]
        )
    print("", file=out)  # type: ignore[arg-type]
    print("federation peers:", file=out)  # type: ignore[arg-type]
    if peer_config_error is not None:
        print(
            f"  [warn] knowledge_deps.json: {peer_config_error}",
            file=out,  # type: ignore[arg-type]
        )
    if not peer_snapshot:
        print("  (none configured)", file=out)  # type: ignore[arg-type]
    else:
        for entry in peer_snapshot:
            status = "ok" if entry["reachable"] else "missing"
            scope_marker = (
                f" scope_id={entry['scope_id'][:8]}…"
                if entry["scope_id"]
                else ""
            )
            print(
                f"  [{status:7s}] {entry['name']:16s} {entry['path']}{scope_marker}",
                file=out,  # type: ignore[arg-type]
            )
    print("", file=out)  # type: ignore[arg-type]
    print("run `aelf regime` for the v1.0 regime classifier.",
          file=out)  # type: ignore[arg-type]
    return 1 if report.failed else 0


def _sentiment_from_prose_state() -> tuple[str, int]:
    """Resolve enabled/disabled status for #193 sentiment-from-prose
    feedback plus a count of `feedback_history` rows attributed to it.

    Failures (missing config, missing DB, decode errors) degrade to
    ("disabled", 0) without raising — keeps `aelf health` informational
    and never causes it to exit non-zero on a config issue.
    """
    from aelfrice.sentiment_feedback import (
        SENTIMENT_INFERRED_SOURCE,
        is_enabled,
    )
    config_dict = _load_aelfrice_config_dict(Path.cwd())
    enabled = is_enabled(config_dict)
    if not enabled:
        return ("disabled", 0)
    try:
        store = _open_store()
    except Exception:  # noqa: BLE001
        return ("enabled", 0)
    try:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM feedback_history WHERE source = ?",
            (SENTIMENT_INFERRED_SOURCE,),
        ).fetchone()
        count = int(row[0]) if row is not None else 0
    except Exception:  # noqa: BLE001
        count = 0
    finally:
        store.close()
    return ("enabled", count)


def _load_aelfrice_config_dict(root: Path) -> dict[str, Any] | None:
    """Walk up from `root` for `.aelfrice.toml`; return the parsed dict
    or None on miss / parse failure. Mirrors `_load_llm_config` walk
    semantics but returns the whole file rather than one section.
    """
    import tomllib

    current = root.resolve() if root.exists() else root
    seen: set[Path] = set()
    candidate = current if current.is_dir() else current.parent
    while candidate not in seen:
        seen.add(candidate)
        cfg_path = candidate / ".aelfrice.toml"
        if cfg_path.is_file():
            try:
                raw = cfg_path.read_bytes()
                parsed: Any = tomllib.loads(
                    raw.decode("utf-8", errors="replace")
                )
            except (OSError, tomllib.TOMLDecodeError):
                return None
            return parsed if isinstance(parsed, dict) else None
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent
    return None


def _cmd_migrate(args: argparse.Namespace, out: object) -> int:
    """Copy beliefs from the legacy global DB into the active project's DB.

    Dry-run by default; `--apply` writes. `--all` skips the project-mention
    filter. `--from` overrides the source path. Reads source via SQLite
    `mode=ro` URI so accidental writes are rejected at the DB layer.
    Exits 0 on success or no-op, 1 when source DB is missing or refuses
    to open.
    """
    legacy = (
        Path(args.from_path) if args.from_path
        else default_legacy_db_path()
    )
    target = db_path()
    project_root = Path.cwd()

    if not legacy.exists():
        print(
            f"legacy DB not found at {legacy} — nothing to migrate",
            file=sys.stderr,
        )
        return 1
    if legacy.resolve() == target.resolve():
        print(
            f"legacy and target are the same DB ({legacy}) — refusing to migrate",
            file=sys.stderr,
        )
        return 1

    try:
        report = _migrate_action(
            legacy_path=legacy,
            target_path=target,
            project_root=project_root,
            apply=args.apply,
            copy_all=args.copy_all,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"migrate failed: {exc}", file=sys.stderr)
        return 1

    _print_migrate_report(report, out=out)
    return 0


def _print_migrate_report(report: MigrateReport, out: object) -> None:
    c = report.counts
    mode = "applied" if report.applied else "dry-run"
    print(f"migrate ({mode}):", file=out)  # type: ignore[arg-type]
    print(f"  source:  {report.legacy_path}", file=out)  # type: ignore[arg-type]
    print(f"  target:  {report.target_path}", file=out)  # type: ignore[arg-type]
    print(f"  legacy beliefs:           {c.legacy_beliefs}", file=out)  # type: ignore[arg-type]
    print(f"  legacy edges:             {c.legacy_edges}", file=out)  # type: ignore[arg-type]
    print(f"  matched (project-filter): {c.matched_beliefs}", file=out)  # type: ignore[arg-type]
    if report.applied:
        verb = "inserted"
    else:
        verb = "would insert"
    print(f"  {verb} beliefs:        {c.inserted_beliefs}", file=out)  # type: ignore[arg-type]
    print(f"  skipped existing:         {c.skipped_existing_beliefs}", file=out)  # type: ignore[arg-type]
    print(f"  {verb} edges:           {c.inserted_edges}", file=out)  # type: ignore[arg-type]
    print(f"  skipped orphan edges:     {c.skipped_orphan_edges}", file=out)  # type: ignore[arg-type]
    if not report.applied:
        print("", file=out)  # type: ignore[arg-type]
        print(
            "re-run with --apply to actually copy "
            "(add --all to skip the project-mention filter).",
            file=out,  # type: ignore[arg-type]
        )


def _cmd_ingest_transcript(args: argparse.Namespace, out: object) -> int:
    """Ingest one or many JSONL turn-logs into the active project's DB.

    Three call shapes:
      - `aelf ingest-transcript PATH` — single-file ingest. Used by
        the transcript-logger PreCompact hook (detached spawn on
        rotation) and by manual recovery / replay.
      - `aelf ingest-transcript --batch DIR` — recurse into DIR for
        every `*.jsonl` and ingest each. Handles the v1.2.0
        transcript-logger format AND the Claude Code internal
        session-log format under `~/.claude/projects/` (issue #115).
      - `aelf ingest-transcript --batch DIR --since YYYY-MM-DD` —
        same, but skip files whose mtime is older than the cutoff.

    Returns 0 on success or empty input. Returns 1 only when the
    explicit single-file `path` does not exist or `--batch` resolves
    to a non-existent directory.
    """
    from aelfrice.ingest import ingest_jsonl, ingest_jsonl_dir

    if args.batch:
        return _cmd_ingest_transcript_batch(args, out, ingest_jsonl_dir)
    if not args.path:
        print(
            "ingest-transcript: provide PATH or --batch DIR",
            file=sys.stderr,
        )
        return 1
    path = Path(args.path)
    if not path.is_file():
        print(f"ingest-transcript: {path} not found", file=sys.stderr)
        return 1
    store = _open_store()
    try:
        result = ingest_jsonl(store, path, source_label=args.source_label)
    finally:
        store.close()
    print(
        f"ingest-transcript: {path.name} "
        f"lines={result.lines_read} "
        f"turns={result.turns_ingested} "
        f"beliefs={result.beliefs_inserted} "
        f"edges={result.edges_inserted} "
        f"skipped={result.skipped_lines}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_ingest_transcript_batch(
    args: argparse.Namespace, out: object, ingest_dir,
) -> int:
    """Helper: handle the `--batch DIR [--since DATE]` path.

    Split out so the single-file path stays simple. The `since` parse
    is intentionally lenient — accepts `YYYY-MM-DD` and full ISO
    timestamps; anything else is rejected up-front.
    """
    directory = Path(args.batch)
    if not directory.is_dir():
        print(
            f"ingest-transcript: --batch directory {directory} not found",
            file=sys.stderr,
        )
        return 1
    cutoff: datetime | None = None
    if args.since:
        try:
            cutoff = datetime.fromisoformat(args.since)
        except ValueError:
            print(
                f"ingest-transcript: --since must be ISO date or "
                f"timestamp; got {args.since!r}",
                file=sys.stderr,
            )
            return 1
    store = _open_store()
    try:
        result = ingest_dir(
            store, directory,
            since=cutoff, source_label=args.source_label,
        )
    finally:
        store.close()
    print(
        f"ingest-transcript --batch {directory}: "
        f"files_walked={result.files_walked} "
        f"files_ingested={result.files_ingested} "
        f"files_skipped_age={result.files_skipped_age} "
        f"lines={result.lines_read} "
        f"turns={result.turns_ingested} "
        f"beliefs={result.beliefs_inserted} "
        f"edges={result.edges_inserted} "
        f"skipped={result.skipped_lines}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_regime(args: argparse.Namespace, out: object) -> int:
    """Print the v1.0 regime classifier output (supersede / ignore / mixed).

    Preserved as a separate command in v1.1.0 after `aelf health` was
    rewritten as the structural auditor. The classifier is informational
    — it never affects exit status.
    """
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


def _print_replay_report(
    report: object,
    out: object,
) -> None:
    """Print a human-readable summary of a FullEqualityReport.

    Format (stable — tests pin specific lines):

        replay full-equality report
        ---------------------------
        log rows considered:       <n>
        excluded (legacy_unknown): <n>
        matched:                   <n>
        mismatched:                <n>
        derived_orphan:            <n>
        canonical_orphan:          <n>
        legacy_origin_backfill:    <n>
        feedback_derived_edges:    <n>   (informational)

    If drift > 0, a "drift examples" section follows with one sub-block
    per non-empty bucket.
    """
    from aelfrice.replay import FullEqualityReport
    assert isinstance(report, FullEqualityReport)
    lines = [
        "replay full-equality report",
        "---------------------------",
        f"log rows considered:       {report.total_log_rows}",
        f"excluded (legacy_unknown): {report.excluded_legacy_unknown}",
        f"matched:                   {report.matched}",
        f"mismatched:                {report.mismatched}",
        f"derived_orphan:            {report.derived_orphan}",
        f"canonical_orphan:          {report.canonical_orphan}",
        f"legacy_origin_backfill:    {report.legacy_origin_backfill}",
        f"feedback_derived_edges:    {report.feedback_derived_edges}"
        "   (informational)",
    ]
    for line in lines:
        print(line, file=out)  # type: ignore[arg-type]

    if report.has_drift:
        print("", file=out)  # type: ignore[arg-type]
        print("drift examples:", file=out)  # type: ignore[arg-type]
        for bucket, examples in report.drift_examples.items():
            if not examples:
                continue
            print(f"  [{bucket}]", file=out)  # type: ignore[arg-type]
            for ex in examples:
                print(f"    {ex}", file=out)  # type: ignore[arg-type]


def _cmd_doctor(args: argparse.Namespace, out: object) -> int:
    """Diagnose hooks + brain-graph health.

    `args.scope`:
      - `hooks`  → settings.json hook validation only.
      - `graph`  → structural audit only (orphan threads, FTS5 sync,
                   locked contradictions).
      - None     → run both (default).

    `--classify-orphans` routes to the targeted reclassification pass
    (issue #206); it bypasses the hooks/graph checks entirely and exits
    after printing the cost/distribution report.

    `--replay` routes to the v2.x full-equality probe (#262); it
    bypasses the hooks/graph checks and exits after printing the report.
    Exit 0 if ``mismatched + derived_orphan == 0`` (or <= ``--max-drift``
    when that flag is set); exit 1 otherwise.

    Exit 1 if any structural failure fires in either subcheck;
    informational warnings never trip exit. The v1.2 deprecated alias
    `aelf health` routes here with scope='graph' implicitly via
    `_cmd_health`.
    """
    if getattr(args, "classify_orphans", False):
        return _cmd_doctor_classify_orphans(args, out)
    if getattr(args, "gc_orphan_feedback", False):
        return _cmd_doctor_gc_orphan_feedback(args, out)
    if getattr(args, "promote_retention", False):
        return _cmd_doctor_promote_retention(args, out)
    if getattr(args, "replay", False):
        return _cmd_doctor_replay(args, out)
    if getattr(args, "dedup", False):
        return _cmd_doctor_dedup(args, out)
    if getattr(args, "relationships", False):
        return _cmd_doctor_relationships(args, out)
    if getattr(args, "detect_stale", False):
        return _cmd_doctor_detect_stale(args, out)
    if getattr(args, "derive_pending", False):
        return _cmd_doctor_derive_pending(args, out)
    if getattr(args, "prune_dormant", False):
        return _cmd_doctor_prune_dormant(args, out)
    if getattr(args, "meta_beliefs", False):
        return _cmd_doctor_meta_beliefs(args, out)
    scope = getattr(args, "scope", None)
    exit_code = 0
    if scope in (None, "hooks"):
        project_root = Path(args.project_root) if args.project_root else None
        user_settings = (
            Path(args.user_settings) if args.user_settings else None
        )
        hook_failures_log = (
            Path(args.hook_failures_log) if args.hook_failures_log else None
        )
        known_subs = _known_cli_subcommands()
        report = diagnose(
            user_settings=user_settings,
            project_root=project_root,
            hook_failures_log=hook_failures_log,
            known_cli_subcommands=known_subs,
            hrr_store_path=str(db_path()),
        )
        print(format_report(report), file=out)  # type: ignore[arg-type]
        _print_doctor_store_check(out)
        if report.broken:
            exit_code = 1
        if getattr(args, "fix", False):
            _cmd_doctor_fix_hooks(args, report, out)
    if scope in (None, "graph"):
        if scope is None:
            print("", file=out)  # type: ignore[arg-type]
        graph_exit = _cmd_health(args, out)
        if graph_exit != 0:
            exit_code = graph_exit
    return exit_code


def _cmd_doctor_fix_hooks(
    args: argparse.Namespace, report: object, out: object,
) -> None:
    """Apply `prune_broken_aelf_hooks` to every settings.json doctor scanned.

    Called from `_cmd_doctor` when `--fix` is set. Reads the list of
    scopes/paths off the already-built `DoctorReport` so the prune
    decision matches what doctor just reported. `args.dry_run` flips
    the prune into report-only mode.

    Custom hooks (non-`aelf-` basename) are never touched by the
    prune helper itself — this wrapper only handles I/O.
    """
    from typing import cast

    dry_run = bool(getattr(args, "dry_run", False))
    scopes_scanned = cast(
        list[tuple[str, Path]],
        getattr(report, "scopes_scanned", []),
    )
    if not scopes_scanned:
        return
    print("", file=out)  # type: ignore[arg-type]
    any_removed = False
    for _scope, path in scopes_scanned:
        prune = prune_broken_aelf_hooks(path, dry_run=dry_run)
        if not prune.total_removed:
            continue
        any_removed = True
        events = ", ".join(
            f"{event}={count}"
            for event, count in sorted(prune.removed_per_event.items())
        )
        verb = "would prune" if dry_run else "pruned"
        noun = "entry" if prune.total_removed == 1 else "entries"
        print(
            f"--fix: {verb} {prune.total_removed} stale aelf-* hook "
            f"{noun} ({events}) from {path}",
            file=out,  # type: ignore[arg-type]
        )
    if not any_removed:
        print(
            "--fix: no stale aelf-* hook entries to prune",
            file=out,  # type: ignore[arg-type]
        )


def _cmd_doctor_meta_beliefs(args: argparse.Namespace, out: object) -> int:
    """Surface installed meta-belief state (#755 substrate diagnostic).

    Lists every row in ``meta_beliefs`` with its config + decayed
    surfaced value + per-signal-class sub-posterior breakdown. Text
    mode emits a human-readable table; ``--json`` emits a stable
    object so downstream tooling (eval-harness gates, dashboards) can
    parse without screen-scraping. Always exits 0; this is a diagnostic
    surface, not a gate.
    """
    import json as _json
    import time as _time
    from aelfrice.meta_beliefs import decay_state

    store = _open_store()
    try:
        states = store.list_meta_beliefs()
    finally:
        store.close()
    now_ts = int(_time.time())
    rows = [decay_state(s, now_ts=now_ts) for s in states]
    if getattr(args, "json_output", False):
        payload = {
            "now_ts": now_ts,
            "count": len(rows),
            "meta_beliefs": [
                {
                    "key": s.key,
                    "value": s.value,
                    "static_default": s.static_default,
                    "half_life_seconds": s.half_life_seconds,
                    "last_updated_ts": s.last_updated_ts,
                    "signal_weights": dict(sorted(s.signal_weights.items())),
                    "posteriors": [
                        {
                            "signal_class": p.signal_class,
                            "alpha": p.alpha,
                            "beta": p.beta,
                            "last_updated_ts": p.last_updated_ts,
                        }
                        for p in sorted(
                            s.posteriors.values(),
                            key=lambda p: p.signal_class,
                        )
                    ],
                }
                for s in rows
            ],
        }
        print(_json.dumps(payload, indent=2, sort_keys=False), file=out)  # type: ignore[arg-type]
        return 0
    if not rows:
        print("aelf doctor --meta-beliefs: no meta-beliefs installed.",
              file=out)  # type: ignore[arg-type]
        return 0
    print(f"aelf doctor --meta-beliefs: {len(rows)} installed (now_ts={now_ts})",
          file=out)  # type: ignore[arg-type]
    for s in rows:
        print(
            f"\n  {s.key}\n"
            f"    value           = {s.value:.6f}\n"
            f"    static_default  = {s.static_default}\n"
            f"    half_life       = {s.half_life_seconds}s\n"
            f"    last_updated_ts = {s.last_updated_ts}",
            file=out,  # type: ignore[arg-type]
        )
        if not s.posteriors:
            print("    posteriors      = (none — cold start)",
                  file=out)  # type: ignore[arg-type]
            continue
        print("    posteriors:", file=out)  # type: ignore[arg-type]
        for cls in sorted(s.posteriors.keys()):
            p = s.posteriors[cls]
            weight = s.signal_weights.get(cls, 0.0)
            print(
                f"      [{cls}] alpha={p.alpha:.4f} beta={p.beta:.4f} "
                f"weight={weight} ts={p.last_updated_ts}",
                file=out,  # type: ignore[arg-type]
            )
    return 0


def _cmd_doctor_dedup(args: argparse.Namespace, out: object) -> int:
    """Run the v2.0 dedup audit (#197 R1).

    Walks the store, finds near-duplicate belief pairs with Jaccard >=
    `--dedup-jaccard` AND Levenshtein ratio >= `--dedup-levenshtein`,
    and prints a clustered report. Read-only: no edges are inserted,
    no beliefs are mutated. The write-path SUPERSEDES hook is the
    bench-gated R2 deferred behind the corpus benchmark.

    Exit 0 on success regardless of cluster count — clusters are
    diagnostic, not failure conditions. Exit 1 only on store-open
    errors.
    """
    from aelfrice.dedup import (
        DedupConfig,
        dedup_audit,
        format_audit_report,
        load_dedup_config,
    )

    config = load_dedup_config()
    j_override = getattr(args, "dedup_jaccard", None)
    l_override = getattr(args, "dedup_levenshtein", None)
    mp_override = getattr(args, "dedup_max_pairs", None)
    config = DedupConfig(
        jaccard_min=(
            float(j_override) if j_override is not None else config.jaccard_min
        ),
        levenshtein_min=(
            float(l_override)
            if l_override is not None
            else config.levenshtein_min
        ),
        max_candidate_pairs=(
            int(mp_override)
            if mp_override is not None
            else config.max_candidate_pairs
        ),
    )

    store = _open_store()
    try:
        report = dedup_audit(
            store,
            jaccard_min=config.jaccard_min,
            levenshtein_min=config.levenshtein_min,
            max_candidate_pairs=config.max_candidate_pairs,
        )
    except ValueError as exc:
        print(f"aelf doctor dedup: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(format_audit_report(report), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_doctor_relationships(args: argparse.Namespace, out: object) -> int:
    """Run the v2.0 semantic-relationship audit (#201).

    Walks the store, classifies near-pair relationships
    (``contradicts`` / ``refines``) by modality + quantifier signal
    divergence, and prints a report. Read-only: no edges are inserted,
    no beliefs are mutated. The write-path ``CONTRADICTS`` hook remains
    the bench-gated R2 deferred behind the corpus benchmark.
    POTENTIALLY_STALE edges are written via ``--detect-stale`` (#387).

    Exit 0 on success regardless of pair count — pairs are diagnostic,
    not failure conditions. Exit 1 only on store-open errors.
    """
    from aelfrice.relationship_detector import (
        RelationshipDetectorConfig,
        format_audit_report,
        load_relationship_detector_config,
        relationships_audit,
    )

    config = load_relationship_detector_config()
    j_override = getattr(args, "relationships_jaccard", None)
    c_override = getattr(args, "relationships_confidence", None)
    mp_override = getattr(args, "relationships_max_pairs", None)
    config = RelationshipDetectorConfig(
        jaccard_min=(
            float(j_override) if j_override is not None else config.jaccard_min
        ),
        residual_overlap_min=config.residual_overlap_min,
        confidence_min=(
            float(c_override)
            if c_override is not None
            else config.confidence_min
        ),
        max_candidate_pairs=(
            int(mp_override)
            if mp_override is not None
            else config.max_candidate_pairs
        ),
    )

    store = _open_store()
    try:
        report = relationships_audit(
            store,
            jaccard_min=config.jaccard_min,
            residual_overlap_min=config.residual_overlap_min,
            confidence_min=config.confidence_min,
            max_candidate_pairs=config.max_candidate_pairs,
        )
    except ValueError as exc:
        print(f"aelf doctor relationships: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(
        format_audit_report(report, confidence_min=config.confidence_min),
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_doctor_detect_stale(args: argparse.Namespace, out: object) -> int:
    """Emit POTENTIALLY_STALE edges for sub-confidence contradicting pairs (#387).

    Runs ``write_potentially_stale_edges`` against the open store, which:

    1. Calls ``relationships_audit`` to find all contradicting pairs.
    2. Filters to sub-confidence pairs (score < ``confidence_min``).
    3. For each such pair emits one POTENTIALLY_STALE edge from the newer
       belief to the older one (newer = greater ``created_at``, tie-break
       on lex ``id``).
    4. Skips pairs whose edge already exists (idempotent).

    Reuses ``--relationships-jaccard`` / ``--relationships-confidence`` /
    ``--relationships-max-pairs`` overrides — semantics are shared with
    ``--relationships``.

    Exit 0 on success. Exit 1 only on store-open or threshold errors.
    """
    from aelfrice.relationship_detector import (
        RelationshipDetectorConfig,
        format_write_report,
        load_relationship_detector_config,
        write_potentially_stale_edges,
    )

    config = load_relationship_detector_config()
    j_override = getattr(args, "relationships_jaccard", None)
    c_override = getattr(args, "relationships_confidence", None)
    mp_override = getattr(args, "relationships_max_pairs", None)
    config = RelationshipDetectorConfig(
        jaccard_min=(
            float(j_override) if j_override is not None else config.jaccard_min
        ),
        residual_overlap_min=config.residual_overlap_min,
        confidence_min=(
            float(c_override)
            if c_override is not None
            else config.confidence_min
        ),
        max_candidate_pairs=(
            int(mp_override)
            if mp_override is not None
            else config.max_candidate_pairs
        ),
    )

    store = _open_store()
    try:
        write_report = write_potentially_stale_edges(
            store,
            jaccard_min=config.jaccard_min,
            residual_overlap_min=config.residual_overlap_min,
            confidence_min=config.confidence_min,
            max_candidate_pairs=config.max_candidate_pairs,
        )
    except ValueError as exc:
        print(f"aelf doctor --detect-stale: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(format_write_report(write_report), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_doctor_derive_pending(
    args: argparse.Namespace,  # noqa: ARG001
    out: object,
) -> int:
    """Run the derivation worker over every unstamped ingest_log row (#264).

    Manual escape hatch for the recover-by-replay crash semantics from
    `docs/v2_derivation_worker.md`. If a worker died between batches and
    left log rows with `derived_belief_ids IS NULL`, this sweep walks
    them, derives, and stamps each.

    Exit 0 on success regardless of how many rows were swept (zero is
    the steady-state expectation, not an error). Exit 1 only on store
    open / worker exception.
    """
    store = _open_store()
    try:
        before = len(store.list_unstamped_ingest_log())
        report = run_worker(store)
        after = len(store.list_unstamped_ingest_log())
    except Exception as exc:  # pragma: no cover - surface and bail
        print(f"aelf doctor --derive-pending: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(
        "derive-pending sweep\n"
        "--------------------\n"
        f"unstamped before:    {before}\n"
        f"rows scanned:        {report.rows_scanned}\n"
        f"beliefs inserted:    {report.beliefs_inserted}\n"
        f"beliefs corroborated:{report.beliefs_corroborated}\n"
        f"rows stamped:        {report.rows_stamped}\n"
        f"persist=False skips: {report.rows_skipped_no_belief}\n"
        f"unstamped after:     {after}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_doctor_replay(args: argparse.Namespace, out: object) -> int:
    """Run the v2.x full-equality replay probe (#262).

    Called from `_cmd_doctor` when ``--replay`` is set.  Opens the
    store, runs `replay_full_equality`, prints the report, and returns
    0 or 1 based on drift counts and the ``--max-drift`` threshold.
    """
    from aelfrice.replay import replay_full_equality

    max_drift: int | None = getattr(args, "max_drift", None)
    _de_raw = getattr(args, "drift_examples", None)
    drift_examples: int = 10 if _de_raw is None else int(_de_raw)
    replay_scope: str = getattr(args, "replay_scope", "all") or "all"

    store = _open_store()
    try:
        report = replay_full_equality(
            store,
            max_drift=max_drift,
            drift_examples=drift_examples,
            scope=replay_scope,  # type: ignore[arg-type]
        )
    finally:
        store.close()

    _print_replay_report(report, out)

    drift_total = report.mismatched + report.derived_orphan
    threshold = max(0, max_drift) if max_drift is not None else 0
    return 0 if drift_total <= threshold else 1


def _cmd_doctor_classify_orphans(
    args: argparse.Namespace, out: object
) -> int:
    """Run the targeted Haiku reclassification pass for orphan beliefs.

    An orphan is a belief where BOTH of the following are true:
      - type = 'unknown' (never successfully typed by onboard/ingest)
      - alpha + beta <= 2 (untouched prior is alpha=1, beta=1; any
        feedback event pushes the sum above 2)

    Reuses the same `classify_batch` path as `aelf onboard
    --llm-classify`; no new LLM client is introduced.

    With --dry-run: count and display orphans + before-distribution
    without making any LLM calls or DB writes.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    max_n_raw = getattr(args, "max", None)
    max_n: int | None = int(max_n_raw) if max_n_raw is not None else None

    # Gates: SDK importable, API key set.
    gate = _llm_check_gates(enabled=True, model=_LLMConfig.default().model)
    if not gate.pass_all and not dry_run:
        if gate.exit_code is not None:
            if gate.message:
                print(gate.message, file=sys.stderr)
            return gate.exit_code

    api_key = os.environ.get(_LLM_ENV_API_KEY, "")
    if not api_key and not dry_run:
        print(
            f"aelf: {_LLM_ENV_API_KEY} not set; --classify-orphans requires it. "
            "Pass --dry-run to count orphans without making LLM calls.",
            file=sys.stderr,
        )
        return 1

    cfg = _LLMConfig.default()
    store = _open_store()
    try:
        orphan_report = _classify_orphans(
            store,
            api_key=api_key,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            max_n=max_n,
            dry_run=dry_run,
        )
    except _LLMAuthError as exc:
        print(
            f"aelf: Anthropic auth failure ({exc}). No beliefs updated. "
            f"Verify {_LLM_ENV_API_KEY}.",
            file=sys.stderr,
        )
        return 1
    except _LLMTokenCapExceeded as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()

    print(_format_orphan_report(orphan_report), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_doctor_gc_orphan_feedback(
    args: argparse.Namespace, out: object
) -> int:
    """Garbage-collect `feedback_history` rows whose belief_id no
    longer resolves in `beliefs` (issue #223).

    Default is dry-run (count only). With `--apply`, deletes the
    orphan rows. Bypasses the hooks/graph checks; never touches
    LLMs.
    """
    apply = bool(getattr(args, "apply", False))
    store = _open_store()
    try:
        report = _gc_orphan_feedback(store, dry_run=not apply)
    finally:
        store.close()
    print(_format_orphan_feedback_report(report), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_doctor_promote_retention(
    args: argparse.Namespace, out: object
) -> int:
    """Promote snapshot beliefs to ``fact`` once corroborated enough
    (issue #290 phase-3).

    A snapshot is promoted when corroborated >= 3 times across >= 2
    distinct sessions with no inbound CONTRADICTS edge. See
    docs/belief_retention_class.md §4.

    Opt-in by design (mirrors --classify-orphans from #206). Use
    --dry-run to count candidates without mutating; --max N to cap
    promotions per run. Bypasses the hooks/graph checks.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    max_n_raw = getattr(args, "max", None)
    max_n: int | None = int(max_n_raw) if max_n_raw is not None else None

    store = _open_store()
    try:
        report = _promote_retention(store, dry_run=dry_run, max_n=max_n)
    finally:
        store.close()
    print(_format_promotion_report(report), file=out)  # type: ignore[arg-type]
    return 0


def _cmd_doctor_prune_dormant(
    args: argparse.Namespace, out: object
) -> int:
    """Identify per-project DBs idle for >= --idle-days (default 30)
    and optionally delete them after explicit per-DB confirmation (#594).

    Default action is dry-run: scan + list. With ``--apply``, the
    user is prompted ``[y/N]`` per DB; only ``y``/``Y`` deletes the
    file. Anything else preserves the DB and proceeds to the next
    one. Bypasses the hooks/graph checks; never opens the canonical
    store.

    Honours ``--idle-days N`` (override default), ``--projects-dir``
    (override scan root, primarily for tests).
    """
    idle_days_raw = getattr(args, "idle_days", None)
    idle_days = (
        int(idle_days_raw) if idle_days_raw is not None
        else DORMANT_IDLE_DAYS_DEFAULT
    )
    if idle_days < 0:
        print(
            f"aelf doctor --prune-dormant: --idle-days must be >= 0 (got {idle_days})",
            file=sys.stderr,
        )
        return 2

    projects_dir_raw = getattr(args, "projects_dir", None)
    projects_dir = Path(projects_dir_raw) if projects_dir_raw else None

    apply = bool(getattr(args, "apply", False))

    dormant = _check_dormant_dbs(
        projects_dir=projects_dir, idle_days=idle_days,
    )
    if not dormant:
        print(
            f"no per-project DBs idle for >= {idle_days} days.",
            file=out,  # type: ignore[arg-type]
        )
        return 0

    total_bytes = sum(d.size_bytes for d in dormant)
    total_beliefs = sum(d.row_count for d in dormant)
    print(
        f"found {len(dormant)} dormant per-project DB(s) "
        f"(idle >= {idle_days}d, {total_beliefs:,} beliefs, "
        f"{total_bytes / 1024:.1f} KiB total):",
        file=out,  # type: ignore[arg-type]
    )
    for entry in dormant:
        print(_format_dormant_entry(entry), file=out)  # type: ignore[arg-type]

    if not apply:
        print(
            "",
            file=out,  # type: ignore[arg-type]
        )
        print(
            "dry-run only. re-run with `--apply` to be prompted [y/N] per DB.",
            file=out,  # type: ignore[arg-type]
        )
        return 0

    deleted = 0
    skipped = 0
    for entry in dormant:
        prompt = f"delete {entry.path}? [y/N] "
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            try:
                entry.path.unlink()
                deleted += 1
                print(f"  deleted {entry.path}", file=out)  # type: ignore[arg-type]
            except OSError as exc:
                print(
                    f"  ERROR deleting {entry.path}: {exc}",
                    file=sys.stderr,
                )
                skipped += 1
        else:
            skipped += 1
            print(f"  kept {entry.path}", file=out)  # type: ignore[arg-type]

    print("", file=out)  # type: ignore[arg-type]
    print(
        f"prune complete: {deleted} deleted, {skipped} kept.",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _format_dormant_entry(entry: DormantDB) -> str:
    """One-line summary of a dormant DB for the prune listing."""
    return (
        f"  {entry.path} "
        f"({entry.row_count:,} beliefs, "
        f"{entry.size_bytes / 1024:.1f} KiB, "
        f"idle {entry.idle_days}d)"
    )


def _print_doctor_store_check(out: object) -> None:
    """Surface the 'is this store populated?' check (issue #116).

    Doctor's settings linter says nothing about the actual belief
    graph, so the user can have all-green hooks and zero beliefs and
    not know it. Printed even on a clean settings run; never affects
    exit status (broken settings are the only exit-1 trigger).
    """
    try:
        store = _open_store()
    except Exception:  # pragma: no cover - defensive
        return
    try:
        n = store.count_beliefs()
    finally:
        try:
            store.close()
        except Exception:  # pragma: no cover - defensive
            pass
    print("", file=out)  # type: ignore[arg-type]
    if n == 0:
        print(
            "store: 0 beliefs (empty — run `aelf onboard <path>` to "
            "populate, or check whether retrieval/ingest hooks are "
            "actually firing)",
            file=out,  # type: ignore[arg-type]
        )
    else:
        print(
            f"store: {n} belief(s) at {db_path()}",
            file=out,  # type: ignore[arg-type]
        )
    _print_doctor_session_ring(out)


def _print_doctor_session_ring(out: object) -> None:
    """Surface the #740 injection-dedup ring state (one line).

    Reports current ring size, cap, and lifetime eviction count.
    Silent when the sentinel file is absent (no session has fired
    UPS yet on this repo). Never affects exit status.
    """
    try:
        from aelfrice.session_ring import read_ring_file  # noqa: PLC0415

        data = read_ring_file()
    except Exception:  # pragma: no cover - defensive
        return
    if not data:
        return
    ring = data.get("ring") if isinstance(data.get("ring"), list) else []
    ring_max = data.get("ring_max")
    evicted = data.get("evicted_total")
    print(
        f"injection ring: {len(ring)}/{ring_max} ids "
        f"(evicted {evicted} this session)",
        file=out,  # type: ignore[arg-type]
    )


def _cmd_sweep_feedback(args: argparse.Namespace, out: object) -> int:
    """Process the deferred-feedback queue (#191).

    Applies +epsilon to the alpha of each belief whose retrieval-
    exposure row has cleared its grace window without a contradicting
    explicit-feedback event. Cancels rows where an explicit signal
    landed within the grace window. Idempotent.

    Exits 0 unless `--strict` is passed and an exception escapes.
    Without `--strict`, errors are logged to stderr and the command
    exits 0 so the subcommand is safe to wire into cron.
    """
    from aelfrice.deferred_feedback import (
        resolve_epsilon,
        resolve_grace_seconds,
        sweep_deferred_feedback,
    )

    grace = (
        int(args.grace_seconds)
        if args.grace_seconds is not None
        else resolve_grace_seconds()
    )
    eps = (
        float(args.epsilon)
        if args.epsilon is not None
        else resolve_epsilon()
    )
    limit = int(args.limit) if args.limit is not None else 10_000

    store = _open_store()
    try:
        result = sweep_deferred_feedback(
            store,
            grace_seconds=grace,
            epsilon=eps,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 - cron-safe by default
        print(f"aelf sweep-feedback: {exc}", file=sys.stderr)
        if getattr(args, "strict", False):
            return 1
        return 0
    finally:
        try:
            store.close()
        except Exception:  # pragma: no cover
            pass

    print(
        f"sweep-feedback: applied={result.applied} "
        f"cancelled={result.cancelled} "
        f"skipped_no_belief={result.skipped_no_belief} "
        f"pending_in_grace={result.pending_unmet_grace} "
        f"epsilon={result.epsilon_used} "
        f"grace_seconds={result.grace_seconds_used}",
        file=out,  # type: ignore[arg-type]
    )
    return 0


def _cmd_clamp_ghosts(args: argparse.Namespace, out: object) -> int:
    """Clamp α on belief rows with α inflation and no audit trail.

    Dry-run by default; pass ``--apply`` to mutate. See
    ``aelfrice/clamp_ghosts.py`` for the structural definition of a
    ghost-α row and the reversibility scheme.
    """
    from aelfrice.clamp_ghosts import clamp_ghost_alphas

    store = _open_store()
    try:
        try:
            result = clamp_ghost_alphas(
                store,
                threshold_alpha=float(args.threshold),
                target_alpha=float(args.target),
                dry_run=not bool(args.apply),
                limit=(int(args.limit) if args.limit is not None else None),
            )
        except ValueError as exc:
            print(f"aelf clamp-ghosts: {exc}", file=sys.stderr)
            return 2
    finally:
        try:
            store.close()
        except Exception:  # pragma: no cover
            pass

    print(
        f"clamp-ghosts: matched={result.matched} clamped={result.clamped} "
        f"skipped={result.skipped} dry_run={result.dry_run} "
        f"threshold={result.threshold_alpha} target={result.target_alpha}",
        file=out,  # type: ignore[arg-type]
    )
    if result.sample:
        print("\ntop matches:", file=out)  # type: ignore[arg-type]
        for s in result.sample:
            print(
                f"  α={s['prior_alpha']:6.2f}  {s['id']}  {s['preview']!r}",
                file=out,  # type: ignore[arg-type]
            )
    if result.dry_run and result.matched > 0:
        print(
            "\nrun with --apply to actually clamp.",
            file=out,  # type: ignore[arg-type]
        )
    return 0


def _cmd_session_delta(args: argparse.Namespace, out: object) -> int:
    """Compute per-session telemetry and append one v=1 row.

    Called by the SessionEnd hook. Exits 0 on every code path — a hook
    failure must never surface to the user as a broken shell session.
    A missing or empty --id is a warning-only no-op (stderr, exit 0).
    """
    from aelfrice.telemetry import DEFAULT_TELEMETRY_PATH, emit_session_delta
    from pathlib import Path as _Path

    session_id: str = args.session_id or ""
    telemetry_path: _Path | None = (
        _Path(args.telemetry_path) if args.telemetry_path else DEFAULT_TELEMETRY_PATH
    )
    store = _open_store()
    try:
        emit_session_delta(session_id, store=store, path=telemetry_path)
    finally:
        store.close()
    return 0


def _cmd_tail(args: argparse.Namespace, out: object) -> int:
    """Live-tail the per-turn hook audit log (#321).

    Reads `<git-common-dir>/aelfrice/hook_audit.jsonl` (the per-turn
    record store written by UserPromptSubmit and SessionStart on every
    fire that produces an injection block) and emits a pretty-printed
    stream. By default follows for new records; `--no-follow` reads the
    current contents once and exits.
    """
    from aelfrice.hook_tail import parse_filter, parse_since, tail_audit

    filters: list[tuple[str, str]] = []
    raw_filters: list[str] | None = getattr(args, "filter", None)
    if raw_filters:
        for spec in raw_filters:
            try:
                filters.append(parse_filter(spec))
            except ValueError as exc:
                print(f"aelf tail: {exc}", file=sys.stderr)
                return 2
    since = None
    raw_since: str | None = getattr(args, "since", None)
    if raw_since:
        try:
            since = parse_since(raw_since)
        except ValueError as exc:
            print(f"aelf tail: {exc}", file=sys.stderr)
            return 2
    follow = not bool(getattr(args, "no_follow", False))
    include_blob = not bool(getattr(args, "no_blob", False))
    return tail_audit(
        filters=filters,
        since=since,
        include_blob=include_blob,
        follow=follow,
        out=out,  # type: ignore[arg-type]
    )


def _cmd_scan_derivation(args: argparse.Namespace, out: object) -> int:
    """N-gram Jaccard similarity gate against a reference document (#681).

    Reads each PATH (or stdin when '-' is given), computes N-gram
    Jaccard similarity against the reference file, and prints a
    one-line result per input.

    Exit codes:
      0  all inputs are below the threshold (clean)
      1  at least one input exceeded the threshold (matched)
      2  the reference file is unreadable

    Designed to drop into a git pre-commit / pre-push hook.
    """
    from pathlib import Path as _Path

    from aelfrice.noise_filter import similarity_to_reference

    reference_raw: str | None = getattr(args, "reference", None)
    threshold: float = float(getattr(args, "threshold", 0.6))
    n: int = int(getattr(args, "n", 3))
    paths: list[str] = list(getattr(args, "paths", []) or [])

    if not reference_raw:
        print("aelf scan-derivation: --reference is required", file=sys.stderr)
        return 2
    reference = _Path(reference_raw)
    if not reference.is_file():
        print(
            f"aelf scan-derivation: reference not found or unreadable: {reference}",
            file=sys.stderr,
        )
        return 2

    # Pre-check that we can read the reference (exit 2 on failure).
    try:
        reference.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(
            f"aelf scan-derivation: cannot read reference {reference}: {exc}",
            file=sys.stderr,
        )
        return 2

    any_matched = False

    def _check_text(label: str, text: str) -> None:
        nonlocal any_matched
        over, score, excerpt = similarity_to_reference(
            text, reference, n=n, threshold=threshold
        )
        if over:
            any_matched = True
            snippet = excerpt or text[:120]
            print(
                f"MATCH [{score:.3f}] {label}: {snippet!r}",
                file=out,  # type: ignore[arg-type]
            )
        else:
            print(
                f"clean [{score:.3f}] {label}",
                file=out,  # type: ignore[arg-type]
            )

    if not paths:
        # No paths: read stdin as a single blob.
        text = sys.stdin.read()
        _check_text("<stdin>", text)
    else:
        for p in paths:
            if p == "-":
                text = sys.stdin.read()
                _check_text("<stdin>", text)
            else:
                try:
                    text = _Path(p).read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    print(
                        f"aelf scan-derivation: cannot read {p}: {exc}",
                        file=sys.stderr,
                    )
                    any_matched = True  # treat unreadable inputs as failures
                    continue
                _check_text(p, text)

    return 1 if any_matched else 0


def _known_cli_subcommands() -> frozenset[str]:
    """Snapshot of the subcommands the running `aelf` parser knows.

    Re-introspecting the parser keeps doctor honest as new
    subcommands land — the source of truth is `build_parser()`,
    not a hardcoded list. Issue #115 acceptance: a slash file
    naming a subcommand the parser doesn't know is the "stale
    slash file" we want to flag.
    """
    parser = build_parser()
    out: set[str] = set()
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            out.update(action.choices.keys())
    return frozenset(out)


# --- Dispatcher ---------------------------------------------------------


class _SuppressSubparsersFormatter(argparse.HelpFormatter):
    """Hide subparsers registered with help=argparse.SUPPRESS from --help.

    argparse's stock HelpFormatter ignores SUPPRESS on subparser
    pseudo-actions, so they leak through as `==SUPPRESS==` literals. We
    filter them out at format time.
    """

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            kept = [
                ca for ca in action._choices_actions  # pyright: ignore[reportPrivateUsage]
                if ca.help is not argparse.SUPPRESS
            ]
            action._choices_actions = kept  # pyright: ignore[reportPrivateUsage]
        return super()._format_action(action)


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Parameters
    ----------
    show_advanced:
        When *True* the parser uses the plain :class:`argparse.HelpFormatter`
        so every subcommand (including those registered with
        ``help=argparse.SUPPRESS``) appears in ``--help`` output.  This is
        the behaviour triggered by ``aelf --advanced [--help]``.
        When *False* (the default) :class:`_SuppressSubparsersFormatter` hides
        the suppressed subcommands from ``--help``.
    """
    formatter = (
        argparse.HelpFormatter if show_advanced else _SuppressSubparsersFormatter
    )
    parser = argparse.ArgumentParser(
        prog="aelf",
        description=(
            "Persistent memory for AI agents. Set up once, stays out of "
            "your way. The subcommands below are the everyday surface; "
            "advanced verbs (diagnostics, archive/uninstall, hook entry-"
            "points) are hidden from --help. See docs/COMMANDS.md for the "
            "complete reference."
        ),
        formatter_class=formatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aelf {_AELFRICE_VERSION}",
        help="print the installed aelfrice version and exit",
    )
    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        metavar="<command>",
    )

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
    p_onboard.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "path to a project directory. Required for the default flow "
            "and for --emit-candidates; omitted for --accept-classifications."
        ),
    )
    p_onboard.add_argument(
        "--emit-candidates",
        dest="emit_candidates",
        action="store_true",
        help=(
            "extract candidates, persist a PENDING onboard session, "
            "and print {session_id, n_already_present, sentences[]} as "
            "JSON to stdout. No network. Used by /aelf:onboard."
        ),
    )
    p_onboard.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help=(
            "read-only pre-scan: report how many candidates would be "
            "filtered as already-present vs handed to the classifier, "
            "then exit without inserting any beliefs or persisting a "
            "session. No network, no LLM consent prompt. Use to decide "
            "whether re-onboard is worth the classification cost (#761)."
        ),
    )
    p_onboard.add_argument(
        "--accept-classifications",
        dest="accept_classifications",
        action="store_true",
        help=(
            "apply host-supplied classifications to a pending onboard "
            "session. Requires --session-id and --classifications-file. "
            "No network. Used by /aelf:onboard."
        ),
    )
    p_onboard.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        help="session id returned by --emit-candidates",
    )
    p_onboard.add_argument(
        "--classifications-file",
        dest="classifications_file",
        default=None,
        help=(
            "JSON file with [{index, belief_type, persist}, ...] entries. "
            "Use '-' to read from stdin."
        ),
    )
    p_onboard.add_argument(
        "--llm-classify",
        dest="llm_classify",
        nargs="?",
        const=True,
        default=None,
        help=(
            "opt in to the LLM-Haiku classifier (v1.3.0+). Requires the "
            "[onboard-llm] extra and ANTHROPIC_API_KEY. Default: off. "
            "Pass --llm-classify=false to override [onboard.llm].enabled."
        ),
    )
    p_onboard.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "print the candidates that would be sent to the classifier "
            "without contacting the network. Implies --llm-classify."
        ),
    )
    p_onboard.add_argument(
        "--revoke-consent",
        dest="revoke_consent",
        action="store_true",
        help=(
            "remove the per-machine LLM-classify consent sentinel and "
            "exit. No scan is run."
        ),
    )
    p_onboard.set_defaults(func=_cmd_onboard)

    p_search = sub.add_parser("search", help="L0 locked + L1 FTS5 retrieval")
    p_search.add_argument("query", help="keyword query")
    p_search.add_argument(
        "--budget", type=int, default=DEFAULT_TOKEN_BUDGET,
        help="output token budget (default 2000)",
    )
    p_search.set_defaults(func=_cmd_search)

    p_reason = sub.add_parser(
        "reason",
        help="surface a reasoning chain over the belief graph (#389)",
    )
    p_reason.add_argument("query", help="keyword query for seed selection")
    p_reason.add_argument(
        "--seed-id", action="append", default=[], dest="seed_id",
        help="explicit seed belief id (repeatable; bypasses BM25 seed selection)",
    )
    p_reason.add_argument(
        "--k", type=int, default=3,
        help="BM25 seed fanout when --seed-id not given (default 3)",
    )
    p_reason.add_argument(
        "--depth", type=int, default=2,
        help="max BFS hop depth (default 2)",
    )
    p_reason.add_argument(
        "--budget", type=int, default=10,
        help="total expansion-node budget (default 10)",
    )
    p_reason.add_argument(
        "--fanout", type=int, default=8,
        help="max edges expanded per frontier entry (default 8)",
    )
    p_reason.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of indented hop tree",
    )
    p_reason.set_defaults(func=_cmd_reason)

    p_wonder = sub.add_parser(
        "wonder",
        help="surface consolidation candidates / phantom beliefs (#389)",
        description=(
            "Modes (#645):\n"
            "  no-arg — graph-walk consolidation candidates from a BFS "
            "expansion around a seed.\n"
            "  QUERY (positional) — axes-spawn-ingest research flow "
            "(gap analysis + research axes JSON for skill-layer dispatch).\n"
            "  --gc — soft-delete stale speculative beliefs.\n"
            "  --emit-phantoms — graph-walk candidates serialised as "
            "Phantom JSON for offline review.\n"
            "  --persist — graph-walk candidates persisted to the store "
            "via wonder_ingest.\n"
            "  --axes QUERY — deprecated alias for the positional QUERY."
        ),
    )
    p_wonder.add_argument(
        "query", nargs="?", default=None,
        help=(
            "research query (#645): when provided, runs the axes-spawn-"
            "ingest flow instead of graph-walk. Equivalent to --axes "
            "QUERY but ergonomic. With no query, graph-walk runs."
        ),
    )
    p_wonder.add_argument(
        "--seed", default=None,
        help="explicit seed belief id (default: highest-degree non-locked)",
    )
    p_wonder.add_argument(
        "--top", type=int, default=10,
        help="number of consolidation candidates to surface (default 10)",
    )
    p_wonder.add_argument(
        "--emit-phantoms", action="store_true", dest="emit_phantoms",
        help=(
            "emit candidates as Phantom JSON objects to stdout "
            "(integration with store deferred to v2.x #229 lane)"
        ),
    )
    p_wonder.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of human-readable rows",
    )
    p_wonder.add_argument(
        "--axes", metavar="QUERY", default=None,
        help=(
            "axes mode (#551): emit gap analysis + research axes JSON "
            "for research-agent dispatch. Skips graph-walk; consumed by "
            "the skill layer to fan out parallel research agents."
        ),
    )
    p_wonder.add_argument(
        "--axes-budget", type=int, default=24, dest="axes_budget",
        help="--axes mode: max candidates pulled from retrieve() (default 24)",
    )
    p_wonder.add_argument(
        "--axes-depth", type=int, default=2, dest="axes_depth",
        help="--axes mode: BFS expansion depth, 0 disables BFS (default 2)",
    )
    p_wonder.add_argument(
        "--axes-agents", type=int, default=4, dest="axes_agents",
        help="--axes mode: agent_count hint for the skill layer (default 4)",
    )
    p_wonder.add_argument(
        "--persist", action="store_true", dest="persist",
        help=(
            "after building phantom candidates, persist them to the store "
            "via wonder_ingest and print the insert summary "
            "(inserted/skipped/edges_created). "
            "Mutually exclusive with --emit-phantoms and --axes."
        ),
    )
    p_wonder.add_argument(
        "--persist-docs", metavar="FILE", default=None, dest="persist_docs",
        help=(
            "skill-layer integration (#552): read subagent research "
            "documents from a JSONL FILE (one row per axis: "
            "{axis_name, content, anchor_ids}), convert to Phantoms, "
            "and persist via wonder_ingest. Used by the "
            "/aelf:wonder --axes flow after the host has collected "
            "subagent responses. Skips graph-walk; ignores --seed / "
            "--top. Mutually exclusive with --persist, --emit-phantoms, "
            "--axes."
        ),
    )
    # gc mode (#549): soft-delete stale speculative beliefs. Promoted from
    # a nested sub-subcommand to a flag in #645 so a positional QUERY can
    # be added to `aelf wonder` without colliding with `wonder gc`.
    p_wonder.add_argument(
        "--gc", action="store_true", dest="gc",
        help=(
            "soft-delete stale speculative beliefs (wonder GC). "
            "Use --gc-dry-run to preview candidates; --gc-ttl-days to "
            "override the 14-day age threshold."
        ),
    )
    p_wonder.add_argument(
        "--gc-dry-run", action="store_true", dest="gc_dry_run",
        help="--gc mode: report candidates without mutating the store",
    )
    p_wonder.add_argument(
        "--gc-ttl-days", type=int, default=14, dest="gc_ttl_days",
        help="--gc mode: age threshold in days for GC candidates (default 14)",
    )
    p_wonder.set_defaults(func=_cmd_wonder)

    # v1.4 (#141): user-facing manual trigger for the context
    # rebuilder. Promoted from hidden to visible because manual mode
    # is now the default `trigger_mode` at v1.4.0; the slash command
    # `/aelf:rebuild` and direct `aelf rebuild` are how users fire it.
    p_rebuild = sub.add_parser(
        "rebuild",
        help="manually fire the context rebuilder (manual trigger)",
    )
    p_rebuild.add_argument(
        "--transcript", default=None,
        help=(
            "path to a Claude Code session JSONL to read recent turns "
            "from. Default: walk upward from cwd for "
            ".git/aelfrice/transcripts/turns.jsonl."
        ),
    )
    p_rebuild.add_argument(
        "--n", type=int, default=None,
        help=(
            "number of recent turns to seed the query (default: "
            "[rebuilder].turn_window_n in .aelfrice.toml, "
            "or 50 when unset)"
        ),
    )
    p_rebuild.add_argument(
        "--budget", type=int, default=None,
        help=(
            "token budget for the rebuild block (default: "
            "[rebuilder].token_budget in .aelfrice.toml, "
            "or 4000 when unset)"
        ),
    )
    p_rebuild.set_defaults(func=_cmd_rebuild)

    p_lock = sub.add_parser("lock", help="insert (or upgrade) a user-locked belief")
    p_lock.add_argument("statement", help="belief text to lock as ground truth")
    p_lock.add_argument(
        "--id", dest="session_id", default=None,
        help=(
            "session_id to stamp on the locked belief and ingest_log row. "
            "Defaults to $AELF_SESSION_ID; warns to stderr and writes NULL "
            "if neither is set (#192)."
        ),
    )
    p_lock.add_argument(
        "--doc", dest="doc_uri", default=None,
        help=(
            "optional doc URI to anchor on this belief (#435). Stored as "
            "anchor_type='manual' on belief_documents; opaque to the "
            "linker beyond non-empty (file:// or https:// recommended)."
        ),
    )
    p_lock.set_defaults(func=_cmd_lock)

    p_locked = sub.add_parser("locked", help="list locked beliefs")
    p_locked.add_argument(
        "--pressured", action="store_true",
        help="only show locks with nonzero demotion_pressure",
    )
    p_locked.set_defaults(func=_cmd_locked)

    # Read-only lens: load-bearing beliefs (locked ∪ corroborated ∪ high-posterior).
    p_core = sub.add_parser(
        "core",
        help="surface load-bearing beliefs: locked ∪ corroborated ∪ high-posterior",
    )
    p_core.add_argument(
        "--json", action="store_true", dest="json",
        help="emit JSON list instead of text",
    )
    p_core.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="cap result count after filtering/sort",
    )
    p_core.add_argument(
        "--min-corroboration", type=int, default=_CORE_MIN_CORROBORATION,
        metavar="N",
        help="corroboration_count threshold (default 2; lower to widen "
             "lens — 0 admits any non-negative count)",
    )
    p_core.add_argument(
        "--min-posterior", type=float, default=_CORE_MIN_POSTERIOR,
        metavar="FLOAT",
        help="posterior-mean threshold (default 2/3; lower to widen "
             "lens — 0.0 admits any belief that passes --min-alpha-beta)",
    )
    p_core.add_argument(
        "--min-alpha-beta", type=int, default=_CORE_MIN_ALPHA_BETA,
        metavar="N",
        help="α+β co-gate for posterior signal (default 4)",
    )
    _core_excl = p_core.add_mutually_exclusive_group()
    _core_excl.add_argument(
        "--locked-only", action="store_true",
        help="return only the locked subset",
    )
    _core_excl.add_argument(
        "--no-locked", action="store_true",
        help="suppress locked beliefs (surface corroboration/posterior only)",
    )
    p_core.set_defaults(func=_cmd_core)

    # Hidden: belief lifecycle inverse of `lock` / `validate`. Power-user.
    p_demote = sub.add_parser(
        "demote",
        help=argparse.SUPPRESS,
    )
    p_demote.add_argument("belief_id", help="id of the belief to demote")
    p_demote.add_argument(
        "--to-scope",
        default=None,
        dest="to_scope",
        metavar="SCOPE",
        help=(
            "flip the belief's visibility scope to SCOPE "
            "('project', 'global', or 'shared:<name>'). "
            "Orthogonal to tier demotion — when supplied, only the scope "
            "is changed. Writes a zero-valence audit row tagged "
            "'scope:<old>-><new>' to feedback_history. "
            "Rejected for foreign belief ids."
        ),
    )
    p_demote.set_defaults(func=_cmd_demote)

    # Hidden: weaker than `aelf lock`; rarely needed at the CLI surface.
    p_validate = sub.add_parser(
        "validate",
        help=argparse.SUPPRESS,
    )
    p_validate.add_argument(
        "belief_id", help="id of the belief to validate",
    )
    p_validate.add_argument(
        "--source", default="user_validated",
        help=(
            "audit-row source suffix; written as 'promotion:<source>' "
            "in feedback_history. Defaults to 'user_validated'."
        ),
    )
    p_validate.set_defaults(func=_cmd_validate)

    # Explicit unlock — clears user-lock, writes lock:unlock audit row.
    p_unlock = sub.add_parser(
        "unlock",
        help="drop a user-lock on a belief (writes audit row)",
    )
    p_unlock.add_argument("belief_id", help="id of the belief to unlock")
    p_unlock.set_defaults(func=_cmd_unlock)

    # Hard-delete: remove a belief and all its edges. Confirmation prompt by
    # default; --yes to bypass. --force allows deletion of locked beliefs.
    p_delete = sub.add_parser(
        "delete",
        help="hard-delete a belief from the store (writes audit row before cascade)",
    )
    p_delete.add_argument("belief_id", help="id of the belief to delete")
    p_delete.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation prompt",
    )
    p_delete.add_argument(
        "--force",
        action="store_true",
        help="allow deletion of beliefs with lock_level=user",
    )
    p_delete.set_defaults(func=_cmd_delete)

    # Explicit user affirmation — bumps Beta-Bernoulli alpha without freezing.
    p_confirm = sub.add_parser(
        "confirm",
        help="affirm a belief: bumps posterior toward truth without locking it",
    )
    p_confirm.add_argument("belief_id", help="id of the belief to affirm")
    p_confirm.add_argument(
        "--source",
        default="user_confirmed",
        help=(
            "source label written to feedback_history. "
            "Defaults to 'user_confirmed'."
        ),
    )
    p_confirm.add_argument(
        "--note",
        default="",
        help="optional free-text annotation (printed on success, not persisted)",
    )
    p_confirm.set_defaults(func=_cmd_confirm)

    # promote: user-facing alias of validate. Same handler, same flags.
    p_promote = sub.add_parser(
        "promote",
        help="promote an agent_inferred belief to user_validated (alias of validate)",
    )
    p_promote.add_argument(
        "belief_id", help="id of the belief to promote",
    )
    p_promote.add_argument(
        "--source", default="user_validated",
        help=(
            "audit-row source suffix; written as 'promotion:<source>' "
            "in feedback_history. Defaults to 'user_validated'."
        ),
    )
    p_promote.add_argument(
        "--to-scope",
        default=None,
        dest="to_scope",
        metavar="SCOPE",
        help=(
            "flip the belief's visibility scope to SCOPE "
            "('project', 'global', or 'shared:<name>'). "
            "Orthogonal to origin promotion — when supplied alongside "
            "an agent_inferred belief, both the origin flip and scope "
            "flip occur. Against an already-validated belief, only the "
            "scope is changed. Writes a zero-valence audit row tagged "
            "'scope:<old>-><new>' to feedback_history. "
            "Rejected for foreign belief ids."
        ),
    )
    p_promote.set_defaults(func=_cmd_promote)

    # Hidden: contradiction-resolution maintenance verb. `aelf doctor` flags
    # when a run is needed; users rarely invoke it directly.
    p_resolve = sub.add_parser(
        "resolve",
        help=argparse.SUPPRESS,
        epilog=(
            "Picks a winner per precedence (user_stated > user_corrected "
            "> document_recent; ties broken by recency, then id) and "
            "creates a SUPERSEDES thread from winner to loser. Each "
            "resolution writes an audit row to feedback_history with "
            "source='contradiction_tiebreaker:<rule>'. Idempotent — "
            "already-resolved pairs are skipped."
        ),
    )
    p_resolve.set_defaults(func=_cmd_resolve)

    # Hidden: agents emit feedback automatically via the MCP path. Manual
    # invocation is rare.
    p_feedback = sub.add_parser("feedback", help=argparse.SUPPRESS)
    p_feedback.add_argument("belief_id", help="id of the belief")
    p_feedback.add_argument("signal", choices=["used", "harmful"],
                             help="feedback signal sign")
    p_feedback.add_argument("--source", default="user",
                             help="audit source label (default 'user')")
    p_feedback.set_defaults(func=_cmd_feedback)

    # `aelf status` — one-screen "what's in this store" snapshot.
    # Renamed from `aelf stats` at v1.3 per CLI_SURFACE_AUDIT.md;
    # `stats` stays as a hidden alias for one minor.
    p_status = sub.add_parser(
        "status",
        help="summary of belief / lock / history counts",
    )
    p_status.set_defaults(func=_cmd_stats)

    # Deprecated alias of `status`. Hidden from --help; deleted at v1.4.
    p_stats = sub.add_parser("stats", help=argparse.SUPPRESS)
    p_stats.set_defaults(func=_cmd_stats)

    # Deprecated alias of `doctor graph`. Hidden from --help; deleted at v1.4.
    p_health = sub.add_parser("health", help=argparse.SUPPRESS)
    p_health.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit JSON: {audit: {...}, features: {edges_by_type: {...}}}",
    )
    p_health.set_defaults(func=_cmd_health)

    # Hidden: research output, not a daily verb.
    p_regime = sub.add_parser("regime", help=argparse.SUPPRESS)
    p_regime.set_defaults(func=_cmd_regime)

    p_migrate = sub.add_parser(
        "migrate",
        help="copy beliefs from a legacy per-project DB into the current one",
    )
    p_migrate.add_argument(
        "--from", dest="from_path", default=None,
        help="legacy DB path (default: ~/.aelfrice/memory.db)",
    )
    p_migrate.add_argument(
        "--apply", action="store_true",
        help="actually copy beliefs (default: dry-run)",
    )
    p_migrate.add_argument(
        "--all", dest="copy_all", action="store_true",
        help="skip the project-mention filter and copy every legacy belief",
    )
    p_migrate.add_argument(
        "--yes", action="store_true",
        help="skip the confirmation prompt under --apply",
    )
    p_migrate.set_defaults(func=_cmd_migrate)

    # Hidden: spawned by the transcript-logger hook on rotation. Manual
    # invocation is for batch backfill of historical JSONL — power-user only.
    p_ingest_transcript = sub.add_parser(
        "ingest-transcript",
        help=argparse.SUPPRESS,
    )
    p_ingest_transcript.add_argument(
        "path", nargs="?", default=None,
        help=(
            "path to one turns.jsonl file (typically under "
            ".git/aelfrice/transcripts/archive/). Mutually exclusive "
            "with --batch."
        ),
    )
    p_ingest_transcript.add_argument(
        "--batch", default=None,
        help=(
            "ingest every *.jsonl under DIR (recursive). Handles both "
            "transcript-logger turns.jsonl and Claude Code session "
            "JSONLs at ~/.claude/projects/. Mutually exclusive with PATH."
        ),
    )
    p_ingest_transcript.add_argument(
        "--since", default=None,
        help=(
            "with --batch, skip files whose mtime is older than this "
            "cutoff. Accepts YYYY-MM-DD or full ISO timestamp."
        ),
    )
    p_ingest_transcript.add_argument(
        "--source-label", default="transcript",
        help="source label written on every belief (default: 'transcript')",
    )
    p_ingest_transcript.set_defaults(func=_cmd_ingest_transcript)

    p_doctor = sub.add_parser(
        "doctor",
        help=(
            "diagnose hooks + brain-graph health. With no subcommand "
            "runs both checks. Exits 1 on any structural failure."
        ),
    )
    p_doctor.add_argument(
        "scope", nargs="?", choices=("hooks", "graph"), default=None,
        help=(
            "limit doctor to one check. 'hooks' validates settings.json "
            "hook commands resolve; 'graph' runs the structural auditor "
            "(orphan threads, FTS5 sync, locked contradictions). "
            "Omit to run both."
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
    p_doctor.add_argument(
        "--hook-failures-log", default=None,
        help=(
            "override the hook-failures log path doctor tails "
            "(default: ~/.aelfrice/logs/hook-failures.log)"
        ),
    )
    p_doctor.add_argument(
        "--classify-orphans",
        dest="classify_orphans",
        action="store_true",
        default=False,
        help=(
            "find beliefs with type='unknown' and no feedback signal "
            "(alpha+beta <= 2), then re-classify them via Haiku batch. "
            "Requires ANTHROPIC_API_KEY and the [onboard-llm] extra "
            "(pip install aelfrice[onboard-llm]). "
            "Bypasses the hooks/graph checks. "
            "Combine with --dry-run to count orphans without LLM calls. "
            "Combine with --max N to cap classifications per run "
            "(recommended: 500 for large stores)."
        ),
    )
    p_doctor.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "with --classify-orphans: print orphan count and "
            "before-distribution without making LLM calls or DB writes. "
            "with --promote-retention: count promotable snapshots "
            "without writing."
        ),
    )
    p_doctor.add_argument(
        "--max",
        dest="max",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --classify-orphans: cap the number of beliefs "
            "classified per run. No cap by default; recommended: 500 "
            "for large stores to bound per-run cost. "
            "with --promote-retention: cap the number of beliefs "
            "promoted per run."
        ),
    )
    p_doctor.add_argument(
        "--promote-retention",
        dest="promote_retention",
        action="store_true",
        default=False,
        help=(
            "promote snapshot beliefs to 'fact' once corroborated >= 3 "
            "times across >= 2 distinct sessions with no inbound "
            "CONTRADICTS edge (issue #290 phase-3). Bypasses the "
            "hooks/graph checks. Combine with --dry-run to count "
            "candidates without writing."
        ),
    )
    p_doctor.add_argument(
        "--gc-orphan-feedback",
        dest="gc_orphan_feedback",
        action="store_true",
        default=False,
        help=(
            "find feedback_history rows whose belief_id no longer "
            "resolves in beliefs (issue #223 residue from pre-#283 "
            "re-ingest) and report the count. Bypasses the "
            "hooks/graph checks. Combine with --apply to delete the "
            "orphan rows; default is dry-run."
        ),
    )
    p_doctor.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=False,
        help=(
            "with --gc-orphan-feedback: actually delete the orphan "
            "rows (default: dry-run)."
        ),
    )
    p_doctor.add_argument(
        "--replay",
        dest="replay",
        action="store_true",
        default=False,
        help=(
            "run the v2.x full-equality replay probe (#262): re-derive "
            "every non-legacy ingest_log row and compare to canonical "
            "beliefs. Exits 0 when mismatched + derived_orphan == 0 "
            "(or <= --max-drift N). Bypasses the hooks/graph checks."
        ),
    )
    p_doctor.add_argument(
        "--derive-pending",
        dest="derive_pending",
        action="store_true",
        default=False,
        help=(
            "v2.x #264 manual sweep: invoke the derivation worker over "
            "every unstamped ingest_log row (recover-by-replay escape "
            "hatch when a prior batch crashed mid-stamp). Idempotent; "
            "zero unstamped rows is the steady state. Bypasses the "
            "hooks/graph checks."
        ),
    )
    p_doctor.add_argument(
        "--prune-dormant",
        dest="prune_dormant",
        action="store_true",
        default=False,
        help=(
            "list per-project DBs under ~/.aelfrice/projects/*/memory.db "
            "that have been idle for >= --idle-days (default 30). "
            "Combine with --apply to be prompted [y/N] per DB and "
            "delete on 'y'. Bypasses the hooks/graph checks; never "
            "opens the canonical store. Issue #594."
        ),
    )
    p_doctor.add_argument(
        "--idle-days",
        dest="idle_days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --prune-dormant: idle threshold in days (default: 30). "
            "Files with mtime older than now - N days qualify."
        ),
    )
    p_doctor.add_argument(
        "--projects-dir",
        dest="projects_dir",
        default=None,
        metavar="PATH",
        help=(
            "with --prune-dormant: override the per-project DB scan "
            "root (default: ~/.aelfrice/projects). Primarily for tests."
        ),
    )
    p_doctor.add_argument(
        "--max-drift",
        dest="max_drift",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --replay: exit 0 when mismatched + derived_orphan <= N "
            "instead of requiring exactly 0."
        ),
    )
    p_doctor.add_argument(
        "--drift-examples",
        dest="drift_examples",
        type=int,
        default=10,
        metavar="N",
        help=(
            "with --replay: maximum representative cases to capture per "
            "drift bucket (default: 10)."
        ),
    )
    p_doctor.add_argument(
        "--replay-scope",
        dest="replay_scope",
        choices=("all", "since-v2"),
        default="all",
        help=(
            "with --replay: 'all' (default) walks every non-legacy "
            "ingest_log row; 'since-v2' is equivalent post-#263 migration "
            "(exists for forward compatibility)."
        ),
    )
    p_doctor.add_argument(
        "--dedup",
        dest="dedup",
        action="store_true",
        default=False,
        help=(
            "find near-duplicate beliefs (Jaccard + Levenshtein gate) "
            "and print clustered candidates (#197). Read-only: no edges "
            "are inserted. Bypasses the hooks/graph checks. Tune via "
            "--dedup-jaccard / --dedup-levenshtein / --dedup-max-pairs "
            "or [dedup] in .aelfrice.toml."
        ),
    )
    p_doctor.add_argument(
        "--dedup-jaccard",
        dest="dedup_jaccard",
        type=float,
        default=None,
        metavar="F",
        help=(
            "with --dedup: override the Jaccard prefilter threshold "
            "(0.0-1.0). Default: [dedup] jaccard_min in .aelfrice.toml > "
            "0.8."
        ),
    )
    p_doctor.add_argument(
        "--dedup-levenshtein",
        dest="dedup_levenshtein",
        type=float,
        default=None,
        metavar="F",
        help=(
            "with --dedup: override the Levenshtein-ratio confirmation "
            "threshold (0.0-1.0). Default: [dedup] levenshtein_min in "
            ".aelfrice.toml > 0.85."
        ),
    )
    p_doctor.add_argument(
        "--dedup-max-pairs",
        dest="dedup_max_pairs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --dedup: cap reported duplicate pairs after Jaccard "
            "prefilter; deterministic truncation by (id_a, id_b). "
            "Default: [dedup] max_candidate_pairs in .aelfrice.toml > 5000."
        ),
    )
    p_doctor.add_argument(
        "--relationships",
        dest="relationships",
        action="store_true",
        default=False,
        help=(
            "classify near-pair belief relationships "
            "(contradicts/refines) by modality + quantifier signals "
            "(#201). Read-only: no edges are inserted. Bypasses the "
            "hooks/graph checks. Tune via --relationships-jaccard / "
            "--relationships-confidence / --relationships-max-pairs "
            "or [relationship_detector] in .aelfrice.toml."
        ),
    )
    p_doctor.add_argument(
        "--relationships-jaccard",
        dest="relationships_jaccard",
        type=float,
        default=None,
        metavar="F",
        help=(
            "with --relationships: minimum Jaccard token overlap "
            "(0.0-1.0) for the candidate-pair prefilter (shared with "
            "dedup). Default: [relationship_detector] jaccard_min in "
            ".aelfrice.toml > 0.4."
        ),
    )
    p_doctor.add_argument(
        "--relationships-confidence",
        dest="relationships_confidence",
        type=float,
        default=None,
        metavar="F",
        help=(
            "with --relationships: score floor at which a "
            "`contradicts` verdict is reported as auto-emit eligible "
            "(for the deferred write-path hook). Sub-confidence pairs "
            "still surface in the audit. Default: "
            "[relationship_detector] confidence_min in .aelfrice.toml "
            "> 0.5."
        ),
    )
    p_doctor.add_argument(
        "--relationships-max-pairs",
        dest="relationships_max_pairs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --relationships: cap reported pairs after Jaccard "
            "prefilter; deterministic truncation by (id_a, id_b). "
            "Default: [relationship_detector] max_candidate_pairs in "
            ".aelfrice.toml > 5000."
        ),
    )
    p_doctor.add_argument(
        "--detect-stale",
        dest="detect_stale",
        action="store_true",
        default=False,
        help=(
            "scan for sub-confidence contradicting pairs and emit "
            "POTENTIALLY_STALE edges (#387). Direction: newer belief → "
            "older belief. Idempotent. Stdlib only. Bypasses the "
            "hooks/graph checks. Tunes share with --relationships."
        ),
    )
    p_doctor.add_argument(
        "--meta-beliefs",
        dest="meta_beliefs",
        action="store_true",
        default=False,
        help=(
            "report installed meta-belief state (#755): per-key static "
            "default, half-life, surfaced value, and per-signal-class "
            "sub-posteriors. Bypasses the hooks/graph checks. Read-only. "
            "Add --json for machine-readable output."
        ),
    )
    p_doctor.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help=(
            "with --meta-beliefs: emit JSON instead of a human-readable "
            "table. No effect on other doctor sub-modes today."
        ),
    )
    p_doctor.add_argument(
        "--fix",
        dest="fix",
        action="store_true",
        default=False,
        help=(
            "with hooks scope: remove `aelf-*` hook entries whose program "
            "no longer resolves (#781). Custom shell hooks (non-`aelf-` "
            "basenames) are never touched. Combine with --dry-run to "
            "print what would be pruned without rewriting settings.json."
        ),
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    p_sweep_feedback = sub.add_parser(
        "sweep-feedback",
        help=(
            "process deferred retrieval-exposure feedback queue (#191): "
            "apply +epsilon to beliefs whose grace window elapsed without "
            "a contradicting explicit signal"
        ),
    )
    p_sweep_feedback.add_argument(
        "--grace-seconds", type=int, default=None,
        help=(
            "override grace window in seconds. Default: "
            "AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS env > "
            "[implicit_feedback] grace_window_seconds in .aelfrice.toml > "
            "1800"
        ),
    )
    p_sweep_feedback.add_argument(
        "--epsilon", type=float, default=None,
        help=(
            "override per-row alpha increment. Default: "
            "AELFRICE_IMPLICIT_FEEDBACK_EPSILON env > "
            "[implicit_feedback] epsilon in .aelfrice.toml > 0.05"
        ),
    )
    p_sweep_feedback.add_argument(
        "--limit", type=int, default=None,
        help="max queue rows to process per invocation (default 10000)",
    )
    p_sweep_feedback.add_argument(
        "--strict", action="store_true",
        help="exit non-zero on any exception (default: log + exit 0 for cron)",
    )
    p_sweep_feedback.set_defaults(func=_cmd_sweep_feedback)

    # Hidden: one-shot maintenance for pre-migration ghost-α rows
    # (rows whose alpha was inflated by a code path that no longer
    # exists, leaving no audit trail in feedback_history or
    # belief_corroborations). See aelfrice/clamp_ghosts.py.
    p_clamp_ghosts = sub.add_parser("clamp-ghosts", help=argparse.SUPPRESS)
    p_clamp_ghosts.add_argument(
        "--threshold", type=float, default=4.0,
        help="α floor above which a row is a ghost candidate (default: 4.0)",
    )
    p_clamp_ghosts.add_argument(
        "--target", type=float, default=4.0,
        help="α value to clamp matching rows down to (default: 4.0)",
    )
    p_clamp_ghosts.add_argument(
        "--apply", action="store_true",
        help="actually clamp (default: dry-run)",
    )
    p_clamp_ghosts.add_argument(
        "--limit", type=int, default=None,
        help="cap rows processed in one call (default: no cap)",
    )
    p_clamp_ghosts.set_defaults(func=_cmd_clamp_ghosts)

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
    p_setup.add_argument(
        "--transcript-ingest", dest="transcript_ingest",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the four transcript-logger hooks "
            "(UserPromptSubmit, Stop, PreCompact, PostCompact) so live "
            "conversation turns are captured to the per-project transcripts "
            "log and ingested at compaction boundaries. Default: ON. "
            "Pass --no-transcript-ingest to skip."
        ),
    )
    p_setup.add_argument(
        "--rebuilder", action="store_true",
        help=(
            "ALSO install the PreCompact hook for the context "
            "rebuilder. Idempotent; coexists with all other hooks. "
            "Augment-mode only at v1.4.0; suppress mode is parked "
            "for v2.x."
        ),
    )
    p_setup.add_argument(
        "--commit-ingest", dest="commit_ingest",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the PostToolUse:Bash hook so each "
            "successful `git commit` runs the triple extractor on its "
            "commit message and persists the resulting beliefs and "
            "edges under a session derived from git context. Default: ON. "
            "Pass --no-commit-ingest to skip."
        ),
    )
    p_setup.add_argument(
        "--session-start", dest="session_start",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the SessionStart hook so each new host session "
            "opens with L0 locked beliefs already injected. "
            "Coexists with the UserPromptSubmit and transcript-ingest hooks. "
            "Default: ON. Pass --no-session-start to skip."
        ),
    )
    p_setup.add_argument(
        "--stop-hook", dest="stop_hook",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the Stop hook so each session-end prompts to lock any "
            "correction-class beliefs created in this session "
            "(see #582). Coexists with the transcript-ingest Stop entry. "
            "Default: ON. Pass --no-stop-hook to skip. "
            "Set AELF_AUTOLOCK_CORRECTIONS=1 to auto-lock instead of prompt."
        ),
    )
    p_setup.add_argument(
        "--search-tool", dest="search_tool",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the PreToolUse:Grep|Glob hook so the agent's own search "
            "queries first run against the per-project belief store and the "
            "results are injected as additionalContext. If memory has the "
            "answer the agent can skip / refine the tool call; if not, the "
            "tool result fills the gap. Default: ON. Pass --no-search-tool to "
            "skip. See docs/search_tool_hook.md."
        ),
    )
    p_setup.add_argument(
        "--search-tool-bash", dest="search_tool_bash",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "wire the PreToolUse:Bash hook so shell search commands "
            "(grep, rg, find, fd, ack) run against the per-project belief "
            "store before firing. Independent of --search-tool; either, "
            "both, or neither may be installed. Default: ON. Pass "
            "--no-search-tool-bash to skip. See "
            "docs/search_tool_hook.md § Bash extension."
        ),
    )
    p_setup.set_defaults(func=_cmd_setup)

    # Hidden: install lifecycle, surfaced by docs not by --help.
    p_uninstall = sub.add_parser(
        "uninstall",
        help=argparse.SUPPRESS,
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

    # Hidden: statusline command target in settings.json. Not a verb humans invoke.
    p_statusline = sub.add_parser("statusline", help=argparse.SUPPRESS)
    p_statusline.set_defaults(func=_cmd_statusline)

    # `aelf mcp`: start the FastMCP stdio server. Visible in --help so
    # MCP-capable hosts configuring a server entry can discover it;
    # the [mcp] extra must be installed for it to actually run.
    p_mcp = sub.add_parser(
        "mcp",
        help="start the FastMCP stdio server (requires aelfrice[mcp])",
    )
    p_mcp.set_defaults(func=_cmd_mcp)

    # Hidden: the orange statusline banner already prompts users when an
    # update is pending — direct CLI invocation is auxiliary.
    #
    # Renamed `upgrade` -> `upgrade-cmd` at #427 to read advisory (the
    # subcommand prints the upgrade command; it does NOT execute the
    # upgrade itself, since replacing the running package is unreliable).
    # `upgrade` stays as a deprecated alias for one minor — wired as a
    # second parser whose `func` prepends a stderr deprecation notice
    # before delegating to the canonical handler.
    p_upgrade = sub.add_parser(
        "upgrade-cmd",
        help=argparse.SUPPRESS,
    )
    p_upgrade.add_argument(
        "--check", action="store_true",
        help="only report status, do not print the upgrade command line",
    )
    p_upgrade.set_defaults(func=_cmd_upgrade)
    # Deprecated alias. Identical args; func wraps with a stderr notice.
    p_upgrade_alias = sub.add_parser(
        "upgrade",
        help=argparse.SUPPRESS,
    )
    p_upgrade_alias.add_argument(
        "--check", action="store_true",
        help="only report status, do not print the upgrade command line",
    )
    p_upgrade_alias.set_defaults(func=_cmd_upgrade_deprecated_alias)

    # Hidden: scriptable counterpart to setup. Humans use `aelf uninstall`.
    p_unsetup = sub.add_parser("unsetup", help=argparse.SUPPRESS)
    _add_hook_scope_args(p_unsetup)
    p_unsetup.add_argument(
        "--command", default=None,
        help=(
            "exact hook command string to remove. Default: remove every "
            f"entry whose command basename is {DEFAULT_HOOK_COMMAND!r} "
            "(matches both bare-name and absolute-path installs)."
        ),
    )
    p_unsetup.add_argument(
        "--transcript-ingest", dest="transcript_ingest",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the four transcript-logger entries "
            "(UserPromptSubmit, Stop, PreCompact, PostCompact). "
            "Default: ON. Pass --no-transcript-ingest to leave them in place."
        ),
    )
    p_unsetup.add_argument(
        "--rebuilder", action="store_true",
        help="also remove the rebuilder PreCompact hook entry.",
    )
    p_unsetup.add_argument(
        "--commit-ingest", dest="commit_ingest",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the PostToolUse:Bash commit-ingest entry. "
            "Default: ON. Pass --no-commit-ingest to leave it in place."
        ),
    )
    p_unsetup.add_argument(
        "--session-start", dest="session_start",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the SessionStart hook entry. Default: ON. "
            "Pass --no-session-start to leave it in place."
        ),
    )
    p_unsetup.add_argument(
        "--stop-hook", dest="stop_hook",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the Stop hook entry (#582). Default: ON. "
            "Pass --no-stop-hook to leave it in place."
        ),
    )
    p_unsetup.add_argument(
        "--search-tool", dest="search_tool",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the PreToolUse:Grep|Glob search-tool hook entry. "
            "Default: ON. Pass --no-search-tool to leave it in place."
        ),
    )
    p_unsetup.add_argument(
        "--search-tool-bash", dest="search_tool_bash",
        action=argparse.BooleanOptionalAction, default=True,
        help=(
            "remove the PreToolUse:Bash search-tool-bash hook entry. "
            "Default: ON. Pass --no-search-tool-bash to leave it in place."
        ),
    )
    p_unsetup.set_defaults(func=_cmd_unsetup)

    # Hidden: CI regression target. Humans run via `python -m aelfrice.benchmark`
    # if needed.
    p_gate = sub.add_parser("gate", help=argparse.SUPPRESS)
    p_gate.add_argument(
        "gate_verb", nargs="?", default="list",
        help="verb: list (default). Future: ratify (read/write API).",
    )
    p_gate.add_argument(
        "--json", dest="gate_json", action="store_true",
        help="emit machine-readable JSON instead of plain text",
    )
    p_gate.set_defaults(func=_cmd_gate)

    p_bench = sub.add_parser("bench", help=argparse.SUPPRESS)
    p_bench.add_argument(
        "target", nargs="?", default=None,
        help=(
            "benchmark target: synthetic (default), verify-clean, "
            "longmemeval-score, mab, locomo, longmemeval, structmemeval, "
            "amabench, all. See benchmarks/README.md."
        ),
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
    # posterior-residual flags (Slice 1 of #151 eval harness).
    p_bench.add_argument(
        "--fixtures", dest="pr_fixtures", default=None, metavar="PATH",
        help=(
            "(posterior-residual) path to a JSONL fixture file. "
            "Default: benchmarks/posterior_ranking/fixtures/default.jsonl"
        ),
    )
    p_bench.add_argument(
        "--seeds", dest="pr_seeds", type=int, default=None, metavar="N",
        help="(posterior-residual) number of random seeds for multi-seed MRR run (default 5)",
    )
    p_bench.add_argument(
        "--mrr-threshold", dest="pr_mrr_threshold", type=float, default=None,
        metavar="F",
        help="(posterior-residual) MRR uplift pass threshold (default 0.05)",
    )
    p_bench.add_argument(
        "--ece-threshold", dest="pr_ece_threshold", type=float, default=None,
        metavar="F",
        help="(posterior-residual) ECE pass threshold (default 0.10)",
    )
    p_bench.add_argument(
        "--json", dest="pr_json", action="store_true",
        help="(posterior-residual) emit machine-readable JSON instead of human-readable text",
    )
    # `aelf bench all` flags — reproducibility harness (#437).
    p_bench.add_argument(
        "--out", dest="bench_out", default=None, metavar="PATH",
        help="(target=all) merged JSON report path. Required for target=all.",
    )
    p_bench.add_argument(
        "--canonical", dest="bench_canonical", action="store_true",
        help=(
            "(target=all) assert run matches CANONICAL_INVOCATIONS and "
            "write to v2.0.0.json. Mismatched cut → refused."
        ),
    )
    p_bench.add_argument(
        "--adapters", dest="bench_adapters", default=None, metavar="CSV",
        help="(target=all) comma-separated filter (mab,locomo,...).",
    )
    p_bench.add_argument(
        "--smoke", dest="bench_smoke", action="store_true",
        help="(target=all) run SMOKE_INVOCATIONS instead of canonical.",
    )
    p_bench.set_defaults(func=_cmd_bench)

    # `aelf eval` — relevance-calibration harness (#365 R4 Phase B).
    # Operator-facing alias of `audit_rebuild_log.py --calibrate-corpus`.
    p_eval = sub.add_parser(
        "eval",
        help=(
            "run relevance-calibration harness (P@K / ROC-AUC / "
            "Spearman ρ) on a synthetic corpus"
        ),
    )
    p_eval.add_argument(
        "--corpus", dest="eval_corpus", default=None, type=Path,
        metavar="PATH",
        help=(
            "JSONL corpus with (query, known_belief_content, "
            "noise_belief_contents) rows. Default: bundled public "
            "synthetic corpus."
        ),
    )
    p_eval.add_argument(
        "--k", dest="eval_k", type=int, default=10,
        help="K for P@K (default 10).",
    )
    p_eval.add_argument(
        "--seed", dest="eval_seed", type=int, default=0,
        help="deterministic seed for noise-belief shuffle (default 0).",
    )
    p_eval.add_argument(
        "--json", dest="eval_json", action="store_true",
        help="emit machine-readable JSON instead of text block.",
    )
    p_eval.set_defaults(func=_cmd_eval)

    # Hidden: invoked by the CwdChanged hook (HOME repo). Pre-loads the
    # active project's SQLite + OS page caches so the next aelf call
    # pays only the second-hit cost. Silent no-op for unknown / denied
    # paths, debounced to 60s per project. Issue #137.
    p_project_warm = sub.add_parser("project-warm", help=argparse.SUPPRESS)
    p_project_warm.add_argument(
        "path", help="path to warm; resolved to a project root by aelfrice",
    )
    p_project_warm.add_argument(
        "--debounce", type=int, default=None,
        help=(
            "override the debounce window in seconds "
            f"(default {_PROJECT_WARM_DEBOUNCE})"
        ),
    )
    p_project_warm.set_defaults(func=_cmd_project_warm)

    # Hidden: advanced telemetry verb. Invoked by the SessionEnd hook in the
    # HOME repo. Computes per-session deltas from the active store and appends
    # one v=1 row to ~/.aelfrice/telemetry.jsonl. Silent on missing session_id
    # (logs to stderr, exits 0). Issue #140.
    p_session_delta = sub.add_parser("session-delta", help=argparse.SUPPRESS)
    p_session_delta.add_argument(
        "--id", dest="session_id", default=None,
        help="Claude Code session id to compute telemetry for",
    )
    p_session_delta.add_argument(
        "--telemetry-path", dest="telemetry_path", default=None,
        help=(
            "path to the telemetry JSONL file "
            "(default: ~/.aelfrice/telemetry.jsonl)"
        ),
    )
    p_session_delta.set_defaults(func=_cmd_session_delta)

    p_export_canvas = sub.add_parser("export-canvas", help=argparse.SUPPRESS)
    p_export_canvas.add_argument(
        "query", nargs="?", default="",
        help="keyword query for seed selection (omit when using --seed-id)",
    )
    p_export_canvas.add_argument(
        "--seed-id", action="append", default=[], dest="seed_id",
        help="explicit seed belief id (repeatable; bypasses BM25 seed selection)",
    )
    p_export_canvas.add_argument(
        "--k", type=int, default=3,
        help="BM25 seed fanout when --seed-id not given (default 3)",
    )
    p_export_canvas.add_argument(
        "--depth", type=int, default=2,
        help="max BFS hop depth (default 2)",
    )
    p_export_canvas.add_argument(
        "--budget", type=int, default=32,
        help="total expansion-node budget (default 32)",
    )
    p_export_canvas.add_argument(
        "--fanout", type=int, default=8,
        help="max edges expanded per frontier entry (default 8)",
    )
    p_export_canvas.add_argument(
        "--out", default="-",
        help="output path (default '-' for stdout)",
    )
    p_export_canvas.set_defaults(func=_cmd_export_canvas)

    p_tail = sub.add_parser(
        "tail",
        help="live-tail the per-turn hook injection audit log",
    )
    p_tail.add_argument(
        "--filter", action="append", default=None, metavar="key=value",
        help=(
            "filter records: hook=user_prompt_submit | "
            "hook=session_start | lane=L0 | lane=L1. Repeatable; all "
            "filters must match (AND)."
        ),
    )
    p_tail.add_argument(
        "--since", default=None, metavar="DUR",
        help=(
            "backfill records from the last DUR (e.g. 30s, 5m, 2h, 1d) "
            "before tailing live"
        ),
    )
    p_tail.add_argument(
        "--no-blob", dest="no_blob", action="store_true",
        help="suppress per-belief snippet bodies (just header + ids)",
    )
    p_tail.add_argument(
        "--no-follow", dest="no_follow", action="store_true",
        help="dump current audit contents once and exit (no live tail)",
    )
    p_tail.set_defaults(func=_cmd_tail)

    p_scan_deriv = sub.add_parser(
        "scan-derivation",
        help=(
            "N-gram Jaccard similarity gate against a reference document. "
            "Exits 0=clean, 1=matched, 2=reference unreadable."
        ),
    )
    p_scan_deriv.add_argument(
        "--reference",
        default=None,
        metavar="PATH",
        help="path to the reference document to compare against (required)",
    )
    p_scan_deriv.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Jaccard similarity threshold above which an input is flagged (default: 0.6)",
    )
    p_scan_deriv.add_argument(
        "--n",
        type=int,
        default=3,
        metavar="INT",
        help="N-gram width in words (default: 3)",
    )
    p_scan_deriv.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help=(
            "files to check; pass '-' to read stdin. "
            "If omitted, reads stdin as a single input."
        ),
    )
    p_scan_deriv.set_defaults(func=_cmd_scan_derivation)

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
    {"upgrade-cmd", "upgrade", "uninstall", "statusline"}
)

# Commands that already mutate or tear down settings.json themselves.
# Running auto-install before them would either be a wasted no-op
# (setup re-writes the same entries) or actively wrong (uninstall is
# about to remove them). Doctor explicitly inspects on-disk state and
# should see what is there, not what auto-install would write.
_AUTO_INSTALL_SKIP_CMDS: Final[frozenset[str]] = frozenset(
    {"setup", "unsetup", "uninstall", "doctor"}
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
        f"\x1b[38;5;208m{_format_update_banner(status.latest)}\x1b[0m",
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

    ``--advanced`` flag
    -------------------
    ``aelf --advanced`` (or ``aelf --help --advanced``) prints the full help
    output including subcommands that are hidden from the default ``--help``
    view (those registered with ``help=argparse.SUPPRESS``).  ``--advanced``
    is a *help-modifier* flag: it is consumed before argparse sees the rest of
    the argv, and always results in help being printed then a clean exit (0).
    """
    if out is None:
        out = sys.stdout

    # Pre-scan argv for --advanced *before* building the parser.  We consume
    # the flag here rather than registering it with argparse so that it can
    # coexist naturally with --help / -h without argparse complaining about
    # conflicting actions.
    effective_argv: list[str] = list(argv) if argv is not None else sys.argv[1:]
    show_advanced = "--advanced" in effective_argv
    if show_advanced:
        adv_parser = build_parser(show_advanced=True)
        # Print full help to *out* (not stdout) so tests can capture it.
        adv_parser.print_help(file=out)  # type: ignore[arg-type]
        print("", file=out)  # type: ignore[arg-type]
        return 0

    parser = build_parser()
    args = parser.parse_args(effective_argv)
    cmd = getattr(args, "cmd", None)
    if cmd not in _AUTO_INSTALL_SKIP_CMDS:
        # Idempotent post-upgrade hook installer (#623). Gated on a
        # version stamp at ~/.aelfrice/installed-manifest-version, so
        # the happy path on subsequent invocations is a single stat +
        # short file read. Honors AELFRICE_NO_AUTO_INSTALL=1. Never
        # raises — failure is logged to stderr and the command runs
        # regardless.
        auto_install_at_cli_entry(installed_version=_AELFRICE_VERSION)
    if not _update_check_disabled() and cmd not in _UPDATE_CHECK_SKIP_CMDS:
        # Fire-and-forget: cache TTL gates duplicate work, never blocks.
        maybe_check_for_update_async()
    code = int(args.func(args, out))
    _maybe_emit_update_banner(cmd)
    return code
