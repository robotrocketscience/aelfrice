"""Census probes for subsystem cli-setup-hosts-export (A040-A076).

Each probe reads one store copy, or the `--global-dir` copy of
`~/.aelfrice` as the `global` context, read-only and returns integers and
fixed labels only.

Traces, from the code under src/aelfrice:
  A043  cli.py _cmd_delete/_cmd_retire/_cmd_restore insert one
        feedback_history row each, source user_deleted[_force],
        user_retired[_force], user_restored. feedback_history has no FK,
        so the delete row survives the hard delete.
  A049  cli.py _record_lock_ingest: ingest_log source_kind='cli_remember'
        (aelf lock, plus the --promote-user copy in the user store).
  A064  lifecycle.maybe_migrate_to_uv: ~/.aelfrice/migrated-to-uv after a
        successful `uv tool install --force aelfrice`.
  A065  auto_install.maybe_install_manifest (only caller:
        auto_install_at_cli_entry) creates ~/.aelfrice/.auto-install.lock.
        The installed-manifest-version stamp is also written by aelf setup.
  A068  project_warm._write_sentinel: ~/.aelfrice/projects/<id>/.last_warm.
        The sentinel sits beside a legacy per-project store, so the probe
        sees it only on stores passed with `--legacy`; a sentinel for a git
        repository stays outside any store folder.
  A071  cli._sync_setup_opt_outs_and_stamp writes the per-hook `opt_out`
        list of ~/.aelfrice/opt-out-hooks.json (aelf setup --no-X only).
        The `opt_out_hosts` key is also written by setup --host codex and
        by uninstall.
  A075  temporal_spine.maybe_backfill_temporal_spine (called from aelf
        setup): ~/.aelfrice/spine-backfilled. Its edges have the ingest
        writer's exact shape, and `aelf spine backfill` writes no sentinel.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from . import core as cc

# --- A043 ------------------------------------------------------------------

CURATION_SOURCES: tuple[str, ...] = (
    "user_deleted", "user_deleted_force", "user_retired",
    "user_retired_force", "user_restored",
)


def _a043(ctx: cc.Ctx) -> cc.Result:
    if not ctx.table_exists("feedback_history"):
        return cc.silent(0, note="no feedback_history table")
    marks = ",".join("?" * len(CURATION_SOURCES))
    detail = ctx.group_count(
        f"SELECT source, COUNT(*) FROM feedback_history WHERE source IN ({marks}) "
        "GROUP BY source", CURATION_SOURCES, CURATION_SOURCES)
    return cc.from_count(sum(detail.values()), detail,
                         note="aelf feedback --source can forge these labels by hand")


# --- A049 ------------------------------------------------------------------

_LOCK_KEYS = ("derived_beliefs", "still_locked", "tier_frozen", "tier_reference",
              "with_expiry")


def _a049(ctx: cc.Ctx) -> cc.Result:
    if not ctx.table_exists("ingest_log"):
        return cc.silent(0, note="no ingest_log table")
    n = ctx.count("SELECT COUNT(*) FROM ingest_log WHERE source_kind = 'cli_remember'")
    detail: dict[str, int] = {"ingest_rows": n}
    if n and ctx.table_exists("beliefs"):
        tier = "b.lock_tier" if ctx.column_exists("beliefs", "lock_tier") else "NULL"
        exp = "b.lock_expires_at" if ctx.column_exists("beliefs", "lock_expires_at") else "NULL"
        row = ctx.counts(
            "SELECT COUNT(DISTINCT b.id),"
            " COUNT(DISTINCT CASE WHEN b.lock_level = 'user' THEN b.id END),"
            f" COUNT(DISTINCT CASE WHEN b.lock_level = 'user' AND {tier} = 'frozen' THEN b.id END),"
            f" COUNT(DISTINCT CASE WHEN b.lock_level = 'user' AND {tier} = 'reference' THEN b.id END),"
            f" COUNT(DISTINCT CASE WHEN b.lock_level = 'user' AND {exp} IS NOT NULL THEN b.id END)"
            " FROM ingest_log il, json_each(il.derived_belief_ids) je"
            " JOIN beliefs b ON b.id = je.value"
            " WHERE il.source_kind = 'cli_remember' AND json_valid(il.derived_belief_ids)")
        detail.update(zip(_LOCK_KEYS, row))
    return cc.from_count(n, detail, note="user store rows include promote-user copies")


# --- A064 / A065 / A075: install-state sentinels under ~/.aelfrice ---------


def _sentinel(name: str, note: str = "") -> Callable[[cc.Ctx], cc.Result]:
    def fn(ctx: cc.Ctx) -> cc.Result:
        return cc.from_count(1 if ctx.global_file(name).is_file() else 0, note=note)
    fn.__name__ = "sentinel_" + name.replace("-", "_").replace(".", "_")
    return fn


def _a065(ctx: cc.Ctx) -> cc.Result:
    lock = 1 if ctx.global_file(".auto-install.lock").is_file() else 0
    stamp = 1 if ctx.global_file("installed-manifest-version").is_file() else 0
    return cc.from_count(lock, {"lock_file": lock, "stamp_file": stamp},
                         note="stamp_file is also written by aelf setup; lock_file is not")


# --- A068 ------------------------------------------------------------------


def _a068(ctx: cc.Ctx) -> cc.Result:
    n = 1 if ctx.sidecar(".last_warm").is_file() else 0
    return cc.from_count(n, note="seen only beside legacy per-project stores")


# --- A071 ------------------------------------------------------------------

MANIFEST_HOOKS: tuple[str, ...] = (
    "user_prompt_submit_retrieval", "transcript_ingest", "commit_ingest",
    "session_start", "stop_lock_prompt", "search_tool", "search_tool_bash",
    "pre_issue_guard", "claude_memory_mirror", "agent_context",
)
HOSTS: tuple[str, ...] = ("claude", "codex")


def _str_list(doc: dict[str, Any], key: str) -> list[str]:
    v = doc.get(key)
    if not isinstance(v, list):
        return []
    return [h for h in v if isinstance(h, str)]  # pyright: ignore[reportUnknownVariableType]


def _a071(ctx: cc.Ctx) -> cc.Result:
    p = ctx.global_file("opt-out-hooks.json")
    if not p.is_file():
        return cc.silent(0)
    try:
        doc: Any = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return cc.silent(0, {"unparseable": 1})
    if not isinstance(doc, dict):
        return cc.silent(0, {"unparseable": 1})
    hooks = _str_list(doc, "opt_out")  # pyright: ignore[reportUnknownArgumentType]
    hosts = _str_list(doc, "opt_out_hosts")  # pyright: ignore[reportUnknownArgumentType]
    detail: dict[str, int] = {}
    for h in hooks:
        k = "hook:" + (h if h in MANIFEST_HOOKS else "other")
        detail[k] = detail.get(k, 0) + 1
    for h in hosts:
        k = "host:" + (h if h in HOSTS else "other")
        detail[k] = detail.get(k, 0) + 1
    return cc.from_count(len(hooks), detail,
                         note="count is per-hook opt-outs; host entries also come from uninstall")


# --- registry --------------------------------------------------------------

_U = cc.untraceable_probe

PROBES: list[cc.Probe] = [
    _U("A040", "aelf bench synthetic harness and bench all",
       "runs on an in-memory store or a --db file the user names; writes no row to a real store"),
    _U("A041", "aelf context source-turn recovery",
       "recover_context only reads; opens the store but writes no row or sidecar"),
    _U("A042", "aelf core load-bearing lens",
       "opens through open_store_for_read and writes nothing"),
    cc.Probe("A043", "aelf delete, retire, restore", "store", _a043),
    _U("A044", "aelf doctor --backfill-ingest",
       "ingest_jsonl with label transcript writes the same rows as ingest-transcript and "
       "the Stop flush; no field tells them apart"),
    _U("A045", "aelf doctor --meta-beliefs and --hot-path",
       "read-only reports over meta_beliefs and belief_touches; no write"),
    _U("A046", "aelf eval relevance-calibration harness",
       "scores a fixture corpus in a throwaway store; never opens the user store"),
    _U("A047", "aelf graph DOT/JSON viewer",
       "opens read-only; output goes to stdout or a file the user names"),
    _U("A048", "aelf label corpus labelling CLI",
       "dev tool; appends to a user-named JSONL outside any store"),
    cc.Probe("A049", "aelf lock derivation (tiers, expiry, doc anchor, category)",
             "store", _a049),
    _U("A050", "aelf rebuild manual trigger",
       "rebuild_v14 gets no rebuild_log_path, so no rebuild_logs row; stdout only"),
    _U("A051", "aelf search CLI",
       "opens read-only; no feedback, injection or last_retrieved_at write"),
    _U("A052", "aelf setup --rebuilder / PreCompact hook",
       "entry lives in host settings.json outside any store; the handler only resets the "
       "injection epoch, which SessionStart also does"),
    _U("A053", "aelf show", "opens through open_store_for_read and writes nothing"),
    _U("A054", "aelf speculative listing", "opens through open_store_for_read and writes nothing"),
    _U("A055", "aelf stale listing", "opens through open_store_for_read and writes nothing"),
    _U("A056", "aelf status and stats/health aliases",
       "status and stats open read-only; health prints counts and writes nothing"),
    _U("A057", "aelf uninstall / unsetup",
       "removes files and settings entries; its one write (host opt-out) is also "
       "written by setup --host codex"),
    _U("A058", "aelf upgrade deprecated alias",
       "prints a rename warning and an upgrade command; writes nothing"),
    _U("A059", "Background PyPI update check and banners",
       "writes update_check.json under HOME .cache, outside any store or global dir"),
    _U("A060", "Codex $aelf-* agent skills",
       "skills land under HOME .agents, outside any store or global dir"),
    _U("A061", "Codex host hooks.json setup",
       "writes CODEX_HOME hooks.json outside any store; codex hook fires write the same "
       "store rows as other host hooks with no host field"),
    _U("A062", "export-canvas exporter",
       "opens the store but writes only the canvas JSON to stdout or a user path"),
    _U("A063", "export-obsidian vault exporter",
       "writes Markdown notes into the user vault; no store write"),
    cc.Probe("A064", "Migrate install to uv tool at setup", "global-log",
             _sentinel("migrated-to-uv")),
    cc.Probe("A065", "Post-upgrade hook auto-install at CLI entry", "global-log", _a065),
    _U("A066", "Pre-issue duplicate guard (firing condition and bypass)",
       "guard writes no log row; the outcome is only an exit code and stderr"),
    _U("A067", "Pre-issue guard Jaccard 0.5 and candidate search",
       "guard writes no log row; scores and candidates are never persisted"),
    cc.Probe("A068", "project-warm (CwdChanged pre-load)", "store", _a068),
    _U("A069", "Settings transaction lock and stale-hook prune at setup",
       "lock file and atomic write sit beside host settings.json outside any store"),
    _U("A070", "Setup next-step and historical-JSONL hints", "hints go to stdout only"),
    cc.Probe("A071", "Setup opt-out sync and host opt-out", "global-log", _a071),
    _U("A072", "Setup scope auto-detect and hook command resolution",
       "result lives in host settings.json command strings outside any store"),
    _U("A073", "Slash-command bundle install and prune",
       "commands land under the host commands dir, outside any store or global dir"),
    _U("A074", "Statusline install at setup",
       "statusLine entry lives in host settings.json outside any store"),
    cc.Probe("A075", "Temporal-spine auto-backfill at setup and aelf spine backfill/clear",
             "global-log",
             _sentinel("spine-backfilled",
                       note="sentinel marks the setup auto-backfill only; manual spine "
                            "backfill edges match the ingest writer")),
    _U("A076", "upgrade-cmd advisory upgrade",
       "prints the upgrade command; check mode refreshes update_check.json under HOME .cache"),
]

TRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind != "untraceable")
UNTRACEABLE: frozenset[str] = frozenset(p.item_id for p in PROBES if p.kind == "untraceable")
