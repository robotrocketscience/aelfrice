"""Codex host target for setup / doctor / uninstall (#1052).

Writes, validates, and removes the aelfrice hook set in Codex's
``~/.codex/hooks.json``. Design constraints established in #1054/#1052
triage:

- **hooks.json only.** The documented ``{"hooks": {"<Event>": [...]}}``
  shape is the stable public surface. The per-hook trust ledger
  (``[hooks.state]`` tables in ``config.toml``, keyed positionally with
  a ``trusted_hash`` over a canonical TOML serialization) is explicitly
  marked for replacement in the Codex source — we never write it.
  Setup instead prints approval guidance: the user runs ``/hooks``
  inside a Codex session to trust the new entries. Until approved,
  Codex silently skips the hooks. The ``hooks`` feature itself is
  stable and on by default (it was the under-development ``codex_hooks``
  flag on Codex 0.11x–0.12x); doctor surfaces an explicit disable.
- **Merge-aware and idempotent.** Entries whose command basename is one
  of ours are replaced wholesale on every setup run; everything else in
  the file is preserved byte-for-byte at the JSON level. An unparseable
  hooks.json is never overwritten without ``force`` — a real-world
  ``~/.codex/hooks.json`` has been observed holding truncated JSON, and
  clobbering user content on a parse error is worse than refusing.
- **Portable hook subset.** Host-agnostic hooks are installed:
  retrieval injection (UserPromptSubmit), the transcript logger
  (UserPromptSubmit / Stop / PreCompact / PostCompact), session-start
  baseline injection (SessionStart, all sources — ``compact`` included,
  which is the rebuild-at-compaction channel per #1054), the stop
  lock-prompt, and the ``Bash``-matcher tool hooks (#1055): Codex
  canonicalizes hook tool names to the compatible surface — shell
  commands report ``tool_name == "Bash"`` — so the memory-first shell
  search, pre-issue duplicate guard, and commit-ingest hooks match
  unchanged. The ``Grep|Glob`` search hook is excluded (no such tools
  exist on Codex; greps arrive via Bash and are covered by the Bash
  matcher), as is the host-specific memory mirror.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from aelfrice.setup import (
    SettingsScope,
    resolve_commit_ingest_command,
    resolve_hook_command,
    resolve_pre_issue_guard_command,
    resolve_search_tool_bash_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
)

CODEX_DIR: Final[Path] = Path.home() / ".codex"
CODEX_HOOKS_FILENAME: Final[str] = "hooks.json"
CODEX_CONFIG_FILENAME: Final[str] = "config.toml"

# SessionStart matcher covering every source, so the baseline block fires
# on fresh sessions and the rebuild block fires post-compaction (#1054:
# SessionStart(source=="compact") is Codex's only compaction-injection
# channel — PreCompact/PostCompact stdout is ignored by the harness).
_SESSION_START_MATCHER: Final[str] = "startup|resume|clear|compact"

# Basenames owned by aelfrice; setup replaces exactly these on re-run and
# uninstall removes exactly these. Mirrors the manifest-ownership posture
# of auto_install (additive within owned basenames).
_OWNED_BASENAMES: Final[frozenset[str]] = frozenset({
    "aelf-hook",
    "aelf-transcript-logger",
    "aelf-session-start-hook",
    "aelf-stop-hook",
    "aelf-search-tool-hook",
    "aelf-pre-issue-hook",
    "aelf-commit-ingest",
})


def codex_hooks_path(codex_dir: Path | None = None) -> Path:
    return (codex_dir if codex_dir is not None else CODEX_DIR) / CODEX_HOOKS_FILENAME


def codex_config_path(codex_dir: Path | None = None) -> Path:
    return (codex_dir if codex_dir is not None else CODEX_DIR) / CODEX_CONFIG_FILENAME


def _handler(command: str, *, timeout: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"type": "command", "command": command}
    if timeout is not None:
        entry["timeout"] = timeout
    return entry


def desired_codex_hooks(scope: SettingsScope = "user") -> dict[str, list[dict[str, object]]]:
    """The aelfrice hook set in Codex hooks.json shape, keyed by event.

    Commands resolve to absolute paths via the same resolvers the Claude
    installers use, so both hosts pin identical executables.
    """
    hook_cmd = resolve_hook_command(scope)
    logger_cmd = resolve_transcript_logger_command(scope)
    session_cmd = resolve_session_start_hook_command(scope)
    stop_cmd = resolve_stop_hook_command(scope)
    search_bash_cmd = resolve_search_tool_bash_command(scope)
    pre_issue_cmd = resolve_pre_issue_guard_command(scope)
    commit_cmd = resolve_commit_ingest_command(scope)
    return {
        "UserPromptSubmit": [
            {"hooks": [_handler(hook_cmd), _handler(logger_cmd)]},
        ],
        "Stop": [
            {"hooks": [_handler(logger_cmd), _handler(stop_cmd)]},
        ],
        "PreCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "PostCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "SessionStart": [
            {
                "matcher": _SESSION_START_MATCHER,
                "hooks": [_handler(session_cmd)],
            },
        ],
        # #1055: Codex reports shell commands as tool_name "Bash", so the
        # Bash-matcher hooks are host-portable verbatim. Grep|Glob is
        # omitted — those tools do not exist on Codex.
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [_handler(search_bash_cmd), _handler(pre_issue_cmd)],
            },
        ],
        "PostToolUse": [
            {"matcher": "Bash", "hooks": [_handler(commit_cmd)]},
        ],
    }


def _command_basename(handler: object) -> str:
    """Basename of a handler's command's first token, '' on shape miss."""
    if not isinstance(handler, dict):
        return ""
    hd = cast(dict[str, object], handler)
    cmd = hd.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return ""
    return Path(cmd.split()[0]).name


def _group_is_owned(group: object) -> bool:
    """A matcher group is aelfrice's iff every handler in it is ours.

    Mixed groups (user handler + aelfrice handler in one group) are left
    untouched — we never edit inside someone else's group.
    """
    if not isinstance(group, dict):
        return False
    gd = cast(dict[str, object], group)
    handlers = gd.get("hooks")
    if not isinstance(handlers, list) or not handlers:
        return False
    return all(
        _command_basename(h) in _OWNED_BASENAMES
        for h in cast(list[object], handlers)
    )


def claude_host_has_aelfrice_hooks(settings_path: Path) -> bool:
    """True iff the Claude-host settings.json wires any aelfrice hook.

    Used by `aelf setup --host codex` (#1053) to distinguish a
    Codex-only machine (write the claude auto-install opt-out) from a
    dual-host one (leave auto-install alone). Shape-tolerant and
    fail-closed: a missing or unreadable settings file counts as "no
    hooks" — the worst case of a false negative is an opt-out the user
    can undo with one explicit `aelf setup`.
    """
    if not settings_path.is_file():
        return False
    try:
        parsed = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    hooks = cast(dict[str, object], parsed).get("hooks")
    if not isinstance(hooks, dict):
        return False
    for groups in cast(dict[str, object], hooks).values():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            if not isinstance(group, dict):
                continue
            gd = cast(dict[str, object], group)
            for handler in cast(list[object], gd.get("hooks", []) or []):
                if _command_basename(handler).startswith("aelf-"):
                    return True
    return False


@dataclass
class CodexInstallResult:
    path: Path
    changed: bool
    installed_events: list[str] = field(default_factory=list[str])
    guidance: list[str] = field(default_factory=list[str])
    error: str | None = None


def install_codex_hooks(
    hooks_path: Path,
    *,
    scope: SettingsScope = "user",
    force: bool = False,
) -> CodexInstallResult:
    """Write the aelfrice hook set into ``hooks_path``, merge-aware.

    Refuses (with ``error`` set) when the existing file is unparseable
    and ``force`` is False; ``force`` replaces the broken file with a
    fresh aelfrice-only document.
    """
    existing: dict[str, object] = {}
    if hooks_path.is_file():
        try:
            parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = cast(dict[str, object], parsed)
            elif not force:
                return CodexInstallResult(
                    path=hooks_path, changed=False,
                    error="existing hooks.json is not a JSON object; "
                          "re-run with --force to replace it",
                )
        except json.JSONDecodeError as exc:
            if not force:
                return CodexInstallResult(
                    path=hooks_path, changed=False,
                    error=f"existing hooks.json is invalid JSON ({exc}); "
                          "re-run with --force to replace it",
                )
            existing = {}

    hooks_obj = existing.get("hooks")
    hooks_map: dict[str, object] = (
        cast(dict[str, object], hooks_obj) if isinstance(hooks_obj, dict) else {}
    )

    desired = desired_codex_hooks(scope)
    before = json.dumps({"hooks": hooks_map}, sort_keys=True)
    for event, groups in desired.items():
        current = hooks_map.get(event)
        current_list = (
            cast(list[object], current) if isinstance(current, list) else []
        )
        kept = [g for g in current_list if not _group_is_owned(g)]
        hooks_map[event] = kept + cast(list[object], groups)
    existing["hooks"] = hooks_map
    after = json.dumps({"hooks": hooks_map}, sort_keys=True)

    changed = before != after or not hooks_path.is_file()
    if changed:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8",
        )
    return CodexInstallResult(
        path=hooks_path,
        changed=changed,
        installed_events=sorted(desired.keys()),
        guidance=[
            (
                "Codex runs a hook only after per-hook trust approval: open "
                + "a Codex session and run /hooks to approve the new entries "
                + "(automation that vets its own hook sources may pass "
                + "--dangerously-bypass-hook-trust instead)."
            ),
            (
                "The Codex `hooks` feature is stable and enabled by default; "
                + "no action is needed unless you disabled it "
                + "([features].hooks = false in config.toml)."
            ),
        ],
    )


def remove_codex_hooks(hooks_path: Path) -> CodexInstallResult:
    """Remove aelfrice-owned matcher groups; drop emptied events.

    A missing or unparseable file is reported, not modified — uninstall
    never destroys content it cannot positively identify as ours.
    """
    if not hooks_path.is_file():
        return CodexInstallResult(path=hooks_path, changed=False)
    try:
        parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CodexInstallResult(
            path=hooks_path, changed=False,
            error=f"hooks.json is invalid JSON ({exc}); not modified",
        )
    if not isinstance(parsed, dict):
        return CodexInstallResult(
            path=hooks_path, changed=False,
            error="hooks.json is not a JSON object; not modified",
        )
    doc = cast(dict[str, object], parsed)
    hooks_obj = doc.get("hooks")
    if not isinstance(hooks_obj, dict):
        return CodexInstallResult(path=hooks_path, changed=False)
    hooks_map = cast(dict[str, object], hooks_obj)

    changed = False
    removed_events: list[str] = []
    for event in list(hooks_map.keys()):
        groups = hooks_map[event]
        if not isinstance(groups, list):
            continue
        kept = [g for g in cast(list[object], groups) if not _group_is_owned(g)]
        if len(kept) != len(cast(list[object], groups)):
            changed = True
            removed_events.append(event)
            if kept:
                hooks_map[event] = kept
            else:
                del hooks_map[event]
    if changed:
        hooks_path.write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8",
        )
    return CodexInstallResult(
        path=hooks_path, changed=changed, installed_events=sorted(removed_events),
    )


# --- Codex agent skills (the `$aelf-*` port of `/aelf:*`) ---------------
#
# Codex's analogue of an `/aelf:*` slash command is an *agent skill*: a
# directory holding a ``SKILL.md`` (name + description frontmatter, then
# natural-language instructions), discovered from the user scope
# ``~/.agents/skills/`` and invoked explicitly as ``$<name>`` or triggered
# implicitly when a task matches the description. Custom prompts
# (``~/.codex/prompts``) are the closer 1:1 to a slash file but are
# deprecated upstream in favour of skills, so we target skills.
#
# The source of truth is the SAME bundle the Claude installer ships
# (``src/aelfrice/slash_commands/*.md``). Each file is transformed on
# install — no second copy is maintained, so editing the slash file
# updates both hosts. The transform: rename ``aelf:foo`` -> ``aelf-foo``
# (colons are invalid in skill/dir names), reduce the frontmatter to the
# required ``name``/``description`` pair, and prepend a short adapter
# preamble that (a) defines ``$ARGUMENTS`` for a host with no positional
# substitution engine and (b) maps the host-specific ``Task`` fan-out
# tool onto Codex's equivalent mechanism (``Task`` is the only tool name
# the adapter maps). The ``<objective>``/``<process>`` body is carried
# over verbatim — except that the host-management commands (setup /
# doctor / uninstall / upgrade) additionally get a ``<host-adapter>``
# note steering their ``aelf`` invocations to the ``--host codex`` form
# (#1136), since the bare form targets another host's configuration.

# Codex USER-scope skill discovery root (the open agent-skills standard
# path, shared with other agents' skills — hence the marker-gated prune).
AGENTS_SKILLS_DIR: Final[Path] = Path.home() / ".agents" / "skills"

# Every generated SKILL.md carries this marker on its first body line. It
# is the prune safety key: uninstall / orphan-prune only ever removes an
# ``aelf-*`` skill directory whose SKILL.md contains this exact marker, so
# a user's hand-authored ``aelf-*`` skill is never destroyed.
_SKILL_MARKER: Final[str] = "AELFRICE-CODEX-SKILL"
_SKILL_PREFIX: Final[str] = "aelf-"
_SKILL_FILENAME: Final[str] = "SKILL.md"

# Host-management commands (#1136). Their bundled bodies instruct bare
# ``aelf setup`` / ``aelf doctor`` / ``aelf unsetup`` / ``aelf uninstall``
# runs, which on this host would install, scan, or tear down ANOTHER
# host's configuration (settings-file hooks, statusline, slash bundle)
# instead of ``~/.codex/hooks.json`` + the ``$aelf-*`` skills. Their
# generated skills carry an adapter note steering every such invocation
# to the ``--host codex`` form.
_HOST_MANAGEMENT_SKILLS: Final[frozenset[str]] = frozenset({
    "aelf-setup",
    "aelf-doctor",
    "aelf-uninstall",
    "aelf-upgrade",
})

# The bundled ``setup`` description names another host's artifacts
# (settings file + statusline snippet). Describe the codex-host effect
# instead, so implicit skill triggering matches what the command
# actually does here (#1136).
_SETUP_DESCRIPTION_OVERRIDE: Final[str] = (
    "Install the aelfrice hooks in ~/.codex/hooks.json and the $aelf-* "
    "agent skills under ~/.agents/skills/ on this host."
)

_HOST_MANAGEMENT_NOTE: Final[str] = (
    "<host-adapter>\n"
    "IMPORTANT — on this host, every `aelf setup`, `aelf doctor`,\n"
    "`aelf unsetup`, or `aelf uninstall` invocation in the steps below\n"
    "MUST use the `--host codex` form (e.g. `uv run aelf setup --host "
    "codex`,\n"
    "`uv run aelf doctor --host codex`, `uv run aelf unsetup --host "
    "codex`,\n"
    "`uv run aelf uninstall <flags> --host codex`). The bare form\n"
    "targets another host's configuration — it would not touch this\n"
    "host's install and must not be run here.\n"
    "</host-adapter>"
)


def _parse_slash_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a slash-command file into (frontmatter, body).

    Frontmatter values in the bundle are single-line ``key: value`` pairs
    (the only multi-line key, ``allowed-tools``, is a list we discard), so
    a line-based parse is exact and dependency-free. Returns the scalar
    keys we consume (``name``, ``description``, ``argument-hint``) and the
    body text after the closing delimiter (verbatim, stripped of a single
    leading newline).
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    # lines[0] == "---"; find the closing delimiter.
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    front: dict[str, str] = {}
    for raw in lines[1:close]:
        if raw[:1] in (" ", "\t") or ":" not in raw:
            # Indented list item (allowed-tools entries) or blank — skip.
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        if key in ("name", "description", "argument-hint"):
            front[key] = val.strip()
    body = "\n".join(lines[close + 1:])
    return front, body.lstrip("\n")


def codex_skill_from_slash(filename: str, text: str) -> tuple[str, str]:
    """Transform one slash-command file into a (skill_name, SKILL.md) pair.

    ``filename`` is the bundle basename (e.g. ``search.md``); ``text`` is
    its full content. Returns the skill directory / ``name`` value (e.g.
    ``aelf-search``) and the rendered SKILL.md text. Deterministic: same
    input bytes -> same output bytes.
    """
    front, body = _parse_slash_frontmatter(text)
    raw_name = front.get("name") or (_SKILL_PREFIX + filename.removesuffix(".md"))
    skill_name = raw_name.replace(":", "-")
    slash_name = raw_name  # original slash form, e.g. "aelf:search"
    description = front.get("description", "")
    if skill_name == "aelf-setup":
        # #1136: the bundled description names another host's artifacts.
        description = _SETUP_DESCRIPTION_OVERRIDE

    adapter: list[str] = [
        f"<!-- {_SKILL_MARKER}: auto-generated from "
        f"src/aelfrice/slash_commands/{filename}. Edit the source file, "
        "not this copy. -->",
        "",
        f"This is the Codex port of the `/{slash_name}` slash "
        f"command; invoke it as `${skill_name}`.",
    ]
    hint = front.get("argument-hint")
    if hint:
        adapter.append(f"Arguments: {hint}")
    if "$ARGUMENTS" in body:
        adapter.append(
            "Where the steps below reference `$ARGUMENTS`, substitute the "
            f"text the user typed after `${skill_name}` (their query and/or "
            "flags)."
        )
    if "Task" in body and ("subagent" in body or "Task tool" in body or "Task subagent" in body):
        adapter.append(
            "Where the steps mention the host's `Task` tool / subagents, "
            "use Codex's own subagent mechanism to fan out the equivalent "
            "work; the dispatch logic and CLI calls are unchanged."
        )
        if filename == "onboard.md":
            # onboard's classification fan-out defaults to a low-cost
            # model tier and lets the user pick the tier at run time; it
            # is tier-abstract, not pinned to a host-specific model (the
            # slash body stopped naming one in #1155). On Codex the
            # cheapest fast tier is a `-mini`-class model; resolve the
            # low-cost default there rather than let it fall through to
            # the session's default model — the expensive, slow path for
            # short-label classification. No model id pinned (names drift).
            adapter.append(
                "This skill's classification step defaults to a low-cost "
                "model tier for the fan-out and lets the user pick the tier "
                "at run time; it no longer names a host-specific model. On "
                "Codex, resolve that low-cost default to Codex's cheapest "
                "fast model tier (a `-mini`-class model) — not the session's "
                "default model, which is more expensive and slower for bulk "
                "short-label classification. Use that same model name where a "
                "step renders or prices the classifier model."
            )
    if skill_name in _HOST_MANAGEMENT_SKILLS:
        adapter.append(_HOST_MANAGEMENT_NOTE)
    adapter.append(
        "Run each `uv run aelf ...` command in your shell and show its "
        "output to the user."
    )

    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {description}",
        "---",
        *adapter,
        "",
        body.rstrip("\n"),
        "",
    ]
    return skill_name, "\n".join(lines)


def _bundled_codex_skills() -> dict[str, str]:
    """Map skill_name -> SKILL.md text for every bundled slash command."""
    from aelfrice.setup import bundled_slash_files

    result: dict[str, str] = {}
    for filename, text in bundled_slash_files().items():
        skill_name, skill_text = codex_skill_from_slash(filename, text)
        result[skill_name] = skill_text
    return result


def _is_owned_skill_dir(skill_dir: Path) -> bool:
    """True iff ``skill_dir`` is an aelfrice-generated skill we may prune.

    Gated on both the ``aelf-`` name prefix AND the marker inside its
    SKILL.md, so a user's own ``aelf-*`` skill (no marker) is left alone.
    """
    if not skill_dir.name.startswith(_SKILL_PREFIX) or not skill_dir.is_dir():
        return False
    skill_md = skill_dir / _SKILL_FILENAME
    if not skill_md.is_file():
        return False
    try:
        return _SKILL_MARKER in skill_md.read_text(encoding="utf-8")
    except OSError:
        return False


@dataclass(frozen=True)
class CodexSkillsResult:
    """Outcome of install/uninstall of the Codex ``$aelf-*`` skills.

    ``skipped`` (#1136): bundled skill names whose on-disk collision is
    an unmarked (non-aelfrice) skill — never overwritten. ``failed``
    (#1136): human-readable ``"<name>: <reason>"`` rows for partial
    removals and other FS errors that previously vanished silently.
    """

    dest_dir: Path
    written: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def _remove_owned_skill_dir(
    child: Path, pruned: list[str], failed: list[str],
) -> None:
    """Remove one owned skill dir: unlink its SKILL.md, then rmdir.

    The two steps are split (#1136) so a half-removal is visible: a
    failed unlink records the skill under ``failed`` and stops; a
    successful unlink followed by a failed rmdir (routine case: a stray
    extra file — e.g. OS metadata — keeps the directory non-empty)
    counts the skill as pruned (its SKILL.md is gone) AND records the
    leftover directory under ``failed``. Nothing is deleted recursively.
    """
    try:
        (child / _SKILL_FILENAME).unlink()
    except OSError as exc:
        failed.append(f"{child.name}: could not remove SKILL.md ({exc})")
        return
    try:
        child.rmdir()
    except OSError as exc:
        pruned.append(child.name)
        failed.append(
            f"{child.name}: SKILL.md removed but directory left in "
            f"place ({exc})"
        )
    else:
        pruned.append(child.name)


def install_codex_skills(dest_dir: Path | None = None) -> CodexSkillsResult:
    """Write every bundled command as a Codex skill under ``dest_dir``.

    Default ``dest_dir`` is ``~/.agents/skills/``. Each skill lands at
    ``<dest>/aelf-<cmd>/SKILL.md``. Idempotent (byte-identical files are
    skipped), atomic (temp + ``os.replace``), and orphan-pruning — but
    both the replace path and pruning are marker-gated: only
    marker-carrying ``aelf-*`` skill dirs are ever overwritten or
    removed, never the other skills that share this directory (#1136).
    """
    import os
    import tempfile

    target = dest_dir if dest_dir is not None else AGENTS_SKILLS_DIR
    bundle = _bundled_codex_skills()

    written: list[str] = []
    already: list[str] = []
    pruned: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    target.mkdir(parents=True, exist_ok=True)

    for skill_name, text in sorted(bundle.items()):
        skill_dir = target / skill_name
        dest_file = skill_dir / _SKILL_FILENAME
        if dest_file.is_file():
            try:
                if dest_file.read_text(encoding="utf-8") == text:
                    already.append(skill_name)
                    continue
            except OSError:
                # Unreadable existing file: ownership cannot be verified
                # either, so the marker gate below fails closed (skip).
                pass
            # Bytes differ: the replace path is marker-gated exactly
            # like prune/remove (#1136) — a colliding skill without our
            # marker is someone else's file and is never overwritten.
            if not _is_owned_skill_dir(skill_dir):
                skipped.append(skill_name)
                continue
        skill_dir.mkdir(parents=True, exist_ok=True)
        encoded = text.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(
            prefix=_SKILL_FILENAME + ".", suffix=".tmp", dir=str(skill_dir)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest_file)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        written.append(skill_name)

    # Prune orphans: marker-carrying aelf-* dirs no longer in the bundle
    # (handles renames/removals). Non-aelfrice skills are never touched.
    for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
        if child.name in bundle:
            continue
        if _is_owned_skill_dir(child):
            _remove_owned_skill_dir(child, pruned, failed)

    return CodexSkillsResult(
        dest_dir=target,
        written=tuple(written),
        already=tuple(already),
        pruned=tuple(pruned),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )


def remove_codex_skills(dest_dir: Path | None = None) -> CodexSkillsResult:
    """Remove all aelfrice-generated ``$aelf-*`` skills from ``dest_dir``.

    Only marker-carrying ``aelf-*`` skill directories are removed; other
    skills sharing the directory are preserved. Returns the removed skill
    names under ``pruned``.
    """
    target = dest_dir if dest_dir is not None else AGENTS_SKILLS_DIR
    pruned: list[str] = []
    failed: list[str] = []
    if target.is_dir():
        for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
            if _is_owned_skill_dir(child):
                _remove_owned_skill_dir(child, pruned, failed)
    return CodexSkillsResult(
        dest_dir=target, pruned=tuple(pruned), failed=tuple(failed),
    )


def count_installed_codex_skills(dest_dir: Path | None = None) -> int:
    """Count marker-carrying ``aelf-*`` skills present under ``dest_dir``."""
    target = dest_dir if dest_dir is not None else AGENTS_SKILLS_DIR
    if not target.is_dir():
        return 0
    return sum(
        1 for child in target.glob(f"{_SKILL_PREFIX}*")
        if _is_owned_skill_dir(child)
    )


@dataclass
class CodexDoctorReport:
    """Structured result of the Codex host scan; render at the CLI."""

    codex_dir_present: bool
    hooks_file_present: bool = False
    hooks_file_valid: bool = False
    parse_error: str | None = None
    owned_handler_count: int = 0
    missing_events: list[str] = field(default_factory=list[str])
    stale_commands: list[str] = field(default_factory=list[str])
    feature_flag_on: bool | None = None
    approved_state_count: int = 0
    warnings: list[str] = field(default_factory=list[str])


def doctor_codex(codex_dir: Path | None = None) -> CodexDoctorReport:
    """Scan the Codex host: hooks.json shape, coverage, flag, trust.

    Read-only. Reports rather than raises; the CLI decides exit codes.
    """
    cdir = codex_dir if codex_dir is not None else CODEX_DIR
    report = CodexDoctorReport(codex_dir_present=cdir.is_dir())
    if not report.codex_dir_present:
        report.warnings.append(f"{cdir} not found — Codex not installed?")
        return report

    hooks_path = codex_hooks_path(cdir)
    report.hooks_file_present = hooks_path.is_file()
    hooks_map: dict[str, object] = {}
    if report.hooks_file_present:
        try:
            parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                report.hooks_file_valid = True
                obj = cast(dict[str, object], parsed).get("hooks")
                if isinstance(obj, dict):
                    hooks_map = cast(dict[str, object], obj)
            else:
                report.parse_error = "top-level JSON is not an object"
        except json.JSONDecodeError as exc:
            report.parse_error = str(exc)
        if not report.hooks_file_valid:
            report.warnings.append(
                f"{hooks_path} is unreadable as hooks config "
                f"({report.parse_error}); Codex will ignore or reject it",
            )

    expected_events = set(desired_codex_hooks().keys())
    covered: set[str] = set()
    for event, groups in hooks_map.items():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            if not _group_is_owned(group):
                continue
            covered.add(event)
            gd = cast(dict[str, object], group)
            for handler in cast(list[object], gd.get("hooks", [])):
                report.owned_handler_count += 1
                hd = cast(dict[str, object], handler)
                cmd = hd.get("command")
                if isinstance(cmd, str):
                    exe = Path(cmd.split()[0])
                    if (
                        exe.is_absolute()
                        and not exe.exists()
                        and cmd not in report.stale_commands
                    ):
                        report.stale_commands.append(cmd)
    report.missing_events = sorted(expected_events - covered)
    for cmd in report.stale_commands:
        report.warnings.append(f"hook command not found on disk: {cmd}")
    if report.owned_handler_count and report.missing_events:
        report.warnings.append(
            "aelfrice hook coverage incomplete; missing events: "
            + ", ".join(report.missing_events),
        )

    config_path = codex_config_path(cdir)
    if config_path.is_file():
        try:
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            report.warnings.append(f"could not parse {config_path}: {exc}")
            cfg = {}
        # Codex 0.145+ names the feature `hooks` (stage: stable, enabled
        # by default). Legacy 0.11x–0.12x named it `codex_hooks` (under
        # development, off by default). A default-on feature is absent from
        # config.toml when left at its default, so absence must read as ON,
        # not off. Honour an explicit setting under either key; treat a
        # parsed-but-unmentioned feature as the current default (on).
        hooks_flag: bool | None = None
        features = cfg.get("features")
        if isinstance(features, dict):
            fdict = cast(dict[str, object], features)
            if "hooks" in fdict:
                hooks_flag = fdict.get("hooks") is True
            elif "codex_hooks" in fdict:
                hooks_flag = fdict.get("codex_hooks") is True
        report.feature_flag_on = True if hooks_flag is None else hooks_flag
        hooks_cfg = cfg.get("hooks")
        if isinstance(hooks_cfg, dict):
            state = cast(dict[str, object], hooks_cfg).get("state")
            if isinstance(state, dict):
                for entry in cast(dict[str, object], state).values():
                    # Key-membership only — the approval digest value
                    # itself is never read, held, or logged.
                    if isinstance(entry, dict) and "trusted_hash" in entry:
                        report.approved_state_count += 1
    if report.feature_flag_on is False:
        report.warnings.append(
            "the Codex `hooks` feature is disabled in config.toml "
            "([features].hooks = false) — Codex will not run any hooks; "
            "remove that line or set it true (`hooks` is stable and on "
            "by default)",
        )
    # Approval-state keying is positional today and slated to change
    # upstream (per-handler keys vs per-group digests), so exact
    # count arithmetic would false-positive on multi-handler groups.
    # Warn only on the unambiguous condition: handlers configured,
    # zero approvals recorded.
    if report.owned_handler_count and report.approved_state_count == 0:
        report.warnings.append(
            f"{report.owned_handler_count} aelfrice handler(s) configured "
            "but no approved [hooks.state] entries exist — unapproved "
            "hooks are silently skipped; run /hooks in a Codex session "
            "to approve them",
        )
    return report
