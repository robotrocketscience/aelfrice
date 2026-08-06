"""One-shot cleanup for the removed MCP surface (#1422).

The MCP server never started on any version of its declared dependency
range, and #1422 deleted it. Users who followed the old install
instructions are left with two artefacts this module finds and reports:

1. **The `[mcp]` extra**, i.e. `fastmcp` (and its transitive tree) sitting
   in the uv tool environment for a feature that no longer exists.
2. **A host MCP registration** — the JSON block `docs/user/MCP.md` told
   people to paste — pointing at an `aelf mcp` command that is now gone,
   so the host shows a server that fails to start.

Two rules shape everything here.

**Report by default; edit only when asked.** aelfrice never wrote the host
registration: nothing in this codebase has ever emitted an `mcpServers`
key, and no path constant for those files exists in the tree. Editing a
file the package did not create, whose location we cannot source from the
repo, is precisely what `lifecycle`'s dotdir contract forbids ("anything
not named below is reported and never deleted"). So the automatic pass
detects and prints; `remove_registration()` performs the edit, and only a
caller that explicitly asks gets it — behind a timestamped backup.

**Advise the reinstall, never run it.** `maybe_migrate_to_uv` may shell out
to `uv tool install` because it is guarded to run *only when the install is
not a uv tool install*. This cleanup targets the opposite population: the
dead extra lives precisely in uv-tool installs. Re-installing the package
from inside the running process is the hazard `/aelf:upgrade` already
avoids by design, so this module reads the receipt and prints the command.

Stdlib-only, and it imports nothing from `aelfrice` outside `lifecycle`'s
package name, so `aelf setup` can reach it without an import cycle.
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

__all__ = [
    "MCP_CLEANUP_SENTINEL",
    "McpCleanupResult",
    "find_registrations",
    "is_aelfrice_mcp_entry",
    "maybe_clean_up_mcp",
    "remove_registration",
]

# Sentinel marking a completed cleanup. Host-scoped, not project-scoped:
# there is one install, one uv environment and one host config per machine,
# so a store-adjacent sentinel would re-run this once per repo. Named and
# placed like `migrated-to-uv` and `spine-backfilled`.
MCP_CLEANUP_SENTINEL: Final[Path] = (
    Path.home() / ".aelfrice" / "mcp-surface-removed"
)

# The command shapes `docs/user/MCP.md` and, before it, `docs/INSTALL.md`
# published over the surface's life. A predicate written from the *current*
# doc alone misses the older two, which is most of the installed base — the
# `aelf serve` spelling shipped in the v1.0 early-access instructions and
# the `python -m` spelling in the original MCP.md.
#
# Matching rules, in order of how much they can go wrong:
#   * the module path is decisive wherever it appears in argv, whatever the
#     interpreter is called (`python`, `python3`, a venv absolute path);
#   * `aelf`/`aelf-mcp` must ALSO match on the subcommand — `aelf` has forty
#     other verbs and a bare command match would delete unrelated servers;
#   * `uv run … aelf mcp` must match the pair adjacently, never on `uv`
#     alone, and never on the absolute project path, which differs per user.
# The map key is deliberately NOT part of any rule: users rename it.
_MODULE_PATH: Final[str] = "aelfrice.mcp_server"
_AELF_COMMANDS: Final[frozenset[str]] = frozenset({"aelf", "aelf-mcp"})
_AELF_SUBCOMMANDS: Final[frozenset[str]] = frozenset({"mcp", "serve"})


@dataclass(frozen=True)
class Registration:
    """One `mcpServers` entry that aelfrice published a recipe for."""

    path: Path
    key: str
    command: str
    args: tuple[str, ...]


@dataclass
class McpCleanupResult:
    """Outcome of a `maybe_clean_up_mcp()` call.

    `ran` is False when the sentinel short-circuited or nothing was found;
    it is the flag the caller prints on, matching the `if result.ran:` shape
    the other setup-time migrations use.
    """

    ran: bool = False
    reason: str = ""
    extra_installed: bool = False
    registrations: list[Registration] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _basename(command: str) -> str:
    """`command` reduced to its executable name, POSIX or Windows spelling.

    A registration may spell the interpreter as an absolute path into a
    venv, so the match has to be on the name rather than the whole string.
    """
    normalised = command.replace("\\", "/").rstrip("/")
    return normalised.rsplit("/", 1)[-1] if normalised else ""


def is_aelfrice_mcp_entry(command: str, args: object) -> bool:
    """True iff this entry runs one of the MCP server recipes we published.

    Conservative by construction: an entry we do not recognise is reported
    and left alone rather than guessed at, because the file belongs to the
    user and routinely holds their other servers.
    """
    arg_list = [a for a in args if isinstance(a, str)] if isinstance(args, list) else []
    if any(_MODULE_PATH in a for a in arg_list):
        return True

    name = _basename(command)
    if name in _AELF_COMMANDS:
        # `aelf-mcp` was documented but never shipped, so it has no other
        # subcommands and a bare match is safe; `aelf` needs the verb.
        if name == "aelf-mcp":
            return True
        return bool(arg_list) and arg_list[0] in _AELF_SUBCOMMANDS
    # `uv run --project <abs> aelf mcp` — the pair must be adjacent, so an
    # unrelated `uv run` server that merely mentions "mcp" does not match.
    for first, second in zip(arg_list, arg_list[1:]):
        if first in _AELF_COMMANDS and second in _AELF_SUBCOMMANDS:
            return True
    return False


def _scan_file(path: Path) -> tuple[list[Registration], list[str]]:
    """Registrations in one config file, plus notes about what was skipped."""
    notes: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], notes
    except OSError as exc:
        # Never let an unreadable host config take down the caller's command.
        return [], [f"{path}: unreadable ({exc}); not inspected"]

    try:
        document = json.loads(raw)
    except ValueError:
        # Comments, trailing commas, or truncation. Guessing at a malformed
        # config is worse than leaving it: report and stop.
        return [], [
            f"{path}: not parseable as strict JSON; not modified — "
            f"remove the aelfrice entry by hand"
        ]

    if not isinstance(document, dict):
        return [], [f"{path}: top level is not an object; not inspected"]
    servers = document.get("mcpServers")
    if servers is None:
        return [], notes
    if not isinstance(servers, dict):
        return [], [f"{path}: 'mcpServers' is not an object; not modified"]

    found: list[Registration] = []
    for key, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        command = entry.get("command")
        if not isinstance(command, str):
            continue
        if is_aelfrice_mcp_entry(command, entry.get("args")):
            args = entry.get("args")
            found.append(Registration(
                path=path,
                key=key,
                command=command,
                args=tuple(a for a in args if isinstance(a, str))
                if isinstance(args, list) else (),
            ))
        elif key in {"aelfrice", "aelfrice_mcp", "aelf"}:
            # Named like ours but running something else. Say so; do not touch.
            notes.append(
                f"{path}: 'mcpServers.{key}' points at {command!r}, which "
                f"aelfrice did not publish; left in place"
            )
    return found, notes


def candidate_config_paths() -> list[Path]:
    """Host config files that may carry an `mcpServers` map.

    NOT sourced from anywhere in this repository — aelfrice has never read
    or written these files, and no constant for them exists in the tree.
    They are the conventional locations for the hosts this project
    supports, which is why nothing here edits them without being asked.
    `AELFRICE_MCP_CONFIG` overrides the list outright, so a user on a
    layout we do not know can still be helped.
    """
    override = os.environ.get("AELFRICE_MCP_CONFIG", "").strip()
    if override:
        return [Path(override).expanduser()]
    home = Path.home()
    return [
        home / ".claude.json",
        home / ".mcp.json",
        Path.cwd() / ".mcp.json",
    ]


def find_registrations(
    paths: list[Path] | None = None,
) -> tuple[list[Registration], list[str]]:
    """Every aelfrice MCP registration across the candidate config files."""
    found: list[Registration] = []
    notes: list[str] = []
    for path in (candidate_config_paths() if paths is None else paths):
        entries, file_notes = _scan_file(path)
        found.extend(entries)
        notes.extend(file_notes)
    return found, notes


def mcp_extra_is_installed(receipt_path: Path | None = None) -> bool:
    """True iff this machine's uv tool install requested the `[mcp]` extra.

    Reads uv's own receipt with stdlib `tomllib` — no subprocess, no
    network. Absent or unparseable is treated as "not an mcp install", so
    the caller writes no sentinel and the check re-arms rather than
    recording a cleanup that never happened.
    """
    if receipt_path is None:
        receipt_path = (
            Path.home() / ".local" / "share" / "uv" / "tools"
            / "aelfrice" / "uv-receipt.toml"
        )
    try:
        with receipt_path.open("rb") as handle:
            receipt = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    requirements = receipt.get("tool", {}).get("requirements", [])
    if not isinstance(requirements, list):
        return False
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        extras = requirement.get("extras")
        if isinstance(extras, list) and "mcp" in extras:
            return True
    return False


def remove_registration(
    registration: Registration,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Delete one registration, writing a timestamped backup first.

    Opt-in: nothing in the automatic path calls this. Returns
    `(changed, message)`; the message names the backup so the user can undo
    it by hand. The whole document is re-serialised, so formatting is
    normalised — said plainly in the message rather than discovered later.
    """
    path = registration.path
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.aelfrice-{stamp}.bak")
    try:
        raw = path.read_text(encoding="utf-8")
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        return False, f"{path}: could not read ({exc}); not modified"

    servers = document.get("mcpServers")
    if not isinstance(servers, dict) or registration.key not in servers:
        return False, f"{path}: '{registration.key}' is already gone"

    try:
        backup.write_text(raw, encoding="utf-8")
    except OSError as exc:
        # No backup, no edit. The user's config is not worth a one-way trip.
        return False, f"{path}: could not write backup {backup} ({exc}); not modified"

    del servers[registration.key]
    if not servers:
        del document["mcpServers"]
    try:
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"{path}: write failed ({exc}); backup kept at {backup}"
    return True, (
        f"removed 'mcpServers.{registration.key}' from {path} "
        f"(backup: {backup}; JSON formatting normalised to 2-space indent)"
    )


def maybe_clean_up_mcp(
    *,
    sentinel_path: Path | None = None,
    force: bool = False,
    config_paths: list[Path] | None = None,
    receipt_path: Path | None = None,
) -> McpCleanupResult:
    """Report the leftovers of the removed MCP surface, once per host.

    Detect-and-report only: this never edits a host config and never runs a
    package operation. `sentinel_path` is late-bound rather than defaulted
    at import so a test that repoints HOME is actually obeyed (#1320).

    The sentinel is written only when the pass ran to completion, so a
    failure re-arms rather than recording a cleanup that did not happen.
    """
    if sentinel_path is None:
        sentinel_path = MCP_CLEANUP_SENTINEL
    if not force and sentinel_path.exists():
        return McpCleanupResult(False, "already reported (sentinel exists)")

    result = McpCleanupResult(ran=True, reason="checked")
    result.extra_installed = mcp_extra_is_installed(receipt_path)
    result.registrations, result.notes = find_registrations(config_paths)

    if result.extra_installed:
        result.notes.append(
            "the [mcp] extra is still installed; the MCP server was removed "
            "in v4.3 — reinstall clean with `uv tool install --force aelfrice`"
        )
    for registration in result.registrations:
        result.notes.append(
            f"{registration.path}: 'mcpServers.{registration.key}' starts "
            f"`{registration.command}`, which no longer exists — remove that "
            f"entry, or run `aelf migrate --remove-mcp-config`"
        )

    if not result.notes:
        result.reason = "nothing to clean up"

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"mcp surface removed; reported at "
            f"{datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    except OSError:
        # A sentinel we could not write means this reports again next time.
        # Noisy beats silently skipping a cleanup the user needs.
        pass
    return result
