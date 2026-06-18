"""Per-turn hook audit log: config, path resolution, and append/rotate.

These primitives were extracted from `hook.py` (#968) so the audit sink can
be reused by callers that must stay off `hook.py`'s import path. Importing
`aelfrice.hook` pulls in the retrieval stack (scipy, ~220ms cold), which
blows the sub-10ms budget of the transcript logger's hook process — this
module has no such dependency, so `transcript_logger` and `hook_tail` import
the audit primitives from here directly.

The belief-coupled record builders (`_write_hook_audit_record`,
`_serialize_belief_for_audit`) stay in `hook.py`: they depend on the Belief
model and are only called from the retrieval hook, which already pays the
heavy import.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final, cast

# ---------------------------------------------------------------------------
# Per-turn audit log (#280 mitigation 3)
# ---------------------------------------------------------------------------

_CONFIG_FILENAME: Final[str] = ".aelfrice.toml"

AUDIT_DEFAULT_MAX_BYTES: Final[int] = 10 * 1024 * 1024
"""Default size cap before rotation (10 MB). Overridable via .aelfrice.toml."""

AUDIT_FILENAME: Final[str] = "hook_audit.jsonl"
"""Live audit log filename, sibling of memory.db under <git-common-dir>/aelfrice/."""

AUDIT_ROTATED_SUFFIX: Final[str] = ".1"
"""Single-slot rotation suffix. Rollover renames hook_audit.jsonl -> hook_audit.jsonl.1."""

_AUDIT_SECTION: Final[str] = "hook_audit"
_AUDIT_ENABLED_KEY: Final[str] = "enabled"
_AUDIT_MAX_BYTES_KEY: Final[str] = "max_bytes"
_AUDIT_ENV_DISABLE: Final[str] = "AELFRICE_HOOK_AUDIT"


@dataclass(frozen=True)
class HookAuditConfig:
    """Resolved configuration for the per-turn hook audit log.

    `enabled` defaults True (audit-on) per #280 ratification — the surface
    is monitored unless the operator explicitly opts out via env var or
    TOML. `max_bytes` controls when the live file is rotated.
    """

    enabled: bool = True
    max_bytes: int = AUDIT_DEFAULT_MAX_BYTES


def load_hook_audit_config(
    start: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    stderr: IO[str] | None = None,
) -> HookAuditConfig:
    """Resolve the [hook_audit] config.

    Resolution order:
    1. `AELFRICE_HOOK_AUDIT=0` env var → disabled (overrides TOML).
    2. Walk up from `start` looking for `.aelfrice.toml`; first hit wins.
    3. Default (enabled=True, max_bytes=AUDIT_DEFAULT_MAX_BYTES).

    Missing file / missing section / malformed TOML / wrong-typed values
    all degrade to the safe default with a stderr trace; never raises.
    """
    serr: IO[str] = stderr if stderr is not None else sys.stderr
    env_map = env if env is not None else dict(os.environ)
    env_val = env_map.get(_AUDIT_ENV_DISABLE)
    if env_val is not None and env_val.strip() == "0":
        return HookAuditConfig(enabled=False)
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / _CONFIG_FILENAME
        if candidate.is_file():
            try:
                raw = candidate.read_bytes()
            except OSError as exc:
                print(
                    f"aelfrice hook: cannot read {candidate}: {exc}",
                    file=serr,
                )
                return HookAuditConfig()
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except tomllib.TOMLDecodeError as exc:
                print(
                    f"aelfrice hook: malformed TOML in {candidate}: {exc}",
                    file=serr,
                )
                return HookAuditConfig()
            section_obj: Any = parsed.get(_AUDIT_SECTION, {})
            if not isinstance(section_obj, dict):
                return HookAuditConfig()
            section = cast(dict[str, Any], section_obj)
            enabled_obj: Any = section.get(_AUDIT_ENABLED_KEY, True)
            if not isinstance(enabled_obj, bool):
                print(
                    f"aelfrice hook: ignoring [{_AUDIT_SECTION}] "
                    f"{_AUDIT_ENABLED_KEY} in {candidate} (expected bool)",
                    file=serr,
                )
                enabled_obj = True
            max_bytes_obj: Any = section.get(
                _AUDIT_MAX_BYTES_KEY, AUDIT_DEFAULT_MAX_BYTES,
            )
            if not isinstance(max_bytes_obj, int) or max_bytes_obj <= 0:
                if not (
                    isinstance(max_bytes_obj, int)
                    and max_bytes_obj == AUDIT_DEFAULT_MAX_BYTES
                ):
                    print(
                        f"aelfrice hook: ignoring [{_AUDIT_SECTION}] "
                        f"{_AUDIT_MAX_BYTES_KEY} in {candidate} "
                        f"(expected positive int)",
                        file=serr,
                    )
                max_bytes_obj = AUDIT_DEFAULT_MAX_BYTES
            return HookAuditConfig(
                enabled=enabled_obj,
                max_bytes=max_bytes_obj,
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    return HookAuditConfig()


def _audit_path_for_db(db_path_val: Path) -> Path:
    """Derive the audit log path from the DB path. Sibling of memory.db."""
    return db_path_val.parent / AUDIT_FILENAME


def _append_audit(
    audit_path: Path,
    record: dict[str, object],
    max_bytes: int,
    *,
    stderr: IO[str] | None = None,
) -> None:
    """Append one record to the audit JSONL. Rotate if size cap exceeded.

    Append-then-rotate semantics: the record always lands. If, after
    writing, the live file exceeds `max_bytes`, it is renamed to
    `<path>.1` (overwriting any prior `.1`) and a fresh empty file is
    started for the next call. Single-slot rotation by spec; no archive.

    Fail-soft: any I/O error is logged to stderr and swallowed.
    """
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record) + "\n"
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        if audit_path.stat().st_size > max_bytes:
            rotated = audit_path.with_name(
                audit_path.name + AUDIT_ROTATED_SUFFIX,
            )
            os.replace(audit_path, rotated)
    except Exception as exc:
        serr = stderr if stderr is not None else sys.stderr
        print(
            f"aelfrice: hook audit write failed (non-fatal): {exc}",
            file=serr,
        )
