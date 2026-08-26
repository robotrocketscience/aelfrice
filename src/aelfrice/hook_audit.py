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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Final, cast

from aelfrice.config_discovery import discover_config

# ---------------------------------------------------------------------------
# Per-turn audit log (#280 mitigation 3)
# ---------------------------------------------------------------------------

AUDIT_DEFAULT_MAX_BYTES: Final[int] = 10 * 1024 * 1024
"""Default size cap before rotation (10 MB). Overridable via .aelfrice.toml."""

AUDIT_FILENAME: Final[str] = "hook_audit.jsonl"
"""Live audit log filename, sibling of memory.db under <git-common-dir>/aelfrice/."""

AUDIT_ROTATED_SUFFIX: Final[str] = ".1"
"""Single-slot rotation suffix. Rollover renames hook_audit.jsonl -> hook_audit.jsonl.1."""

AUDIT_ROTATION_HOOK: Final[str] = "audit_rotation"
"""`hook` value of the rotation marker row (#1528).

Rotation is still single-slot: the second rollover overwrites the first
`.1`, and the history in it is gone. Retention was NOT widened — N slots
would multiply the on-disk cost by N on every repo that has an audit log,
and the property #1528 asks for is *detectability*, not retention. So each
rollover instead stamps one JSONL row as the first line of the freshly
emptied live file, carrying this file's generation number, a summary of the
file just retired (row count and ts range), and the one fact nothing else
in the data can reconstruct: how many archives have already been discarded.

Every existing reader filters on a specific `hook` value
(`user_prompt_submit`, `session_start`, ...), so the marker is inert to
them. A log rotated before this shipped simply carries no marker, which
`audit_window` reads as generation 1 with nothing discarded — the correct
answer for a log that has rotated at most once.
"""

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
    # Shared discovery (#1304): inside a `config_discovery_scope`
    # N readers cost one walk instead of N. Semantics unchanged —
    # the loop this replaces already stopped at the first
    # `.aelfrice.toml` it found and never continued past it.
    candidate = discover_config(start)
    if candidate is not None:
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
        # `bool` is a subclass of `int`, so a bare `isinstance(..., int)`
        # accepts `max_bytes = true`, and `True <= 0` is False -- the value
        # passes both arms, no trace prints, and the config carries a cap
        # of 1 byte, rotating the log on every append. Guarded explicitly,
        # in the spelling `dedup`, `relationship_detector`, `cadence`,
        # `auto_install`, `hook` and `noise_filter` already use (#1340).
        if (
            isinstance(max_bytes_obj, bool)
            or not isinstance(max_bytes_obj, int)
            or max_bytes_obj <= 0
        ):
            if not (
                not isinstance(max_bytes_obj, bool)
                and isinstance(max_bytes_obj, int)
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
    return HookAuditConfig()


def _audit_path_for_db(db_path_val: Path) -> Path:
    """Derive the audit log path from the DB path. Sibling of memory.db."""
    return db_path_val.parent / AUDIT_FILENAME


def _is_int(value: object) -> bool:
    """True for a real int. `bool` is an int subclass and is excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class _FileScan:
    """One audit file's self-description. Defaults are the no-marker case."""

    generation: int = 1
    discarded: int = 0
    records: int = 0
    first_ts: str | None = None
    last_ts: str | None = None


def _scan_audit_file(path: Path) -> _FileScan:
    """Read one audit file's generation, row count, and ts range.

    `generation` comes from the file's own rotation-marker row and is **1**
    when the file has none — a file written before #1528, or the very first
    generation, which never had a predecessor to stamp it. Marker rows are
    not counted as records and their `ts` does not widen the range: the
    marker is bookkeeping about the file, not a fire in it.

    `discarded` is read off the marker rather than derived from
    `generation`. Deriving it is off by one at exactly the boundary that
    matters: the first rotation produces a live file at generation 2 and a
    `.1` holding generation 1, and that pair is a COMPLETE history with
    nothing discarded.

    Unparseable and non-object lines are skipped, matching every other
    reader of this log.
    """
    generation = 1
    discarded = 0
    records = 0
    first_ts: str | None = None
    last_ts: str | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed: Any = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            rec = cast(dict[str, Any], parsed)
            if rec.get("hook") == AUDIT_ROTATION_HOOK:
                gen: Any = rec.get("generation")
                if _is_int(gen) and gen >= 1:
                    generation = cast(int, gen)
                lost: Any = rec.get("discarded_generations")
                if _is_int(lost) and lost >= 0:
                    discarded = cast(int, lost)
                continue
            records += 1
            ts: Any = rec.get("ts")
            if isinstance(ts, str) and ts:
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
    return _FileScan(
        generation=generation,
        discarded=discarded,
        records=records,
        first_ts=first_ts,
        last_ts=last_ts,
    )


def _rotation_marker(audit_path: Path) -> dict[str, object]:
    """Build the marker row for the generation that succeeds `audit_path`.

    `discarded_generations` is `generation - 1`: rotating generation 1
    fills the empty `.1` slot and loses nothing; rotating generation 2
    overwrites generation 1's archive, and so on. That subtraction is the
    whole point of the marker — after the `os.replace` the discarded file
    is gone and no amount of reading what remains can recover the count.

    COST, measured on this branch against a synthetic 10.00 MiB file (the
    default cap) of 5,143 realistic audit rows: 43.8 ms for the scan, on
    the one fire that rotates, amortised to 0.0085 ms per fire. The scan
    parses every row because `first_ts`/`last_ts` are a true min/max —
    concurrent hook processes append to this file, so taking the first and
    last LINE instead would be an ordering assumption rather than a
    measurement. A byte-oriented newline count runs in 6.3 ms but yields
    only the row count, which is why it was not used.
    """
    scan = _scan_audit_file(audit_path)
    return {
        "hook": AUDIT_ROTATION_HOOK,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation": scan.generation + 1,
        "discarded_generations": scan.generation - 1,
        "rotated_from": {
            "generation": scan.generation,
            "records": scan.records,
            "first_ts": scan.first_ts,
            "last_ts": scan.last_ts,
        },
    }


@dataclass(frozen=True)
class AuditWindow:
    """What a set of audit files actually covers (#1528).

    `truncated` is the question a benchmark has to be able to ask: were
    older archives destroyed by rotation? Before the marker existed a
    reader could not tell a short history from a truncated one, so every
    long-horizon rate derived from this log was silently bounded to the
    last <=20 MB of fires.
    """

    generation: int = 1
    discarded_generations: int = 0
    records: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    rotated_present: bool = False

    @property
    def truncated(self) -> bool:
        """True iff at least one rotation generation has been discarded."""
        return self.discarded_generations > 0


def audit_window(paths: Iterable[Path]) -> AuditWindow:
    """Summarise the window covered by `paths` (live + rotated audit logs).

    Degrades to the honest default on a log with no marker anywhere:
    generation 1, nothing discarded, `truncated` False. That is exactly
    right for a log that has rotated at most once, and it is the only claim
    such a file supports — pre-#1528 logs in the wild must keep parsing,
    and they do, because the marker is read as an optional row rather than
    a required header.

    Missing / unreadable paths are skipped: this is a diagnostic.

    `discarded_generations` is a LOWER bound when the live file is absent
    from `paths`: each file states what had been lost when it was created,
    so a set holding only the rotated `.1` cannot see the loss its own
    rotation caused.
    """
    generation = 1
    discarded = 0
    records = 0
    first_ts: str | None = None
    last_ts: str | None = None
    rotated_present = False
    for path in paths:
        if path.name.endswith(AUDIT_ROTATED_SUFFIX):
            rotated_present = rotated_present or path.is_file()
        if not path.is_file():
            continue
        try:
            scan = _scan_audit_file(path)
        except OSError:
            continue
        generation = max(generation, scan.generation)
        discarded = max(discarded, scan.discarded)
        records += scan.records
        lo, hi = scan.first_ts, scan.last_ts
        if lo is not None and (first_ts is None or lo < first_ts):
            first_ts = lo
        if hi is not None and (last_ts is None or hi > last_ts):
            last_ts = hi
    return AuditWindow(
        generation=generation,
        discarded_generations=discarded,
        records=records,
        first_ts=first_ts,
        last_ts=last_ts,
        rotated_present=rotated_present,
    )


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
    `<path>.1` (overwriting any prior `.1`) and a fresh file is started
    for the next call. Still single-slot by spec; no archive.

    #1528: the fresh file opens with one `audit_rotation` marker row, so a
    reader can tell a truncated window from a short one. See
    `AUDIT_ROTATION_HOOK` for why detectability rather than retention.

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
            # Read the generation BEFORE the replace: afterwards the file
            # under `audit_path` is gone and the count is unrecoverable.
            marker = json.dumps(_rotation_marker(audit_path)) + "\n"
            os.replace(audit_path, rotated)
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(marker)
                f.flush()
                os.fsync(f.fileno())
    except Exception as exc:
        serr = stderr if stderr is not None else sys.stderr
        print(
            f"aelfrice: hook audit write failed (non-fatal): {exc}",
            file=serr,
        )
