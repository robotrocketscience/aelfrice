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
them.

A log rotated before this shipped carries no marker, and that case is
UNKNOWN rather than clean: one rollover discards nothing, a second
discards the first archive, and the surviving files cannot say which.
Both the reader (`audit_window`) and the writer (`_rotation_marker`,
which sees the doomed `.1` before `os.replace` removes it) report it as
unknown with a lower bound, never as complete. Claiming completeness
there would be the same unearned claim this marker exists to prevent.
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
    marker_present: bool = False
    """Whether this file carries a rotation marker at all.

    Load-bearing, not diagnostic. `generation` defaults to 1 for a file
    with no marker, and 1 is also the truthful answer for a file that has
    genuinely never rotated — so the value alone cannot tell "generation 1"
    from "unknown generation, written before #1528". This flag can.
    """
    discarded_unknown: bool = False
    """Whether `generation`/`discarded` are LOWER BOUNDS rather than counts.

    Set once an unmarked history has been destroyed: nothing survives that
    could say how many generations preceded it, so the true counts are
    permanently unknowable. Propagates forward through every later marker,
    because a bound never becomes exact again.
    """


def _scan_audit_file(path: Path) -> _FileScan:
    """Read one audit file's generation, row count, and ts range.

    `generation` comes from the file's own rotation-marker row and is **1**
    when the file has none — a file written before #1528, or the very first
    generation, which never had a predecessor to stamp it. Those two cases
    are *not* the same claim, and `marker_present` is what separates them:
    a first generation has provably discarded nothing, while a pre-#1528
    file has discarded an unknown amount. Callers must not read a bare
    `generation == 1` as "nothing lost". Marker rows are not counted as
    records and their `ts` does not widen the range: the marker is
    bookkeeping about the file, not a fire in it.

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
    marker_present = False
    discarded_unknown = False
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
                marker_present = True
                gen: Any = rec.get("generation")
                if _is_int(gen) and gen >= 1:
                    generation = cast(int, gen)
                lost: Any = rec.get("discarded_generations")
                if _is_int(lost) and lost >= 0:
                    discarded = cast(int, lost)
                if rec.get("discarded_unknown") is True:
                    discarded_unknown = True
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
        marker_present=marker_present,
        discarded_unknown=discarded_unknown,
    )


def _rotation_marker(
    audit_path: Path, *, rotated_present: bool,
) -> dict[str, object]:
    """Build the marker row for the generation that succeeds `audit_path`.

    `discarded_generations` is `generation - 1`: rotating generation 1
    fills the empty `.1` slot and loses nothing; rotating generation 2
    overwrites generation 1's archive, and so on. That subtraction is the
    whole point of the marker — after the `os.replace` the discarded file
    is gone and no amount of reading what remains can recover the count.

    `rotated_present` is whether a `.1` already exists, read at the call
    site BEFORE the `os.replace` that overwrites it. It is what makes the
    pre-#1528 population honest, and it cannot be inferred from the scan.
    A live file carrying no marker scans as generation 1, so without this
    flag every log already rotated in the wild — the population the issue
    was filed about, three of seventeen on the reporter's machine, each
    beside a 10.5 MB `.1` — would stamp `generation: 2,
    discarded_generations: 0` on the very rollover that destroyed that
    archive. That is a false completeness claim, strictly worse than the
    silence it replaced.

    Three cases, and the middle one is the fix:

    * marker on the source -> generation is known, carry it forward.
    * no marker, `.1` present -> the source rotated at least once before
      #1528 shipped and the archive dying in this `os.replace` is real
      history. Treated as generation >= 2, so `discarded_generations` is
      >= 1, and `discarded_unknown` marks both as LOWER BOUNDS. How many
      generations an unmarked history discarded is not knowable; "unknown,
      at least one destroyed" is the truthful answer and "nothing
      discarded" is not.
    * no marker, no `.1` -> genuinely the first rollover. Nothing lost,
      and the counts are exact.

    COST, re-derived on this branch against a synthetic 10.01 MiB file of
    10,360 realistic audit rows — sized to `AUDIT_DEFAULT_MAX_BYTES`, the
    10 MiB cap a rotation actually fires at: 56 ms (min of 5) for the
    scan, paid on the one fire in 10,360 that rotates, so 0.0054 ms
    amortised per fire.

    BOTH FIGURES ARE AT THE DEFAULT CAP ONLY, and neither is a constant.
    The scan reads and parses every row, so its cost scales with the file
    — and `.aelfrice.toml`'s `[hook_audit] max_bytes` sets that file's
    size to whatever the operator likes. Doubling `max_bytes` roughly
    doubles the scan and roughly halves the amortised figure, because
    rotations get rarer in proportion. Quote the ratio, not the
    milliseconds, for any other cap.

    The scan parses every row because `first_ts`/`last_ts` are a true
    min/max — concurrent hook processes append to this file, so taking
    the first and last LINE instead would be an ordering assumption
    rather than a measurement. A byte-oriented newline count is roughly
    an order of magnitude cheaper — 3.7 ms against 31.2 ms over a
    synthetic 10.00 MiB file of 9,939 rows, medians of 7 — but it yields
    only the row count, which is why it was not used. Both absolutes
    track row CONTENT, not just file size, so they move with the corpus
    as well as with the cap; the order of magnitude between them is the
    part that holds.
    """
    scan = _scan_audit_file(audit_path)
    if scan.marker_present:
        source_generation = scan.generation
        unknown = scan.discarded_unknown
    elif rotated_present:
        source_generation = 2
        unknown = True
    else:
        source_generation = 1
        unknown = False
    return {
        "hook": AUDIT_ROTATION_HOOK,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation": source_generation + 1,
        "discarded_generations": source_generation - 1,
        "discarded_unknown": unknown,
        "rotated_from": {
            "generation": source_generation,
            "records": scan.records,
            "first_ts": scan.first_ts,
            "last_ts": scan.last_ts,
            # The retired file's own counts are exact even when the
            # generation numbering is a bound: they were just measured.
            "marker_present": scan.marker_present,
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

    The answer is THREE-valued, not two, and collapsing it to a bool is
    how a false completeness claim gets shipped. Read `truncated` and
    `complete` together:

    * `truncated` -> at least `discarded_generations` archives are
      provably gone.
    * `complete` -> nothing was discarded, and that is provable.
    * neither -> unknown. The files on disk cannot settle it, because the
      history that would have said so was itself destroyed unmarked.

    `truncated` is deliberately NOT true in the unknown case: a warning
    that fires whenever the answer is merely unavailable is not a signal.
    """

    generation: int = 1
    discarded_generations: int = 0
    records: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    rotated_present: bool = False
    discarded_unknown: bool = False
    """True when `generation`/`discarded_generations` are lower bounds.

    Set when the marker cannot settle the question: an unmarked live file
    beside a `.1` (rotated before #1528 -- one rollover discards nothing,
    a second discards the first archive, and neither file can say which),
    a set holding no live file at all (each file states what had been lost
    when it was CREATED, so archives cannot see the loss their own
    rotation caused), or a marker that already carries the flag forward.
    """

    @property
    def truncated(self) -> bool:
        """True iff at least one generation is PROVABLY discarded."""
        return self.discarded_generations > 0

    @property
    def complete(self) -> bool:
        """True iff nothing was discarded and that is provable.

        Not `not truncated`. The unknown case is neither.
        """
        return not self.truncated and not self.discarded_unknown


def audit_window(paths: Iterable[Path]) -> AuditWindow:
    """Summarise the window covered by `paths` (live + rotated audit logs).

    Degrades to the honest default on a log with no marker and no `.1`:
    generation 1, nothing discarded, `complete` True. Such a file has
    never rotated, so that is exact rather than merely permissive —
    pre-#1528 logs in the wild must keep parsing, and they do, because the
    marker is read as an optional row rather than a required header.

    Missing / unreadable paths are skipped: this is a diagnostic.

    `generation` and `discarded_generations` are LOWER bounds whenever
    `discarded_unknown` is set, which happens in three ways:

    * a scanned marker already carries the flag (an unmarked history was
      destroyed at some earlier rollover);
    * a `.1` is present but the live file carries no marker — the
      pre-#1528 pair, where one rollover discarded nothing and a second
      discarded the first archive, and the files cannot say which. A `.1`
      counts as present when it is on disk beside a live path handed in,
      whether or not the caller also handed in the `.1` itself;
    * a `.1` is present but no live file was handed in at all. Each file
      states what had been lost when it was CREATED, so a set holding only
      archives cannot see the loss its own rotation caused.
    """
    generation = 1
    discarded = 0
    records = 0
    first_ts: str | None = None
    last_ts: str | None = None
    rotated_present = False
    live_marker_seen = False
    unknown = False
    for path in paths:
        is_rotated = path.name.endswith(AUDIT_ROTATED_SUFFIX)
        if is_rotated:
            rotated_present = rotated_present or path.is_file()
        else:
            # Probe the sibling rather than infer "never rotated" from the
            # caller's argument list. Handing in the live path alone is the
            # documented single-file invocation, and every log already
            # rotated in the wild is unmarked -- so inferring from the
            # arguments reports the pre-#1528 population, the one this
            # module exists for, as exactly complete.
            sibling = path.with_name(path.name + AUDIT_ROTATED_SUFFIX)
            rotated_present = rotated_present or sibling.is_file()
        if not path.is_file():
            continue
        try:
            scan = _scan_audit_file(path)
        except OSError:
            continue
        if not is_rotated and scan.marker_present:
            live_marker_seen = True
        unknown = unknown or scan.discarded_unknown
        generation = max(generation, scan.generation)
        discarded = max(discarded, scan.discarded)
        records += scan.records
        lo, hi = scan.first_ts, scan.last_ts
        if lo is not None and (first_ts is None or lo < first_ts):
            first_ts = lo
        if hi is not None and (last_ts is None or hi > last_ts):
            last_ts = hi
    # A `.1` whose generation nothing in this set can state. Without the
    # live file's marker the count of discarded archives is a floor, not a
    # measurement, and reporting it as complete is the unearned claim the
    # marker exists to prevent.
    if rotated_present and not live_marker_seen:
        unknown = True
    return AuditWindow(
        generation=generation,
        discarded_generations=discarded,
        records=records,
        first_ts=first_ts,
        last_ts=last_ts,
        rotated_present=rotated_present,
        discarded_unknown=unknown,
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
            # `rotated.exists()` has to be read here too — the replace on
            # the next line destroys that archive, and on a log that
            # rotated before #1528 its existence is the ONLY evidence that
            # history is being discarded at all.
            marker = json.dumps(
                _rotation_marker(
                    audit_path, rotated_present=rotated.exists(),
                )
            ) + "\n"
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
