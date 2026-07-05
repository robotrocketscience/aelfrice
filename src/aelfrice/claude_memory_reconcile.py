"""Full-set reconciliation of the claude-memory fact-file set into the
belief graph (#1089), plus the shared per-file ingest core it reuses from
the #985 PostToolUse mirror.

The #985 mirror is write-event-triggered: it ingests a fact file when the
agent writes or edits it in-session, mapping the ``metadata.type``
frontmatter to a belief origin/prior. Pre-existing files, and any write the
hook did not observe, never reach the graph. This module closes that hole
with a full-set sweep of the project's ``.../memory/<name>.md`` files, via
the same :func:`ingest_memory_text` the hook now delegates to (so the two
cannot drift).

Following the G4 temporal-spine migration pattern (#1064), the sweep runs
**once per project** behind a sentinel, is announced at ``aelf setup``, and
is gated on the user not having explicitly opted the mirror out. The same
first-run event flips the mirror's effective default on (see
:func:`aelfrice.claude_memory.is_mirror_enabled`), so curated memory stays
in sync thereafter without an operator flag flip.

Contract (inherited from #985): one-way and non-authoritative — aelfrice
never writes back to the memory files. Idempotent — belief ids are
content-derived, so a re-run corroborates rather than duplicates.

The heavy derivation/store imports are module-top here (this module is not
a hot path); the PostToolUse hook keeps its lazy-import discipline by
importing this module only once a memory-file write is confirmed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aelfrice.classification_core import (
    USER_SOURCE,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.claude_memory import derive_memory_dir, parse_memory_file
from aelfrice.derivation import DerivationInput, RouteOverrides, derive
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_CLAUDE_MEMORY,
    INGEST_SOURCE_CLAUDE_MEMORY,
    ORIGIN_USER_VALIDATED,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Cap the body we ingest so a pathologically large memory file cannot blow
# the latency budget or the belief content column (mirrors the #985 hook).
_BODY_BYTE_CAP: Final[int] = 16384


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _BODY_BYTE_CAP:
        return text
    return encoded[:_BODY_BYTE_CAP].decode("utf-8", errors="ignore")


def ingest_memory_text(store: "MemoryStore", text: str) -> str | None:
    """Parse one memory fact-file's ``text``, derive a belief, and
    ``insert_or_corroborate`` it into ``store``. Returns the belief id on a
    successful ingest, else ``None`` (no frontmatter, empty body, or a
    non-persisting classification).

    This is the single home for the #985 frontmatter -> origin/prior
    mapping (ratified 2026-06-23) so the PostToolUse mirror and the #1089
    reconcile sweep cannot drift:

    - ``metadata.type`` ``user`` / ``feedback`` -> ``origin=user_validated``
      with the undeflated prior, frozen as a ``route_overrides`` decision
      (the driving frontmatter lives in ``raw_meta``, which replay nulls,
      so we freeze the decision rather than recompute it).
    - ``project`` / ``reference`` / absent -> ``route_overrides=None``, which
      lets ``derive()`` run the deterministic classifier path ->
      ``origin=agent_inferred`` with the deflated prior.
    - The mirror NEVER locks: L0 stays reserved for explicit ``aelf lock``.

    The caller owns ``store`` (the sweep opens one and loops; the hook opens
    one per write) so this function neither opens nor closes it.
    """
    parsed = parse_memory_file(text)
    if parsed is None:
        return None  # no frontmatter / empty body -> nothing to mirror

    body = _truncate(parsed.body)

    route_overrides = None
    if parsed.memory_type in ("user", "feedback"):
        result = classify_sentence(body, USER_SOURCE)
        if result.persist:
            belief_type, alpha, beta = (
                result.belief_type,
                result.alpha,
                result.beta,
            )
        else:
            belief_type = BELIEF_FACTUAL
            alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, USER_SOURCE)
        route_overrides = RouteOverrides(
            belief_type=belief_type,
            origin=ORIGIN_USER_VALIDATED,
            alpha=alpha,
            beta=beta,
        )

    output = derive(
        DerivationInput(
            raw_text=body,
            source_kind=INGEST_SOURCE_CLAUDE_MEMORY,
            source_path=None,
            route_overrides=route_overrides,
        )
    )
    if output.belief is None:
        return None

    store.insert_or_corroborate(
        output.belief,
        source_type=CORROBORATION_SOURCE_CLAUDE_MEMORY,
    )
    return output.belief.id


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of a reconcile call.

    ``ran`` is True iff the sweep actually executed (in ``maybe_*`` this
    means the sentinel was absent and the mirror was not opted out).
    ``n_files`` is the fact files seen; ``n_ingested`` is how many produced
    a belief (parse/classify may drop some, and re-runs corroborate rather
    than insert). ``reason`` is a one-line description for a stderr notice,
    populated on the skipped paths too.
    """

    ran: bool
    n_files: int
    n_ingested: int
    reason: str


def reconcile_claude_memory(
    store: "MemoryStore",
    memory_dir: Path,
) -> ReconcileResult:
    """Ingest every fact file under ``memory_dir`` into ``store``.

    Scans ``<memory_dir>/*.md`` (non-recursive — the upstream tool writes
    fact files flat), skipping the ``MEMORY.md`` index. Each file goes
    through :func:`ingest_memory_text`, so the sweep and the write-through
    mirror share one code path and one provenance mapping. Idempotent:
    re-running ingests no duplicates. An unreadable file is skipped, not
    fatal.
    """
    if not memory_dir.is_dir():
        return ReconcileResult(True, 0, 0, f"no memory dir at {memory_dir}")

    n_files = 0
    n_ingested = 0
    for path in sorted(memory_dir.glob("*.md")):
        if path.name == "MEMORY.md" or not path.is_file():
            continue
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ingest_memory_text(store, text) is not None:
            n_ingested += 1

    return ReconcileResult(
        True,
        n_files,
        n_ingested,
        f"reconciled {n_ingested}/{n_files} fact file(s) from {memory_dir}",
    )


def maybe_reconcile_claude_memory(
    store: "MemoryStore",
    *,
    project_path: str | Path,
    sentinel_path: Path,
    opted_out: bool | None = None,
) -> ReconcileResult:
    """One-shot, sentinel-gated reconcile for a project (#1089).

    Mirrors the G4 ``maybe_backfill_temporal_spine`` shape so ``aelf setup``
    can call it every run and it fires exactly once per project. The first
    run is the "consent event": it is announced by the caller and it flips
    the mirror's effective default on (the sentinel it writes is what
    :func:`claude_memory.is_mirror_enabled` reads).

    The caller passes ``project_path`` (to derive the memory dir) and
    ``sentinel_path`` (see :func:`reconcile_sentinel_path`) rather than
    having this reach into store internals — ``aelf setup`` already resolves
    both.

    Short-circuit order (cheapest first):

      1. sentinel exists -> no-op (already reconciled + consented).
      2. the mirror is explicitly opted out (``AELFRICE_MIRROR_CLAUDE_MEMORY``
         / ``[memory] mirror_claude_memory`` set false) -> no-op, and the
         sentinel is NOT written, so the check re-arms and fires on the
         first setup after the opt-out is removed.
      3. otherwise reconcile, write the sentinel, and report the counts.

    Never raises — any failure returns a result describing it and leaves the
    store untouched. Reversible: delete the sentinel (or use the CLI
    ``--force``) to re-run; the ingested beliefs are removed with the normal
    belief-delete tooling.
    """
    if sentinel_path.exists():
        return ReconcileResult(
            False, 0, 0, "already reconciled (sentinel exists)"
        )

    if opted_out is None:
        from aelfrice.claude_memory import mirror_opted_out  # noqa: PLC0415

        opted_out = mirror_opted_out()
    if opted_out:
        return ReconcileResult(
            False, 0, 0,
            "mirror opted out — reconcile deferred until opt-out removed",
        )

    memory_dir = derive_memory_dir(project_path)
    try:
        result = reconcile_claude_memory(store, memory_dir)
    except Exception as exc:  # noqa: BLE001 — never break `aelf setup`
        return ReconcileResult(False, 0, 0, f"reconcile failed: {exc}")

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"claude-memory reconciled {result.n_ingested} fact(s) "
            f"at {time.time():.0f}\n"
        )
    except OSError as exc:
        # Reconcile is done + idempotent; a missing sentinel only costs a
        # harmless re-run next setup (which corroborates, ingesting 0 new).
        return ReconcileResult(
            True, result.n_files, result.n_ingested,
            f"{result.reason} (sentinel write failed: {exc})",
        )
    return result
