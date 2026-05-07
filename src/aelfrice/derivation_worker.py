"""Derivation worker — materializes beliefs from ingest_log rows.

v2.x slice of the write-log-as-truth refactor (#264). Today every ingest
entry point calls `derive()` itself and writes to `beliefs` directly,
with `record_ingest()` running in parallel as audit. After this worker
lands, the desired call shape is:

    1. Entry point appends a row to `ingest_log` with no derived ids.
    2. Entry point invokes `run_worker(store)` (synchronously, end of
       batch).
    3. Worker tails unstamped rows, calls `derive()`, writes the canonical
       belief (and edges) via the existing `insert_or_corroborate()` API,
       and stamps the log row with the derived ids.

`beliefs` becomes computed state; `ingest_log` is canonical. The view
flip in a later issue stops the parallel write entirely.

The worker is a function called synchronously after each ingest batch in
v2.x. Async / daemon mode is a v3 concern.

Design properties:

- **Idempotency.** Re-invoking `run_worker()` over the same log produces
  identical canonical state. A row whose `derived_belief_ids` is non-empty
  is skipped. A row whose stamp is empty but whose derived belief already
  exists in `beliefs` is re-stamped without a duplicate `insert_belief`
  (deterministic id + content_hash UNIQUE on belief gives idempotency at
  the storage layer; the worker just stamps the orphan).

- **Crash recovery.** If the worker dies between `insert_belief` and
  `update_ingest_derived_ids`, the next run notices the orphan (row with
  empty derived_belief_ids whose deterministic belief id already exists
  in `beliefs`) and stamps it.

- **Concurrency.** Belief id is sha256(source + text)[:16]; two parallel
  workers producing the same id converge on one row via the content_hash
  UNIQUE constraint inside `insert_or_corroborate`. Each worker stamps its
  own log row independently — log rows are 1:1 with raw inputs, not
  beliefs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    CORROBORATION_SOURCE_CLI_REMEMBER,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_MCP_REMEMBER,
)
from aelfrice.store import MemoryStore

# Multiple call sites share `source_kind=filesystem` (transcript ingest
# vs scanner / classification). The log row carries the disambiguating
# `call_site` key in `raw_meta`; when absent we fall back to the
# `source_kind`-implied default below. Entry points migrating into this
# worker SHOULD stash `call_site` explicitly so the corroboration audit
# is unambiguous on replay.
_CORROBORATION_BY_SOURCE_KIND: dict[str, str] = {
    INGEST_SOURCE_FILESYSTEM: CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    INGEST_SOURCE_GIT: CORROBORATION_SOURCE_COMMIT_INGEST,
    INGEST_SOURCE_CLI_REMEMBER: CORROBORATION_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_MCP_REMEMBER: CORROBORATION_SOURCE_MCP_REMEMBER,
}

# Sentinel keys recognised inside `raw_meta` by the worker.
_META_CALL_SITE: str = "call_site"
_META_OVERRIDE_BELIEF_TYPE: str = "override_belief_type"
_META_SOURCE_PATH_HASH: str = "source_path_hash"

# v2.x #265 view-flip flag. Default off. When on, `beliefs` becomes a
# materialized view of `ingest_log` and direct `insert_belief()` calls
# from outside the derivation worker raise. The flag itself is wired
# here in PR-A and consumed by PR-B (insert_belief gating) + PR-C
# (`aelf rebuild` re-derive). Reading inline matches the pattern used
# by AELFRICE_BFS / AELFRICE_BM25F / AELFRICE_HEAT_KERNEL in retrieval.py.
ENV_WRITE_LOG_AUTHORITATIVE: Final[str] = "AELFRICE_WRITE_LOG_AUTHORITATIVE"
_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def is_write_log_authoritative(env: dict[str, str] | None = None) -> bool:
    """Return True iff `AELFRICE_WRITE_LOG_AUTHORITATIVE` is set truthy.

    Two-state (vs the tri-state pattern used by AELFRICE_BFS) because
    no kwarg / TOML precedence layer exists for this flag — env is the
    only signal. Default off; any unset / unrecognised value is off.
    `env` is `os.environ`-like; `None` reads the live process env.
    """
    src = os.environ if env is None else env
    raw = src.get(ENV_WRITE_LOG_AUTHORITATIVE)
    if raw is None:
        return False
    return raw.strip().lower() in _ENV_TRUTHY


@dataclass(frozen=True)
class WorkerResult:
    """Aggregate counts from one `run_worker()` invocation.

    `rows_scanned` is every unstamped row visited.
    `beliefs_inserted` counts new canonical rows written to `beliefs`.
    `beliefs_corroborated` counts hits on the content_hash UNIQUE
        constraint (existing belief, corroboration row appended).
    `rows_stamped` counts log rows whose `derived_belief_ids` we wrote
        (orphan recovery + new derivations both contribute).
    `rows_skipped_no_belief` counts rows where `derive()` returned no
        belief (classifier marked persist=False).
    """

    rows_scanned: int = 0
    beliefs_inserted: int = 0
    beliefs_corroborated: int = 0
    rows_stamped: int = 0
    rows_skipped_no_belief: int = 0


_TRANSCRIPT_CALL_SITE: str = CORROBORATION_SOURCE_TRANSCRIPT_INGEST


def _resolve_corroboration_source(row: dict[str, object]) -> str:
    raw_meta = row.get("raw_meta") if isinstance(row, dict) else None
    if isinstance(raw_meta, dict):
        explicit = raw_meta.get(_META_CALL_SITE)
        if isinstance(explicit, str) and explicit:
            return explicit
    source_kind = str(row.get("source_kind") or "")
    return _CORROBORATION_BY_SOURCE_KIND.get(
        source_kind, CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    )


def _derivation_input_from_row(row: dict[str, object]) -> DerivationInput:
    raw_meta_obj = row.get("raw_meta")
    raw_meta: dict[str, object] | None = (
        raw_meta_obj if isinstance(raw_meta_obj, dict) else None
    )
    override = None
    if raw_meta is not None:
        ov = raw_meta.get(_META_OVERRIDE_BELIEF_TYPE)
        if isinstance(ov, str) and ov:
            override = ov
    source_path = row.get("source_path")
    session_id = row.get("session_id")
    classifier_version = row.get("classifier_version")
    rule_set_hash = row.get("rule_set_hash")
    return DerivationInput(
        raw_text=str(row.get("raw_text") or ""),
        source_kind=str(row.get("source_kind") or ""),
        source_path=source_path if isinstance(source_path, str) else None,
        raw_meta=raw_meta,
        session_id=session_id if isinstance(session_id, str) else None,
        ts=str(row.get("ts") or ""),
        classifier_version=(
            classifier_version if isinstance(classifier_version, str) else None
        ),
        rule_set_hash=(
            rule_set_hash if isinstance(rule_set_hash, str) else None
        ),
        override_belief_type=override,
    )


def run_worker(store: MemoryStore) -> WorkerResult:
    """Materialize beliefs for every unstamped ingest_log row.

    Synchronous, all-rows-in-one-pass. Safe to call after every batch.

    Returns aggregate `WorkerResult` counts. Caller can ignore them in
    production paths and inspect them in tests.
    """
    rows = store.list_unstamped_ingest_log()
    out = WorkerResult()
    for row in rows:
        out = _process_row(store, row, out)
    return out


def _process_row(
    store: MemoryStore,
    row: dict[str, object],
    acc: WorkerResult,
) -> WorkerResult:
    log_id = str(row.get("id") or "")
    if not log_id:
        return acc
    rows_scanned = acc.rows_scanned + 1

    inp = _derivation_input_from_row(row)
    out = derive(inp)
    if out.belief is None:
        # Stamp the row with an explicit empty list so a subsequent
        # worker pass treats it as covered (vs ambiguous NULL = unstamped).
        store.update_ingest_derived_ids(log_id, derived_belief_ids=[])
        return WorkerResult(
            rows_scanned=rows_scanned,
            beliefs_inserted=acc.beliefs_inserted,
            beliefs_corroborated=acc.beliefs_corroborated,
            rows_stamped=acc.rows_stamped + 1,
            rows_skipped_no_belief=acc.rows_skipped_no_belief + 1,
        )

    corroboration_source = _resolve_corroboration_source(row)
    raw_meta = row.get("raw_meta") if isinstance(row, dict) else None
    source_path_hash: str | None = None
    if isinstance(raw_meta, dict):
        sph = raw_meta.get(_META_SOURCE_PATH_HASH)
        if isinstance(sph, str) and sph:
            source_path_hash = sph

    actual_id, was_inserted = store.insert_or_corroborate(
        out.belief,
        source_type=corroboration_source,
        session_id=inp.session_id,
        source_path_hash=source_path_hash,
    )

    derived_edge_ids: list[tuple[str, str, str]] = []
    for edge in out.edges:
        store.insert_edge(edge)
        derived_edge_ids.append((edge.src, edge.dst, edge.type))

    store.update_ingest_derived_ids(
        log_id,
        derived_belief_ids=[actual_id],
        derived_edge_ids=derived_edge_ids if derived_edge_ids else None,
    )

    return WorkerResult(
        rows_scanned=rows_scanned,
        beliefs_inserted=acc.beliefs_inserted + (1 if was_inserted else 0),
        beliefs_corroborated=(
            acc.beliefs_corroborated + (0 if was_inserted else 1)
        ),
        rows_stamped=acc.rows_stamped + 1,
        rows_skipped_no_belief=acc.rows_skipped_no_belief,
    )
