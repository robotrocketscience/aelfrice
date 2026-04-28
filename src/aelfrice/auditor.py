"""Structural auditor for `aelf health`.

Three mechanical checks in v1.1.0. Each fires `severity='fail'` when the
invariant is violated; `aelf health` exits 1 if any failure is present
so CI can gate on it. Informational metrics (credal gap, thread counts,
feedback coverage, average confidence) are reported alongside but do
not affect exit status.

Checks:
  - orphan_threads     threads (graph relationships) whose src or dst
                       no longer exists in beliefs
  - fts_sync           beliefs_fts row count matches beliefs row count
  - locked_contradicts pairs of locked beliefs joined by a CONTRADICTS thread

Threshold-tuned checks (isolated clusters, decay anomalies) are deferred
to v1.2.0+ — they require calibration data the v1.1.0 store doesn't yet
make available.

The internal schema and code keep "edges"; user-facing labels surface as
"threads" per the v1.1.0 cosmetic rename. `CHECK_ORPHAN_EDGES` is kept
as a deprecated alias for the constant (same value as the new
`CHECK_ORPHAN_THREADS`) for v1.0 importer compatibility; remove in v1.2.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from aelfrice.store import MemoryStore

SEVERITY_FAIL: Final[str] = "fail"
SEVERITY_INFO: Final[str] = "info"

CHECK_ORPHAN_THREADS: Final[str] = "orphan_threads"
# Deprecated alias for v1.0 importer compatibility. Drop in v1.2.0.
CHECK_ORPHAN_EDGES: Final[str] = CHECK_ORPHAN_THREADS
CHECK_FTS_SYNC: Final[str] = "fts_sync"
CHECK_LOCKED_CONTRADICTS: Final[str] = "locked_contradicts"


@dataclass(frozen=True)
class AuditFinding:
    """One row of the audit report."""

    check: str
    severity: str
    count: int
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """Result of `audit(store)`."""

    findings: list[AuditFinding]
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """True iff any finding has severity `fail`."""
        return any(f.severity == SEVERITY_FAIL for f in self.findings)


def _check_orphan_edges(store: MemoryStore) -> AuditFinding:
    n = store.count_orphan_edges()
    if n == 0:
        return AuditFinding(
            check=CHECK_ORPHAN_THREADS,
            severity=SEVERITY_INFO,
            count=0,
            detail="all threads resolve to existing beliefs",
        )
    return AuditFinding(
        check=CHECK_ORPHAN_THREADS,
        severity=SEVERITY_FAIL,
        count=n,
        detail=f"{n} thread(s) reference deleted or missing beliefs",
    )


def _check_fts_sync(store: MemoryStore) -> AuditFinding:
    beliefs = store.count_beliefs()
    fts = store.count_fts_rows()
    if beliefs == fts:
        return AuditFinding(
            check=CHECK_FTS_SYNC,
            severity=SEVERITY_INFO,
            count=0,
            detail=f"FTS5 mirror in sync ({beliefs} rows)",
        )
    drift = abs(beliefs - fts)
    return AuditFinding(
        check=CHECK_FTS_SYNC,
        severity=SEVERITY_FAIL,
        count=drift,
        detail=f"beliefs={beliefs} vs beliefs_fts={fts} (drift {drift})",
    )


def _check_locked_contradicts(store: MemoryStore) -> AuditFinding:
    pairs = store.list_locked_contradicts_pairs()
    if not pairs:
        return AuditFinding(
            check=CHECK_LOCKED_CONTRADICTS,
            severity=SEVERITY_INFO,
            count=0,
            detail="no unresolved contradictions between locked beliefs",
        )
    sample = ", ".join(f"({a},{b})" for a, b in pairs[:3])
    if len(pairs) > 3:
        sample += f", … ({len(pairs) - 3} more)"
    return AuditFinding(
        check=CHECK_LOCKED_CONTRADICTS,
        severity=SEVERITY_FAIL,
        count=len(pairs),
        detail=f"{len(pairs)} locked CONTRADICTS thread(s): {sample} — run `aelf resolve`",
    )


def _gather_metrics(store: MemoryStore) -> dict[str, float | int]:
    """Informational metrics displayed alongside the audit. Read-only."""
    n_beliefs = store.count_beliefs()
    n_edges = store.count_edges()
    n_locked = store.count_locked()
    n_feedback = store.count_feedback_events()
    edges_by_type = store.count_edges_by_type()
    pairs = store.alpha_beta_pairs()
    if pairs:
        avg_confidence = sum(a / (a + b) for a, b in pairs if a + b > 0) / len(pairs)
        # credal gap: beliefs whose alpha+beta is at the type prior — i.e.
        # never received feedback. v1.0 type priors all sum to mass <= 4.0
        # (factual 4.0, requirement 9.5, preference 5.0, correction 9.0);
        # we approximate "untested" as (alpha+beta) <= the smallest prior
        # mass observed (3.5, the factual prior alpha=3.0, beta=0.5 ≈ 3.5).
        # Beliefs at higher mass have received feedback events.
        credal_gap = sum(1 for a, b in pairs if a + b <= 3.5)
    else:
        avg_confidence = 0.0
        credal_gap = 0
    metrics: dict[str, float | int] = {
        "beliefs": n_beliefs,
        # v1.1.0 user-facing rename: graph relationships surface as
        # "threads" in CLI output. Internal schema keeps `edges`.
        "threads": n_edges,
        "locked": n_locked,
        "feedback_events": n_feedback,
        "avg_confidence": round(avg_confidence, 3),
        "credal_gap": credal_gap,
    }
    for edge_type, count in sorted(edges_by_type.items()):
        metrics[f"threads_{edge_type.lower()}"] = count
    return metrics


def audit(store: MemoryStore) -> AuditReport:
    """Run the v1.1.0 mechanical-auditor checks against `store`.

    Pure read-only operation. Returns a structured report; the caller
    decides how to render it and what exit code to emit.
    """
    findings = [
        _check_orphan_edges(store),
        _check_fts_sync(store),
        _check_locked_contradicts(store),
    ]
    return AuditReport(findings=findings, metrics=_gather_metrics(store))
