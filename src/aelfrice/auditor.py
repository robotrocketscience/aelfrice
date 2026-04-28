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
SEVERITY_WARN: Final[str] = "warn"

CHECK_ORPHAN_THREADS: Final[str] = "orphan_threads"
# Deprecated alias for v1.0 importer compatibility. Drop in v1.2.0.
CHECK_ORPHAN_EDGES: Final[str] = CHECK_ORPHAN_THREADS
CHECK_FTS_SYNC: Final[str] = "fts_sync"
CHECK_LOCKED_CONTRADICTS: Final[str] = "locked_contradicts"
CHECK_CORPUS_VOLUME: Final[str] = "corpus_volume"

# Default minimum belief count below which the corpus_volume check warns.
# Override via AELFRICE_CORPUS_MIN env var (read in the CLI handler).
CORPUS_MIN_DEFAULT: Final[int] = 50

# Project must be at least this many days old before a low corpus is
# considered alarming -- a brand-new project legitimately has 0 beliefs.
CORPUS_MIN_PROJECT_AGE_DAYS: Final[int] = 7


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


def _check_corpus_volume(
    store: MemoryStore,
    *,
    threshold: int,
    project_age_days: int | None,
) -> AuditFinding:
    """Warn when an established project has too few beliefs to be useful.

    Fires `severity='warn'` when belief count is below `threshold` AND
    the project has been active for at least `CORPUS_MIN_PROJECT_AGE_DAYS`
    days. Brand-new projects (project_age_days < threshold age, or
    unknown) are never warned — they legitimately start empty (issue
    #116).

    Severity is `warn` rather than `fail` because a low corpus is not
    a structural failure of the store; it is a usage signal. `aelf
    health` keeps exiting 0 in this case so CI is unaffected.
    """
    n = store.count_beliefs()
    if project_age_days is None or project_age_days < CORPUS_MIN_PROJECT_AGE_DAYS:
        return AuditFinding(
            check=CHECK_CORPUS_VOLUME,
            severity=SEVERITY_INFO,
            count=n,
            detail=(
                f"{n} belief(s); project age check skipped"
                if project_age_days is None
                else f"{n} belief(s); project ~{project_age_days}d old (under "
                     f"{CORPUS_MIN_PROJECT_AGE_DAYS}d threshold, no warning)"
            ),
        )
    if n >= threshold:
        return AuditFinding(
            check=CHECK_CORPUS_VOLUME,
            severity=SEVERITY_INFO,
            count=n,
            detail=(
                f"{n} belief(s) (>= {threshold}; project ~{project_age_days}d old)"
            ),
        )
    return AuditFinding(
        check=CHECK_CORPUS_VOLUME,
        severity=SEVERITY_WARN,
        count=n,
        detail=(
            f"{n} belief(s) — project is ~{project_age_days}d old but corpus "
            f"is below {threshold}. Run `aelf onboard <path>` to populate the "
            f"belief graph, or check if hooks are recording activity (`aelf "
            f"doctor`)."
        ),
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


def audit(
    store: MemoryStore,
    *,
    corpus_min: int = CORPUS_MIN_DEFAULT,
    project_age_days: int | None = None,
) -> AuditReport:
    """Run the v1.1.0+ mechanical-auditor checks against `store`.

    Pure read-only operation. Returns a structured report; the caller
    decides how to render it and what exit code to emit. The
    corpus-volume check (added in #116) needs `project_age_days`
    supplied by the caller — auditor.py stays pure-store and never
    shells out to git.
    """
    findings = [
        _check_orphan_edges(store),
        _check_fts_sync(store),
        _check_locked_contradicts(store),
        _check_corpus_volume(
            store,
            threshold=corpus_min,
            project_age_days=project_age_days,
        ),
    ]
    return AuditReport(findings=findings, metrics=_gather_metrics(store))
