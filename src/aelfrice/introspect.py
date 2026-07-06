"""`aelf introspect` — a read-only, honest-signal view over stored beliefs (#1081).

Users asked for a native answer to "look at my conversations and analyse the
beliefs it extracted" — today they hand-roll an ad-hoc digest that groups
beliefs into themes and flags which are half-thoughts versus decided. This
module is the deterministic core of that view: given a store (and optional
session / project filters) it groups the active beliefs and, for each, surfaces
the signals that already exist in the store but are never shown together:

* **posterior mean μ** and its **evidence weight** (α+β) — how confident, and
  on how much evidence.
* **recurrence** (corroboration count) — how many times the belief was
  re-asserted. Surfaced with an explicit caveat: recurrence is *not* truth
  (a junk line re-captured every session scores high on recurrence alone;
  #1081/#1086).
* **grounding** — the entity-persistence signal (#1096): `durable` (grounds to
  file paths / error codes / symbols), `ephemeral` (grounds to version /
  branch / bare issue-number coordination chatter), or `neutral` (prose that
  extracts only noun phrases / URLs, or nothing — no grounding signal either
  way). This is the "standalone-meaningful vs context-bound" axis.
* **status** — floated vs decided, read off `RESOLVES` / `POTENTIALLY_STALE`
  edges. Most beliefs are `floated`; a belief that resolves (or is resolved by)
  another is `decides` / `decided`.
* **noise** — the stranded-capture predicate (#1081): orphan headers and shell
  echoes that were stored as standalone beliefs. These are the prime retire
  candidates, so the view floats them to the top of each group.

Everything here is deterministic: pure counts and edge lookups, no model, no
clock. The command is read-only — curation stays in the existing verbs
(`aelf retire` / `aelf lock` / `aelf resolve`), which the CLI footer points at.
"""
from __future__ import annotations

from dataclasses import dataclass

from aelfrice import noise_filter
from aelfrice.models import (
    EDGE_POTENTIALLY_STALE,
    EDGE_RESOLVES,
    LOCK_USER,
)

# entity_persistence S1 = durable / (durable + transient + 1). A single durable
# entity with no transient ones lands at 0.5; transient-dominated content lands
# below it. So >= 0.5 reads as durable grounding, < 0.5 as ephemeral. Beliefs
# absent from the map (no entities, or only grounding-neutral ones) carry no
# grounding signal and are reported as `neutral`.
_GROUNDING_DURABLE_THRESHOLD = 0.5

GROUNDING_DURABLE = "durable"
GROUNDING_EPHEMERAL = "ephemeral"
GROUNDING_NEUTRAL = "neutral"

STATUS_FLOATED = "floated"
STATUS_DECIDES = "decides"
STATUS_DECIDED = "decided"
STATUS_STALE = "stale?"

GROUP_BY_SESSION = "session"
GROUP_BY_PROJECT = "project"

# Suspicion ordering within a group: noise first, then least-grounded, then
# lowest-confidence — i.e. the beliefs most worth a curation look rise to top.
_GROUNDING_RANK = {
    GROUNDING_EPHEMERAL: 0,
    GROUNDING_NEUTRAL: 1,
    GROUNDING_DURABLE: 2,
}


@dataclass(frozen=True)
class BeliefSignals:
    """The honest-signal snapshot for one belief in the introspection view."""

    id: str
    content: str
    posterior_mean: float
    evidence: float  # alpha + beta
    recurrence: int  # corroboration_count
    grounding: str  # durable | ephemeral | neutral
    status: str  # floated | decides | decided | stale?
    noise: bool
    lock_level: str
    lock_tier: str | None
    origin: str

    def _sort_key(self) -> tuple[bool, int, float, str]:
        # `not noise` puts noise (True) first; then least-grounded; then
        # lowest posterior; id is the deterministic final tiebreak.
        return (
            not self.noise,
            _GROUNDING_RANK[self.grounding],
            self.posterior_mean,
            self.id,
        )


@dataclass(frozen=True)
class Group:
    """A set of beliefs sharing a grouping key (session id or project ctx)."""

    key: str  # raw key ("" for the no-session / no-project bucket)
    label: str  # display label
    beliefs: tuple[BeliefSignals, ...]

    @property
    def count(self) -> int:
        return len(self.beliefs)

    @property
    def noise_count(self) -> int:
        return sum(1 for b in self.beliefs if b.noise)


@dataclass(frozen=True)
class IntrospectReport:
    group_by: str
    groups: tuple[Group, ...]
    total: int
    noise_total: int


def _grounding(score: float | None) -> str:
    if score is None:
        return GROUNDING_NEUTRAL
    return (
        GROUNDING_DURABLE
        if score >= _GROUNDING_DURABLE_THRESHOLD
        else GROUNDING_EPHEMERAL
    )


def _status(store: object, belief_id: str) -> str:
    """Floated vs decided, read off RESOLVES / POTENTIALLY_STALE edges.

    Incoming RESOLVES (some belief resolves this one) → decided; incoming
    POTENTIALLY_STALE → stale?; outgoing RESOLVES (this belief resolves
    another) → decides; otherwise floated. Incoming signals win because a
    belief that has *been* resolved is the more decided of the two.
    """
    incoming = store.edges_to(belief_id)  # type: ignore[attr-defined]
    if any(e.type == EDGE_RESOLVES for e in incoming):
        return STATUS_DECIDED
    if any(e.type == EDGE_POTENTIALLY_STALE for e in incoming):
        return STATUS_STALE
    outgoing = store.edges_from(belief_id)  # type: ignore[attr-defined]
    if any(e.type == EDGE_RESOLVES for e in outgoing):
        return STATUS_DECIDES
    return STATUS_FLOATED


def _signals_for(store: object, belief: object, persistence: dict[str, float]) -> BeliefSignals:
    alpha: float = belief.alpha  # type: ignore[attr-defined]
    beta: float = belief.beta  # type: ignore[attr-defined]
    ab = alpha + beta
    lock_level: str = belief.lock_level  # type: ignore[attr-defined]
    return BeliefSignals(
        id=belief.id,  # type: ignore[attr-defined]
        content=belief.content,  # type: ignore[attr-defined]
        posterior_mean=(alpha / ab) if ab > 0 else 0.0,
        evidence=ab,
        recurrence=belief.corroboration_count,  # type: ignore[attr-defined]
        grounding=_grounding(persistence.get(belief.id)),  # type: ignore[attr-defined]
        status=_status(store, belief.id),  # type: ignore[attr-defined]
        noise=noise_filter.is_stranded_capture_noise(belief.content),  # type: ignore[attr-defined]
        lock_level=lock_level,
        lock_tier=belief.lock_tier if lock_level == LOCK_USER else None,  # type: ignore[attr-defined]
        origin=belief.origin,  # type: ignore[attr-defined]
    )


def build_report(
    store: object,
    *,
    group_by: str = GROUP_BY_SESSION,
    session: str | None = None,
    project: str | None = None,
    only_noise: bool = False,
    limit: int | None = 100,
) -> IntrospectReport:
    """Build the introspection report over the store's active beliefs.

    ``group_by`` is ``"session"`` (default) or ``"project"``. ``session`` /
    ``project`` filter to one conversation/session or one project_context.
    ``only_noise`` keeps just the stranded-capture-flagged beliefs (the
    curation shortlist). ``limit`` caps the belief count after filtering,
    newest first (``None`` = no cap). Deterministic: same store state and
    args yield the same report.
    """
    if group_by not in (GROUP_BY_SESSION, GROUP_BY_PROJECT):
        raise ValueError(
            f"group_by must be {GROUP_BY_SESSION!r} or {GROUP_BY_PROJECT!r}, "
            f"got {group_by!r}"
        )
    beliefs = store.list_active_beliefs(limit=None, order="recent")  # type: ignore[attr-defined]
    if session is not None:
        beliefs = [b for b in beliefs if b.session_id == session]
    if project is not None:
        beliefs = [b for b in beliefs if b.project_context == project]
    if limit is not None:
        beliefs = beliefs[:limit]

    ids = [b.id for b in beliefs]
    persistence = store.entity_persistence_scores(ids) if ids else {}  # type: ignore[attr-defined]

    # Group first (needs the Belief's raw key), then attach signals.
    buckets: dict[str, list[object]] = {}
    for b in beliefs:
        if group_by == GROUP_BY_SESSION:
            raw = b.session_id or ""
        else:
            raw = b.project_context or ""
        buckets.setdefault(raw, []).append(b)

    groups: list[Group] = []
    noise_total = 0
    total = 0
    for raw in sorted(buckets, key=lambda k: (k == "", k)):
        sigs = [_signals_for(store, b, persistence) for b in buckets[raw]]
        if only_noise:
            sigs = [s for s in sigs if s.noise]
        if not sigs:
            continue
        sigs.sort(key=lambda s: s._sort_key())
        label = _group_label(group_by, raw)
        groups.append(Group(key=raw, label=label, beliefs=tuple(sigs)))
        total += len(sigs)
        noise_total += sum(1 for s in sigs if s.noise)

    return IntrospectReport(
        group_by=group_by,
        groups=tuple(groups),
        total=total,
        noise_total=noise_total,
    )


def _group_label(group_by: str, raw: str) -> str:
    if raw:
        return raw
    return "(no session)" if group_by == GROUP_BY_SESSION else "(no project)"
