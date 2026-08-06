"""Clamp α on ghost belief rows that lack an audit trail.

A "ghost-α" belief row has `alpha` inflated above the Beta(1,1) prior
with zero supporting evidence in either audit table:

- no rows in ``feedback_history`` (no apply_feedback / hook / deferred /
  sentiment event has run for this belief)
- no rows in ``belief_corroborations`` (no insert_or_corroborate hit
  has re-asserted this belief)

Every current α-mutation path leaves at least one of those trails:

- ``feedback.apply_feedback`` → always inserts into feedback_history
  (`store.insert_feedback_event` at the bottom of the function).
- ``deferred_feedback.sweep_deferred_feedback`` → writes both
  ``UPDATE beliefs SET alpha = alpha + ?`` AND the feedback_history
  row inside the same ``BEGIN IMMEDIATE`` transaction.
- ``store.insert_or_corroborate`` content-hash hit → does NOT bump α
  but DOES record a corroboration row.

That list has one hole, and it is the *insert* path. `derive()` stamps
the undeflated `TYPE_PRIORS` α straight onto a brand-new row for
user-sourced content — α=9.0 for a `requirement` or `correction`, 7.0
for a `preference` — and a row that was created five minutes ago has
neither a feedback event nor a corroboration yet. Such a row matched
all four conditions above by construction, so the selector could not
tell a fabricated ghost from a legitimate belief that is merely new,
and `CLAMP_SOURCE` then wrote an audit row attributing the clamp to
itself. `USER_PRIOR_ORIGINS` closes that hole: origins whose insert
path writes the full undeflated prior are excluded outright.

On the **insert path** that exclusion is complete: non-user origins get
α deflated by `_AGENT_INFERRED_DEFLATION`, and the maximum reachable
there is 1.8 (`correction` from a non-user source), so no insert on a
clampable origin can clear the α=4.0 threshold. That bound is pinned by
`test_no_deterministic_non_user_insert_can_reach_the_clamp_threshold`
rather than left to this paragraph.

It bounds α by **source**, while the selector excludes by **origin**,
and two paths join those two differently. Both are named here because
the value of this file is that its selector's justification is
auditable.

**`route_overrides` writes (origin, α) verbatim.** `derive()` skips the
classifier entirely on that branch (`derivation.py`, the
`inp.route_overrides is not None` block), so `get_source_adjusted_prior`
never runs and nothing structurally caps α. Neither shipped producer
reaches a clampable origin with an inflated α, and they miss it for
different reasons — worth stating, because the safety here rests on the
producers, not on the branch:

- `llm_classifier` (`aelf onboard --llm-classify`, default off) is
  restricted to `{agent_inferred, document_recent}` by
  `_PERMITTED_ORIGINS`, and takes its α from `get_source_adjusted_prior`
  on the candidate's `doc:`/`ast:`/`git:` source label — so it is
  deflated like any other non-user insert and tops out at 1.8, not 9.0.
- `claude_memory_reconcile` (#985 mirror, #1089 sweep) does write the
  **undeflated** prior, up to α=9.0 — but on `origin=user_validated`,
  which `USER_PRIOR_ORIGINS` excludes.

So the exposure is a *future* producer pairing a high α with a
clampable origin, not a current one.

**`origin='unknown'` is the gap, and it is deliberate.** `unknown` is a
non-user origin the selector does **not** exclude, and it is the one
clampable origin that carries α above 1.8 in practice. `migrate()`
copies a legacy row's α verbatim and stamps `unknown` on every unlocked
non-correction row, while copying neither `feedback_history` nor
`belief_corroborations` (out of scope per that module's header). A
belief whose α was legitimately earned in the source store therefore
arrives looking exactly like a fabricated ghost.

It stays clampable because it is not a bystander here — it is the
target population. Every row the clamp has ever actually clamped
carried it; excluding `unknown` would make the tool a no-op on the only
store it has run against. `created_before` is the mitigation: an
operator who has just run a migration should fence the clamp to rows
predating it.

**This is why the one-shot framing needs a caveat.** Empirically (lab
campaign ``retrieval-corpus-bloat`` → ``alpha-gain-third-path``,
2026-05-11) the ghosts on the development store are pre-migration
artifacts, and the live population above 1.8 is narrower than that
survey suggests: it sits on an exact k·(0.6, 1.0) lattice, i.e. k
copies of the deflated *factual* prior summed, which is
``_maybe_consolidate_content_hash_duplicates`` (#219) adding α and β
across a content-hash duplicate group — not ``_read_legacy_beliefs``,
which copies α through unchanged and cannot manufacture that lattice.
That consolidation is marker-gated (`content_hash_dedup_complete`) and
short-circuits forever once set, so it genuinely cannot re-run.

``migrate()`` can. It is reachable today via ``aelf migrate --apply``
and ``aelf doctor``'s in-place upgrade, and a legacy store imported
after a clamp will land fresh candidates. So "one-shot" is a statement
about a *given* store's existing rows, not a property of the tool:
re-run the clamp after any migration, and scope it with
``created_before``.

Reversibility. Every clamp writes a single negative-valence row to
``feedback_history`` with ``source='clamp_ghosts'`` and
``valence = -(prior_alpha - target_alpha)``. To reverse a clamp:

    UPDATE beliefs SET alpha = alpha + (
        SELECT -fh.valence FROM feedback_history fh
        WHERE fh.belief_id = beliefs.id AND fh.source = 'clamp_ghosts'
    )
    WHERE id IN (SELECT belief_id FROM feedback_history
                 WHERE source = 'clamp_ghosts' AND belief_id = beliefs.id);
    DELETE FROM feedback_history WHERE source = 'clamp_ghosts';

Idempotency. After one successful clamp the rows have feedback_history
entries, so the EXISTS filter excludes them on re-run. A bare
``aelf clamp-ghosts --apply`` is safe to run twice.

The default ``target_alpha = 4.0`` is a posterior-fitting choice: at
α=4, β=1 the posterior mean is 0.8 — still highly retrievable, but
not dominating the BM25 ordering against legitimately-confirmed
beliefs. ``--target`` flag lets the operator pick a different value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO

from aelfrice.models import (
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
)
from aelfrice.store import MemoryStore


USER_PRIOR_ORIGINS: tuple[str, ...] = (
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
)
"""Origins whose insert path writes the full undeflated `TYPE_PRIORS` α.

A row on one of these origins can sit at α=9.0 with an empty audit
trail and still be entirely legitimate — it is simply new. They are
excluded from the ghost selector so the clamp cannot fire on them.

A tuple, not a set: it reaches the selector as one bound JSON array
(``_ELIGIBILITY_SQL`` below), and a set's iteration order would make
that array's byte value vary between runs. It is serialised sorted for
the same reason."""


_ELIGIBILITY_SQL: str = (
    "  AND b.origin NOT IN (SELECT je.value FROM json_each(?) je) "
    "  AND (? IS NULL OR b.created_at < ?) "
    "  AND NOT EXISTS ("
    "    SELECT 1 FROM feedback_history fh WHERE fh.belief_id = b.id"
    "  ) "
    "  AND NOT EXISTS ("
    "    SELECT 1 FROM belief_corroborations bc WHERE bc.belief_id = b.id"
    "  ) "
)
"""The eligibility predicate, shared verbatim by both selectors.

The enumeration query and the under-the-write-lock re-check must agree
exactly, or the re-check becomes a second, looser selector that can
clamp a row the enumeration would never have offered. Making it one
module constant rather than a locally-composed fragment is what makes
that agreement checkable — see
``test_both_selectors_share_one_eligibility_predicate``.

The SQL text is fully static. The origin exclusion arrives as a single
bound JSON array read by ``json_each`` rather than an interpolated
``IN (?, ?, …)`` list, and the optional ``created_before`` cutoff is a
bound ``? IS NULL`` disjunction rather than a conditionally-appended
clause. So there is no placeholder count to keep in sync with a
parameter count, and no injection-shaped construction for a reader (or
a scanner) to have to reason about — the same mechanism, and the same
motivation, as ``store.list_stale_speculative_ids`` (#1171).
``json_each`` is available in SQLite ≥ 3.38, standard in Python 3.12+.

Takes three bound parameters, in order: the origins JSON array, then
``created_before`` twice."""


def _eligibility_params(created_before: str | None) -> list[object]:
    """Bound parameters for one ``_ELIGIBILITY_SQL`` splice, in order.

    Kept next to the predicate so a change to one is a change to the
    other in the same edit; both call sites go through it, so they
    cannot drift in parameter order either.
    """
    return [
        json.dumps(sorted(USER_PRIOR_ORIGINS)),
        created_before,
        created_before,
    ]


_GHOST_SELECT_SQL: str = (
    "SELECT b.id AS id, b.alpha AS alpha, b.beta AS beta, "
    "substr(b.content, 1, 100) AS preview "
    "FROM beliefs b "
    "WHERE b.lock_level = 'none' "
    "  AND b.alpha > ? " + _ELIGIBILITY_SQL + "ORDER BY b.alpha DESC LIMIT ?"
)
"""Enumeration query. Parameters: threshold_alpha, *eligibility, limit.

``LIMIT`` is always bound rather than conditionally appended; SQLite
reads a negative limit as no upper bound, so the no-cap case passes
``-1`` instead of building a second query string."""


_GHOST_RECHECK_SQL: str = (
    "SELECT b.alpha AS alpha "
    "FROM beliefs b "
    "WHERE b.id = ? "
    "  AND b.lock_level = 'none' "
    "  AND b.alpha > ? " + _ELIGIBILITY_SQL
)
"""Under-the-write-lock re-check. Parameters: id, target_alpha, *eligibility."""


_NO_LIMIT: int = -1
"""``LIMIT`` value meaning "no cap" — SQLite treats a negative limit as unbounded."""


CLAMP_SOURCE: str = "clamp_ghosts"
"""``feedback_history.source`` value for clamp-driven audit rows.

Stable across versions: downstream forensic queries
(``SELECT ... WHERE source = 'clamp_ghosts'``) depend on this string."""


DEFAULT_THRESHOLD_ALPHA: float = 4.0
"""α above which a row is considered a ghost candidate. At α=4 the
posterior is ~0.8; below that the row is already in the normal
high-recall band and clamping would be a no-op or worse."""


DEFAULT_TARGET_ALPHA: float = 4.0
"""α value the clamp writes to matching rows. Same number as the
threshold by design — anything matched (α > 4) gets pulled back to
α = 4."""


@dataclass
class ClampResult:
    """Outcome of one clamp_ghost_alphas invocation.

    The count fields satisfy `matched == clamped + skipped`. `matched`
    is the number of rows selected by the SQL filter, `clamped` the
    number actually mutated, and `skipped` is `matched - clamped`.
    In the dry-run path nothing is mutated, so `skipped == matched`
    and `clamped == 0`. In the apply path a matched row is skipped
    rather than clamped only when it fails the eligibility re-check
    performed under the write lock (e.g. a concurrent --apply run
    already clamped it), so `matched` and `skipped` are not
    generally equal there.
    """
    matched: int
    clamped: int
    skipped: int
    dry_run: bool
    threshold_alpha: float
    target_alpha: float
    sample: list[dict] = field(default_factory=list)


def clamp_ghost_alphas(
    store: MemoryStore,
    *,
    threshold_alpha: float = DEFAULT_THRESHOLD_ALPHA,
    target_alpha: float = DEFAULT_TARGET_ALPHA,
    dry_run: bool = True,
    limit: int | None = None,
    created_before: str | None = None,
    stderr: IO[str] | None = None,
) -> ClampResult:
    """Clamp α to ``target_alpha`` on ghost-α belief rows.

    A "ghost-α" row is one where:

    - ``lock_level = 'none'`` (locked beliefs are never clamped — their
      α reflects an explicit user assertion)
    - ``origin`` is not in ``USER_PRIOR_ORIGINS`` (a user-sourced row is
      born at the full type prior with no audit trail, so an empty trail
      says nothing about it)
    - ``alpha > threshold_alpha``
    - ``created_at < created_before``, when that cutoff is supplied
    - no rows in ``feedback_history`` for this belief
    - no rows in ``belief_corroborations`` for this belief

    Under ``dry_run=True`` (the default) the function returns the
    sample without mutating anything. Pass ``dry_run=False`` to apply.

    Within the apply path, each row's clamp is one ``BEGIN IMMEDIATE``
    transaction containing two writes: the α update on ``beliefs``
    and one negative-valence row on ``feedback_history``. Mirrors the
    transactional shape of ``deferred_feedback.sweep_deferred_feedback``
    so the same atomicity guarantees hold.

    Args:
        store: open MemoryStore.
        threshold_alpha: α floor above which a row is a candidate.
        target_alpha: α value to clamp matching rows down to.
        dry_run: when True, enumerate matches but do not mutate.
        limit: optional cap on rows processed in one call (None = no cap).
        created_before: optional ISO-8601 ``created_at`` cutoff; only
            rows created strictly before it are candidates. Lets an
            operator confine a one-shot clamp to rows that predate the
            migration that produced the ghosts.
        stderr: optional stream for any non-fatal warnings.

    Returns:
        ClampResult with counts and a 10-row sample of matches.

    Raises:
        ValueError: if threshold_alpha < target_alpha (would no-op
            silently); if either value is non-positive.
    """
    if threshold_alpha <= 0 or target_alpha <= 0:
        raise ValueError(
            f"threshold_alpha and target_alpha must be positive; "
            f"got threshold={threshold_alpha} target={target_alpha}"
        )
    if target_alpha > threshold_alpha:
        raise ValueError(
            f"target_alpha ({target_alpha}) must be <= threshold_alpha "
            f"({threshold_alpha}); otherwise no row would be clamped down"
        )

    conn = store._conn  # noqa: SLF001 — same access pattern as deferred_feedback.py:300

    # Normalise falsy-but-not-None to None so the `? IS NULL` arm of
    # _ELIGIBILITY_SQL reproduces the previous truthiness test exactly:
    # created_before="" must mean "no cutoff", not "created before the
    # empty string" (which would match nothing).
    created_before = created_before or None

    params: list[object] = [threshold_alpha]
    params.extend(_eligibility_params(created_before))
    params.append(_NO_LIMIT if limit is None else int(limit))

    rows = conn.execute(_GHOST_SELECT_SQL, params).fetchall()
    matched = len(rows)
    sample = [
        {
            "id": r["id"],
            "prior_alpha": round(float(r["alpha"]), 3),
            "preview": (r["preview"] or "")[:80],
        }
        for r in rows[:10]
    ]

    if dry_run:
        return ClampResult(
            matched=matched,
            clamped=0,
            skipped=matched,
            dry_run=True,
            threshold_alpha=threshold_alpha,
            target_alpha=target_alpha,
            sample=sample,
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    clamped = 0
    for r in rows:
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Re-check eligibility under the write lock so concurrent
            # --apply runs can't both clamp the same belief and insert
            # duplicate CLAMP_SOURCE audit rows (breaks the one-row
            # reversible-audit invariant).
            recheck_params: list[object] = [r["id"], target_alpha]
            recheck_params.extend(_eligibility_params(created_before))
            current = conn.execute(_GHOST_RECHECK_SQL, recheck_params).fetchone()
            if current is None:
                conn.execute("ROLLBACK")
                continue
            prior = float(current["alpha"])
            delta = -(prior - target_alpha)  # negative valence: reversible audit
            conn.execute(
                "UPDATE beliefs SET alpha = ? WHERE id = ?",
                (target_alpha, r["id"]),
            )
            conn.execute(
                "INSERT INTO feedback_history "
                "(belief_id, valence, source, created_at) "
                "VALUES (?, ?, ?, ?)",
                (r["id"], delta, CLAMP_SOURCE, now_iso),
            )
            conn.execute("COMMIT")
            clamped += 1
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return ClampResult(
        matched=matched,
        clamped=clamped,
        skipped=matched - clamped,
        dry_run=False,
        threshold_alpha=threshold_alpha,
        target_alpha=target_alpha,
        sample=sample,
    )
