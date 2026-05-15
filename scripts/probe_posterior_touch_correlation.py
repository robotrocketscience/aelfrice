"""Probe posterior-touch correlation on a real aelfrice memory.db.

Diagnostic for the hot-path `belief_touches` substrate (#748 / #816 /
PR #821). Measures Spearman ρ between `posterior_mean = α / (α + β)`
(from `beliefs`) and per-session `touch_count` (from `injection_events`)
on a chosen session. Output is purely statistical; safe to copy-paste
into a public issue comment.

Use it when:

- You want to verify the #848 R7c finding on your own corpus shape.
- You're considering proposing a new H3 mechanism and want to know
  whether the synthetic-vs-production correlation gap is similar on
  your project's DB.
- You're extending the R7c sweep (more sessions, more corpora) to
  tighten the M=2 trial-equivalent verdict at #848.

This is a contributor diagnostic, not a CI gate. It reads production
data, so it runs locally only.

## What it measures

For a chosen session, computes Spearman ρ between:

- `posterior_mean(b) = α / (α + β)` per belief (from `beliefs` table).
- `touch_count(b, session_id) = count of injection_events rows`
  (from `injection_events` table, scoped to that session).

Reports two correlations:

1. **Touched-only**: ρ restricted to beliefs touched in the session
   (touch_count ≥ 1). Whether posterior tracks *frequency* of touch
   among already-touched beliefs.
2. **Touched + sampled-untouched**: ρ over all touched beliefs plus
   a random sample of untouched beliefs (touch=0) of comparable size.
   Closest match to the original R4-family comparison shape; this is
   the load-bearing number for the #848 H3-defer call.

## Decision framework (carried from R7b)

- ρ < 0.30: signal robust; consumer-build justified.
- 0.30 ≤ ρ < 0.60: signal partially survives; partial verdict.
- ρ ≥ 0.60: signal mostly synthetic-corpus artifact; consumer-defer.

Per #848, both production corpora dispatched on 2026-05-15 landed
above the 0.60 crossover (ρ_mixed = +0.87 / +0.72). Re-opening
conditions for H3 include: an extended sweep yielding ρ_mixed < 0.60
on a meaningful fraction of cells.

## Usage

    python3 scripts/probe_posterior_touch_correlation.py \\
        --db <project-root>/.git/aelfrice/memory.db \\
        [--session-id <session_id>]

    # Or via uv:
    uv run python scripts/probe_posterior_touch_correlation.py \\
        --db <project-root>/.git/aelfrice/memory.db

If `--session-id` is omitted, picks the most recent session with at
least 5 injection_events rows.

If the DB has no `injection_events` table (older aelfrice versions,
pre-#779), the script exits 2 with a clear message.

## Privacy

Reads only schema columns: `beliefs.alpha`, `beliefs.beta`,
`injection_events.belief_id`, `injection_events.session_id`,
`injection_events.injected_at`. Does NOT touch `beliefs.text`,
`belief_documents`, or any other content column. Output is purely
statistical (ρ, n samples, verdict string).
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from pathlib import Path


SPEARMAN_CROSSOVER = 0.60
ROBUST_THRESHOLD = 0.30
ARTIFACT_THRESHOLD = 0.60


def spearman_rho(a: list[float], b: list[float]) -> float:
    """Pure-stdlib Spearman rank correlation with average-rank ties."""
    n = len(a)
    if n < 2:
        return 0.0

    def _rank(xs: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[indexed[j + 1]] == xs[indexed[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    ra = _rank(a)
    rb = _rank(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    var_a = sum((ra[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((rb[i] - mean_b) ** 2 for i in range(n))
    if var_a == 0 or var_b == 0:
        return 0.0
    return num / math.sqrt(var_a * var_b)


def verdict_for_rho(rho: float) -> tuple[str, str]:
    """Map ρ to (verdict, one-paragraph implication)."""
    if rho < ROBUST_THRESHOLD:
        return (
            "BUILD_PIPELINE",
            "Production correlation is low; the synthetic top-K shift "
            "signal that motivated the deferred touch-temperature "
            "consumer survives at this corpus shape. A consumer "
            "rebuild would plausibly produce measurable fidelity gain.",
        )
    elif rho < ARTIFACT_THRESHOLD:
        return (
            "PARTIAL",
            "Production correlation is moderate; the synthetic signal "
            "partially survives. A consumer rebuild has plausible but "
            "damped expected effect; weigh build cost vs the partial "
            "expected lift.",
        )
    else:
        return (
            "SHIP_H4_ONLY",
            "Production correlation is high; the synthetic top-K shift "
            "signal mostly evaporates at this corpus shape. Consistent "
            "with the #848 H3-defer verdict.",
        )


def _pick_recent_session(
    conn: sqlite3.Connection, min_events: int = 5
) -> str | None:
    cur = conn.execute(
        """
        SELECT session_id, COUNT(*) as c, MAX(injected_at) as last
        FROM injection_events
        GROUP BY session_id
        HAVING c >= ?
        ORDER BY last DESC
        LIMIT 1
        """,
        (min_events,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _load_touched_counts(
    conn: sqlite3.Connection, session_id: str
) -> dict[str, int]:
    cur = conn.execute(
        """
        SELECT belief_id, COUNT(*) as cnt
        FROM injection_events
        WHERE session_id = ?
        GROUP BY belief_id
        """,
        (session_id,),
    )
    return {row[0]: row[1] for row in cur.fetchall()}


def _load_posterior_means(
    conn: sqlite3.Connection, belief_ids: list[str] | None = None
) -> dict[str, float]:
    if belief_ids is None:
        cur = conn.execute("SELECT id, alpha, beta FROM beliefs")
    else:
        placeholders = ",".join("?" * len(belief_ids))
        cur = conn.execute(
            f"SELECT id, alpha, beta FROM beliefs WHERE id IN ({placeholders})",
            belief_ids,
        )
    out: dict[str, float] = {}
    for bid, alpha, beta in cur.fetchall():
        denom = alpha + beta
        if denom <= 0:
            continue
        out[bid] = alpha / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe posterior-touch correlation on a real aelfrice "
            "memory.db. Outputs Spearman ρ + decision-band verdict "
            "per the #848 H3-defer framework. Privacy-safe: reads "
            "only schema columns, never text."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        help=(
            "Path to an aelfrice memory.db. Per-project DBs live at "
            "<project-root>/.git/aelfrice/memory.db."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Session to measure. If omitted, picks the most recent "
            "session with >= --min-events injection_events rows."
        ),
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="Minimum injection_events count when auto-picking a session.",
    )
    parser.add_argument(
        "--untouched-sample-ratio",
        type=float,
        default=1.0,
        help=(
            "Number of untouched beliefs to sample, as a multiple of "
            "the touched belief count. 1.0 (default) is balanced."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the untouched-sample draw. Determinism.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"ERROR: db not found at {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='injection_events'"
        )
        if cur.fetchone() is None:
            print(
                "ERROR: injection_events table missing. "
                "Requires the post-#779 schema (PR #789 onward). "
                "Update aelfrice and re-run.",
                file=sys.stderr,
            )
            return 2

        session_id = args.session_id or _pick_recent_session(
            conn, min_events=args.min_events
        )
        if session_id is None:
            print(
                f"ERROR: no session found with >= {args.min_events} "
                "injection_events. Try a smaller --min-events or "
                "pass --session-id explicitly.",
                file=sys.stderr,
            )
            return 2

        touched_counts = _load_touched_counts(conn, session_id)
        if not touched_counts:
            print(
                f"ERROR: session_id={session_id} has 0 injection_events.",
                file=sys.stderr,
            )
            return 2

        touched_posteriors = _load_posterior_means(
            conn, list(touched_counts.keys())
        )
        all_posteriors = _load_posterior_means(conn, None)

        touched_ids = sorted(
            bid for bid in touched_counts if bid in touched_posteriors
        )
        touched_post_vec = [touched_posteriors[bid] for bid in touched_ids]
        touched_count_vec = [
            float(touched_counts[bid]) for bid in touched_ids
        ]
        rho_touched = spearman_rho(touched_post_vec, touched_count_vec)

        rng = random.Random(args.seed)
        untouched_pool = sorted(
            bid for bid in all_posteriors if bid not in touched_counts
        )
        n_sample = min(
            int(len(touched_ids) * args.untouched_sample_ratio),
            len(untouched_pool),
        )
        sampled_untouched = (
            rng.sample(untouched_pool, n_sample) if n_sample > 0 else []
        )
        mixed_ids = touched_ids + sampled_untouched
        mixed_post_vec = [all_posteriors[bid] for bid in mixed_ids]
        mixed_count_vec = [
            float(touched_counts.get(bid, 0)) for bid in mixed_ids
        ]
        rho_mixed = spearman_rho(mixed_post_vec, mixed_count_vec)

    finally:
        conn.close()

    touched_verdict, _ = verdict_for_rho(rho_touched)
    mixed_verdict, mixed_implication = verdict_for_rho(rho_mixed)

    print(f"=== Posterior-touch correlation probe ===")
    print(f"  db:              {db_path}")
    print(f"  session_id:      {session_id}")
    print(f"  touched beliefs: {len(touched_ids)}")
    print(f"  untouched samp:  {len(sampled_untouched)}")
    print()
    print(
        f"  Touched-only Spearman ρ:  {rho_touched:+.4f}  "
        f"→ {touched_verdict}"
    )
    print(
        f"  Mixed (touched + cold)  ρ: {rho_mixed:+.4f}  "
        f"→ {mixed_verdict}"
    )
    print()
    print("  Decision framework (#848 / R7b crossover):")
    print(f"    ρ < 0.30  →  BUILD_PIPELINE   (signal robust)")
    print(f"    ρ < 0.60  →  PARTIAL          (signal partially survives)")
    print(f"    ρ ≥ 0.60  →  SHIP_H4_ONLY    (signal mostly artifact)")
    print()
    print("  Implication (mixed ρ — closest to R4-family comparison):")
    for line in mixed_implication.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")
    print()
    print(
        f"  Safe-to-share summary: "
        f"ρ_touched={rho_touched:+.3f}, "
        f"ρ_mixed={rho_mixed:+.3f}, "
        f"verdict={mixed_verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
