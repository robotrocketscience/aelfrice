"""Cadence shadow-log scoring (#875).

Reads ``<project>/.git/aelfrice/cadence_shadow/<session_id>.jsonl`` rows
written by the Stop-hook shadow-evaluation path and emits aggregate
statistics: per-policy fire-rate (counts of ``would_fire=true``),
selected-policy live fire-rate, and a 2x2 agreement matrix between the
original two policies (P1 vs P2), retained for backward compatibility —
see ``pairwise_agreement`` on ``ShadowSummary`` for the generalised
table across all four currently-implemented policies.

Designed to surface as ``aelf cadence-score``. The scoring is offline:
inputs are the already-written shadow log; no live cadence state is
read. Determinism (#605): same on-disk rows -> same report. Discretion
(``ab96e9d3501b1c14``): consumes only the shadow log; no
``~/.claude/``-derived state.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aelfrice.cadence import (
    CADENCE_SHADOW_DIRNAME,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
)


@dataclass(frozen=True)
class ShadowSummary:
    """Aggregate statistics over a set of shadow-log rows.

    ``per_policy_fire_count`` and ``per_policy_total`` are keyed by
    policy name — these already span every policy present in the rows
    (p1, p2, p3_velocity, p3_substantive). ``selected_fire_count``
    counts rows where the selected policy fired live.

    ``agreement_matrix`` is the legacy 2x2 ``{(p1_fire, p2_fire):
    count}`` dict, retained for backward compatibility.
    ``pairwise_agreement`` generalises it to every unordered policy
    pair present in the rows (#876 four-policy bake): keyed by a
    lexicographically-ordered ``(policy_a, policy_b)`` tuple, the value
    is a count dict with ``both`` / ``a_only`` / ``b_only`` / ``neither``
    — the head-to-head divergence the bake reads to pick a winner.
    """
    total_rows: int
    sessions: int
    earliest_ts: str | None
    latest_ts: str | None
    per_policy_fire_count: dict[str, int] = field(default_factory=dict)
    per_policy_total: dict[str, int] = field(default_factory=dict)
    selected_fire_count: dict[str, int] = field(default_factory=dict)
    selected_total: dict[str, int] = field(default_factory=dict)
    agreement_matrix: dict[tuple[bool, bool], int] = field(default_factory=dict)
    pairwise_agreement: dict[tuple[str, str], dict[str, int]] = field(
        default_factory=dict
    )


def iter_shadow_rows(shadow_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON rows from every ``*.jsonl`` under ``shadow_dir``.

    Malformed lines / unreadable files are skipped silently — the
    scoring path is best-effort and shouldn't fail on a single bad row.
    Yields in filesystem-iteration order; caller sorts if a stable
    order matters (the summary aggregates so order is irrelevant).
    """
    if not shadow_dir.is_dir():
        return
    for path in shadow_dir.glob("*.jsonl"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def compute_summary(
    rows: Iterable[dict[str, Any]],
    *,
    session_filter: str | None = None,
) -> ShadowSummary:
    """Aggregate shadow rows into a :class:`ShadowSummary`.

    ``session_filter`` (optional): if given, drop rows whose
    ``session_id`` field does not exactly match. Used by the CLI
    ``--session`` flag.

    Returns a frozen summary with totals, per-policy counts, the legacy
    2x2 P1/P2 agreement matrix, and the generalised pairwise-agreement
    table across every policy pair present. Empty input -> all zeros.
    """
    total = 0
    sessions: set[str] = set()
    earliest: str | None = None
    latest: str | None = None
    per_policy_fire: dict[str, int] = {}
    per_policy_total: dict[str, int] = {}
    selected_fire: dict[str, int] = {}
    selected_total: dict[str, int] = {}
    agreement: dict[tuple[bool, bool], int] = {
        (True, True): 0,
        (True, False): 0,
        (False, True): 0,
        (False, False): 0,
    }
    pairwise: dict[tuple[str, str], dict[str, int]] = {}

    for row in rows:
        sid = row.get("session_id")
        if session_filter is not None and sid != session_filter:
            continue
        total += 1
        if isinstance(sid, str):
            sessions.add(sid)
        ts = row.get("ts")
        if isinstance(ts, str):
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts

        selected = row.get("selected")
        fired = row.get("fired")
        if isinstance(selected, str):
            selected_total[selected] = selected_total.get(selected, 0) + 1
            if fired is True:
                selected_fire[selected] = selected_fire.get(selected, 0) + 1

        shadow = row.get("shadow")
        if not isinstance(shadow, dict):
            continue

        p1_fire = False
        p2_fire = False
        row_fire: dict[str, bool] = {}
        for policy, decision in shadow.items():
            if not isinstance(decision, dict):
                continue
            would = decision.get("would_fire")
            if not isinstance(would, bool):
                continue
            per_policy_total[policy] = per_policy_total.get(policy, 0) + 1
            if would:
                per_policy_fire[policy] = per_policy_fire.get(policy, 0) + 1
            row_fire[policy] = would
            if policy == POLICY_P1_EVERY_K_TURNS:
                p1_fire = would
            elif policy == POLICY_P2_CTX_THRESHOLD:
                p2_fire = would
        agreement[(p1_fire, p2_fire)] += 1

        # Generalised pairwise agreement over every unordered policy pair
        # whose decision was logged on this row. The pair key is ordered
        # lexicographically (a < b); ``a_only``/``b_only`` follow that order.
        policies = sorted(row_fire)
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                a, b = policies[i], policies[j]
                cell = pairwise.setdefault(
                    (a, b),
                    {"both": 0, "a_only": 0, "b_only": 0, "neither": 0},
                )
                fa, fb = row_fire[a], row_fire[b]
                if fa and fb:
                    cell["both"] += 1
                elif fa:
                    cell["a_only"] += 1
                elif fb:
                    cell["b_only"] += 1
                else:
                    cell["neither"] += 1

    return ShadowSummary(
        total_rows=total,
        sessions=len(sessions),
        earliest_ts=earliest,
        latest_ts=latest,
        per_policy_fire_count=per_policy_fire,
        per_policy_total=per_policy_total,
        selected_fire_count=selected_fire,
        selected_total=selected_total,
        agreement_matrix=agreement,
        pairwise_agreement=pairwise,
    )


def _rate(num: int, denom: int) -> str:
    if denom <= 0:
        return "0/0 (n/a)"
    pct = (num / denom) * 100
    return f"{num}/{denom} ({pct:.1f}%)"


def format_report(summary: ShadowSummary, *, as_json: bool = False) -> str:
    """Render a :class:`ShadowSummary` as a human-readable report.

    Pass ``as_json=True`` for a machine-readable JSON object suitable
    for piping into downstream tooling. Defaults to the human form.
    """
    if as_json:
        payload = {
            "total_rows": summary.total_rows,
            "sessions": summary.sessions,
            "earliest_ts": summary.earliest_ts,
            "latest_ts": summary.latest_ts,
            "per_policy_fire_count": summary.per_policy_fire_count,
            "per_policy_total": summary.per_policy_total,
            "selected_fire_count": summary.selected_fire_count,
            "selected_total": summary.selected_total,
            "agreement_matrix": {
                f"p1={int(k[0])},p2={int(k[1])}": v
                for k, v in summary.agreement_matrix.items()
            },
            "pairwise_agreement": {
                f"{a}|{b}": cell
                for (a, b), cell in summary.pairwise_agreement.items()
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    lines: list[str] = []
    lines.append(f"total rows:    {summary.total_rows}")
    lines.append(f"sessions:      {summary.sessions}")
    lines.append(
        f"date range:    {summary.earliest_ts or '-'} .. {summary.latest_ts or '-'}"
    )
    lines.append("")
    lines.append("per-policy would_fire rate:")
    for policy in sorted(summary.per_policy_total):
        num = summary.per_policy_fire_count.get(policy, 0)
        denom = summary.per_policy_total[policy]
        lines.append(f"  {policy:<24} {_rate(num, denom)}")
    lines.append("")
    lines.append("selected-policy live fire rate:")
    for policy in sorted(summary.selected_total):
        num = summary.selected_fire_count.get(policy, 0)
        denom = summary.selected_total[policy]
        lines.append(f"  {policy:<24} {_rate(num, denom)}")
    lines.append("")
    lines.append("P1 vs P2 agreement matrix:")
    lines.append("                p2=fire   p2=skip")
    lines.append(
        "  p1=fire     "
        f"{summary.agreement_matrix.get((True, True), 0):>9}"
        f"{summary.agreement_matrix.get((True, False), 0):>9}"
    )
    lines.append(
        "  p1=skip     "
        f"{summary.agreement_matrix.get((False, True), 0):>9}"
        f"{summary.agreement_matrix.get((False, False), 0):>9}"
    )
    if summary.pairwise_agreement:
        lines.append("")
        lines.append("pairwise policy agreement (both / a-only / b-only / neither):")
        for (a, b) in sorted(summary.pairwise_agreement):
            cell = summary.pairwise_agreement[(a, b)]
            both = cell.get("both", 0)
            a_only = cell.get("a_only", 0)
            b_only = cell.get("b_only", 0)
            neither = cell.get("neither", 0)
            n = both + a_only + b_only + neither
            diverge = a_only + b_only
            div_pct = (diverge / n * 100) if n else 0.0
            lines.append(
                f"  {a} vs {b}: "
                f"{both}/{a_only}/{b_only}/{neither} "
                f"(diverge {diverge}/{n} = {div_pct:.1f}%)"
            )
    return "\n".join(lines) + "\n"


def resolve_shadow_dir(project_root: Path) -> Path:
    """Resolve <project_root>/.git/aelfrice/cadence_shadow."""
    return project_root / ".git" / "aelfrice" / CADENCE_SHADOW_DIRNAME
