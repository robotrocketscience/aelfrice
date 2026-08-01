"""#1291 / #1177 proposal 16 — free precision proxy for attributed correction.

Proposal 16 replaces the uniform valence smear in
`sentiment_feedback.apply_sentiment_to_pending` — which credits every belief
injected on the prior turn with the *same* valence — with winner-take-all
attribution plus an abstention rule. Its own cheapest kill experiment is a free
precision proxy: over real injection sets paired with the correction that
followed them, compute the distribution of max-Jaccard.

Pre-registered kill rule, quoted from the proposal and unchanged:

    KILL if the top candidate rarely clears theta=0.15, or clears it for many
    candidates at once. PROCEED only if a substantial share of correction turns
    yield exactly one clear winner with a margin over the runner-up.

**Two corpora, measured independently, because `hook_audit.jsonl` exists twice.**
The proposal names `hook_audit.jsonl`. There is a stale copy in `~/.aelfrice/`
holding 4 rows from one 7-minute window; the live one the hooks actually write
is **repo-local**, at `<repo>/.git/aelfrice/hook_audit.jsonl` (plus a `.1`
rotation), and holds ~700 `user_prompt_submit` rows. Resolve it via
`hook._open_store().db_path` rather than assuming the dotdir — measuring the
dotdir copy silently answers the wrong question.

  * `--hook-audit` (primary) runs the proposal's experiment **as specified**.
    Each UPS row carries `prompt_prefix` and the injected `beliefs`, so
    consecutive rows in a session give (correction text, prior injected set)
    with no join at all.
  * `--transcripts` (optional) runs the same measurement over
    `injection_events` joined to the session transcript archive. Independent
    corpus, independent failure modes; agreement between the two is the point.

**`prompt_prefix` is capped at 200 chars, which is exactly
`sentiment_feedback.MAX_PROMPT_CHARS`.** That collision is load-bearing rather
than a nuisance: `detect_sentiment` ignores any prompt longer than the cap, so
every prompt it *would* act on is stored complete. Rows at the cap are dropped —
a 200-char prefix cannot be distinguished from a truncated longer prompt, and
either way it is at or over the guard.

The store is opened **read-only**; point `--db` at a copy if you want belt and
braces. Output is aggregate statistics only: no prompt text and no belief text
is printed, written or retained.

Run:
    python benchmarks/correction_attribution_proxy.py \
        --db <store.db> --hook-audit <.git/aelfrice/hook_audit.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

from aelfrice.bm25 import tokenize_stemmed
from aelfrice.relationship_detector import _STOPWORDS
from aelfrice.sentiment_feedback import (
    MAX_PROMPT_CHARS,
    NEGATIVE,
    detect_sentiment,
)

# The proposal's theta. Not a tunable here — changing it invalidates the
# pre-registered rule this script exists to adjudicate.
THETA = 0.15

# `injection_events` is 2.1% non-locked before this date and 50.3% after
# (#1016-B landed reference-tier locks). Never pool across it.
REGIME_BREAK = "2026-06-30"

# An injection fires at UserPromptSubmit for the turn it belongs to, so the
# injection row and the user turn share a wall-clock instant to within hook
# latency. 30s is generous enough to absorb a slow hook and far tighter than
# the gap between consecutive turns.
ALIGN_TOLERANCE_S = 30.0


def content_tokens(text: str) -> set[str]:
    """The proposal's `content_tokens`: stemmed tokens minus stopwords."""
    return {t for t in tokenize_stemmed(text) if t not in _STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_ts(raw: str) -> float | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def load_injections(db: Path) -> tuple[dict[str, list], dict[str, set[str]]]:
    """Read-only. Returns (session -> [(t, injected_at, belief_ids)], tokens)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT session_id, turn_id, injected_at, belief_id "
            "FROM injection_events"
        ).fetchall()
        beliefs = dict(con.execute("SELECT id, content FROM beliefs").fetchall())
    finally:
        con.close()

    turns: dict[tuple[str, str], dict] = {}
    for sid, tid, at, bid in rows:
        slot = turns.setdefault(
            (sid, tid), {"t": parse_ts(at), "at": at, "ids": []}
        )
        slot["ids"].append(bid)

    by_session: dict[str, list] = {}
    for (sid, _tid), v in turns.items():
        if v["t"] is None:
            continue
        by_session.setdefault(sid, []).append((v["t"], v["at"], v["ids"]))
    for sid in by_session:
        by_session[sid].sort()

    return by_session, {
        bid: content_tokens(c) for bid, c in beliefs.items() if c
    }


def user_turns(path: Path) -> list[tuple[float, str]]:
    """`[(epoch, text)]` for genuine user prose turns, oldest first.

    Harness-generated entries (slash-command envelopes, tool results) are not
    user prose and would inflate the denominator of the correction funnel.
    """
    out: list[tuple[float, str]] = []
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "user":
                continue
            msg = rec.get("message") or {}
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            if not text.strip():
                continue
            if "<local-command" in text or "tool_use_id" in text:
                continue
            t = parse_ts(rec.get("timestamp", ""))
            if t is not None:
                out.append((t, text))
    out.sort()
    return out


def hook_audit_rows(path: Path) -> list[dict]:
    """UPS rows from `path` and its `.1` rotation, oldest first."""
    out: list[dict] = []
    for candidate in (path, path.with_suffix(path.suffix + ".1")):
        if not candidate.is_file():
            continue
        with candidate.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("hook") == "user_prompt_submit":
                    out.append(rec)
    out.sort(key=lambda r: r.get("ts", ""))
    return out


def run_hook_audit(
    path: Path, tok: dict[str, set[str]]
) -> tuple[list[dict], int]:
    """The proposal's experiment as specified — no join required.

    Consecutive UPS rows in one session give (correction text, prior
    injected set) directly: `prompt_prefix` is this turn's prompt and the
    previous row's `beliefs` is what was injected before it.
    """
    rows = hook_audit_rows(path)
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r.get("session_id", ""), []).append(r)

    total = complete = fired = negative = 0
    records: list[dict] = []
    for _sid, session_rows in by_session.items():
        for idx, row in enumerate(session_rows):
            prompt = row.get("prompt_prefix") or ""
            total += 1
            # At the cap, a prefix cannot be told from a truncated longer
            # prompt — and either way it is at or over detect_sentiment's
            # own guard, so it is ineligible under both readings.
            if len(prompt) >= MAX_PROMPT_CHARS:
                continue
            complete += 1
            signal = detect_sentiment(prompt)
            if signal is None:
                continue
            fired += 1
            if signal.sentiment != NEGATIVE:
                continue
            negative += 1
            if idx == 0:
                continue
            prior = session_rows[idx - 1].get("beliefs") or []
            ids = [b.get("id") for b in prior if isinstance(b, dict)]
            scores = sorted(
                (jaccard(content_tokens(prompt), tok[b]) for b in ids if b in tok),
                reverse=True,
            )
            if not scores:
                continue
            records.append({
                "at": (session_rows[idx - 1].get("ts") or "")[:10],
                "p": len(scores),
                "max": scores[0],
                "runner": scores[1] if len(scores) > 1 else 0.0,
                "n_clear": sum(1 for x in scores if x >= THETA),
            })

    print()
    print("=" * 70)
    print("CORPUS A — hook_audit (the proposal's own corpus, no join)")
    print("=" * 70)
    denom = max(total, 1)
    print(f"  UPS rows                        : {total} "
          f"({len(by_session)} sessions)")
    print(f"  prompt complete (< {MAX_PROMPT_CHARS} chars)   : {complete} "
          f"({100 * complete / denom:.1f}%)  <- the rest are at/over the guard")
    print(f"  detect_sentiment fired          : {fired} "
          f"({100 * fired / denom:.2f}%)")
    print(f"  ...and NEGATIVE                 : {negative} "
          f"({100 * negative / denom:.2f}%)")
    print(f"  ...with a prior injected set    : {len(records)}")
    return records, total


def _describe(label: str, subset: list[dict]) -> None:
    print()
    print("=" * 70)
    print(f"{label}   n={len(subset)}")
    print("=" * 70)
    if not subset:
        print("  (empty)")
        return
    mx = sorted(r["max"] for r in subset)
    print(f"  |P| median            : "
          f"{statistics.median(r['p'] for r in subset):.0f}")
    print(f"  max-Jaccard           : min={mx[0]:.4f} median="
          f"{statistics.median(mx):.4f} p90={mx[int(len(mx) * 0.9)]:.4f} "
          f"max={mx[-1]:.4f}")
    clears = [r for r in subset if r["max"] >= THETA]
    print(f"  clears theta={THETA}      : {len(clears)}/{len(subset)} "
          f"({100 * len(clears) / len(subset):.1f}%)")
    if clears:
        nc = sorted(r["n_clear"] for r in clears)
        margins = [r["max"] - r["runner"] for r in clears]
        print(f"    |B*| median         : {statistics.median(nc):.0f} "
              f"(max {nc[-1]})")
        print(f"    exactly one clears  : "
              f"{sum(1 for r in clears if r['n_clear'] == 1)}/{len(clears)}")
        print(f"    winner-runner margin: {statistics.median(margins):.4f}")


def _reading_note() -> None:
    print()
    print("=" * 70)
    print("READING THIS")
    print("=" * 70)
    print("  The verdict lives in the correction funnels, not the Jaccard")
    print("  tables. If the NEGATIVE count is small, the max-Jaccard rows are")
    print("  underpowered and must not be quoted as a precision result.")
    print("  Corpus A is the proposal's own; corpus B is independent. They")
    print("  are reported separately on purpose — agreement is the evidence.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path,
                    help="store containing injection_events (opened read-only)")
    ap.add_argument("--hook-audit", type=Path, default=None,
                    help="live hook_audit.jsonl (repo-local .git/aelfrice/, "
                         "NOT the stale ~/.aelfrice copy)")
    ap.add_argument("--transcripts", type=Path, default=None,
                    help="optional second corpus: directory searched "
                         "recursively for <session-id>.jsonl")
    args = ap.parse_args(argv)

    if not args.db.is_file():
        print(f"no such store: {args.db}", file=sys.stderr)
        return 2
    if args.hook_audit is None and args.transcripts is None:
        print("give at least one corpus: --hook-audit and/or --transcripts",
              file=sys.stderr)
        return 2
    if args.hook_audit is not None and not args.hook_audit.is_file():
        print(f"no such hook-audit file: {args.hook_audit}", file=sys.stderr)
        return 2
    if args.transcripts is not None and not args.transcripts.is_dir():
        print(f"no such directory: {args.transcripts}", file=sys.stderr)
        return 2

    by_session, tok = load_injections(args.db)
    print(f"beliefs with content       : {len(tok)}")

    if args.hook_audit is not None:
        audit_records, audit_total = run_hook_audit(args.hook_audit, tok)
        if audit_total == 0:
            # An all-zero funnel reads exactly like a real result, and it
            # points the same way as this script's conclusion: a corpus
            # that is empty because the path was wrong is indistinguishable
            # from a population that is genuinely tiny. That already
            # happened once here — the stale ~/.aelfrice copy answered with
            # 4 rows and nothing complained. Refuse instead.
            print(f"no user_prompt_submit rows in {args.hook_audit} — wrong "
                  "path, or an audit that has never been written",
                  file=sys.stderr)
            return 1
        _describe("CORPUS A — max-Jaccard", audit_records)

    if args.transcripts is None:
        _reading_note()
        return 0

    index = {p.stem: p for p in args.transcripts.rglob("*.jsonl")}

    total_turns = sum(len(v) for v in by_session.values())
    print()
    print("=" * 70)
    print("CORPUS B — injection_events joined to the transcript archive")
    print("=" * 70)
    print(f"  sessions with injections : {len(by_session)}")
    print(f"  injection turns          : {total_turns}")

    joined = 0
    joined_injection_turns = 0
    user_turn_total = 0
    over_guard = 0
    fired = 0
    negative = 0
    records: list[dict] = []

    for sid, turns in by_session.items():
        tpath = index.get(sid)
        if tpath is None:
            continue
        joined += 1
        joined_injection_turns += len(turns)

        uturns = user_turns(tpath)
        user_turn_total += len(uturns)
        if not uturns:
            continue

        # Align each injection to the nearest user turn in time.
        by_time: dict[float, list[tuple[str, list[str]]]] = {}
        for t, at, ids in turns:
            best = min(uturns, key=lambda u: abs(u[0] - t))
            if abs(best[0] - t) <= ALIGN_TOLERANCE_S:
                by_time.setdefault(best[0], []).append((at, ids))

        for idx, (_t_now, text_now) in enumerate(uturns):
            if len(text_now) > MAX_PROMPT_CHARS:
                over_guard += 1
                continue
            signal = detect_sentiment(text_now)
            if signal is None:
                continue
            fired += 1
            if signal.sentiment != NEGATIVE:
                continue
            negative += 1

            # A correction at turn i is about the set injected at turn i-1.
            if idx == 0:
                continue
            prior = by_time.get(uturns[idx - 1][0])
            if not prior:
                continue
            at_prev, ids = prior[0]

            c_tok = content_tokens(text_now)
            scores = sorted(
                (jaccard(c_tok, tok[b]) for b in ids if b in tok), reverse=True
            )
            if not scores:
                continue
            records.append({
                "at": at_prev,
                "p": len(scores),
                "max": scores[0],
                "runner": scores[1] if len(scores) > 1 else 0.0,
                "n_clear": sum(1 for s in scores if s >= THETA),
            })

    print()
    print("=" * 70)
    print("CORPUS B — join feasibility")
    print("=" * 70)
    pct = 100 * joined_injection_turns / total_turns if total_turns else 0.0
    print(f"  sessions joined to a transcript : {joined}/{len(by_session)}")
    print(f"  injection turns reachable       : {joined_injection_turns}"
          f"/{total_turns} ({pct:.1f}%)")

    if user_turn_total == 0:
        # Same reasoning as corpus A: a join that reached nothing prints a
        # funnel of zeros that reads as a measured result.
        print(f"no user prose turns joined from {args.transcripts} — the "
              "transcript archive does not cover these sessions",
              file=sys.stderr)
        return 1

    print()
    print("=" * 70)
    print("CORPUS B — correction funnel")
    print("=" * 70)
    denom = max(user_turn_total, 1)
    print(f"  user prose turns                : {user_turn_total}")
    print(f"  over MAX_PROMPT_CHARS ({MAX_PROMPT_CHARS})     : {over_guard} "
          f"({100 * over_guard / denom:.1f}%)  <- None before any pattern runs")
    print(f"  detect_sentiment fired          : {fired} "
          f"({100 * fired / denom:.2f}%)")
    print(f"  ...and NEGATIVE                 : {negative} "
          f"({100 * negative / denom:.2f}%)")
    print(f"  ...with a prior injected set    : {len(records)}")

    _describe("CORPUS B — ALL", records)
    _describe(f"POST-BREAK (> {REGIME_BREAK})",
              [r for r in records if r["at"] > REGIME_BREAK])
    _describe(f"PRE-BREAK (<= {REGIME_BREAK})",
              [r for r in records if r["at"] <= REGIME_BREAK])

    _reading_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
