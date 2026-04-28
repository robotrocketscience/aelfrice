"""Transcript-replay loader for the context-rebuilder eval harness.

Reads a `turns.jsonl` file (per `docs/transcript_ingest.md` schema)
and walks the per-turn agent state, returning a structured
`ReplayResult` that includes per-turn `token_budget_delta` and
`hook_latency_ms` measurements.

Scaffolding only -- this does NOT score continuation fidelity. That
deliverable lives in #138 (continuation-fidelity scorer).

Schema of `turns.jsonl`, lifted from `transcript_logger._build_turn_line`:

    {
      "schema_version": 1,
      "ts":             "...",
      "role":           "user" | "assistant",
      "text":           "...",
      "session_id":     "...",
      "turn_id":        "...",
      "context":        {"cwd": "...", ...}
    }

A small subset of compaction-marker lines may also appear:

    {"schema_version": 1, "ts": "...", "event": "compaction_start"}
    {"schema_version": 1, "ts": "...", "event": "compaction_complete"}

These are skipped on replay -- they don't carry turn content.

The fixture shape is the contract the synthetic generator + any
captured-corpus replay both honour.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, cast

from benchmarks.context_rebuilder.inject import ClearInjection
from benchmarks.context_rebuilder.measure import (
    estimate_tokens,
    hook_latency_ms,
    token_budget_delta,
)

#: Approximate per-turn rebuild-block overhead, in tokens. Models the
#: cost of the `<aelfrice-rebuild>...<continue/></aelfrice-rebuild>`
#: envelope plus the synthetic per-turn marker. The exact value is
#: re-tuned by the v1.4.x calibration runs; this is only the
#: scaffolding default.
DEFAULT_REBUILD_OVERHEAD_TOKENS: Final[int] = 32


@dataclass(frozen=True)
class ReplayTurn:
    """One normalized turn read from a `turns.jsonl` fixture.

    Subset of the wire schema; fields the scaffolding actually uses.
    Other fields (`turn_id`, `context`, `ts`) are intentionally
    discarded -- they're not part of the per-turn measurement
    surface and pulling them through would couple the harness to
    fields the rebuilder spec doesn't read either.
    """
    index: int
    role: str
    text: str
    session_id: str | None


@dataclass(frozen=True)
class ReplayTurnResult:
    """Per-turn measurement record. One per content turn replayed.

    Three fields the v1.4.0 acceptance criterion requires:

    * `turn_index` -- 0-based content-turn index in the fixture
      (compaction markers are skipped, so this is the position
      among `(role, text)`-bearing lines, not the file line number).
    * `token_budget_delta` -- difference, in tokens, between the
      cumulative full-replay baseline at this turn and the
      rebuilder-substituted baseline. Negative deltas mean the
      rebuilder saved tokens; positive deltas mean it didn't (or a
      rebuild has not yet fired). Pre-clear, this is always 0.
    * `hook_latency_ms` -- wall-clock time spent in the rebuilder's
      hook simulation for this turn, in milliseconds. Always
      monotonic non-negative; 0 for turns where the hook did not
      fire.

    The `cleared` flag is informational -- it marks the turn at
    which the midpoint-clear injection ran (if any).
    """
    turn_index: int
    role: str
    token_budget_delta: int
    hook_latency_ms: float
    cleared: bool = False


@dataclass(frozen=True)
class ReplayResult:
    """Top-level result of one replay run.

    Serialized to JSON by `__main__` for the eval-harness JSON
    output schema (#136 acceptance criterion).
    """
    fixture_path: str
    n_turns: int
    n_skipped_lines: int
    clear_injected_at: int | None
    full_replay_baseline_tokens: int
    rebuild_block_tokens: int
    turns: list[ReplayTurnResult] = field(
        default_factory=lambda: [],  # typed-empty for pyright strict
    )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view used by `__main__`."""
        return {
            "fixture_path": self.fixture_path,
            "n_turns": self.n_turns,
            "n_skipped_lines": self.n_skipped_lines,
            "clear_injected_at": self.clear_injected_at,
            "full_replay_baseline_tokens": self.full_replay_baseline_tokens,
            "rebuild_block_tokens": self.rebuild_block_tokens,
            "turns": [asdict(t) for t in self.turns],
        }


class FixtureError(Exception):
    """Raised when a fixture is missing, empty, or unreadable.

    Distinct from JSON / schema errors on individual lines, which
    are skipped silently and counted in `n_skipped_lines` -- the
    same tolerance the production transcript_logger applies.
    """


def load_turns(fixture: Path) -> tuple[list[ReplayTurn], int]:
    """Read a `turns.jsonl` fixture into `ReplayTurn` records.

    Returns `(turns, n_skipped)` where `n_skipped` counts blank
    lines, malformed JSON, compaction-marker events, and lines that
    don't match the turn schema. Raises `FixtureError` if the
    fixture file is missing OR has zero usable turns; that's the
    "clear error, no crash" contract from the issue.
    """
    if not fixture.exists():
        raise FixtureError(f"fixture not found: {fixture}")
    if not fixture.is_file():
        raise FixtureError(f"fixture is not a file: {fixture}")
    try:
        raw = fixture.read_text(encoding="utf-8")
    except OSError as exc:
        raise FixtureError(f"cannot read fixture {fixture}: {exc}") from exc

    turns: list[ReplayTurn] = []
    n_skipped = 0
    for line in raw.splitlines():
        if not line.strip():
            n_skipped += 1
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n_skipped += 1
            continue
        if not isinstance(obj, dict):
            n_skipped += 1
            continue
        rec = cast(dict[str, object], obj)
        # Compaction markers carry "event" instead of "role"/"text".
        if "event" in rec and "text" not in rec:
            n_skipped += 1
            continue
        role = rec.get("role")
        text = rec.get("text")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            n_skipped += 1
            continue
        if not isinstance(text, str):
            n_skipped += 1
            continue
        sid_raw = rec.get("session_id")
        sid = sid_raw if isinstance(sid_raw, str) else None
        turns.append(
            ReplayTurn(
                index=len(turns), role=role, text=text, session_id=sid,
            )
        )

    if not turns:
        raise FixtureError(
            f"fixture has zero usable turns: {fixture} "
            f"(skipped {n_skipped} non-turn line(s))"
        )
    return turns, n_skipped


def run(
    fixture: Path,
    *,
    inject: ClearInjection | None = None,
    rebuild_overhead_tokens: int = DEFAULT_REBUILD_OVERHEAD_TOKENS,
) -> ReplayResult:
    """End-to-end replay of one fixture against the scaffolding harness.

    Walks the fixture turn-by-turn, accumulates the full-replay
    baseline token count, and -- if `inject` is provided -- forces
    a synthetic context-clear at `inject.clear_at` and substitutes
    the simulated rebuild block thereafter. Per turn, records:

      * `token_budget_delta` (signed int): rebuild-substituted
        cumulative tokens minus full-replay cumulative tokens.
      * `hook_latency_ms` (float >= 0): wall-clock for the rebuild
        simulation; 0 on turns where the hook did not fire.

    Pure-ish -- the only impurity is `time.monotonic()` for the
    latency measurement. Determinism guarantee: re-running on the
    same fixture with `inject=None` produces identical numeric
    output (apart from `hook_latency_ms`, which is always 0 in that
    branch).
    """
    turns, n_skipped = load_turns(fixture)

    # Full-replay baseline: sum of all turn token estimates.
    per_turn_tokens = [estimate_tokens(t.text) for t in turns]
    full_replay_baseline_tokens = sum(per_turn_tokens)

    cleared_at: int | None = None
    if inject is not None and inject.clear_at < len(turns):
        cleared_at = inject.clear_at

    # Rebuild block size: synthetic budget = pre-clear cumulative
    # tokens compressed to the rebuild-block overhead. The real
    # rebuilder produces a `<aelfrice-rebuild>` block; the scaffold
    # just measures size. Mirrors the spec's "rebuild block size as
    # fraction of full-replay baseline" headline ratio.
    rebuild_block_tokens = (
        rebuild_overhead_tokens if cleared_at is not None else 0
    )

    # Per-turn measurements.
    turn_results: list[ReplayTurnResult] = []
    cumulative_full = 0
    cumulative_rebuild = 0
    for t, tok in zip(turns, per_turn_tokens):
        cumulative_full += tok
        latency_ms = 0.0
        on_clear_turn = cleared_at is not None and t.index == cleared_at
        if on_clear_turn:
            t0 = time.monotonic()
            # Simulated hook work: nothing, deliberately. The scaffold
            # measures the *shape* of the latency channel, not real
            # rebuild cost. Real cost lands when #138 wires the
            # fidelity scorer in and replaces this with a call into
            # `aelfrice.context_rebuilder.rebuild()`.
            cumulative_rebuild = rebuild_block_tokens + tok
            latency_ms = hook_latency_ms(t0)
        elif cleared_at is not None and t.index > cleared_at:
            cumulative_rebuild += tok
        else:
            cumulative_rebuild = cumulative_full
        delta = token_budget_delta(
            full=cumulative_full, rebuilt=cumulative_rebuild,
        )
        turn_results.append(
            ReplayTurnResult(
                turn_index=t.index,
                role=t.role,
                token_budget_delta=delta,
                hook_latency_ms=latency_ms,
                cleared=on_clear_turn,
            )
        )

    return ReplayResult(
        fixture_path=str(fixture),
        n_turns=len(turns),
        n_skipped_lines=n_skipped,
        clear_injected_at=cleared_at,
        full_replay_baseline_tokens=full_replay_baseline_tokens,
        rebuild_block_tokens=rebuild_block_tokens,
        turns=turn_results,
    )


if __name__ == "__main__":
    # `python -m benchmarks.context_rebuilder.replay <fixture>` invokes
    # this branch. Defer to __main__.py for the CLI surface so
    # `python -m benchmarks.context_rebuilder` and `python -m
    # benchmarks.context_rebuilder.replay` produce identical behaviour.
    # Lazy import avoids the import cycle that would arise if __main__
    # were imported at top of file (it imports back from us).
    import runpy as _runpy
    import sys as _sys

    _runpy.run_module(
        "benchmarks.context_rebuilder", run_name="__main__", alter_sys=True,
    )
    _sys.exit(0)
