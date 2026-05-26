"""Offline deterministic replay bench for cadence policies (#876).

The cadence-policy bake (#749 / #875 shadow mode) normally needs an
operator-week of live shadow data before ``aelf cadence-score`` can
compare policies. This harness produces the *same* comparison from a
self-contained synthetic fixture instead — feeding per-tick inputs
through every implemented ``would_fire`` predicate (p1, p2,
p3_velocity, p3_substantive) and emitting shadow-log-shaped rows that
:func:`aelfrice.cadence_score.compute_summary` aggregates exactly as it
would a live bake.

Determinism (#605): same fixture → same report. No wall-clock, no
random sampling, no network. Discretion (``ab96e9d3501b1c14``): the
fixture is synthetic input authored alongside the bench; the harness
reads no ``~/.claude/``-derived state and writes only the capture the
caller names.

Fixture schema (JSON)::

    {
      "config": {
        "k": 15,
        "ctx_threshold": 0.8,
        "ctx_byte_window": 100000,
        "p3_velocity_threshold": 3000,
        "p3_substantive_window": 5,
        "p3_substantive_threshold": 0.6
      },
      "selected": "p1_every_k_turns",   # optional; default "off"
      "ticks": [
        {
          "fire_idx": 7,
          "bytes_at_last_fire": 0,
          "fire_idx_at_last_fire": 0,
          "transcript_bytes": 70000,
          "last_prompt": "ok next",
          "classifications": [true, true, true, true, true]
        },
        ...
      ]
    }

Each tick yields one shadow row. The optional ``selected`` policy (also
overridable per-tick) drives the ``fired`` field so the selected-policy
live-fire rate is meaningful; the four ``would_fire`` decisions are
policy-agnostic and always recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from aelfrice.cadence import (
    CadenceConfig,
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
    POLICY_P3_SUBSTANTIVE,
    POLICY_P3_VELOCITY,
    estimate_transcript_bytes,
    would_fire_p1,
    would_fire_p2,
    would_fire_p3_substantive,
    would_fire_p3_velocity,
)
from aelfrice.cadence_score import compute_summary, format_report


def _config_from_fixture(raw: dict[str, Any]) -> CadenceConfig:
    """Build a CadenceConfig from the fixture ``config`` block.

    Missing knobs fall back to CadenceConfig defaults. ``enabled`` and
    ``policy`` are forced (enabled=True, policy=OFF) — the replay
    evaluates policy-agnostic would_fire predicates, so the selected
    policy on the config object is irrelevant to the comparison.
    """
    cfg = raw.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}
    defaults = CadenceConfig()
    return CadenceConfig(
        enabled=True,
        policy=POLICY_OFF,
        k=int(cfg.get("k", defaults.k)),
        ctx_threshold=float(cfg.get("ctx_threshold", defaults.ctx_threshold)),
        ctx_byte_window=int(cfg.get("ctx_byte_window", defaults.ctx_byte_window)),
        p3_velocity_threshold=int(
            cfg.get("p3_velocity_threshold", defaults.p3_velocity_threshold)
        ),
        p3_substantive_window=int(
            cfg.get("p3_substantive_window", defaults.p3_substantive_window)
        ),
        p3_substantive_threshold=float(
            cfg.get("p3_substantive_threshold", defaults.p3_substantive_threshold)
        ),
    )


def _write_transcript(path: Path, target_bytes: int, last_prompt: str) -> None:
    """Write a synthetic transcript of ~``target_bytes`` ending in a user line.

    The final line is the user prompt (so ``read_last_user_prompt`` and the
    P2 phase-boundary check see it); preceding filler assistant lines pad
    the file toward ``target_bytes``. The exact size need not be byte-perfect
    — fixtures pick byte counts comfortably above/below the watermark.
    """
    user_line = json.dumps({"message": {"role": "user", "content": last_prompt}})
    filler = json.dumps({"message": {"role": "assistant", "content": "x" * 800}})
    user_size = len(user_line.encode("utf-8")) + 1
    filler_size = len(filler.encode("utf-8")) + 1
    pad = max(0, target_bytes - user_size)
    n_filler = pad // filler_size
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n_filler):
            f.write(filler + "\n")
        f.write(user_line + "\n")


def replay_fixture(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay every tick in ``fixture`` into shadow-log-shaped rows.

    Returns a list of rows with the same shape the Stop-hook shadow
    logger writes, so :func:`compute_summary` aggregates them directly.
    Deterministic and pure aside from the per-tick temp transcript file
    (written inside a TemporaryDirectory and discarded).
    """
    cfg = _config_from_fixture(fixture)
    default_selected = str(fixture.get("selected", POLICY_OFF))
    ticks = fixture.get("ticks")
    ticks = ticks if isinstance(ticks, list) else []

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cadence_replay_") as tmpdir:
        tmp = Path(tmpdir)
        for idx, tick in enumerate(ticks):
            if not isinstance(tick, dict):
                continue
            fire_idx = int(tick.get("fire_idx", 0))
            bytes_at_last_fire = int(tick.get("bytes_at_last_fire", 0))
            fire_idx_at_last_fire = int(tick.get("fire_idx_at_last_fire", 0))
            target_bytes = int(tick.get("transcript_bytes", 0))
            last_prompt = str(tick.get("last_prompt", ""))
            raw_classes = tick.get("classifications")
            classifications = (
                [bool(c) for c in raw_classes] if isinstance(raw_classes, list) else []
            )
            selected = str(tick.get("selected", default_selected))

            transcript = tmp / f"tick-{idx}.jsonl"
            _write_transcript(transcript, target_bytes, last_prompt)
            # Use the actual written file size for both P2 (reads the file)
            # and p3_velocity (takes the int), so the two predicates see a
            # consistent transcript-byte figure.
            actual_bytes = estimate_transcript_bytes(transcript)
            turns_since = fire_idx - fire_idx_at_last_fire
            window = cfg.p3_substantive_window
            substantive_count = sum(
                1 for c in classifications[-window:] if c is True
            )

            p1_fire, p1_reason = would_fire_p1(fire_idx=fire_idx, config=cfg)
            p2_fire, p2_reason = would_fire_p2(
                transcript_path=transcript,
                last_user_prompt=last_prompt,
                config=cfg,
            )
            p3v_fire, p3v_reason = would_fire_p3_velocity(
                bytes_at_last_fire=bytes_at_last_fire,
                transcript_bytes=actual_bytes,
                turns_since_last_fire=turns_since,
                config=cfg,
            )
            p3s_fire, p3s_reason = would_fire_p3_substantive(
                substantive_count=substantive_count,
                config=cfg,
            )

            fired_by_selected = {
                POLICY_P1_EVERY_K_TURNS: p1_fire,
                POLICY_P2_CTX_THRESHOLD: p2_fire,
                POLICY_P3_VELOCITY: p3v_fire,
                POLICY_P3_SUBSTANTIVE: p3s_fire,
            }.get(selected, False)

            rows.append({
                "ts": f"replay-{idx:06d}",
                "session_id": "replay",
                "selected": selected,
                "fired": fired_by_selected,
                "shadow": {
                    POLICY_P1_EVERY_K_TURNS: {
                        "would_fire": p1_fire, "reason": p1_reason,
                    },
                    POLICY_P2_CTX_THRESHOLD: {
                        "would_fire": p2_fire, "reason": p2_reason,
                    },
                    POLICY_P3_VELOCITY: {
                        "would_fire": p3v_fire, "reason": p3v_reason,
                    },
                    POLICY_P3_SUBSTANTIVE: {
                        "would_fire": p3s_fire, "reason": p3s_reason,
                    },
                },
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline deterministic replay bench for cadence policies (#876)",
    )
    parser.add_argument(
        "--fixture", required=True, metavar="PATH",
        help="Path to the JSON replay fixture.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the ShadowSummary as JSON instead of the text report.",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Write the JSON summary to PATH (in addition to stdout).",
    )
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    if not fixture_path.is_file():
        print(f"cadence-replay: fixture not found: {fixture_path}", file=sys.stderr)
        return 2
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cadence-replay: cannot read fixture: {exc}", file=sys.stderr)
        return 2
    if not isinstance(fixture, dict):
        print("cadence-replay: fixture must be a JSON object", file=sys.stderr)
        return 2

    rows = replay_fixture(fixture)
    summary = compute_summary(rows)
    print(format_report(summary, as_json=args.json), end="")
    if args.output:
        Path(args.output).write_text(
            format_report(summary, as_json=True), encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
