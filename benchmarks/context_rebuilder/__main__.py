"""Command-line entry point for `python -m benchmarks.context_rebuilder.replay`.

Invocation form (per the v1.4.0 acceptance criteria):

    python -m benchmarks.context_rebuilder.replay <fixture> [--clear-at N] [--out PATH]

Default behaviour: replay the fixture full-baseline (no clear
injection), print the resulting JSON to stdout. With `--clear-at N`,
inject a synthetic context-clear at content-turn index N. With
`--out PATH`, write the JSON to that path instead of stdout (the
parent directory is created if missing).

Scaffolding only. The fidelity scorer (#138) lands later; this
entry point only verifies the harness *runs* against a synthetic
fixture and emits the documented output schema.

Note on module-as-script: Python's `-m` flag treats the package's
`__main__.py` as the module body when the package itself is named.
The issue's acceptance phrasing (`python -m
benchmarks.context_rebuilder.replay <fixture>`) names the `replay`
submodule explicitly, so this file mirrors `python -m
benchmarks.context_rebuilder` to the same CLI for both invocation
forms. `replay.py` does not have its own `if __name__ ==
"__main__":` block; it routes through here via the
`benchmarks.context_rebuilder.replay` module's import-time alias
established at the bottom of this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from benchmarks.context_rebuilder.inject import ClearInjection
from benchmarks.context_rebuilder.replay import (
    DEFAULT_REBUILD_OVERHEAD_TOKENS,
    FixtureError,
    run,
)
from benchmarks.context_rebuilder.score import DEFAULT_SCORE_METHOD, ScoreMethod

#: Score-method strings the CLI accepts. Mirrors `score.ScoreMethod`
#: but spelled as a tuple of strings so argparse can validate via
#: `choices=`. Methods other than `'exact'` raise NotImplementedError
#: at the scorer call site (parked at v1.4.0); we still surface them
#: in `--score-method --help` so the toggle path is documented.
_SCORE_METHOD_CHOICES: tuple[str, ...] = ("exact", "embedding", "llm-judge")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.context_rebuilder.replay",
        description=(
            "Context-rebuilder eval-harness replay. Scaffolding only -- "
            "does NOT score continuation fidelity. Fidelity scoring is #138."
        ),
    )
    p.add_argument(
        "fixture",
        type=Path,
        help="Path to a synthetic turns.jsonl fixture.",
    )
    p.add_argument(
        "--clear-at",
        type=int,
        default=None,
        metavar="N",
        help=(
            "0-based content-turn index at which to inject a synthetic "
            "context-clear. Omitted = no injection (full-replay baseline)."
        ),
    )
    p.add_argument(
        "--rebuild-overhead-tokens",
        type=int,
        default=DEFAULT_REBUILD_OVERHEAD_TOKENS,
        metavar="N",
        help=(
            f"Synthetic rebuild-block size in tokens. Default: "
            f"{DEFAULT_REBUILD_OVERHEAD_TOKENS}."
        ),
    )
    p.add_argument(
        "--score-method",
        choices=_SCORE_METHOD_CHOICES,
        default=DEFAULT_SCORE_METHOD,
        help=(
            "Continuation-fidelity scoring method. Default: "
            f"'{DEFAULT_SCORE_METHOD}' (deterministic, reproducible, "
            "no outbound calls). 'embedding' and 'llm-judge' are "
            "parked at v1.4.0 and raise NotImplementedError -- see "
            "benchmarks/context_rebuilder/score.py and #138."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write the JSON output to PATH instead of stdout. Parent "
            "directories are created if missing."
        ),
    )
    return p


def main(
    argv: list[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """CLI entry point. Returns 0 on success, non-zero on fixture errors.

    Argument parsing failures still bypass this function (argparse
    calls `sys.exit(2)` before returning to us); we treat that as
    "exit code 2 from argparse" and don't try to override it.
    """
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    parser = _build_parser()
    args = parser.parse_args(argv)

    fixture: Path = args.fixture  # type: ignore[assignment]
    clear_at: int | None = args.clear_at  # type: ignore[assignment]
    rebuild_overhead: int = args.rebuild_overhead_tokens  # type: ignore[assignment]
    out_path: Path | None = args.out  # type: ignore[assignment]
    score_method_raw: str = args.score_method  # type: ignore[assignment]

    # `argparse choices=` already restricts to _SCORE_METHOD_CHOICES;
    # the `cast` keeps pyright strict-happy without re-validating.
    score_method: ScoreMethod = score_method_raw  # type: ignore[assignment]

    inject: ClearInjection | None = None
    if clear_at is not None:
        try:
            inject = ClearInjection(clear_at=clear_at)
        except ValueError as exc:
            print(f"error: {exc}", file=err_stream)
            return 2

    try:
        result = run(
            fixture,
            inject=inject,
            rebuild_overhead_tokens=rebuild_overhead,
            score_method=score_method,
        )
    except FixtureError as exc:
        print(f"error: {exc}", file=err_stream)
        return 1
    except NotImplementedError as exc:
        # Parked-method paths (`embedding`, `llm-judge`). Surface as
        # exit 2 (caller config error), not exit 1 (fixture error).
        print(f"error: {exc}", file=err_stream)
        return 2

    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}", file=out_stream)
    else:
        print(payload, file=out_stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
