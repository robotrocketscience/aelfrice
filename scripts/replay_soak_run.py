#!/usr/bin/env python3
"""Daily replay-soak runner (#403 A).

Invoked by `.github/workflows/replay-soak.yml` on a cron schedule
against `main`. Loads `tests/corpus/replay_soak/v0.1/`, runs the
fixture through `replay_full_equality`, and appends one row to
`.replay-soak-status.json` (append-only JSONL, despite the `.json`
suffix per the 2026-05-04 ratification).

Per-row fields (ratified):

- `date`: UTC date the run executed (`YYYY-MM-DD`).
- `sha`: HEAD sha of the run.
- `replay_full_equality_result`: `"pass"` if `has_drift is False`,
  `"fail"` otherwise.
- `total_log_rows`: rows the harness derived.
- `mismatched`: drift bucket count.
- `derived_orphan`: drift bucket count.

Exit code: 0 on pass; 1 on fail. The workflow does not block on
failure — soak history records the streak; the PR check (deliverable
C) interprets it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure the repository's `tests/` is importable when running from the
# workflow's checkout (no editable install needed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aelfrice.store import MemoryStore  # noqa: E402
from tests.replay_soak_runner import run_replay_soak  # noqa: E402


def _git_head_sha(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=_REPO_ROOT / ".replay-soak-status.json",
        help="Path to the append-only JSONL status file.",
    )
    parser.add_argument(
        "--today",
        type=str,
        default=None,
        help="Override today's date (YYYY-MM-DD). For deterministic tests.",
    )
    args = parser.parse_args()

    today = (
        args.today
        if args.today is not None
        else datetime.now(timezone.utc).date().isoformat()
    )
    sha = _git_head_sha(_REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "replay_soak.db"
        store = MemoryStore(str(db_path))
        try:
            report = run_replay_soak(store)
        finally:
            store.close()

    entry = {
        "date": today,
        "sha": sha,
        "replay_full_equality_result": "pass" if not report.has_drift else "fail",
        "total_log_rows": report.total_log_rows,
        "mismatched": report.mismatched,
        "derived_orphan": report.derived_orphan,
    }

    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    with args.status_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(json.dumps(entry, indent=2))
    return 0 if not report.has_drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
