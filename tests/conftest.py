"""Shared pytest fixtures (#319 v2.0 bench-gate harness).

The bench-gate harness consumes the v2.0 evaluation corpus from #307. Corpus
content lives in the private lab repo only; the public repo carries the
schema contract and harness scaffold. The `AELFRICE_CORPUS_ROOT` env var
points the harness at a mounted corpus. When the var is unset, or when a
specific module directory is empty, bench-gate tests skip cleanly so public
CI passes without corpus access.

See `tests/corpus/v2_0/README.md` for the schema contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

CORPUS_ENV_VAR = "AELFRICE_CORPUS_ROOT"


def _corpus_root() -> Path | None:
    raw = os.environ.get(CORPUS_ENV_VAR)
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


@pytest.fixture(scope="session")
def aelfrice_corpus_root() -> Path:
    """Resolve `AELFRICE_CORPUS_ROOT` to a directory; skip the test otherwise.

    Tests that depend on labeled corpus rows should request this fixture and
    will skip on public CI where the env var is unset.
    """
    root = _corpus_root()
    if root is None:
        pytest.skip(
            f"{CORPUS_ENV_VAR} not set or not a directory; "
            "skipping bench-gate test (lab corpus absent)"
        )
    return root


def load_corpus_module(root: Path, module: str) -> list[dict]:
    """Load every `*.jsonl` row under `root/<module>/`. Skip if empty."""
    mod_dir = root / module
    if not mod_dir.is_dir():
        pytest.skip(f"corpus module {module!r} missing under {root}")
    rows: list[dict] = []
    for p in sorted(mod_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        pytest.skip(f"corpus module {module!r} empty under {root}")
    return rows
