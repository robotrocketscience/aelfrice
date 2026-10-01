"""#1622: the fix command follows the real platform, not a patched one.

`tests/test_lock_gaps_1622.py` pins `_on_windows` with an autouse fixture so
each branch can be tested on any host. That leaves the probe itself
untested: inverted, or hard-coded either way, it passes there. This file
has no such fixture, so it reads the platform the code really runs on.
"""
from __future__ import annotations

import hashlib
import os

from aelfrice import lock_gaps
from aelfrice.lock_gaps import LockGap


def test_the_probe_reports_the_real_platform() -> None:
    assert lock_gaps._on_windows() is (os.name == "nt")  # pyright: ignore[reportPrivateUsage]


def test_the_fix_command_follows_the_real_platform() -> None:
    statement = "Keep every widget in the blue drawer."
    gap = LockGap(
        hashlib.sha256(statement.encode("utf-8")).hexdigest(), statement,
        len(statement), "exception", "2026-01-01T00:00:00Z", None, 1,
    )
    if os.name == "nt":
        assert gap.fix_command is None
    else:
        assert gap.fix_command == f"aelf lock '{statement}'"
