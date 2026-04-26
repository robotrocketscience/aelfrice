"""Minimum config surface for v0.1.0.

Just the DB path and storage defaults. Larger config (retrieval budgets,
feedback thresholds, etc.) lands with the modules that need them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# Default DB location; overridable via AELFRICE_DB env var.
DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"

# Beta-Bernoulli prior. Jeffreys (0.5, 0.5) -- noninformative.
DEFAULT_ALPHA_PRIOR: Final[float] = 0.5
DEFAULT_BETA_PRIOR: Final[float] = 0.5

# propagate_valence defaults.
DEFAULT_VALENCE_DECAY: Final[float] = 0.5
DEFAULT_VALENCE_MAX_HOPS: Final[int] = 3
DEFAULT_VALENCE_MIN_THRESHOLD: Final[float] = 0.05


def db_path() -> Path:
    """Resolve the DB path from env or defaults. Caller creates the dir."""
    override: str | None = os.environ.get("AELFRICE_DB")
    if override:
        return Path(override)
    return DEFAULT_DB_DIR / DEFAULT_DB_FILENAME
