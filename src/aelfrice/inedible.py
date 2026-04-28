"""User-controlled exclusion: skip files whose basename contains INEDIBLE.

A file or filename containing the literal string `INEDIBLE` (all caps,
anywhere in the basename) is unconditionally skipped by every aelfrice
ingest path:

- `scan_repo` filesystem and AST walks.
- `ingest_jsonl` (single file and `--batch DIR` recursion).

This is a privacy/security primitive. The user opts a file out of the
brain graph by renaming it; aelfrice respects the choice
deterministically. The check is on the file's basename, not the
content — content is not read at all when `is_inedible` returns True.

Marker policy:
- Case-sensitive. `INEDIBLE` matches; `Inedible` and `inedible` do not.
  Reason: the marker should be unmistakable in directory listings; a
  case-insensitive check would surprise users who lowercase
  reflexively.
- Anywhere in the basename. `INEDIBLE_secrets.md`,
  `notes_INEDIBLE.txt`, and `partINEDIBLEpart.py` all match.
- Basename only. A directory named `INEDIBLE/` does not propagate to
  files underneath; users who want directory-scoped exclusion should
  rename the directory itself (which then matches via its basename
  during the walk) or use `.gitignore`-style patterns (deferred to a
  future feature).

The marker string and predicate are deliberately tiny and side-effect
free so they're cheap to call on every file the walkers visit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

INEDIBLE_MARKER: Final[str] = "INEDIBLE"


def is_inedible(path: Path | str) -> bool:
    """Return True iff `path`'s basename contains the INEDIBLE marker.

    Accepts either a Path or a raw string; the basename is taken via
    Path() so callers don't have to normalise.
    """
    return INEDIBLE_MARKER in Path(path).name
