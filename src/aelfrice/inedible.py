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
- `is_inedible` inspects the basename only. The walkers that need
  directory-scoped exclusion (the `ingest_jsonl` transcript paths,
  which glob a tree rather than recursing entry-by-entry) call
  `is_inedible_path`, which also inspects ancestor directory names so
  a file beneath an `INEDIBLE/` directory is excluded — matching
  `scan_repo`, which prunes such directories during its walk.

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


def is_inedible_path(path: Path | str, *, root: Path | str | None = None) -> bool:
    """Return True iff `path`'s basename OR any ancestor directory name
    carries the INEDIBLE marker.

    `is_inedible` checks the basename only, which is correct for
    `scan_repo` (it prunes INEDIBLE-named directories entry-by-entry, so
    a file beneath one is never reached). The `ingest_jsonl` transcript
    paths glob a whole tree instead of recursing, so they would descend
    into an INEDIBLE-named directory unless the ancestor names are
    inspected too. This helper closes that gap (issue #958).

    `root`, when given, bounds the ancestor check to the directory
    components strictly between `root` and the file — the root's own
    name is not inspected, matching `scan_repo`, which pushes its walk
    root unconditionally. Use it for the batch path. With `root=None`
    (the single-file path, which has no walk root) every ancestor up to
    the filesystem root is inspected.
    """
    p = Path(path)
    if INEDIBLE_MARKER in p.name:
        return True
    if root is not None:
        try:
            ancestor_names = p.relative_to(Path(root)).parts[:-1]
        except ValueError:
            # `path` is not under `root`; fall back to the full walk.
            ancestor_names = tuple(parent.name for parent in p.parents)
    else:
        ancestor_names = tuple(parent.name for parent in p.parents)
    return any(INEDIBLE_MARKER in name for name in ancestor_names)
