"""Project scanner for onboarding.

Three extractors only in v1.0 (pre-commit `MVP_SCOPE.md`):
  - filesystem walk over docs (.md / .rst / .txt / .adoc)
  - basic git log (last N commits)
  - simple AST over .py files (top-level docstrings, class/function names)

Each extractor returns a list of `SentenceCandidate(text, source)`. The
`scan_repo` orchestrator combines them, classifies each candidate via
`classification.classify_sentence`, and inserts the persistable results
as Beliefs into a Store.

The other five extractors from the previous codebase (HRR-related,
doc-cross-reference, citation, sentence-decomposition, structural) are
explicitly deferred to a later release. v1.0 ships only the three
above.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Directories the scanner never descends into. Standard "build artefact"
# locations and version-control internals.
_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".egg-info",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    "vendor",
    ".next",
    ".nuxt",
})

# Documentation file extensions handled by the filesystem extractor.
_DOC_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".md",
    ".rst",
    ".txt",
    ".adoc",
})

# Minimum paragraph length (in chars) to treat as a belief candidate.
# Stops trivial one-liners from polluting the store.
_MIN_PARAGRAPH_CHARS: Final[int] = 24


@dataclass
class SentenceCandidate:
    """A unit of text extracted from a project file, awaiting classification.

    Fields:
    - text: paragraph-shaped string, stripped of surrounding whitespace
    - source: stable identifier for the origin (e.g. `doc:README.md`,
      `git:commit:abcdef`, `ast:src/foo.py:bar`); read by the
      classifier and stored on the Belief for provenance
    """

    text: str
    source: str


def _iter_doc_files(root: Path) -> list[Path]:
    """Return the doc files under root in deterministic sorted order.

    Skips _SKIP_DIRS at every level. Symlinks are not followed (avoids
    infinite loops and lets users opt out of scanning shared volumes by
    symlinking them in).
    """
    out: list[Path] = []
    if not root.exists() or not root.is_dir():
        return out

    # Manual recursion so we can prune SKIP_DIRS deterministically.
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                stack.append(entry)
                continue
            if entry.suffix.lower() in _DOC_EXTENSIONS:
                out.append(entry)
    out.sort()
    return out


def _split_paragraphs(text: str) -> list[str]:
    """Split a document into paragraph-shaped units.

    Paragraph boundary: one or more blank lines. Leading/trailing
    whitespace stripped per paragraph. Paragraphs shorter than
    _MIN_PARAGRAPH_CHARS dropped (markdown headings, list bullets, code
    fence markers, etc.).
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in raw_paragraphs if len(p) >= _MIN_PARAGRAPH_CHARS]


_GIT_LOG_DEFAULT_LIMIT: Final[int] = 100
_GIT_LOG_TIMEOUT_SECONDS: Final[float] = 5.0


def extract_git_log(
    root: Path,
    limit: int = _GIT_LOG_DEFAULT_LIMIT,
) -> list[SentenceCandidate]:
    """Read recent commit subjects via `git log` and emit one candidate
    per commit.

    Returns the empty list when:
    - `root` is not a directory
    - `root` has no `.git` (not a git repo)
    - `git` binary is missing
    - the subprocess errors or times out (5s budget)
    - the repo has no commits

    Source format: `git:commit:<short-hash>`.

    Output order matches `git log` default (most recent first), capped
    at `limit`. Pure stdlib (subprocess); no third-party git library.
    """
    if not root.exists() or not root.is_dir():
        return []
    if not (root / ".git").exists():
        return []
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--format=%H%x09%s",
                "-n",
                str(limit),
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_LOG_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    candidates: list[SentenceCandidate] = []
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        sha, _, subject = line.partition("\t")
        subject = subject.strip()
        if not subject:
            continue
        candidates.append(
            SentenceCandidate(
                text=subject,
                source=f"git:commit:{sha[:7]}",
            )
        )
    return candidates


def extract_filesystem(root: Path) -> list[SentenceCandidate]:
    """Walk `root` and emit one candidate per doc paragraph.

    Pure-stdlib, deterministic for any stable input tree. Handles
    missing/non-directory roots gracefully (returns empty list).
    Source string format: `doc:<relative_path>:p<paragraph-index>`.
    """
    candidates: list[SentenceCandidate] = []
    if not root.exists() or not root.is_dir():
        return candidates

    for path in _iter_doc_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        for idx, para in enumerate(_split_paragraphs(text)):
            candidates.append(
                SentenceCandidate(
                    text=para,
                    source=f"doc:{rel}:p{idx}",
                )
            )
    return candidates
