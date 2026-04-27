"""Project scanner for onboarding.

Three extractors only in v1.0 (pre-commit `MVP_SCOPE.md`):
  - filesystem walk over docs (.md / .rst / .txt / .adoc)
  - basic git log (last N commits)
  - simple AST over .py files (top-level docstrings, class/function names)

Each extractor returns a list of `SentenceCandidate(text, source)`. The
`scan_repo` orchestrator combines them, classifies each candidate via
`classification.classify_sentence`, and inserts the persistable results
as Beliefs into a MemoryStore.

The other five extractors from the previous codebase (HRR-related,
doc-cross-reference, citation, sentence-decomposition, structural) are
explicitly deferred to a later release. v1.0 ships only the three
above.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aelfrice.classification import classify_sentence
from aelfrice.models import LOCK_NONE, Belief
from aelfrice.noise_filter import is_noise
from aelfrice.store import MemoryStore

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


def scan_repo(
    store: MemoryStore,
    root: Path,
    now: str | None = None,
) -> ScanResult:
    """Run all three extractors on `root`, classify each candidate, and
    insert persistable results as Beliefs in the store.

    Idempotent: a belief id is `sha256(source\\x00text)[:16]`, so
    re-scanning the same tree produces no duplicates. Existing beliefs
    are detected via `MemoryStore.get_belief(id)` and skipped.

    Non-persistable candidates (classifier returned `persist=False` —
    empty paragraphs, question-form sentences) are counted in
    `skipped_non_persisting` for visibility but never reach the store.

    No new edges are formed by `scan_repo` in v1.0. Edge formation at
    onboard time is deferred to a later release; for now beliefs land
    standalone and edges are added by feedback flows or future
    explicit-edge tools.

    Returns a ScanResult summarizing what happened. Pure-stdlib
    orchestration: no third-party deps, deterministic for any stable
    input tree + clock supplied via `now`.
    """
    timestamp = now if now is not None else _utc_now_iso()
    candidates: list[SentenceCandidate] = []
    candidates.extend(extract_filesystem(root))
    candidates.extend(extract_git_log(root))
    candidates.extend(extract_ast(root))

    inserted = 0
    skipped_existing = 0
    skipped_non_persisting = 0
    skipped_noise = 0

    for candidate in candidates:
        if is_noise(candidate.text):
            skipped_noise += 1
            continue
        result = classify_sentence(candidate.text, candidate.source)
        if not result.persist:
            skipped_non_persisting += 1
            continue
        belief_id = _derive_belief_id(candidate.text, candidate.source)
        if store.get_belief(belief_id) is not None:
            skipped_existing += 1
            continue
        store.insert_belief(Belief(
            id=belief_id,
            content=candidate.text,
            content_hash=_content_hash(candidate.text),
            alpha=result.alpha,
            beta=result.beta,
            type=result.belief_type,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=timestamp,
            last_retrieved_at=None,
        ))
        inserted += 1

    return ScanResult(
        inserted=inserted,
        skipped_existing=skipped_existing,
        skipped_non_persisting=skipped_non_persisting,
        total_candidates=len(candidates),
        skipped_noise=skipped_noise,
    )


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

_PY_EXTENSION: Final[str] = ".py"

_BELIEF_ID_HEX_LEN: Final[int] = 16


@dataclass
class ScanResult:
    """Aggregate outcome of a scan_repo run.

    Fields:
    - inserted: count of new beliefs added to the store
    - skipped_existing: count of candidates whose deterministic id was
      already present (re-scan idempotence)
    - skipped_non_persisting: count of candidates the classifier
      flagged persist=False (questions, empty paragraphs)
    - total_candidates: sum across all extractors before filtering
    - skipped_noise: count of candidates dropped by the v1.0.1
      noise_filter before classification (markdown headings,
      checklist blocks, three-word fragments, license boilerplate)
    """

    inserted: int
    skipped_existing: int
    skipped_non_persisting: int
    total_candidates: int
    skipped_noise: int = 0


def _derive_belief_id(text: str, source: str) -> str:
    """Deterministic id from (text, source). Re-running scan on the same
    tree yields the same ids — that's how the orchestrator stays
    idempotent without a separate dedupe table.
    """
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _iter_py_files(root: Path) -> list[Path]:
    """Return .py files under root in deterministic sorted order.

    Honors the same _SKIP_DIRS exclusions as the doc walk.
    """
    out: list[Path] = []
    if not root.exists() or not root.is_dir():
        return out
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
            if entry.suffix == _PY_EXTENSION:
                out.append(entry)
    out.sort()
    return out


def _extract_from_module(
    tree: ast.Module, rel: str
) -> list[SentenceCandidate]:
    """One pass over the module: collect module docstring + top-level
    function and class docstrings.

    No-docstring symbols are skipped (not enough signal). Nested
    functions and methods are skipped (top-level only) — per the
    v1.0 'simple AST' contract.
    """
    out: list[SentenceCandidate] = []

    module_doc = ast.get_docstring(tree)
    if module_doc:
        out.append(
            SentenceCandidate(
                text=module_doc.strip(),
                source=f"ast:{rel}:module",
            )
        )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            if doc:
                out.append(
                    SentenceCandidate(
                        text=doc.strip(),
                        source=f"ast:{rel}:func:{node.name}",
                    )
                )
        elif isinstance(node, ast.AsyncFunctionDef):
            doc = ast.get_docstring(node)
            if doc:
                out.append(
                    SentenceCandidate(
                        text=doc.strip(),
                        source=f"ast:{rel}:func:{node.name}",
                    )
                )
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                out.append(
                    SentenceCandidate(
                        text=doc.strip(),
                        source=f"ast:{rel}:class:{node.name}",
                    )
                )
    return out


def extract_ast(root: Path) -> list[SentenceCandidate]:
    """Walk .py files under root and extract docstrings as candidates.

    Three sources only — module docstrings, top-level function
    docstrings, top-level class docstrings — per the v1.0 'simple AST'
    contract. Nested functions and methods are skipped intentionally;
    the next release can expand if usage justifies it.

    Files that fail to parse are skipped silently (returns the
    candidates from the rest of the tree). Pure stdlib (`ast` module),
    deterministic for any stable input tree.

    Source formats:
      `ast:<rel-path>:module`
      `ast:<rel-path>:func:<name>`
      `ast:<rel-path>:class:<name>`
    """
    candidates: list[SentenceCandidate] = []
    for path in _iter_py_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(root).as_posix()
        candidates.extend(_extract_from_module(tree, rel))
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
