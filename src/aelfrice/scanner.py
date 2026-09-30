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
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from aelfrice.derivation_worker import run_worker
from aelfrice.extraction import CODE_FENCE_RE
from aelfrice.inedible import is_inedible
from aelfrice.models import (
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    INGEST_SOURCE_FILESYSTEM,
)
from aelfrice.noise_filter import NoiseConfig, is_noise
from aelfrice.store import MemoryStore

# Optional v1.3+ LLM-classify routing path. `LLMRouter` is a Protocol
# the scanner depends on; production callers pass an instance built in
# `aelfrice.llm_classifier`. The actual `anthropic` SDK is NOT imported
# here — the SDK call lives behind the Protocol. Default install never
# builds a router so this surface is never reached without explicit
# opt-in. v1.0/v1.2 callers keep `llm_router=None` and the regex path
# is unchanged.
from typing import Protocol

# Directories the scanner never descends into. Standard "build artefact"
# locations, version-control internals, and coding-agent host config
# (.claude holds settings plus full repo copies under worktrees/ —
# scanning it inflates candidates ~27x and pollutes provenance, #954).
_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".claude",
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

# Maximum paragraph length (in chars); a longer paragraph is dropped
# (#1616). A data file with no blank lines is one "paragraph", and one
# such file produced a 54 MB candidate that went whole into a classifier
# prompt. 4,000 sits just above the measured p99 of real doc paragraphs
# (3,541), so it drops about 1% of candidates and keeps a candidate
# near 1k tokens.
_MAX_PARAGRAPH_CHARS: Final[int] = 4_000


@dataclass
class SentenceCandidate:
    """A unit of text extracted from a project file, awaiting classification.

    Fields:
    - text: paragraph-shaped string, stripped of surrounding whitespace
    - source: stable identifier for the origin (e.g. `doc:README.md`,
      `git:commit:abcdef`, `ast:src/foo.py:bar`); read by the
      classifier and stored on the Belief for provenance
    - commit_date: ISO-8601 author date of the most recent commit that
      touched the source file (or, for `git:commit:*` candidates, the
      commit's own date). `None` for files outside any git work-tree
      and for files with no commit history (newly added, untracked).
      v1.1.0 git-recency feature: when set, scan_repo uses this as
      `belief.created_at`. The posterior decay this was written to feed
      was never wired and was deleted in #1369, so the date now serves
      the temporal lanes and the age reports that read `created_at`.
    """

    text: str
    source: str
    commit_date: str | None = None


class LLMRouter(Protocol):
    """Minimal interface scanner needs from an LLM-classify driver.

    Production callers pass `aelfrice.llm_classifier.ScannerRouter`
    (defined there, with the SDK import kept lazy inside its methods); tests pass a
    mock. The protocol decouples scanner.py from the optional
    `anthropic` SDK at module-load.

    Contract: `classify` returns one route per candidate in input
    order. The router is responsible for batching, telemetry, and
    regex fallback on transient failure. Auth failures and token-cap
    aborts are signalled to the caller by raising — scanner does not
    catch them so the cli wrapper can map to exit codes.
    """

    def classify(
        self,
        candidates: "list[SentenceCandidate]",
    ) -> "list[LLMRoute]":
        ...


@dataclass
class LLMRoute:
    """Per-candidate output of the router.

    `belief_type`, `origin`, `persist`, `alpha`, `beta` are the
    fields scanner needs to write a Belief. `audit_source` (when set)
    is written to feedback_history for fallback insertions.
    """

    belief_type: str
    origin: str
    persist: bool
    alpha: float
    beta: float
    audit_source: str | None = None


def _derive_scan_session_id(root: Path, timestamp: str) -> str:
    """Synthetic session_id for filesystem scans.

    Stable across two scans of the same tree at the same timestamp; the
    timestamp differs per invocation so each scan run gets its own id.
    Mirrors `hook_commit_ingest._derive_session_id` in shape (16-hex
    sha256 prefix) so downstream consumers can treat all session_id
    values uniformly.
    """
    raw = f"scan:{root}:{timestamp}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def scan_repo(
    store: MemoryStore,
    root: Path,
    now: str | None = None,
    *,
    noise_config: NoiseConfig | None = None,
    llm_router: LLMRouter | None = None,
    session_id: str | None = None,
) -> ScanResult:
    """Run all three extractors on `root`, classify each candidate, and
    insert persistable results as Beliefs in the store.

    Idempotent: a belief id is `sha256(source\\x00text)[:16]`, so
    re-scanning the same tree produces no duplicates. Existing beliefs
    are detected via `MemoryStore.insert_or_corroborate()`'s content-hash
    lookup (`get_belief_by_content_hash`) and corroborated instead of
    re-inserted.

    Non-persistable candidates (classifier returned `persist=False` —
    empty paragraphs, question-form sentences) are counted in
    `skipped_non_persisting` for visibility but never reach the store.
    Noise candidates (markdown headings, checklist blocks, three-word
    fragments, license boilerplate, plus any extra patterns from
    `.aelfrice.toml`) are counted in `skipped_noise` and never reach
    the classifier.

    `noise_config` controls the noise filter. `None` (default)
    discovers `.aelfrice.toml` by walking up from `root`; pass an
    explicit `NoiseConfig` to override discovery (tests, library use).

    No new edges are formed by `scan_repo` in v1.0. Edge formation at
    onboard time is deferred to a later release; for now beliefs land
    standalone and edges are added by feedback flows or future
    explicit-edge tools.

    Returns a ScanResult summarizing what happened. Pure-stdlib
    orchestration: no third-party deps, deterministic for any stable
    input tree + clock supplied via `now`.
    """
    timestamp = now if now is not None else _utc_now_iso()
    sid = session_id if session_id is not None else _derive_scan_session_id(
        root, timestamp,
    )
    cfg: NoiseConfig = (
        noise_config
        if noise_config is not None
        else NoiseConfig.discover(root)
    )

    # v1.1.0 git-recency: one git invocation produces a map of
    # {relative-path: most-recent-author-date}. Extractors enrich each
    # SentenceCandidate with the matching commit_date so the eventual
    # belief gets `created_at = commit_date` instead of wall-clock now.
    # Files outside git, untracked files, and the entire fallback when
    # git is unavailable all yield commit_date=None and the wall-clock
    # path applies as before.
    recency = _build_file_recency_map(root)

    candidates: list[SentenceCandidate] = []
    candidates.extend(extract_filesystem(root, recency=recency))
    candidates.extend(extract_git_log(root))
    candidates.extend(extract_ast(root, recency=recency))

    inserted = 0
    skipped_existing = 0
    skipped_non_persisting = 0
    skipped_noise = 0

    # Both paths defer belief materialization to
    # `derivation_worker.run_worker` (#264 part 2 for regex; #265 PR-B
    # for the LLM-classify path). Each persistable candidate appends one
    # unstamped row to `ingest_log`; one worker invocation at end-of-scan
    # stamps every row, dedup-or-corroborates against `beliefs`, and
    # writes edges. Snapshot the canonical id set up front so we can
    # back-fill scanner accounting from log entries after the worker
    # runs (preserves the public ScanResult contract).
    #
    # The LLM-vs-regex distinction lives in `raw_meta["route_overrides"]`:
    # when present, the worker post-splices the router's
    # type/origin/alpha/beta onto derive()'s output and emits the
    # audit_source feedback row on insert. When absent, the worker runs
    # the deterministic classifier and writes no audit row.
    ids_before: set[str] = set(store.list_belief_ids())
    log_ids: list[str] = []

    # Two-pass when an llm_router is supplied: first collect the
    # noise-filtered candidates, then route them all through the
    # batched LLM call. Single-pass when no router (default OFF):
    # the regex path delegates to the derivation worker after the loop.
    filtered: list[SentenceCandidate] = []
    for candidate in candidates:
        if is_noise(candidate.text, cfg):
            skipped_noise += 1
            continue
        filtered.append(candidate)

    routes: list[LLMRoute] | None = None
    if llm_router is not None:
        routes = llm_router.classify(filtered)
        if len(routes) != len(filtered):
            # The router contract requires one-route-per-candidate
            # in input order. A length mismatch is a programming
            # error, not a user-visible state.
            raise RuntimeError(
                f"llm_router.classify returned {len(routes)} routes for "
                f"{len(filtered)} candidates"
            )

    for idx, candidate in enumerate(filtered):
        created_at = candidate.commit_date or timestamp
        raw_meta: dict[str, Any] = {
            "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
        }
        if routes is not None:
            route = routes[idx]
            if not route.persist:
                # persist=False is consumed before any state change —
                # no log row, no worker work. Counted for visibility.
                skipped_non_persisting += 1
                continue
            overrides: dict[str, Any] = {
                "belief_type": route.belief_type,
                "origin": route.origin,
                "alpha": route.alpha,
                "beta": route.beta,
            }
            if route.audit_source is not None:
                overrides["audit_source"] = route.audit_source
            raw_meta["route_overrides"] = overrides
        log_id = store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=candidate.source,
            raw_text=candidate.text,
            session_id=sid,
            ts=created_at,
            raw_meta=raw_meta,
        )
        log_ids.append(log_id)

    # End-of-batch worker invocation. Idempotent: scans every unstamped
    # row in the store, including any orphaned by a crashed prior scan.
    # Ours are the rows we just appended; the worker derives, dedups,
    # and stamps each with `derived_belief_ids`. Calling once per scan
    # rather than per candidate keeps the cost O(unstamped) instead of
    # O(unstamped × candidates).
    if log_ids:
        run_worker(store)

        # Back-fill scanner accounting from the stamped log rows. Three
        # outcomes per log row, mirroring the pre-#264 inline branches:
        #   - empty derived_belief_ids → derive() returned no belief
        #     (persist=False) → skipped_non_persisting.
        #   - derived id already in `ids_before` → known belief
        #     corroborated → skipped_existing.
        #   - derived id new since `ids_before` → inserted. Counted at
        #     most once per id so two log rows that converge on the
        #     same belief still report inserted=1.
        seen_new: set[str] = set()
        for log_id in log_ids:
            entry = store.get_ingest_log_entry(log_id)
            if entry is None:
                continue
            ids = entry.get("derived_belief_ids")
            if not isinstance(ids, list) or not ids:
                skipped_non_persisting += 1
                continue
            for bid in ids:
                if not isinstance(bid, str):
                    continue
                if bid in ids_before:
                    skipped_existing += 1
                elif bid not in seen_new:
                    seen_new.add(bid)
                    inserted += 1
                else:
                    # Same belief id already counted in this scan
                    # (two log rows hashed to the same id).
                    skipped_existing += 1

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
                if is_inedible(entry):
                    continue
                stack.append(entry)
                continue
            if entry.suffix.lower() in _DOC_EXTENSIONS:
                if is_inedible(entry):
                    continue
                out.append(entry)
    out.sort()
    return out


def _split_paragraphs(text: str) -> list[str]:
    """Split a document into paragraph-shaped units.

    Paragraph boundary: one or more blank lines. Leading/trailing
    whitespace stripped per paragraph. Paragraphs shorter than
    _MIN_PARAGRAPH_CHARS dropped (markdown headings, list bullets, code
    fence markers, etc.).

    Fenced code regions are removed first (#1371 §10), using the same
    `extraction.CODE_FENCE_RE` the transcript path uses, so one document
    ingested through onboard and through `extract_sentences` yields the
    same beliefs. Before this, `is_noise`'s four categories had no fence
    rule and a fenced block over `_MIN_PARAGRAPH_CHARS` was stored
    verbatim as a belief on the onboard path only.

    Stripping precedes the split, matching `extract_sentences`' ordering:
    a fence sitting between blank lines collapses to whitespace and its
    paragraph drops on length, while a fence inline in a prose paragraph
    leaves the prose intact. An *unterminated* fence matches nothing and
    is left alone — also matching the transcript path, so the two agree
    on the malformed case rather than only the well-formed one.
    """
    raw_paragraphs = [p.strip() for p in CODE_FENCE_RE.sub(" ", text).split("\n\n")]
    return [p for p in raw_paragraphs if len(p) >= _MIN_PARAGRAPH_CHARS]


_GIT_LOG_DEFAULT_LIMIT: Final[int] = 100
_GIT_LOG_TIMEOUT_SECONDS: Final[float] = 5.0

_PY_EXTENSION: Final[str] = ".py"


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_date_to_utc(iso: str) -> str:
    """Rewrite a git `%aI` author date in the stored UTC form (#1611).

    git keeps the author's local offset (`...T10:00:00-08:00`), and
    `created_at` is ordered as text, so a store that mixes offsets with
    the `Z` rows every other writer produces sorts out of real order.
    A value that does not parse as an aware datetime is returned as is:
    this function only changes the form of a date, never which line is
    taken as one.
    """
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if parsed.tzinfo is None:
        return iso
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_file_recency_map(root: Path) -> dict[str, str]:
    """Return `{relative-path: most-recent-author-date-iso}` for every
    file in the git work-tree.

    Walks `git log --name-only --cc -z --pretty=format:%x00%aI` once.
    Newer commits come first; we record the first date seen for each
    file (which is the most recent).

    `--cc` makes a merge commit list the paths it changed relative to
    every parent (#1612). Without it git lists no paths for a merge, so a
    file that arrived only through a merge, such as a history import,
    had no date, and a conflict resolved to new content was dated to the
    last edit before the merge. A merge that took a path unchanged from
    one parent does not list it, so that path keeps the date of the
    commit that made the change, not of the merge. It is the same test
    git's default history simplification applies to a merge in
    `git log -1 -- <path>`. One case still differs from that per-path
    answer: a side-branch edit that a merge discarded is still seen by
    this walk, because only a pathspec lets git prune it.

    The parse reads framing git emits, never the content of a line
    (#1629, #1621). With `-z`, paths are NUL-terminated and printed raw,
    so git does not C-quote non-ASCII names, quotes, or backslashes.
    Split on NUL, a commit is `<date>` or `<date>\\n<first path>`,
    followed by its remaining paths. git never emits an empty path, so
    an empty token only ever comes from the commit separator or the
    leading `%x00`, and the token after one is always a commit header.
    A path whose text looks like a date is therefore read as a path.

    Returns an empty dict when:
    - `root` is not a directory
    - `root` is not a git work-tree
    - `git` binary is missing or rev-parse fails or times out

    Pure stdlib subprocess. One fork per `scan_repo` call regardless of
    file count.
    """
    if not root.exists() or not root.is_dir():
        return {}
    if not (root / ".git").exists():
        return {}
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--name-only",
                "--cc",
                "-z",
                "--pretty=format:%x00%aI",
            ],
            capture_output=True,
            timeout=_GIT_LOG_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}

    out: dict[str, str] = {}
    current_date: str | None = None
    at_header = False
    for token in result.stdout.split(b"\x00"):
        if not token:
            at_header = True
            continue
        if at_header:
            at_header = False
            date, _, first_path = token.partition(b"\n")
            current_date = _git_date_to_utc(date.decode("ascii", errors="replace"))
            if not first_path:
                continue
            token = first_path
        if current_date is None:
            continue
        # Decode as the filesystem does, so the key equals the
        # `relative_to(root).as_posix()` string every lookup computes.
        path = os.fsdecode(token)
        # First-seen wins (newer commits come first). Don't overwrite.
        if path not in out:
            out[path] = current_date
    return out


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
                "--format=%H%x09%aI%x09%s",
                "-n",
                str(limit),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_LOG_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    candidates: list[SentenceCandidate] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        sha, iso_date, subject = parts
        subject = subject.strip()
        if not subject:
            continue
        candidates.append(
            SentenceCandidate(
                text=subject,
                source=f"git:commit:{sha[:7]}",
                commit_date=_git_date_to_utc(iso_date) if iso_date else None,
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
                if is_inedible(entry):
                    continue
                stack.append(entry)
                continue
            if entry.suffix == _PY_EXTENSION:
                if is_inedible(entry):
                    continue
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


def extract_ast(
    root: Path,
    *,
    recency: dict[str, str] | None = None,
) -> list[SentenceCandidate]:
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

    `recency` maps relative-path -> ISO-8601 author date of the most
    recent commit that touched the file. When provided and the file
    has an entry, every candidate from that file is tagged with the
    matching `commit_date` so scan_repo writes it as `belief.created_at`.
    """
    rec = recency or {}
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
        commit_date = rec.get(rel)
        for c in _extract_from_module(tree, rel):
            if commit_date is not None:
                c.commit_date = commit_date
            candidates.append(c)
    return candidates


def extract_filesystem(
    root: Path,
    *,
    recency: dict[str, str] | None = None,
) -> list[SentenceCandidate]:
    """Walk `root` and emit one candidate per doc paragraph.

    Pure-stdlib, deterministic for any stable input tree. Handles
    missing/non-directory roots gracefully (returns empty list).
    Source string format: `doc:<relative_path>:p<paragraph-index>`.

    `recency` maps relative-path -> ISO-8601 author date of the most
    recent commit that touched the file. When provided and the file
    has an entry, every candidate from that file is tagged with the
    matching `commit_date` so scan_repo writes it as `belief.created_at`.
    """
    rec = recency or {}
    candidates: list[SentenceCandidate] = []
    if not root.exists() or not root.is_dir():
        return candidates

    for path in _iter_doc_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        commit_date = rec.get(rel)
        for idx, para in enumerate(_split_paragraphs(text)):
            # Dropped after numbering, not in _split_paragraphs: the index
            # is part of the source and so of the belief id, and filtering
            # first would renumber every later paragraph in the file.
            if len(para) > _MAX_PARAGRAPH_CHARS:
                continue
            candidates.append(
                SentenceCandidate(
                    text=para,
                    source=f"doc:{rel}:p{idx}",
                    commit_date=commit_date,
                )
            )
    return candidates
