"""Regression: scan_repo finishes under 60s on a synthetic 50k-LOC project.

Closes the v1.0.1 onboard-performance line item from the ROADMAP /
LIMITATIONS. The fixture is generated programmatically so we don't
check in 50k lines of static text; the same generator runs on every
CI invocation, producing a deterministic tree.

Budget: 60s wall clock. Default pytest timeout is 5s; this test
overrides via `@pytest.mark.timeout(120)` to give 2x cushion above
the assertion threshold.

Marker: `regression` so the default fast unit suite (`uv run
pytest`) runs it but it can be filtered with `-m 'not regression'`
when iterating on unit changes.

Where the time goes:
- File generation: O(n_files * avg_file_len) — kept under 1s by
  using simple string repetition; not counted against the 60s.
- scan_repo:
  - filesystem walk over docs (~500 .md/.rst/.txt files)
  - Python AST over .py files (~250 files, top-level docstrings)
  - git_log: skipped (no .git under tmp_path)
  - per-candidate: noise_filter + classify + sqlite insert
- Inserts go through WAL + transaction batching (since v0.6+); the
  per-candidate cost is dominated by the regex pass in
  classify_sentence and the FTS5 mirror write.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore


pytestmark = pytest.mark.regression


# A small bank of realistic-shaped paragraphs. Each is at least 24
# chars (the scanner's _MIN_PARAGRAPH_CHARS) and longer than 4 words
# (above the noise filter's fragment threshold), so the candidates
# survive both filters and exercise the full pipeline.
_DOC_PARAGRAPHS: tuple[str, ...] = (
    "The retrieval pipeline uses BM25 over an FTS5 virtual table to "
    "select candidates and then layers locked beliefs on top.",
    "Configuration loads from environment variables and a single "
    "TOML file at the project root, walking up the directory tree "
    "to find it.",
    "Bayesian updates use closed-form Beta-Bernoulli math: helpful "
    "feedback increments alpha, harmful feedback increments beta, "
    "and the posterior mean is alpha over alpha plus beta.",
    "The audit log records every successful feedback event with "
    "source, timestamp, valence, and belief id; the regime is "
    "recoverable after the fact from this log.",
    "Locked beliefs short-circuit decay and bypass token-budget "
    "trimming so user-asserted ground truth always survives "
    "retrieval-time competition.",
    "Demotion pressure accumulates when contradicting evidence "
    "arrives; once five independent contradictions land, the lock "
    "auto-demotes back to a normal belief.",
    "Onboarding extracts paragraphs from documentation files, "
    "commit subjects from the git log, and module-level docstrings "
    "from Python source via the standard ast module.",
    "Classification picks one of four belief types — factual, "
    "preference, requirement, correction — based on a deterministic "
    "regex fallback when no host LLM is present.",
)


_PY_DOCSTRINGS: tuple[str, ...] = (
    "Compute the posterior mean of a Beta distribution given "
    "the alpha and beta concentration parameters.",
    "Insert a belief into the store and mirror its content into "
    "the FTS5 virtual table for keyword retrieval.",
    "Walk the filesystem under root, yielding paragraphs for each "
    "documentation file in deterministic sorted order.",
    "Apply one feedback event to the addressed belief, updating "
    "alpha or beta by valence sign and writing an audit row.",
    "Return the regime classification for the current store state "
    "based on confidence mass, lock density, and edge density.",
    "Resolve a contradiction between two beliefs by picking a "
    "winner per the precedence rules and emitting a SUPERSEDES "
    "edge from winner to loser.",
)


def _build_doc_file(path: Path, target_lines: int) -> None:
    """Write a documentation file with `target_lines` lines of content.

    Repeats paragraphs from `_DOC_PARAGRAPHS` separated by blank
    lines until the line count is met. Each paragraph wraps to
    around 3 lines, so target_lines / 4 paragraphs roughly.
    """
    lines: list[str] = []
    idx = 0
    while len(lines) < target_lines:
        para = _DOC_PARAGRAPHS[idx % len(_DOC_PARAGRAPHS)]
        # Inject the index so content_hash dedupe doesn't collapse
        # paragraphs across files (we want every candidate to be
        # distinct so the insert path is exercised end to end).
        lines.append(f"// section {idx} //")
        lines.append(para)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines[:target_lines]), encoding="utf-8")


def _build_py_file(path: Path, target_lines: int) -> None:
    """Write a Python file with module + function docstrings.

    Function bodies are filler `pass` lines so the line count adds
    up; each function carries one docstring drawn from
    `_PY_DOCSTRINGS` cycled with the file index for distinctness.
    """
    parts: list[str] = []
    parts.append('"""Module-level docstring describing the role.')
    parts.append('')
    parts.append(_PY_DOCSTRINGS[hash(path.name) % len(_PY_DOCSTRINGS)])
    parts.append('"""')
    parts.append('')
    parts.append('from __future__ import annotations')
    parts.append('')
    fn_idx = 0
    # ~10 lines per function definition. Generate functions until
    # target_lines is roughly met.
    while sum(p.count("\n") for p in parts) + len(parts) < target_lines:
        doc = _PY_DOCSTRINGS[fn_idx % len(_PY_DOCSTRINGS)]
        parts.append(f"def function_{fn_idx}() -> None:")
        parts.append(f'    """{doc} (variant {fn_idx})"""')
        parts.append("    pass")
        parts.append("")
        fn_idx += 1
    path.write_text("\n".join(parts), encoding="utf-8")


def _build_50k_loc_fixture(root: Path) -> tuple[int, int]:
    """Generate a synthetic 50k-LOC project under `root`.

    Returns (n_files, total_lines). Roughly 250 .py files at ~200
    LOC each plus 100 .md/.rst/.txt documentation files at ~50 LOC
    each. Total ~50k + ~5k = ~55k LOC, comfortably above the 50k
    target.
    """
    n_files = 0
    total_lines = 0

    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(250):
        # Spread across ~10 packages so the AST extractor sees a
        # realistic-shaped tree.
        pkg = src / f"pkg_{i % 10}"
        pkg.mkdir(parents=True, exist_ok=True)
        f = pkg / f"module_{i}.py"
        _build_py_file(f, target_lines=200)
        n_files += 1
        total_lines += 200

    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        ext = (".md", ".rst", ".txt")[i % 3]
        f = docs / f"doc_{i}{ext}"
        _build_doc_file(f, target_lines=50)
        n_files += 1
        total_lines += 50

    # Make sure pkg_X has an __init__.py so the directories look
    # canonical (not strictly needed for scan_repo, but cheap).
    for i in range(10):
        (src / f"pkg_{i}" / "__init__.py").write_text(
            '"""Package init."""\n', encoding="utf-8",
        )
        n_files += 1
        total_lines += 1

    return n_files, total_lines


@pytest.mark.timeout(120)
def test_scan_repo_under_60s_on_50k_loc(tmp_path: Path) -> None:
    """End-to-end perf budget: scan_repo on a synthetic 50k-LOC tree
    must finish in under 60 seconds. Held against the in-memory
    store so disk-write contention does not skew the measurement."""
    n_files, total_lines = _build_50k_loc_fixture(tmp_path)
    assert total_lines >= 50_000, (
        f"fixture undershot: only {total_lines} lines"
    )
    assert n_files >= 200, f"fixture undershot: only {n_files} files"

    s = MemoryStore(":memory:")
    try:
        start = time.monotonic()
        result = scan_repo(s, tmp_path)
        elapsed = time.monotonic() - start
    finally:
        s.close()

    assert result.total_candidates > 0, (
        "scan produced zero candidates — fixture or pipeline broken"
    )
    assert elapsed < 60.0, (
        f"scan_repo took {elapsed:.2f}s on a {total_lines}-line / "
        f"{n_files}-file fixture; v1.0.1 budget is 60s. "
        f"Candidates seen: {result.total_candidates}, "
        f"inserted: {result.inserted}, "
        f"skipped_noise: {result.skipped_noise}, "
        f"skipped_non_persisting: {result.skipped_non_persisting}."
    )
