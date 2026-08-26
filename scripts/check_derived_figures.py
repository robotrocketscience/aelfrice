#!/usr/bin/env python3
"""#1469 — a published figure must name the script that re-derives it.

Six of eight PRs reviewed in the 2026-08-10 board sweep shipped at least one
stale or false published figure. Every one was caught by a human re-deriving
it; none by CI. This is the gate that changes that.

## The marker

A published figure carries a machine-readable marker naming its producer, the
key that producer emits it under, and the value that was published:

    <!-- derived: benchmarks/published_constants.py#stop_prompt_max_items = 20 -->

In a Python file the identical marker is written inside a comment:

    # <!-- derived: benchmarks/published_constants.py#stop_prompt_max_items = 20 -->

The syntax is the same everywhere so one scanner reads both surfaces. Six of
the seven #1469 instances shipped in source as well as in the CHANGELOG, and
#1445's number shipped in three files, so a CHANGELOG-only scanner would have
guarded a third of the corpus.

## Two classes, and the reader can tell them apart

**Store-free** — no `corpus=` attribute. CI re-runs the producer and hard-fails
on any difference between the emitted value and the published one. This is the
real gate.

**Store-backed** — carries `corpus=<label>@<YYYY-MM-DD>` and
`producer-sha=<12 hex>`. `benchmarks/stop_prompt_block_bounds.py`,
`benchmarks/scan_admission_funnel.py` and `benchmarks/sidecar_rebuild_rate.py`
read a real belief store; a public runner has none and the lab corpus must not
go there (#1456). Re-running them in CI is impossible, so the marker instead
records *which corpus, on what date* produced the figure, and CI checks the two
things it still can see:

  * **self-consistency** (hard) — the same `producer#key` published in several
    files must carry the same value. #1449 shipped 44,683 in one file and
    44,687 in four others, in one PR; that is exactly this check.
  * **code staleness** (advisory) — has the producer's source changed since the
    figure was stamped? #1445 is the clean example: correct when measured, and
    one commit later the code moved underneath it. Advisory because a producer
    edit does not prove the figure moved, and because the re-measure needs a
    store nobody in public CI has.

The class is readable off the marker itself: `corpus=` present means "this one
cannot be re-run here, and here is what it was measured against".

## Grandfathering, and the one claim that is not grandfathered

A figure with no marker is allowed. The annotation backlog is large and a gate
that fails on it would be turned off within a day. `--list-unmarked` enumerates
what is still unannotated so the backlog is countable rather than notional.

The exception is the overclaim sentence, whose shape is
``<script> re-derives **every** figure here``. #1445 and #1447 both shipped it
over scripts that emitted about a
third of the entry's numbers. That sentence is only permitted in an entry where
every figure carries a marker, and that check is **hard**. Making a strong claim
is allowed; making it for free is not.

## Usage

    python3 scripts/check_derived_figures.py                # text checks only
    uv run python scripts/check_derived_figures.py --mode all
    python3 scripts/check_derived_figures.py --list-unmarked CHANGELOG/v4.md

`--mode text` is stdlib-only and needs no installed package, so it runs in the
`release-docs-check` job beside the other every-PR document gates. `--mode
producers` executes the store-free producers and therefore needs the package;
it runs in its own `ci.yml` job. `--mode all` is the local form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories scanned for markers and for overclaim sentences.
DEFAULT_ROOTS: tuple[str, ...] = (
    "CHANGELOG",
    "src",
    "benchmarks",
    "docs",
    "scripts",
    "tests",
    "README.md",
)

SCANNED_SUFFIXES: frozenset[str] = frozenset({".md", ".py"})

# Directories never scanned: generated output, fixtures and vendored trees.
SKIP_PARTS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "results", "fixtures", "oracle_fixtures", "corpus",
})

# The marker. `producer` is a repo-relative path, `key` the name the producer
# emits the figure under, `value` the published number as written.
MARKER_RE = re.compile(
    r"<!--\s*derived:\s*"
    r"(?P<producer>[^\s#]+)#(?P<key>[A-Za-z0-9_.\-]+)"
    r"\s*=\s*(?P<value>[^\s]+)"
    r"(?P<attrs>(?:\s+[a-z][a-z-]*=[^\s]+)*)"
    r"\s*-->"
)

ATTR_RE = re.compile(r"([a-z][a-z-]*)=([^\s]+)")

# `corpus=` is the class discriminator; `producer-sha=` is the staleness stamp.
CORPUS_ATTR = "corpus"
SHA_ATTR = "producer-sha"
KNOWN_ATTRS: frozenset[str] = frozenset({CORPUS_ATTR, SHA_ATTR})

# A corpus identity is a label and the date it was measured. Both halves are
# load-bearing: the label says which store, the date says which snapshot of it.
# #1449's 44,683-vs-44,687 split is two real snapshots five days apart, not an
# arithmetic error, and only the date says so.
CORPUS_RE = re.compile(r"^[A-Za-z0-9_.:/+-]+@\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{12}$")

# The overclaim. Deliberately narrow — it fires on the sentence shape that
# actually shipped, not on every mention of re-derivation.
# `[*_]{0,2}` around each word is not decoration: #1445 shipped the sentence as
# ``re-derives **every** figure here``, and a pattern that could not see through
# the emphasis would have missed the exact instance this rule exists for.
_EMPH = r"[*_]{0,2}"
OVERCLAIM_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"re-?derives?\s+{_EMPH}(?:every|all|each){_EMPH}\s+"
        rf"{_EMPH}(?:figure|number|value)",
        re.I,
    ),
    re.compile(
        rf"{_EMPH}(?:every|all|each){_EMPH}\s+{_EMPH}(?:figure|number|value)s?{_EMPH}\s+"
        r"(?:here|in this entry|in this section)[^.]{0,60}re-?derived",
        re.I,
    ),
)

# Inline code is a citation, not a claim. The rule has to be discussable in
# prose -- this file's own docstring quotes the sentence, and so does the
# CHANGELOG entry that introduces the rule -- and a checker that fires on every
# mention of itself is a checker with an exemption list, which is worse. Quoting
# the sentence inside backticks is the escape, and it is the same convention the
# repo already uses for naming code in prose.
# Double-backtick spans first: a citation of the sentence has to be able to
# contain a backtick, because the sentence itself names a script in code font.
_INLINE_CODE_RE = re.compile(r"``.*?``|`[^`]*`", re.S)


def _uncited(text: str) -> str:
    """`text` with inline-code spans blanked, for claim detection."""
    return _INLINE_CODE_RE.sub(" ", text)


# Figure extraction. Applied only inside an entry that carries an overclaim
# sentence, or under --list-unmarked.
#
# Masked out before extraction, because none of these is a measured figure:
# inline code, markdown links, issue refs, dotted version strings, ISO dates,
# section refs, and the markers themselves.
_MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<!--.*?-->", re.S),
    re.compile(r"`[^`]*`"),
    re.compile(r"\[[^\]]*\]\([^)]*\)"),
    re.compile(r"https?://\S+"),
    re.compile(r"#\d+"),
    re.compile(r"\bv?\d+(?:\.\d+){2,}"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"§\s*\d+(?:\.\d+)*"),
)

# Thousands groups are matched as `,ddd` / `_ddd` rather than as a loose
# `[\d,_]*` class, which swallowed the sentence comma after a figure and
# reported `41,929,` as the unmarked value -- a diagnostic the author cannot
# grep for is a diagnostic they ignore.
FIGURE_RE = re.compile(r"(?<![\w.])\d+(?:[,_]\d{3})*(?:\.\d+)?\s*(?:%|x|×)?")


class Marker:
    """One `<!-- derived: ... -->` occurrence."""

    __slots__ = ("path", "line", "producer", "key", "value", "corpus", "sha", "errors")

    def __init__(
        self,
        path: Path,
        line: int,
        producer: str,
        key: str,
        value: str,
        corpus: str | None,
        sha: str | None,
        errors: list[str],
    ) -> None:
        self.path = path
        self.line = line
        self.producer = producer
        self.key = key
        self.value = value
        self.corpus = corpus
        self.sha = sha
        self.errors = errors

    @property
    def ident(self) -> str:
        return f"{self.producer}#{self.key}"

    @property
    def store_backed(self) -> bool:
        return self.corpus is not None


def normalise(value: str) -> str:
    """Canonical form for comparing a published rendering to an emitted value.

    `11,508`, `11508` and `11_508` are one figure; so are `299.7x` and `299.7`,
    and `8.69%` and `8.69`. Trailing zeros are dropped so `20` and `20.0` agree
    -- a producer emitting an int and prose writing a float is not a defect.
    """
    text = str(value).strip().rstrip("%xX×").replace(",", "").replace("_", "")
    try:
        num = float(text)
    except ValueError:
        return text.casefold()
    if num == int(num):
        return str(int(num))
    return repr(num)


def iter_files(roots: list[str]) -> list[Path]:
    """Every scanned file under `roots`, sorted, deterministic."""
    out: list[Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if base.is_file():
            if base.suffix in SCANNED_SUFFIXES:
                out.append(base)
            continue
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
                continue
            if SKIP_PARTS & set(path.relative_to(REPO_ROOT).parts):
                continue
            out.append(path)
    return sorted(set(out))


def parse_markers(path: Path, text: str) -> list[Marker]:
    """Every marker in `text`, with grammar errors attached rather than raised.

    A malformed marker is reported, never skipped. A marker the scanner cannot
    read is a figure nobody is guarding while the prose says otherwise, which is
    worse than no marker at all.
    """
    markers: list[Marker] = []
    for match in MARKER_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        attrs: dict[str, str] = {}
        errors: list[str] = []
        for name, val in ATTR_RE.findall(match.group("attrs") or ""):
            if name in attrs:
                errors.append(f"duplicate attribute {name!r}")
            attrs[name] = val
        for name in attrs:
            if name not in KNOWN_ATTRS:
                errors.append(
                    f"unknown attribute {name!r}; known: {sorted(KNOWN_ATTRS)}"
                )
        corpus = attrs.get(CORPUS_ATTR)
        sha = attrs.get(SHA_ATTR)
        if corpus is not None:
            if not CORPUS_RE.match(corpus):
                errors.append(
                    f"corpus={corpus!r} is not '<label>@<YYYY-MM-DD>'; the date "
                    "is what distinguishes two snapshots of one store"
                )
            if sha is None:
                errors.append(
                    "store-backed marker (corpus=) must also carry "
                    "producer-sha=<12 hex> or staleness cannot be checked"
                )
        elif sha is not None:
            errors.append(
                "producer-sha= without corpus=: a store-free figure is checked "
                "by re-running the producer, so the stamp is misleading"
            )
        if sha is not None and not SHA_RE.match(sha):
            errors.append(f"producer-sha={sha!r} is not 12 lowercase hex digits")
        markers.append(
            Marker(
                path=path,
                line=line,
                producer=match.group("producer"),
                key=match.group("key"),
                value=match.group("value"),
                corpus=corpus,
                sha=sha,
                errors=errors,
            )
        )
    return markers


# The stamp is excluded from the bytes it is a stamp over. Without this the
# hash of a producer that documents its own figures is a moving target: writing
# the new stamp into its docstring changes the bytes, which changes the hash,
# which invalidates the stamp just written. Measured, not reasoned about --
# `--restamp` ran twice on `benchmarks/spine_fan_in_baseline.py` and reported a
# different "current" hash each time. Stripping the stamp value makes
# restamping a fixed point in one pass, and costs nothing: the stamp is the one
# span of the file that can never be the reason a figure moved.
_SHA_STRIP_RE = re.compile(rb"producer-sha=[0-9a-f]{12}")


def producer_sha(producer: str) -> str | None:
    """First 12 hex of sha256 over the producer's bytes, or None if missing.

    Every `producer-sha=` value in the file is blanked first; see above.
    """
    path = REPO_ROOT / producer
    if not path.is_file():
        return None
    body = _SHA_STRIP_RE.sub(b"producer-sha=", path.read_bytes())
    return hashlib.sha256(body).hexdigest()[:12]


def split_entries(path: Path, text: str) -> list[tuple[int, str]]:
    """Split a file into the units the overclaim claim is scoped to.

    A CHANGELOG entry is a top-level `- ` bullet plus its indented continuation
    lines -- the unit a reader reads as one claim, and the unit
    `CHANGELOG/unreleased/` stores one per file. Anywhere else the unit is a
    blank-line-delimited paragraph, which is the widest scope a comment block
    can reasonably be held to.

    Returns `(first_line_number, block_text)` pairs.
    """
    lines = text.splitlines()
    entries: list[tuple[int, str]] = []
    if any(line.startswith("- ") for line in lines):
        start: int | None = None
        buf: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if line.startswith("- "):
                if start is not None:
                    entries.append((start, "\n".join(buf)))
                start, buf = idx, [line]
            elif line.startswith("#") or line.startswith("## "):
                if start is not None:
                    entries.append((start, "\n".join(buf)))
                start, buf = None, []
            elif start is not None:
                buf.append(line)
        if start is not None:
            entries.append((start, "\n".join(buf)))
        return entries
    start = 1
    buf = []
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if not buf:
                start = idx
            buf.append(line)
        elif buf:
            entries.append((start, "\n".join(buf)))
            buf = []
    if buf:
        entries.append((start, "\n".join(buf)))
    return entries


def extract_figures(block: str) -> list[str]:
    """Numeric figures in `block`, in order, deduplicated by normalised value."""
    masked = block
    for pattern in _MASKS:
        masked = pattern.sub(" ", masked)
    seen: set[str] = set()
    out: list[str] = []
    for match in FIGURE_RE.finditer(masked):
        raw = match.group(0).strip()
        norm = normalise(raw)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(raw)
    return out


def rel(path: Path) -> str:
    """Repo-relative path, falling back to the absolute one.

    The fallback is not decoration: the tests drive the same functions over a
    tmp_path tree, and a `relative_to` that raised there would mean the live
    scan and the unit scan run different code.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class Report:
    """Accumulates findings; hard ones set the exit code, advisory ones do not."""

    def __init__(self, github: bool) -> None:
        self.github = github
        self.hard: list[str] = []
        self.advisory: list[str] = []

    def fail(self, path: Path, line: int, message: str) -> None:
        self.hard.append(f"{rel(path)}:{line}: {message}")
        if self.github:
            print(f"::error file={rel(path)},line={line}::{message}")

    def warn(self, path: Path, line: int, message: str) -> None:
        self.advisory.append(f"{rel(path)}:{line}: {message}")
        if self.github:
            print(f"::warning file={rel(path)},line={line}::{message}")


def check_text(files: list[Path], report: Report) -> list[Marker]:
    """Grammar, self-consistency, staleness and the overclaim sentence."""
    markers: list[Marker] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "derived:" in text:
            markers.extend(parse_markers(path, text))

    for marker in markers:
        for err in marker.errors:
            report.fail(marker.path, marker.line, f"malformed marker: {err}")
        if not (REPO_ROOT / marker.producer).is_file():
            report.fail(
                marker.path,
                marker.line,
                f"producer {marker.producer!r} does not exist",
            )

    # Self-consistency: one producer#key, one published value, everywhere.
    by_ident: dict[str, list[Marker]] = {}
    for marker in markers:
        by_ident.setdefault(marker.ident, []).append(marker)
    for ident, group in sorted(by_ident.items()):
        values = {normalise(m.value) for m in group}
        if len(values) > 1:
            sites = ", ".join(f"{rel(m.path)}:{m.line}={m.value}" for m in group)
            for marker in group:
                report.fail(
                    marker.path,
                    marker.line,
                    f"{ident} is published with {len(values)} different values "
                    f"({sites}); one figure, one value",
                )

    # Code staleness: advisory, and only meaningful for store-backed figures.
    for marker in markers:
        if not marker.store_backed or marker.sha is None:
            continue
        current = producer_sha(marker.producer)
        if current is None or current == marker.sha:
            continue
        report.warn(
            marker.path,
            marker.line,
            f"{marker.ident} was stamped against {marker.producer} at "
            f"{marker.sha}, which is now {current}. The producer changed since "
            "this figure was measured; re-derive it against "
            f"{marker.corpus} or restamp if the change cannot move it.",
        )

    # The overclaim sentence. Hard.
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not any(p.search(_uncited(text)) for p in OVERCLAIM_RES):
            continue
        for start, block in split_entries(path, text):
            if not any(p.search(_uncited(block)) for p in OVERCLAIM_RES):
                continue
            published = {normalise(m.value) for m in parse_markers(path, block)}
            missing = [f for f in extract_figures(block) if normalise(f) not in published]
            if missing:
                shown = ", ".join(missing[:12])
                more = "" if len(missing) <= 12 else f" (+{len(missing) - 12} more)"
                report.fail(
                    path,
                    start,
                    "this entry claims every figure is re-derived, but "
                    f"{len(missing)} of them carry no marker: {shown}{more}. "
                    "Annotate them, or narrow the sentence to what the script "
                    "actually emits.",
                )
    return markers


def check_producers(markers: list[Marker], report: Report) -> None:
    """Run every store-free producer and diff its output against the prose."""
    by_producer: dict[str, list[Marker]] = {}
    for marker in markers:
        if marker.store_backed or marker.errors:
            continue
        by_producer.setdefault(marker.producer, []).append(marker)

    for producer, group in sorted(by_producer.items()):
        path = REPO_ROOT / producer
        if not path.is_file():
            continue  # already reported by check_text
        proc = subprocess.run(
            [sys.executable, str(path), "--emit-figures"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            for marker in group:
                report.fail(
                    marker.path,
                    marker.line,
                    f"producer {producer} exited {proc.returncode} under "
                    f"--emit-figures: {proc.stderr.strip()[:400]}",
                )
            continue
        try:
            emitted = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            for marker in group:
                report.fail(
                    marker.path,
                    marker.line,
                    f"producer {producer} did not emit JSON on stdout ({exc})",
                )
            continue
        if not isinstance(emitted, dict):
            for marker in group:
                report.fail(
                    marker.path,
                    marker.line,
                    f"producer {producer} emitted {type(emitted).__name__}, "
                    "expected a JSON object of key -> value",
                )
            continue
        for marker in group:
            if marker.key not in emitted:
                report.fail(
                    marker.path,
                    marker.line,
                    f"producer {producer} emits no key {marker.key!r}; it emits "
                    f"{sorted(emitted)}",
                )
                continue
            got = normalise(emitted[marker.key])
            want = normalise(marker.value)
            if got != want:
                report.fail(
                    marker.path,
                    marker.line,
                    f"published {marker.ident} = {marker.value}, but "
                    f"{producer} now emits {emitted[marker.key]}. The figure is "
                    "stale: re-derive it, or fix the producer.",
                )


def restamp(files: list[Path]) -> int:
    """Rewrite every `producer-sha=` to the producer's current hash.

    Two uses, and only two. A producer whose own docstring carries a marker
    cannot be stamped by hand — writing the hash changes the bytes the hash is
    over — so the stamp has to be applied after the edit rather than during it.
    And a producer edit that provably cannot move the figure (a docstring
    correction, a rename) leaves an advisory warning standing on every file
    quoting it, which is how a warning becomes wallpaper.

    Restamping asserts "I have re-derived this figure, or I know this edit
    cannot have moved it". It is never a way to clear a warning that has not
    been looked at, and it is deliberately not run by CI.
    """
    changed = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "derived:" not in text:
            continue
        out = text
        for marker in parse_markers(path, text):
            if marker.sha is None:
                continue
            current = producer_sha(marker.producer)
            if current is None or current == marker.sha:
                continue
            out = out.replace(
                f"{SHA_ATTR}={marker.sha}", f"{SHA_ATTR}={current}"
            )
        if out != text:
            path.write_text(out, encoding="utf-8")
            changed += 1
            print(f"restamped {rel(path)}")
    print(f"{changed} file(s) restamped.")
    return 0


def list_unmarked(files: list[Path]) -> int:
    """Enumerate figures that carry no marker. Reporting only; always exit 0."""
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for start, block in split_entries(path, text):
            published = {normalise(m.value) for m in parse_markers(path, block)}
            missing = [f for f in extract_figures(block) if normalise(f) not in published]
            if not missing:
                continue
            total += len(missing)
            print(f"{rel(path)}:{start}: {len(missing)} unmarked: {', '.join(missing[:20])}")
    print(f"\n{total} unmarked figures across {len(files)} files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("text", "producers", "all"),
        default="text",
        help="text: grammar/self-consistency/staleness/overclaim, stdlib only. "
        "producers: re-run store-free producers (needs the package). "
        "all: both.",
    )
    ap.add_argument("--list-unmarked", action="store_true")
    ap.add_argument(
        "--restamp",
        action="store_true",
        help="rewrite producer-sha= to each producer's current hash. Local "
        "only; asserts the figure was re-derived or cannot have moved.",
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit ::error/::warning workflow annotations as well as text",
    )
    ap.add_argument("paths", nargs="*", default=None)
    args = ap.parse_args(argv)

    roots = args.paths if args.paths else list(DEFAULT_ROOTS)
    files = iter_files(roots)
    if not files:
        print("no files scanned; check the paths given", file=sys.stderr)
        return 1

    if args.restamp:
        return restamp(files)
    if args.list_unmarked:
        return list_unmarked(files)

    report = Report(github=args.github)
    markers = check_text(files, report)
    if args.mode in ("producers", "all"):
        check_producers(markers, report)

    store_free = sum(1 for m in markers if not m.store_backed)
    print(
        f"{len(markers)} derived-figure markers across {len(files)} files "
        f"({store_free} store-free, {len(markers) - store_free} store-backed)."
    )
    for line in report.advisory:
        print(f"advisory: {line}")
    for line in report.hard:
        print(f"ERROR: {line}", file=sys.stderr)
    if report.hard:
        print(f"\n{len(report.hard)} hard failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
