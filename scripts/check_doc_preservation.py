#!/usr/bin/env python3
"""Check that a documentation rewrite changed the prose and nothing else (#1509).

A style conversion touches every sentence in a document. That is a large diff
with no behaviour change, and it is exactly the shape of diff a reviewer skims.
The failure it hides is a lost fact: a benchmark figure that got rounded, a
citation that fell out of a shortened sentence, a table row that went missing
when a section was reflowed. None of those show up as a broken build, and none
of them are visible without reading the two versions side by side.

So the property is checked mechanically instead. The rule for an ASD-STE100
conversion is that prose changes and evidence does not:

    Every number, citation, link, table and figure is preserved unchanged;
    section order and the article title are unchanged.

This script compares a document against its pre-rewrite form and fails on any
loss. Prose differences are never reported, because prose changing is the
point.

## Direction

The check is one-directional. A rewrite may ADD a subheading, because breaking
a long section into named parts is a legitimate conversion move. It may never
drop one. Additions print as `NOTE` and pass; losses print as `LOSS` and fail.

## What is deliberately not checked

Heading *wording*. Removing idiom from a heading is required by the style, not
a defect, so comparing heading text would forbid the conversion this script
exists to verify. What is checked instead is that the title is unchanged, that
no section disappeared, and that no in-document link was orphaned by a
rewording -- the one way a reworded heading does lose something.

Number *attachment*. Numbers compare as a multiset, so a rewrite that swapped
two figures between sentences still passes. That case needs a reader, and the
script says so rather than implying a coverage it does not have.

## Usage

    scripts/check_doc_preservation.py ORIGINAL REWRITTEN

Exit 0 = no losses, 1 = at least one loss, 2 = usage error.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# A fence may be indented: markdown allows up to three spaces anywhere, and a
# fence inside a list item is indented to the item's content column. Anchoring
# hard to the line start leaves those blocks unstripped, and the stray backtick
# run then mis-pairs every inline code span after it -- which reports junk
# tokens as losses and can bury a real one.
FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>```+|~~~+)(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
# Inline code spans are scanned by _code_spans() rather than matched by a
# regex, because the delimiter is a RUN of backticks and only a run of equal
# length closes it. A single-backtick pattern splits ``` into a pair plus a
# stray, and that stray then pairs with the next real span's backtick -- which
# reports junk tokens as losses and can bury a real one. See _code_spans.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.MULTILINE)
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")
HTML_ATTR_URL_RE = re.compile(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']")
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
FOOTNOTE_RE = re.compile(r"\[\^([^\]]+)\]")
ANCHOR_RE = re.compile(r"\]\(#([^)\s]+)\)|href\s*=\s*[\"']#([^\"']+)[\"']")
EXPLICIT_ANCHOR_RE = re.compile(r"(?:name|id)\s*=\s*[\"']([^\"']+)[\"']")
BRACE_ANCHOR_RE = re.compile(r"\{#([^}]+)\}")


def _strip_code(text: str) -> str:
    """Drop fenced blocks so their contents are not counted twice."""
    return FENCE_RE.sub("\n", text)


def _code_spans(text: str) -> list[str]:
    """Inline code spans, honouring backtick runs the way markdown does.

    A run of N backticks opens a span, and only a run of exactly N backticks
    closes it. A run with no partner of its own length is literal text, and
    scanning resumes after it rather than inside it.

    The rule matters for a literal ``` inside a table cell, which is ordinary
    prose and not a fence. Pairing backticks one at a time turns that run into
    a span plus a leftover backtick, and the leftover swallows everything up to
    the next real span -- so `docs/user/CONFIG.md` reported a phantom loss of
    the token 'fences) +' and froze that cell against any rewrite.

    A span may still wrap one line break, because a conversion reflows
    paragraphs and markdown renders the break as a space. More than one break
    means the opener never had a partner in the same paragraph, so it is
    treated as literal and the scan cannot run away across the document.
    """
    spans: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "`":
            i += 1
            continue
        open_start = i
        while i < n and text[i] == "`":
            i += 1
        run = i - open_start
        body_start = i
        j = i
        while j < n:
            if text[j] != "`":
                j += 1
                continue
            close_start = j
            while j < n and text[j] == "`":
                j += 1
            if j - close_start != run:
                continue  # a run of another length is body text
            body = text[body_start:close_start]
            if body.count("\n") <= 1:
                spans.append(body)
                i = j
            break
    return spans


def fences(text: str) -> Counter[str]:
    return Counter(m.group("body") for m in FENCE_RE.finditer(text))


def fence_langs(text: str) -> Counter[str]:
    return Counter(m.group("info").strip() for m in FENCE_RE.finditer(text))


def inline_code(text: str) -> Counter[str]:
    """Inline code spans, with internal whitespace collapsed.

    A conversion reflows paragraphs, so a span that wrapped a line break in the
    original may sit on one line afterwards. That is the same span. Collapsing
    the whitespace keeps a reflow from reading as a lost identifier, while a
    genuinely renamed identifier still compares unequal.
    """
    return Counter(
        " ".join(m.split()) for m in _code_spans(_strip_code(text))
    )


def headings(text: str) -> list[tuple[int, str]]:
    return [
        (len(m.group(1)), m.group(2).strip())
        for m in HEADING_RE.finditer(_strip_code(text))
    ]


def urls(text: str) -> Counter[str]:
    found: list[str] = []
    found += URL_RE.findall(text)
    found += MD_LINK_RE.findall(text)
    found += HTML_ATTR_URL_RE.findall(text)
    # Markdown prose glues sentence punctuation onto a bare URL; strip it so
    # the same link does not compare unequal to itself across a reflow.
    cleaned = (u.rstrip(".,;:") for u in found)
    # Fragment-only targets are in-document links, and they belong to the
    # anchor check instead. Counting them here would report a LOSS when a
    # heading and the link to it are correctly renamed together -- which is
    # the exact repair the anchor check asks for.
    return Counter(u for u in cleaned if not u.startswith("#"))


def numbers(text: str) -> Counter[str]:
    """Numeric literals outside fenced code, normalised.

    Thousands separators are removed and a trailing ``.0`` is dropped, so
    ``1,744`` and ``1744`` compare equal. Numbers inside code blocks are
    already covered by the byte-exact fence comparison.
    """
    out: list[str] = []
    for raw in NUMBER_RE.findall(_strip_code(text)):
        token = raw.replace(",", "").lstrip("+")
        if token.endswith(".0"):
            token = token[:-2]
        out.append(token)
    return Counter(out)


def frontmatter_keys(text: str) -> set[str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return set()
    keys = set()
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "-", "\t")):
            keys.add(line.split(":", 1)[0].strip())
    return keys


def table_rows(text: str) -> int:
    return len(TABLE_ROW_RE.findall(_strip_code(text)))


def footnotes(text: str) -> Counter[str]:
    return Counter(FOOTNOTE_RE.findall(_strip_code(text)))


def heading_slug(text: str) -> str:
    """Return the anchor slug a markdown renderer gives this heading.

    Inline markup is removed first: backticks, emphasis and HTML tags do not
    appear in the generated anchor.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_]", "", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def broken_anchors(text: str) -> set[str]:
    """In-document link targets that match no heading or explicit anchor."""
    targets = {a or b for a, b in ANCHOR_RE.findall(text)}
    available = {heading_slug(t) for _, t in headings(text)}
    available |= set(EXPLICIT_ANCHOR_RE.findall(text))
    available |= set(BRACE_ANCHOR_RE.findall(text))
    return {t for t in targets if t and t not in available}


def _lost(before: Counter[str], after: Counter[str]) -> list[tuple[str, int, int]]:
    return sorted(
        (item, n, after.get(item, 0))
        for item, n in before.items()
        if after.get(item, 0) < n
    )


def _gained(before: Counter[str], after: Counter[str]) -> list[tuple[str, int, int]]:
    return sorted(
        (item, before.get(item, 0), m)
        for item, m in after.items()
        if m > before.get(item, 0)
    )


def compare(before: str, after: str) -> tuple[list[str], list[str]]:
    """Return (losses, notes) for a rewrite of ``before`` into ``after``."""
    losses: list[str] = []
    notes: list[str] = []

    def check(
        name: str,
        b: Counter[str],
        a: Counter[str],
        *,
        report_gains: bool = True,
        limit: int = 25,
    ) -> None:
        rows = _lost(b, a)
        for item, n, m in rows[:limit]:
            losses.append(f"LOSS  {name}: {item!r} appeared {n}x, now {m}x")
        if len(rows) > limit:
            losses.append(f"LOSS  {name}: ...and {len(rows) - limit} more")
        if report_gains:
            for item, n, m in _gained(b, a)[:limit]:
                notes.append(f"NOTE  {name} added: {item!r} {n}x -> {m}x")

    # A dropped citation is the most damaging thing a rewrite can do to a
    # technical document, so links are checked first.
    check("url", urls(before), urls(after))

    # Code is not prose. It survives byte for byte.
    check("code block", fences(before), fences(after), report_gains=False)
    check("code fence language", fence_langs(before), fence_langs(after))
    check("inline code", inline_code(before), inline_code(after))

    # Every number is a claim somebody measured.
    check("number", numbers(before), numbers(after))
    check("footnote ref", footnotes(before), footnotes(after))

    hb, ha = headings(before), headings(after)

    title_b = next((t for lvl, t in hb if lvl == 1), None)
    title_a = next((t for lvl, t in ha if lvl == 1), None)
    if title_b is not None and title_b != title_a:
        losses.append(f"LOSS  title changed: {title_b!r} -> {title_a!r}")

    if len(ha) < len(hb):
        losses.append(f"LOSS  section count dropped: {len(hb)} headings -> {len(ha)}")

    levels_b = [lvl for lvl, _ in hb]
    levels_a = [lvl for lvl, _ in ha]
    if levels_b == levels_a:
        for (lvl, tb), (_, ta) in zip(hb, ha):
            if tb != ta:
                notes.append(f"NOTE  heading reworded: {'#' * lvl} {tb!r} -> {ta!r}")
    else:
        notes.append(
            f"NOTE  heading structure changed: levels {levels_b} -> {levels_a}. "
            "Confirm no section moved. Added subheadings are allowed."
        )

    for anchor in sorted(broken_anchors(after) - broken_anchors(before)):
        losses.append(
            f"LOSS  in-document link '#{anchor}' now points at no heading. "
            "A reworded heading orphaned it. Update the link or keep the anchor."
        )

    for key in sorted(frontmatter_keys(before) - frontmatter_keys(after)):
        losses.append(f"LOSS  front-matter key: {key!r}")

    tb, ta = table_rows(before), table_rows(after)
    if ta < tb:
        losses.append(f"LOSS  table rows: {tb} -> {ta}")

    return losses, notes


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: check_doc_preservation.py ORIGINAL REWRITTEN", file=sys.stderr
        )
        return 2
    before = Path(argv[1]).read_text(encoding="utf-8")
    after = Path(argv[2]).read_text(encoding="utf-8")
    losses, notes = compare(before, after)

    print(f"=== {argv[2]} ===")
    print(f"words {len(before.split())} -> {len(after.split())}")
    for line in notes:
        print(line)
    for line in losses:
        print(line)
    if losses:
        print(f"FAIL {len(losses)} preservation loss(es)")
        return 1
    print("PASS no preservation losses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
