"""Sentence extraction from conversation text.

Implements the Exp 57/61 dumb extraction pipeline: strip noise, split on boundaries,
discard fragments. No classification, no keyword filtering.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import NamedTuple

from aelfrice.models import ORIGIN_USER_TRANSCRIPT


# Triple-backtick region, including the language tag. Exported rather
# than inlined because the onboard path strips fences too (#1371 §10) and
# the two must agree: a second copy of this pattern is a divergence
# waiting to happen, and the whole defect was that only one path had it.
#
# Anchored to line starts (`(?m)^`), and deliberately so. Unanchored, a
# ``` appearing *inside prose* — a sentence about markdown, an inline code
# span — pairs with the opening delimiter of the next real fence and flips
# parity for the rest of the document: the prose tail is deleted as fence
# interior and the code body is promoted to prose. On the onboard path,
# which walks a repo's own documentation, that is common enough to matter
# — this repo's `docs/user/CONFIG.md` loses 128 of its 248 paragraphs to
# it. Both paths inherit the anchoring, which is the point of sharing one
# object. Up to three leading spaces are allowed, matching CommonMark; a
# closing delimiter is still required, so an unterminated fence matches
# nothing and its content survives, unchanged from before.
#
# The closing delimiter takes only trailing whitespace, never an info
# string. CommonMark allows a language on the *opening* fence alone, so a
# ```` ```python ```` line inside a block is body text — matching it as a
# close ends the span early, leaking the remainder of the block into the
# cleaned output along with the real closing delimiter as stray prose.
CODE_FENCE_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]{0,3}```[^\n]*\n[\s\S]*?^[ \t]{0,3}```[ \t]*\r?$"
)

# Minimum character length for a sentence fragment to be kept.
_MIN_LEN: int = 10

_CODE_SPAN_RE: re.Pattern[str] = re.compile(r"`[^`]*`")
_URL_RE: re.Pattern[str] = re.compile(r"https?://\S+")

# Spans that stay removed, because they commonly hold a credential (operator
# ruling, #1874). This is a fixed list of shapes, not a secret scanner: a token
# with no known shape in a URL's path or fragment, or alone in a code span, is
# kept. The rule may remove too much, because removal is what happened to every
# span before #1874.
#
# In a code span: any `=`, a `?` with text right after it (a query, with or
# without a scheme), a `:` and then `@` (userinfo, under any scheme), "bearer", or a
# credential word followed by a colon or a value.
_CREDENTIAL_CODE_RE: re.Pattern[str] = re.compile(
    r"=|\?\S|:\S*@|bearer"
    r"|(?:key|token|secret|passw(?:or)?d|pwd|auth\w*|cookie|credential)s?[\"']?(?:\s*:|\s+\S)",
    re.IGNORECASE,
)
# Known credential shapes, alone in a code span or anywhere in a URL: vendor key
# prefixes, a chat app's token or webhook, a JWT, and a private key header.
_KNOWN_CREDENTIAL_RE: re.Pattern[str] = re.compile(
    r"\b(?:sk-[\w-]{16,}|AIza[\w-]{30,}|gh[pousr]_\w{20,}|github_pat_\w{20,}|glpat-[\w-]{16,}"
    r"|A[KS]IA[0-9A-Z]{16}|[rs]k_live_\w{16,}|npm_\w{30,}|pypi-[\w-]{16,}|hf_\w{30,}"
    r"|xox[a-z]-[\w-]{10,}|\d{8,10}:[\w-]{30,}|eyJ[\w-]{5,}\.[\w-]{5,}\.)"
    r"|hooks\.slack\.com/services/|discord(?:app)?\.com/api/webhooks/"
    r"|BEGIN [A-Z ]*PRIVATE KEY"
)
# A run this long, with this many bits per character, reads as random. On the
# #1874 corpus it removes 0 of 344 spans. A hexadecimal token tops out at 4.0
# bits, so a digest stays.
_LONG_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9+=_-]{32,}")
_RANDOM_MIN_BITS: float = 4.5
# A code span right after one of these is the value of an assignment or a query.
_VALUE_INTRODUCERS: str = "=?&"
# A span longer than this stays removed. The longest in the #1874 corpus is 118
# characters. Entity extraction is quadratic in a run of dots, and a span at this
# length costs it about 0.13 ms, where 100,000 characters cost 10 s.
_SPAN_MAX_CHARS: int = 256


def _bits_per_char(token: str) -> float:
    """Shannon entropy of `token`, in bits per character."""
    length = len(token)
    return -sum(n / length * math.log2(n / length) for n in Counter(token).values())


def _holds_known_credential(span: str) -> bool:
    """True for a known credential shape, or a long run that reads as random."""
    return bool(_KNOWN_CREDENTIAL_RE.search(span)) or any(
        _bits_per_char(token) >= _RANDOM_MIN_BITS for token in _LONG_TOKEN_RE.findall(span)
    )


def _is_credential_url(url: str) -> bool:
    """True for a URL with a query string, an `@` anywhere, or a known credential.

    The whole URL goes, because a partial one needs a second rule about which
    part is safe. A `?` that only ends a sentence isn't a query string.
    """
    return (
        "@" in url
        or "?" in url.rstrip(".,;:!?)]'\"`*_")
        or _holds_known_credential(url)
    )


# Stands in for each kept span while the markdown and sentence passes run. Every
# pattern and `.strip()` reads it as whitespace, and `splitlines` doesn't break on
# it, so a pass sees what it would see if the span were a space. Only `[ \t]` can
# tell the two apart, so the list patterns name it.
_HOLD: str = "\x1f"


class SentenceForms(NamedTuple):
    """One sentence in the two forms that #1874 separates.

    `holed` has each inline code span and URL removed. Every decision reads it:
    the noise gate, the sub-floor rule, the length floor, and the classifier.
    `whole` keeps the spans, and it's the form that the store gets.
    """

    holed: str
    whole: str


def holed_form(sentence: str) -> str:
    """The form of a stored transcript sentence that decisions read (#1874).

    For a caller that holds only the stored text. It equals the `holed` half of
    `sentence_forms`, with one known exception: markdown emphasis inside a URL
    scheme, as in `*http*://example.invalid/a`. The emphasis pass assembles that
    URL after the URL pass has run, so `holed` keeps it, and once stored it can't
    be told apart from a URL that was held. This function removes it.
    """
    return _URL_RE.sub(" ", _CODE_SPAN_RE.sub(" ", sentence)).strip()


def decision_text(content: str, origin: str) -> str:
    """A stored belief's text as a detector reads it (#1874).

    Only a sentence the user typed into a transcript keeps its inline code and
    URLs, and a rule or value quoted inside a span doesn't count as stated.
    """
    return holed_form(content) if origin == ORIGIN_USER_TRANSCRIPT else content


def _spans_only(match: re.Match[str]) -> str:
    """What a pass that removes `match` leaves: the spans inside it, if any.

    Spans that touched stay touching, and spans with anything between them keep
    one space between them.
    """
    runs = re.findall(f"{_HOLD}+", match.group(0))
    return " ".join(runs) + " " if runs else ""


def sentence_forms(text: str) -> list[SentenceForms]:
    """Extract atomic sentences from conversation text, in both forms.

    Process:
    1. Strip code blocks (triple-backtick regions)
    2. Hold inline code spans and URLs out of the later passes
    3. Strip markdown formatting (headers, bold, italic, table rows, list markers)
    4. Split on newlines first
    5. Within each line, split on sentence-ending punctuation followed by space
    6. Discard fragments under 10 characters once the held spans are removed

    A U+001F in the input reads as a space, because that's the stand-in.
    """
    # Step 1: strip code blocks (triple-backtick regions, including language tag)
    cleaned: str = CODE_FENCE_RE.sub(" ", text).replace(_HOLD, " ")

    # Step 2: hold inline code spans and URLs out of the later passes. Backticks
    # pair first, across the whole text, and a URL ends at the hole a span leaves.
    code: list[str] = []
    held: list[str] = []

    def hold_code(match: re.Match[str]) -> str:
        span = match.group(0)
        # Backticks paired across lines aren't inline code, and what they
        # enclose is often a pasted diff or log, so that text stays removed.
        if len(span.splitlines()) > 1 or len(span) > _SPAN_MAX_CHARS:
            return " "
        if _CREDENTIAL_CODE_RE.search(span[1:-1]) or _holds_known_credential(span[1:-1]):
            return " "
        start = match.start()
        if start and match.string[start - 1] in _VALUE_INTRODUCERS:
            return " "
        code.append(span)
        return _HOLD

    cleaned = _CODE_SPAN_RE.sub(hold_code, cleaned)
    codes = iter(code)

    def hold_in_order(match: re.Match[str]) -> str:
        span = match.group(0)
        if span != _HOLD and (len(span) > _SPAN_MAX_CHARS or _is_credential_url(span)):
            return " "
        held.append(next(codes) if span == _HOLD else span)
        return _HOLD

    cleaned = re.sub(f"{_HOLD}|{_URL_RE.pattern}", hold_in_order, cleaned)

    # Step 3: strip markdown formatting
    # Headers: lines starting with one or more # characters
    cleaned = re.sub(r"^#{1,6}\s+", _spans_only, cleaned, flags=re.MULTILINE)
    # Bold: **text** or __text__
    cleaned = re.sub(r"\*{2}([^*]*)\*{2}", r"\1", cleaned)
    # Underscore-bold/italic must be at a true word boundary so
    # snake_case identifiers and file paths like ``auth_service.py``
    # are not mangled. Negative lookbehind/lookahead require non-
    # alphanumeric context outside the underscore (``\b`` does not
    # work because ``_`` is in ``\w``).
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_{2}([^_]*)_{2}(?![A-Za-z0-9])", r"\1", cleaned,
    )
    # Italic: *text* or _text_ (single, not double)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])", r"\1", cleaned,
    )
    # Markdown table rows: lines containing | characters
    cleaned = re.sub(
        r"^\s*\|.*\|\s*$",
        lambda m: " " + _spans_only(m),
        cleaned,
        flags=re.MULTILINE,
    )
    # List markers: leading -, *, +, or numbered list items
    cleaned = re.sub(rf"^[ \t{_HOLD}]*[-*+]\s+", _spans_only, cleaned, flags=re.MULTILINE)
    cleaned = re.sub(rf"^[ \t{_HOLD}]*\d+\.\s+", _spans_only, cleaned, flags=re.MULTILINE)

    # Every pass above keeps every stand-in, in order, so the next one read is
    # always the stand-in for the next held span, even in a discarded fragment.
    spans = iter(held)
    forms: list[SentenceForms] = []
    # Step 4: split on newlines first
    for line in cleaned.splitlines():
        # Step 5: within each line, split on sentence-ending punctuation followed
        # by space. The stop stays with the preceding sentence, and a span in the
        # whitespace goes with the following one.
        parts = re.sub(
            r"(?<=[.!?])\s+",
            lambda m: "\n" + _spans_only(m),
            line,
        ).split("\n")
        for part in parts:
            holed = part.replace(_HOLD, " ").strip()
            whole = re.sub(_HOLD, lambda _: next(spans), part).strip()
            # Step 6: discard fragments under 10 characters
            if len(holed) >= _MIN_LEN:
                forms.append(SentenceForms(holed, whole))

    return forms


def extract_sentences(text: str) -> list[str]:
    """The sentences of `text` as decisions read them: inline code and URLs removed.

    `sentence_forms` also gives the form with them kept. The result is what this
    function returned before #1874, except that a U+001F in the input reads as a
    space.
    """
    return [form.holed for form in sentence_forms(text)]
