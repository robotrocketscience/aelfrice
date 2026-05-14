"""Type-aware compression for retrieval pack (#434).

Pure deterministic compressor over `(Belief, locked)` per
`docs/design/feature-type-aware-compression.md`. Strategy is dispatched by
`belief.retention_class`:

  fact      → verbatim         (locked or not)
  unknown   → verbatim         (migration safety; locked or not)
  snapshot  → headline if not locked, else verbatim
  transient → stub if not locked, else verbatim

Locks always override retention class — this mirrors the existing
"L0 beliefs are never trimmed" rule in `retrieval.py`.

The module has no store reads, no clock reads, no env reads, no random.
It depends only on `models` for the `Belief` shape and retention-class
constants. Token estimator is duplicated from `retrieval._estimate_tokens`
to avoid a circular import (retrieval consumes this module).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .models import (
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_TRANSIENT,
    RETENTION_UNKNOWN,
    Belief,
)

STRATEGY_VERBATIM: Final[str] = "verbatim"
STRATEGY_HEADLINE: Final[str] = "headline"
STRATEGY_STUB: Final[str] = "stub"

MAX_HEADLINE_CHARS: Final[int] = 240

_HEADLINE_ELLIPSIS: Final[str] = "…"
_CODE_FENCE: Final[str] = "```"

# Mirrors `retrieval._CHARS_PER_TOKEN`. Duplicated to keep this module
# free of a retrieval import (retrieval.py imports compression, not the
# other way around).
_CHARS_PER_TOKEN: Final[float] = 4.0


@dataclass(frozen=True)
class CompressedBelief:
    """A `Belief` plus its packed-render form.

    `rendered_tokens` is monotone-non-increasing in `_estimate_tokens(belief.content)`:
    the compressor never produces a render that costs more than the source.
    """

    belief: Belief
    rendered: str
    rendered_tokens: int
    strategy: str


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _content_is_single_code_fence(content: str) -> bool:
    s = content.strip()
    if not s.startswith(_CODE_FENCE) or not s.endswith(_CODE_FENCE):
        return False
    inner = s[len(_CODE_FENCE):-len(_CODE_FENCE)]
    return _CODE_FENCE not in inner


def _headline(content: str) -> str:
    """Extract the leading sentence as a headline.

    - If content already fits within MAX_HEADLINE_CHARS and has no
      internal sentence boundary, return content unchanged (the
      headline strategy never expands the source).
    - If content is wholly a single code fence, return content
      unchanged (per impl-PR decision on open-question 3).
    - Otherwise split on the first `. ` or `.\\n` that falls outside
      a balanced code-fence span. If that split lands within
      MAX_HEADLINE_CHARS, return prefix + `…`.
    - Else hard-truncate at the last whitespace ≤ MAX_HEADLINE_CHARS
      and append `…`.
    """
    if _content_is_single_code_fence(content):
        return content

    sentence_end = _first_sentence_end_outside_fence(content)
    if sentence_end is not None and sentence_end <= MAX_HEADLINE_CHARS:
        prefix = content[:sentence_end].rstrip()
        if sentence_end >= len(content):
            return prefix
        return prefix + _HEADLINE_ELLIPSIS

    if len(content) <= MAX_HEADLINE_CHARS and sentence_end is None:
        return content

    cut = content.rfind(" ", 0, MAX_HEADLINE_CHARS + 1)
    if cut <= 0:
        cut = MAX_HEADLINE_CHARS
    return content[:cut].rstrip() + _HEADLINE_ELLIPSIS


def _first_sentence_end_outside_fence(content: str) -> int | None:
    """Return the index just past the first `. ` or `.\\n` that falls
    outside a balanced ``` code fence, or None if none exists."""
    in_fence = False
    i = 0
    n = len(content)
    while i < n:
        if content.startswith(_CODE_FENCE, i):
            in_fence = not in_fence
            i += len(_CODE_FENCE)
            continue
        if not in_fence and content[i] == ".":
            nxt = content[i + 1] if i + 1 < n else ""
            if nxt == " " or nxt == "\n":
                return i + 1
        i += 1
    return None


def _stub(belief: Belief) -> str:
    return f"[stub: belief={belief.id} class={RETENTION_TRANSIENT}]"


def compress_for_retrieval(belief: Belief, *, locked: bool) -> CompressedBelief:
    """Compress a belief for the retrieval pack.

    Pure and deterministic. See module docstring and
    `docs/design/feature-type-aware-compression.md` for the strategy table.

    Invariant: `rendered_tokens <= _estimate_tokens(belief.content)`.
    """
    source_cost = _estimate_tokens(belief.content)
    rc = belief.retention_class

    if locked or rc == RETENTION_FACT or rc == RETENTION_UNKNOWN:
        return CompressedBelief(
            belief=belief,
            rendered=belief.content,
            rendered_tokens=source_cost,
            strategy=STRATEGY_VERBATIM,
        )

    if rc == RETENTION_SNAPSHOT:
        rendered = _headline(belief.content)
        cost = _estimate_tokens(rendered)
        if cost > source_cost or rendered == belief.content:
            return CompressedBelief(
                belief=belief,
                rendered=belief.content,
                rendered_tokens=source_cost,
                strategy=STRATEGY_VERBATIM,
            )
        return CompressedBelief(
            belief=belief,
            rendered=rendered,
            rendered_tokens=cost,
            strategy=STRATEGY_HEADLINE,
        )

    if rc == RETENTION_TRANSIENT:
        rendered = _stub(belief)
        cost = _estimate_tokens(rendered)
        if cost > source_cost:
            return CompressedBelief(
                belief=belief,
                rendered=belief.content,
                rendered_tokens=source_cost,
                strategy=STRATEGY_VERBATIM,
            )
        return CompressedBelief(
            belief=belief,
            rendered=rendered,
            rendered_tokens=cost,
            strategy=STRATEGY_STUB,
        )

    return CompressedBelief(
        belief=belief,
        rendered=belief.content,
        rendered_tokens=source_cost,
        strategy=STRATEGY_VERBATIM,
    )
