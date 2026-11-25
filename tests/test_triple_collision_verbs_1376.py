"""#1159 §3 / #1376: single-token relation verbs that are also plural nouns.

`triple_extractor` registers bare `tests`, `covers`, `follows`, `extends`,
`supports` and `replaces` as relation verbs. Each is also an ordinary plural
noun or a non-relational verb in commit prose. #1376 dropped them from a
separate bank that the commit hook stored triples from, and kept the read
bank that `rebuild_log` queries with byte-identical (operator ruling
2026-08-06). The commit hook no longer stores anything, and its bank went
with it. What stays pinned here is the read side: the same collision rows
still fire under the read bank, and its regexes, types, and swaps are
unchanged.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aelfrice.triple_extractor import _PATTERNS, extract_triples

# --- Real commit prose that the dropped templates fire on --------------
#
# The first two rows are #1376's own executed examples. The rest are drawn
# verbatim from `github/main` commit bodies in the 400-body window #1376
# measured, chosen because each fires a *different* one of the six dropped
# templates.
COLLISION_ROWS: tuple[str, ...] = (
    # #1376's seed rows, both executed in the issue body.
    "Two tests that encoded the old behaviour are removed.",
    "codex-skills tests unaffected",
    # `tests` — the dominant collision, 51 of 82 fires on the sample.
    "The two tests that missed their branches are covered now.",
    "Both tests pin that None stays distinguishable from absent.",
    # `covers` — 12 of 82.
    "The recognition predicate covers all four shapes this project emits.",
    "Once the gate covers every non-advisory check the required set is whole.",
    # `replaces` — 3 of 82.
    "The way the tests it replaces were written is not preserved.",
    # `follows` — 1 of 82.
    "The lowercase stage that follows it is str.lower, not casefold.",
    # `supports` and `extends` never fired on the sample, so the corpus
    # carries a constructed row for each. Otherwise nothing here would go
    # red if one were quietly removed from the read bank.
    "The three supports that hold the frame are bolted, not welded.",
    "Every belief that extends past the window is dropped.",
)

# --- The read path is byte-identical ----------------------------------


@pytest.mark.parametrize("text", COLLISION_ROWS)
def test_read_path_still_fires_on_the_same_rows(text: str) -> None:
    """Dropping the six templates from `_PATTERNS` would change what
    `rebuild_log` queries on every prompt. These rows keep that visible."""
    assert extract_triples(text) != [], (
        f"{text!r} no longer fires under the read bank; a template the "
        "commit hook's bank dropped has left the read path too"
    )


def test_read_path_bank_behaviour_is_byte_identical_to_pre_1376() -> None:
    """`_PATTERNS`' *digest* moved under #1376; its behaviour must not have.

    `_RelationPattern` gained a `template` field so a bank can be filtered
    by verb, which changed the dataclass repr and therefore the pinned
    manifest digest — forcing a `DETECTOR_THRESHOLDS_VERSION` bump for a
    change that alters nothing the extractor does. `DIGEST_HISTORY[2]`
    claims exactly that, and this is the check behind the claim.

    The signature below is over (regex source, edge type, swap) — the
    three things that decide what the read path matches and emits. It was
    computed on `github/main` at dde5a5f5, before this change, and is
    equal at this head. If a later change moves it, the read path moved
    with it and `context_rebuilder` is affected.
    """
    signature = hashlib.sha256(
        json.dumps(
            [(p.regex.pattern, p.edge_type, p.swap) for p in _PATTERNS]
        ).encode("utf-8")
    ).hexdigest()
    assert signature == (
        "ed7dc10d976d0f0ec0b0e222b8378987e4791227cd80c71ed8caaed27729afb7"
    )
