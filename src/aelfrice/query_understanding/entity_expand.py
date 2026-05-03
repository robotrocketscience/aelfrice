"""R1 entity-expansion query rewriter (#291).

Detects capitalised tokens in the raw query, lowercases them, and
appends them to the BM25 term list. Each detected entity is emitted
`qf_multiplier` times. The BM25 integer-qf contract represents 2x
weight as a duplicated term -- fractional qf is not encodable.

The lab R1 round measured this exact rule (capitalised token detect
-> lowercase -> append at 2x qf) at +0.104 P@10 on the synthetic v2
corpus (ENTITY_POOL=12). Magnitude on live stores is unknown until
the #288 phase-1b retest -- the rule is carried as written.

Pure regex over the raw query string. Deterministic; no I/O; no
store coupling. Sub-microsecond per call.
"""
from __future__ import annotations

import re
from typing import Final

# Capitalised-token rule: an uppercase letter followed by at least one
# lowercase letter, then any mix of letters or digits. The two-character
# minimum avoids matching the sentence pronoun "I" while still catching
# short product names like "Go" or "Qt".
_CAPITALISED: Final[re.Pattern[str]] = re.compile(r"\b[A-Z][a-z][a-zA-Z0-9]*\b")

# 2x is the value the lab R1 round measured. The integer-qf BM25
# contract cannot encode fractional weights, so "boost X" means
# "emit X copies of the term".
DEFAULT_QF_MULTIPLIER: Final[int] = 2


def expand_with_capitalised_entities(
    raw_query: str,
    base_terms: list[str],
    *,
    qf_multiplier: int = DEFAULT_QF_MULTIPLIER,
) -> list[str]:
    """Return `base_terms` with capitalised-entity expansions appended.

    Detects capitalised tokens in `raw_query` (left-to-right match
    order), lowercases each, and appends `qf_multiplier` copies to
    a new list following `base_terms`. `base_terms` is not mutated.

    Entities that already appear in `base_terms` are still appended;
    the duplication is the boost mechanism in the integer-qf BM25
    contract.

    `qf_multiplier=1` reduces to "append once" (no boost). Values
    below 1 raise `ValueError`.
    """
    if qf_multiplier < 1:
        raise ValueError(f"qf_multiplier must be >= 1, got {qf_multiplier}")
    expansions: list[str] = []
    for match in _CAPITALISED.finditer(raw_query):
        token = match.group(0).lower()
        for _ in range(qf_multiplier):
            expansions.append(token)
    return list(base_terms) + expansions
