"""Reworded-restatement matcher for the SUPPORTS writer (#1650).

Decides whether one sentence the user typed restates one sentence of an
existing belief in other words. A later change logs each match as a
candidate SUPPORTS edge and applies it only to phantoms, after a
precision gate (see ``docs/design/feature-supports-writer.md``). This
module only decides. It opens no store, writes nothing, and nothing
calls it yet.

The rule is **stemmed residual containment**:

0. Normalize each sentence: apostrophe look-alikes become ``'``, NFKC
   folds fullwidth and compatibility forms, and format characters such
   as the soft hyphen and zero-width space are deleted. Every step below
   and every veto reads the normalized text.
1. Tokenize both texts with ``relationship_detector.extract_signals``.
   The residual is the tokens left after dropping stopwords, negation
   and quantifier words. Direction words (``DIRECTION_TOKENS``) stay in
   the residual even though the detector drops them, so "over the
   threshold" never matches "under the threshold".
2. Stem each residual token with the BM25 Porter stemmer.
3. Split the target belief into the sentences ingest would store
   (``ingest.stored_sentences``). The user sentence matches target
   sentence ``i`` when every stemmed residual token of the user
   sentence appears in sentence ``i``, and the user sentence has at
   least ``REWORDED_MIN_RESIDUAL_TOKENS`` distinct residual tokens.

A containing sentence is still rejected by any of these vetoes:

* ``mixed_script``: a word in either sentence mixes Latin letters with
  Cyrillic, Greek, Armenian or Cherokee ones, such as a Cyrillic ``o``
  inside ``not``.
* ``token_stuffing``: one residual word appears in the user sentence
  ``REWORDED_MAX_TOKEN_REPEATS`` times or more, counted before stemming.
* ``negation_mismatch``: exactly one of the two sentences is negated.
* ``quantifier_mismatch``: the sentences carry different sets of
  quantifier positions (``relationship_detector.QUANT_AXIS``),
  including one having a quantifier and the other none.
* ``value_conflict``: ``value_compare.find_conflicts`` reports a
  numeric or enum slot conflict between the user sentence and the
  containing target sentence.
* ``order_mismatch``: the shared tokens appear in a different order,
  so "A depends on B" never matches "B depends on A".

The first target sentence that contains the user sentence and passes
every veto is the match. The function is pure: no model, no
embeddings, no network, and the same input always gives the same
output.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from aelfrice.bm25 import tokenize_stemmed
from aelfrice.ingest import stored_sentences
from aelfrice.relationship_detector import QUANT_AXIS, Signals, extract_signals
from aelfrice.value_compare import extract_values, find_conflicts

# --- Pinned constants (see detector_thresholds.py) -----------------------

# The fewest distinct stemmed residual tokens a user sentence needs
# before it can match. Below this, a short sentence such as "the cache"
# is contained in almost any belief that mentions the same noun.
REWORDED_MIN_RESIDUAL_TOKENS: Final[int] = 3

# A user sentence in which one residual word, as typed, appears this
# many times is rejected. Containment is a set test, so repeating a
# word adds nothing a real restatement needs. Counted before stemming,
# so "iterate ... iterative ... iteratively" is not stuffing.
REWORDED_MAX_TOKEN_REPEATS: Final[int] = 3

# Direction and ordering words that the relationship detector drops as
# stopwords but that reverse a sentence's meaning when swapped. They
# stay in the residual so a swap fails containment. Each word whose
# partner is listed is listed too (``on`` with ``off``, ``in`` with
# ``out``), and passive ``by`` is kept so "A is called by B" does not
# match "A calls B".
DIRECTION_TOKENS: Final[frozenset[str]] = frozenset({
    "to", "from", "in", "into", "on", "onto", "out", "over", "under",
    "up", "down", "off", "above", "below", "before", "after", "upward",
    "downward", "inward", "outward", "by",
})

RULE_STEMMED_RESIDUAL_CONTAINMENT: Final[str] = "stemmed_residual_containment"

# --- Rejection reasons ----------------------------------------------------

REJECT_TOO_SHORT: Final[str] = "too_short"
REJECT_TOKEN_STUFFING: Final[str] = "token_stuffing"
REJECT_NO_TARGET_SENTENCES: Final[str] = "no_target_sentences"
REJECT_NOT_CONTAINED: Final[str] = "not_contained"
REJECT_NEGATION_MISMATCH: Final[str] = "negation_mismatch"
REJECT_QUANTIFIER_MISMATCH: Final[str] = "quantifier_mismatch"
REJECT_VALUE_CONFLICT: Final[str] = "value_conflict"
REJECT_ORDER_MISMATCH: Final[str] = "order_mismatch"
REJECT_MIXED_SCRIPT: Final[str] = "mixed_script"

REJECTION_REASONS: Final[tuple[str, ...]] = (
    REJECT_TOO_SHORT,
    REJECT_TOKEN_STUFFING,
    REJECT_NO_TARGET_SENTENCES,
    REJECT_NOT_CONTAINED,
    REJECT_NEGATION_MISMATCH,
    REJECT_QUANTIFIER_MISMATCH,
    REJECT_VALUE_CONFLICT,
    REJECT_ORDER_MISMATCH,
    REJECT_MIXED_SCRIPT,
)


@dataclass(frozen=True)
class RewordedMatch:
    """One user sentence matched to one sentence of a target belief.

    ``target_sentence_index`` indexes ``ingest.stored_sentences`` of the
    target's content, the same split the #1650 promotion count uses.
    ``matched_tokens`` are the user sentence's stemmed residual tokens
    in the order the user typed them.
    """

    target_sentence_index: int
    target_sentence: str
    matched_tokens: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class _Analysis:
    """Residual tokens and modality signals for one sentence.

    ``text`` is the normalized sentence the other fields were read from.
    """

    text: str
    mixed_script: bool
    surface: tuple[str, ...]
    residual: tuple[str, ...]
    signals: Signals
    quantifiers: frozenset[float]


# Entries kept per cache. One user sentence is checked against many
# beliefs, and one belief against many user sentences, so both sides
# repeat. Caching changes cost only: every cached value is immutable.
_CACHE_SIZE: Final[int] = 4096

# Characters that stand in for an apostrophe. Mapped to ``'`` before
# NFKC, which leaves most of them alone, so ``don\u02bct`` reads as
# ``don't`` to the negation check.
_APOSTROPHE_MAP: Final[dict[int, str]] = str.maketrans({
    "`": "'",  # grave accent
    "\u00b4": "'",  # acute accent
    "\u02bc": "'",  # modifier letter apostrophe
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u2032": "'",  # prime
})

# Scripts with letters that look like Latin ones. A word that mixes
# Latin with one of these is treated as a spoof.
_CONFUSABLE_SCRIPTS: Final[frozenset[str]] = frozenset({
    "ARMENIAN", "CHEROKEE", "CYRILLIC", "GREEK",
})

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\w+")


def _normalize(text: str) -> str:
    """The form of ``text`` that every rule and veto reads.

    Apostrophe look-alikes become ``'`` so a contraction such as
    ``don't`` reads as negated, NFKC folds fullwidth and compatibility
    forms to their plain letters, and format characters (soft hyphen,
    zero-width space and joiners, bidi controls) are deleted so they
    cannot split or hide a word.
    """
    text = text.translate(_APOSTROPHE_MAP)
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _has_mixed_script_word(text: str) -> bool:
    """True when one word mixes Latin letters with a look-alike script.

    A Cyrillic ``o`` (U+043E) inside ``not`` reads as ``not`` but tokenizes as a
    different word, which hides the negation from every veto.
    """
    for word in _WORD_RE.findall(text):
        scripts = {
            unicodedata.name(ch, "").split(" ", 1)[0]
            for ch in word if ch.isalpha()
        }
        if "LATIN" in scripts and scripts & _CONFUSABLE_SCRIPTS:
            return True
    return False


@lru_cache(maxsize=_CACHE_SIZE)
def _analyse(text: str) -> _Analysis:
    text = _normalize(text)
    signals = extract_signals(text)
    surface = tuple(
        tok for tok in signals.tokens
        if tok in signals.residual_content or tok in DIRECTION_TOKENS
    )
    residual = tuple(stem for tok in surface for stem in tokenize_stemmed(tok))
    quantifiers = frozenset(
        QUANT_AXIS[tok] for tok in signals.tokens if tok in QUANT_AXIS
    )
    return _Analysis(
        text=text, mixed_script=_has_mixed_script_word(text),
        surface=surface, residual=residual, signals=signals,
        quantifiers=quantifiers,
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _target_sentences(
    target_text: str,
) -> tuple[tuple[str, _Analysis, frozenset[str]], ...]:
    """Each stored sentence of a target with its analysis and residual set."""
    result: list[tuple[str, _Analysis, frozenset[str]]] = []
    for sentence in stored_sentences(target_text):
        analysis = _analyse(sentence)
        result.append((sentence, analysis, frozenset(analysis.residual)))
    return tuple(result)


def _first_occurrences(tokens: tuple[str, ...], keep: frozenset[str]) -> list[str]:
    return [tok for tok in dict.fromkeys(tokens) if tok in keep]


def _veto(
    user: _Analysis,
    user_set: frozenset[str],
    target: _Analysis,
) -> str | None:
    """The first veto that rejects a containing target sentence."""
    if target.mixed_script:
        return REJECT_MIXED_SCRIPT
    if user.signals.has_negation != target.signals.has_negation:
        return REJECT_NEGATION_MISMATCH
    if user.quantifiers != target.quantifiers:
        return REJECT_QUANTIFIER_MISMATCH
    if find_conflicts(extract_values(user.text), extract_values(target.text)):
        return REJECT_VALUE_CONFLICT
    if (_first_occurrences(user.residual, user_set)
            != _first_occurrences(target.residual, user_set)):
        return REJECT_ORDER_MISMATCH
    return None


def _evaluate(
    user_sentence: str, target_text: str,
) -> tuple[RewordedMatch | None, str | None]:
    user = _analyse(user_sentence)
    user_set = frozenset(user.residual)
    if len(user_set) < REWORDED_MIN_RESIDUAL_TOKENS:
        return None, REJECT_TOO_SHORT
    if user.mixed_script:
        return None, REJECT_MIXED_SCRIPT
    if max(Counter(user.surface).values()) >= REWORDED_MAX_TOKEN_REPEATS:
        return None, REJECT_TOKEN_STUFFING
    sentences = _target_sentences(target_text)
    if not sentences:
        return None, REJECT_NO_TARGET_SENTENCES
    first_veto: str | None = None
    for index, (sentence, target, target_set) in enumerate(sentences):
        if not user_set <= target_set:
            continue
        veto = _veto(user, user_set, target)
        if veto is None:
            match = RewordedMatch(
                target_sentence_index=index,
                target_sentence=sentence,
                matched_tokens=tuple(dict.fromkeys(user.residual)),
                rule=RULE_STEMMED_RESIDUAL_CONTAINMENT,
            )
            return match, None
        if first_veto is None:
            first_veto = veto
    return None, first_veto or REJECT_NOT_CONTAINED


def match_reworded(user_sentence: str, target_text: str) -> RewordedMatch | None:
    """Return the match if ``user_sentence`` restates a sentence of ``target_text``.

    ``user_sentence`` is one sentence as ingest stored it. ``target_text``
    is a belief's full content, split here with ``stored_sentences``.
    Returns ``None`` when no target sentence matches.
    """
    return _evaluate(user_sentence, target_text)[0]


def explain_rejection(user_sentence: str, target_text: str) -> str | None:
    """Return why ``match_reworded`` rejects the pair, or ``None`` if it matches.

    The value is one of ``REJECTION_REASONS``. When several target
    sentences contain the user sentence and all are vetoed, the reason
    is the veto of the first one.
    """
    return _evaluate(user_sentence, target_text)[1]


__all__ = [
    "DIRECTION_TOKENS",
    "REJECTION_REASONS",
    "REJECT_MIXED_SCRIPT",
    "REJECT_NEGATION_MISMATCH",
    "REJECT_NOT_CONTAINED",
    "REJECT_NO_TARGET_SENTENCES",
    "REJECT_ORDER_MISMATCH",
    "REJECT_QUANTIFIER_MISMATCH",
    "REJECT_TOKEN_STUFFING",
    "REJECT_TOO_SHORT",
    "REJECT_VALUE_CONFLICT",
    "REWORDED_MAX_TOKEN_REPEATS",
    "REWORDED_MIN_RESIDUAL_TOKENS",
    "RULE_STEMMED_RESIDUAL_CONTAINMENT",
    "RewordedMatch",
    "explain_rejection",
    "match_reworded",
]
