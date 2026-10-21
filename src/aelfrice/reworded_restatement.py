"""Reworded-restatement matcher for the SUPPORTS writer (#1650).

Decides whether one sentence the user typed restates one sentence of an
existing belief in other words. ``aelfrice.supports_log`` logs each
match as an unapplied SUPPORTS candidate, behind a default-off flag; a
later change applies them to phantoms only, after a precision gate (see
``docs/design/feature-supports-writer.md``). This module only decides.
It opens no store and writes nothing.

The rule is **stemmed residual containment**:

0. Normalize each sentence: characters that look like an apostrophe
   become ``'``, NFKC folds fullwidth and compatibility forms, format
   characters such as the soft hyphen and zero-width space are
   deleted, and ``cannot``
   becomes ``can not`` so it reads as negated, like ``can't``. Every
   step below and every veto reads the normalized text.
1. Tokenize both texts with ``relationship_detector.extract_signals``.
   The residual is the tokens left after dropping stopwords, negation
   and quantifier words. Direction words (``DIRECTION_TOKENS``) stay in
   the residual even though the detector drops them, so "over the
   threshold" never matches "under the threshold".
2. Stem each residual token with the BM25 Porter stemmer.
3. Split the target belief into the sentences ingest would store
   (``ingest.stored_sentences``), and split each sentence, and the user
   sentence, into clauses at commas, colons, semicolons, brackets and
   the words in ``_CLAUSE_WORDS``. The user sentence matches target
   sentence ``i`` when each user clause fits inside one clause of
   sentence ``i`` (all its stemmed residual tokens appear there), the
   next user clause fits the same target clause, or the one after it
   when the two sentences break there with the same kind of break and
   the user clause opens with the stem that target clause opens with,
   and the user sentence has at least ``REWORDED_MIN_RESIDUAL_TOKENS``
   distinct residual tokens that are not direction words. So neither
   "the scanner calls the lexer" nor "the scanner, calls the lexer"
   matches "the scanner calls the parser and the parser calls the
   lexer".

A containing sentence is still rejected by any of these vetoes:

* ``mixed_script``: a word in either sentence mixes Latin letters with
  Cyrillic, Greek, Armenian or Cherokee ones, such as a Cyrillic ``o``
  inside ``not``. A unit sign after a number (``20µs``, ``10kΩ``,
  ``5 °C``) is not read as a letter here or in the lookalike veto.
* ``lookalike``: the target sentence holds a word with a letter, mark,
  digit or symbol outside ASCII that the user sentence lacks, or a word
  the user sentence lacks that spells a negation, quantifier or
  polarity word with digits (``n0t``). Either one can hide a negation
  as an unknown extra word, such as a small-capital or Coptic ``o``
  inside ``not``, or an accented ``nevér`` beside a ``café`` the user
  also typed.
* ``token_stuffing``: one residual word other than a direction word
  appears in the user sentence ``REWORDED_MAX_TOKEN_REPEATS`` times or
  more, counted before stemming.
* ``negation_mismatch``: exactly one of the two sentences is negated.
* ``negation_scope``: both are negated, but some negation in the user
  sentence has no negation between the same residual words in the
  matched target clauses, so "do not deploy the billing service" never
  matches "deploy the billing service and do not deploy the search
  service".
* ``quantifier_mismatch``: the sentences carry different sets of
  quantifier positions (``relationship_detector.QUANT_AXIS``),
  including one having a quantifier and the other none.
* ``polarity_mismatch``: the sentences hold different sets of
  ``POLARITY_TOKENS``, the hedge, limit, condition, contrast,
  alternative and past-state words. A number whose digits are all zero
  counts as ``zero``, unless a colon joins it to another number as in
  the clock time ``10:00``. Containment only checks user tokens, so without
  this a target that adds "unable", "few", "unless", "hardly" or
  "used to" would match. Words that frame a claim as false or
  disputed ("wrong", "myth", "the claim that") count too, and a
  target sentence with a question mark is rejected unless the user
  sentence has one, since a question asserts nothing.
* ``value_conflict``: the user sentence holds a whole dotted or
  comma-grouped number (``3.12``, ``1,000``) that the target sentence
  does not, or ``value_compare.find_conflicts`` reports a numeric or
  enum slot conflict between the user sentence and the containing
  target sentence.
* ``order_mismatch``: the shared tokens appear in a different order
  in the matched target clauses, so "A depends on B" never matches
  "B depends on A".

The first target sentence, and within it the first run of clauses,
that contains the user sentence and passes every veto is the match.
The function is pure: no model, no embeddings, no network, and the
same input always gives the same output.

Known limits: containment checks only the user's tokens, so a target
that adds a content word outside ``POLARITY_TOKENS`` and the detector's
negation and quantifier words still matches, even when that word
weakens or reverses the claim. A floor on how much of the target the
user sentence covers was tried and reverted: legitimate rewordings sit
below any floor that rejects the probes. A user sentence crosses a
target clause boundary only at the same kind of break the target uses
there, so a two-clause restatement that swaps an equivalent break,
such as ``and`` for ``;`` or ``,``, or ``then`` for ``and then``, does
not match. In a synthetic sweep of 6000 two-clause pairs that differ
only in the break, this rule kept 678 of the 4463 pairs the
per-sentence rule matched. The rule gives up that recall to block a
user sentence that takes its subject from one target clause and its
object from another. The lookalike veto compares whole words that hold
a non-ASCII letter, mark, digit or symbol, so an inflected accented
word (``résumés`` against ``résumé``) or a symbol with different
spacing (``a → b`` against ``a→b``) reads as a lookalike and does not
match. Polarity words compare surface tokens, so an inflection change
(``lacks`` against ``is lacking``, ``refuses`` against ``refused``)
does not match, and ``once`` fires in its frequency sense as well as
its past-state sense. The later precision gate is the backstop.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from aelfrice.bm25 import tokenize_stemmed
from aelfrice.relationship_detector import QUANT_AXIS, Signals, extract_signals
from aelfrice.sentence_partition import stored_sentences
from aelfrice.value_compare import (
    NumericSlot,
    ValueSlots,
    extract_values,
    find_conflicts,
)

# --- Pinned constants (see detector_thresholds.py) -----------------------

# The fewest distinct stemmed residual tokens a user sentence needs
# before it can match. Below this, a short sentence such as "the cache"
# is contained in almost any belief that mentions the same noun.
# Direction words do not count, so "up to over" is too short.
REWORDED_MIN_RESIDUAL_TOKENS: Final[int] = 3

# A user sentence in which one residual word, as typed, appears this
# many times is rejected. Containment is a set test, so repeating a
# word adds nothing a real restatement needs. Counted before stemming,
# so "iterate ... iterative ... iteratively" is not stuffing. Direction
# words do not count, since "from A to B to C to D" repeats one legitimately.
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

# Words that weaken, limit, condition, date or reverse a claim. Most are
# detector stopwords, so containment never sees them, and a target that
# adds one ("The worker is unable to reach the host", "Few tests are flaky",
# "Deploys fail unless the cache is warm") says something other than
# the user sentence. Both sentences must hold the same set of them.
POLARITY_TOKENS: Final[frozenset[str]] = frozenset({
    # inability, failure and near-negation
    "unable", "fail", "fails", "failed", "failing", "hardly",
    "barely", "scarcely", "non", "zero", "lack", "lacks", "lacked",
    "lacking", "refuse", "refuses", "refused", "refusing",
    # hedges and modals
    "may", "might", "maybe", "perhaps", "possibly", "probably", "likely",
    "unlikely", "could", "would", "should", "seem", "seems", "appear",
    "appears", "allegedly", "supposedly", "reportedly", "apparently",
    "presumably",
    # partial quantities and limits
    "few", "fewer", "less", "more", "most", "many", "several", "some",
    "any", "half", "only", "almost", "nearly", "approximately", "roughly",
    # conditions and exceptions
    "if", "unless", "except", "excluding", "when", "whenever", "until",
    "while", "provided", "assuming",
    # contrast and alternatives
    "but", "although", "though", "however", "whereas", "or", "either",
    "instead", "rather", "otherwise",
    # past state or change of state
    "was", "were", "had", "used", "previously", "formerly", "anymore",
    "became", "went", "changed", "stopped", "ceased", "removed", "once",
    "drops", "dropped", "deprecate", "deprecates", "deprecated",
    # negative states
    "disabled", "unavailable", "unsupported", "invalid", "false",
    # frames that dispute or deny the claim they hold
    "wrong", "incorrect", "untrue", "mistaken", "myth", "misconception",
    "rumor", "claim", "claims", "claimed", "whether",
})

# A number such as ``0`` or ``0.0`` reads as the polarity word ``zero``:
# "0 tests pass" says what "no tests pass" says.
_ZERO_RE: Final[re.Pattern[str]] = re.compile(r"0+(?:[.,]0+)*")

# A colon and a digit on either side of a number, as in ``10:00``.
_CLOCK_BEFORE_RE: Final[re.Pattern[str]] = re.compile(r"\d:\Z")
_CLOCK_AFTER_RE: Final[re.Pattern[str]] = re.compile(r":\d")

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
REJECT_LOOKALIKE: Final[str] = "lookalike"
REJECT_POLARITY_MISMATCH: Final[str] = "polarity_mismatch"
REJECT_NEGATION_SCOPE: Final[str] = "negation_scope"

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
    REJECT_LOOKALIKE,
    REJECT_POLARITY_MISMATCH,
    REJECT_NEGATION_SCOPE,
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


# Words that start a new clause. A user sentence must fit inside the
# matching target clauses, so a subject from one clause and an object
# from the next never combine into a claim the target does not make.
# Polarity words such as ``but`` and ``or`` are left out: the polarity
# veto already rejects a target that adds one.
_CLAUSE_WORDS: Final[tuple[str, ...]] = ("and", "because", "since", "so", "then")

# Where one clause ends and the next begins.
_CLAUSE_BREAK_RE: Final[re.Pattern[str]] = re.compile(
    r"[,:;()\[\]]|\b(?:"
    + "|".join(_CLAUSE_WORDS) + r")\b",
    re.IGNORECASE,
)


# The stems of the residual words just before and just after one
# negation marker, or None where the sentence has none on that side.
_Anchor = tuple[str | None, str | None]


@dataclass(frozen=True)
class _Analysis:
    """Residual tokens and modality signals for one sentence.

    ``text`` is the normalized sentence the other fields were read from.
    """

    text: str
    mixed_script: bool
    foreign: frozenset[str]
    digit_spelled: frozenset[str]
    surface: tuple[str, ...]
    residual: tuple[str, ...]
    clauses: tuple[tuple[str, ...], ...]
    clause_anchors: tuple[frozenset[_Anchor], ...]
    gaps: tuple[frozenset[str], ...]
    signals: Signals
    quantifiers: frozenset[float]
    numbers: frozenset[str]
    polarity: frozenset[str]
    question: bool


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

# A run of characters between whitespace and ASCII punctuation.
_FOREIGN_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[^\s!-/:-@\[-`{-~]+")

# The detector reads ``can't`` as negation but not ``cannot``.
_CANNOT_RE: Final[re.Pattern[str]] = re.compile(r"\b(can)(not)\b", re.IGNORECASE)

# A unit sign after a number, with an optional space and SI prefix, as
# NFKC leaves it: the micro sign becomes Greek mu and the ohm sign Greek
# omega. Removed before the script checks, so ``20µs`` and ``10kΩ`` are
# quantities, not mixed-script or lookalike words.
_UNIT_SIGN_RE: Final[re.Pattern[str]] = re.compile(
    r"(\d\s?[A-Za-z]?)[\u03bc\u03a9\u03c9\u00b0]"
)

# Digits that stand in for letters. ``1`` reads as either ``l`` or ``i``.
_DIGIT_LETTERS: Final[tuple[dict[int, int], ...]] = (
    str.maketrans("0134578", "oleastb"),
    str.maketrans("0134578", "oieastb"),
)

# A whole dotted or comma-grouped number such as ``3.12.1`` or
# ``1,000,000``. The tokenizer splits it into ``3``, ``12`` and ``1``, so
# containment alone lets ``3.12`` match ``3.12.1`` and ``1,000`` match
# ``1,000,000``.
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\d+(?:[.,]\d+)*")


def _normalize(text: str) -> str:
    """The form of ``text`` that every rule and veto reads.

    Characters that look like an apostrophe become ``'`` so a
    contraction such as ``don't`` reads as negated, NFKC folds fullwidth and compatibility
    forms to their plain letters, and format characters (soft hyphen,
    zero-width space and joiners, bidi controls) are deleted so they
    cannot split or hide a word. ``cannot`` becomes ``can not`` so the
    detector reads it as negated, as it reads ``can't``.
    """
    text = text.translate(_APOSTROPHE_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _CANNOT_RE.sub(r"\1 \2", text)


def _reads_zero(text: str) -> bool:
    """True when ``text`` holds a number whose digits are all zero.

    A number that a colon joins to another, such as the ``00`` in the
    clock time ``10:00``, is not one.
    """
    return any(
        _ZERO_RE.fullmatch(found.group(0))
        and not _CLOCK_BEFORE_RE.search(text, 0, found.start())
        and not _CLOCK_AFTER_RE.match(text, found.end())
        for found in _NUMBER_RE.finditer(text)
    )


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


def _foreign_words(text: str) -> frozenset[str]:
    """Lowercased words that hold a letter, mark, digit or symbol outside ASCII.

    A word here is a run between whitespace and ASCII punctuation, so a
    symbol such as the arrow U+2192 is a word of its own. Words, not characters,
    are compared: a user sentence with ``café`` must not let the target
    hide a negation as ``nevér``.
    """
    return frozenset(
        word.lower() for word in _FOREIGN_WORD_RE.findall(text)
        if any(
            ord(ch) > 0x7F and unicodedata.category(ch)[0] in "LMNS"
            for ch in word
        )
    )


def _digit_spelled_words(tokens: tuple[str, ...]) -> frozenset[str]:
    """Tokens that spell a negation, quantifier or polarity word with digits.

    ``n0t`` reads as ``not`` and ``d0n`` before ``t`` as ``don't``, but
    the detector sees neither as negation.
    """
    found: set[str] = set()
    for index, tok in enumerate(tokens):
        if tok.isdigit() or tok.isalpha():
            continue
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        for table in _DIGIT_LETTERS:
            word = tok.translate(table)
            if (_negates(word) or word in QUANT_AXIS or word in POLARITY_TOKENS
                    or (nxt == "t" and _negates(f"{word}'t"))):
                found.add(tok)
    return frozenset(found)


@lru_cache(maxsize=_CACHE_SIZE)
def _negates(word: str) -> bool:
    """True when the detector reads ``word`` alone as a negation."""
    return extract_signals(word).has_negation


def _negation_anchors(
    tokens: tuple[str, ...], surface: frozenset[str],
) -> frozenset[_Anchor]:
    """The residual words on each side of every negation marker.

    A marker is a detector negation word, or a contraction that the
    tokenizer split into ``doesn`` and ``t``. Each anchor holds the
    stem of the nearest residual token before the marker and after it,
    or ``None`` when ``tokens`` hold none on that side. Callers pass one
    clause at a time.
    """
    anchors: set[_Anchor] = set()
    for index, tok in enumerate(tokens):
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if not (_negates(tok) or (nxt == "t" and _negates(f"{tok}'t"))):
            continue
        before = [t for t in tokens[:index] if t in surface]
        after = [t for t in tokens[index + 1:] if t in surface]
        anchors.add((
            _stem(before[-1]) if before else None,
            _stem(after[0]) if after else None,
        ))
    return frozenset(anchors)


def _stem(token: str) -> str:
    stems = tokenize_stemmed(token)
    return stems[0] if stems else token


def _anchor_held(anchor: _Anchor, held: frozenset[_Anchor]) -> bool:
    """True when some anchor in ``held`` agrees with ``anchor`` on each side.

    ``None`` on either side agrees with anything, so a negation with no
    residual word before it ("do not deploy X") still matches one that
    names its subject.
    """
    return any(
        all(a is None or b is None or a == b for a, b in zip(anchor, other))
        for other in held
    )


def _residual(signals: Signals) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The residual words as typed and their stems, in sentence order."""
    surface = tuple(
        tok for tok in signals.tokens
        if tok in signals.residual_content or tok in DIRECTION_TOKENS
    )
    return surface, tuple(stem for tok in surface for stem in tokenize_stemmed(tok))


def _break_kind(found: str) -> str:
    """The kind of one clause break: the lowercased word, or its punctuation.

    An opening and a closing bracket are one kind.
    """
    if found.isalpha():
        return found.lower()
    return "()" if found in "()[]" else found


def _gap_kind(kinds: set[str]) -> frozenset[str]:
    """The kind of the gap between two clauses, from the breaks inside it.

    A clause word decides it when the gap has one, so ``, and`` and
    ``and`` are one kind.
    """
    words = {kind for kind in kinds if kind.isalpha()}
    return frozenset(words or kinds)


def _clauses(
    text: str,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[frozenset[_Anchor], ...],
    tuple[frozenset[str], ...],
]:
    """The stemmed residual and negation anchors of each clause that has a
    residual, and the kind of the gap after each clause but the last.

    Anchors are read inside one clause, so a negation never anchors on a
    word from the clause before or after it. A word in ``_CLAUSE_WORDS``
    stays at the start of the clause it opens: the detector keeps
    ``because``, ``since`` and ``so`` as residual content, so the clauses
    must hold every residual token the whole sentence holds.
    """
    chunks: list[str] = []
    kinds: list[str] = []
    start = 0
    for found in _CLAUSE_BREAK_RE.finditer(text):
        chunks.append(text[start:found.start()])
        kinds.append(_break_kind(found.group(0)))
        start = found.start() if found.group(0).isalpha() else found.end()
    chunks.append(text[start:])
    stems_out: list[tuple[str, ...]] = []
    anchors_out: list[frozenset[_Anchor]] = []
    gaps_out: list[frozenset[str]] = []
    pending: set[str] = set()
    for index, chunk in enumerate(chunks):
        if index:
            pending.add(kinds[index - 1])
        signals = extract_signals(chunk)
        surface, stems = _residual(signals)
        if not stems:
            continue
        if stems_out:
            gaps_out.append(_gap_kind(pending))
        pending = set()
        stems_out.append(stems)
        anchors_out.append(_negation_anchors(signals.tokens, frozenset(surface)))
    return tuple(stems_out), tuple(anchors_out), tuple(gaps_out)


@lru_cache(maxsize=_CACHE_SIZE)
def _analyse(text: str) -> _Analysis:
    text = _normalize(text)
    signals = extract_signals(text)
    surface, residual = _residual(signals)
    clauses, clause_anchors, gaps = _clauses(text)
    numbers = frozenset(_NUMBER_RE.findall(text))
    script_text = _UNIT_SIGN_RE.sub(r"\1", text)
    quantifiers = frozenset(
        QUANT_AXIS[tok] for tok in signals.tokens if tok in QUANT_AXIS
    )
    return _Analysis(
        text=text, mixed_script=_has_mixed_script_word(script_text),
        foreign=_foreign_words(script_text),
        digit_spelled=_digit_spelled_words(signals.tokens),
        surface=surface, residual=residual, clauses=clauses,
        clause_anchors=clause_anchors, gaps=gaps, signals=signals,
        quantifiers=quantifiers,
        numbers=numbers, question="?" in text,
        polarity=(frozenset(signals.tokens) & POLARITY_TOKENS) | (
            frozenset({"zero"}) if _reads_zero(text) else frozenset[str]()
        ),
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


def _numeric_close(a: NumericSlot, b: NumericSlot) -> bool:
    """True when ``find_conflicts`` sees no conflict between the two values."""
    return not find_conflicts(ValueSlots((a,), ()), ValueSlots((b,), ()))


def _values_conflict(user: ValueSlots, target: ValueSlots) -> bool:
    """True when the two sentences bind a shared slot to different values.

    ``find_conflicts`` compares every pair across both sides, so a
    sentence naming two values of one key (``from v2.0 to v2.1``)
    conflicts with itself. Here a numeric key conflicts only when some
    value on one side has no close value on the other, so a sentence
    always matches its own restatement. Enum slots go to
    ``find_conflicts`` unchanged: it already compares groups as sets.
    """
    if find_conflicts(ValueSlots((), user.enum), ValueSlots((), target.enum)):
        return True
    shared = {s.key for s in user.numeric} & {s.key for s in target.numeric}
    for key in sorted(shared):
        mine = [s for s in user.numeric if s.key == key]
        theirs = [s for s in target.numeric if s.key == key]
        if any(not any(_numeric_close(a, b) for b in theirs) for a in mine):
            return True
        if any(not any(_numeric_close(a, b) for a in mine) for b in theirs):
            return True
    return False


def _first_occurrences(tokens: tuple[str, ...], keep: frozenset[str]) -> list[str]:
    return [tok for tok in dict.fromkeys(tokens) if tok in keep]


def _windows(user: _Analysis, target: _Analysis) -> list[tuple[int, int]]:
    """The runs of target clauses, as ``(first, last)``, that hold the user clauses.

    Each user clause must fit inside one target clause: every stemmed
    residual token of the one appears in the other. The next user clause
    fits the same target clause, so a user sentence split by more
    punctuation than the target still matches. It moves on to the next
    target clause only where the target splits the same content at the
    same kind of break: the user gap and the target gap are the same
    kind (``_gap_kind``), and the user clause opens with the stem the
    target clause opens with. So "the scanner, calls the lexer" and "the
    scanner calls the parser and the lexer" never take their subject
    from one target clause and their object from the next.
    """
    have = [frozenset(clause) for clause in target.clauses]
    runs: set[tuple[int, int]] = set()
    for first in range(len(have)):
        ends: set[int] = (
            {first} if frozenset(user.clauses[0]) <= have[first] else set()
        )
        for gap, clause in zip(user.gaps, user.clauses[1:]):
            ends = {
                nxt for end in ends for nxt in (end, end + 1)
                if nxt < len(have) and frozenset(clause) <= have[nxt]
                and (nxt == end or (
                    target.gaps[end] == gap
                    and clause[0] == target.clauses[nxt][0]
                ))
            }
        runs.update((first, end) for end in ends)
    return sorted(runs)


def _veto(
    user: _Analysis,
    user_set: frozenset[str],
    target: _Analysis,
    run: tuple[int, int],
) -> str | None:
    """The first veto that rejects a run of clauses holding the user sentence.

    ``run`` gives the first and last target clause of the run. Negation
    scope and order are compared inside the run only, so a negation or
    a repeated word in another clause of the target does not count.
    """
    held = frozenset[_Anchor]().union(
        *target.clause_anchors[run[0]:run[1] + 1]
    )
    if target.mixed_script:
        return REJECT_MIXED_SCRIPT
    if (not target.foreign <= user.foreign
            or not target.digit_spelled <= frozenset(user.signals.tokens)):
        return REJECT_LOOKALIKE
    if user.signals.has_negation != target.signals.has_negation:
        return REJECT_NEGATION_MISMATCH
    if not all(
        _anchor_held(anchor, held)
        for anchors in user.clause_anchors for anchor in anchors
    ):
        return REJECT_NEGATION_SCOPE
    if user.quantifiers != target.quantifiers:
        return REJECT_QUANTIFIER_MISMATCH
    if user.polarity != target.polarity or (target.question and not user.question):
        return REJECT_POLARITY_MISMATCH
    if not user.numbers <= target.numbers:
        return REJECT_VALUE_CONFLICT
    if _values_conflict(extract_values(user.text), extract_values(target.text)):
        return REJECT_VALUE_CONFLICT
    window = tuple(
        stem for clause in target.clauses[run[0]:run[1] + 1] for stem in clause
    )
    if (_first_occurrences(user.residual, user_set)
            != _first_occurrences(window, user_set)):
        return REJECT_ORDER_MISMATCH
    return None


def _content_stems(user: _Analysis) -> frozenset[str]:
    """The stems of the residual words that are not direction words."""
    return frozenset(
        stem for tok in user.surface if tok not in DIRECTION_TOKENS
        for stem in tokenize_stemmed(tok)
    )


def _evaluate(
    user_sentence: str, target_text: str,
) -> tuple[RewordedMatch | None, str | None]:
    user = _analyse(user_sentence)
    user_set = frozenset(user.residual)
    if len(_content_stems(user)) < REWORDED_MIN_RESIDUAL_TOKENS:
        return None, REJECT_TOO_SHORT
    if user.mixed_script:
        return None, REJECT_MIXED_SCRIPT
    typed = Counter(tok for tok in user.surface if tok not in DIRECTION_TOKENS)
    if max(typed.values(), default=0) >= REWORDED_MAX_TOKEN_REPEATS:
        return None, REJECT_TOKEN_STUFFING
    sentences = _target_sentences(target_text)
    if not sentences:
        return None, REJECT_NO_TARGET_SENTENCES
    first_veto: str | None = None
    for index, (sentence, target, target_set) in enumerate(sentences):
        if not user_set <= target_set:
            continue
        for run in _windows(user, target):
            veto = _veto(user, user_set, target, run)
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


def candidate_terms(user_sentence: str) -> tuple[str, ...]:
    """The words a belief must hold before ``user_sentence`` can match it.

    The residual words of the normalized sentence as typed, without
    direction words, each once, in sentence order. A target sentence
    matches only when it holds the stem of every residual word, so a
    search that requires all of these words narrows the candidates
    without ranking them away. The search index stems with its own
    stemmer, so a word pair this matcher stems alike and the index does
    not can still be missed. Returns ``()`` when the sentence has too
    few words to match anything (``REJECT_TOO_SHORT``).
    """
    user = _analyse(user_sentence)
    if len(_content_stems(user)) < REWORDED_MIN_RESIDUAL_TOKENS:
        return ()
    return tuple(dict.fromkeys(
        tok for tok in user.surface if tok not in DIRECTION_TOKENS
    ))


def explain_rejection(user_sentence: str, target_text: str) -> str | None:
    """Return why ``match_reworded`` rejects the pair, or ``None`` if it matches.

    The value is one of ``REJECTION_REASONS``. When several target
    sentences contain the user sentence and all are vetoed, the reason
    is the veto of the first one.
    """
    return _evaluate(user_sentence, target_text)[1]


__all__ = [
    "DIRECTION_TOKENS",
    "POLARITY_TOKENS",
    "REJECTION_REASONS",
    "REJECT_LOOKALIKE",
    "REJECT_MIXED_SCRIPT",
    "REJECT_NEGATION_MISMATCH",
    "REJECT_NOT_CONTAINED",
    "REJECT_NO_TARGET_SENTENCES",
    "REJECT_ORDER_MISMATCH",
    "REJECT_POLARITY_MISMATCH",
    "REJECT_QUANTIFIER_MISMATCH",
    "REJECT_TOKEN_STUFFING",
    "REJECT_TOO_SHORT",
    "REJECT_VALUE_CONFLICT",
    "REWORDED_MAX_TOKEN_REPEATS",
    "REWORDED_MIN_RESIDUAL_TOKENS",
    "RULE_STEMMED_RESIDUAL_CONTAINMENT",
    "RewordedMatch",
    "candidate_terms",
    "explain_rejection",
    "match_reworded",
]
