"""BM25F / FTS5 tokenisation parity (#1348, parent #1158 defect 2).

The BM25F index lane and the FTS5 lane are advertised as
interchangeable — `retrieval` swaps one for the other at
`posterior_weight = 0.0`, and `AELFRICE_BM25F=0` is the documented way
to debug one against the other. That only holds if they key their
vocabularies the same way, and they did not: `beliefs_fts` is declared
`porter unicode61` while `tokenize_stemmed` used `\\w+` and an unguarded
Porter stemmer.

What makes this worth a dedicated module rather than another case in
`test_bm25_index.py` is the shape of the bug that preceded it. The
comment justifying the old regex cited
`test_bm25_index.py::test_w0_equivalence_with_fts5` — a test that has
never existed — and the real test it meant,
`test_w0_topk_matches_fts5_baseline`, runs on three pure-ASCII
documents with no underscore and no diacritic. The approximation was
guarded only by inputs on which it could not fail.

So the oracle here is a **real** SQLite `fts5` table declared exactly as
`store.py` declares `beliefs_fts`, read back through
`fts5vocab(..., 'instance')` — never a second Python reimplementation of
what unicode61 is believed to do. And `test_corpus_distinguishes_the_two_
implementations` pins the corpus itself: it asserts the pre-#1348
pipeline *fails* on this fixture, so a future edit cannot quietly make
the parity test green by softening the inputs.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata

import pytest

# One import style, module-qualified throughout. Everything here is
# read through `bm25.` at call time so the tests pin whatever is bound
# then — the point is to pin the shipped pipeline, not function objects
# captured at import.
import aelfrice.bm25 as bm25
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# Exactly the declaration `store.py` uses for `beliefs_fts`. Written out
# rather than imported: if that declaration changes, this module should
# start failing, not silently follow it.
FTS5_DECL = "porter unicode61"

# The fixture the acceptance criterion asks for. Every line carries at
# least one input on which `\w+` + unguarded Porter disagrees with
# unicode61, and between them they cover each divergence class:
# underscore-joined identifiers, diacritics, mixed case, and short words
# (the class the byte guard fixes, and the one a hand-picked corpus is
# most likely to omit).
PARITY_CORPUS = (
    "ADD_TO_LIST appends to the queue",
    "snake_case_name and CamelCase and SCREAMING_SNAKE",
    "café naïve résumé jalapeño",
    "max_retry_count is 5 as of v4.1.0",
    "x86_64 builds vs arm64 builds",
    "it is as good as it was, s'il vous plaît",
    "MixedCase_Identifiers Are Common",
    "__dunder__ and _leading and trailing_",
    "running as it was, abusive organizations",
)

# Inputs the fix must not BREAK, as opposed to inputs it repairs. Kept
# out of PARITY_CORPUS deliberately: the pre-#1348 pipeline handled all
# of these correctly, so they fail that corpus's distinguishing
# assertion by construction. They are here because the first cut of this
# fix regressed every one of them, and the Latin-only corpus above could
# not see it — the same blind spot, one layer up.
NON_LATIN_REGRESSION_GUARD = (
    "الوَلَد",         # Arabic harakat: separators, not diacritics
    "בְּרֵאשִׁית",        # Hebrew points: separators
    "हिन्दी",           # Devanagari matras: separators
    "ភាសាខ្មែរ",        # Khmer: separators
    "한국어",           # NFD explodes these into conjoining jamo
    "Tiếng Việt",    # two marks: unicode61 folds neither
    "ばか",            # dakuten is not a removable diacritic
    "ǖber",          # U+01D5, the remove_diacritics=1 limit
    # Greek and Cyrillic: a *removable* mark (U+0301, U+0308, U+0306) on a
    # NON-ASCII base. Every entry above exercises a separator, a
    # normalisation blow-up or a mark unicode61 never removes, so none of
    # them can see a mark that IS in `_REMOVED_MARKS` sitting on a base
    # SQLite has no fold entry for — which is the one class the first cut
    # of this fix got wrong. `πότε` (when) and `ποτέ` (ever) are here as a
    # pair on purpose: over-folding collapses two distinct words into one
    # term, so the cost is precision inside the lane, not just a gap
    # against FTS5.
    "мой",           # и + U+0306; SQLite keeps it, ASCII-base rule keeps it
    "ёлка",          # е + U+0308
    "всё",
    "Ελλάδα",        # α + U+0301
    "πότε",
    "ποτέ",
)

# unicode61 folds U+00B5 MICRO SIGN to Greek mu (U+03BC) — a
# compatibility mapping that canonical decomposition does not perform,
# so `_fold_diacritics` cannot reach it. Held here as a named, asserted
# residual rather than dropped from the corpus, because a divergence
# that nothing records is how #1348 happened.
KNOWN_RESIDUAL = "the µs latency budget"


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-08-05T00:00:00Z",
        last_retrieved_at=None,
    )


def fts5_terms(text: str) -> list[str]:
    """The terms SQLite indexes for `text`, in offset order.

    Both virtual tables live in `temp` because the three-argument
    `fts5vocab('temp', <tbl>, 'instance')` form only parses there — the
    same reason `store.py::_ensure_fts5_query_probe` creates its probe
    in `temp` too.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE temp.probe USING fts5(t, "
            f"tokenize='{FTS5_DECL}')"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE temp.probe_terms "
            "USING fts5vocab('temp', 'probe', 'instance')"
        )
        conn.execute("INSERT INTO temp.probe(rowid, t) VALUES (1, ?)", (text,))
        rows = conn.execute(
            "SELECT term, offset FROM temp.probe_terms ORDER BY offset"
        ).fetchall()
    finally:
        conn.close()
    return [term for term, _offset in rows]


_LEGACY_PATTERN = re.compile(r"\w+", re.UNICODE)


def _legacy_tokenize_stemmed(text: str) -> list[str]:
    """The pre-#1348 pipeline: `\\w+`, no fold, unguarded Porter.

    Spelled out here rather than reached for through the shipped code,
    so this stays a fixed historical baseline even as `bm25` changes.
    """
    return [
        bm25._PORTER_STEMMER.stemWord(m.group(0).lower())
        for m in _LEGACY_PATTERN.finditer(text)
    ]


@pytest.mark.parametrize("text", PARITY_CORPUS)
def test_tokenize_stemmed_matches_the_fts5_term_list(text: str) -> None:
    """`tokenize_stemmed` reproduces what FTS5 actually indexes.

    Order-sensitive on purpose: comparing sorted sets would hide a
    pipeline that emits the right terms in the wrong positions, and the
    BM25F lane's term-frequency counts depend on the multiset, not the
    set.
    """
    assert bm25.tokenize_stemmed(text) == fts5_terms(text)


@pytest.mark.parametrize("text", PARITY_CORPUS)
def test_corpus_distinguishes_the_two_implementations(text: str) -> None:
    """Every fixture line must be one the old pipeline got wrong.

    This is the assertion that keeps the parity test honest. Without it
    someone can make `test_tokenize_stemmed_matches_the_fts5_term_list`
    pass by replacing the corpus with ASCII lowercase prose — which is
    precisely how the approximation survived review the first time.
    """
    assert _legacy_tokenize_stemmed(text) != fts5_terms(text)


def test_micro_sign_is_a_known_and_bounded_residual() -> None:
    """Parity is close, not exact, and the gap is named.

    SQLite folds U+00B5 to Greek mu; NFD does not. Special-casing it
    would buy 12 of 44,683 live beliefs, so it is recorded instead. If a
    later change makes this pass, the assertion should be deleted and
    the residual count in `tokenize_stemmed`'s docstring updated — it
    failing is the signal, not an excuse to loosen it.
    """
    ours = bm25.tokenize_stemmed(KNOWN_RESIDUAL)
    theirs = fts5_terms(KNOWN_RESIDUAL)
    assert ours != theirs
    assert "µ" in ours and "μ" in theirs
    # Everything apart from the micro sign still agrees.
    assert [t for t in ours if t != "µ"] == [t for t in theirs if t != "μ"]


@pytest.mark.parametrize(
    "text", ["ﬁle", "①", "x²", "ＡＢＣ", "½ cup", "Ⅸ"],
)
def test_compatibility_forms_are_not_folded(text: str) -> None:
    """The fold is NFD, and swapping it for NFKD is a regression.

    unicode61 applies essentially no compatibility mappings; NFKD
    applies all of them. Re-derived over 44,683 live beliefs by
    `benchmarks/bm25_fts5_divergence.py`, whose fold table scores each
    form against a no-fold baseline rather than in isolation: NFKD fixes
    23 documents that baseline gets wrong and breaks 49 it gets right,
    net -26 — it is net-negative against doing nothing, which is why #1348 did not
    take the normalisation form its own body proposed. These inputs are
    the ones that separate the two forms.
    """
    assert bm25.tokenize_stemmed(text) == fts5_terms(text)


def test_shipped_word_class_and_stem_guards_are_the_measured_ones() -> None:
    """Pin the values, not just the mechanism.

    The tests above would still pass if the pattern and the guard bounds
    were monkeypatched at runtime, so pin what actually ships. `3` and
    `64` are SQLite's own bounds, measured in UTF-8 bytes.
    """
    assert bm25._FTS5_TOKEN_PATTERN.pattern == r"[^\W_]+"

    # 2 bytes: below SQLite's floor, left alone. `is` -> `i` was the
    # single largest source of divergence on the live store.
    assert bm25._stem("is") == "is"
    assert bm25._stem("as") == "as"
    # 3 bytes, including a multi-byte character: stemmed.
    assert bm25._stem("μs") == "μ"
    assert bm25._stem("running") == "run"
    # 64 bytes stemmed, 65 passed through.
    assert bm25._stem("a" * 63 + "s") == "a" * 63
    assert bm25._stem("a" * 64 + "s") == "a" * 64 + "s"


def test_tokenize_still_keeps_underscores_whole() -> None:
    """`tokenize()` is deliberately *not* changed by #1348.

    Its consumers — consolidation blocking shingles, dedup,
    `relationship_detector`, `query_understanding.strategy` — compare
    against their own vocabularies and owe FTS5 nothing. Splitting them
    on `_` would move dedup behaviour with no measurement behind it, so
    the divergence is intentional and pinned here.
    """
    assert bm25.tokenize("ADD_TO_LIST") == ["add_to_list"]
    assert bm25.tokenize("max_retry_count") == ["max_retry_count"]
    assert bm25.tokenize("café") == ["café"]
    assert bm25.tokenize_stemmed("ADD_TO_LIST") == ["add", "to", "list"]


def test_bm25f_lane_answers_the_queries_fts5_answers() -> None:
    """The operator-visible symptom, end to end.

    Before #1348 the default BM25F lane returned zero results for each
    of these while the FTS5 lane returned a hit — the divergence was not
    a ranking nuance, it was a blank result set.
    """
    store = MemoryStore(":memory:")
    store.insert_belief(
        _mk("b1", "the max_retry_count knob caps reconnection attempts")
    )
    store.insert_belief(_mk("b2", "the café menu changes weekly"))
    store.insert_belief(_mk("b3", "garden grows tomatoes"))

    index = bm25.BM25Index.build(store, anchor_weight=0)
    for query in ("retry", "count", "cafe", "max_retry_count", "café"):
        bm25f = {bid for bid, _score in index.score(query, top_k=10)}
        fts5 = {b.id for b in store.search_beliefs(query, limit=10)}
        assert bm25f, f"BM25F lane returned nothing for {query!r}"
        assert bm25f == fts5, f"lanes disagree on {query!r}"


def test_which_precomposed_codepoints_fold_matches_sqlite() -> None:
    """Sweep every base+one-mark codepoint, not a sample of them.

    `test_removed_marks_matches_sqlite` establishes *which marks* are
    removable, and it does so by probing with an ASCII base (``zz{mark}zz``).
    That is the right way to learn the mark set and the wrong way to learn
    the fold rule: ``remove_diacritics=1`` is a per-**codepoint** table, so
    "U+0301 is removable after a" says nothing about U+0301 after alpha.
    Applying the mark set across scripts folded `ά`, `й` and `ё`, which
    SQLite leaves alone.

    A sample corpus cannot close this — `NON_LATIN_REGRESSION_GUARD` had
    eight entries and none of them could see it, because separators,
    conjoining jamo and non-removable marks are all different mechanisms.
    So this asks SQLite about the entire class: all 727 codepoints whose
    NFD is a base plus exactly one combining mark.

    The rule that comes back is exact — SQLite folds precisely those with
    an ASCII base, 375 of them, with no false folds and no false keeps.
    The four near-misses worth naming are `ǣ`, `ǯ`, `ǽ` and `ǿ`: their
    bases are ae, ezh and o-with-stroke, so a rule saying "Latin base"
    rather than "ASCII base" would wrongly fold all four.
    """
    folds: set[str] = set()
    keeps: set[str] = set()
    for codepoint in range(0x20, 0x110000):
        ch = chr(codepoint)
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) != 2 or not unicodedata.combining(decomposed[1]):
            continue
        if unicodedata.combining(decomposed[0]):
            continue
        terms = fts5_terms(ch)
        if terms == [decomposed[0].lower()]:
            folds.add(ch)
        elif terms == [ch.lower()]:
            keeps.add(ch)
        # anything else is a separator producing no token; not a fold rule.

    assert folds, "oracle produced no folds at all — the probe is broken"
    predicted = {
        ch
        for ch in folds | keeps
        if unicodedata.normalize("NFD", ch)[1] in bm25._REMOVED_MARKS
        and unicodedata.normalize("NFD", ch)[0].isascii()
        and unicodedata.normalize("NFD", ch)[0].isalpha()
    }
    assert predicted == folds, (
        "the shipped fold rule disagrees with SQLite. Wrongly folded: "
        f"{sorted(predicted - folds)}; wrongly kept: {sorted(folds - predicted)}"
    )

    # And the rule as written is the one actually running.
    for ch in sorted(folds):
        assert bm25._fold_diacritics(ch) == unicodedata.normalize("NFD", ch)[0]
    for ch in sorted(keeps):
        assert bm25._fold_diacritics(ch) == ch


def test_removed_marks_matches_sqlite() -> None:
    """Re-derive `_REMOVED_MARKS` from SQLite rather than trusting it.

    The set is the whole correctness argument for `_fold_diacritics`:
    unicode61 *removes* 25 combining marks and treats 628 others as
    token separators, so a fold keyed on `unicodedata.combining()`
    instead of on this set is wrong for 628 of 653 marks. Nothing in
    Python's Unicode tables encodes which is which — the only authority
    is SQLite, so this asks SQLite every run.

    A drift here means the host SQLite disagrees with the set the fold
    was measured against, which is a real signal and not a flaky test:
    it says the BM25F lane and `beliefs_fts` have started to disagree on
    this machine.
    """
    removed: set[str] = set()
    separators: set[str] = set()
    for codepoint in range(0x300, 0x2000):
        mark = chr(codepoint)
        if not unicodedata.combining(mark):
            continue
        terms = fts5_terms(f"zz{mark}zz")
        if len(terms) > 1:
            separators.add(mark)
        elif terms == ["zzzz"]:
            removed.add(mark)

    assert removed == set(bm25._REMOVED_MARKS), (
        "SQLite removes a different set of combining marks than "
        "_REMOVED_MARKS records"
    )
    # The asymmetry is the reason the set exists at all; if it ever
    # inverted, `unicodedata.combining()` would have been fine.
    assert len(separators) > 10 * len(removed)


@pytest.mark.parametrize("text", NON_LATIN_REGRESSION_GUARD)
def test_non_latin_text_is_not_over_folded(text: str) -> None:
    """The class the first cut of this fix got wrong.

    A blanket NFD + strip-combining-marks welded `الوَلَد` into one token
    where FTS5 emits three, turned `한국어` into jamo that match no row,
    and folded `ばか` to `はか` — which additionally makes the BM25F lane
    return documents FTS5 does not, a precision loss rather than a mere
    cross-lane gap.
    """
    assert bm25.tokenize_stemmed(text) == fts5_terms(text)


@pytest.mark.parametrize("text", NON_LATIN_REGRESSION_GUARD)
def test_non_latin_guard_is_a_regression_guard_not_a_fix(text: str) -> None:
    """State the premise the guard rests on: main already got these right.

    Without this, someone reading `test_non_latin_text_is_not_over_folded`
    would reasonably assume those inputs are part of what #1348 repairs
    and might weaken them alongside PARITY_CORPUS. They are the opposite
    — the pre-#1348 pipeline matched FTS5 on every one, so any failure
    there is a regression this branch introduced, not a gap it inherited.
    """
    assert _legacy_tokenize_stemmed(text) == fts5_terms(text)


# The step-1b undoubling pairs on which SQLite's Porter and
# snowballstemmer disagree, swept against the oracle rather than reasoned
# from the two rule statements (#1389). Reasoning from them is exactly
# how the first cut of this docstring got the class wrong: snowball
# undoubles only `bb dd ff gg mm nn pp rr tt`, so "every other double
# consonant" looks right — but SQLite's rule carries its own exception,
# `not (*L or *S or *Z)`, which puts `ll`, `ss` and `zz` back on the
# agreeing side.
UNDOUBLING_DIVERGENT_PAIRS = frozenset(
    {"cc", "hh", "jj", "kk", "qq", "vv", "ww", "xx", "yy"}
)


def test_undoubling_divergence_is_exactly_the_documented_pairs() -> None:
    """Pin the class `tokenize_stemmed`'s docstring names.

    A prose class in a docstring rots silently; this is the same claim,
    enforced. The probe stems `gra<XX>ed` so the stem ends in the doubled
    pair, which is what step 1b keys on.

    Asserted as set equality in **both** directions on purpose. A subset
    check passes when a pair stops diverging, and a superset check passes
    when a new one starts — and either would be a real change in the
    BM25F-vs-FTS5 residual that the docstring would then misdescribe.
    """
    diverging: set[str] = set()
    for consonant in "bcdfghjklmnpqrstvwxyz":
        word = f"gra{consonant}{consonant}ed"
        if bm25.tokenize_stemmed(word) != fts5_terms(word):
            diverging.add(consonant * 2)

    assert diverging == set(UNDOUBLING_DIVERGENT_PAIRS)

    # And the exception that makes the class non-obvious: these are
    # outside snowball's nine pairs yet still agree, because SQLite
    # exempts them too.
    for pair in ("ll", "ss", "zz"):
        word = f"gra{pair}ed"
        assert bm25.tokenize_stemmed(word) == fts5_terms(word), (
            f"{pair} is outside snowball's nine pairs but SQLite's "
            "*L/*S/*Z exception should keep both stemmers in agreement"
        )

def _unguarded_fold(text: str) -> str:
    """`_fold_diacritics` without the ASCII fast path (#1387).

    Spelled out rather than reached for, for the same reason as
    `_legacy_tokenize_stemmed`: this is the fixed reference the shortcut
    is measured against, so it must not follow `bm25` when `bm25`
    changes. `_REMOVED_MARKS` is read through the module on purpose —
    the mark set is shared data, and only the loop is duplicated here.
    """
    out: list[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            if ch not in bm25._REMOVED_MARKS:
                out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        if (
            len(decomposed) == 2
            and decomposed[1] in bm25._REMOVED_MARKS
            and decomposed[0].isascii()
            and decomposed[0].isalpha()
        ):
            out.append(decomposed[0])
        else:
            out.append(ch)
    return "".join(out)


_GUARD_CORPUS = PARITY_CORPUS + NON_LATIN_REGRESSION_GUARD + (KNOWN_RESIDUAL,)


def test_the_guard_corpus_exercises_both_sides_of_the_fast_path() -> None:
    """Both branches are reached, so the identity tests are not vacuous.

    A corpus that happened to be all-ASCII would make the differential
    below pass while proving only that the shortcut returns its input,
    and an all-non-ASCII corpus would never enter the shortcut at all.
    Assert the split rather than trust the fixtures to keep it.
    """
    taken = [t for t in _GUARD_CORPUS if t.isascii()]
    not_taken = [t for t in _GUARD_CORPUS if not t.isascii()]
    assert taken, "no all-ASCII fixture: the fast path is never entered"
    assert not_taken, "no non-ASCII fixture: the loop is never entered"


@pytest.mark.parametrize("text", _GUARD_CORPUS)
def test_ascii_fast_path_is_identity_on_the_corpus(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortcut changes speed and nothing else.

    Asserted at both levels the acceptance criterion names: the fold
    itself, and the token stream `BM25Index.build` consumes. The second
    is not implied by the first for a reader — `tokenize_stemmed` folds,
    splits, lowercases and stems, and it is the tokens that reach the
    index.

    The token-stream half swaps the fold out from under the *shipped*
    `tokenize_stemmed` rather than restating its pipeline here. A second
    copy of the split/lowercase/stem chain would be a second source of
    truth: it could drift from the real one and still agree with itself,
    which is how a parity test comes to assert nothing.
    """
    assert bm25._fold_diacritics(text) == _unguarded_fold(text)

    guarded_tokens = bm25.tokenize_stemmed(text)
    monkeypatch.setattr(bm25, "_fold_diacritics", _unguarded_fold)
    assert bm25.tokenize_stemmed(text) == guarded_tokens


def test_ascii_fast_path_is_identity_over_every_ascii_codepoint() -> None:
    """Exhaustive on the branch the shortcut actually changes.

    The corpus above samples; this does not. The fast path fires on
    exactly one class of input — strings where every codepoint is ASCII
    — so that class can be swept whole rather than sampled, which is the
    same standard `test_which_precomposed_codepoints_fold_matches_sqlite`
    holds the fold rule to. Every single ASCII codepoint and every
    ordered ASCII pair is 16,512 inputs, cheap and total.

    The argument the shortcut rests on is that ASCII takes neither loop
    branch, so both halves of it are asserted directly below rather than
    only through their consequence — a future Unicode revision that gave
    an ASCII codepoint a combining class or a two-codepoint NFD would
    silently invalidate the shortcut, and this says so in that voice
    instead of failing as an unexplained fold mismatch.
    """
    ascii_chars = [chr(cp) for cp in range(0x80)]
    for ch in ascii_chars:
        assert not unicodedata.combining(ch), (
            f"U+{ord(ch):04X} is now combining — the ASCII fast path in "
            "_fold_diacritics is no longer identity"
        )
        assert len(unicodedata.normalize("NFD", ch)) == 1, (
            f"U+{ord(ch):04X} now has a multi-codepoint NFD — the ASCII "
            "fast path in _fold_diacritics is no longer identity"
        )

    for a in ascii_chars:
        assert bm25._fold_diacritics(a) == _unguarded_fold(a)
        for b in ascii_chars:
            pair = a + b
            assert bm25._fold_diacritics(pair) == _unguarded_fold(pair)


def test_the_ascii_fast_path_is_actually_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one thing #1387 ships, and the only test that would notice it go.

    Every other test here asserts the guarded fold agrees with the
    unguarded one. The *unguarded* code satisfies that by construction, so
    deleting `if text.isascii(): return text` leaves all of them green — a
    perf change whose entire deliverable no assertion can see.

    This counts the per-character work instead. On all-ASCII input the loop
    must not run at all; on input with one non-ASCII codepoint it must, or
    the test would pass against a fold that had stopped folding.
    """
    calls: list[str] = []
    real_combining = unicodedata.combining
    monkeypatch.setattr(
        bm25.unicodedata,
        "combining",
        lambda ch: (calls.append(ch), real_combining(ch))[1],
    )

    assert bm25._fold_diacritics("plain ascii text, 123") == "plain ascii text, 123"
    assert calls == [], (
        "the ASCII fast path was not taken: _fold_diacritics entered the "
        "per-character loop on an all-ASCII string"
    )

    # The distinguishing half. Without it, a fold that returned `text`
    # unconditionally would pass the assertion above.
    assert bm25._fold_diacritics("café") == "cafe"
    assert calls != [], "the per-character loop must still run on non-ASCII input"
