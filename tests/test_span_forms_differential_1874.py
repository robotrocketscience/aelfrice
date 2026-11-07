"""#1874: keeping inline code and URLs changes the stored text and no decision.

The differential compares the shipped code with a frozen copy of the extraction
that ran before #1874. Admitted sentences, their boundaries, and their type
verdicts must be equal once span text is removed from what's stored.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from aelfrice.classification_core import USER_SOURCE, classify_sentence
from aelfrice.contradiction import _slot_conflict_preextracted
from aelfrice.derivation import DerivationInput, derive
from aelfrice.extraction import (
    CODE_FENCE_RE,
    extract_sentences,
    holed_form,
    sentence_forms,
)
from aelfrice.hook import (
    _belief_is_lock_candidate,
    _directive_window_spec,
    _format_stop_prompt,
)
from aelfrice.ingest import _ingest_turn
from aelfrice.introspect import _signals_for
from aelfrice.models import INGEST_SOURCE_TRANSCRIPT, ORIGIN_USER_TRANSCRIPT, Belief
from aelfrice.noise_filter import (
    is_stranded_capture_noise,
    is_transcript_noise,
    is_transcript_scaffolding,
)
from aelfrice.phantom_promotion_opportunity import _PHANTOM_RESIDUE_ALLOWED
from aelfrice.sentence_partition import (
    _looks_like_subfloor_noise,
    partition_sentences,
    stored_sentences,
)
from aelfrice.store import MemoryStore
from aelfrice.supports_log import iter_reworded_candidates
from aelfrice.value_compare import extract_values


def _frozen_extract_sentences(text: str) -> list[str]:
    """`extract_sentences` as it was before #1874. Don't update this copy."""
    cleaned: str = CODE_FENCE_RE.sub(" ", text)
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*{2}([^*]*)\*{2}", r"\1", cleaned)
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_{2}([^_]*)_{2}(?![A-Za-z0-9])", r"\1", cleaned,
    )
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])", r"\1", cleaned,
    )
    cleaned = re.sub(r"^\s*\|.*\|\s*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    sentences: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if len(part) >= 10:
                sentences.append(part)
    return sentences


def _frozen_partition(text: str) -> tuple[list[str], list[list[str]]]:
    """`partition_sentences` before #1874, on the frozen extraction."""
    full: list[str] = []
    between: list[list[str]] = []
    pending: list[str] = []
    for sentence in _frozen_extract_sentences(text):
        if is_transcript_noise(sentence):
            continue
        if _looks_like_subfloor_noise(sentence):
            pending.append(sentence)
        else:
            full.append(sentence)
            between.append(pending)
            pending = []
    return full, between


def _frozen_gate(prompt: str) -> bool:
    """The transcript logger's gate before #1874: True when it logs `prompt`."""
    if is_transcript_scaffolding(prompt):
        return False
    if is_transcript_noise(prompt):
        parts = [s for s in _frozen_extract_sentences(prompt) if s.strip()]
        return not all(is_transcript_noise(s) for s in parts)
    return True


_HEADS = (
    "Yes", "No", "Ready", "Nothing", "Polling", "yes", "ok", "thanks", "lgtm",
    "/review", "/loop 5m", "git log", "gh pr view", "uv run", "pytest", "python",
    "cd /tmp", "Running.", "<task-notification>", "<system-reminder>", "</x>",
    "⏺", "⎿", "┌─", "$", ">>>", "INFO", "ERROR", "remote:",
    "To", "curl", "Note:", "Run", "- ", "1. ", "# ", "| a |", "**", "_", "*",
)
_SPANS = (
    "`a`", "`make all`", "`a. b`", "`x? y`", "`long --flag --value=3`",
    "https://ex.invalid/a", "https://ex.invalid/a?t=1", "http://e.io/x.html",
    "`https://ex.invalid/c`", "`__init__`", "`don't do that`", "`never push to main`",
    "`prefer_fast`", "`maybe we go`", "https://ex.invalid/required-status-checks",
    "https://u:p@ex.invalid/a", "`KEY=value`", "`Authorization: Bearer abc`",
)
_TAILS = (
    "", ".", "!", "?", ":", " ok", " ok.", " now.", " please", " then go",
    " Then go ahead now.", " and then the rest of it.", "\nmore text on a line here",
    "\n`x y z`", ". Then go", ". ok", " |", "*", "_", "**",
)
_WORDS = (
    "word", "the", "widget", "**bold**", "*it*", "__u__", "_i_", "snake_case",
    "a.b.c", "src/a/b.py", "e.g.", "x.", "y?", "z!", "- ", "1. ", "# ", "| t |",
    "\n", "\n\n", "No", "Yes", "<x>", "www.ex.invalid/a", "`", "``", "```", "```py\n",
    "`a`", "`b c`", "`d\ne`", "https://ex.invalid/a", "https://ex.invalid/a.",
    "https://ex.invalid/a`b", "https://", "\t", "|", " ", "\x0b", "\x00",
)


def _corpus() -> list[str]:
    rng = random.Random(18742)
    prompts = [h + " " + s + t for h in _HEADS for s in _SPANS for t in _TAILS]
    # A span where a line's markup starts, which a pattern must read as a space.
    prompts += [
        s + mark + " the rest of this line stays here."
        for s in _SPANS for mark in (" 1.", " -", " *", " +", " #", " |", " 12.")
    ]
    prompts += [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 14)))
        for _ in range(6000)
    ]
    return prompts


CORPUS = _corpus()


def test_the_corpus_holds_the_shapes_that_flipped_before() -> None:
    """Without these, a green differential says nothing about the review's flips."""
    span_prompts = sum(1 for p in CORPUS if "`" in p or "://" in p)
    assert span_prompts > 12000
    assert "Yes https://ex.invalid/a. Then go" in CORPUS
    differing = sum(
        1 for p in CORPUS for form in sentence_forms(p) if form.holed != form.whole
    )
    assert differing > 3000


def test_extract_sentences_equals_the_frozen_extraction() -> None:
    mismatches = [p for p in CORPUS if extract_sentences(p) != _frozen_extract_sentences(p)]
    assert mismatches == []


def test_holed_form_recovers_the_holed_half_from_the_stored_text() -> None:
    """derive() and the hooks only hold the stored text, so they rely on this."""
    mismatches = [
        (p, form) for p in CORPUS for form in sentence_forms(p)
        if holed_form(form.whole) != form.holed
    ]
    assert mismatches == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Run *http*://example.invalid/a before the widget release.",
        "Run **https**://example.invalid/a before the widget release.",
        "Run http*s*://example.invalid/a before the widget release.",
        "Run __https__://example.invalid/a before the widget release.",
    ],
)
def test_holed_form_can_not_recover_a_url_that_emphasis_assembled(sentence: str) -> None:
    """The known limit in `holed_form`'s docstring: the URL is removed, where `holed` keeps it."""
    forms = sentence_forms(sentence)
    assert [form.holed for form in forms] == _frozen_extract_sentences(sentence)
    assert [form.holed for form in forms] == [form.whole for form in forms]
    assert holed_form(forms[0].whole) == "Run   before the widget release."
    assert holed_form(forms[0].whole) != forms[0].holed


def test_a_unit_separator_in_the_input_reads_as_a_space() -> None:
    """The one text change in `extract_sentences`, because it's the stand-in."""
    assert extract_sentences("The widget\x1frelease needs a look.") == ["The widget release needs a look."]
    assert _frozen_extract_sentences("The widget\x1frelease needs a look.") == ["The widget\x1frelease needs a look."]


def test_the_admitted_sentences_and_the_sub_floor_clauses_equal_the_frozen_ones() -> None:
    def holed(partition: tuple[list[str], list[list[str]]]) -> tuple[list[str], list[list[str]]]:
        full, between = partition
        return [holed_form(s) for s in full], [[holed_form(c) for c in g] for g in between]

    mismatches = [p for p in CORPUS if holed(partition_sentences(p)) != _frozen_partition(p)]
    assert mismatches == []


def test_stored_sentences_reads_the_holed_form() -> None:
    mismatches = [p for p in CORPUS if stored_sentences(p) != _frozen_partition(p)[0]]
    assert mismatches == []


def test_every_type_verdict_equals_the_one_on_the_holed_text() -> None:
    mismatches = []
    for prompt in CORPUS:
        whole, _ = partition_sentences(prompt)
        for sentence, frozen in zip(whole, _frozen_partition(prompt)[0], strict=True):
            derived = derive(
                DerivationInput(
                    raw_text=sentence,
                    source_kind=INGEST_SOURCE_TRANSCRIPT,
                    source_path="transcript",
                    raw_meta={"role": "user"},
                    ts="2026-11-06T00:00:00+00:00",
                )
            ).belief
            expected = classify_sentence(frozen, USER_SOURCE)
            got = (derived.type, derived.alpha, derived.beta) if derived else None
            want = (
                (expected.belief_type, expected.alpha, expected.beta)
                if expected.persist else None
            )
            if got != want:
                mismatches.append((prompt, sentence, got, want))
    assert mismatches == []


@pytest.mark.parametrize(
    ("sentence", "belief_type"),
    [
        ("See https://ex.invalid/required-status-checks for the widget.", "factual"),
        ("Use `don't do that` as the widget title.", "factual"),
        ("Set `prefer_fast` on the widget.", "factual"),
    ],
)
def test_span_text_can_not_set_the_type(sentence: str, belief_type: str) -> None:
    belief = derive(
        DerivationInput(
            raw_text=sentence,
            source_kind=INGEST_SOURCE_TRANSCRIPT,
            source_path="transcript",
            raw_meta={"role": "user"},
            ts="2026-11-06T00:00:00+00:00",
        )
    ).belief
    assert belief is not None
    assert belief.type == belief_type
    assert belief.content == sentence


@pytest.mark.parametrize(
    "prompt",
    [
        "Which `a. b` one?",
        "Is it `a? b` right?",
        "Use `maybe we go` here as title.",
        "| col | https://ex.invalid/x|",
        "thanks https://ex.invalid/a",
        "curl -sS https://ex.invalid/a",
        "GET https://ex.invalid/x 200",
        "ok `git status`",
    ],
)
def test_span_text_can_not_change_what_ingest_stores(tmp_path: Path, prompt: str) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _ingest_turn(store, prompt, "transcript", role="user")
        count = store.count_beliefs()
    finally:
        store.close()
    frozen = [
        s for s in _frozen_partition(prompt)[0] if classify_sentence(s, USER_SOURCE).persist
    ]
    assert count == len(frozen)


def test_a_rule_quoted_in_a_span_is_not_a_lock_candidate() -> None:
    def belief(content: str, origin: str) -> Belief:
        return Belief(
            id="b1", content=content, content_hash="h", alpha=3.0, beta=1.0,
            type="factual", lock_level="none", locked_at=None,
            created_at="2026-11-06T00:00:00+00:00", last_retrieved_at=None,
            session_id="s1", origin=origin,
        )

    quoted = "Use `never push to main` as the widget title."
    assert not _belief_is_lock_candidate(belief(quoted, ORIGIN_USER_TRANSCRIPT), "s1")
    assert _belief_is_lock_candidate(
        belief("Never push to main on the widget.", ORIGIN_USER_TRANSCRIPT), "s1",
    )


def test_stored_sentences_is_exact_on_an_emphasized_scheme() -> None:
    """It reads the holed half directly, so a phantom's split never moves."""
    text = "Run *http*://example.invalid/a before the widget release."
    assert stored_sentences(text) == _frozen_partition(text)[0]


def test_a_detector_reads_other_origins_as_stored() -> None:
    """Only a sentence from a prompt has a holed form. Onboard text never did."""
    onboard = Belief(
        id="b1", content="Use `never push to main` as the widget title.", content_hash="h",
        alpha=3.0, beta=1.0, type="factual", lock_level="none", locked_at=None,
        created_at="2026-11-06T00:00:00+00:00", last_retrieved_at=None,
        session_id="s1", origin="user_validated",
    )
    assert _belief_is_lock_candidate(onboard, "s1")


@pytest.mark.parametrize(
    ("source_kind", "source_path", "role"),
    [
        ("filesystem", "docs/a.md", None),
        (INGEST_SOURCE_TRANSCRIPT, "session-end", "user"),
        (INGEST_SOURCE_TRANSCRIPT, "transcript", None),
    ],
)
def test_the_classifier_reads_other_sources_as_stored(
    source_kind: str, source_path: str, role: str | None,
) -> None:
    text = "Use `never push to main` as the widget title."
    derived = derive(
        DerivationInput(
            raw_text=text, source_kind=source_kind, source_path=source_path,
            raw_meta={"role": role} if role else None,
            ts="2026-11-06T00:00:00+00:00",
        )
    ).belief
    assert derived is not None
    assert derived.type == classify_sentence(text, source_path).belief_type
    assert derived.type == "correction"


def test_a_lock_prompt_takes_no_window_from_inside_a_span() -> None:
    """A correction-class belief is a candidate, so its window must come from prose."""
    quoted = Belief(
        id="b1", content="Use `always remember this for 7 days` as the widget title.",
        content_hash="h", alpha=3.0, beta=1.0, type="correction", lock_level="none",
        locked_at=None, created_at="2026-11-06T00:00:00+00:00", last_retrieved_at=None,
        session_id="s1", origin=ORIGIN_USER_TRANSCRIPT,
    )
    assert _directive_window_spec(quoted.content) is not None
    assert "--for" not in _format_stop_prompt([quoted])


def _typed(content: str, origin: str = ORIGIN_USER_TRANSCRIPT) -> Belief:
    return Belief(
        id="b1", content=content, content_hash="h", alpha=3.0, beta=1.0, type="factual",
        lock_level="none", locked_at=None, created_at="2026-11-06T00:00:00+00:00",
        last_retrieved_at=None, session_id="s1", origin=origin,
    )


def test_a_value_in_a_span_does_not_conflict_with_a_lock() -> None:
    lock = _typed("The widget timeout is 60 seconds always.", "user_stated")
    pairs = [(lock, extract_values(lock.content))]
    quoted = "Set `timeout 30` on the widget before the release."
    assert _slot_conflict_preextracted(_typed(quoted), pairs) is None
    assert _slot_conflict_preextracted(_typed(quoted, "user_validated"), pairs) == "b1"


def test_the_introspect_noise_flag_reads_the_holed_form(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        # A stranded header once holed, and no header as stored.
        typed = _typed("Recommendation: `zqtool --check`")
        assert is_stranded_capture_noise(holed_form(typed.content))
        assert not is_stranded_capture_noise(typed.content)
        assert _signals_for(store, typed, {}).noise
    finally:
        store.close()


def test_a_sub_floor_clause_keeps_its_span_for_the_anchor() -> None:
    text = "Run the first thing here now.\nUse this one: `zqtool`\nThen do the second thing now."
    full, between = partition_sentences(text)
    assert full == ["Run the first thing here now.", "Then do the second thing now."]
    assert between == [[], ["Use this one: `zqtool`"]]


def test_a_phantom_with_a_span_still_fails_the_residue_check() -> None:
    """Promotion keeps the phantom's whole text, and a span in it stays residue."""
    phantom = "Always run `zqtool --frobnicate` before the widget release starts."
    residue = phantom
    for sentence in stored_sentences(phantom):
        residue = residue.replace(sentence, " ", 1)
    assert any(not c.isspace() and c not in _PHANTOM_RESIDUE_ALLOWED for c in residue)


# --- the transcript logger's gate, on the real handler ----------------------


def _log(tmp_path: Path, prompt: str, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    from aelfrice import transcript_logger

    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dot"))
    tdir = tmp_path / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transcript_logger, "transcripts_dir", lambda: tdir)
    transcript_logger._handle_user_prompt_submit(
        {"session_id": "s1", "prompt": prompt, "cwd": "/tmp"}
    )
    out = tdir / transcript_logger.TURNS_FILENAME
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


@pytest.mark.parametrize(
    ("prompt", "logged"),
    [
        # Noise as a whole, and its sentences are [False, True] once holed: the
        # gate keeps it only by reading sentences.
        ("No `x` it broke. Yes keep it", True),
        # Noise as a whole, with no sentence past the floor once holed.
        ("No https://ex.invalid/a. Then go", False),
        ("Yes https://ex.invalid/a. ok", False),
        ("Ready http://e.io/x.html. ok", False),
        ("Yes `widget_flag_name`.", False),
        # Noise as a whole, and every holed sentence is noise, while a sentence
        # that keeps its span isn't. These pin that the gate reads the holed form.
        ("Yes ok. `x` Yes keep going", False),
        ("Yes ok. https://e.io/a Yes keep going", False),
        ("Yes done! `a` No more of it", False),
    ],
)
def test_the_logger_gate_equals_the_frozen_one_on_the_real_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    logged: bool,
) -> None:
    assert _frozen_gate(prompt) is logged
    assert bool(_log(tmp_path, prompt, monkeypatch)) is logged


def test_the_support_log_matches_on_the_holed_form(tmp_path: Path) -> None:
    """A reworded restatement can't be earned or lost through span text."""
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        store.insert_belief(Belief(
            id="p1",
            content="The scanner reliably detects leaked secrets in committed source code.",
            content_hash="h_p1", alpha=1.0, beta=1.0, type="factual", lock_level="none",
            locked_at=None, created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
            session_id="s1", origin="agent_inferred",
        ))
        turn = _ingest_turn(
            store, "The scanner detects `zqextra` leaked secrets in source code.",
            "transcript", session_id="s2", created_at="2026-10-02T00:00:00Z", role="user",
        )
        bid = turn.inserted[0]
        log_id = store._conn.execute(  # noqa: SLF001
            "SELECT id FROM ingest_log WHERE derived_belief_ids LIKE ?", (f'%"{bid}"%',),
        ).fetchone()[0]
        candidates = list(iter_reworded_candidates(store, [(bid, str(log_id))]))
    finally:
        store.close()
    assert [c.target_id for c in candidates if c.match is not None] == ["p1"]


# --- spans that stay removed (operator ruling) ------------------------------

# Built at runtime, so this source holds no full key shape for a pattern scan to hit.
# The body is low in entropy, so a prefix's own pattern is what removes each one.
_BODY = "a1b2c3d4" * 5
_KNOWN_SHAPES = {
    "model vendor, new": "sk" + "-ant-api03-" + _BODY,
    "model vendor, project": "sk" + "-proj-" + _BODY,
    "model vendor, old": "sk" + "-" + _BODY,
    "model vendor, cloud": "AI" + "za" + _BODY + "abcde",
    "code host, classic": "gh" + "p_" + _BODY,
    "code host, oauth": "gh" + "o_" + _BODY,
    "code host, server": "gh" + "s_" + _BODY,
    "code host, refresh": "gh" + "r_" + _BODY,
    "code host, user": "gh" + "u_" + _BODY,
    "code host, fine-grained": "github" + "_pat_" + _BODY,
    "code host, other": "gl" + "pat-" + _BODY[:20],
    "cloud access key": "AK" + "IA" + "Q1W2E3R4T5Y6U7I8",
    "cloud session key": "AS" + "IA" + "Q1W2E3R4T5Y6U7I8",
    "payments": "sk" + "_live_" + _BODY,
    "payments, restricted": "rk" + "_live_" + _BODY,
    "package registry": "np" + "m_" + _BODY,
    "package index": "py" + "pi-" + _BODY,
    "model hub": "h" + "f_" + _BODY,
    "chat bot token": "xo" + "xb-1234567890-" + _BODY,
    "chat user token": "xo" + "xp-1234567890-" + _BODY,
    "chat app bot token": "123456" + "7890:" + _BODY,
    "JWT": "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0In0." + _BODY,
    "RSA key header": "-----BEG" + "IN RSA PRIVATE KEY-----",
    "key header": "-----BEG" + "IN PRIVATE KEY-----",
    "SSH key header": "-----BEG" + "IN OPENSSH PRIVATE KEY-----",
    "random token": "Zk3vQ9xL" + "2mN7bT5c" + "R8wY1pH6" + "dF4gJ0sA" + "3eU7iO9q",
    "encoded token": "aGVsbG8g" + "d29ybGQg" + "dGhpcyBp" + "cyBhIHNl" + "Y3JldCB2" + "YWx1",
}


@pytest.mark.parametrize("value", _KNOWN_SHAPES.values(), ids=_KNOWN_SHAPES.keys())
def test_a_known_credential_alone_in_a_code_span_stays_removed(value: str) -> None:
    sentence = f"Run `{value}` before the widget release."
    assert [form.whole for form in sentence_forms(sentence)] == _frozen_extract_sentences(sentence)


@pytest.mark.parametrize(
    "value",
    # A key header holds spaces, so it can't sit inside a URL.
    [value for name, value in _KNOWN_SHAPES.items() if "header" not in name],
    ids=[name for name in _KNOWN_SHAPES if "header" not in name],
)
def test_a_known_credential_in_a_url_stays_removed(value: str) -> None:
    sentence = f"Read https://example.invalid/x/{value} before the widget release."
    assert [form.whole for form in sentence_forms(sentence)] == _frozen_extract_sentences(sentence)


@pytest.mark.parametrize(
    "host", ["hooks." + "slack.com/services/T0/B0", "discord" + ".com/api/webhooks/1/a", "discordapp" + ".com/api/webhooks/1/a"],
)
def test_a_chat_webhook_url_stays_removed(host: str) -> None:
    sentence = f"Post to https://{host} for the widget release."
    assert [form.whole for form in sentence_forms(sentence)] == _frozen_extract_sentences(sentence)


def test_a_known_credential_never_reaches_the_store(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        for i, value in enumerate(_KNOWN_SHAPES.values()):
            _ingest_turn(store, f"Run `{value}` before release {i}.", "transcript", role="user")
        stored = store._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs UNION ALL SELECT raw_text FROM ingest_log "
            "UNION ALL SELECT entity_lower FROM belief_entities"
        ).fetchall()
    finally:
        store.close()
    assert stored
    assert not [row for row in stored if "`" in str(row[0]) or _BODY.lower() in str(row[0]).lower()]


def _token(distinct: int, length: int) -> str:
    """`distinct` symbols used equally, so the entropy is log2(distinct) when `length` divides."""
    symbols = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz"[:distinct]
    return "".join(symbols[i % distinct] for i in range(length))


@pytest.mark.parametrize(
    ("token", "kept"),
    [
        (_token(32, 32), False),  # 5.0 bits at the length floor
        (_token(23, 46), False),  # 4.52 bits
        (_token(22, 44), True),  # 4.46 bits
        (_token(31, 31), True),  # under the length floor
        (_token(16, 64), True),  # 4.0 bits, as a digest has
        ("a" * 64, True),
        ("test_a_credential_shape_stays_removed_as_before", True),
    ],
)
def test_the_entropy_check_has_a_length_and_a_bit_floor(token: str, kept: bool) -> None:
    sentence = f"Run `{token}` before the widget release."
    whole = [form.whole for form in sentence_forms(sentence)]
    assert whole == ([sentence] if kept else _frozen_extract_sentences(sentence))


@pytest.mark.parametrize(
    "span",
    ["`risk-assessment-for-the-widget`", "`task_manager_for_widget`", "`build-1234567:ok`", "`-----BEGIN CERTIFICATE-----`"],
)
def test_a_known_shape_needs_its_whole_prefix(span: str) -> None:
    sentence = f"Check {span} in the widget config."
    assert [form.whole for form in sentence_forms(sentence)] == [sentence]


@pytest.mark.parametrize(
    ("prompt", "token"),
    [
        ("Fetch the widget export from https://example.invalid/export?token=ZQAURLQ1111 now.", "ZQAURLQ1111"),
        ("Fetch the widget export from https://example.invalid/?ZQAURLQ1112 now.", "ZQAURLQ1112"),
        ("Fetch the widget export from https://example.invalid/a?k=ZQAURLQ1113. Then go on.", "ZQAURLQ1113"),
        ("Mirror the widget from https://bot:ZQAURLA2222@example.invalid/widget.tar now.", "ZQAURLA2222"),
        ("Mirror the widget from https://ZQAURLA2223@example.invalid/widget.tar now.", "ZQAURLA2223"),
        ("Run `export ZQ_API_KEY=ZQATICK3333` before the widget release.", "ZQATICK3333"),
        ("Run `ZQ_API_KEY = ZQATICK3334` before the widget release.", "ZQATICK3334"),
        ("Use `Authorization: Bearer ZQATICH4444` for the widget endpoint.", "ZQATICH4444"),
        ("Use `authorization: ZQATICH4445` for the widget endpoint.", "ZQATICH4445"),
        ("Use `bearer ZQATICH4446` for the widget endpoint.", "ZQATICH4446"),
        ("Run `curl https://example.invalid/a?k=ZQATICK3335` for the widget.", "ZQATICK3335"),
        ("Run `git clone https://bot:ZQATICK3336@example.invalid/r` for the widget.", "ZQATICK3336"),
        ("thanks https://example.invalid/hook?key=ZQALONE5555", "ZQALONE5555"),
        ("curl -sS https://example.invalid/api?apikey=ZQACURL6666", "ZQACURL6666"),
        ("Never paste `ZQ_API_KEY=ZQALOCK1212` into the widget channel.", "ZQALOCK1212"),
        # Any scheme, in any case, once it's in a code span.
        ("Run `HTTPS://example.invalid/x?ZQASCHM0001` for the widget now.", "ZQASCHM0001"),
        ("Run `git clone HTTPS://bot:ZQASCHM0002@example.invalid/r` for the widget.", "ZQASCHM0002"),
        ("Run `Https://bot:ZQASCHM0003@example.invalid/r` for the widget.", "ZQASCHM0003"),
        ("Use `postgres://bot:ZQASCHM0004@db.invalid/w` for the widget.", "ZQASCHM0004"),
        ("Use `redis://:ZQASCHM0005@db.invalid/0` for the widget.", "ZQASCHM0005"),
        ("Use `ftp://bot:ZQASCHM0006@e.invalid/w` for the widget.", "ZQASCHM0006"),
        ("Use `ssh://ZQASCHM0007@e.invalid/w` for the widget.", "ZQASCHM0007"),
        ("Use `bot:ZQASCHM0008@db.invalid/w` for the widget.", "ZQASCHM0008"),
        ("See `example.invalid/x?ZQASCHM0009` for the widget endpoint.", "ZQASCHM0009"),
        ("See `?ZQASCHM0010` for the widget endpoint.", "ZQASCHM0010"),
        # An `=` in any company.
        ('Set `"api_key"="ZQAEQLS0001"` before the widget release.', "ZQAEQLS0001"),
        ("Set `API_KEY:=ZQAEQLS0002` before the widget release.", "ZQAEQLS0002"),
        ("Set `API_KEY+=ZQAEQLS0003` before the widget release.", "ZQAEQLS0003"),
        ("Set `cfg[0]=ZQAEQLS0004` before the widget release.", "ZQAEQLS0004"),
        # A credential word, then a colon or a value.
        ('Set `{"token":"ZQAWORD0001"}` before the widget release.', "ZQAWORD0001"),
        ("Use `Bearer: ZQAWORD0002` for the widget endpoint.", "ZQAWORD0002"),
        ("Use `my_authorization: ZQAWORD0003` for the widget endpoint.", "ZQAWORD0003"),
        ("Use `X-Api-Key: ZQAWORD0004` for the widget endpoint.", "ZQAWORD0004"),
        ("Use `password: ZQAWORD0005` for the widget release.", "ZQAWORD0005"),
        ("Use `client_secret: ZQAWORD0006` for the widget release.", "ZQAWORD0006"),
        ("Use `Cookies: ZQAWORD0007` for the widget release.", "ZQAWORD0007"),
        ("Use `credentials: ZQAWORD0008` for the widget release.", "ZQAWORD0008"),
        ("Use `passwd: ZQAWORD0009` for the widget release.", "ZQAWORD0009"),
        ("Use `pwd: ZQAWORD0010` for the widget release.", "ZQAWORD0010"),
        ("Run `tool --token ZQAWORD0011` for the widget endpoint.", "ZQAWORD0011"),
        ("Run `curl -H Token ZQAWORD0012` for the widget endpoint.", "ZQAWORD0012"),
        # An `@` anywhere in a bare URL.
        ("Use https://bot:a#ZQAURLA2224@example.invalid/w for the widget.", "ZQAURLA2224"),
        ("Use https://bot:ZQAURLA2225/x@example.invalid/w for the widget.", "ZQAURLA2225"),
        # A span that's the value of a query or an assignment.
        ("Fetch https://example.invalid/x?k=`ZQAVALU0001` for the widget now.", "ZQAVALU0001"),
        ("Fetch https://example.invalid/x?a=1&`ZQAVALU0002` for the widget now.", "ZQAVALU0002"),
    ],
)
def test_a_credential_shape_stays_removed_as_before(
    tmp_path: Path, prompt: str, token: str,
) -> None:
    assert [form.whole for form in sentence_forms(prompt)] == _frozen_extract_sentences(prompt)
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _ingest_turn(store, prompt, "transcript", role="user")
        stored = store._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs UNION ALL SELECT raw_text FROM ingest_log "
            "UNION ALL SELECT entity_lower FROM belief_entities"
        ).fetchall()
    finally:
        store.close()
    assert not [row for row in stored if token.lower() in str(row[0]).lower()]


@pytest.mark.parametrize(
    "sentence",
    [
        # What the rule keeps on purpose.
        "Is it https://example.invalid/a? Then read the second part.",
        "See `https://example.invalid/a?` for the widget docs.",
        "Is `this one?` the widget flag to use here?",
        "See `git@example.invalid:zq/widget.git` for the widget docs.",
        "Use `pkg@1.2.3` for the widget build.",
        # What the rule misses. It's a list of shapes and not a scanner, and
        # these invented credentials reach the store.
        "Post to https://hooks.example.invalid/services/T0/B0/ZQMISS1 for the widget.",
        "Use https://example.invalid/cb#access_token=ZQMISS2 for the widget.",
        "Use `https://example.invalid/cb#ZQMISS3` for the widget endpoint.",
        "Use `zqk-ZQMISS4` as the widget credential.",
        "Run `curl -u " + "bot:ZQMISS5 https://example.invalid/a` for the widget.",
        "The widget API key is ZQMISS6 for the release.",
        "Read https://example.invalid/ZQMISS7? for the widget release.",
        "Read https://example.invalid/a%3Fk=ZQMISS8 for the widget release.",
        "Read HTTPS://bot:ZQMISS9@example.invalid/a for the widget release.",
        "Read postgres://bot:ZQMISS10@db.invalid/w for the widget release.",
        "Use `KEY\uff1dZQMISS11` for the widget release.",
        "Use `session ZQMISS12` for the widget release.",
    ],
)
def test_the_rule_is_a_list_of_shapes_and_not_a_scanner(sentence: str) -> None:
    assert [form.whole for form in sentence_forms(sentence)] == [sentence]


@pytest.mark.parametrize(
    "span",
    ["`a == b`", "`--flag=3`", "`monkey patch`", "`key: value`", "`auth: on`"],
)
def test_the_rule_may_remove_more_than_a_secret(span: str) -> None:
    """Over-removal costs what every span cost before #1874, so it's accepted."""
    sentence = f"Check {span} in the widget config."
    assert [form.whole for form in sentence_forms(sentence)] == _frozen_extract_sentences(sentence)


def test_a_span_that_opens_the_text_has_no_introducer() -> None:
    """Position zero has nothing before it, so the text's last character doesn't count."""
    sentence = "`zqtool` runs the widget release when a="
    assert [form.whole for form in sentence_forms(sentence)] == [sentence]


def test_a_span_right_after_a_question_mark_stays_removed() -> None:
    sentence = "Fetch https://example.invalid/x?`ZQAVALU0003` for the widget now."
    assert "ZQAVALU0003" not in " ".join(form.whole for form in sentence_forms(sentence))


def test_a_span_after_prose_keeps_its_text_when_a_stop_ends_the_word() -> None:
    sentence = "Is it this one? `zqtool` is the widget flag to use."
    assert [form.whole for form in sentence_forms(sentence)] == [
        "Is it this one?", "`zqtool` is the widget flag to use.",
    ]
