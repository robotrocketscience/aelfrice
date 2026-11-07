"""#1371 §1 — the transcript filter stops deleting real beliefs.

Two corpora, and both halves are required. `must_survive_1371.txt` alone is
passed perfectly by a filter that discards nothing; `must_die_1371.txt` is
the control that makes it mean something.

The rule under test is one signal, applied to two categories: **terminal
sentence punctuation separates written prose from a pasted command or a
chat ack.** `git add .` fails it (the stop follows a space), `git push is
forbidden on main.` passes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.noise_filter import (
    _TRANSCRIPT_ACK_PHRASES,
    _TRANSCRIPT_ACK_RE,
    _looks_like_written_prose,
    is_transcript_noise,
    is_transcript_scaffolding,
)

_DATA = Path(__file__).parent / "data"


def _corpus(name: str) -> list[str]:
    rows = []
    for line in (_DATA / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


MUST_SURVIVE = _corpus("must_survive_1371.txt")
MUST_DIE = _corpus("must_die_1371.txt")


def test_the_corpora_are_not_empty_and_do_not_overlap() -> None:
    """A misparsed corpus file would silently make every row below vacuous —
    `_corpus` returning `[]` turns a parametrized test into zero tests, which
    reports as green."""
    assert len(MUST_SURVIVE) >= 20
    assert len(MUST_DIE) >= 25
    assert not (set(MUST_SURVIVE) & set(MUST_DIE))


#: The nine strings #1371's AC1 names. #1159's audit finding executed each
#: against the pre-fix predicate and reported all nine as discarded, so the
#: AC is met by these strings verbatim or it is not met. Duplicated here
#: rather than read from the corpus: the point of the assertion below is
#: that the corpus file still contains them.
_AC1_EXECUTED_EXAMPLES = (
    "No vector embeddings, ever.",
    "No tests exist for the temporal spine.",
    "No we should use uv instead of pip.",
    "Nothing in retrieval may call the network.",
    "Ready means the gate passed.",
    "git history rewrite is denied by the ruleset.",
    "pytest is the only test runner we support.",
    "gh api is the only way to delete refs.",
    "python 3.13 is the minimum supported version.",
)


def test_ac1_names_nine_strings_and_the_corpus_carries_all_nine() -> None:
    """AC1's corpus clause, pinned as a membership claim.

    The parametrized test above only asserts that whatever rows are in the
    file survive; it says nothing about which rows are in the file. Dropping
    one of the nine — or paraphrasing it into a structural analogue — would
    leave that test green and quietly unsatisfy the AC.

    Falsifiable by deleting any row of the AC1 block in
    `tests/data/must_survive_1371.txt`.
    """
    missing = [s for s in _AC1_EXECUTED_EXAMPLES if s not in MUST_SURVIVE]
    assert not missing, f"AC1 examples absent from the must-survive corpus: {missing}"


@pytest.mark.parametrize("sentence", MUST_SURVIVE, ids=range(len(MUST_SURVIVE)))
def test_a_real_belief_is_not_discarded(sentence: str) -> None:
    """#1159 AC4's must-survive corpus. Each row is a sentence the filter
    was measurably discarding, or one the issue quotes."""
    assert is_transcript_noise(sentence) is False, sentence


@pytest.mark.parametrize("sentence", MUST_DIE, ids=range(len(MUST_DIE)))
def test_scaffolding_and_bare_acks_are_still_discarded(sentence: str) -> None:
    """The control. Rescuing everything passes the corpus above and fails
    here."""
    assert is_transcript_noise(sentence) is True, sentence


def test_the_ack_allowlist_is_closed_and_every_member_is_pinned() -> None:
    """The allowlist is the one place the prose rule is overridden by name,
    so its membership is the thing to pin, not its behaviour in general.

    Every phrase must appear in the control corpus: a phrase in the frozenset
    with no corpus row is unpinned (deleting it leaves the suite green), and a
    corpus row with no frozenset entry means the row is being discarded by
    some other arm than the one the block claims. The count is asserted
    because the release notes state it — an auditor counting rows must find
    the number the changelog gives.
    """
    assert len(_TRANSCRIPT_ACK_PHRASES) == 8
    assert set(_TRANSCRIPT_ACK_PHRASES) <= set(MUST_DIE)


def test_the_prose_test_is_what_separates_them() -> None:
    """The single signal both categories rest on.

    `git add .` and `cd /tmp/.` end in a dot that is an *argument*; the stop
    has to follow a word character to count as prose. Falsifiable by
    relaxing the pattern to `\\.$`: both then read as prose and stop being
    filtered, which the control corpus catches.
    """
    assert _looks_like_written_prose("git push is forbidden on main.") is True
    assert _looks_like_written_prose("No behavior change.") is True
    assert _looks_like_written_prose('He said "no".') is True
    assert _looks_like_written_prose("git add .") is False
    assert _looks_like_written_prose("cd /tmp/.") is False
    assert _looks_like_written_prose("Yes keep working") is False


def test_trailing_whitespace_does_not_flip_the_prose_verdict() -> None:
    """`_looks_like_written_prose` rstrips, and that is behaviour, not tidiness.

    The transcript logger writes prompts verbatim and `extract_sentences`
    does not guarantee a stripped tail, so a real sentence can arrive with a
    trailing space. Without the rstrip the anchored `$` no longer sees the
    full stop, the sentence fails the prose test, and the ack arm deletes it
    — which is the #1371 §1 defect coming back through the whitespace door.
    The rstrip must not rescue a pasted command, so `git add . ` is asserted
    on the other side.

    Falsifiable by dropping `.rstrip()` from `_looks_like_written_prose`.
    """
    assert _looks_like_written_prose("No behavior change. ") is True
    assert is_transcript_noise("No behavior change. ") is False
    assert _looks_like_written_prose("git add . ") is False
    assert is_transcript_noise("git add . ") is True


def test_the_scaffolding_split_is_proper_over_the_control_corpus() -> None:
    """The split the logger's two arms rest on, asserted as a *partition*.

    The containment on its own is not a test. `is_transcript_noise` opens
    with `if is_transcript_scaffolding(sentence): return True`, so
    `scaffolding(x) -> noise(x)` holds for every input by construction — a
    version of this that only walked the corpus asserting the implication
    stayed green even with `return True` wired into the top of
    `is_transcript_scaffolding`, and its docstring named a falsifier
    ("add a category the noise function does not consult") that the
    delegation makes impossible.

    What is *not* free is that the subset is proper, and proper in the
    direction the logger depends on: scaffolding condemns the whole payload,
    so an ack must never be scaffolding or a prompt is dropped again on its
    leading ack, which is the #1371 §1 defect. Every ack the filter knows
    about is checked, not two hand-picked literals — a widening that swallows
    the unpunctuated chat acks is invisible to a spot check.

    Falsifiable by widening `is_transcript_scaffolding` to claim any ack
    (`return True` at the top turns this red; so does folding
    `_TRANSCRIPT_ACK_PHRASES` into the scaffolding branch).
    """
    scaffolding = {s for s in MUST_DIE if is_transcript_scaffolding(s)}
    assert scaffolding, "no control row is structural — the split is degenerate"
    assert set(MUST_DIE) - scaffolding, "every control row is structural"

    acks = set(_TRANSCRIPT_ACK_PHRASES) | {
        s for s in MUST_DIE if _TRANSCRIPT_ACK_RE.match(s) is not None
    }
    assert not (acks & scaffolding), sorted(acks & scaffolding)

    # And the containment itself, stated as the invariant the logger reads.
    for sentence in MUST_DIE + MUST_SURVIVE:
        if is_transcript_scaffolding(sentence):
            assert is_transcript_noise(sentence), sentence


def test_scaffolding_covers_structure_and_not_acks() -> None:
    """The split the logger depends on: a harness tag marks the whole
    payload, an ack marks only its own sentence."""
    assert is_transcript_scaffolding("<task-notification>") is True
    assert is_transcript_scaffolding("git status") is True
    assert is_transcript_scaffolding("⏺ ran a tool") is True
    # acks and progress emits are noise but NOT scaffolding
    assert is_transcript_scaffolding("Yes.") is False
    assert is_transcript_noise("Yes.") is True
    assert is_transcript_scaffolding("Running.") is False
    assert is_transcript_noise("Running.") is True


# --- the transcript_logger granularity fix --------------------------------


def _log(tmp_path: Path, prompt: str, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Drive the real UserPromptSubmit handler and return the rows it wrote.

    Points `transcripts_dir` at a tmp path rather than stubbing the append,
    so the gate under test is the shipped one and the assertion is on the
    file the rebuilder would actually read.
    """
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dot"))
    from aelfrice import transcript_logger

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


def test_a_multi_sentence_prompt_survives_a_leading_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #1371 §1 acceptance case, and it is not hypothetical — this exact
    prompt is in this repo's archived transcripts and was dropped whole.

    Falsifiable by restoring the bare `if is_transcript_noise(prompt)` gate.
    """
    rows = _log(tmp_path, "No work around. It needs to be fixed", monkeypatch)
    assert rows, "a real multi-sentence prompt was dropped on its leading ack"


def test_a_mixed_prompt_is_kept_because_every_sentence_must_be_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quantifier itself, which nothing else in the file reaches.

    `"No it broke. Yes keep it"` is noise as a whole prompt (the ack pattern
    matches `No` plus a 22-char tail with no prose-terminal stop), and its two
    sentences carry noise flags `[False, True]` — the only shape that tells
    `all` from `any`. `No it broke.` is durable content, so the prompt stays.

    Falsifiable by weakening `all(...)` to `any(...)` at the logger's
    every-sentence gate: this prompt is then dropped and no row is written.
    The other logger tests cannot catch that — `'No work around. It needs to
    be fixed'` is `[False, False]`, the `<task-notification>` block exits one
    line earlier on `is_transcript_scaffolding`, and `"Yes."` splits to no
    sentences at all.

    Both sentences are >= 10 characters so `extraction._MIN_LEN` keeps them.
    """
    from aelfrice.extraction import extract_sentences

    prompt = "No it broke. Yes keep it"
    parts = [s for s in extract_sentences(prompt) if s.strip()]
    assert [is_transcript_noise(s) for s in parts] == [False, True], (
        "fixture must be mixed, or it cannot discriminate the quantifier"
    )
    assert _log(tmp_path, prompt, monkeypatch), (
        "a prompt with one durable sentence was dropped on its trailing ack"
    )


def test_a_harness_block_is_still_dropped_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression the acceptance text's literal reading would have caused.

    A `<task-notification>` block's prose lines are not individually noise,
    so plain sentence-granularity keeps it — measured, that is 751 of this
    repo's 6,274 archived prompts, which is exactly the flooding #747 added
    the gate to stop. Scaffolding is judged on the whole payload instead.

    Falsifiable by dropping the `is_transcript_scaffolding` arm from the
    logger and gating only on "every sentence is noise".
    """
    block = (
        "<task-notification>\n<task-id>bx7oplash</task-id>\n"
        "<summary>Monitor fired</summary>\n"
        "The background command finished successfully.\n"
        "</task-notification>"
    )
    from aelfrice.extraction import extract_sentences

    parts = [s for s in extract_sentences(block) if s.strip()]
    assert not all(is_transcript_noise(s) for s in parts), (
        "fixture must have at least one non-noise sentence, or it does not "
        "discriminate the two logger rules"
    )
    assert _log(tmp_path, block, monkeypatch) == []


def test_a_pasted_command_prompt_is_still_dropped_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The *shell* half of the same whole-payload line, which nothing reached.

    `test_a_harness_block_is_still_dropped_whole` drives a
    `<task-notification>` fixture, so it exercises only the XML-prefix arm of
    `is_transcript_scaffolding`. Narrowing the logger's gate to
    `is_transcript_scaffolding(prompt) and not prompt.startswith(("cd /",
    "git ", "gh ", "uv run", "pytest", "python "))` — dropping the pasted-command
    half while keeping the tag half — left the whole suite green while changing
    shipped behaviour: this prompt goes from unlogged to logged.

    The fixture is mixed on purpose. Its sentences carry noise flags
    `[True, False]`, so `all(...)` is False and the every-sentence arm does
    *not* condemn it; the only rule that can drop it is the scaffolding one.
    Without that assertion the case would pass under a logger that had lost
    the scaffolding arm entirely.
    """
    from aelfrice.extraction import extract_sentences

    prompt = "git rebase -i HEAD~3\nSquash the last two commits and force push"
    parts = [s for s in extract_sentences(prompt) if s.strip()]
    assert not all(is_transcript_noise(s) for s in parts), (
        "fixture must have at least one non-noise sentence, or it does not "
        "discriminate the scaffolding arm from the all-sentences arm"
    )
    assert _log(tmp_path, prompt, monkeypatch) == []


def test_a_bare_ack_prompt_is_still_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every sentence is noise, so the whole prompt goes — unchanged."""
    assert _log(tmp_path, "Yes.", monkeypatch) == []
