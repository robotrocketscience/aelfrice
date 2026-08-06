"""#1371 §1: statements the write-side admission filter must never delete.

The corpus lives in `tests/fixtures/must_survive_beliefs.txt` — one
statement per line — so it can be extended without touching test code.
Every row is a real durable belief; most of them were returned as
`is_transcript_noise(...) is True` on `github/main` before #1371, i.e.
they were erased from the turn log before ingest ever saw them.

Two halves, both required:

* the corpus must survive `is_transcript_noise` (the narrowing), and
* the ack / shell-command categories must still fire on real
  scaffolding (the guard against "narrowing" by deleting the category).

The second half is why this module also pins the negative cases: a
change that made `is_transcript_noise` return False unconditionally
would turn the corpus green and is caught here instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.noise_filter import (
    is_agent_ack,
    is_shell_command,
    is_transcript_noise,
)

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CORPUS_PATH: Path = _REPO_ROOT / "tests" / "fixtures" / "must_survive_beliefs.txt"


def _load_corpus() -> list[str]:
    """Return the corpus rows: non-empty, non-comment lines."""
    text = _CORPUS_PATH.read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


MUST_SURVIVE: list[str] = _load_corpus()


def test_corpus_file_is_populated() -> None:
    """Guard the guard: an empty or missing corpus passes vacuously."""
    assert _CORPUS_PATH.is_file()
    assert len(MUST_SURVIVE) >= 15


@pytest.mark.parametrize("statement", MUST_SURVIVE, ids=range(len(MUST_SURVIVE)))
def test_must_survive_statement_is_not_transcript_noise(statement: str) -> None:
    assert is_transcript_noise(statement) is False, statement


# --- the categories must still fire (anti-gutting half) -------------------

_STILL_ACK: tuple[str, ...] = (
    "Yes.",
    "No.",
    "Ready.",
    "Nothing.",
    "Polling.",
    "Standing by.",
    "Ready",
    "Polling",
    "Nothing to report.",
    "Ready when you are.",
    "Polling for results.",
    "Standing by for your direction.",
)

_STILL_COMMAND: tuple[str, ...] = (
    "cd /home/user/projects",
    "git checkout main",
    "gh pr view 675",
    "uv run pytest tests/",
    "pytest tests/test_ingest.py -v",
    "python script.py --flag",
    "git add .",
    "uv run --no-sync pytest -q --ignore=tests/e2e",
)


@pytest.mark.parametrize("utterance", _STILL_ACK)
def test_agent_ack_still_fires(utterance: str) -> None:
    assert is_agent_ack(utterance) is True, utterance
    assert is_transcript_noise(utterance) is True, utterance


@pytest.mark.parametrize("utterance", _STILL_COMMAND)
def test_shell_command_still_fires(utterance: str) -> None:
    assert is_shell_command(utterance) is True, utterance
    assert is_transcript_noise(utterance) is True, utterance


# --- the three independent narrowing conditions ---------------------------
#
# Each assertion below is rescued by exactly one of the three ack
# conditions. Reverting any single condition turns exactly one of them
# red, which is what makes the mutation check meaningful.


def test_ack_word_cap_rejects_long_clause() -> None:
    """Seven words. Rescued by `_ACK_MAX_WORDS` alone."""
    assert is_agent_ack("Nothing in retrieval may call the network.") is False


def test_ack_tail_lead_rejects_content_word() -> None:
    """Tail opens on a content word. Rescued by `_ACK_TAIL_LEAD_WORDS`."""
    assert is_agent_ack("Ready means the gate passed.") is False
    assert is_agent_ack("No vector embeddings, ever.") is False


def test_ack_main_clause_auxiliary_rejects_assertion() -> None:
    """Tail opens on an allowed lead but predicates in the main clause.

    "Yes" + "to" is an allowed adjunct lead; the finite `is` with no
    preceding subordinator is what disqualifies it. Rescued by the
    `_ACK_FINITE_VERBS` scan alone.
    """
    assert is_agent_ack("No to us is the wrong default.") is False
    # ...and the subordinator carve-out keeps the real ack green.
    assert is_agent_ack("Ready when you are.") is True


def test_shell_command_rejects_sentence_final_punctuation() -> None:
    """Rescued by the sentence-final-punctuation check alone.

    No token here is a determiner, pronoun or finite auxiliary, so the
    closed-class scan cannot save it — only the trailing full stop does.
    """
    assert is_shell_command("pytest collects tests/regression too.") is False


def test_shell_command_rejects_clause_backbone() -> None:
    """Rescued by `_COMMAND_DISQUALIFYING_TOKENS` alone (no full stop)."""
    assert is_shell_command("pytest is the only test runner we support") is False
    assert is_shell_command("git history rewrite is denied by policy") is False
